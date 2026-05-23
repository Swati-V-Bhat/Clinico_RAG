# src/train.py
import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, classification_report

from .dataset import CXRDataset, MontgomeryDataset, SegDS, collate_fn
from .models import MedXRVEncoder, build_unet, dice_bce_loss


def free_gpu(*models):
    for m in models:
        if m is not None:
            m.cpu()
    gc.collect()
    torch.cuda.empty_cache()


# ── Stage A: U-Net Training ───────────────────────────────────
def train_unet(cfg, device):
    paths = cfg['paths']
    ucfg  = cfg['unet']

    if not os.path.exists(os.path.join(paths['montgomery_mask'], 'leftMask')):
        print("⚠️  No Montgomery masks found — skipping U-Net")
        return None

    unet = build_unet(device)
    ds   = SegDS(paths['montgomery_img'], paths['montgomery_mask'])
    dl   = DataLoader(ds, batch_size=ucfg['batch_size'], shuffle=True, num_workers=2)
    opt  = optim.AdamW(unet.parameters(), lr=ucfg['lr'])
    sch  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ucfg['epochs'])
    best = float('inf')

    os.makedirs(paths['output_dir'], exist_ok=True)
    print("Training U-Net (ResNet-34, BCE+Dice loss)...")

    for ep in range(ucfg['epochs']):
        unet.train(); ls = 0
        for imgs, masks in dl:
            imgs, masks = imgs.to(device), masks.to(device)
            opt.zero_grad()
            loss = dice_bce_loss(unet(imgs), masks)
            loss.backward(); opt.step()
            ls += loss.item()
        sch.step()
        avg = ls / len(dl)
        if avg < best:
            best = avg
            torch.save(unet.state_dict(),
                       os.path.join(paths['output_dir'], 'unet.pth'))
        print(f"  Epoch {ep+1:2d}/{ucfg['epochs']}  Loss:{avg:.4f}")

    print(f"✅ U-Net saved  (best loss={best:.4f})")
    return unet


def eval_unet(cfg, device, unet_model=None):
    """Evaluate U-Net: Dice, IoU, Pixel Accuracy."""
    paths = cfg['paths']

    if unet_model is not None:
        unet = unet_model.to(device)
    else:
        unet = build_unet(device)
        unet.load_state_dict(torch.load(
            os.path.join(paths['output_dir'], 'unet.pth'), map_location=device))
    unet.eval()

    ds = SegDS(paths['montgomery_img'], paths['montgomery_mask'])
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)

    dice_scores, iou_scores, pixel_accs = [], [], []
    with torch.no_grad():
        for imgs, masks in dl:
            imgs, masks = imgs.to(device), masks.to(device)
            preds_bin   = (torch.sigmoid(unet(imgs)) > 0.5).float()
            for p, m in zip(preds_bin, masks):
                p = p.view(-1); m = m.view(-1)
                dice  = (2*(p*m).sum()+1) / (p.sum()+m.sum()+1)
                inter = (p*m).sum()
                union = p.sum()+m.sum()-inter
                iou   = (inter+1) / (union+1)
                pacc  = (p == m).float().mean()
                dice_scores.append(dice.item())
                iou_scores.append(iou.item())
                pixel_accs.append(pacc.item())

    results = {
        'dice':      np.mean(dice_scores),
        'iou':       np.mean(iou_scores),
        'pixel_acc': np.mean(pixel_accs),
        'n':         len(dice_scores),
    }
    print(f"\n{'='*45}")
    print(f"  U-Net Segmentation Accuracy (Montgomery)")
    print(f"{'='*45}")
    print(f"  Mean Dice Score    : {results['dice']:.4f}")
    print(f"  Mean IoU           : {results['iou']:.4f}")
    print(f"  Mean Pixel Accuracy: {results['pixel_acc']:.4f}")
    print(f"  Evaluated on       : {results['n']} images")
    print(f"{'='*45}")
    return results


# ── Stage B: MedXRVEncoder Training ──────────────────────────
def train_encoder(cfg, device):
    """
    Two-phase domain-aware training (Section 3.3.5 of paper).
    Returns model, val_dl, labels, preds, probs, history.
    """
    paths  = cfg['paths']
    tcfg   = cfg['training']
    p1cfg  = tcfg['phase1']
    p2cfg  = tcfg['phase2']
    os.makedirs(paths['output_dir'], exist_ok=True)

    # ── Build splits ──────────────────────────────────────────
    full_ds = CXRDataset(paths['shenzhen_img'],
                          paths.get('shenzhen_txt'), train=False)
    n    = len(full_ds)
    n_tr = int((1 - tcfg['val_split']) * n)
    g    = torch.Generator(); g.manual_seed(tcfg['seed'])
    idx  = torch.randperm(n, generator=g).tolist()

    shenz_train = Subset(
        CXRDataset(paths['shenzhen_img'],
                   paths.get('shenzhen_txt'), train=True),
        idx[:n_tr])

    mont_train = None
    if os.path.exists(paths.get('montgomery_img', '')):
        mont_train = MontgomeryDataset(paths['montgomery_img'], train=True)
        print(f"  + Montgomery: {len(mont_train)} images for cross-site training")

    val_ds = Subset(full_ds, idx[n_tr:])

    combined = ConcatDataset([shenz_train, mont_train]) \
        if mont_train else shenz_train

    train_dl = DataLoader(combined, batch_size=tcfg['batch_size'], shuffle=True,
                           collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=tcfg['batch_size'], shuffle=False,
                           collate_fn=collate_fn, num_workers=2)

    print(f"  Train: {len(combined)}  |  Val: {len(val_ds)}")

    model  = MedXRVEncoder().to(device)
    scaler = torch.cuda.amp.GradScaler()
    crit   = nn.CrossEntropyLoss(label_smoothing=tcfg['label_smoothing'])

    history = {'phase1': {'loss': [], 'f1': []},
               'phase2': {'loss': [], 'f1': []}}
    best_f1, best_state = 0, None

    # ── Phase 1: frozen backbones ─────────────────────────────
    print("\nPhase 1: Frozen backbones (MCSA + head only)...")
    model.freeze_backbones()
    opt1 = optim.AdamW([*model.fusion.parameters(),
                         *model.classifier.parameters()],
                        lr=p1cfg['lr_head'], weight_decay=p1cfg['weight_decay'])
    sch1 = optim.lr_scheduler.CosineAnnealingLR(
        opt1, T_max=p1cfg['epochs'], eta_min=p1cfg['eta_min'])

    for ep in range(p1cfg['epochs']):
        model.train(); ls = 0
        for imgs, lbls, _, _ in train_dl:
            imgs, lbls = imgs.to(device), lbls.to(device)
            opt1.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss = crit(model(imgs), lbls)
            scaler.scale(loss).backward()
            scaler.unscale_(opt1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt1); scaler.update()
            ls += loss.item()
        sch1.step()

        ep_f1 = _val_f1(model, val_dl, device)
        avg_loss = ls / len(train_dl)
        history['phase1']['loss'].append(avg_loss)
        history['phase1']['f1'].append(ep_f1)
        print(f"  Ph1 Ep {ep+1:2d}/{p1cfg['epochs']}  "
              f"Loss:{avg_loss:.4f}  F1:{ep_f1:.4f}  "
              f"LR:{opt1.param_groups[0]['lr']:.1e}")
        if ep_f1 > best_f1:
            best_f1   = ep_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f"  Phase 1 best F1 = {best_f1:.4f}")

    # ── Phase 2: full fine-tuning ─────────────────────────────
    print(f"\nPhase 2: Full fine-tuning "
          f"(backbone LR={p2cfg['lr_backbone']:.0e}, head LR={p2cfg['lr_head']:.0e})...")
    model.unfreeze_backbones()
    opt2 = optim.AdamW([
        {'params': model.med_backbone.parameters(), 'lr': p2cfg['lr_backbone']},
        {'params': model.eff.parameters(),           'lr': p2cfg['lr_backbone']},
        {'params': model.fusion.parameters(),        'lr': p2cfg['lr_head']},
        {'params': model.classifier.parameters(),    'lr': p2cfg['lr_head']},
    ], weight_decay=p2cfg['weight_decay'])
    sch2 = optim.lr_scheduler.CosineAnnealingLR(
        opt2, T_max=p2cfg['epochs'], eta_min=p2cfg['eta_min'])

    patience = 0
    for ep in range(p2cfg['epochs']):
        model.train(); ls = 0
        for imgs, lbls, _, _ in train_dl:
            imgs, lbls = imgs.to(device), lbls.to(device)
            opt2.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss = crit(model(imgs), lbls)
            scaler.scale(loss).backward()
            scaler.unscale_(opt2)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt2); scaler.update()
            ls += loss.item()
        sch2.step()

        ep_f1    = _val_f1(model, val_dl, device)
        avg_loss = ls / len(train_dl)
        lr_now   = opt2.param_groups[2]['lr']
        history['phase2']['loss'].append(avg_loss)
        history['phase2']['f1'].append(ep_f1)
        print(f"  Ph2 Ep {ep+1:2d}/{p2cfg['epochs']}  "
              f"Loss:{avg_loss:.4f}  F1:{ep_f1:.4f}  LR:{lr_now:.1e}")

        if ep_f1 > best_f1:
            best_f1 = ep_f1; patience = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_state,
                       os.path.join(paths['output_dir'], 'model_best.pth'))
        else:
            patience += 1
            if patience >= p2cfg['patience']:
                print(f"  Early stop at epoch {ep+1} — best F1={best_f1:.4f}")
                break

    model.load_state_dict(best_state)

    # ── Final evaluation ──────────────────────────────────────
    labels, preds, probs = _full_eval(model, val_dl, device)
    _print_metrics(labels, preds, probs)

    return model, val_dl, labels, preds, probs, history


# ── Helpers ───────────────────────────────────────────────────
def _val_f1(model, val_dl, device):
    model.eval(); preds, trues = [], []
    with torch.no_grad():
        for imgs, lbls, _, _ in val_dl:
            with torch.cuda.amp.autocast():
                out = model(imgs.to(device))
            preds.extend(out.argmax(1).cpu().numpy())
            trues.extend(lbls.numpy())
    return f1_score(trues, preds, zero_division=0)


def _full_eval(model, val_dl, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, lbls, _, _ in val_dl:
            with torch.cuda.amp.autocast():
                out = model(imgs.to(device))
            all_probs.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(lbls.numpy())
    return all_labels, all_preds, all_probs


def _print_metrics(labels, preds, probs):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, zero_division=0)
    auc = roc_auc_score(labels, probs)
    print(f"\n{'='*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1       : {f1:.4f}")
    print(f"  AUC      : {auc:.4f}")
    print(f"{'='*50}")
    print(classification_report(labels, preds, target_names=['Normal', 'TB']))
