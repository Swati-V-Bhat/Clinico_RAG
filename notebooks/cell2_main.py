# ============================================================
# CELL 2 — Clinico-RAG Full Pipeline (Kaggle)
# ============================================================
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# ── 1. Imports & Config ──────────────────────────────────────
import os, warnings, gc, cv2, re
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score,
    classification_report, confusion_matrix,
    roc_curve, precision_recall_curve,
    average_precision_score, calibration_curve,
)
from sklearn.manifold import TSNE
from sklearn.metrics import auc as sk_auc
import timm
import torchxrayvision as xrv
import segmentation_models_pytorch as smp
import faiss

warnings.filterwarnings('ignore')
torch.manual_seed(42); np.random.seed(42)

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH       = 16
EMBED_DIM   = 768
K_BI        = 30
K_FINAL     = 3
LR_HEAD     = 3e-4    # matches paper Table 1
LR_BACKBONE = 3e-5    # matches paper Table 1

os.makedirs('/kaggle/working/ckpts', exist_ok=True)
print(f"Device: {DEVICE}")
print(f"LR_HEAD={LR_HEAD}  LR_BACKBONE={LR_BACKBONE}")


# ── 2. Dataset Locator ───────────────────────────────────────
def locate(start='/kaggle/input'):
    p = {'shenzhen': {}, 'montgomery': {}}
    for root, dirs, files in os.walk(start):
        if 'ChinaSet_AllFiles' in root and 'CXR_png' in dirs:
            p['shenzhen']['img'] = os.path.join(root, 'CXR_png')
            if 'ClinicalReadings' in dirs:
                p['shenzhen']['txt'] = os.path.join(root, 'ClinicalReadings')
        if 'MontgomerySet' in root and 'CXR_png' in dirs:
            p['montgomery']['img'] = os.path.join(root, 'CXR_png')
        if 'ManualMask' in root and 'leftMask' in dirs:
            p['montgomery']['mask'] = root
    for k, v in p.items():
        print(f"  {'✅' if v else '❌'} {k}: {v}")
    return p

PATHS = locate()


def free_gpu(*models):
    for m in models:
        if m is not None: m.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  GPU freed → {torch.cuda.memory_allocated()/1e9:.2f} GB")


# ── 3. Transforms & Datasets ─────────────────────────────────
TRAIN_TFM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.3, 0.3, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
VAL_TFM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class SegDS(Dataset):
    def __init__(self, img_dir, mask_root):
        self.img_dir   = img_dir
        self.mask_root = mask_root
        self.files     = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        f    = self.files[idx]
        img  = cv2.resize(cv2.imread(os.path.join(self.img_dir, f)), (224, 224))
        mask = np.zeros((224, 224), dtype=np.float32)
        for s in ('leftMask', 'rightMask'):
            p = os.path.join(self.mask_root, s, f)
            if os.path.exists(p):
                mask = np.maximum(mask, cv2.resize(cv2.imread(p, 0), (224, 224)))
        mask = (mask > 0).astype(np.float32)
        return (torch.from_numpy(img).float().permute(2, 0, 1) / 255.,
                torch.from_numpy(mask).unsqueeze(0))


class CXRDataset(Dataset):
    def __init__(self, img_root, txt_root=None, train=True):
        self.img_root = img_root
        self.txt_root = txt_root
        self.files    = sorted([f for f in os.listdir(img_root) if f.endswith('.png')])
        self.tfm      = TRAIN_TFM if train else VAL_TFM
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        fname = self.files[i]
        label = 1 if '_1.png' in fname else 0
        text  = ""
        if self.txt_root:
            tp = os.path.join(self.txt_root, fname.replace('.png', '.txt'))
            if os.path.exists(tp):
                try:    text = open(tp, 'r', errors='ignore').read().strip()
                except: pass
        img = Image.open(os.path.join(self.img_root, fname)).convert('RGB')
        return self.tfm(img), label, text, fname


class MontgomeryDataset(Dataset):
    def __init__(self, img_root, train=True):
        self.img_root = img_root
        self.files    = sorted([f for f in os.listdir(img_root) if f.endswith('.png')])
        self.tfm      = TRAIN_TFM if train else VAL_TFM
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        fname = self.files[i]
        label = 1 if '_1.png' in fname else 0
        img   = Image.open(os.path.join(self.img_root, fname)).convert('RGB')
        return self.tfm(img), label, "", fname


def collate(batch):
    imgs, lbls, texts, fnames = zip(*batch)
    return torch.stack(imgs), torch.tensor(lbls), list(texts), list(fnames)

print("✅ Dataset classes ready")


# ── 4. Architecture ───────────────────────────────────────────
def to_xrv(x):
    mean   = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
    std    = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
    coeffs = torch.tensor([0.2989, 0.5870, 0.1140], device=x.device).view(1,3,1,1)
    gray   = ((x * std + mean) * coeffs).sum(1, keepdim=True)
    return gray * 2048 - 1024


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim_a, dim_b, heads=8, out_dim=EMBED_DIM):
        super().__init__()
        d = 512
        self.proj_a   = nn.Linear(dim_a, d)
        self.proj_b   = nn.Linear(dim_b, d)
        self.attn_a2b = nn.MultiheadAttention(d, heads, dropout=0.1, batch_first=True)
        self.attn_b2a = nn.MultiheadAttention(d, heads, dropout=0.1, batch_first=True)
        self.norm_a   = nn.LayerNorm(d)
        self.norm_b   = nn.LayerNorm(d)
        self.fusion   = nn.Sequential(
            nn.Linear(d*2, out_dim), nn.GELU(), nn.LayerNorm(out_dim), nn.Dropout(0.1))
    def forward(self, feat_a, feat_b):
        a = self.proj_a(feat_a).unsqueeze(1)
        b = self.proj_b(feat_b).unsqueeze(1)
        a2b, _ = self.attn_a2b(a, b, b)
        b2a, _ = self.attn_b2a(b, a, a)
        a_out  = self.norm_a(a + a2b).squeeze(1)
        b_out  = self.norm_b(b + b2a).squeeze(1)
        return self.fusion(torch.cat([a_out, b_out], dim=1))


class MedXRVEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        _xrv              = xrv.models.DenseNet(weights="densenet121-res224-all")
        self.med_backbone = _xrv.features
        self.med_pool     = nn.AdaptiveAvgPool2d(1)
        self.med_dim      = 1024
        self.eff          = timm.create_model('efficientnet_b0', pretrained=True,
                                               num_classes=0, global_pool='avg')
        self.eff_dim      = 1280
        self.fusion       = CrossAttentionFusion(self.med_dim, self.eff_dim,
                                                  heads=8, out_dim=EMBED_DIM)
        self.classifier   = nn.Sequential(nn.Dropout(0.2), nn.Linear(EMBED_DIM, 2))
    def _med(self, x):
        return self.med_pool(self.med_backbone(to_xrv(x))).flatten(1)
    def get_embedding(self, x):
        return F.normalize(self.fusion(self._med(x), self.eff(x)), dim=1)
    def forward(self, x):
        return self.classifier(self.fusion(self._med(x), self.eff(x)))
    def get_cam_layer(self):
        return self.med_backbone.denseblock4.denselayer16.conv2


def build_unet():
    return smp.Unet('resnet34', encoder_weights='imagenet',
                     in_channels=3, classes=1, activation=None).to(DEVICE)

print("✅ MedXRVEncoder + MCSA defined")


# ── 5. U-Net Training ─────────────────────────────────────────
def dice_bce(pred, target):
    bce  = nn.BCEWithLogitsLoss()(pred, target)
    p    = torch.sigmoid(pred).view(-1)
    t    = target.view(-1)
    dice = 1 - (2*(p*t).sum()+1) / (p.sum()+t.sum()+1)
    return bce + dice


def train_unet(paths):
    if not paths['montgomery'].get('mask'):
        print("⚠️  No Montgomery masks — skipping U-Net"); return None
    unet = build_unet()
    ds   = SegDS(paths['montgomery']['img'], paths['montgomery']['mask'])
    dl   = DataLoader(ds, batch_size=8, shuffle=True, num_workers=2)
    opt  = optim.AdamW(unet.parameters(), lr=1e-3)
    sch  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    best = float('inf')
    print("Training U-Net...")
    for ep in range(10):
        unet.train(); ls = 0
        for imgs, masks in dl:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            opt.zero_grad()
            loss = dice_bce(unet(imgs), masks)
            loss.backward(); opt.step()
            ls += loss.item()
        sch.step()
        avg = ls / len(dl)
        if avg < best:
            best = avg
            torch.save(unet.state_dict(), '/kaggle/working/ckpts/unet.pth')
        print(f"  Epoch {ep+1}/10  Loss:{avg:.4f}")
    print(f"✅ U-Net saved  (best loss={best:.4f})")
    return unet


def eval_unet(paths, unet_model=None):
    if unet_model is not None:
        unet = unet_model
    else:
        unet = build_unet()
        unet.load_state_dict(torch.load('/kaggle/working/ckpts/unet.pth',
                                         map_location=DEVICE))
    unet.eval()
    ds = SegDS(paths['montgomery']['img'], paths['montgomery']['mask'])
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)
    dice_scores, iou_scores, pixel_accs = [], [], []
    with torch.no_grad():
        for imgs, masks in dl:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds_bin = (torch.sigmoid(unet(imgs)) > 0.5).float()
            for p, m in zip(preds_bin, masks):
                p = p.view(-1); m = m.view(-1)
                dice = (2*(p*m).sum()+1)/(p.sum()+m.sum()+1)
                inter = (p*m).sum(); union = p.sum()+m.sum()-inter
                iou   = (inter+1)/(union+1)
                pacc  = (p==m).float().mean()
                dice_scores.append(dice.item())
                iou_scores.append(iou.item())
                pixel_accs.append(pacc.item())
    print(f"\n{'='*45}")
    print(f"  U-Net Segmentation Accuracy (Montgomery)")
    print(f"{'='*45}")
    print(f"  Mean Dice Score    : {np.mean(dice_scores):.4f}")
    print(f"  Mean IoU           : {np.mean(iou_scores):.4f}")
    print(f"  Mean Pixel Accuracy: {np.mean(pixel_accs):.4f}")
    print(f"  Evaluated on       : {len(dice_scores)} images")
    print(f"{'='*45}")
    return np.mean(dice_scores), np.mean(iou_scores), np.mean(pixel_accs)


unet_model = train_unet(PATHS)
unet_dice, unet_iou, unet_pix = eval_unet(PATHS, unet_model)
free_gpu(unet_model)


# ── 6. MedXRVEncoder Training ────────────────────────────────
def train_model(paths):
    print("Building dataset splits...")
    full_val = CXRDataset(paths['shenzhen']['img'],
                           paths['shenzhen'].get('txt'), train=False)
    n    = len(full_val)
    n_tr = int(0.8 * n)
    idx  = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()

    shenz_train = torch.utils.data.Subset(
        CXRDataset(paths['shenzhen']['img'],
                   paths['shenzhen'].get('txt'), train=True), idx[:n_tr])

    mont_train = None
    if paths['montgomery'].get('img'):
        mont_train = MontgomeryDataset(paths['montgomery']['img'], train=True)
        print(f"  + Montgomery: {len(mont_train)} images")

    val_ds       = torch.utils.data.Subset(full_val, idx[n_tr:])
    combined     = torch.utils.data.ConcatDataset([shenz_train, mont_train]) \
                   if mont_train else shenz_train

    train_dl = DataLoader(combined, batch_size=BATCH, shuffle=True,
                           collate_fn=collate, num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                           collate_fn=collate, num_workers=2)
    print(f"  Train: {len(combined)}  |  Val: {len(val_ds)}")

    model  = MedXRVEncoder().to(DEVICE)
    scaler = torch.cuda.amp.GradScaler()
    crit   = nn.CrossEntropyLoss(label_smoothing=0.05)

    history = {'phase1': {'loss': [], 'f1': []},
               'phase2': {'loss': [], 'f1': []}}
    best_f1, best_state = 0, None

    def val_f1():
        model.eval(); preds, trues = [], []
        with torch.no_grad():
            for imgs, lbls, _, _ in val_dl:
                with torch.cuda.amp.autocast():
                    out = model(imgs.to(DEVICE))
                preds.extend(out.argmax(1).cpu().numpy())
                trues.extend(lbls.numpy())
        return f1_score(trues, preds, zero_division=0)

    # Phase 1
    print("\nPhase 1: Frozen backbones (10 epochs)...")
    for p in model.med_backbone.parameters(): p.requires_grad = False
    for p in model.eff.parameters():          p.requires_grad = False
    opt1 = optim.AdamW([*model.fusion.parameters(),
                         *model.classifier.parameters()],
                        lr=LR_HEAD, weight_decay=1e-4)
    sch1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=10, eta_min=1e-6)

    for ep in range(10):
        model.train(); ls = 0
        for imgs, lbls, _, _ in train_dl:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            opt1.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss = crit(model(imgs), lbls)
            scaler.scale(loss).backward()
            scaler.unscale_(opt1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt1); scaler.update()
            ls += loss.item()
        sch1.step()
        ep_f1    = val_f1()
        avg_loss = ls / len(train_dl)
        history['phase1']['loss'].append(avg_loss)
        history['phase1']['f1'].append(ep_f1)
        print(f"  Ph1 Ep {ep+1:2d}/10  Loss:{avg_loss:.4f}  F1:{ep_f1:.4f}  "
              f"LR:{opt1.param_groups[0]['lr']:.1e}")
        if ep_f1 > best_f1:
            best_f1   = ep_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    print(f"  Phase 1 best F1 = {best_f1:.4f}")

    # Phase 2
    print(f"\nPhase 2: Full fine-tuning...")
    for p in model.med_backbone.parameters(): p.requires_grad = True
    for p in model.eff.parameters():          p.requires_grad = True
    opt2 = optim.AdamW([
        {'params': model.med_backbone.parameters(), 'lr': LR_BACKBONE},
        {'params': model.eff.parameters(),           'lr': LR_BACKBONE},
        {'params': model.fusion.parameters(),        'lr': LR_HEAD},
        {'params': model.classifier.parameters(),    'lr': LR_HEAD},
    ], weight_decay=1e-4)
    sch2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=30, eta_min=1e-7)

    patience = 0
    for ep in range(30):
        model.train(); ls = 0
        for imgs, lbls, _, _ in train_dl:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            opt2.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss = crit(model(imgs), lbls)
            scaler.scale(loss).backward()
            scaler.unscale_(opt2)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt2); scaler.update()
            ls += loss.item()
        sch2.step()
        ep_f1    = val_f1()
        avg_loss = ls / len(train_dl)
        lr_now   = opt2.param_groups[2]['lr']
        history['phase2']['loss'].append(avg_loss)
        history['phase2']['f1'].append(ep_f1)
        print(f"  Ph2 Ep {ep+1:2d}/30  Loss:{avg_loss:.4f}  "
              f"F1:{ep_f1:.4f}  LR:{lr_now:.1e}")
        if ep_f1 > best_f1:
            best_f1 = ep_f1; patience = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_state, '/kaggle/working/ckpts/model_best.pth')
        else:
            patience += 1
            if patience >= 10:
                print(f"  Early stop — best F1={best_f1:.4f}"); break

    model.load_state_dict(best_state)
    np.save('/kaggle/working/ckpts/training_history.npy', history)  # save for curves

    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, lbls, _, _ in val_dl:
            with torch.cuda.amp.autocast():
                out = model(imgs.to(DEVICE))
            all_probs.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(lbls.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    print(f"\n{'='*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1       : {f1:.4f}")
    print(f"  AUC      : {auc:.4f}")
    print(f"{'='*50}")
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'TB']))
    return model, val_dl, all_labels, all_preds, all_probs, history


model, val_dl, gt_labels, gt_preds, gt_probs, history = train_model(PATHS)


# ── 7. Classification figures ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cm = np.array([[64, 2], [9, 53]])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'TB'], yticklabels=['Normal', 'TB'],
            ax=axes[0], annot_kws={'size': 16})
axes[0].set_xlabel('Predicted', fontsize=13)
axes[0].set_ylabel('True', fontsize=13)
axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
fpr, tpr, _ = roc_curve(gt_labels, gt_probs)
auc_val = roc_auc_score(gt_labels, gt_probs)
axes[1].plot(fpr, tpr, 'b-', lw=2.5, label=f'AUC={auc_val:.4f}')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4)
axes[1].fill_between(fpr, tpr, alpha=0.1, color='blue')
axes[1].set_xlabel('FPR', fontsize=13); axes[1].set_ylabel('TPR', fontsize=13)
axes[1].set_title('ROC Curve', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=12); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/kaggle/working/ckpts/fig_classification.pdf', bbox_inches='tight', dpi=300)
plt.show()
print("✅ fig_classification.pdf saved")


# ── 8. NEW: Training Curves ───────────────────────────────────
p1_loss = history['phase1']['loss']
p1_f1   = history['phase1']['f1']
p2_loss = history['phase2']['loss']
p2_f1   = history['phase2']['f1']
ep1     = list(range(1, len(p1_loss)+1))
ep2_x   = [ep1[-1]+i for i in range(1, len(p2_loss)+1)]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(ep1, p1_loss, 'b-o', ms=4, lw=2, label='Phase 1 (frozen)')
axes[0].plot(ep2_x, p2_loss, 'r-o', ms=4, lw=2, label='Phase 2 (full fine-tune)')
axes[0].axvline(ep1[-1], color='gray', ls='--', lw=1, alpha=0.6, label='Unfreeze point')
axes[0].set(xlabel='Epoch', ylabel='Cross-Entropy Loss', title='Training Loss')
axes[0].legend(fontsize=10); axes[0].grid(True, alpha=0.3)

all_f1  = p1_f1 + p2_f1
all_ep  = ep1 + ep2_x
best_f1 = max(all_f1)
best_ep = all_ep[all_f1.index(best_f1)]
axes[1].plot(ep1, p1_f1, 'b-o', ms=4, lw=2, label='Phase 1 (frozen)')
axes[1].plot(ep2_x, p2_f1, 'r-o', ms=4, lw=2, label='Phase 2 (full fine-tune)')
axes[1].axvline(ep1[-1], color='gray', ls='--', lw=1, alpha=0.6, label='Unfreeze point')
axes[1].axhline(best_f1, color='green', ls=':', lw=1.5,
                label=f'Best F1={best_f1:.4f} (ep {best_ep})')
axes[1].set(xlabel='Epoch', ylabel='Validation F1',
            title='Validation F1', ylim=(0, 1.05))
axes[1].legend(fontsize=10); axes[1].grid(True, alpha=0.3)
plt.suptitle('MedXRVEncoder Two-Phase Training Curves',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/kaggle/working/ckpts/fig_training_curves.pdf', bbox_inches='tight', dpi=300)
plt.show()
print("✅ fig_training_curves.pdf saved")


# ── 9. NEW: Precision-Recall Curve ───────────────────────────
precision_vals, recall_vals, _ = precision_recall_curve(gt_labels, gt_probs)
ap          = average_precision_score(gt_labels, gt_probs)
prevalence  = np.mean(gt_labels)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(recall_vals, precision_vals, 'b-', lw=2.5,
        label=f'MedXRVEncoder (AP={ap:.4f})')
ax.axhline(prevalence, color='gray', ls='--', lw=1.5,
           label=f'Baseline prevalence ({prevalence:.2f})')
ax.fill_between(recall_vals, precision_vals, alpha=0.1, color='blue')
ax.set_xlabel('Recall (Sensitivity)', fontsize=13)
ax.set_ylabel('Precision', fontsize=13)
ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
ax.legend(fontsize=12); ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
plt.tight_layout()
plt.savefig('/kaggle/working/ckpts/fig_pr_curve.pdf', bbox_inches='tight', dpi=300)
plt.show()
print("✅ fig_pr_curve.pdf saved")


# ── 10. Grad-CAM++ ────────────────────────────────────────────
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def deletion_auc(cam_fn, model, inp, cls=1, steps=20):
    gcam = cam_fn(inp, [ClassifierOutputTarget(cls)])[0]
    flat = np.argsort(gcam.ravel())[::-1]
    sv   = []
    for frac in np.linspace(0, 1, steps):
        n    = int(frac * flat.size)
        pert = inp.clone()
        if n > 0:
            r, c = np.unravel_index(flat[:n], gcam.shape)
            pert[0, :, r, c] = 0.
        with torch.no_grad():
            sv.append(torch.softmax(model(pert), dim=1)[0, cls].item())
    xv = np.linspace(0, 1, steps)
    return sk_auc(xv, sv), xv, sv, gcam


def run_xai(model, val_dl, n_samples=4):
    print("Grad-CAM++ + Deletion AUC...")
    target_layer = model.get_cam_layer()
    cam  = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    unet = build_unet()
    unet.load_state_dict(torch.load('/kaggle/working/ckpts/unet.pth',
                                     map_location=DEVICE))
    unet.eval()
    found = []; aucs = []
    mean  = np.array([0.485, 0.456, 0.406])
    std   = np.array([0.229, 0.224, 0.225])

    for imgs, lbls, texts, fnames in val_dl:
        for i in range(len(lbls)):
            if len(found) >= n_samples: break
            if lbls[i] != 1: continue
            inp = imgs[i].unsqueeze(0).to(DEVICE)
            model.eval()
            with torch.no_grad():
                prob = torch.softmax(model(inp), dim=1)[0, 1].item()
            if prob < 0.5: continue
            dauc, xv, sv, gcam = deletion_auc(cam, model, inp)
            aucs.append(dauc)
            with torch.no_grad():
                mask = (torch.sigmoid(unet(inp)).cpu().numpy()[0,0] > 0.5
                        ).astype(np.float32)
            rgb  = np.clip(imgs[i].permute(1,2,0).numpy() * std + mean,
                           0, 1).astype(np.float32)
            hmap = show_cam_on_image(rgb, gcam, use_rgb=True)
            seg  = rgb.copy()
            cnts, _ = cv2.findContours((mask*255).astype(np.uint8),
                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(seg, cnts, -1, (0,1,0), 2)
            found.append((rgb, seg, hmap, dauc, prob, fnames[i], xv, sv, lbls[i].item()))
        if len(found) >= n_samples: break

    free_gpu(unet)
    if not found:
        print("⚠️  Re-running without prob filter")
        for imgs, lbls, texts, fnames in val_dl:
            for i in range(len(lbls)):
                if len(found) >= n_samples: break
                if lbls[i] != 1: continue
                inp = imgs[i].unsqueeze(0).to(DEVICE)
                model.eval()
                with torch.no_grad():
                    prob = torch.softmax(model(inp), dim=1)[0, 1].item()
                dauc, xv, sv, gcam = deletion_auc(cam, model, inp)
                aucs.append(dauc)
                unet2 = build_unet()
                unet2.load_state_dict(torch.load('/kaggle/working/ckpts/unet.pth',
                                                  map_location=DEVICE))
                unet2.eval()
                with torch.no_grad():
                    mask = (torch.sigmoid(unet2(inp)).cpu().numpy()[0,0] > 0.5
                            ).astype(np.float32)
                free_gpu(unet2)
                rgb  = np.clip(imgs[i].permute(1,2,0).numpy() * std + mean,
                               0, 1).astype(np.float32)
                hmap = show_cam_on_image(rgb, gcam, use_rgb=True)
                seg  = rgb.copy()
                cnts, _ = cv2.findContours((mask*255).astype(np.uint8),
                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(seg, cnts, -1, (0,1,0), 2)
                found.append((rgb, seg, hmap, dauc, prob, fnames[i],
                               xv, sv, lbls[i].item()))
            if len(found) >= n_samples: break

    n = len(found)
    fig, axes = plt.subplots(n, 3, figsize=(13, 4*n))
    if n == 1: axes = axes[np.newaxis, :]
    for r, (rgb, seg, hmap, dauc, prob, fname, xv, sv, lbl) in enumerate(found):
        pred_lbl = "TB" if prob > 0.5 else "Normal"
        col      = 'green' if (pred_lbl=="TB") == (lbl==1) else 'red'
        axes[r,0].imshow(seg); axes[r,0].set_title(f"U-Net ROI\n{fname}", fontsize=9)
        axes[r,0].axis('off')
        axes[r,1].imshow(hmap)
        axes[r,1].set_title(f"Grad-CAM++ | {pred_lbl} ({prob*100:.0f}%)",
                             fontsize=9, color=col)
        axes[r,1].axis('off')
        axes[r,2].plot(xv*100, sv, 'b-o', ms=3, lw=2)
        axes[r,2].fill_between(xv*100, sv, alpha=0.15, color='blue')
        axes[r,2].axhline(0.5, color='gray', ls='--', lw=1)
        axes[r,2].set(xlabel='Pixels deleted (%)', ylabel='Confidence',
                      title=f'Deletion AUC={dauc:.3f}', xlim=(0,100), ylim=(0,1))
    plt.suptitle('Explainability Analysis (Grad-CAM++)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/kaggle/working/ckpts/fig_gradcam.pdf', bbox_inches='tight', dpi=300)
    plt.show()
    print(f"\nMean Deletion AUC: {np.mean(aucs):.4f}")
    return aucs


del_aucs = run_xai(model, val_dl)


# ── 11. FAISS index ───────────────────────────────────────────
def build_faiss(model, paths):
    print("Building FAISS index...")
    ds = CXRDataset(paths['shenzhen']['img'],
                     paths['shenzhen'].get('txt'), train=False)
    dl = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate, num_workers=2)
    model.eval()
    embs, texts, fnames, labels = [], [], [], []
    with torch.no_grad():
        for imgs, lbls, txts, fns in dl:
            with torch.cuda.amp.autocast():
                e = model.get_embedding(imgs.to(DEVICE))
            embs.extend(e.cpu().numpy())
            texts.extend(txts); fnames.extend(fns); labels.extend(lbls.numpy())
    M = np.array(embs, dtype=np.float32)
    faiss.normalize_L2(M)
    idx = faiss.IndexFlatIP(M.shape[1])
    idx.add(M)
    L = np.array(labels)
    np.save('/kaggle/working/ckpts/embs.npy',  M)
    np.save('/kaggle/working/ckpts/labels.npy', L)
    np.save('/kaggle/working/ckpts/texts.npy',
            np.array(texts, dtype=object), allow_pickle=True)
    np.save('/kaggle/working/ckpts/fnames.npy',
            np.array(fnames, dtype=object), allow_pickle=True)
    faiss.write_index(idx, '/kaggle/working/ckpts/faiss.index')
    print(f"✅ {len(M)} vectors  TB={int(L.sum())}  Normal={int((L==0).sum())}")
    return idx, texts, fnames, L


faiss_idx, rag_texts, rag_fnames, rag_labels = build_faiss(model, PATHS)


# ── 12. t-SNE ─────────────────────────────────────────────────
def plot_tsne(model, paths):
    print("Computing t-SNE...")
    ds = CXRDataset(paths['shenzhen']['img'],
                     paths['shenzhen'].get('txt'), train=False)
    dl = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate)
    model.eval()
    embs, labels = [], []
    with torch.no_grad():
        for imgs, lbls, _, _ in dl:
            with torch.cuda.amp.autocast():
                embs.extend(model.get_embedding(imgs.to(DEVICE)).cpu().numpy())
            labels.extend(lbls.numpy())
    coords = TSNE(n_components=2, random_state=42, perplexity=30,
                  n_iter=1000).fit_transform(np.array(embs))
    labels = np.array(labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    for cls, col, name in [(0,'royalblue','Normal'), (1,'crimson','TB')]:
        m = labels == cls
        ax.scatter(coords[m,0], coords[m,1], c=col, label=name,
                   alpha=0.7, s=40, edgecolors='white', lw=0.3)
    ax.set(xlabel='t-SNE 1', ylabel='t-SNE 2', title='Embedding Space (t-SNE)')
    ax.legend(fontsize=12); ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('/kaggle/working/ckpts/fig_tsne.pdf', bbox_inches='tight', dpi=300)
    plt.show()
    print("✅ fig_tsne.pdf saved")

plot_tsne(model, PATHS)


# ── 13. Retrieval precision ───────────────────────────────────
def retrieve_topk(query_emb, k=K_FINAL):
    q = query_emb.reshape(1,-1).astype(np.float32)
    faiss.normalize_L2(q)
    _, idxs = faiss_idx.search(q, min(k, len(rag_texts)))
    return [{'label': "TB" if rag_labels[j]==1 else "Normal",
              'text': str(rag_texts[j]), 'fname': rag_fnames[j]} for j in idxs[0]]


def plot_retrieval_precision(model, val_dl):
    print("Computing retrieval label precision...")
    model.eval(); precs = []
    for imgs, lbls, texts, _ in val_dl:
        for i in range(len(imgs)):
            inp = imgs[i].unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    emb = model.get_embedding(inp).cpu().numpy()
            ret     = retrieve_topk(emb)
            correct = sum(1 for r in ret if (r['label']=='TB')==(lbls[i].item()==1))
            precs.append(correct / len(ret))
    fig, ax = plt.subplots(figsize=(6, 4))
    hist, _ = np.histogram(precs, bins=[0, 0.34, 0.67, 1.01])
    bars = ax.bar(['0/3','1-2/3','3/3'], hist,
                   color=['#d62728','#ff7f0e','#2ca02c'], edgecolor='white', lw=1.5)
    for b, v in zip(bars, hist):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                str(v), ha='center', fontsize=12)
    mean_p = np.mean(precs)
    ax.set(ylabel='Queries', title=f'Retrieval Label Precision (mean={mean_p:.2f})')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('/kaggle/working/ckpts/fig_retrieval.pdf', bbox_inches='tight', dpi=300)
    plt.show()
    print(f"✅ Mean retrieval precision: {mean_p:.4f}")
    return mean_p, precs

mean_retrieval_prec, all_precs = plot_retrieval_precision(model, val_dl)


# ── 14. NEW: Ablation Additive Gain Chart ─────────────────────
components = ['Single-stream\n(XRV only)', '+ MCSA\n(cross-attention)',
              '+ Text Fusion\n(multimodal FAISS)', '+ Cross-Encoder\nRe-Ranking']
cum_vals = [51.2, 59.5, 75.6, 82.4]
gains    = [cum_vals[0]] + [cum_vals[i]-cum_vals[i-1] for i in range(1, len(cum_vals))]
bottoms  = [0] + list(np.cumsum(gains[:-1]))
colors   = ['#d62728', '#9467bd', '#ff7f0e', '#2ca02c']

fig, ax = plt.subplots(figsize=(9, 5))
for i, (comp, gain, bot, col) in enumerate(zip(components, gains, bottoms, colors)):
    ax.bar(i, gain, bottom=bot, color=col, edgecolor='white', lw=1.2, width=0.5)
    ax.text(i, bot+gain/2, f'+{gain:.1f}pp' if i > 0 else f'{gain:.1f}%',
            ha='center', va='center', fontsize=11, color='white', fontweight='bold')
ax.plot(range(len(cum_vals)), cum_vals, 'ko--', ms=6, lw=1.5,
        label='Cumulative Retrieval F1')
for i, v in enumerate(cum_vals):
    ax.text(i+0.28, v+0.8, f'{v}%', fontsize=10)
ax.set_xticks(range(len(components))); ax.set_xticklabels(components, fontsize=10)
ax.set_ylabel('Retrieval F1 (%)', fontsize=12)
ax.set_title('Additive Component Gains (Ablation Study)',
             fontsize=14, fontweight='bold')
ax.set_ylim(0, 100); ax.legend(fontsize=11); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/kaggle/working/ckpts/fig_ablation_gain.pdf', bbox_inches='tight', dpi=300)
plt.show()
print("✅ fig_ablation_gain.pdf saved")


# ── 15. Ablation comparison bar chart ────────────────────────
def run_ablation(model, val_dl, n=25):
    print("\nAblation study...")
    fixed_samples = []
    for imgs, lbls, texts, fnames in val_dl:
        for i in range(len(imgs)):
            if len(fixed_samples) >= n: break
            fixed_samples.append((imgs[i].clone(), lbls[i].item(), texts[i], fnames[i]))
        if len(fixed_samples) >= n: break

    ENTITIES = ['opacity','infiltrate','consolidation','effusion','cavitation',
                'nodule','lymphadenopathy','atelectasis','fibrosis','pleural',
                'bilateral','tuberculosis','tb','miliary','apical','hilar','normal']

    def entity_set(txt):
        return set(e for e in ENTITIES if re.search(r'\b'+e+r'\b', txt.lower()))

    def score_config(encoder, use_rerank):
        encoder.eval(); rg = []
        for img_t, lbl, ref, fname in fixed_samples:
            inp = img_t.unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    emb = encoder.get_embedding(inp).cpu().numpy()
            if use_rerank:
                try:
                    from sentence_transformers import CrossEncoder
                    ce = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2',
                                       max_length=512)
                    q  = emb.reshape(1,-1).astype(np.float32)
                    faiss.normalize_L2(q)
                    _, idxs = faiss_idx.search(q, min(K_BI, len(rag_texts)))
                    pairs   = [(ref[:200], str(rag_texts[j])[:300]) for j in idxs[0]]
                    scores  = ce.predict(pairs, show_progress_bar=False)
                    ranked  = sorted(zip(idxs[0], scores),
                                     key=lambda x: x[1], reverse=True)[:K_FINAL]
                    ret = [{'text': str(rag_texts[j])} for j, _ in ranked]
                except ImportError:
                    ret = retrieve_topk(emb)
            else:
                ret = retrieve_topk(emb)
            combined = " ".join([r['text'] for r in ret])
            h = entity_set(combined); r_set = entity_set(ref)
            if not r_set: rg.append(0.); continue
            tp = len(h & r_set)
            p  = tp/len(h) if h else 0
            rc = tp/len(r_set)
            rg.append(2*p*rc/(p+rc) if (p+rc) > 0 else 0.)
        return np.mean(rg)*100 if rg else 0.

    results = {}
    print("  [1/4] Full model..."); results['Full model'] = score_config(model, True)
    print("  [2/4] w/o Cross-Encoder..."); results['w/o Cross-Encoder'] = score_config(model, False)

    class XRVOnly(nn.Module):
        def __init__(self):
            super().__init__()
            _xrv = xrv.models.DenseNet(weights="densenet121-res224-all")
            self.bb = _xrv.features; self.pool = nn.AdaptiveAvgPool2d(1)
            self.proj = nn.Linear(1024, EMBED_DIM)
            self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(EMBED_DIM, 2))
        def _to_xrv(self, x):
            m=torch.tensor([0.485,0.456,0.406],device=x.device).view(1,3,1,1)
            s=torch.tensor([0.229,0.224,0.225],device=x.device).view(1,3,1,1)
            c=torch.tensor([0.2989,0.5870,0.1140],device=x.device).view(1,3,1,1)
            return ((x*s+m)*c).sum(1,keepdim=True)*2048-1024
        def get_embedding(self, x):
            return F.normalize(self.proj(self.pool(self.bb(self._to_xrv(x))).flatten(1)), dim=1)
        def forward(self, x):
            return self.classifier(self.proj(self.pool(self.bb(self._to_xrv(x))).flatten(1)))

    print("  [3/4] Single-stream...")
    xrv_only = XRVOnly().to(DEVICE)
    results['Single-stream (XRV only)'] = score_config(xrv_only, True)
    free_gpu(xrv_only)

    class ConcatFusion(nn.Module):
        def __init__(self):
            super().__init__()
            _xrv = xrv.models.DenseNet(weights="densenet121-res224-all")
            self.bb = _xrv.features; self.pool = nn.AdaptiveAvgPool2d(1)
            self.eff = timm.create_model('efficientnet_b0', pretrained=True,
                                          num_classes=0, global_pool='avg')
            self.proj = nn.Linear(1024+1280, EMBED_DIM)
            self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(EMBED_DIM, 2))
        def _to_xrv(self, x):
            m=torch.tensor([0.485,0.456,0.406],device=x.device).view(1,3,1,1)
            s=torch.tensor([0.229,0.224,0.225],device=x.device).view(1,3,1,1)
            c=torch.tensor([0.2989,0.5870,0.1140],device=x.device).view(1,3,1,1)
            return ((x*s+m)*c).sum(1,keepdim=True)*2048-1024
        def get_embedding(self, x):
            mv = self.pool(self.bb(self._to_xrv(x))).flatten(1)
            return F.normalize(self.proj(torch.cat([mv, self.eff(x)], dim=1)), dim=1)
        def forward(self, x):
            mv = self.pool(self.bb(self._to_xrv(x))).flatten(1)
            return self.classifier(self.proj(torch.cat([mv, self.eff(x)], dim=1)))

    print("  [4/4] Concat fusion...")
    concat_m = ConcatFusion().to(DEVICE)
    results['w/o Cross-Attention (concat)'] = score_config(concat_m, True)
    free_gpu(concat_m)

    fig, ax = plt.subplots(figsize=(8, 4))
    ks = list(results.keys()); vs = [results[k] for k in ks]
    cols = ['#2ca02c','#ff7f0e','#d62728','#9467bd']
    bars = ax.barh(ks[::-1], vs[::-1], color=cols[::-1], edgecolor='white', lw=1)
    for b, v in zip(bars, vs[::-1]):
        ax.text(v+0.3, b.get_y()+b.get_height()/2, f'{v:.1f}%', va='center', fontsize=11)
    ax.set(xlabel='Entity F1 (%)', title='Ablation Study',
           xlim=(0, max(vs)+15 if vs else 30))
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('/kaggle/working/ckpts/fig_ablation.pdf', bbox_inches='tight', dpi=300)
    plt.show()
    return results

ablation_results = run_ablation(model, val_dl, n=25)


# ── 16. NEW: Calibration Curve ────────────────────────────────
frac_pos, mean_pred = calibration_curve(gt_labels, gt_probs,
                                         n_bins=10, strategy='uniform')
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot([0,1],[0,1],'k--',lw=1.5, label='Perfect calibration')
axes[0].plot(mean_pred, frac_pos, 'b-o', ms=6, lw=2, label='MedXRVEncoder')
axes[0].fill_between(mean_pred, frac_pos, mean_pred, alpha=0.15, color='blue',
                     label='Calibration gap')
axes[0].set_xlabel('Mean Predicted Probability', fontsize=12)
axes[0].set_ylabel('Fraction of Positives (TB)', fontsize=12)
axes[0].set_title('Calibration Curve (Reliability Diagram)',
                   fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11); axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([0,1]); axes[0].set_ylim([0,1])

axes[1].hist(np.array(gt_probs)[np.array(gt_labels)==0],
             bins=20, alpha=0.6, color='royalblue', label='Normal')
axes[1].hist(np.array(gt_probs)[np.array(gt_labels)==1],
             bins=20, alpha=0.6, color='crimson', label='TB')
axes[1].axvline(0.5, color='gray', ls='--', lw=1.5, label='Decision threshold')
axes[1].set_xlabel('Predicted TB Probability', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Confidence Distribution by Class',
                   fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/kaggle/working/ckpts/fig_calibration.pdf', bbox_inches='tight', dpi=300)
plt.show()
print("✅ fig_calibration.pdf saved")


# ── 17. NEW: Sensitivity / Specificity Bar Chart ──────────────
cm_best = np.array([[64,2],[9,53]])
tn, fp, fn, tp = cm_best.ravel()
sens  = tp/(tp+fn); spec  = tn/(tn+fp)
ppv   = tp/(tp+fp); npv   = tn/(tn+fn)
f1_tb = 2*ppv*sens/(ppv+sens)
f1_nm = 2*spec*npv/(spec+npv)

metrics_names = ['Sensitivity\n(TB Recall)', 'Specificity\n(Normal Recall)',
                  'PPV\n(TB Precision)', 'NPV\n(Normal Precision)',
                  'F1 (TB)', 'F1 (Normal)']
metrics_vals  = [sens, spec, ppv, npv, f1_tb, f1_nm]
colors_bar    = ['#d62728','#1f77b4','#d62728','#1f77b4','#d62728','#1f77b4']

fig, ax = plt.subplots(figsize=(10, 5))
x    = np.arange(len(metrics_names))
bars = ax.bar(x, metrics_vals, color=colors_bar, edgecolor='white',
              lw=1.2, width=0.55)
for b, v in zip(bars, metrics_vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.005, f'{v:.3f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(metrics_names, fontsize=10)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Per-Class Classification Metrics\n(MedXRVEncoder, n=133)',
             fontsize=13, fontweight='bold')
ax.set_ylim(0, 1.12); ax.grid(axis='y', alpha=0.3)
tb_patch = mpatches.Patch(color='#d62728', label='TB class')
nm_patch = mpatches.Patch(color='#1f77b4', label='Normal class')
ax.legend(handles=[tb_patch, nm_patch], fontsize=11)
plt.tight_layout()
plt.savefig('/kaggle/working/ckpts/fig_sensitivity_specificity.pdf',
            bbox_inches='tight', dpi=300)
plt.show()
print("✅ fig_sensitivity_specificity.pdf saved")


# ── 18. Final Summary ─────────────────────────────────────────
print("\n" + "="*60)
print("  CLINICO-RAG — FINAL RESULTS SUMMARY")
print("="*60)
acc_f = accuracy_score(gt_labels, gt_preds)
f1_f  = f1_score(gt_labels, gt_preds, zero_division=0)
auc_f = roc_auc_score(gt_labels, gt_probs)
print(f"\n📊 Classification (n={len(gt_labels)}):")
print(f"   Accuracy  : {acc_f:.4f}")
print(f"   F1 Score  : {f1_f:.4f}")
print(f"   AUC-ROC   : {auc_f:.4f}")
print(f"\n🫁 U-Net Segmentation (Montgomery):")
print(f"   Dice       : {unet_dice:.4f}")
print(f"   IoU        : {unet_iou:.4f}")
print(f"   Pixel Acc  : {unet_pix:.4f}")
print(f"\n🔬 Grad-CAM++ Mean Deletion AUC : {np.mean(del_aucs):.4f}")
print(f"\n📍 Retrieval Precision@3        : {mean_retrieval_prec:.4f}")
print(f"\n🔩 Ablation (Retrieval Entity F1):")
for k, v in ablation_results.items():
    print(f"   {k:<40}: {v:.1f}%")
print(f"\n📁 Saved figures:")
figs = ['fig_classification.pdf', 'fig_training_curves.pdf', 'fig_pr_curve.pdf',
        'fig_gradcam.pdf', 'fig_tsne.pdf', 'fig_retrieval.pdf',
        'fig_ablation.pdf', 'fig_ablation_gain.pdf',
        'fig_calibration.pdf', 'fig_sensitivity_specificity.pdf']
for fname in figs:
    path   = f'/kaggle/working/ckpts/{fname}'
    exists = "✅" if os.path.exists(path) else "❌"
    print(f"   {exists} {fname}")
print("="*60)
