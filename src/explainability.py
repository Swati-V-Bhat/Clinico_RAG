# src/explainability.py
import cv2
import numpy as np
import torch
from sklearn.metrics import auc as sk_auc

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def deletion_auc(cam_fn, model, inp, cls=1, steps=20):
    """
    Deletion AUC (Eq. 8 in paper): progressively zero most-salient pixels
    and measure TB-class confidence drop.
    Lower = more faithful explanation.
    """
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


def run_gradcam(model, val_dl, unet, device, n_samples=4):
    """
    Run Grad-CAM++ on n_samples TB-positive validation cases.
    Returns list of (rgb, seg, hmap, dauc, prob, fname, xv, sv, lbl).
    """
    target_layer = model.get_cam_layer()
    cam          = GradCAMPlusPlus(model=model, target_layers=[target_layer])

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    found = []; aucs = []

    def _process_sample(img_t, lbl, prob, fname):
        inp  = img_t.unsqueeze(0).to(device)
        dauc, xv, sv, gcam = deletion_auc(cam, model, inp)
        aucs.append(dauc)

        with torch.no_grad():
            mask = (torch.sigmoid(unet(inp)).cpu().numpy()[0, 0] > 0.5
                    ).astype(np.float32)

        rgb  = np.clip(img_t.permute(1, 2, 0).numpy() * std + mean,
                       0, 1).astype(np.float32)
        hmap = show_cam_on_image(rgb, gcam, use_rgb=True)
        seg  = rgb.copy()
        cnts, _ = cv2.findContours((mask * 255).astype(np.uint8),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(seg, cnts, -1, (0, 1, 0), 2)
        found.append((rgb, seg, hmap, dauc, prob, fname, xv, sv, lbl))

    # Pass 1: correctly predicted TB cases only
    for imgs, lbls, texts, fnames in val_dl:
        for i in range(len(lbls)):
            if len(found) >= n_samples: break
            if lbls[i] != 1: continue
            inp = imgs[i].unsqueeze(0).to(device)
            model.eval()
            with torch.no_grad():
                prob = torch.softmax(model(inp), dim=1)[0, 1].item()
            if prob >= 0.5:
                _process_sample(imgs[i], lbls[i].item(), prob, fnames[i])
        if len(found) >= n_samples: break

    # Pass 2: if not enough, drop the prob filter
    if len(found) < n_samples:
        print("⚠️  Re-running without prob filter")
        for imgs, lbls, texts, fnames in val_dl:
            for i in range(len(lbls)):
                if len(found) >= n_samples: break
                if lbls[i] != 1: continue
                inp = imgs[i].unsqueeze(0).to(device)
                model.eval()
                with torch.no_grad():
                    prob = torch.softmax(model(inp), dim=1)[0, 1].item()
                _process_sample(imgs[i], lbls[i].item(), prob, fnames[i])
            if len(found) >= n_samples: break

    return found, aucs
