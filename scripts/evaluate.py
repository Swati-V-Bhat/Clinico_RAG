#!/usr/bin/env python3
# scripts/evaluate.py
"""
Full Clinico-RAG evaluation pipeline.
Loads trained checkpoints and generates all paper figures.

Usage:
    python scripts/evaluate.py --config configs/default.yaml
"""
import os
import sys
import argparse
import yaml
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import MedXRVEncoder, build_unet
from src.train import eval_unet, _full_eval
from src.dataset import CXRDataset, collate_fn
from src.retrieval import (build_faiss_index, retrieve_topk,
                             retrieve_with_rerank, entity_f1)
from src.explainability import run_gradcam
from src.visualize import (
    plot_classification, plot_gradcam, plot_tsne,
    plot_retrieval_precision, plot_ablation,
    plot_training_curves, plot_pr_curve, plot_ablation_gain,
    plot_calibration, plot_sensitivity_specificity,
)
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def main(cfg):
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = cfg['paths']['output_dir']
    os.makedirs(out_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────
    print("Loading MedXRVEncoder checkpoint...")
    model = MedXRVEncoder().to(device)
    ckpt  = os.path.join(out_dir, 'model_best.pth')
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print("✅ Model loaded")

    # ── Val dataloader ────────────────────────────────────────
    full_ds = CXRDataset(cfg['paths']['shenzhen_img'],
                          cfg['paths'].get('shenzhen_txt'), train=False)
    n    = len(full_ds)
    n_tr = int((1 - cfg['training']['val_split']) * n)
    g    = torch.Generator(); g.manual_seed(cfg['training']['seed'])
    idx  = torch.randperm(n, generator=g).tolist()
    from torch.utils.data import Subset
    val_ds = Subset(full_ds, idx[n_tr:])
    val_dl = DataLoader(val_ds, batch_size=cfg['training']['batch_size'],
                         shuffle=False, collate_fn=collate_fn, num_workers=2)

    # ── Classification metrics ────────────────────────────────
    labels, preds, probs = _full_eval(model, val_dl, device)

    # ── U-Net evaluation ──────────────────────────────────────
    unet_results = eval_unet(cfg, device)

    # ── FAISS index ───────────────────────────────────────────
    faiss_idx, rag_texts, rag_fnames, rag_labels = \
        build_faiss_index(model, cfg, device)

    # ── Retrieval precision ───────────────────────────────────
    print("Computing retrieval label precision...")
    model.eval(); precs = []
    for imgs, lbls, texts, _ in val_dl:
        for i in range(len(imgs)):
            inp = imgs[i].unsqueeze(0).to(device)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    emb = model.get_embedding(inp).cpu().numpy()
            ret     = retrieve_topk(emb, faiss_idx, rag_labels,
                                     rag_texts, rag_fnames,
                                     k=cfg['retrieval']['k_final'])
            correct = sum(1 for r in ret
                          if (r['label'] == 'TB') == (lbls[i].item() == 1))
            precs.append(correct / len(ret))
    mean_prec = np.mean(precs)
    print(f"  Mean retrieval precision: {mean_prec:.4f}")

    # ── Grad-CAM++ ────────────────────────────────────────────
    print("Running Grad-CAM++...")
    unet = build_unet(device)
    unet.load_state_dict(torch.load(os.path.join(out_dir, 'unet.pth'),
                                     map_location=device))
    unet.eval()
    found, del_aucs = run_gradcam(model, val_dl, unet, device, n_samples=4)
    unet.cpu()

    # ── Ablation ──────────────────────────────────────────────
    from scripts.run_ablation import run_ablation
    ablation_results = run_ablation(model, val_dl, faiss_idx,
                                     rag_texts, rag_fnames, rag_labels, cfg, device)

    # ══════════════════════════════════════════════════════════
    # Generate all figures
    # ══════════════════════════════════════════════════════════
    print("\nGenerating figures...")

    plot_classification(labels, preds, probs, out_dir)
    plot_gradcam(found, out_dir)
    plot_tsne(model, cfg['paths'], device, out_dir)
    plot_retrieval_precision(precs, out_dir)
    plot_ablation(ablation_results, out_dir)

    # New figures
    plot_pr_curve(labels, probs, out_dir)
    plot_ablation_gain(out_dir)
    plot_calibration(labels, probs, out_dir)
    plot_sensitivity_specificity(labels, preds, probs, out_dir)

    # Training curves — loaded from saved history if available
    history_path = os.path.join(out_dir, 'training_history.npy')
    if os.path.exists(history_path):
        history = np.load(history_path, allow_pickle=True).item()
        plot_training_curves(history, out_dir)
    else:
        print("⚠️  training_history.npy not found — "
              "training curves will be generated during training")

    # ── Final summary ─────────────────────────────────────────
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, zero_division=0)
    auc = roc_auc_score(labels, probs)

    print(f"\n{'='*60}")
    print(f"  CLINICO-RAG — FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"\n📊 Classification (n={len(labels)}):")
    print(f"   Accuracy  : {acc:.4f}")
    print(f"   F1 Score  : {f1:.4f}")
    print(f"   AUC-ROC   : {auc:.4f}")
    print(f"\n🫁 U-Net Segmentation:")
    print(f"   Dice       : {unet_results['dice']:.4f}")
    print(f"   IoU        : {unet_results['iou']:.4f}")
    print(f"   Pixel Acc  : {unet_results['pixel_acc']:.4f}")
    print(f"\n🔬 Grad-CAM++ Mean Deletion AUC : {np.mean(del_aucs):.4f}")
    print(f"\n📍 Retrieval Precision@3        : {mean_prec:.4f}")
    print(f"\n📁 Figures saved to: {out_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
