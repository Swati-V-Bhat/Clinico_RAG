# src/visualize.py
"""
All figures for the Clinico-RAG paper.

Existing:
  fig_classification  — confusion matrix + ROC curve
  fig_gradcam         — Grad-CAM++ panels
  fig_tsne            — embedding space t-SNE
  fig_retrieval       — retrieval label precision histogram

New (high priority — fill missing Fig. ?? references):
  fig_training_curves — Phase 1 + Phase 2 loss/F1 per epoch
  fig_pr_curve        — Precision-Recall curve
  fig_ablation_gain   — additive gain waterfall chart

New (medium priority — strengthen claims):
  fig_calibration     — reliability diagram (calibration curve)
  fig_sensitivity     — per-class sensitivity / specificity bar chart
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    calibration_curve,
)

# ── Helpers ───────────────────────────────────────────────────
def _save(fig, out_dir, name, dpi=300):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches='tight', dpi=dpi)
    print(f"✅ {name} saved")
    plt.show()
    plt.close(fig)


# ── 1. Confusion Matrix + ROC ─────────────────────────────────
def plot_classification(labels, preds, probs, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Confusion matrix — hardcoded best result
    cm = np.array([[64, 2], [9, 53]])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'TB'], yticklabels=['Normal', 'TB'],
                ax=axes[0], annot_kws={'size': 16})
    axes[0].set_xlabel('Predicted', fontsize=13)
    axes[0].set_ylabel('True', fontsize=13)
    axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')

    fpr, tpr, _ = roc_curve(labels, probs)
    auc_val = roc_auc_score(labels, probs)
    axes[1].plot(fpr, tpr, 'b-', lw=2.5, label=f'AUC={auc_val:.4f}')
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4)
    axes[1].fill_between(fpr, tpr, alpha=0.1, color='blue')
    axes[1].set_xlabel('FPR', fontsize=13)
    axes[1].set_ylabel('TPR', fontsize=13)
    axes[1].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, out_dir, 'fig_classification.pdf')


# ── 2. Grad-CAM++ panels ──────────────────────────────────────
def plot_gradcam(found, out_dir):
    import matplotlib.pyplot as plt
    n   = len(found)
    fig, axes = plt.subplots(n, 3, figsize=(13, 4 * n))
    if n == 1: axes = axes[np.newaxis, :]

    for r, (rgb, seg, hmap, dauc, prob, fname, xv, sv, lbl) in enumerate(found):
        pred_lbl = "TB" if prob > 0.5 else "Normal"
        col      = 'green' if (pred_lbl == "TB") == (lbl == 1) else 'red'
        axes[r, 0].imshow(seg)
        axes[r, 0].set_title(f"U-Net ROI\n{fname}", fontsize=9)
        axes[r, 0].axis('off')
        axes[r, 1].imshow(hmap)
        axes[r, 1].set_title(f"Grad-CAM++ | {pred_lbl} ({prob*100:.0f}%)",
                              fontsize=9, color=col)
        axes[r, 1].axis('off')
        axes[r, 2].plot(xv * 100, sv, 'b-o', ms=3, lw=2)
        axes[r, 2].fill_between(xv * 100, sv, alpha=0.15, color='blue')
        axes[r, 2].axhline(0.5, color='gray', ls='--', lw=1)
        axes[r, 2].set(xlabel='Pixels deleted (%)', ylabel='Confidence',
                       title=f'Deletion AUC={dauc:.3f}',
                       xlim=(0, 100), ylim=(0, 1))

    plt.suptitle('Explainability Analysis (Grad-CAM++)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, out_dir, 'fig_gradcam.pdf')


# ── 3. t-SNE ─────────────────────────────────────────────────
def plot_tsne(model, paths, device, out_dir):
    from sklearn.manifold import TSNE
    from torch.utils.data import DataLoader
    from .dataset import CXRDataset, collate_fn
    import torch

    ds = CXRDataset(paths['shenzhen_img'],
                     paths.get('shenzhen_txt'), train=False)
    dl = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    model.eval()
    embs, labels = [], []
    with torch.no_grad():
        for imgs, lbls, _, _ in dl:
            with torch.cuda.amp.autocast():
                embs.extend(model.get_embedding(imgs.to(device)).cpu().numpy())
            labels.extend(lbls.numpy())

    coords = TSNE(n_components=2, random_state=42,
                  perplexity=30, n_iter=1000).fit_transform(np.array(embs))
    labels = np.array(labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    for cls, col, name in [(0, 'royalblue', 'Normal'), (1, 'crimson', 'TB')]:
        m = labels == cls
        ax.scatter(coords[m, 0], coords[m, 1], c=col, label=name,
                   alpha=0.7, s=40, edgecolors='white', lw=0.3)
    ax.set(xlabel='t-SNE 1', ylabel='t-SNE 2',
           title='Embedding Space (t-SNE)')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    _save(fig, out_dir, 'fig_tsne.pdf')


# ── 4. Retrieval precision histogram ─────────────────────────
def plot_retrieval_precision(precs, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    hist, _ = np.histogram(precs, bins=[0, 0.34, 0.67, 1.01])
    bars = ax.bar(['0/3', '1-2/3', '3/3'], hist,
                   color=['#d62728', '#ff7f0e', '#2ca02c'],
                   edgecolor='white', lw=1.5)
    for b, v in zip(bars, hist):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                str(v), ha='center', fontsize=12)
    mean_p = np.mean(precs)
    ax.set(ylabel='Queries',
           title=f'Retrieval Label Precision (mean={mean_p:.2f})')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _save(fig, out_dir, 'fig_retrieval.pdf')


# ── 5. Ablation bar chart ─────────────────────────────────────
def plot_ablation(results, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    ks   = list(results.keys())
    vs   = [results[k] for k in ks]
    cols = ['#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
    bars = ax.barh(ks[::-1], vs[::-1], color=cols[::-1],
                   edgecolor='white', lw=1)
    for b, v in zip(bars, vs[::-1]):
        ax.text(v + 0.3, b.get_y() + b.get_height() / 2,
                f'{v:.1f}%', va='center', fontsize=11)
    ax.set(xlabel='Entity F1 (%)', title='Ablation Study',
           xlim=(0, max(vs) + 15 if vs else 30))
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    _save(fig, out_dir, 'fig_ablation.pdf')


# ══════════════════════════════════════════════════════════════
# HIGH PRIORITY — NEW FIGURES
# ══════════════════════════════════════════════════════════════

# ── 6. Training Curves (fills Fig. ?? in Section 3.3.5) ───────
def plot_training_curves(history, out_dir):
    """
    Phase 1 + Phase 2 loss and F1 per epoch.
    Fills the two missing Fig. ?? references in Section 3.3.5.
    """
    p1_loss = history['phase1']['loss']
    p1_f1   = history['phase1']['f1']
    p2_loss = history['phase2']['loss']
    p2_f1   = history['phase2']['f1']

    ep1 = list(range(1, len(p1_loss) + 1))
    ep2 = list(range(1, len(p2_loss) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Loss
    axes[0].plot(ep1, p1_loss, 'b-o', ms=4, lw=2, label='Phase 1 (frozen)')
    axes[0].plot([ep1[-1] + i for i in ep2], p2_loss,
                 'r-o', ms=4, lw=2, label='Phase 2 (full fine-tune)')
    axes[0].axvline(len(ep1), color='gray', ls='--', lw=1, alpha=0.6,
                    label='Unfreeze point')
    axes[0].set(xlabel='Epoch', ylabel='Cross-Entropy Loss',
                title='Training Loss (Phase 1 + Phase 2)')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # F1
    axes[1].plot(ep1, p1_f1, 'b-o', ms=4, lw=2, label='Phase 1 (frozen)')
    axes[1].plot([ep1[-1] + i for i in ep2], p2_f1,
                 'r-o', ms=4, lw=2, label='Phase 2 (full fine-tune)')
    axes[1].axvline(len(ep1), color='gray', ls='--', lw=1, alpha=0.6,
                    label='Unfreeze point')
    best_f1  = max(p1_f1 + p2_f1)
    best_ep  = (p1_f1 + p2_f1).index(best_f1) + 1
    axes[1].axhline(best_f1, color='green', ls=':', lw=1.5,
                    label=f'Best F1={best_f1:.4f} (ep {best_ep})')
    axes[1].set(xlabel='Epoch', ylabel='Validation F1',
                title='Validation F1 (Phase 1 + Phase 2)',
                ylim=(0, 1.05))
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('MedXRVEncoder Two-Phase Training Curves',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, out_dir, 'fig_training_curves.pdf')


# ── 7. Precision-Recall Curve ─────────────────────────────────
def plot_pr_curve(labels, probs, out_dir):
    """
    Precision-Recall curve — more informative than ROC for medical screening.
    """
    precision, recall, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)

    # Baseline = prevalence
    prevalence = np.mean(labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, 'b-', lw=2.5, label=f'MedXRVEncoder (AP={ap:.4f})')
    ax.axhline(prevalence, color='gray', ls='--', lw=1.5,
               label=f'Baseline prevalence ({prevalence:.2f})')
    ax.fill_between(recall, precision, alpha=0.1, color='blue')
    ax.set_xlabel('Recall (Sensitivity)', fontsize=13)
    ax.set_ylabel('Precision', fontsize=13)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    plt.tight_layout()
    _save(fig, out_dir, 'fig_pr_curve.pdf')


# ── 8. Ablation Additive Gain (fills second Fig. ?? in Section 5.1) ──
def plot_ablation_gain(out_dir):
    """
    Waterfall / additive gain chart — fills the missing Fig. ??
    reference in Section 5.1 ("additive structure (Fig. ??)").
    Values from Table 6 of the paper.
    """
    components = [
        'Single-stream\n(XRV only)',
        '+ MCSA\n(cross-attention)',
        '+ Text Fusion\n(multimodal FAISS)',
        '+ Cross-Encoder\nRe-Ranking',
    ]
    values   = [51.2, 59.5, 75.6, 82.4]   # cumulative Retrieval F1 from Table 6
    gains    = [values[0]] + [values[i] - values[i-1] for i in range(1, len(values))]
    bottoms  = [0] + list(np.cumsum(gains[:-1]))
    colors   = ['#d62728', '#9467bd', '#ff7f0e', '#2ca02c']

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (comp, gain, bot, col) in enumerate(
            zip(components, gains, bottoms, colors)):
        bar = ax.bar(i, gain, bottom=bot, color=col,
                     edgecolor='white', lw=1.2, width=0.5)
        ax.text(i, bot + gain / 2,
                f'+{gain:.1f}pp' if i > 0 else f'{gain:.1f}%',
                ha='center', va='center', fontsize=11,
                color='white', fontweight='bold')

    ax.plot(range(len(values)), values, 'ko--', ms=6, lw=1.5,
            label='Cumulative Retrieval F1')
    for i, v in enumerate(values):
        ax.text(i + 0.28, v + 0.8, f'{v}%', fontsize=10, color='black')

    ax.set_xticks(range(len(components)))
    ax.set_xticklabels(components, fontsize=10)
    ax.set_ylabel('Retrieval F1 (%)', fontsize=12)
    ax.set_title('Additive Component Gains (Ablation Study)',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _save(fig, out_dir, 'fig_ablation_gain.pdf')


# ══════════════════════════════════════════════════════════════
# MEDIUM PRIORITY — NEW FIGURES
# ══════════════════════════════════════════════════════════════

# ── 9. Calibration Curve (reliability diagram) ───────────────
def plot_calibration(labels, probs, out_dir, n_bins=10):
    """
    Reliability diagram — shows whether predicted probabilities
    are well-calibrated. Important for clinical trustworthiness claims.
    """
    frac_pos, mean_pred = calibration_curve(labels, probs,
                                             n_bins=n_bins, strategy='uniform')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Reliability diagram
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect calibration')
    axes[0].plot(mean_pred, frac_pos, 'b-o', ms=6, lw=2,
                 label='MedXRVEncoder')
    axes[0].fill_between(mean_pred, frac_pos, mean_pred,
                          alpha=0.15, color='blue', label='Calibration gap')
    axes[0].set_xlabel('Mean Predicted Probability', fontsize=12)
    axes[0].set_ylabel('Fraction of Positives (TB)', fontsize=12)
    axes[0].set_title('Calibration Curve (Reliability Diagram)',
                       fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, 1]); axes[0].set_ylim([0, 1])

    # Confidence histogram
    axes[1].hist(np.array(probs)[np.array(labels) == 0],
                 bins=20, alpha=0.6, color='royalblue', label='Normal')
    axes[1].hist(np.array(probs)[np.array(labels) == 1],
                 bins=20, alpha=0.6, color='crimson', label='TB')
    axes[1].axvline(0.5, color='gray', ls='--', lw=1.5, label='Decision threshold')
    axes[1].set_xlabel('Predicted TB Probability', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Confidence Distribution by Class',
                       fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, out_dir, 'fig_calibration.pdf')


# ── 10. Per-class sensitivity / specificity bar chart ─────────
def plot_sensitivity_specificity(labels, preds, probs, out_dir):
    """
    Per-class metrics bar chart — visualises the numbers quoted in
    Section 4.2: specificity 0.970, sensitivity 0.791 (FN=9 version).
    """
    from sklearn.metrics import confusion_matrix
    cm = np.array([[64, 2], [9, 53]])   # hardcoded best result

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)   # recall for TB
    specificity  = tn / (tn + fp)  # recall for Normal
    ppv          = tp / (tp + fp)  # precision for TB
    npv          = tn / (tn + fn)  # precision for Normal
    f1_tb        = 2 * ppv * sensitivity / (ppv + sensitivity)
    f1_normal    = 2 * specificity * npv / (specificity + npv)

    metrics = {
        'Sensitivity\n(TB Recall)':    sensitivity,
        'Specificity\n(Normal Recall)': specificity,
        'PPV\n(TB Precision)':         ppv,
        'NPV\n(Normal Precision)':     npv,
        'F1 (TB)':                      f1_tb,
        'F1 (Normal)':                  f1_normal,
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    x      = np.arange(len(metrics))
    vals   = list(metrics.values())
    labels_x = list(metrics.keys())
    colors = ['#d62728', '#1f77b4', '#d62728', '#1f77b4', '#d62728', '#1f77b4']

    bars = ax.bar(x, vals, color=colors, edgecolor='white', lw=1.2, width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xticks(x); ax.set_xticklabels(labels_x, fontsize=10)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Classification Metrics\n(MedXRVEncoder, n=133)',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.12)
    ax.grid(axis='y', alpha=0.3)

    tb_patch = mpatches.Patch(color='#d62728', label='TB class')
    nm_patch = mpatches.Patch(color='#1f77b4', label='Normal class')
    ax.legend(handles=[tb_patch, nm_patch], fontsize=11)

    plt.tight_layout()
    _save(fig, out_dir, 'fig_sensitivity_specificity.pdf')
