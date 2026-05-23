#!/usr/bin/env python3
# scripts/train_encoder.py
"""
Stage B: Train MedXRVEncoder (two-phase domain-aware training).
Saves model_best.pth and training_history.npy.

Usage:
    python scripts/train_encoder.py --config configs/default.yaml
"""
import os, sys, argparse, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from src.train import train_encoder


def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model, val_dl, labels, preds, probs, history = train_encoder(cfg, device)

    # Save training history for plot_training_curves()
    out_dir = cfg['paths']['output_dir']
    np.save(os.path.join(out_dir, 'training_history.npy'), history)
    print(f"✅ training_history.npy saved to {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
