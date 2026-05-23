#!/usr/bin/env python3
# scripts/build_index.py
"""
Stage C: Build multimodal FAISS index from trained MedXRVEncoder.

Usage:
    python scripts/build_index.py --config configs/default.yaml
"""
import os, sys, argparse, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.models import MedXRVEncoder
from src.retrieval import build_faiss_index


def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = MedXRVEncoder().to(device)
    ckpt  = os.path.join(cfg['paths']['output_dir'], 'model_best.pth')
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print("✅ Model loaded from", ckpt)

    build_faiss_index(model, cfg, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
