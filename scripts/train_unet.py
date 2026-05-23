#!/usr/bin/env python3
# scripts/train_unet.py
"""
Stage A: Train ResNet-34 U-Net for lung segmentation.

Usage:
    python scripts/train_unet.py --config configs/default.yaml
"""
import os, sys, argparse, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.train import train_unet, eval_unet
from src.train import free_gpu


def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    unet = train_unet(cfg, device)
    if unet is not None:
        eval_unet(cfg, device, unet_model=unet)
        free_gpu(unet)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
