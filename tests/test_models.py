# tests/test_models.py
"""
Basic unit tests for MedXRVEncoder and U-Net.
Run with: pytest tests/
"""
import pytest
import torch
import numpy as np


def test_cross_attention_fusion():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.models import CrossAttentionFusion
    fusion = CrossAttentionFusion(dim_a=1024, dim_b=1280, heads=8, out_dim=768)
    a = torch.randn(2, 1024)
    b = torch.randn(2, 1280)
    out = fusion(a, b)
    assert out.shape == (2, 768), f"Expected (2, 768), got {out.shape}"


def test_to_xrv_range():
    from src.models import to_xrv
    x = torch.rand(2, 3, 224, 224)   # ImageNet-normalized range
    out = to_xrv(x)
    assert out.shape == (2, 1, 224, 224)
    # Output should be roughly in [-1024, 1024]
    assert out.min().item() > -1200
    assert out.max().item() < 1200


def test_dice_bce_loss():
    from src.models import dice_bce_loss
    pred   = torch.randn(4, 1, 224, 224)
    target = (torch.rand(4, 1, 224, 224) > 0.5).float()
    loss   = dice_bce_loss(pred, target)
    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_entity_f1():
    from src.retrieval import entity_f1
    retrieved = ["opacity in left upper lobe with consolidation"]
    reference = "bilateral opacity consolidation tuberculosis"
    f1 = entity_f1(retrieved, reference)
    assert 0.0 <= f1 <= 1.0


def test_entity_f1_empty_reference():
    from src.retrieval import entity_f1
    f1 = entity_f1(["some text"], "")
    assert f1 == 0.0
