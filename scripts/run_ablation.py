#!/usr/bin/env python3
# scripts/run_ablation.py
"""
Ablation study — evaluates all pipeline configurations on the same fixed
sample set to ensure comparability.
"""
import os, sys, re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
import timm
import torchxrayvision as xrv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval import retrieve_topk, entity_f1

EMBED_DIM = 768


def run_ablation(model, val_dl, faiss_idx, rag_texts, rag_fnames,
                 rag_labels, cfg, device, n=25):
    print("\nAblation study (fixed eval set)...")

    # ── Collect fixed samples ─────────────────────────────────
    fixed_samples = []
    for imgs, lbls, texts, fnames in val_dl:
        for i in range(len(imgs)):
            if len(fixed_samples) >= n: break
            fixed_samples.append((imgs[i].clone(), lbls[i].item(),
                                   texts[i], fnames[i]))
        if len(fixed_samples) >= n: break
    print(f"  Fixed evaluation set: {len(fixed_samples)} samples")

    def score_config(encoder, use_rerank):
        encoder.eval(); rg = []
        for img_t, lbl, ref, fname in fixed_samples:
            inp = img_t.unsqueeze(0).to(device)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    emb = encoder.get_embedding(inp).cpu().numpy()

            if use_rerank:
                try:
                    from sentence_transformers import CrossEncoder
                    ce = CrossEncoder(cfg['retrieval']['cross_encoder'],
                                       max_length=512)
                    q  = emb.reshape(1, -1).astype(np.float32)
                    faiss.normalize_L2(q)
                    _, idxs = faiss_idx.search(
                        q, min(cfg['retrieval']['k_bi'], len(rag_texts)))
                    pairs  = [(ref[:200], str(rag_texts[j])[:300])
                               for j in idxs[0]]
                    scores = ce.predict(pairs, show_progress_bar=False)
                    ranked = sorted(zip(idxs[0], scores),
                                    key=lambda x: x[1], reverse=True
                                    )[:cfg['retrieval']['k_final']]
                    ret = [{'text': str(rag_texts[j])} for j, _ in ranked]
                except ImportError:
                    ret = retrieve_topk(emb, faiss_idx, rag_labels,
                                         rag_texts, rag_fnames)
            else:
                ret = retrieve_topk(emb, faiss_idx, rag_labels,
                                     rag_texts, rag_fnames)

            ef1 = entity_f1([r['text'] for r in ret], ref)
            rg.append(ef1)
        return np.mean(rg) * 100 if rg else 0.

    results = {}

    print("  [1/4] Full model (MCSA + FAISS + CrossEncoder)...")
    results['Full model'] = score_config(model, use_rerank=True)

    print("  [2/4] w/o Cross-Encoder re-ranking...")
    results['w/o Cross-Encoder'] = score_config(model, use_rerank=False)

    print("  [3/4] Single-stream (XRV DenseNet only)...")

    class XRVOnly(nn.Module):
        def __init__(self):
            super().__init__()
            _xrv      = xrv.models.DenseNet(weights="densenet121-res224-all")
            self.bb   = _xrv.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.proj = nn.Linear(1024, EMBED_DIM)
            self.classifier = nn.Sequential(
                nn.Dropout(0.2), nn.Linear(EMBED_DIM, 2))

        def _to_xrv(self, x):
            mean = torch.tensor([0.485, 0.456, 0.406],
                                  device=x.device).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225],
                                  device=x.device).view(1, 3, 1, 1)
            coef = torch.tensor([0.2989, 0.5870, 0.1140],
                                  device=x.device).view(1, 3, 1, 1)
            return ((x * std + mean) * coef).sum(1, keepdim=True) * 2048 - 1024

        def get_embedding(self, x):
            return F.normalize(
                self.proj(self.pool(self.bb(self._to_xrv(x))).flatten(1)), dim=1)

        def forward(self, x):
            return self.classifier(
                self.proj(self.pool(self.bb(self._to_xrv(x))).flatten(1)))

    xrv_only = XRVOnly().to(device)
    results['Single-stream (XRV only)'] = score_config(xrv_only, use_rerank=True)
    xrv_only.cpu()

    print("  [4/4] w/o Cross-Attention (concat fusion)...")

    class ConcatFusion(nn.Module):
        def __init__(self):
            super().__init__()
            _xrv      = xrv.models.DenseNet(weights="densenet121-res224-all")
            self.bb   = _xrv.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.eff  = timm.create_model('efficientnet_b0', pretrained=True,
                                           num_classes=0, global_pool='avg')
            self.proj = nn.Linear(1024 + 1280, EMBED_DIM)
            self.classifier = nn.Sequential(
                nn.Dropout(0.2), nn.Linear(EMBED_DIM, 2))

        def _to_xrv(self, x):
            mean = torch.tensor([0.485, 0.456, 0.406],
                                  device=x.device).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225],
                                  device=x.device).view(1, 3, 1, 1)
            coef = torch.tensor([0.2989, 0.5870, 0.1140],
                                  device=x.device).view(1, 3, 1, 1)
            return ((x * std + mean) * coef).sum(1, keepdim=True) * 2048 - 1024

        def get_embedding(self, x):
            m = self.pool(self.bb(self._to_xrv(x))).flatten(1)
            return F.normalize(
                self.proj(torch.cat([m, self.eff(x)], dim=1)), dim=1)

        def forward(self, x):
            m = self.pool(self.bb(self._to_xrv(x))).flatten(1)
            return self.classifier(
                self.proj(torch.cat([m, self.eff(x)], dim=1)))

    concat_m = ConcatFusion().to(device)
    results['w/o Cross-Attention (concat)'] = score_config(
        concat_m, use_rerank=True)
    concat_m.cpu()

    print(f"\n{'Configuration':<40}  {'Retrieval F1':>12}")
    print('-' * 56)
    for k, v in results.items():
        print(f"  {k:<38}  {v:>10.1f}%")

    return results
