# src/retrieval.py
import os
import re
import numpy as np
import faiss
import torch
from torch.utils.data import DataLoader

from .dataset import CXRDataset, collate_fn


def build_faiss_index(model, cfg, device):
    """
    Build multimodal FAISS index (Stage C).
    Each entry = alpha_img * visual_emb + alpha_txt * miniLM_text_emb.
    Falls back to pure image embedding when no clinical text is available.
    """
    paths  = cfg['paths']
    rcfg   = cfg['retrieval']
    alpha_img = rcfg['alpha_img']
    alpha_txt = rcfg['alpha_txt']

    print("Building FAISS index from full Shenzhen dataset (662 images)...")

    # ── Visual embeddings ─────────────────────────────────────
    ds = CXRDataset(paths['shenzhen_img'],
                     paths.get('shenzhen_txt'), train=False)
    dl = DataLoader(ds, batch_size=32, shuffle=False,
                     collate_fn=collate_fn, num_workers=2)
    model.eval()
    vis_embs, texts, fnames, labels = [], [], [], []
    with torch.no_grad():
        for imgs, lbls, txts, fns in dl:
            with torch.cuda.amp.autocast():
                e = model.get_embedding(imgs.to(device))
            vis_embs.extend(e.cpu().numpy())
            texts.extend(txts); fnames.extend(fns); labels.extend(lbls.numpy())

    V = np.array(vis_embs, dtype=np.float32)
    faiss.normalize_L2(V)

    # ── Text embeddings via MiniLM ────────────────────────────
    try:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(rcfg['miniLM_model'])
        raw_txt_embs = st.encode(texts, batch_size=64,
                                  show_progress_bar=True,
                                  normalize_embeddings=True)
        # Pad/repeat 384-d → 768-d to match visual embedding dim
        T = np.repeat(raw_txt_embs, 2, axis=1).astype(np.float32)
        faiss.normalize_L2(T)
        # Weighted fusion
        M = alpha_img * V + alpha_txt * T
    except Exception as e:
        print(f"  ⚠️  MiniLM unavailable ({e}) — falling back to image-only index")
        M = V.copy()

    faiss.normalize_L2(M)
    idx = faiss.IndexFlatIP(M.shape[1])
    idx.add(M)
    L = np.array(labels)

    os.makedirs(paths['output_dir'], exist_ok=True)
    np.save(os.path.join(paths['output_dir'], 'embs.npy'),   V)   # visual only
    np.save(os.path.join(paths['output_dir'], 'joint_embs.npy'), M)
    np.save(os.path.join(paths['output_dir'], 'labels.npy'), L)
    np.save(os.path.join(paths['output_dir'], 'texts.npy'),
            np.array(texts, dtype=object), allow_pickle=True)
    np.save(os.path.join(paths['output_dir'], 'fnames.npy'),
            np.array(fnames, dtype=object), allow_pickle=True)
    faiss.write_index(idx, os.path.join(paths['output_dir'], 'faiss.index'))

    print(f"✅ {len(M)} vectors  dim={M.shape[1]}  "
          f"TB={int(L.sum())}  Normal={int((L==0).sum())}")
    return idx, texts, fnames, L


def load_faiss_index(cfg):
    paths = cfg['paths']
    idx   = faiss.read_index(os.path.join(paths['output_dir'], 'faiss.index'))
    texts  = np.load(os.path.join(paths['output_dir'], 'texts.npy'),
                      allow_pickle=True).tolist()
    fnames = np.load(os.path.join(paths['output_dir'], 'fnames.npy'),
                      allow_pickle=True).tolist()
    labels = np.load(os.path.join(paths['output_dir'], 'labels.npy'))
    return idx, texts, fnames, labels


def retrieve_topk(query_emb, faiss_idx, rag_labels, rag_texts, rag_fnames,
                  k=3):
    """Bi-encoder retrieval — returns top-k candidates."""
    q = query_emb.reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(q)
    _, idxs = faiss_idx.search(q, min(k, len(rag_texts)))
    return [{'label': "TB" if rag_labels[j] == 1 else "Normal",
              'text':  str(rag_texts[j]),
              'fname': rag_fnames[j]} for j in idxs[0]]


def retrieve_with_rerank(query_emb, query_text, faiss_idx, rag_labels,
                          rag_texts, rag_fnames, cfg):
    """Two-stage retrieval: FAISS bi-encoder → MiniLM cross-encoder re-rank."""
    rcfg  = cfg['retrieval']
    k_bi  = rcfg['k_bi']
    k_fin = rcfg['k_final']

    q = query_emb.reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(q)
    _, idxs = faiss_idx.search(q, min(k_bi, len(rag_texts)))

    try:
        from sentence_transformers import CrossEncoder
        ce     = CrossEncoder(rcfg['cross_encoder'], max_length=512)
        pairs  = [(query_text[:200], str(rag_texts[j])[:300]) for j in idxs[0]]
        scores = ce.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(idxs[0], scores),
                        key=lambda x: x[1], reverse=True)[:k_fin]
        return [{'label': "TB" if rag_labels[j] == 1 else "Normal",
                  'text':  str(rag_texts[j]),
                  'fname': rag_fnames[j],
                  'score': float(s)} for j, s in ranked]
    except ImportError:
        return retrieve_topk(query_emb, faiss_idx, rag_labels,
                              rag_texts, rag_fnames, k=k_fin)


ENTITIES = [
    'opacity', 'infiltrate', 'consolidation', 'effusion', 'cavitation',
    'nodule', 'lymphadenopathy', 'atelectasis', 'fibrosis', 'pleural',
    'bilateral', 'tuberculosis', 'tb', 'miliary', 'apical', 'hilar', 'normal',
]


def entity_set(txt):
    return set(e for e in ENTITIES
               if re.search(r'\b' + e + r'\b', txt.lower()))


def entity_f1(retrieved_texts, reference_text):
    """Entity-level F1 between retrieved cases and reference clinical text."""
    combined = " ".join(retrieved_texts)
    h     = entity_set(combined)
    r_set = entity_set(reference_text)
    if not r_set:
        return 0.0
    tp = len(h & r_set)
    p  = tp / len(h) if h else 0
    rc = tp / len(r_set)
    return 2 * p * rc / (p + rc) if (p + rc) > 0 else 0.0
