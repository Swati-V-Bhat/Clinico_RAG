Clinico-RAG: A Framework for Multimodal Evidence-Based TB Diagnosis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

Official implementation of **"Clinico-RAG: A Framework for Multimodal Evidence-Based TB Diagnosis"**.

---

## Overview

Clinico-RAG is a unified five-stage diagnostic framework for tuberculosis detection and structured report generation from chest radiographs:

| Stage | Component | Description |
|-------|-----------|-------------|
| A | U-Net Segmentation | ResNet-34 U-Net, BCE+Dice loss, lung ROI isolation |
| B | MedXRVEncoder | Dual-stream MCSA encoder (DenseNet-121 + EfficientNet-B0) |
| C | Multimodal Retrieval | FAISS bi-encoder + MiniLM cross-encoder re-ranking |
| D | Report Generation | BioMistral-7B (4-bit NF4) + dynamic PubMed grounding |
| E | Explainability | Grad-CAM++ + Deletion AUC faithfulness evaluation |

---

## Results

| Metric | Value |
|--------|-------|
| Accuracy | 0.9412 |
| F1 Score | 0.9348 |
| AUC-ROC | 0.9682 |
| U-Net Dice | 0.9837 |
| Mean Deletion AUC | 0.2841 |
| Retrieval Precision@3 | 0.89 |
| BLEU | 0.3247 |

---

## Repository Structure


```text
clinico-rag/
├── configs/
│   └── default.yaml        # All hyperparameters
├── notebooks/
│   ├── cell1_install.py    # Kaggle environment dependencies setup
│   └── cell2_main.py       # Clinico-RAG full pipeline execution
├── scripts/
│   ├── build_index.py      # Stage C: build FAISS index
│   ├── evaluate.py         # Full evaluation + all figures
│   ├── run_ablation.py     # Ablation studies execution
│   ├── train_encoder.py    # Stage B: train MedXRVEncoder
│   └── train_unet.py       # Stage A: train segmentation model
├── src/
│   ├── __init__.py
│   ├── dataset.py          # CXRDataset, MontgomeryDataset, SegDS
│   ├── explainability.py   # Grad-CAM++, Deletion AUC
│   ├── models.py           # MedXRVEncoder, CrossAttentionFusion, U-Net
│   ├── retrieval.py        # FAISS index, bi-encoder, cross-encoder re-ranking
│   ├── train.py            # Two-phase training loop
│   └── visualize.py        # Visualization utilities
├── tests/
│   └── test_models.py      # Unit tests
├── README.md
├── requirements.txt
└── setup.py

```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt

```

### 2. Download datasets

* **Shenzhen Hospital CXR Set**: [Kaggle](https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-shenzhen)
* **Montgomery County CXR Set**: [Kaggle](https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-montgomery)

Set paths in `configs/default.yaml`.

---

## Usage

### Train U-Net (Stage A)

```bash
python scripts/train_unet.py --config configs/default.yaml

```

### Train MedXRVEncoder (Stage B)

```bash
python scripts/train_encoder.py --config configs/default.yaml

```

### Build FAISS Index (Stage C)

```bash
python scripts/build_index.py --config configs/default.yaml

```

### Full Evaluation + All Figures

```bash
python scripts/evaluate.py --config configs/default.yaml

```

### Kaggle Execution

Run the provided scripts sequentially in your Kaggle environment:

```bash
python notebooks/cell1_install.py
python notebooks/cell2_main.py

```

