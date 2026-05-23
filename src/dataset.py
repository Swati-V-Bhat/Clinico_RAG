# src/dataset.py
import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

TRAIN_TFM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.3, 0.3, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TFM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class SegDS(Dataset):
    """Montgomery — image + binary lung mask for U-Net training."""

    def __init__(self, img_dir, mask_root):
        self.img_dir   = img_dir
        self.mask_root = mask_root
        self.files     = [f for f in os.listdir(img_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f   = self.files[idx]
        img = cv2.resize(cv2.imread(os.path.join(self.img_dir, f)), (224, 224))
        mask = np.zeros((224, 224), dtype=np.float32)
        for s in ('leftMask', 'rightMask'):
            p = os.path.join(self.mask_root, s, f)
            if os.path.exists(p):
                mask = np.maximum(mask, cv2.resize(cv2.imread(p, 0), (224, 224)))
        mask = (mask > 0).astype(np.float32)
        return (torch.from_numpy(img).float().permute(2, 0, 1) / 255.,
                torch.from_numpy(mask).unsqueeze(0))


class CXRDataset(Dataset):
    """Shenzhen — image, binary label, clinical text, filename."""

    def __init__(self, img_root, txt_root=None, train=True):
        self.img_root = img_root
        self.txt_root = txt_root
        self.files    = sorted([f for f in os.listdir(img_root) if f.endswith('.png')])
        self.tfm      = TRAIN_TFM if train else VAL_TFM

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        fname = self.files[i]
        label = 1 if '_1.png' in fname else 0
        text  = ""
        if self.txt_root:
            tp = os.path.join(self.txt_root, fname.replace('.png', '.txt'))
            if os.path.exists(tp):
                try:
                    text = open(tp, 'r', errors='ignore').read().strip()
                except Exception:
                    pass
        img = Image.open(os.path.join(self.img_root, fname)).convert('RGB')
        return self.tfm(img), label, text, fname


class MontgomeryDataset(Dataset):
    """Montgomery — image + binary label only (no text).
    Used for cross-site generalisation during MedXRVEncoder training."""

    def __init__(self, img_root, train=True):
        self.img_root = img_root
        self.files    = sorted([f for f in os.listdir(img_root) if f.endswith('.png')])
        self.tfm      = TRAIN_TFM if train else VAL_TFM

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        fname = self.files[i]
        label = 1 if '_1.png' in fname else 0
        img   = Image.open(os.path.join(self.img_root, fname)).convert('RGB')
        return self.tfm(img), label, "", fname


def collate_fn(batch):
    imgs, lbls, texts, fnames = zip(*batch)
    return torch.stack(imgs), torch.tensor(lbls), list(texts), list(fnames)
