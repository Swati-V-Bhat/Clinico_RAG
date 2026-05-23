# src/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchxrayvision as xrv
import segmentation_models_pytorch as smp

EMBED_DIM = 768


def to_xrv(x):
    """3-ch ImageNet-norm → 1-ch XRV [-1024, 1024] (Eq. 2 in paper)."""
    mean   = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std    = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    coeffs = torch.tensor([0.2989, 0.5870, 0.1140], device=x.device).view(1, 3, 1, 1)
    gray   = ((x * std + mean) * coeffs).sum(1, keepdim=True)
    return gray * 2048 - 1024


class CrossAttentionFusion(nn.Module):
    """
    Mutual Cross-Stream Attention (MCSA) — Equations 3-7 in paper.

    Medical stream queries texture stream (A→B):
        Which texture patterns match the detected pathology?
    Texture stream queries medical stream (B→A):
        Which pathological region does this texture belong to?
    """

    def __init__(self, dim_a, dim_b, heads=8, out_dim=EMBED_DIM):
        super().__init__()
        d = 512
        self.proj_a   = nn.Linear(dim_a, d)
        self.proj_b   = nn.Linear(dim_b, d)
        self.attn_a2b = nn.MultiheadAttention(d, heads, dropout=0.1, batch_first=True)
        self.attn_b2a = nn.MultiheadAttention(d, heads, dropout=0.1, batch_first=True)
        self.norm_a   = nn.LayerNorm(d)
        self.norm_b   = nn.LayerNorm(d)
        self.fusion   = nn.Sequential(
            nn.Linear(d * 2, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Dropout(0.1),
        )

    def forward(self, feat_a, feat_b):
        a   = self.proj_a(feat_a).unsqueeze(1)   # (B, 1, d)
        b   = self.proj_b(feat_b).unsqueeze(1)   # (B, 1, d)
        a2b, _ = self.attn_a2b(a, b, b)
        b2a, _ = self.attn_b2a(b, a, a)
        a_out  = self.norm_a(a + a2b).squeeze(1)
        b_out  = self.norm_b(b + b2a).squeeze(1)
        return self.fusion(torch.cat([a_out, b_out], dim=1))


class MedXRVEncoder(nn.Module):
    """
    Dual-stream encoder (Stage B).

    Stream 1 — Medical:  XRV DenseNet-121 (700k CXRs)  → 1024-d
    Stream 2 — Texture:  EfficientNet-B0 (ImageNet)     → 1280-d
    Fusion:               MCSA                           → 768-d (L2-norm)
    """

    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        _xrv              = xrv.models.DenseNet(weights="densenet121-res224-all")
        self.med_backbone = _xrv.features
        self.med_pool     = nn.AdaptiveAvgPool2d(1)
        self.med_dim      = 1024

        self.eff     = timm.create_model('efficientnet_b0', pretrained=True,
                                          num_classes=0, global_pool='avg')
        self.eff_dim = 1280

        self.fusion     = CrossAttentionFusion(self.med_dim, self.eff_dim,
                                                heads=8, out_dim=embed_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(embed_dim, 2),
        )

    def _med_features(self, x):
        return self.med_pool(self.med_backbone(to_xrv(x))).flatten(1)

    def get_embedding(self, x):
        """L2-normalised 768-d embedding for retrieval."""
        return F.normalize(self.fusion(self._med_features(x), self.eff(x)), dim=1)

    def forward(self, x):
        return self.classifier(self.fusion(self._med_features(x), self.eff(x)))

    def get_cam_layer(self):
        """Target layer for Grad-CAM++: deepest conv in DenseNet."""
        return self.med_backbone.denseblock4.denselayer16.conv2

    def freeze_backbones(self):
        for p in self.med_backbone.parameters(): p.requires_grad = False
        for p in self.eff.parameters():          p.requires_grad = False

    def unfreeze_backbones(self):
        for p in self.med_backbone.parameters(): p.requires_grad = True
        for p in self.eff.parameters():          p.requires_grad = True


def build_unet(device='cpu'):
    """ResNet-34 U-Net for lung segmentation (Stage A)."""
    return smp.Unet(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
        activation=None,
    ).to(device)


def dice_bce_loss(pred, target):
    """Composite segmentation loss: L_seg = L_BCE + L_Dice (Eq. 1 in paper)."""
    bce  = nn.BCEWithLogitsLoss()(pred, target)
    p    = torch.sigmoid(pred).view(-1)
    t    = target.view(-1)
    dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)
    return bce + dice
