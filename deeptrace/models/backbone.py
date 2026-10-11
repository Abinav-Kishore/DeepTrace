"""Backbone: a pretrained CNN used in two ways.

1. BASELINE: score each frame independently, then average per-frame
   probabilities to get a video-level score (frame-averaging baseline).
2. TEMPORAL: use the same CNN (frozen by default) as a per-frame feature
   extractor; the sequence of embeddings feeds a BiGRU / transformer.

The pretrained weights are ImageNet weights (documented choice: no large
face-forensics-pretrained backbone is available offline, and ImageNet
features transfer well enough for a small fine-tuned head).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

from deeptrace.config import BACKBONE, BACKBONE_PRETRAINED, FEATURE_DIM


def _build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    """Return (backbone_feature_extractor, feature_dim) for the configured CNN."""
    weights = None
    if pretrained:
        if name == "resnet18":
            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        elif name == "resnet50":
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        elif name == "efficientnet_b0":
            weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
        else:
            raise ValueError(f"Unknown backbone: {name}")

    if name == "resnet18":
        model = torchvision.models.resnet18(weights=weights)
        feat_dim = model.fc.in_features
        model.fc = nn.Identity()  # replace classifier with identity => (N, feat_dim)
    elif name == "resnet50":
        model = torchvision.models.resnet50(weights=weights)
        feat_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif name == "efficientnet_b0":
        model = torchvision.models.efficientnet_b0(weights=weights)
        feat_dim = model.classifier[1].in_features
        model.classifier = nn.Identity()  # (N, feat_dim)
    else:
        raise ValueError(f"Unknown backbone: {name}")

    return model, feat_dim


def build_feature_extractor(
    backbone: str = BACKBONE,
    pretrained: bool = BACKBONE_PRETRAINED,
    expected_dim: int = FEATURE_DIM,
) -> nn.Module:
    """Build + validate the shared per-frame feature extractor."""
    model, feat_dim = _build_backbone(backbone, pretrained)
    assert feat_dim == expected_dim, (
        f"FEATURE_DIM config ({expected_dim}) != built backbone dim ({feat_dim}); "
        "update deeptrace/config.py after changing BACKBONE."
    )
    return model


def freeze_backbone(model: nn.Module) -> nn.Module:
    """Freeze every parameter (used for the default frozen-backbone temporal run)."""
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model
