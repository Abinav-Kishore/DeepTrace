"""TEMPORAL model: per-frame CNN features -> sequence model -> video logit.

Default configuration (documented in README):
- backbone: ImageNet-pretrained ResNet18, FROZEN during training (flag below)
- sequence model: 1-layer bidirectional GRU, hidden 128, dropout 0.3
- head: linear over the GRU's final hidden state (concat of both directions)
- ablation: small Transformer encoder over frame embeddings
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from deeptrace.config import (
    BACKBONE,
    BACKBONE_PRETRAINED,
    FEATURE_DIM,
    TEMPORAL_HIDDEN,
    GRU_LAYERS,
    DROPOUT,
    TEMPORAL_MODEL,
    TF_HEADS,
    TF_LAYERS,
    TF_DIM_FEEDFORWARD,
    TF_DROPOUT,
)
from deeptrace.models.backbone import build_feature_extractor


class TemporalModel(nn.Module):
    """frozen-or-trainable CNN encoder + BiGRU (or transformer ablation)."""

    def __init__(
        self,
        backbone: str = BACKBONE,
        pretrained: bool = BACKBONE_PRETRAINED,
        feature_dim: int = FEATURE_DIM,
        hidden: int = TEMPORAL_HIDDEN,
        gru_layers: int = GRU_LAYERS,
        dropout: float = DROPOUT,
        frozen_backbone: bool = True,
        sequence_model: str = TEMPORAL_MODEL,  # 'bgru' | 'transformer'
    ) -> None:
        super().__init__()
        self.sequence_model_kind = sequence_model
        self.features = build_feature_extractor(backbone, pretrained, feature_dim)
        if frozen_backbone:
            for p in self.features.parameters():
                p.requires_grad_(False)
            self.features.eval()
        self.frozen_backbone = frozen_backbone

        in_dim = feature_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        if sequence_model == "bgru":
            self.temporal = nn.GRU(
                input_size=hidden,
                hidden_size=hidden,
                num_layers=gru_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if gru_layers > 1 else 0.0,
            )
            out_dim = hidden * 2
        elif sequence_model == "transformer":
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=TF_HEADS,
                dim_feedforward=TF_DIM_FEEDFORWARD,
                dropout=TF_DROPOUT,
                batch_first=True,
            )
            self.temporal = nn.TransformerEncoder(enc_layer, num_layers=TF_LAYERS)
            out_dim = hidden
        else:
            raise ValueError(f"Unknown sequence model: {sequence_model}")

        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(out_dim, 1))

    def train(self, mode: bool = True) -> "TemporalModel":
        super().train(mode)
        if self.frozen_backbone:
            self.features.eval()  # keep backbone in eval mode even during training
        return self

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (B, T, C, H, W) -> video-level logit (B,)."""
        b, t, c, h, w = frames.shape
        flat = frames.reshape(b * t, c, h, w)
        feats = self.features(flat)  # (B*T, D)
        seq = self.proj(feats).view(b, t, -1)  # (B, T, hidden)

        if self.sequence_model_kind == "bgru":
            out, _ = self.temporal(seq)  # (B, T, 2*hidden)
            pooled = out[:, -1, :]  # last time step of both directions
        else:
            out = self.temporal(seq)  # (B, T, hidden)
            pooled = out.mean(dim=1)  # mean-pool over time

        return self.head(pooled).squeeze(-1)  # (B,)

    def frame_embeddings(self, frames: torch.Tensor) -> torch.Tensor:
        """Per-frame post-GRU embeddings before the head: (B, T, out_dim)."""
        b, t, c, h, w = frames.shape
        flat = frames.reshape(b * t, c, h, w)
        feats = self.features(flat)
        seq = self.proj(feats).view(b, t, -1)
        if self.sequence_model_kind == "bgru":
            out, _ = self.temporal(seq)
        else:
            out = self.temporal(seq)
        return out
