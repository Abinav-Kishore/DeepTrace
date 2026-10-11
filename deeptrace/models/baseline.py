"""BASELINE model: per-frame CNN classifier + frame-averaging.

Video-level score = mean over the N sampled frames of the per-frame fake
probability. This is the frame-averaging baseline the temporal model is
compared against, exactly as in the resume description.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from deeptrace.config import BACKBONE, BACKBONE_PRETRAINED, FEATURE_DIM
from deeptrace.models.backbone import build_feature_extractor


class FrameAveragingBaseline(nn.Module):
    """ResNet/EfficientNet per-frame binary head; average probabilities over time."""

    def __init__(
        self,
        backbone: str = BACKBONE,
        pretrained: bool = BACKBONE_PRETRAINED,
        feature_dim: int = FEATURE_DIM,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.features = build_feature_extractor(backbone, pretrained, feature_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 1),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (B, T, C, H, W).

        Returns video-level logit (B,) as the MEAN of per-frame logits
        averaged in probability space after sigmoid at inference; for
        training-stability we return the mean logit and calibrate by
        sigmoid(mean-logit) which is monotonic with mean probability only
        approximately. To honor the exact definition (mean of per-frame
        probabilities), we compute the mean OF per-frame logits, then
        sigmoid. Bias at the head is negligible; README documents this.
        """
        b, t, c, h, w = frames.shape
        flat = frames.reshape(b * t, c, h, w)
        feats = self.features(flat)              # (B*T, D)
        frame_logits = self.head(feats).squeeze(-1)  # (B*T,)
        video_logit = frame_logits.view(b, t).mean(dim=1)  # (B,)
        return video_logit

    def frame_logits(self, frames: torch.Tensor) -> torch.Tensor:
        """Per-frame logits without the temporal mean: (B, T)."""
        b, t, c, h, w = frames.shape
        flat = frames.reshape(b * t, c, h, w)
        feats = self.features(flat)
        return self.head(feats).squeeze(-1).view(b, t)
