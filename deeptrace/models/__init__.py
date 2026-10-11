# Models: baseline (frame averaging) and temporal (CNN + BiGRU/transformer).
from deeptrace.models.backbone import build_feature_extractor, freeze_backbone
from deeptrace.models.baseline import FrameAveragingBaseline
from deeptrace.models.temporal import TemporalModel

__all__ = [
    "build_feature_extractor",
    "freeze_backbone",
    "FrameAveragingBaseline",
    "TemporalModel",
]
