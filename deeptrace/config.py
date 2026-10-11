"""
Single source of truth for DeepTrace configuration.

All hard-coded numbers live here so experiments are reproducible and
the README results can be regenerated exactly.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED: int = 42  # fixed random seed for every randomized step

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
# DFDC sample dataset layout (Kaggle "deepfake-detection-challenge" sample).
# - train_sample_videos/*.mp4  +  metadata.json
# - test_videos/*.mp4  (no labels in public sample; we hold out a portion of
#   the *labeled* train set as our test split, since the public test set is
#   unlabeled and can't be used for measured metrics).
#
# DEFAULT_SOURCE points at a folder you have locally. The real DFDC sample is
# obtained via Kaggle (see README / scripts/download_dfdc_sample.py). When the
# folder is not present, the pipeline can still run in "dummy data" mode for
# tests / CI / smoke runs.
DEFAULT_SOURCE: Path = Path("data/dfdc_sample")

# Video-level split ratios (by VIDEO, never by frame) on the LABELED set.
# val is used for backbone fine-tune / early-stop decisions; test is held out.
TRAIN_RATIO: float = 0.7
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9, "split ratios must sum to 1"

# Subset of labeled videos to actually use. Set to None to use the full set.
# Kept tiny by default for CPU / free-GPU runs; configurable for your hardware.
SUBSET_SIZE: int | None = 40  # configurable; None = use all labeled videos

# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------
FRAMES_PER_VIDEO: int = 16  # evenly spaced frames extracted per video
FACE_CROP_SIZE: int = 160    # MTCNN face crop size fed to the model
# If a video has no detectable face, we keep a centered, resized full frame
# placeholder so the sequence length stays fixed (documented in README limits).
USE_FACE_CROP: bool = True

# MTCNN runs on CPU here; keep these modest to stay fast.
MTCNN_IMAGE_SIZE: int = 160
MTCNN_MARGIN: float = 0.3

# Image normalization fed to the pretrained backbone (ImageNet stats).
MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# Model input size (square) after face crop / resize.
INPUT_SIZE: int = 112  # fits ResNet18 / EfficientNet-B0 well on CPU

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
BACKBONE: str = "resnet18"  # resnet18 | resnet50 | efficientnet_b0
# Backbone weights: ImageNet pretrained by default (documented choice).
BACKBONE_PRETRAINED: bool = True
FEATURE_DIM: int = (
    512 if BACKBONE == "resnet18"
    else (2048 if BACKBONE == "resnet50" else 1280)
)

# Temporal model (sequence of per-frame embeddings).
TEMPORAL_HIDDEN: int = 128
TEMPORAL_MODEL: str = "bgru"  # bgru | transformer
GRU_LAYERS: int = 1
DROPOUT: float = 0.3

# If True, backbone is frozen during temporal-model training (documented default).
FROZEN_BACKBONE: bool = True

# Ablation: if True, also train/eval a small Transformer encoder temporal model.
TEMPORAL_TRANSFORMER_ABLATION: bool = False
TF_HEADS: int = 4
TF_LAYERS: int = 2
TF_DIM_FEEDFORWARD: int = 128
TF_DROPOUT: float = 0.1

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE: int = 8  # sequence batch size (videos per batch)
LEARNING_RATE: float = 1e-3
NUM_EPOCHS: int = 6  # small default for CPU / free-GPU reproducibility
NUM_WORKERS: int = 0  # 0 on Windows/CPython-friendly dataloaders


# ---------------- convenience helpers ----------------


def torch_cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return torch.cuda.is_available()


def _resolve_device() -> str:
    if torch_cuda_available():
        return "cuda"
    # Prefer MPS on Apple Silicon when present.
    try:
        import torch
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def num_gpus() -> int:
    try:
        import torch
    except Exception:
        return 0
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


DEVICE: str = _resolve_device()

# ---------------------------------------------------------------------------
# Robustness study (degrades TEST videos only)
# ---------------------------------------------------------------------------
# Levels are applied to the held-out TEST split ONLY; never to train/val.
# Each level is a dict describing one degradation; order matters for plots.
ROBUSTNESS_LEVELS: list[dict] = [
    {"name": "clean", "desc": "Original test videos (no degradation)", "degrade": None},
    {"name": "h264_500kbps", "desc": "H.264 re-encode @ ~500 kbps", "degrade": "h264"},
    {"name": "h264_150kbps", "desc": "H.264 re-encode @ ~150 kbps", "degrade": "h264_low"},
    {"name": "down_480p", "desc": "Downscale to ~480p height", "degrade": "down480"},
    {"name": "down_240p", "desc": "Downscale to ~240p height", "degrade": "down240"},
    {"name": "blur_sigma3", "desc": "Gaussian blur sigma=3", "degrade": "blur"},
    {"name": "noise_sigma20", "desc": "Gaussian noise sigma=20", "degrade": "noise"},
]

# Where to write degraded test videos for the robustness study (never committed).
ROBUSTNESS_CACHE: Path = Path("deeptrace/results/robustness_cache")
