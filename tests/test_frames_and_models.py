"""Tests for frame extraction shapes and model forward-pass shapes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from deeptrace.config import FRAMES_PER_VIDEO, INPUT_SIZE
from deeptrace.data.dataset import _resample_frames
from deeptrace.data.dummy import create_dummy_dfdc_sample
from deeptrace.inference import frames_from_video


@pytest.fixture(scope="module")
def dummy_video(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("data")
    create_dummy_dfdc_sample(root, n_labeled=6)
    return root / "train_sample_videos" / "dmy000.mp4"


def test_extract_frames_count_and_shape(dummy_video):
    frames = frames_from_video(dummy_video, frames_per_video=FRAMES_PER_VIDEO)
    assert len(frames) == FRAMES_PER_VIDEO
    for f in frames:
        assert f.ndim == 3 and f.shape[2] == 3  # RGB
        assert f.dtype == np.uint8


def test_resample_frames_evenly():
    paths = [Path(f"f{i}.jpg") for i in range(30)]
    out = _resample_frames(paths, 16)
    assert len(out) == 16
    # monotonic, distinct (no repeats when n >= k)
    idxs = [paths.index(p) for p in out]
    assert idxs == sorted(idxs) and len(set(idxs)) == 16


def test_resample_frames_pads_when_few():
    paths = [Path(f"f{i}.jpg") for i in range(5)]
    out = _resample_frames(paths, 16)
    assert len(out) == 16
    assert out[-1] == out[-2]  # padded by repeating last frame


def test_resample_frames_empty_raises():
    with pytest.raises(RuntimeError):
        _resample_frames([], 16)


def test_baseline_forward_shape():
    from deeptrace.models import FrameAveragingBaseline

    model = FrameAveragingBaseline(pretrained=False)
    model.eval()
    x = torch.randn(2, 5, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2,)


def test_temporal_forward_shape():
    from deeptrace.models import TemporalModel

    model = TemporalModel(pretrained=False, frozen_backbone=True)
    model.eval()
    x = torch.randn(2, 5, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2,)
    # frozen backbone must actually be frozen
    assert not any(p.requires_grad for p in model.features.parameters())


def test_temporal_transformer_forward_shape():
    from deeptrace.models import TemporalModel

    model = TemporalModel(pretrained=False, frozen_backbone=True, sequence_model="transformer")
    model.eval()
    x = torch.randn(1, 4, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1,)
