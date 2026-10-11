"""API tests: /predict on a tiny dummy sample video (trained model required)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("DEEPTRACE_TEST", "1")

from deeptrace.data.dummy import create_dummy_dfdc_sample  # noqa: E402

fastapi_testclient = pytest.importorskip("fastapi.testclient", reason="fastapi not installed")


@pytest.fixture(scope="module")
def trained_results(tmp_path_factory):
    """Train tiny models on the dummy sample so /predict has checkpoints."""
    root = tmp_path_factory.mktemp("apidata")
    data_root = create_dummy_dfdc_sample(root, n_labeled=6)
    import deeptrace.config as cfg
    from deeptrace import train

    original = ("results",)
    out_dir = tmp_path_factory.mktemp("apimodels")
    train.main(
        data_root=data_root,
        subset_size=6,
        epochs=1,
        out_dir=out_dir,
    )
    # point the inference module to the temp checkpoints
    import deeptrace.inference as inf

    inf.MODEL_DIR = out_dir
    inf._MODEL_CACHE.clear()
    yield out_dir
    inf._MODEL_CACHE.clear()


def test_predict_on_dummy_video(trained_results, tmp_path_factory):
    from fastapi.testclient import TestClient
    from deeptrace.api.main import app

    root = tmp_path_factory.mktemp("apidata2")
    create_dummy_dfdc_sample(root, n_labeled=2)
    video = root / "train_sample_videos" / "dmy000.mp4"

    client = TestClient(app)
    with open(video, "rb") as f:
        r = client.post(
            "/predict",
            files={"file": ("dmy000.mp4", f, "video/mp4")},
            data={"model": "temporal"},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["verdict"] in ("REAL", "FAKE")
    assert 0.0 <= body["video_probability"] <= 1.0
    assert len(body["per_frame_scores"]) > 0
    assert body["disclaimer"].startswith("Research prototype")


def test_predict_rejects_bad_file(trained_results):
    from fastapi.testclient import TestClient
    from deeptrace.api.main import app

    client = TestClient(app)
    # empty upload
    r = client.post(
        "/predict",
        files={"file": ("empty.mp4", b"", "video/mp4")},
    )
    assert r.status_code == 400
    # not a valid video
    r = client.post(
        "/predict",
        files={"file": ("bad.mp4", b"this is not a video", "video/mp4")},
    )
    assert r.status_code == 400


def test_predict_rejects_wrong_type(trained_results):
    from fastapi.testclient import TestClient
    from deeptrace.api.main import app

    client = TestClient(app)
    r = client.post(
        "/predict",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 415
