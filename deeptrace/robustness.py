"""Robustness study: degrade held-out TEST videos only, re-evaluate both models.

Degradations (applied to the test split ONLY — never to train/val):
- H.264 re-encode at low bitrates (via PyAV/libx264 — no external ffmpeg needed)
- downscaling (to ~480p and ~240p heights)
- Gaussian blur (sigma=3)
- Gaussian noise (sigma=20)

Pipeline: ffmpeg-python is NOT required; we implement all degradations in
memory with PyAV + OpenCV and re-encode to H.264 with libx264 (PyAV bundles
FFmpeg libs). Raw (uncropped, undegraded) per-video frames are cached once,
then every degradation re-extracts faces from the degraded frames so the face
crop reflects the degraded video, as it would in deployment.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import av
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from deeptrace.config import (
    ROBUSTNESS_LEVELS, ROBUSTNESS_CACHE, FRAMES_PER_VIDEO, SEED, DEFAULT_SOURCE,
)
from deeptrace.data import (
    load_metadata, build_video_splits, leak_check_splits, get_transform,
    DeepTraceDataset,
)
from deeptrace.data.frames import _get_mtcnn, detect_and_crop_face
from deeptrace.train import make_loaders, collect_predictions


# --------------------------------------------------------------------------
# Frame re-encoding helpers (PyAV based, no external ffmpeg binary)
# --------------------------------------------------------------------------


def _encode_rgb_frames(frames: list[np.ndarray], bitrate: int, out_path: Path) -> Path:
    """Encode a list of RGB frames to an H.264 MP4 at the given bitrate."""
    assert frames, "no frames to encode"
    h, w = frames[0].shape[:2]
    container = av.open(str(out_path), mode="w")
    stream = container.add_stream("libx264", rate=15)
    stream.pix_fmt = "yuv420p"
    stream.width = w
    stream.height = h
    stream.bit_rate = bitrate
    stream.options = {"preset": "medium"}

    img_last = None
    for packet in stream.encode(_wrap(frames[0])):
        container.mux(packet)
    for f in frames[1:]:
        img_last = f
        for packet in stream.encode(_wrap(f)):
            container.mux(packet)
    if img_last is not None:
        for packet in stream.encode(_wrap(img_last)):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return out_path


def _wrap(frame: np.ndarray) -> "av.VideoFrame":
    return av.VideoFrame.from_ndarray(frame, format="rgb24")


def _degraded_frames(
    raw_frames: list[np.ndarray],
    degrade: str,
    target_height: int | None = None,
) -> list[np.ndarray]:
    """Apply one degradation to a list of RGB uint8 frames."""
    out: list[np.ndarray] = []
    if degrade in ("h264", "h264_low"):
        # Compression is applied at re-encode time; frames pass through.
        return raw_frames
    for f in raw_frames:
        if degrade in ("down480", "down240"):
            h0, w0 = f.shape[:2]
            scale = target_height / h0
            f2 = cv2.resize(f, (max(2, int(round(w0 * scale))), target_height), interpolation=cv2.INTER_AREA)
            downscale_cache = None
        elif degrade == "blur":
            f2 = cv2.GaussianBlur(f, (0, 0), sigmaX=3.0, sigmaY=3.0)
        elif degrade == "noise":
            rng = np.random.default_rng(SEED)
            noise = rng.normal(0, 20, size=f.shape)
            f2 = np.clip(f.astype(np.float64) + noise, 0, 255).astype(np.uint8)
        else:
            f2 = f
        out.append(f2)
    return out


_CODEC_BITRATE = {"h264": 500_000, "h264_low": 150_000}


def degrade_raw_frames(
    video_path: Path,
    degrade: str,
    frames_per_video: int,
    out_dir: Path,
) -> list[Path]:
    """Cache degraded video for a level, return the frame jpg paths inside.

    All frames (evenly sampled at the SAME timestamps as training extraction)
    pass through read -> degrade -> re-encode to H.264 (when applicable) ->
    decode -> face crop. This exercises the real re-encode/decode pipeline.
    """
    level_dir = out_dir / degrade
    frame_dir = level_dir / "frames"
    if frame_dir.exists():
        cached = sorted(frame_dir.glob("*.jpg"))
        if cached:  # an empty dir from a crashed earlier run is NOT a cache hit
            return cached

    level_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f"Cannot open {video_path}"
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idxs = (
        set(np.linspace(0, total - 1, frames_per_video).round().astype(int).tolist())
        if total > 0
        else set(range(frames_per_video))
    )
    picked: list[np.ndarray] = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in idxs:
            picked.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    assert picked, f"No frames readable from {video_path}"

    # Re-encode at low bitrate (this step itself applies codec compression),
    # and for downscale levels resize frames BEFORE the encode.
    codec_degrade = _CODEC_BITRATE.get(degrade)
    if codec_degrade:
        rt_frames = picked
    elif degrade in ("down480", "down240"):
        target_h = 480 if degrade == "down480" else 240
        rt_frames = [
            cv2.resize(
                f,
                (
                    max(2, int(round(f.shape[1] * target_h / f.shape[0]))),
                    target_h,
                ),
                interpolation=cv2.INTER_AREA,
            )
            for f in picked
        ]
    elif degrade == "blur":
        rt_frames = [cv2.GaussianBlur(f, (0, 0), sigmaX=3.0, sigmaY=3.0) for f in picked]
    elif degrade == "noise":
        rng = np.random.default_rng(SEED)
        rt_frames = [
            np.clip(f.astype(np.float64) + rng.normal(0, 20, size=f.shape), 0, 255).astype(np.uint8)
            for f in picked
        ]
    else:
        rt_frames = picked

    # For compression levels actually run the decode step too: encode to a
    # temp mp4, then decode it back to frames (end-to-end codec round trip).
    if codec_degrade:
        tmp = level_dir / video_path.name
        _encode_rgb_frames(rt_frames, codec_degrade, tmp)
        cap = cv2.VideoCapture(str(tmp))
        decoded: list[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            decoded.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        tmp.unlink(missing_ok=True)
        rt_frames = decoded
        assert rt_frames, f"Codec round-trip produced no frames for {video_path}"

    mtcnn = _get_mtcnn()
    saved: list[Path] = []
    for j, f in enumerate(rt_frames):
        face = detect_and_crop_face(f, mtcnn=mtcnn)
        p = frame_dir / f"frame_{j:03d}.jpg"
        face.save(str(p))
        saved.append(p)
    return sorted(frame_dir.glob("*.jpg"))


def _clean_frame_paths(video_id: str, frames_cache: Path) -> list[Path]:
    return sorted((frames_cache / video_id).glob("*.jpg"))


def run_robustness(
    metrics_path: Path = Path("results/metrics.json"),
    out_dir: Path | None = None,
    models_dir: Path | None = None,
    data_root: Path | None = None,
    subset_size: int | None = None,
    frames_cache: Path | None = None,
) -> dict:
    """Evaluate BOTH models on every degradation level of the TEST split."""
    out_dir = out_dir or metrics_path.parent
    models_dir = models_dir or out_dir
    data_root = data_root or DEFAULT_SOURCE

    # Rebuild the exact same split (same seed) — no leakage by construction.
    source = Path(data_root)
    videos = load_metadata(source)
    from deeptrace.config import SUBSET_SIZE

    if subset_size is None:
        from deeptrace import train as _train
        subset_size = _train.SUBSET_SIZE
    if subset_size is not None and subset_size < len(videos):
        import random as _r
        ids = sorted(videos)
        rng = _r.Random(SEED)
        rng.shuffle(ids)
        ids = ids[: subset_size]
        videos = {k: videos[k] for k in ids}
    splits = build_video_splits(videos)
    leak_check_splits(splits)

    frames_cache = frames_cache or Path("deeptrace/results/frames_cache")

    # Load trained checkpoints.
    from deeptrace.models import FrameAveragingBaseline, TemporalModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline = FrameAveragingBaseline()
    baseline.load_state_dict(torch.load(models_dir / "baseline.pt", map_location=device))
    baseline = baseline.to(device).eval()
    temporal = TemporalModel(frozen_backbone=True)
    temporal.load_state_dict(torch.load(models_dir / "temporal.pt", map_location=device))
    temporal = temporal.to(device).eval()

    # Load the clean (already measured) metrics to double-check consistency.
    results = json.loads(metrics_path.read_text(encoding="utf-8"))
    test_ids = sorted(splits["test"])
    for name, model in (("baseline", baseline), ("temporal", temporal)):
        m = results["models"][name]
        if m.get("test_ids") and sorted(m["test_ids"]) != test_ids:
            print(
                f"[robustness] WARNING: metrics.json {name} test_ids differ from rebuilt "
                "split for this run; continuing with rebuilt split for degradation study."
            )

    level_results: dict[str, dict] = {}
    for level in ROBUSTNESS_LEVELS:
        lname, degrade = level["name"], level["degrade"]
        print(f"[robustness] level={lname} degrade={degrade}")
        # Build a per-video degraded frame index for TEST videos only.
        index: dict[str, list[Path]] = {}
        for vid in test_ids:
            if degrade is None:
                index[vid] = _clean_frame_paths(vid, frames_cache)
            else:
                index[vid] = degrade_raw_frames(
                    videos[vid].path, degrade, FRAMES_PER_VIDEO, ROBUSTNESS_CACHE
                )
            assert index[vid], f"No degraded frames for {vid} at {lname}"

        transform = get_transform(train=False)
        ds = DeepTraceDataset(test_ids, videos, index, transform, FRAMES_PER_VIDEO)
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

        labels = np.asarray([0.0 if videos[v].label == "REAL" else 1.0 for v in test_ids])
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

        level_results[lname] = {"desc": level["desc"]}
        for name, model in (("baseline", baseline), ("temporal", temporal)):
            probs, ytrue, _ = collect_predictions(model, loader, str(device))
            y = ytrue if len(ytrue) else labels
            preds = (probs >= 0.5).astype(int)
            level_results[lname][name] = {
                "accuracy": float(accuracy_score(y.astype(int), preds)),
                "auc": float(roc_auc_score(y.astype(int), probs)) if len(set(y.tolist())) > 1 else None,
                "f1": float(f1_score(y.astype(int), preds, zero_division=0)),
                "n": int(len(y)),
            }
            r = level_results[lname][name]
            print(
                f"    {name}: acc={r['accuracy']:.3f} auc={r['auc'] and round(r['auc'],3)} f1={r['f1']:.3f}"
            )

    results["robustness"] = level_results
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _robustness_plots(results, out_dir)
    return results


def _robustness_plots(results: dict, out_dir: Path) -> None:
    levels = list(results.get("robustness", {}).keys())
    if not levels:
        return
    x = np.arange(len(levels))

    for metric, fname, ylabel in (
        ("accuracy", "robustness_accuracy.png", "accuracy on degraded test videos"),
        ("auc", "robustness_auc.png", "AUC on degraded test videos"),
    ):
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        for name, marker in (("baseline", "o"), ("temporal", "s")):
            vals = []
            for lvl in levels:
                v = results["robustness"][lvl].get(name, {}).get(metric)
                vals.append(float("nan") if v is None else v)
            ax.plot(x, vals, marker=marker, label=name)
        ax.set_xticks(x, levels, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{metric} vs degradation (held-out test split only)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Run the low-quality video robustness study on the TEST split.")
    p.add_argument("--metrics", default="results/metrics.json")
    p.add_argument("--data-root", default=None)
    p.add_argument("--subset", type=int, default=None)
    args = p.parse_args()
    run_robustness(
        metrics_path=Path(args.metrics),
        data_root=Path(args.data_root) if args.data_root else None,
        subset_size=args.subset,
    )
