"""Shared inference helpers for the FastAPI service and Streamlit app.

Both load the trained temporal model (default) or baseline, extract frames,
and return a JSON-friendly prediction dict with a verdict (probability based),
per-frame scores, and graceful handling of common failure modes.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from deeptrace.config import FRAMES_PER_VIDEO, INPUT_SIZE, MEAN, STD, DEVICE
from deeptrace.data.frames import _get_mtcnn, detect_and_crop_face

MODEL_DIR = Path("results")

_MODEL_CACHE: dict[str, object] = {}


def _load_model(kind: str = "temporal", device: str = DEVICE):
    """Load (and cache) the trained model checkpoint from results/."""
    key = f"{kind}:{device}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    from deeptrace.models import FrameAveragingBaseline, TemporalModel

    is_temporal = kind == "temporal"
    fname = MODEL_DIR / ("temporal.pt" if is_temporal else "baseline.pt")

    class _NotTrained(RuntimeError):
        pass

    if not fname.exists():
        raise _NotTrained(
            f"Checkpoint missing at {fname}. Run `python -m deeptrace.train` first."
        )
    state = torch.load(fname, map_location=device)
    if is_temporal:
        model = TemporalModel(frozen_backbone=True)
    else:
        model = FrameAveragingBaseline()
    model.load_state_dict(state)
    model = model.to(device).eval()
    _MODEL_CACHE[key] = model
    return model


def frames_from_video(path: Path, frames_per_video: int = FRAMES_PER_VIDEO) -> list[np.ndarray]:
    """Extract evenly spaced RGB frames; raise on corrupt/empty video."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        raise ValueError(f"Video has no readable frames: {path}")
    idxs = set(np.linspace(0, total - 1, min(frames_per_video, total)).round().astype(int).tolist())
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
    if not picked:
        raise ValueError(f"Video frames not decodable: {path}")
    return picked


def _to_tensor_batch(frames_rgb: list[np.ndarray], mtcnn) -> torch.Tensor:
    """Face-crop each frame and stack into a (1, T, C, H, W) tensor."""
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
    from PIL import Image

    tfm = Compose(
        [
            Resize(INPUT_SIZE + 16, interpolation=Image.BICUBIC),
            CenterCrop(INPUT_SIZE),
            ToTensor(),
            Normalize(mean=MEAN, std=STD),
        ]
    )
    tensors: list[torch.Tensor] = []
    used_face = 0
    for f in frames_rgb:
        pil = Image.fromarray(f)
        face = detect_and_crop_face(f, mtcnn=mtcnn)
        # detect_and_crop_face returns a full-frame fallback when no face is
        # found; count how many frames actually contained a face for the API.
        if face is not None and _mtcnn_hit(pil, mtcnn):
            used_face += 1
        tensors.append(tfm(face if face is not None else pil))
    return torch.stack(tensors).unsqueeze(0), used_face


def _mtcnn_hit(pil_img, mtcnn) -> bool:
    try:
        boxes, _ = mtcnn.detect(pil_img)
        return bool(boxes is not None and len(boxes) > 0)
    except Exception:
        return False


def predict_video(
    video_path: Path,
    kind: str = "temporal",
    device: str = DEVICE,
    frames_per_video: int = FRAMES_PER_VIDEO,
) -> dict:
    """Full inference: returns verdict, video prob, per-frame scores + meta."""
    model = _load_model(kind, device)
    frames = frames_from_video(video_path, frames_per_video=frames_per_video)
    try:
        mtcnn = _get_mtcnn(device)
    except Exception:
        mtcnn = None
    x, faces_found = _to_tensor_batch(frames, mtcnn)
    x = x.to(device)

    with torch.no_grad():
        if kind == "temporal":
            emb = model.frame_embeddings(x)           # (1, T, D')
            per_window = model.head(emb).squeeze(-1).squeeze(0)  # (T,)
            video_logit = model(x)
        else:
            fl = model.frame_logits(x).squeeze(0)
            per_window = fl
            video_logit = x.new_tensor([fl.mean()])

    probs = torch.sigmoid(per_window).cpu().tolist()
    video_prob = float(torch.sigmoid(video_logit).item())
    return {
        "verdict": "FAKE" if video_prob >= 0.5 else "REAL",
        "video_probability": round(video_prob, 4),
        "per_frame_scores": [round(p, 4) for p in probs],
        "model": kind,
        "frames_extracted": len(frames),
        "faces_detected": faces_found,
        "disclaimer": "Research prototype; not a forensic tool.",
    }
