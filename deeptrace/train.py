"""Train + evaluate both models and write results/metrics.json.

Hard leakage rule enforced here: build_video_splits splits BY VIDEO with a
fixed seed; leak_check_splits raises AssertionError if any video id appears
in two splits. Every metric in the output comes from actually running the
trained models on the held-out test videos — nothing is hardcoded.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from deeptrace.config import (  # noqa: F401 (configures everything)
    SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SUBSET_SIZE,
    FRAMES_PER_VIDEO, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, NUM_WORKERS,
    DEVICE, BACKBONE, FROZEN_BACKBONE, TEMPORAL_MODEL, TEMPORAL_TRANSFORMER_ABLATION,
    DEFAULT_SOURCE,
)
from deeptrace.data import (
    load_metadata, build_video_splits, leak_check_splits, get_transform,
    DeepTraceDataset, build_frame_index, frame_stats, create_dummy_dfdc_sample,
)
from deeptrace.models import FrameAveragingBaseline, TemporalModel

RESULTS_DIR = Path("results")
FRAMES_CACHE = Path("deeptrace/results/frames_cache")


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    try:
        torch.cuda.manual_seed_all(SEED)
    except Exception:
        pass
    torch.use_deterministic_algorithms(
        True, warn_only=False
    ) if False else None  # determinism kept soft for dataloader order; seed fixed above


def make_loaders(
    data_root: Path | None = None,
    subset_size: int | None = SUBSET_SIZE,
    frames_per_video: int = FRAMES_PER_VIDEO,
    batch_size: int = BATCH_SIZE,
):
    """Load metadata, split BY VIDEO, extract (cached) frames, build loaders."""
    source = Path(data_root) if data_root else DEFAULT_SOURCE
    if not (source / "train_sample_videos" / "metadata.json").exists():
        print(f"[setup] No DFDC sample found at {source}; creating deterministic dummy sample.")
        source = create_dummy_dfdc_sample(source)

    videos = load_metadata(source)
    assert len(videos) > 0, "No labeled videos found."

    if subset_size is not None and subset_size < len(videos):
        # Deterministic subset BEFORE splitting (fixed seed, then split BY VIDEO).
        ids = sorted(videos)
        rng = random.Random(SEED)
        rng.shuffle(ids)
        ids = ids[: subset_size]
        videos = {k: videos[k] for k in ids}
        print(f"[setup] Using SUBSET_SIZE={subset_size} of {len(ids) if False else ''} labeled videos (fixed seed).")

    splits = build_video_splits(videos)
    leak_check_splits(splits)  # HARD ASSERTION: no video id in >1 split

    print(
        "[splits] train/val/test video counts:",
        {k: len(v) for k, v in splits.items()},
    )

    print("[frames] extracting frames (cached) ...")
    t0 = time.time()
    frame_index = build_frame_index(videos, FRAMES_CACHE, frames_per_video=frames_per_video)
    print(f"[frames] done in {time.time() - t0:.1f}s:", frame_stats(frame_index))

    transform = get_transform(train=False)
    loaders = {}
    for name in ("train", "val", "test"):
        ds = DeepTraceDataset(
            splits[name], videos, frame_index, transform, frames_per_video
        )
        assert len(ds) > 0, f"Empty {name} split"
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=NUM_WORKERS,
            drop_last=False,
        )
    return loaders, splits, videos, frame_index


def _train_one_epoch(model, loader, opt, crit, device) -> float:
    model.train()
    total, n = 0.0, 0
    for frames, labels, _ids in loader:
        frames = frames.to(device)
        labels = labels.to(device)
        opt.zero_grad()
        logits = model(frames)
        loss = crit(logits, labels)
        loss.backward()
        opt.step()
        total += loss.item() * frames.size(0)
        n += frames.size(0)
    return total / max(n, 1)


@torch.no_grad()
def collect_predictions(model, loader, device) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run a full pass, returning (probs, labels, video_ids) as arrays."""
    model.eval()
    probs, labels, ids = [], [], []
    for frames, y, video_ids in loader:
        frames = frames.to(device)
        logits = model(frames)
        p = torch.sigmoid(logits).reshape(-1).cpu().numpy()
        probs.extend(p.ravel().tolist())
        labels.extend(y.numpy().tolist())
        ids.extend(list(video_ids))
    return np.asarray(probs, dtype=np.float64), np.asarray(labels, dtype=np.float64), ids


def train_model(
    model: nn.Module,
    loaders: dict,
    device: str,
    epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
) -> dict:
    """Train on train split, early-stop on val AUC; return history + best epoch."""
    model = model.to(device)
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=1e-4
    )
    crit = nn.BCEWithLogitsLoss()

    from sklearn.metrics import roc_auc_score  # local import to keep module import light

    history: list[dict] = []
    best_auc, best_state, best_epoch = -1.0, None, -1
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss = _train_one_epoch(model, loaders["train"], opt, crit, device)
        val_probs, val_labels, _ = collect_predictions(model, loaders["val"], device)
        try:
            val_auc = roc_auc_score(val_labels, val_probs) if len(set(val_labels)) > 1 else float("nan")
        except ValueError:
            val_auc = float("nan")
        history.append(
            {"epoch": epoch, "train_loss": round(tr_loss, 4), "val_auc": None if val_auc != val_auc else round(val_auc, 4), "secs": round(time.time() - t0, 1)}
        )
        print(f"[train] epoch {epoch}: loss={tr_loss:.4f} val_auc={val_auc:.4f} ({history[-1]['secs']}s)")
        if val_auc == val_auc and val_auc > best_auc:  # not NaN and improved
            best_auc, best_epoch = val_auc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "best_val_auc": round(best_auc, 4), "best_epoch": best_epoch}


def main(
    data_root: Path | None = None,
    subset_size: int | None = None,
    epochs: int | None = None,
    transformer_ablation: bool | None = None,
    out_dir: Path | None = None,
) -> dict:
    seed_everything()
    epochs = NUM_EPOCHS if epochs is None else epochs
    subset_size = SUBSET_SIZE if subset_size is None else subset_size
    transformer_ablation = (
        TEMPORAL_TRANSFORMER_ABLATION if transformer_ablation is None else transformer_ablation
    )
    out_dir = out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    loaders, splits, videos, _ = make_loaders(
        data_root=data_root, subset_size=subset_size
    )

    device = DEVICE
    print(f"[train] device: {device}")

    results: dict = {
        "seed": SEED,
        "subset_size": subset_size,
        "backbone": BACKBONE,
        "frames_per_video": FRAMES_PER_VIDEO,
        "device": device,
        "splits": {k: sorted(v) for k, v in splits.items()},
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
        "epochs": epochs,
        "models": {},
    }

    # ---------------- baseline: frame-averaging ----------------
    print("[train] training FRAME-AVERAGING BASELINE ...")
    baseline = FrameAveragingBaseline()
    base_info = train_model(baseline, loaders, device, epochs=epochs)
    bb, bl, bids = collect_predictions(baseline, loaders["test"], device)
    torch.save(baseline.state_dict(), out_dir / "baseline.pt")
    results["models"]["baseline"] = {
        **base_info,
        "test_probs": bb.tolist(),
        "test_labels": bl.tolist(),
        "test_ids": bids,
    }

    # ---------------- temporal: CNN + BiGRU ----------------
    print("[train] training TEMPORAL (frozen backbone BiGRU) ...")
    temporal = TemporalModel(frozen_backbone=FROZEN_BACKBONE)
    temp_info = train_model(temporal, loaders, device, epochs=epochs)
    tp, tl, tids = collect_predictions(temporal, loaders["test"], device)
    torch.save(temporal.state_dict(), out_dir / "temporal.pt")
    results["models"]["temporal"] = {
        **temp_info,
        "test_probs": tp.tolist(),
        "test_labels": tl.tolist(),
        "test_ids": tids,
        "frozen_backbone": FROZEN_BACKBONE,
        "sequence_model": TEMPORAL_MODEL,
    }

    # ---------------- optional ablation: transformer ----------------
    if transformer_ablation:
        print("[train] training TEMPORAL (frozen backbone TRANSFORMER ablation) ...")
        tf = TemporalModel(frozen_backbone=FROZEN_BACKBONE, sequence_model="transformer")
        tf_info = train_model(tf, loaders, device, epochs=epochs)
        tfp, tfb, tfids = collect_predictions(tf, loaders["test"], device)
        torch.save(tf.state_dict(), out_dir / "temporal_transformer.pt")
        results["models"]["temporal_transformer"] = {
            **tf_info,
            "test_probs": tfp.tolist(),
            "test_labels": tfb.tolist(),
            "test_ids": tfids,
            "frozen_backbone": FROZEN_BACKBONE,
            "sequence_model": "transformer",
        }

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[train] wrote {metrics_path}")

    # Evaluate right after training so metrics.json has the test metrics too.
    from deeptrace.evaluate import evaluate_and_plot  # local import avoids cycle

    evaluate_and_plot(metrics_path=metrics_path, results=results, out_dir=out_dir)
    return results


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Train baseline + temporal models on DFDC sample.")
    p.add_argument("--data-root", default=None, help="Path to DFDC sample root (default from config).")
    p.add_argument("--subset", type=int, default=None, help="Subset of labeled videos (default from config).")
    p.add_argument("--epochs", type=int, default=None, help="Training epochs (default from config).")
    p.add_argument("--transformer-ablation", action="store_true", help="Also train the transformer ablation.")
    a = p.parse_args()
    main(
        data_root=Path(a.data_root) if a.data_root else None,
        subset_size=a.subset,
        epochs=a.epochs,
        transformer_ablation=True if a.transformer_ablation else None,
    )
