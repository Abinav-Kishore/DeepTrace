"""Evaluate baseline vs temporal on the held-out TEST split.

Writes metrics into results/metrics.json and generates plots:
- ROC curves (clean test data)
- confusion matrices for both models
- baseline-vs-temporal comparison bar chart
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _video_level_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).astype(int)
    y = labels.astype(int)
    # Zero-division guard for tiny subsets: report 0.0 rather than crash.
    return {
        "auc": float(roc_auc_score(y, probs)) if len(set(y.tolist())) > 1 else None,
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "threshold": threshold,
        "n": int(len(y)),
        "confusion_matrix": confusion_matrix(y, preds, labels=[0, 1]).tolist(),
    }


def evaluate_and_plot(
    metrics_path: Path = Path("results/metrics.json"),
    results: dict | None = None,
    out_dir: Path | None = None,
) -> dict:
    """Compute clean-test metrics for every trained model, save + plot."""
    out_dir = out_dir or metrics_path.parent
    if results is None:
        results = json.loads(metrics_path.read_text(encoding="utf-8"))

    for name, m in results["models"].items():
        probs = np.asarray(m["test_probs"], dtype=np.float64)
        labels = np.asarray(m["test_labels"], dtype=np.float64)
        mm = _video_level_metrics(probs, labels)
        m["test_metrics"] = mm
        print(
            f"[eval] {name}: auc={mm['auc'] and round(mm['auc'], 4)} "
            f"acc={mm['accuracy']:.3f} p={mm['precision']:.3f} "
            f"r={mm['recall']:.3f} f1={mm['f1']:.3f} (n={mm['n']})"
        )

    # Honest delta; report even if temporal is NOT better.
    if "baseline" in results["models"] and "temporal" in results["models"]:
        b = results["models"]["baseline"]["test_metrics"]
        t = results["models"]["temporal"]["test_metrics"]
        results["baseline_vs_temporal"] = {
            "auc_delta": (None if b["auc"] is None or t["auc"] is None else round(t["auc"] - b["auc"], 4)),
            "accuracy_delta": round(t["accuracy"] - b["accuracy"], 4),
            "f1_delta": round(t["f1"] - b["f1"], 4),
            "note": "delta = temporal - baseline (positive = temporal better)",
        }

    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _plots(results, out_dir)
    print(f"[eval] wrote {metrics_path} and plots in {out_dir}")
    return results


def _plots(results: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- ROC curves ----
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in ("baseline", "temporal", "temporal_transformer"):
        m = results["models"].get(name)
        if not m:
            continue
        probs = np.asarray(m["test_probs"])
        labels = np.asarray(m["test_labels"]).astype(int)
        if len(set(labels.tolist())) <= 1:
            continue
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = results["models"][name]["test_metrics"]["auc"]
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC on held-out test videos (clean)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "roc_clean.png", dpi=150)
    plt.close(fig)

    # ---- confusion matrices ----
    names = [k for k in ("baseline", "temporal") if k in results["models"]]
    if names:
        fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.2))
        axes = np.atleast_1d(axes)
        for ax, name in zip(axes, names):
            cm = np.asarray(results["models"][name]["test_metrics"]["confusion_matrix"])
            ax.imshow(cm, cmap="Blues")
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, str(v), ha="center", va="center")
            ax.set_xticks([0, 1], labels=["REAL", "FAKE"])
            ax.set_yticks([0, 1], labels=["REAL", "FAKE"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"Confusion: {name}")
        fig.tight_layout()
        fig.savefig(out_dir / "confusion_matrices.png", dpi=150)
        plt.close(fig)

    # ---- comparison bar chart ----
    metrics_to_plot = ["accuracy", "precision", "recall", "f1"]
    avail = [k for k in ("baseline", "temporal") if k in results["models"]]
    if avail:
        x = np.arange(len(metrics_to_plot))
        width = 0.8 / len(avail)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for i, name in enumerate(avail):
            vals = [results["models"][name]["test_metrics"][k] for k in metrics_to_plot]
            ax.bar(x + i * width, vals, width, label=name)
        b_auc = results["models"].get("baseline", {}).get("test_metrics", {}).get("auc")
        t_auc = results["models"].get("temporal", {}).get("test_metrics", {}).get("auc")
        if b_auc is not None and t_auc is not None:
            ax.set_title(f"Test metrics (clean) — AUC: baseline {b_auc:.3f} vs temporal {t_auc:.3f}")
        else:
            ax.set_title("Test metrics (clean)")
        ax.set_xticks(x + width * (len(avail) - 1) / 2, metrics_to_plot)
        ax.set_ylim(0, 1.05)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "baseline_vs_temporal.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    evaluate_and_plot()
