#!/usr/bin/env python
"""Download the real DFDC sample dataset from a public HuggingFace mirror.

Source: https://huggingface.co/datasets/191fa07121/deepfake-detection-challenge
Contains 800 mp4 files (400 labeled train_sample_videos + 400 unlabeled
test_videos) plus metadata.json with the official DFDC labels
(label/split/original). It is the same data as the Kaggle
'deepfake-detection-challenge' competition sample.

Usage:
    python scripts/download_dfdc_sample.py               # everything (~1.5 GB)
    python scripts/download_dfdc_sample.py --labeled-only # 400 labeled videos only
    python scripts/download_dfdc_sample.py --labeled-only --first 120  # subset for weak machines

If you prefer provenance straight from Kaggle, run instead:
    kaggle competitions download -c deepfake-detection-challenge -f sample.zip
and unzip sample.zip into data/dfdc_sample/ (same layout).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/191fa07121/deepfake-detection-challenge/resolve/main/"
DEST = Path("data/dfdc_sample")


def _download(fname: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(BASE + fname, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(dest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labeled-only", action="store_true", help="skip the unlabeled test_videos")
    ap.add_argument("--first", type=int, default=None, help="only the first N labeled videos (deterministic order)")
    args = ap.parse_args()

    req = urllib.request.Request(
        "https://huggingface.co/api/datasets/191fa07121/deepfake-detection-challenge",
        headers={"User-Agent": "deeptrace-download"},
    )
    d = json.load(urllib.request.urlopen(req, timeout=60))
    files = [s["rfilename"] for s in d.get("siblings", [])]

    labeled = sorted(f for f in files if f.endswith(".mp4") and not f.startswith("processed_"))
    print(f"Found {len(labeled)} labeled-candidate mp4s on the mirror.")

    if args.first:
        labeled = sorted(labeled)[: args.first]
        print(f"Restricted to first {args.first}: {labeled[:3]} ...")

    (DEST / "train_sample_videos").mkdir(parents=True, exist_ok=True)
    (DEST / "test_videos").mkdir(parents=True, exist_ok=True)

    # 1. metadata (always full — labels for all 400)
    print("Downloading metadata.json ...")
    _download("metadata.json", DEST / "train_sample_videos" / "metadata.json")

    # 2. move metadata copy to repo root of the source for reference (optional)
    ok, fail = 0, 0
    for i, f in enumerate(labeled):
        try:
            _download(f, DEST / "train_sample_videos" / f)
            ok += 1
            if i % 25 == 0:
                print(f"  {i + 1}/{len(labeled)}")
        except Exception as e:
            fail += 1
            print(f"  FAIL {f}: {e}")

    print(f"Done. downloaded={ok} failed={fail}")

    # Validate: keep metadata entries whose mp4 exists and drop the rest
    meta_p = DEST / "train_sample_videos" / "metadata.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    keep = {k: v for k, v in meta.items() if (DEST / "train_sample_videos" / k).exists()}
    meta_p.write_text(json.dumps(keep, indent=2) + "\n", encoding="utf-8")
    print(f"metadata.json: {len(meta)} entries -> {len(keep)} with a local mp4")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
