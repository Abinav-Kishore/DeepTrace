"""Tests for video-level split (no leakage) using the deterministic dummy sample."""

from __future__ import annotations

import random

import pytest

from deeptrace.data import build_video_splits, leak_check_splits


def _make_videos(n=20):
    return {f"vid{i:03d}": i for i in range(n)}


def test_no_video_in_two_splits():
    videos = _make_videos(20)
    splits = build_video_splits(videos)
    leak_check_splits(splits)  # must not raise

    sets = [set(v) for v in splits.values()]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            assert not (sets[i] & sets[j]), "video id appears in two splits"


def test_all_videos_accounted_for():
    videos = _make_videos(30)
    splits = build_video_splits(videos)
    all_ids = set(videos)
    assert set().union(*[set(s) for s in splits.values()]) == all_ids


def test_leak_check_raises():
    bad = {"train": ["a", "b"], "val": ["b", "c"], "test": ["d"]}
    with pytest.raises(AssertionError):
        leak_check_splits(bad)


def test_deterministic_with_seed():
    videos = _make_videos(15)
    s1 = build_video_splits(videos, seed=42)
    s2 = build_video_splits(videos, seed=42)
    assert s1 == s2
    # different seed can give a different split
    s3 = build_video_splits(videos, seed=7)
    assert s1 != s3
