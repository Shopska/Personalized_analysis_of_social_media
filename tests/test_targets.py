"""Tests for creator_analytics.targets — no real data, no network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creator_analytics.targets import add_targets, build_target, get_nonmonotonic_video_ids


@pytest.fixture
def eng():
    """Two creators, two videos each, records at days 7 and 14 only."""
    return pd.DataFrame(
        [
            # author a
            {"video_id": "v1", "days_since_post": 7, "play_count": 100},
            {"video_id": "v1", "days_since_post": 14, "play_count": 200},
            {"video_id": "v2", "days_since_post": 7, "play_count": 300},
            {"video_id": "v2", "days_since_post": 14, "play_count": 400},
            # author b
            {"video_id": "v3", "days_since_post": 7, "play_count": 1000},
            {"video_id": "v3", "days_since_post": 14, "play_count": 1500},
            {"video_id": "v4", "days_since_post": 7, "play_count": 800},
            {"video_id": "v4", "days_since_post": 14, "play_count": 1200},
        ]
    )


@pytest.fixture
def vids():
    return pd.DataFrame(
        [
            {"video_id": "v1", "author_id": "a"},
            {"video_id": "v2", "author_id": "a"},
            {"video_id": "v3", "author_id": "b"},
            {"video_id": "v4", "author_id": "b"},
        ]
    )


def test_absolute_applies_log1p(eng, vids):
    result = build_target(eng, vids, day=14, mode="absolute")
    row = result[result["video_id"] == "v1"].iloc[0]
    assert np.isclose(row["target"], np.log1p(200))


def test_absolute_day_filter_returns_only_exact_day(eng, vids):
    result = build_target(eng, vids, day=7, mode="absolute")
    assert set(result["video_id"]) == {"v1", "v2", "v3", "v4"}
    assert len(result) == 4


def test_missing_day_returns_empty_dataframe(eng, vids):
    result = build_target(eng, vids, day=30, mode="absolute")
    assert len(result) == 0
    assert "video_id" in result.columns and "target" in result.columns


def test_invalid_mode_raises(eng, vids):
    with pytest.raises(ValueError, match="mode"):
        build_target(eng, vids, day=14, mode="log_ratio")


def test_relative_requires_train_ids(eng, vids):
    with pytest.raises(ValueError, match="train_video_ids"):
        build_target(eng, vids, day=14, mode="relative", train_video_ids=None)


def test_relative_uses_train_only_not_test(eng, vids):
    """Creator median must use ONLY train videos, not test videos.

    Setup: train = {v1 (author a), v3 (author b)}, test = {v2, v4}.
    Author a train median = log1p(200) (v1 only).
    If test-set leakage occurred, median would use v2 too → different value.
    Verify: v1 relative target = log1p(200) - log1p(200) = 0.0 exactly.
    Verify: v2 relative target = log1p(400) - log1p(200)  (NOT using v2 in median).
    """
    train_ids = ["v1", "v3"]
    result = build_target(eng, vids, day=14, mode="relative", train_video_ids=train_ids)

    a_train_median = np.log1p(200)  # only v1 is in train for author a
    b_train_median = np.log1p(1500)  # only v3 is in train for author b

    v1 = result[result["video_id"] == "v1"].iloc[0]
    assert np.isclose(v1["target"], 0.0, atol=1e-10), (
        "Train video v1 should have relative target 0.0 when it's the only train video for its creator"
    )

    v2 = result[result["video_id"] == "v2"].iloc[0]
    expected_v2 = np.log1p(400) - a_train_median
    assert np.isclose(v2["target"], expected_v2, atol=1e-10)

    v4 = result[result["video_id"] == "v4"].iloc[0]
    expected_v4 = np.log1p(1200) - b_train_median
    assert np.isclose(v4["target"], expected_v4, atol=1e-10)


def test_relative_train_residuals_sum_to_zero_per_creator(eng, vids):
    """When all videos are in train, per-creator median of relative target = 0."""
    train_ids = ["v1", "v2", "v3", "v4"]
    result = build_target(eng, vids, day=14, mode="relative", train_video_ids=train_ids)
    result = result.merge(vids, on="video_id")
    for _, group in result.groupby("author_id"):
        assert abs(group["target"].median()) < 1e-10


def test_build_target_respects_videos_filter():
    """build_target must only return rows for video_ids in the videos argument.
    Previously it returned all engagement rows regardless of the videos scope.
    """
    engagement = pd.DataFrame(
        {
            "video_id": ["v1", "v2", "v3"],
            "days_since_post": [14, 14, 14],
            "play_count": [100, 200, 300],
        }
    )
    # Pass only v1 and v2 — v3 must NOT appear in the output
    videos_subset = pd.DataFrame({"video_id": ["v1", "v2"], "author_id": ["a", "a"]})
    result = build_target(engagement, videos_subset, day=14, mode="absolute")
    assert set(result["video_id"]) == {"v1", "v2"}
    assert len(result) == 2


def test_get_nonmonotonic_flags_decreasing_video():
    engagement = pd.DataFrame(
        {
            "video_id": ["v1", "v1", "v1", "v2", "v2"],
            "days_since_post": [7, 14, 21, 7, 14],
            "play_count": [100, 200, 150, 100, 200],  # v1 drops 200→150
        }
    )
    bad = get_nonmonotonic_video_ids(engagement)
    assert "v1" in bad
    assert "v2" not in bad


def test_get_nonmonotonic_monotonic_video_is_clean():
    engagement = pd.DataFrame(
        {
            "video_id": ["v1", "v1"],
            "days_since_post": [7, 14],
            "play_count": [100, 200],
        }
    )
    assert len(get_nonmonotonic_video_ids(engagement)) == 0


def test_add_targets_baseline_plus_relative_equals_absolute():
    engagement = pd.DataFrame(
        {
            "video_id": ["v1", "v2", "v3", "v4"],
            "days_since_post": [14, 14, 14, 14],
            "play_count": [100, 300, 1000, 50],
        }
    )
    df = pd.DataFrame(
        {
            "video_id": ["v1", "v2", "v3", "v4"],
            "author_id": ["a", "a", "a", "b"],
            "split": ["train", "train", "test", "train"],
        }
    )
    out = add_targets(df, engagement, day=14)
    assert np.allclose(out["creator_baseline"] + out["target_rel"], out["target_abs"])
    assert np.isclose(out.loc[0, "creator_baseline"], np.median(np.log1p([100, 300])))
