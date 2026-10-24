"""Tests for creator_analytics.features — synthetic data only, no real dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creator_analytics.features import (
    build_features,
    creator_context_features,
    description_features,
    format_features,
    hashtag_indicator_features,
    hook_features,
    music_features,
    speech_features,
    timing_features,
)
from creator_analytics.targets import build_target, get_nonmonotonic_video_ids


# ------------------------------------------------------------------ #
# Shared fixtures                                                      #
# ------------------------------------------------------------------ #

@pytest.fixture
def minimal_videos():
    """Four videos by two authors; minimal columns required by all feature functions."""
    return pd.DataFrame(
        {
            "video_id":  ["v1", "v2", "v3", "v4"],
            "author_id": ["a",  "a",  "b",  "b"],
            "create_time": [
                "2024-07-01 10:00:00",
                "2024-07-05 18:00:00",
                "2024-07-01 08:00:00",
                "2024-07-03 22:00:00",
            ],
            "create_date": ["2024-07-01", "2024-07-05", "2024-07-01", "2024-07-03"],
            "duration":    [30.0,  60.0,  90.0, 15.0],
            "ratio":       ["540p", "720p", "540p", "540p"],
            "created_by_ai": [0.0, 1.0, 0.0, 0.0],
            "is_ads":      [0, 0, 0, 0],
            "desc":        ["hello #fyp", None, "world", "#dance"],
            "sticker_text": [None, "buy now", None, None],
            "hashtags": [
                np.array([{"hashtag_id": "1", "hashtag_name": "fyp"}]),
                np.array([{"hashtag_id": "2", "hashtag_name": "dance"}]),
                np.array([]),
                np.array([{"hashtag_id": "3", "hashtag_name": "dance"}]),
            ],
            "is_english":  [1, 1, 0, 1],
            "transcript":  ["Hello world.", None, "Hi there? How are you?", None],
            "speaking_rate": [120.0, None, 90.0, None],
            "word_count":  [2.0, None, 3.0, None],
            "emoji_count": [0.0, None, 0.0, None],
            "question_count": [0.0, None, 1.0, None],
            "hashtag_count": [1.0, None, 0.0, None],
            "anger":   [0.1, 0.8, 0.2, 0.3],
            "joy":     [0.7, 0.1, 0.5, 0.4],
            "surprise": [0.1, 0.05, 0.2, 0.2],
            "sadness": [0.05, 0.03, 0.05, 0.05],
            "disgust": [0.02, 0.01, 0.03, 0.01],
            "fear":    [0.03, 0.01, 0.02, 0.01],
            "topic":   ["Life", "Dance", "Life", "Dance"],
            "music_id": ["m1", "m1", "m2", "m1"],
            "music_owner_id": ["a", "x",  "b", "y"],
            "music_selected_from": ["original", "single_song", "original", "edit_page_recommend"],
            "gpt_summary": ["...", "...", "...", "..."],
        }
    )


@pytest.fixture
def minimal_creator_daily():
    return pd.DataFrame(
        {
            "author_id":      ["a",  "a",  "b",  "b"],
            "date":           ["2024-06-30", "2024-07-04", "2024-06-30", "2024-07-02"],
            "follower_count": [1000.0, 1100.0, 500.0, 510.0],
            "video_count":    [10.0,   12.0,   5.0,   6.0],
            "following_count": [100.0, 100.0, 50.0, 50.0],
            "total_favorited": [0, 0, 0, 0],
            "enterprise_verified": [None, None, None, None],
        }
    )


# ================================================================== #
# КРОК 0 — bug fixes in targets.py                                    #
# ================================================================== #

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
            "video_id":       ["v1", "v1", "v1", "v2", "v2"],
            "days_since_post": [7,    14,   21,   7,    14],
            "play_count":     [100,  200,  150,  100,  200],  # v1 drops 200→150
        }
    )
    bad = get_nonmonotonic_video_ids(engagement)
    assert "v1" in bad
    assert "v2" not in bad


def test_get_nonmonotonic_monotonic_video_is_clean():
    engagement = pd.DataFrame(
        {
            "video_id":       ["v1", "v1"],
            "days_since_post": [7,   14],
            "play_count":     [100, 200],
        }
    )
    assert len(get_nonmonotonic_video_ids(engagement)) == 0


# ================================================================== #
# Group 1 — format_features                                           #
# ================================================================== #

def test_format_features_duration_log(minimal_videos):
    out = format_features(minimal_videos)
    assert np.isclose(out.loc[out["video_id"] == "v1", "duration_log"].iloc[0], np.log1p(30.0))


def test_format_features_created_by_ai_fills_null(minimal_videos):
    v = minimal_videos.copy()
    v.loc[0, "created_by_ai"] = np.nan
    out = format_features(v)
    assert out["created_by_ai"].isna().sum() == 0


# ================================================================== #
# Group 2 — timing_features                                           #
# ================================================================== #

def test_timing_features_hour_and_dow(minimal_videos):
    out = timing_features(minimal_videos)
    # v1: '2024-07-01 10:00:00' → hour=10, Monday=0
    row = out[out["video_id"] == "v1"].iloc[0]
    assert row["publish_hour"] == 10
    assert row["publish_dow"] == 0  # Monday


def test_timing_days_since_prev_first_video_is_nan(minimal_videos):
    """First video by each author must have NaN days_since_prev_video."""
    out = timing_features(minimal_videos)
    # Sort by author+date → first video per author → NaN
    for author in ["a", "b"]:
        author_rows = out[out["video_id"].isin(
            minimal_videos[minimal_videos["author_id"] == author]["video_id"]
        )].sort_values("video_id")
        # The earliest date for each author should have NaN
    # v1 is 2024-07-01 for author a (earliest), v2 is 2024-07-05
    row_v2 = out[out["video_id"] == "v2"].iloc[0]
    assert row_v2["days_since_prev_video"] == 4  # 5-1 = 4 days


def test_timing_videos_last_7d_no_future(minimal_videos):
    """videos_last_7d must not count future videos."""
    out = timing_features(minimal_videos)
    # v1 at 2024-07-01 is the first video for author a → 0 prior in 7d
    row_v1 = out[out["video_id"] == "v1"].iloc[0]
    assert row_v1["videos_last_7d"] == 0


def test_timing_videos_last_7d_counts_prior(minimal_videos):
    # v2 at 2024-07-05, v1 at 2024-07-01 (4 days earlier) → should count v1
    out = timing_features(minimal_videos)
    row_v2 = out[out["video_id"] == "v2"].iloc[0]
    assert row_v2["videos_last_7d"] == 1


# ================================================================== #
# Group 3 — description_features                                      #
# ================================================================== #

def test_description_null_desc_gives_zero_len(minimal_videos):
    out = description_features(minimal_videos)
    row = out[out["video_id"] == "v2"].iloc[0]
    assert row["desc_len"] == 0


def test_description_n_hashtags_from_list(minimal_videos):
    out = description_features(minimal_videos)
    row = out[out["video_id"] == "v1"].iloc[0]
    assert row["n_hashtags"] == 1


def test_description_has_sticker_text(minimal_videos):
    out = description_features(minimal_videos)
    assert out.loc[out["video_id"] == "v2", "has_sticker_text"].iloc[0] == 1
    assert out.loc[out["video_id"] == "v1", "has_sticker_text"].iloc[0] == 0


# ================================================================== #
# Group 4 — speech_features                                           #
# ================================================================== #

def test_speech_has_transcript(minimal_videos):
    out = speech_features(minimal_videos)
    assert out.loc[out["video_id"] == "v1", "has_transcript"].iloc[0] == 1
    assert out.loc[out["video_id"] == "v2", "has_transcript"].iloc[0] == 0


def test_speech_transcript_n_words_null_for_no_transcript(minimal_videos):
    out = speech_features(minimal_videos)
    assert pd.isna(out.loc[out["video_id"] == "v2", "transcript_n_words"].iloc[0])


# ================================================================== #
# Group 8 — music_features (temporal leakage test)                    #
# ================================================================== #

def test_sound_popularity_first_video_is_zero(minimal_videos):
    """The first video using a music_id should have sound_popularity = 0."""
    out = music_features(minimal_videos)
    # v1 is 2024-07-01 and is the first video with music_id=m1
    row_v1 = out[out["video_id"] == "v1"].iloc[0]
    assert row_v1["sound_popularity"] == 0


def test_sound_popularity_later_video_counts_past(minimal_videos):
    """v2 (2024-07-05, music_id=m1) should see v1 (2024-07-01, m1) as prior."""
    out = music_features(minimal_videos)
    row_v2 = out[out["video_id"] == "v2"].iloc[0]
    assert row_v2["sound_popularity"] >= 1  # at least v1 used m1 before v2


def test_sound_popularity_same_day_not_counted():
    """Two videos with the same music_id on the same day should each see 0 past uses."""
    videos = pd.DataFrame(
        {
            "video_id":   ["x1", "x2", "x3"],
            "author_id":  ["a", "b", "c"],
            "create_date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "music_id":   ["m", "m", "m"],
            "music_owner_id": ["a", "b", "c"],
            "music_selected_from": ["original", "original", "original"],
        }
    )
    out = music_features(videos)
    # x1 and x2 are on day 1 → 0 prior uses of m
    assert out.loc[out["video_id"] == "x1", "sound_popularity"].iloc[0] == 0
    assert out.loc[out["video_id"] == "x2", "sound_popularity"].iloc[0] == 0
    # x3 is on day 2 → 2 prior uses (x1 and x2)
    assert out.loc[out["video_id"] == "x3", "sound_popularity"].iloc[0] == 2


def test_original_sound_flag(minimal_videos):
    """original_sound=1 when music_owner_id == author_id."""
    out = music_features(minimal_videos)
    # v1: author_id=a, music_owner_id=a → 1
    assert out.loc[out["video_id"] == "v1", "original_sound"].iloc[0] == 1
    # v2: author_id=a, music_owner_id=x → 0
    assert out.loc[out["video_id"] == "v2", "original_sound"].iloc[0] == 0


# ================================================================== #
# Group 9 — hashtag_indicator_features (leakage test)                 #
# ================================================================== #

def test_hashtag_top_uses_train_only():
    """Top hashtags must be determined from train videos only.

    Setup: train has only #fyp; test has only #dance.
    The indicator column for #fyp should exist; there should be no #dance column
    (or it should not appear in top-20 computed from train alone).
    """
    videos = pd.DataFrame(
        {
            "video_id": ["v1", "v2"],
            "hashtags": [
                np.array([{"hashtag_id": "1", "hashtag_name": "fyp"}]),
                np.array([{"hashtag_id": "2", "hashtag_name": "dance"}]),
            ],
        }
    )
    train_ids = ["v1"]  # only v1 is train
    out = hashtag_indicator_features(videos, train_video_ids=train_ids)

    assert "hashtag_fyp" in out.columns, "#fyp must appear (it's in train top-1)"
    assert "hashtag_dance" not in out.columns, "#dance must NOT appear (test-only)"


def test_hashtag_indicator_values_correct():
    videos = pd.DataFrame(
        {
            "video_id": ["v1", "v2"],
            "hashtags": [
                np.array([{"hashtag_id": "1", "hashtag_name": "fyp"}]),
                np.array([{"hashtag_id": "2", "hashtag_name": "cats"}]),
            ],
        }
    )
    out = hashtag_indicator_features(videos, train_video_ids=["v1", "v2"])
    # v1 has #fyp → indicator = 1; v2 does not → 0
    assert out.loc[out["video_id"] == "v1", "hashtag_fyp"].iloc[0] == 1
    assert out.loc[out["video_id"] == "v2", "hashtag_fyp"].iloc[0] == 0


# ================================================================== #
# build_features — no engagement_daily columns                        #
# ================================================================== #

def test_build_features_no_engagement_columns(minimal_videos, minimal_creator_daily):
    """build_features output must contain no engagement_daily column names."""
    engagement_cols = {
        "play_count", "like_count", "comment_count", "share_count",
        "collect_count", "download_count", "whatsapp_share_count",
        "days_since_post", "date",
    }
    out = build_features(minimal_videos, minimal_creator_daily, train_video_ids=["v1", "v2"])
    leaked = set(out.columns) & engagement_cols
    assert not leaked, f"Engagement columns leaked into features: {leaked}"


def test_build_features_has_video_id_for_all_inputs(minimal_videos, minimal_creator_daily):
    out = build_features(minimal_videos, minimal_creator_daily)
    assert set(out["video_id"]) == set(minimal_videos["video_id"])


# ================================================================== #
# creator_context_features — as-of join is backward-only             #
# ================================================================== #

def test_creator_context_uses_past_follower_count(minimal_videos, minimal_creator_daily):
    """follower_count_log must use the snapshot on or before publish date, not after."""
    out = creator_context_features(minimal_videos, minimal_creator_daily)
    # v1: author_id=a, create_date=2024-07-01
    #   creator_daily has rows for a on 2024-06-30 (1000) and 2024-07-04 (1100)
    #   backward as-of: closest before 2024-07-01 → 2024-06-30 → 1000
    row_v1 = out[out["video_id"] == "v1"].iloc[0]
    assert np.isclose(row_v1["follower_count_log"], np.log1p(1000.0))
    # v2: author_id=a, create_date=2024-07-05
    #   closest before → 2024-07-04 → 1100
    row_v2 = out[out["video_id"] == "v2"].iloc[0]
    assert np.isclose(row_v2["follower_count_log"], np.log1p(1100.0))
