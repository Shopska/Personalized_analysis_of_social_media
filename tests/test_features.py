"""Tests for creator_analytics.features — synthetic data only, no real dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creator_analytics.data import prepare_dataset
from creator_analytics.features import (
    add_categorical_features,
    add_creator_context_features,
    add_description_features,
    add_posting_history_features,
    add_time_columns,
    add_vs_creator_features,
    build_features,
    count_prev_in_window,
    describe_feature,
    follower_asof_join,
    hashtag_names,
    impute_numeric,
    select_features,
    sound_history,
    top_hashtags,
)


@pytest.fixture
def videos():
    return pd.DataFrame(
        {
            "video_id": ["v1", "v2", "v3", "v4"],
            "author_id": ["a", "a", "b", "b"],
            "split": ["train", "test", "train", "train"],
            "create_time": [
                "2024-07-01 10:00:00",
                "2024-07-05 18:00:00",
                "2024-07-01 08:00:00",
                "2024-07-03 22:00:00",
            ],
            "create_date": ["2024-07-01", "2024-07-05", "2024-07-01", "2024-07-03"],
            "duration": [30.0, 60.0, 90.0, 15.0],
            "ratio": ["540p", "720p", "540p", "540p"],
            "created_by_ai": [0.0, 1.0, 0.0, 0.0],
            "is_english": [1, 1, 0, 1],
            "desc": ["hello #fyp @bob?", None, "world", "#dance"],
            "sticker_text": [None, "buy now", None, None],
            "hashtags": [
                np.array([{"hashtag_id": "1", "hashtag_name": "fyp"}]),
                np.array([{"hashtag_id": "2", "hashtag_name": "dance"}]),
                np.array([]),
                np.array([{"hashtag_id": "3", "hashtag_name": "Dance"}]),
            ],
            "transcript": ["Hello world.", None, "Hi there?", None],
            "anger": [0.1, 0.8, 0.2, np.nan],
            "joy": [0.7, 0.1, 0.5, 0.4],
            "surprise": [0.1, 0.05, 0.2, 0.2],
            "sadness": [0.05, 0.03, 0.05, 0.05],
            "disgust": [0.02, 0.01, 0.03, 0.01],
            "fear": [0.03, 0.01, 0.02, 0.01],
            "topic": ["Life", "Dance", "Life", "Dance"],
            "music_id": ["m1", "m1", "m2", "m1"],
            "music_owner_id": ["a", "x", "b", "y"],
            "music_title": ["original sound - a", "song", "song", "song"],
            "music_selected_from": ["original", "single_song", "original", None],
        }
    )


@pytest.fixture
def creator_daily():
    return pd.DataFrame(
        {
            "author_id": ["a", "a", "a", "b", "b"],
            "date": ["2024-06-30", "2024-07-01", "2024-07-04", "2024-06-30", "2024-07-02"],
            "follower_count": [1000.0, 5000.0, 1100.0, 500.0, 510.0],
            "following_count": [100.0, 100.0, 100.0, 50.0, 50.0],
            "total_favorited": [10, 10, 12, 5, 6],
            "video_count": [10.0, 11.0, 12.0, 5.0, 6.0],
        }
    )


def test_hashtag_names_parses_dicts_strings_and_empty():
    assert hashtag_names(np.array([{"hashtag_name": "FYP"}, {"hashtag_name": " "}])) == ["fyp"]
    assert hashtag_names(np.array([])) == []
    assert hashtag_names(None) == []
    assert hashtag_names("#a #b") == ["a", "b"]


def test_count_prev_in_window_counts_only_past_same_group():
    frame = pd.DataFrame(
        {
            "g": ["x", "x", "x", "y"],
            "t": pd.to_datetime(["2024-07-01", "2024-07-03", "2024-07-20", "2024-07-02"]),
        }
    )
    assert count_prev_in_window(frame, "g", "t", 7).tolist() == [0, 1, 0, 0]


def test_prepare_dataset_excludes_nonmonotonic_and_small_creators():
    videos = pd.DataFrame(
        {
            "video_id": [f"v{i}" for i in range(9)],
            "author_id": ["a"] * 5 + ["b"] * 3 + ["c"],
        }
    )
    rows = [(f"v{i}", 14, 100) for i in range(9)] + [("v0", 21, 50)]
    engagement = pd.DataFrame(rows, columns=["video_id", "days_since_post", "play_count"])
    df = prepare_dataset(videos, engagement, min_videos=3, test_size=0.34, seed=0)
    assert "v0" not in set(df["video_id"])
    assert set(df["author_id"]) == {"a", "b"}
    assert set(df.loc[df["split"] == "test", "author_id"]) == {"a", "b"}


def test_posting_history_first_video_is_flagged(videos):
    df, _ = add_posting_history_features(add_time_columns(videos))
    first_a = df[df["video_id"] == "v1"].iloc[0]
    second_a = df[df["video_id"] == "v2"].iloc[0]
    assert first_a["is_first_in_window"] == 1 and np.isnan(first_a["log_hours_since_prev"])
    assert second_a["is_first_in_window"] == 0
    assert second_a["videos_prev_7d"] == 1


def test_creator_context_uses_snapshot_strictly_before_publish_day(videos, creator_daily):
    df, _ = add_creator_context_features(add_time_columns(videos), creator_daily)
    v1 = df[df["video_id"] == "v1"].iloc[0]
    assert np.isclose(v1["followers_log"], np.log1p(1000.0))


def test_creator_context_falls_back_to_same_day_snapshot(videos, creator_daily):
    later = creator_daily[creator_daily["date"] >= "2024-07-01"]
    df, _ = add_creator_context_features(add_time_columns(videos), later)
    v1 = df[df["video_id"] == "v1"].iloc[0]
    assert np.isclose(v1["followers_log"], np.log1p(5000.0))


def test_sound_history_uses_only_earlier_videos(videos):
    hist = sound_history(videos)
    assert hist.loc["v1", "sound_prev_uses"] == 0
    assert hist.loc["v4", "sound_prev_uses"] == 1
    assert hist.loc["v2", "sound_prev_uses"] == 2
    assert hist.loc["v2", "creator_reused_sound"] == 1
    assert hist.loc["v4", "creator_reused_sound"] == 0


def test_top_hashtags_use_train_only(videos):
    df, _ = add_description_features(videos)
    tags = top_hashtags(df, top_n=10)
    assert "dance" in tags.index
    assert tags["dance"] == 1


def test_impute_numeric_uses_train_values_only(videos):
    df = videos.assign(x=[np.nan, 100.0, 2.0, 4.0])
    df.loc[df["video_id"] == "v2", "split"] = "test"
    out = impute_numeric(df, ["x"])
    assert out.loc[out["video_id"] == "v1", "x"].iloc[0] == 3.0


def test_vs_creator_ignores_test_values(videos):
    df = videos.assign(duration_log=[1.0, 50.0, 2.0, 4.0])
    out, cols = add_vs_creator_features(df, ["duration_log"])
    assert cols == ["duration_log_vs_creator"]
    assert out.loc[out["video_id"] == "v2", "duration_log_vs_creator"].iloc[0] == 49.0


def test_categorical_rare_levels_go_to_other(videos):
    out, groups, levels = add_categorical_features(videos, ["topic"], min_share=0.5)
    assert levels["topic"] == ["Life", "other"]
    assert "topic_other" in groups["cat_topic"]
    assert out.loc[out["video_id"] == "v2", "topic_other"].iloc[0] == 1


def test_select_features_drops_duplicates_and_respects_priority():
    df = pd.DataFrame(
        {
            "split": ["train"] * 5,
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],
            "c": [5, 3, 4, 1, 2],
            "const": [1, 1, 1, 1, 1],
        }
    )
    sel = select_features(df, {"g": ["a", "b", "c", "const"]})
    assert sel.zero_variance == ["const"]
    assert sel.redundant == ["b"]
    sel = select_features(df, {"g": ["a", "b", "c", "const"]}, keep_priority=["b"])
    assert sel.redundant == ["a"]


def test_build_features_returns_clean_matrix(videos, creator_daily):
    feats = build_features(videos.drop(columns="split"), creator_daily, {"v1", "v3", "v4"})
    assert set(feats["video_id"]) == set(videos["video_id"])
    assert not feats.drop(columns="video_id").isna().any().any()
    assert not {"play_count", "like_count", "target"} & set(feats.columns)


def test_describe_feature_handles_generated_names():
    assert "#fyp" in describe_feature("ht_fyp")
    assert "relative" in describe_feature("duration_log_vs_creator")
    assert "Life" in describe_feature("topic_Life")


def test_follower_asof_join_kept_for_notebook_02(videos, creator_daily):
    out = follower_asof_join(videos, creator_daily)
    assert out.loc[out["video_id"] == "v1", "follower_count_at_publish"].iloc[0] == 5000.0
