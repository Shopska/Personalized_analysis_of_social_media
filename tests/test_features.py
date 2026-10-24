"""Tests for creator_analytics.features on small synthetic data (no download needed).

The most important ones check for data leakage: a feature of a video must not change
when the video's own target, or the target of any later video, changes.
"""

import numpy as np
import pandas as pd
import pytest

from creator_analytics.features import (
    ID_COLUMNS,
    TARGET_COLUMNS,
    add_creator_history_features,
    add_follower_features,
    add_music_features,
    build_features,
    encode_categoricals,
    get_feature_columns,
)


def _videos(n_per_creator: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for author in ["a", "b", "c"]:
        start = pd.Timestamp("2024-07-01")
        offsets = np.sort(rng.uniform(0, 90 * 24 * 3600, n_per_creator))
        for i, off in enumerate(offsets):
            rows.append(
                {
                    "video_id": f"{author}{i}",
                    "author_id": author,
                    "create_time": str(start + pd.Timedelta(seconds=int(off))),
                    "log_views": rng.normal(7, 2),
                    "engagement_rate": rng.uniform(0, 0.2),
                    "music_id": rng.choice(["m1", "m2", None]),
                    "music_owner_id": rng.choice([author, "other"]),
                }
            )
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


def _brute_force_hist_mean(df: pd.DataFrame, gap_days: int) -> pd.Series:
    t = pd.to_datetime(df["create_time"])
    out = []
    for i in range(len(df)):
        mask = (df["author_id"] == df.loc[i, "author_id"]) & (
            t < t[i] - pd.Timedelta(days=gap_days)
        )
        out.append(df.loc[mask, "log_views"].mean() if mask.any() else np.nan)
    return pd.Series(out)


@pytest.mark.parametrize("gap_days", [0, 7])
def test_history_mean_matches_brute_force(gap_days):
    df = _videos()
    fast = add_creator_history_features(df, gap_days=gap_days)["hist_mean_log_views"]
    slow = _brute_force_hist_mean(df, gap_days)
    np.testing.assert_allclose(fast.to_numpy(), slow.to_numpy(), equal_nan=True)


def test_history_ignores_own_and_future_targets():
    df = _videos()
    before = add_creator_history_features(df, gap_days=7)

    # Change the target of the latest video of creator "a": nothing of "a" may change,
    # because no other video of "a" is later than it.
    latest = pd.to_datetime(df["create_time"]).where(df["author_id"] == "a").idxmax()
    changed = df.copy()
    changed.loc[latest, "log_views"] += 100
    after = add_creator_history_features(changed, gap_days=7)

    hist_cols = [c for c in before.columns if c.startswith("hist_")]
    pd.testing.assert_frame_equal(before[hist_cols], after[hist_cols])


def test_history_is_per_creator():
    df = _videos()
    before = add_creator_history_features(df, gap_days=7)
    changed = df.copy()
    changed.loc[changed["author_id"] == "b", "log_views"] += 100
    after = add_creator_history_features(changed, gap_days=7)
    mask = df["author_id"] != "b"
    pd.testing.assert_series_equal(
        before.loc[mask, "hist_mean_log_views"], after.loc[mask, "hist_mean_log_views"]
    )


def test_first_videos_have_no_history():
    df = _videos()
    out = add_creator_history_features(df, gap_days=7)
    first = pd.to_datetime(df["create_time"]).groupby(df["author_id"]).idxmin()
    assert out.loc[first, "hist_n_videos"].eq(0).all()
    assert out.loc[first, "hist_mean_log_views"].isna().all()
    assert out.loc[first, "days_since_prev_video"].isna().all()


def test_followers_use_only_earlier_snapshots():
    videos = pd.DataFrame(
        {
            "author_id": ["a", "a", "a"],
            "create_time": ["2024-07-01 10:00", "2024-07-03 10:00", "2024-07-10 10:00"],
        }
    )
    creator_daily = pd.DataFrame(
        {
            "author_id": ["a", "a", "a"],
            "date": ["2024-07-01", "2024-07-02", "2024-07-03"],
            "follower_count": [100.0, 200.0, 300.0],
            "following_count": [1.0, 1.0, 1.0],
            "total_favorited": [10, 20, 30],
            "video_count": [5.0, 6.0, 7.0],
        }
    )
    out = add_follower_features(videos, creator_daily)
    # Day 1: same-day snapshot is not allowed -> missing. Day 3: snapshot of day 2.
    assert np.isnan(out.loc[0, "log_followers"])
    assert out.loc[0, "has_account_stats"] == 0
    assert out.loc[1, "log_followers"] == pytest.approx(np.log1p(200))
    assert out.loc[2, "log_followers"] == pytest.approx(np.log1p(300))


def test_music_trend_counts_only_previous_uses():
    df = pd.DataFrame(
        {
            "author_id": ["a", "b", "c", "d"],
            "create_time": ["2024-07-01", "2024-07-02", "2024-07-20", "2024-07-21"],
            "music_id": ["m", "m", "m", None],
            "music_owner_id": ["a", "a", None, None],
        }
    )
    out = add_music_features(df, trend_window_days=7)
    assert out["music_uses_prev_window"].tolist()[:3] == [0, 1, 0]
    assert np.isnan(out.loc[3, "music_uses_prev_window"])
    assert out["is_own_sound"].tolist() == [1, 0, 0, 0]


def test_encode_categoricals_groups_rare_values():
    df = pd.DataFrame({"topic": ["x"] * 5 + ["y"] * 2 + [None]})
    out = encode_categoricals(df, ["topic"], min_count=3)
    assert set(out.columns) == {"topic__x", "topic__other"}
    assert out.sum().to_dict() == {"topic__x": 5, "topic__other": 3}


def test_build_features_has_no_leaky_columns():
    df = _videos()
    df["play_count"] = 1
    df["like_rate"] = 0.1
    df["follower_count"] = 5
    for col in ["duration", "is_english", "created_by_ai"]:
        df[col] = 1.0
    df["ratio"] = "720p"
    df["is_ads"] = "0"
    df["desc"] = "#a #b"
    df["hashtags"] = [["a", "b"]] * len(df)
    df["sticker_text"] = None
    df["transcript"] = "hello"
    for col in [
        "word_count",
        "emoji_count",
        "question_count",
        "hashtag_count",
        "speaking_rate",
        "anger",
        "joy",
        "surprise",
        "sadness",
        "disgust",
        "fear",
    ]:
        df[col] = 0.5
    df["topic"] = "Lifestyle"
    df["music_selected_from"] = "original"
    creator_daily = pd.DataFrame(
        {
            "author_id": ["a"],
            "date": ["2024-07-15"],
            "follower_count": [1000.0],
            "following_count": [10.0],
            "total_favorited": [50],
            "video_count": [20.0],
        }
    )
    config = {
        "data": {"author_column": "author_id"},
        "features": {
            "history_gap_days": 7,
            "history_recent_k": 5,
            "music_trend_window_days": 7,
            "rare_category_min_count": 1,
            "categorical": ["topic", "music_selected_from"],
        },
    }
    table = build_features(df, creator_daily, config)
    features = get_feature_columns(table)

    assert len(table) == len(df)
    assert not {"play_count", "like_rate", "follower_count"} & set(table.columns)
    assert not set(ID_COLUMNS + TARGET_COLUMNS) & set(features)
    assert "topic__Lifestyle" in features
