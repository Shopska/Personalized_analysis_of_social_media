"""Feature engineering: turn ``videos_with_target.parquet`` into a model-ready table.

Every feature here must be **known at the moment the video is posted**. That single rule
drives most design decisions in this module:

- engagement counters (views, likes, comments, shares, ...) and everything derived from
  them are never features — they are what we predict;
- follower counts come from the last ``creator_daily`` snapshot strictly *before* the
  posting day (the EDA used the nearest snapshot, which can be after posting);
- a creator's "typical performance" is computed only from their *earlier* videos, and
  only from videos posted at least ``gap_days`` before, because a video posted
  yesterday has not finished collecting views yet.

The public entry point is :func:`build_features`; the ``add_*`` functions are the
individual steps, kept separate so the notebook can show and check each one.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

ID_COLUMNS = ["video_id", "author_id", "create_time"]
TARGET_COLUMNS = ["log_views", "engagement_rate"]
EMOTIONS = ["anger", "joy", "surprise", "sadness", "disgust", "fear"]
TEXT_STATS = ["word_count", "emoji_count", "question_count", "hashtag_count", "speaking_rate"]

# Columns of videos_with_target.parquet that must never become features: they are
# measured after posting (the target itself or its components).
LEAKY_COLUMNS = [
    "play_count",
    "like_count",
    "comment_count",
    "share_count",
    "collect_count",
    "like_rate",
    "comment_rate",
    "share_rate",
    "collect_rate",
    "observed_day",
    "follower_count",  # EDA version used the *nearest* snapshot, possibly after posting
]

# Feature groups. Kept as explicit lists so that models and SHAP plots can refer to
# whole groups (e.g. "the global model without creator features" = drop CREATOR_*).
CONTENT_FEATURES = (
    ["duration", "resolution", "is_english", "created_by_ai", "is_ads"]
    + ["has_desc", "desc_length", "n_hashtags", "has_sticker_text", "sticker_text_length"]
    + ["has_transcript", "has_text_stats"]
    + TEXT_STATS
    + ["has_emotions"]
    + EMOTIONS
)
TIME_FEATURES = ["hour_sin", "hour_cos", "weekday", "is_weekend"]
MUSIC_FEATURES = ["is_own_sound", "music_uses_prev_window"]
ACCOUNT_FEATURES = [
    "log_followers",
    "log_following",
    "log_total_favorited",
    "account_video_count",
    "has_account_stats",
]
HISTORY_FEATURES = [
    "hist_n_videos",
    "hist_mean_log_views",
    "hist_std_log_views",
    "hist_recent_mean_log_views",
    "hist_mean_engagement_rate",
    "days_since_prev_video",
    "n_videos_prev_7d",
]

_SECONDS_PER_DAY = 86_400
# Group codes are multiplied by this so that "group, then time" sorts as one int64 key.
# Unix timestamps in seconds are ~1.7e9, safely below it.
_GROUP_STRIDE = 10**10


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def _posting_time(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["create_time"])


def _sorted_group_keys(groups: pd.Series, times: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Sort rows by (group, time) and return ``(order, sorted_key)``.

    ``sorted_key`` is one increasing int64 per row (``group_code * stride + seconds``),
    which lets every "how many rows of the same group happened before time t" question
    be answered with a vectorized :func:`numpy.searchsorted` instead of a slow Python
    loop over 200k rows.
    """
    codes, _ = pd.factorize(groups, sort=True)
    seconds = times.astype("datetime64[s]").astype("int64").to_numpy()
    raw_key = codes.astype("int64") * _GROUP_STRIDE + seconds
    order = np.argsort(raw_key, kind="stable")
    return order, raw_key[order]


def _prefix(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prefix sums of values, squared values and non-missing counts (with a leading 0)."""
    present = ~np.isnan(values)
    filled = np.where(present, values, 0.0)
    zero = np.zeros(1)
    return (
        np.concatenate([zero, np.cumsum(filled)]),
        np.concatenate([zero, np.cumsum(filled**2)]),
        np.concatenate([zero, np.cumsum(present)]),
    )


def _window_mean(
    cumsum: np.ndarray, cumcount: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> np.ndarray:
    """Mean of the non-missing values in sorted rows ``[lo, hi)``, via prefix sums."""
    total = cumsum[hi] - cumsum[lo]
    count = cumcount[hi] - cumcount[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


# --------------------------------------------------------------------------------------
# Feature groups
# --------------------------------------------------------------------------------------


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Posting hour as a point on a circle, weekday, and a weekend flag.

    Hour is encoded as sin/cos so that 23:00 and 00:00 are neighbours for the model,
    not 23 units apart.
    """
    out = df.copy()
    t = _posting_time(out)
    hour = t.dt.hour + t.dt.minute / 60
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["weekday"] = t.dt.dayofweek
    out["is_weekend"] = (out["weekday"] >= 5).astype(int)
    return out


def add_content_features(df: pd.DataFrame) -> pd.DataFrame:
    """Video format and text features that are fully known before posting.

    Missing values that mean "absent" (no description, no sticker text) become 0 plus a
    flag. The five text statistics are missing *together* for ~38% of videos, so they
    stay NaN (both RandomForest and XGBoost handle NaN) and get one ``has_text_stats``
    flag instead of being filled with a misleading 0.
    """
    out = df.copy()
    out["resolution"] = pd.to_numeric(
        out["ratio"].astype("string").str.extract(r"(\d+)")[0], errors="coerce"
    ).astype(float)
    # float, not pandas' nullable Int64: scikit-learn can't read pd.NA
    out["is_ads"] = pd.to_numeric(out["is_ads"], errors="coerce").astype(float)
    out["created_by_ai"] = pd.to_numeric(out["created_by_ai"], errors="coerce").astype(float)

    desc = out["desc"].fillna("").astype(str)
    out["has_desc"] = (desc.str.len() > 0).astype(int)
    out["desc_length"] = desc.str.len()
    # `hashtags` is complete, unlike `hashtag_count` (part of the 38%-missing text stats).
    out["n_hashtags"] = out["hashtags"].map(
        lambda tags: len(tags) if isinstance(tags, (list, tuple, np.ndarray)) else 0
    )

    sticker = out["sticker_text"].fillna("").astype(str)
    out["has_sticker_text"] = (sticker.str.len() > 0).astype(int)
    out["sticker_text_length"] = sticker.str.len()

    out["has_transcript"] = out["transcript"].notna().astype(int)
    out["has_text_stats"] = out[TEXT_STATS].notna().all(axis=1).astype(int)
    out["has_emotions"] = out[EMOTIONS].notna().all(axis=1).astype(int)
    return out


def add_music_features(df: pd.DataFrame, trend_window_days: int = 7) -> pd.DataFrame:
    """Own sound vs borrowed sound, and how "trending" the sound is at posting time.

    ``music_uses_prev_window`` counts videos in the dataset that used the same
    ``music_id`` during the ``trend_window_days`` days *before* this video. It is a
    rough proxy: the dataset is a sample of creators, not all of TikTok.
    """
    out = df.copy()
    own = out["music_owner_id"].astype("string") == out["author_id"].astype("string")
    out["is_own_sound"] = own.fillna(False).astype(int)

    uses = np.full(len(out), np.nan)
    has_music = out["music_id"].notna().to_numpy()
    if has_music.any():
        sub = out.loc[has_music]
        order, key = _sorted_group_keys(sub["music_id"].astype(str), _posting_time(sub))
        first_same_time = np.searchsorted(key, key, side="left")
        window_start = np.searchsorted(key, key - trend_window_days * _SECONDS_PER_DAY, side="left")
        counts = np.empty(len(sub))
        counts[order] = first_same_time - window_start
        uses[has_music] = counts
    out["music_uses_prev_window"] = uses
    return out


def add_follower_features(
    df: pd.DataFrame, creator_daily: pd.DataFrame, author_column: str = "author_id"
) -> pd.DataFrame:
    """Account size from the last ``creator_daily`` snapshot strictly before posting day.

    Strictly before (``allow_exact_matches=False``) because a snapshot dated on the
    posting day may have been taken after the video went live. Videos with no earlier
    snapshot get NaN and ``has_account_stats = 0``.
    """
    out = df.copy()
    columns = ["follower_count", "following_count", "total_favorited", "video_count"]
    snap = creator_daily.rename(columns={"author_id": author_column})
    snap = snap[[author_column, "date"] + columns].copy()
    snap["date"] = pd.to_datetime(snap["date"]).astype("datetime64[ns]")
    snap[author_column] = snap[author_column].astype(str)
    snap = snap.dropna(subset=["follower_count"]).sort_values("date")

    left = pd.DataFrame(
        {
            "_row": np.arange(len(out)),
            author_column: out[author_column].astype(str).to_numpy(),
            "_day": _posting_time(out).dt.normalize().astype("datetime64[ns]").to_numpy(),
        }
    ).sort_values("_day")
    merged = pd.merge_asof(
        left,
        snap,
        left_on="_day",
        right_on="date",
        by=author_column,
        direction="backward",
        allow_exact_matches=False,
    ).sort_values("_row")

    def numeric(col: str) -> np.ndarray:
        return pd.to_numeric(merged[col], errors="coerce").clip(lower=0).to_numpy()

    out["log_followers"] = np.log1p(numeric("follower_count"))
    out["log_following"] = np.log1p(numeric("following_count"))
    out["log_total_favorited"] = np.log1p(numeric("total_favorited"))
    out["account_video_count"] = numeric("video_count")
    out["has_account_stats"] = merged["follower_count"].notna().astype(int).to_numpy()
    return out


def add_creator_history_features(
    df: pd.DataFrame,
    author_column: str = "author_id",
    gap_days: int = 7,
    recent_k: int = 5,
) -> pd.DataFrame:
    """Each creator's past performance and posting rhythm, from earlier videos only.

    Performance history (uses targets of *other, earlier* videos, posted at least
    ``gap_days`` before this one):

    - ``hist_n_videos`` — how many such videos the creator has;
    - ``hist_mean_log_views`` / ``hist_std_log_views`` — the creator's typical level
      and volatility of log views;
    - ``hist_recent_mean_log_views`` — mean over the last ``recent_k`` of them
      ("current form");
    - ``hist_mean_engagement_rate`` — typical engagement rate.

    Posting rhythm (no targets involved, so no gap needed):
    ``days_since_prev_video`` and ``n_videos_prev_7d``.

    Why the gap: views keep growing for ~2 weeks after posting (EDA: ~86% of final views
    by day 5, ~92% by day 10). The final views of a video posted yesterday did not exist
    yet when today's video was posted.
    """
    out = df.copy()
    order, key = _sorted_group_keys(out[author_column].astype(str), _posting_time(out))
    n = len(out)

    group_start = np.searchsorted(key, (key // _GROUP_STRIDE) * _GROUP_STRIDE, side="left")
    first_same_time = np.searchsorted(key, key, side="left")
    hist_end = np.searchsorted(key, key - gap_days * _SECONDS_PER_DAY, side="left")
    hist_end = np.maximum(hist_end, group_start)
    recent_start = np.maximum(hist_end - recent_k, group_start)

    def to_original_order(values_sorted: np.ndarray) -> np.ndarray:
        result = np.empty(n)
        result[order] = values_sorted
        return result

    s, s2, c = _prefix(out["log_views"].to_numpy(dtype=float)[order])
    mean = _window_mean(s, c, group_start, hist_end)
    count = c[hist_end] - c[group_start]
    with np.errstate(invalid="ignore"):
        var = (s2[hist_end] - s2[group_start]) / np.maximum(count, 1) - mean**2
    std = np.where(count >= 2, np.sqrt(np.clip(var, 0, None)), np.nan)

    er_s, _, er_c = _prefix(out["engagement_rate"].to_numpy(dtype=float)[order])

    seconds = key % _GROUP_STRIDE
    prev = first_same_time - 1
    days_since_prev = np.where(
        prev >= group_start,
        (seconds - seconds[np.maximum(prev, 0)]) / _SECONDS_PER_DAY,
        np.nan,
    )
    week_start = np.searchsorted(key, key - 7 * _SECONDS_PER_DAY, side="left")

    out["hist_n_videos"] = to_original_order((hist_end - group_start).astype(float))
    out["hist_mean_log_views"] = to_original_order(mean)
    out["hist_std_log_views"] = to_original_order(std)
    out["hist_recent_mean_log_views"] = to_original_order(
        _window_mean(s, c, recent_start, hist_end)
    )
    out["hist_mean_engagement_rate"] = to_original_order(
        _window_mean(er_s, er_c, group_start, hist_end)
    )
    out["days_since_prev_video"] = to_original_order(days_since_prev)
    out["n_videos_prev_7d"] = to_original_order((first_same_time - week_start).astype(float))
    return out


def encode_categoricals(df: pd.DataFrame, columns: list[str], min_count: int = 500) -> pd.DataFrame:
    """One-hot encode categorical columns, merging rare categories into ``"other"``.

    One-hot (rather than XGBoost's native categories) because RandomForest in
    scikit-learn needs numbers, and because each category then gets its own SHAP value,
    which is easy to explain ("music from edit_page_recent: +0.3"). Rare categories are
    merged so that models and SHAP plots aren't cluttered by labels with few videos.
    """
    out = df.copy()
    for col in columns:
        values = out[col].astype("string").fillna("missing")
        counts = values.value_counts()
        rare = counts[counts < min_count].index
        values = values.where(~values.isin(rare), "other")
        dummies = pd.get_dummies(values, prefix=col, prefix_sep="__", dtype=int)
        out = pd.concat([out.drop(columns=[col]), dummies], axis=1)
    return out


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """All columns of a built feature table that are model inputs (not ids, not targets)."""
    return [c for c in df.columns if c not in ID_COLUMNS + TARGET_COLUMNS]


def build_features(
    videos: pd.DataFrame, creator_daily: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """Run every step above and return ids + targets + features, one row per video.

    Parameters
    ----------
    videos : DataFrame
        ``data/interim/videos_with_target.parquet`` (output of the EDA notebook).
    creator_daily : DataFrame
        The ``creator_daily`` table of the dataset.
    config : dict
        Loaded ``config/default.yaml``; uses ``data.author_column`` and the
        ``features`` section.
    """
    author = config["data"]["author_column"]
    fcfg = config["features"]

    df = videos.drop(columns=[c for c in LEAKY_COLUMNS if c in videos.columns])
    df = add_time_features(df)
    df = add_content_features(df)
    df = add_music_features(df, trend_window_days=fcfg["music_trend_window_days"])
    df = add_follower_features(df, creator_daily, author_column=author)
    df = add_creator_history_features(
        df,
        author_column=author,
        gap_days=fcfg["history_gap_days"],
        recent_k=fcfg["history_recent_k"],
    )

    categorical = fcfg["categorical"]
    keep = (
        ID_COLUMNS
        + TARGET_COLUMNS
        + CONTENT_FEATURES
        + TIME_FEATURES
        + MUSIC_FEATURES
        + ACCOUNT_FEATURES
        + HISTORY_FEATURES
        + categorical
    )
    table = encode_categoricals(df[keep], categorical, min_count=fcfg["rare_category_min_count"])
    return table.reset_index(drop=True)
