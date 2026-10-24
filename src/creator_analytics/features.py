"""Feature engineering for TikTok creator analytics.

Each ``add_*`` function takes the prepared video table (one row per video, with a
``split`` column from :func:`creator_analytics.data.prepare_dataset`) and returns
``(table_with_new_columns, new_feature_names)``. The notebook
``notebooks/03-feature-engineering.ipynb`` calls them step by step;
:func:`build_feature_table` runs the same steps end to end.

Leakage rules
-------------
- No ``engagement_daily`` column is ever used as a feature.
- Anything learned from data (fill values, creator medians, top hashtags, category
  levels) is computed on the train split only and then applied to all videos.
- Anything time-dependent (posting history, follower counts, sound popularity) uses
  only data from before the video's publish time.
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EMOTIONS = ["anger", "joy", "surprise", "sadness", "disgust", "fear"]
CREATOR_DAILY_COLS = ["follower_count", "following_count", "total_favorited", "video_count"]
CATEGORICAL = ["ratio", "topic", "music_source", "dominant_emotion"]
VS_CREATOR_BASE = [
    "duration_log",
    "desc_len_log",
    "n_hashtags",
    "sticker_len_log",
    "log_hours_since_prev",
    "emotion_max",
    "sound_prev_uses_log",
]
META_COLUMNS = [
    "video_id",
    "author_id",
    "split",
    "create_dt",
    "target_abs",
    "target_rel",
    "creator_baseline",
]
ORIGINAL_SOUND_PATTERN = "original sound|оригінальний звук|son original|sonido original"
EMOJI_PATTERN = (
    "[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff"
    "\U0001f1e0-\U0001f1ff\U00002702-\U000027b0\U0001f900-\U0001f9ff"
    "\U0001fa70-\U0001faff\U00002600-\U000026ff]"
)

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "duration_log": "Video duration, log1p(seconds)",
    "is_english": "Description language detected as English (0/1)",
    "created_by_ai": "Video marked as AI-generated (0/1)",
    "publish_hour": "Publish hour (timezone unknown, likely UTC)",
    "publish_dow": "Publish day of week (0 = Monday)",
    "is_weekend": "Published on Saturday or Sunday (0/1)",
    "hour_sin": "Publish hour, cyclic encoding (sin)",
    "hour_cos": "Publish hour, cyclic encoding (cos)",
    "dow_sin": "Publish day of week, cyclic encoding (sin)",
    "dow_cos": "Publish day of week, cyclic encoding (cos)",
    "log_hours_since_prev": "Hours since the creator's previous video, log1p",
    "is_first_in_window": "First video of the creator in the dataset window (0/1)",
    "videos_prev_1d": "Creator's videos in the 24 hours before this one",
    "videos_prev_7d": "Creator's videos in the 7 days before this one",
    "followers_log": "Followers at publish time, log1p",
    "following_log": "Accounts the creator follows at publish time, log1p",
    "total_likes_log": "Creator's total likes at publish time, log1p",
    "creator_videos_log": "Creator's total video count at publish time, log1p",
    "likes_per_follower_log": "Total likes per follower at publish time, log scale",
    "follower_growth_7d": "Follower growth over the 7 days before publishing, log scale",
    "has_desc": "Video has a description (0/1)",
    "desc_len_log": "Description length in characters, log1p",
    "desc_n_words": "Words in the description",
    "desc_n_emojis": "Emojis in the description",
    "desc_n_mentions": "@-mentions in the description",
    "desc_has_question": "Description contains a question mark (0/1)",
    "n_hashtags": "Number of hashtags",
    "has_sticker_text": "Video has on-screen text (0/1)",
    "sticker_len_log": "On-screen text length in characters, log1p",
    "has_transcript": "Video has a speech transcript (0/1)",
    "anger": "Anger score of the content",
    "joy": "Joy score of the content",
    "surprise": "Surprise score of the content",
    "sadness": "Sadness score of the content",
    "disgust": "Disgust score of the content",
    "fear": "Fear score of the content",
    "emotion_missing": "Emotion scores were missing and filled (0/1)",
    "emotion_entropy": "How mixed the emotional tone is (entropy)",
    "emotion_max": "Strength of the dominant emotion",
    "original_sound": "Creator's own original sound (0/1)",
    "sound_prev_uses_log": "Earlier uses of this sound in the dataset, log1p",
    "sound_prev_authors_log": "Earlier distinct creators using this sound, log1p",
    "sound_prev_7d_log": "Uses of this sound in the previous 7 days, log1p",
    "creator_reused_sound": "Creator has used this sound before (0/1)",
}


@dataclass
class FeatureSelection:
    """Result of :func:`select_features`."""

    features: list[str]
    groups: dict[str, list[str]]
    zero_variance: list[str]
    redundant: list[str]
    pairs: pd.Series


def describe_feature(name: str) -> str:
    """Human-readable description, including generated hashtag and one-hot columns."""
    if name in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[name]
    if name.endswith("_vs_creator"):
        base = name[: -len("_vs_creator")]
        return f"{describe_feature(base)}, relative to the creator's usual value"
    if name.startswith("ht_"):
        return f"Video uses hashtag #{name[3:]} (0/1)"
    for cat in CATEGORICAL:
        if name.startswith(f"{cat}_"):
            return f"{cat} is {name[len(cat) + 1:]} (0/1)"
    return name


def to_datetime_any(s: pd.Series) -> pd.Series:
    """Parse timestamps given either as date strings or as unix seconds/milliseconds."""
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.9:
        unit = "ms" if num.median() > 1e11 else "s"
        return pd.to_datetime(num, unit=unit, errors="coerce")
    return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_convert(None)


def _to_seconds(dt: pd.Series) -> np.ndarray:
    return (dt - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy().astype("int64")


def count_prev_in_window(
    frame: pd.DataFrame, group_col: str, time_col: str, days: int
) -> pd.Series:
    """For each row, count rows of the same group published in the ``days`` before it."""
    codes = frame[group_col].astype("category").cat.codes.to_numpy().astype("int64")
    key = codes * 10**10 + _to_seconds(frame[time_col])
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    left = np.searchsorted(sorted_key, sorted_key - days * 86_400, side="left")
    counts = np.empty(len(key), dtype="int64")
    counts[order] = np.arange(len(key)) - left
    return pd.Series(counts, index=frame.index)


def rare_to_other(s: pd.Series, train_mask: pd.Series, min_share: float = 0.01) -> pd.Series:
    """Replace categories rarer than ``min_share`` on train with ``"other"``."""
    share = s[train_mask].value_counts(normalize=True)
    keep = share[share >= min_share].index
    return s.where(s.isin(keep), "other")


def hashtag_names(cell) -> list[str]:
    """Lowercase hashtag names from one cell of the ``hashtags`` column."""
    if cell is None or isinstance(cell, float):
        return []
    if isinstance(cell, str):
        return [t.lower() for t in re.findall(r"#?(\w+)", cell)]
    names = []
    for item in cell:
        name = item.get("hashtag_name") if isinstance(item, dict) else item
        if isinstance(name, str) and name.strip():
            names.append(name.strip().lower())
    return names


def _train_mask(df: pd.DataFrame) -> pd.Series:
    return df["split"].eq("train")


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add parsed ``create_dt`` / ``create_day`` and sort by creator and publish time."""
    out = df.copy()
    out["create_dt"] = to_datetime_any(out["create_time"])
    out["create_day"] = pd.to_datetime(out["create_date"], errors="coerce")
    return out.sort_values(["author_id", "create_dt"]).reset_index(drop=True)


def add_format_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    out["duration_log"] = np.log1p(out["duration"].clip(lower=0))
    out["is_english"] = out["is_english"].fillna(0).astype("int8")
    out["created_by_ai"] = out["created_by_ai"].fillna(0).astype("int8")
    return out, ["duration_log", "is_english", "created_by_ai"]


def add_timing_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    out["publish_hour"] = out["create_dt"].dt.hour
    out["publish_dow"] = out["create_dt"].dt.dayofweek
    out["is_weekend"] = (out["publish_dow"] >= 5).astype("int8")
    out["hour_sin"] = np.sin(2 * np.pi * out["publish_hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["publish_hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["publish_dow"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["publish_dow"] / 7)
    cols = ["publish_hour", "publish_dow", "is_weekend", "hour_sin", "hour_cos", "dow_sin"]
    return out, cols + ["dow_cos"]


def add_posting_history_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    ordered = out.sort_values(["author_id", "create_dt"])
    prev_dt = ordered.groupby("author_id")["create_dt"].shift(1).reindex(out.index)
    hours = (out["create_dt"] - prev_dt).dt.total_seconds() / 3600
    out["log_hours_since_prev"] = np.log1p(hours.clip(lower=0))
    out["is_first_in_window"] = hours.isna().astype("int8")
    out["videos_prev_1d"] = count_prev_in_window(out, "author_id", "create_dt", 1)
    out["videos_prev_7d"] = count_prev_in_window(out, "author_id", "create_dt", 7)
    return out, ["log_hours_since_prev", "is_first_in_window", "videos_prev_1d", "videos_prev_7d"]


def _asof_snapshot(
    keys: pd.DataFrame, snapshots: pd.DataFrame, lag_days: int, strict: bool
) -> pd.DataFrame:
    left = (
        keys.assign(lookup_day=keys["create_day"] - pd.Timedelta(days=lag_days))
        .dropna(subset=["lookup_day"])
        .sort_values("lookup_day")
    )
    merged = pd.merge_asof(
        left,
        snapshots,
        left_on="lookup_day",
        right_on="snap_day",
        by="author_id",
        direction="backward",
        allow_exact_matches=not strict,
    )
    return merged.set_index("video_id")[CREATOR_DAILY_COLS]


def add_creator_context_features(
    df: pd.DataFrame, creator_daily: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """Creator account state from the last snapshot strictly before the publish day.

    Falls back to the same-day snapshot only when no earlier one exists.
    """
    out = df.copy()
    snaps = creator_daily.copy()
    snaps["snap_day"] = pd.to_datetime(snaps["date"], errors="coerce")
    snaps = snaps.dropna(subset=["snap_day"]).sort_values("snap_day")
    snaps = snaps[["author_id", "snap_day"] + CREATOR_DAILY_COLS]

    keys = out[["video_id", "author_id", "create_day"]]
    snap = _asof_snapshot(keys, snaps, 0, strict=True).combine_first(
        _asof_snapshot(keys, snaps, 0, strict=False)
    )
    snap_7d = _asof_snapshot(keys, snaps, 7, strict=False)

    at_publish = {c: out["video_id"].map(snap[c]).clip(lower=0) for c in CREATOR_DAILY_COLS}
    followers_7d = out["video_id"].map(snap_7d["follower_count"]).clip(lower=0)

    out["followers_log"] = np.log1p(at_publish["follower_count"])
    out["following_log"] = np.log1p(at_publish["following_count"])
    out["total_likes_log"] = np.log1p(at_publish["total_favorited"])
    out["creator_videos_log"] = np.log1p(at_publish["video_count"])
    out["likes_per_follower_log"] = out["total_likes_log"] - out["followers_log"]
    out["follower_growth_7d"] = out["followers_log"] - np.log1p(followers_7d)
    cols = [
        "followers_log",
        "following_log",
        "total_likes_log",
        "creator_videos_log",
        "likes_per_follower_log",
        "follower_growth_7d",
    ]
    return out, cols


def add_description_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Description, hashtag count and on-screen text features.

    Also adds the helper columns ``hashtag_list`` / ``hashtag_set`` (not features).
    """
    out = df.copy()
    desc = out["desc"].fillna("").astype(str)
    sticker = out["sticker_text"].fillna("").astype(str)

    out["has_desc"] = desc.str.strip().ne("").astype("int8")
    out["desc_len_log"] = np.log1p(desc.str.len())
    out["desc_n_words"] = desc.str.split().str.len().fillna(0)
    out["desc_n_emojis"] = desc.str.count(EMOJI_PATTERN)
    out["desc_n_mentions"] = desc.str.count(r"@\w+")
    out["desc_has_question"] = desc.str.contains("?", regex=False).astype("int8")

    out["hashtag_list"] = out["hashtags"].apply(hashtag_names)
    out["hashtag_set"] = out["hashtag_list"].apply(set)
    out["n_hashtags"] = out["hashtag_list"].str.len()

    out["has_sticker_text"] = out["sticker_text"].notna().astype("int8")
    out["sticker_len_log"] = np.log1p(sticker.str.len())
    cols = [
        "has_desc",
        "desc_len_log",
        "desc_n_words",
        "desc_n_emojis",
        "desc_n_mentions",
        "desc_has_question",
        "n_hashtags",
        "has_sticker_text",
        "sticker_len_log",
    ]
    return out, cols


def add_transcript_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Only ``has_transcript``.

    The precomputed text statistics (``word_count``, ``speaking_rate`` and others) are
    not used: their missingness depends on the publish month (fully missing from
    October 2024), so they encode the data-collection batch, not the content.
    """
    out = df.copy()
    out["has_transcript"] = out["transcript"].notna().astype("int8")
    return out, ["has_transcript"]


def add_emotion_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    train = _train_mask(out)
    out["emotion_missing"] = out[EMOTIONS].isna().any(axis=1).astype("int8")
    out[EMOTIONS] = out[EMOTIONS].fillna(out.loc[train, EMOTIONS].mean())

    probs = out[EMOTIONS].clip(lower=1e-9)
    probs = probs.div(probs.sum(axis=1), axis=0)
    out["emotion_entropy"] = -(probs * np.log(probs)).sum(axis=1)
    out["emotion_max"] = out[EMOTIONS].max(axis=1)
    out["dominant_emotion"] = out[EMOTIONS].idxmax(axis=1)
    return out, EMOTIONS + ["emotion_missing", "emotion_entropy", "emotion_max"]


def sound_history(all_videos: pd.DataFrame) -> pd.DataFrame:
    """Past-only usage statistics of each video's sound, over the full videos table."""
    mus = all_videos[["video_id", "author_id", "music_id", "create_time"]].copy()
    mus["create_dt"] = to_datetime_any(mus["create_time"])
    mus = mus.dropna(subset=["music_id", "create_dt"])
    mus = mus.sort_values(["music_id", "create_dt"]).reset_index(drop=True)

    first_use = (~mus.duplicated(["music_id", "author_id"])).astype(int)
    mus["sound_prev_uses"] = mus.groupby("music_id").cumcount()
    mus["sound_prev_authors"] = first_use.groupby(mus["music_id"]).cumsum() - first_use
    mus["creator_reused_sound"] = (1 - first_use).astype("int8")
    mus["sound_prev_7d"] = count_prev_in_window(mus, "music_id", "create_dt", 7)
    cols = ["sound_prev_uses", "sound_prev_authors", "creator_reused_sound", "sound_prev_7d"]
    return mus.set_index("video_id")[cols]


def add_music_features(
    df: pd.DataFrame, all_videos: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Sound features; popularity is counted over ``all_videos`` (default: ``df``)."""
    out = df.copy()
    owner = out["music_owner_id"].fillna("").astype(str)
    title = out["music_title"].fillna("").astype(str).str.lower()
    out["music_source"] = out["music_selected_from"].fillna("unknown").astype(str)
    out["original_sound"] = (
        owner.eq(out["author_id"].astype(str)) | title.str.contains(ORIGINAL_SOUND_PATTERN)
    ).astype("int8")

    history = sound_history(df if all_videos is None else all_videos)
    for c in ["sound_prev_uses", "sound_prev_authors", "sound_prev_7d"]:
        out[f"{c}_log"] = np.log1p(out["video_id"].map(history[c]).fillna(0))
    reused = out["video_id"].map(history["creator_reused_sound"])
    out["creator_reused_sound"] = reused.fillna(0).astype("int8")
    cols = [
        "original_sound",
        "sound_prev_uses_log",
        "sound_prev_authors_log",
        "sound_prev_7d_log",
        "creator_reused_sound",
    ]
    return out, cols


def top_hashtags(df: pd.DataFrame, top_n: int = 25) -> pd.Series:
    """Hashtags used by the most distinct creators on train."""
    train_tags = (
        df.loc[_train_mask(df), ["author_id", "hashtag_list"]]
        .explode("hashtag_list")
        .dropna()
        .drop_duplicates()
    )
    return train_tags["hashtag_list"].value_counts().head(top_n)


def add_hashtag_features(
    df: pd.DataFrame, top_n: int = 25
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Multi-hot ``ht_<tag>`` columns for the top hashtags (by distinct train creators)."""
    out = df.copy()
    mapping: dict[str, str] = {}
    for tag in top_hashtags(out, top_n).index:
        col = "ht_" + re.sub(r"\W+", "_", tag).strip("_")
        while col in mapping.values() or col in out.columns:
            col += "_"
        mapping[tag] = col
        out[col] = out["hashtag_set"].apply(lambda s, t=tag: t in s).astype("int8")
    return out, list(mapping.values()), mapping


def impute_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Fill NaN with the creator's train median, falling back to the global train median."""
    out = df.copy()
    train = _train_mask(out)
    creator_med = out.loc[train].groupby("author_id")[cols].median()
    fill = (
        creator_med.reindex(out["author_id"])
        .fillna(out.loc[train, cols].median())
        .set_axis(out.index)
    )
    out[cols] = out[cols].fillna(fill)
    return out


def add_vs_creator_features(
    df: pd.DataFrame, base_cols: list[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """``<feature>_vs_creator`` = value minus the creator's train median of that feature."""
    out = df.copy()
    base = VS_CREATOR_BASE if base_cols is None else base_cols
    train = _train_mask(out)
    ref = (
        out.loc[train]
        .groupby("author_id")[base]
        .median()
        .reindex(out["author_id"])
        .fillna(out.loc[train, base].median())
        .set_axis(out.index)
    )
    names = []
    for c in base:
        out[f"{c}_vs_creator"] = out[c] - ref[c]
        names.append(f"{c}_vs_creator")
    return out, names


def add_categorical_features(
    df: pd.DataFrame, cols: list[str] | None = None, min_share: float = 0.01
) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, list[str]]]:
    """One-hot encode categoricals; levels rarer than ``min_share`` on train -> ``other``."""
    out = df.copy()
    cats = CATEGORICAL if cols is None else cols
    train = _train_mask(out)
    groups: dict[str, list[str]] = {}
    levels: dict[str, list[str]] = {}
    for c in cats:
        out[c] = rare_to_other(out[c].fillna("unknown").astype(str), train, min_share)
        levels[c] = sorted(out.loc[train, c].unique().tolist())
        categories = levels[c] + ([] if "other" in levels[c] else ["other"])
        dummies = pd.get_dummies(
            pd.Categorical(out[c], categories=categories), prefix=c, dtype="int8"
        )
        dummies.columns = [re.sub(r"\W+", "_", col).strip("_") for col in dummies.columns]
        dummies.index = out.index
        out = pd.concat([out, dummies], axis=1)
        groups[f"cat_{c}"] = dummies.columns.tolist()
    return out, groups, levels


def select_features(
    df: pd.DataFrame,
    groups: dict[str, list[str]],
    threshold: float = 0.95,
    keep_priority: Collection[str] = (),
) -> FeatureSelection:
    """Drop zero-variance features and one feature of each near-duplicate pair.

    A pair is near-duplicate when |Spearman rho| on train exceeds ``threshold``. The
    later feature of the pair is dropped, unless it is listed in ``keep_priority``.
    """
    train = _train_mask(df)
    all_features = [c for cols in groups.values() for c in cols]
    zero_var = [c for c in all_features if df.loc[train, c].nunique() <= 1]
    candidates = [c for c in all_features if c not in zero_var]

    corr = df.loc[train, candidates].corr(method="spearman").abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack()
    pairs = pairs[pairs > threshold].sort_values(ascending=False).rename("abs_spearman")

    redundant: list[str] = []
    for a, b in pairs.index:
        if a in redundant or b in redundant:
            continue
        redundant.append(a if b in keep_priority and a not in keep_priority else b)

    features = [c for c in candidates if c not in redundant]
    kept = {g: [c for c in cols if c in features] for g, cols in groups.items()}
    kept = {g: cols for g, cols in kept.items() if cols}
    return FeatureSelection(features, kept, zero_var, redundant, pairs)


def build_feature_table(
    df: pd.DataFrame,
    creator_daily: pd.DataFrame,
    all_videos: pd.DataFrame | None = None,
    top_n_hashtags: int = 25,
    rare_min_share: float = 0.01,
    redundancy_threshold: float = 0.95,
    keep_priority: Collection[str] = ("emotion_max",),
) -> tuple[pd.DataFrame, FeatureSelection, dict]:
    """Run every feature step on a prepared table (needs ``split``; targets optional).

    Returns the full table, the feature selection and extra metadata
    (top hashtags and categorical levels) for ``feature_spec.json``.
    """
    groups: dict[str, list[str]] = {}
    out = add_time_columns(df)
    out, groups["format"] = add_format_features(out)
    out, groups["timing"] = add_timing_features(out)
    out, groups["posting_history"] = add_posting_history_features(out)
    out, groups["creator_context"] = add_creator_context_features(out, creator_daily)
    out, groups["description"] = add_description_features(out)
    out, groups["transcript"] = add_transcript_features(out)
    out, groups["emotions"] = add_emotion_features(out)
    out, groups["music"] = add_music_features(out, all_videos)
    out, groups["hashtags"], hashtag_map = add_hashtag_features(out, top_n_hashtags)
    out = impute_numeric(out, [c for cols in groups.values() for c in cols])
    out, groups["vs_creator"] = add_vs_creator_features(out)
    out, cat_groups, levels = add_categorical_features(out, min_share=rare_min_share)
    groups.update(cat_groups)
    selection = select_features(out, groups, redundancy_threshold, keep_priority)
    extra = {"top_hashtags": hashtag_map, "categorical_levels": levels}
    return out, selection, extra


def build_features(
    videos: pd.DataFrame,
    creator_daily: pd.DataFrame,
    train_video_ids: Collection[str] | None = None,
) -> pd.DataFrame:
    """Feature matrix with ``video_id`` + selected features (used by ``cli.py``).

    ``videos`` must already be filtered. If ``train_video_ids`` is None, all videos
    are treated as train.
    """
    df = videos.copy()
    train_ids = set(df["video_id"]) if train_video_ids is None else set(train_video_ids)
    df["split"] = np.where(df["video_id"].isin(train_ids), "train", "test")
    table, selection, _ = build_feature_table(df, creator_daily)
    return table[["video_id"] + selection.features].reset_index(drop=True)


def save_feature_table(
    df: pd.DataFrame,
    selection: FeatureSelection,
    processed_dir: str | Path,
    spec_extra: dict | None = None,
) -> dict:
    """Write ``features.parquet``, ``dataset_index.parquet`` and ``feature_spec.json``.

    ``features.parquet`` holds ``video_id`` + features only; ``dataset_index.parquet``
    holds ids, split and targets. Returns the spec dict.
    """
    out_dir = Path(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = [c for c in META_COLUMNS if c in df.columns]
    df[["video_id"] + selection.features].to_parquet(out_dir / "features.parquet", index=False)
    df[meta].to_parquet(out_dir / "dataset_index.parquet", index=False)
    spec = {
        "features": selection.features,
        "groups": selection.groups,
        "dropped_zero_variance": selection.zero_variance,
        "dropped_redundant": selection.redundant,
        "n_videos": int(len(df)),
        "n_creators": int(df["author_id"].nunique()),
        **(spec_extra or {}),
    }
    with open(out_dir / "feature_spec.json", "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)
    return spec


def load_feature_table(processed_dir: str | Path) -> tuple[pd.DataFrame, dict]:
    """Load ids, split, targets and features as one table, plus the feature spec."""
    in_dir = Path(processed_dir)
    with open(in_dir / "feature_spec.json", encoding="utf-8") as f:
        spec = json.load(f)
    index = pd.read_parquet(in_dir / "dataset_index.parquet")
    features = pd.read_parquet(in_dir / "features.parquet")
    return index.merge(features, on="video_id", how="inner"), spec


def follower_asof_join(videos: pd.DataFrame, creator_daily: pd.DataFrame) -> pd.DataFrame:
    """Add ``follower_count_at_publish`` (last snapshot with ``date <= create_date``).

    Kept for notebook 02; the feature pipeline uses the stricter
    :func:`add_creator_context_features`.
    """
    vids = videos[["video_id", "author_id", "create_date"]].copy()
    vids["_create_dt"] = pd.to_datetime(vids["create_date"], errors="coerce")
    vids = vids.dropna(subset=["_create_dt"]).sort_values("_create_dt")

    crd = creator_daily[["author_id", "date", "follower_count"]].copy()
    crd["_date_dt"] = pd.to_datetime(crd["date"], errors="coerce")
    crd = crd.dropna(subset=["_date_dt"]).sort_values("_date_dt")

    merged = pd.merge_asof(
        vids,
        crd[["author_id", "_date_dt", "follower_count"]],
        left_on="_create_dt",
        right_on="_date_dt",
        by="author_id",
        direction="backward",
    )
    fc = merged[["video_id", "follower_count"]].rename(
        columns={"follower_count": "follower_count_at_publish"}
    )
    return videos.merge(fc, on="video_id", how="left")
