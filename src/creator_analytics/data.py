"""Loading, filtering, splitting, and persisting the TikTok engagement dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def raw_table_path(raw_dir: str | Path, table: str) -> Path:
    """Return the local Parquet path for a named table."""
    return Path(raw_dir) / f"{table}.parquet"


def materialize_raw_table(
    raw_dir: str | Path,
    table: str,
    hf_repo: str,
    hf_split: str = "train",
) -> None:
    """Download one HF dataset subset and write it directly to Parquet (no pandas).

    Uses ``Dataset.to_parquet`` so the full table never materialises as a pandas
    DataFrame — important for engagement_daily (~6 M rows).

    Parameters
    ----------
    raw_dir : str or Path
        Directory where ``<table>.parquet`` will be written.
    table : str
        HF subset/config name, e.g. ``"videos"``, ``"creator_daily"``,
        ``"engagement_daily"``.
    hf_repo : str
        Dataset id on the Hub, e.g. ``lingbow/tiktok-video-engagement-200k``.
    hf_split : str
        Which split to download, usually ``"train"``.
    """
    from datasets import load_dataset  # imported here: only this function needs it

    path = raw_table_path(raw_dir, table)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(hf_repo, table, split=hf_split)
    dataset.to_parquet(str(path))


def load_raw_table(raw_dir: str | Path, table: str) -> pd.DataFrame:
    """Read a downloaded table from disk.

    Raises
    ------
    FileNotFoundError
        With a hint to run ``python scripts/download_data.py`` first.
    """
    path = raw_table_path(raw_dir, table)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist yet. Run `python scripts/download_data.py` first."
        )
    return pd.read_parquet(path)


def filter_creators(df: pd.DataFrame, author_column: str, min_videos: int) -> pd.DataFrame:
    """Keep only creators with at least ``min_videos`` videos.

    Why this matters: a per-creator model trained on 5 videos learns noise, not
    patterns. This threshold is the main knob of the proposal's plan B, so it lives in
    config/default.yaml, not hardcoded here.
    """
    counts = df[author_column].value_counts()
    keep = counts[counts >= min_videos].index
    return df[df[author_column].isin(keep)].reset_index(drop=True)


def creator_video_counts(df: pd.DataFrame, author_column: str) -> pd.Series:
    """Number of videos per creator, largest first. Handy for EDA."""
    return df[author_column].value_counts()


def split_features_target(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into feature columns ``X`` and the target series ``y``."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def train_test_split_by_creator(
    df: pd.DataFrame, author_column: str, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split so that every creator has videos in both train and test, in the same ratio.

    Why this matters: the project compares a per-creator model with one global model.
    That comparison is only fair if both are tested on exactly the same videos, and a
    per-creator model can only be tested on a creator it has seen in training.
    Stratifying by creator guarantees both.

    Requires every creator to have at least 2 videos — run ``filter_creators`` first.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[author_column],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def prepare_dataset(
    videos: pd.DataFrame,
    engagement: pd.DataFrame,
    author_column: str = "author_id",
    min_videos: int = 30,
    test_size: float = 0.2,
    seed: int = 42,
    target_day: int = 14,
) -> pd.DataFrame:
    """Clean, filter and split the videos table into the modelling set.

    Drops videos with a non-monotonic ``play_count`` (data-collection resets) and
    videos with no engagement record at exactly ``target_day``, keeps creators with at
    least ``min_videos`` videos, and adds a ``split`` column (``"train"``/``"test"``)
    stratified by creator.
    """
    from creator_analytics.targets import get_nonmonotonic_video_ids

    bad_ids = get_nonmonotonic_video_ids(engagement)
    day_ids = set(engagement.loc[engagement["days_since_post"] == target_day, "video_id"])
    keep = ~videos["video_id"].isin(bad_ids) & videos["video_id"].isin(day_ids)
    df = filter_creators(videos[keep], author_column, min_videos)

    train_df, _ = train_test_split_by_creator(df, author_column, test_size, seed)
    df = df.copy()
    is_train = df["video_id"].isin(set(train_df["video_id"]))
    df["split"] = is_train.map({True: "train", False: "test"})
    df.attrs["n_nonmonotonic_excluded"] = len(bad_ids)
    return df
