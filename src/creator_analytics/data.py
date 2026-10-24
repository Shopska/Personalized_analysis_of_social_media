"""Loading, filtering, splitting, and persisting the TikTok engagement dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def materialize_raw_data(
    output_path: str | Path,
    hf_repo: str,
    hf_config: str | None = None,
    hf_split: str = "train",
) -> pd.DataFrame:
    """Download the dataset from Hugging Face and save it to ``output_path`` as Parquet.

    This is the only place the dataset is fetched from the internet: everything else in
    the project reads the local Parquet file. Parquet (not CSV) because the dataset has
    long text columns (``transcript``, ``gpt_summary``) that CSV handles badly, and
    210k rows are several times smaller and faster to read as Parquet.

    Parameters
    ----------
    output_path : str or Path
        Where to write the file, e.g. ``data/raw/videos.parquet``.
    hf_repo : str
        Dataset id on the Hub, e.g. ``lingbow/tiktok-video-engagement-200k``.
    hf_config : str, optional
        Subset name, if the dataset has several (check its page on the Hub).
    hf_split : str
        Which split to download, usually ``"train"``.
    """
    from datasets import load_dataset  # imported here: only this function needs it

    dataset = load_dataset(hf_repo, hf_config, split=hf_split)
    df = dataset.to_pandas()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return df


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Read the downloaded dataset back from disk.

    Raises
    ------
    FileNotFoundError
        With a hint to run ``python scripts/download_data.py`` first.
    """
    path = Path(path)
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
