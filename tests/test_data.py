"""Tests for creator_analytics.data on a small synthetic dataset (no download needed)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from creator_analytics.data import (
    filter_creators,
    load_raw_table,
    materialize_raw_table,
    raw_table_path,
    split_features_target,
    train_test_split_by_creator,
)


@pytest.fixture
def videos() -> pd.DataFrame:
    # Creators with 3, 20, 30 and 50 videos.
    authors = ["a"] * 3 + ["b"] * 20 + ["c"] * 30 + ["d"] * 50
    return pd.DataFrame({"author_id": authors, "views": range(len(authors))})


def test_filter_creators_drops_small_creators(videos):
    filtered = filter_creators(videos, "author_id", min_videos=10)
    assert set(filtered["author_id"]) == {"b", "c", "d"}
    assert len(filtered) == 100


def test_split_keeps_every_creator_in_train_and_test(videos):
    filtered = filter_creators(videos, "author_id", min_videos=10)
    train_df, test_df = train_test_split_by_creator(filtered, "author_id", test_size=0.2, seed=42)
    assert set(train_df["author_id"]) == set(test_df["author_id"]) == {"b", "c", "d"}


def test_split_has_no_overlap(videos):
    filtered = filter_creators(videos, "author_id", min_videos=10)
    train_df, test_df = train_test_split_by_creator(filtered, "author_id", test_size=0.2, seed=42)
    assert set(train_df["views"]).isdisjoint(test_df["views"])
    assert len(train_df) + len(test_df) == len(filtered)


def test_split_features_target(videos):
    X, y = split_features_target(videos, target_column="views")
    assert "views" not in X.columns
    assert len(X) == len(y) == len(videos)


# --- tests for three-table API (no network) ---


def test_raw_table_path(tmp_path):
    p = raw_table_path(tmp_path / "raw", "videos")
    assert p == tmp_path / "raw" / "videos.parquet"


def _make_fake_dataset(df: pd.DataFrame):
    """Return a mock HF Dataset whose to_parquet writes a real Parquet file."""

    def _to_parquet(path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

    from unittest.mock import MagicMock

    mock_ds = MagicMock()
    mock_ds.to_parquet.side_effect = _to_parquet
    return mock_ds


def test_materialize_raw_table_creates_file(tmp_path):
    fake_df = pd.DataFrame({"video_id": [1, 2, 3], "author_id": ["x", "y", "z"]})
    fake_ds = _make_fake_dataset(fake_df)

    with patch("datasets.load_dataset", return_value=fake_ds):
        materialize_raw_table(tmp_path / "raw", "videos", hf_repo="fake/repo", hf_split="train")

    out = tmp_path / "raw" / "videos.parquet"
    assert out.exists()
    fake_ds.to_parquet.assert_called_once_with(str(out))


def test_materialize_raw_table_passes_subset_name(tmp_path):
    """load_dataset must be called with the table name as the config/subset argument."""
    fake_ds = _make_fake_dataset(pd.DataFrame({"a": [1]}))

    with patch("datasets.load_dataset", return_value=fake_ds) as mock_ld:
        materialize_raw_table(tmp_path, "creator_daily", hf_repo="fake/repo", hf_split="train")

    mock_ld.assert_called_once_with("fake/repo", "creator_daily", split="train")


def test_load_raw_table_reads_file(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    df = pd.DataFrame({"x": [10, 20, 30]})
    df.to_parquet(raw_dir / "videos.parquet", index=False)

    result = load_raw_table(raw_dir, "videos")
    pd.testing.assert_frame_equal(result, df)


def test_load_raw_table_raises_if_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data.py"):
        load_raw_table(tmp_path / "raw", "videos")
