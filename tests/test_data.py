"""Tests for creator_analytics.data on a small synthetic dataset (no download needed)."""

import pandas as pd
import pytest

from creator_analytics.data import (
    filter_creators,
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
