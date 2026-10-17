"""The same seed must give the same split; that's what makes results reproducible."""

import pandas as pd

from creator_analytics.data import train_test_split_by_creator


def _videos() -> pd.DataFrame:
    authors = ["a"] * 20 + ["b"] * 30 + ["c"] * 50
    return pd.DataFrame({"author_id": authors, "views": range(len(authors))})


def test_same_seed_gives_same_split():
    train_1, test_1 = train_test_split_by_creator(_videos(), "author_id", test_size=0.2, seed=42)
    train_2, test_2 = train_test_split_by_creator(_videos(), "author_id", test_size=0.2, seed=42)
    pd.testing.assert_frame_equal(train_1, train_2)
    pd.testing.assert_frame_equal(test_1, test_2)


def test_different_seed_gives_different_split():
    _, test_1 = train_test_split_by_creator(_videos(), "author_id", test_size=0.2, seed=42)
    _, test_2 = train_test_split_by_creator(_videos(), "author_id", test_size=0.2, seed=7)
    assert set(test_1["views"]) != set(test_2["views"])
