"""Tests for models.py and evaluate.py.

Key invariants:
- search_* functions have no X_test/y_test parameters (test-isolation by signature).
- sum(per_creator["n_test"]) == len(y_test) (per-creator covers all global rows).
- compute_metrics returns all required keys.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from creator_analytics.evaluate import compute_metrics, compute_per_creator_metrics
from creator_analytics.models import (
    TrainedModel,
    make_cv_splits,
    prepare_X,
    search_random_forest,
    search_xgboost,
    trivial_creator_mean,
    trivial_zero_rmse,
)


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _make_tiny_train(n: int = 60, n_creators: int = 6, seed: int = 0):
    """Minimal feature DataFrame + target + author_ids."""
    rng = np.random.RandomState(seed)
    author_ids = pd.Series(
        [f"creator_{i % n_creators}" for i in range(n)], name="author_id"
    )
    X = pd.DataFrame({
        "duration_log": rng.rand(n).astype("float64"),
        "desc_len": rng.randint(0, 200, n).astype("float64"),
        "ratio": pd.array(["9:16"] * (n // 2) + ["1:1"] * (n - n // 2), dtype=object),
        "topic": pd.array(["music"] * n, dtype=object),
        "music_selected_from": pd.array(["original"] * n, dtype=object),
        "dominant_emotion": pd.array(["joy"] * n, dtype=object),
    })
    y = pd.Series(rng.randn(n), name="target")
    return X, y, author_ids


# ------------------------------------------------------------------ #
# Signature tests: no test data in search_* signatures                 #
# ------------------------------------------------------------------ #

def test_search_random_forest_has_no_test_params():
    sig = inspect.signature(search_random_forest)
    params = set(sig.parameters)
    assert "X_test" not in params
    assert "y_test" not in params


def test_search_xgboost_has_no_test_params():
    sig = inspect.signature(search_xgboost)
    params = set(sig.parameters)
    assert "X_test" not in params
    assert "y_test" not in params


# ------------------------------------------------------------------ #
# prepare_X                                                            #
# ------------------------------------------------------------------ #

def test_prepare_x_drops_video_id():
    df = pd.DataFrame({"video_id": ["a", "b"], "duration_log": [1.0, 2.0]})
    X = prepare_X(df)
    assert "video_id" not in X.columns


def test_prepare_x_drops_is_ads():
    df = pd.DataFrame({"is_ads": [0, 1], "duration_log": [1.0, 2.0]})
    X = prepare_X(df)
    assert "is_ads" not in X.columns


def test_prepare_x_converts_nullable_int():
    df = pd.DataFrame({"publish_hour": pd.array([10, None, 14], dtype="Int8")})
    X = prepare_X(df)
    assert X["publish_hour"].dtype == np.float64
    assert np.isnan(X["publish_hour"].iloc[1])


# ------------------------------------------------------------------ #
# make_cv_splits                                                        #
# ------------------------------------------------------------------ #

def test_make_cv_splits_count():
    author_ids = pd.Series(["a"] * 20 + ["b"] * 20 + ["c"] * 20)
    splits = make_cv_splits(author_ids, n_splits=5, seed=42)
    assert len(splits) == 5


def test_make_cv_splits_cover_all_indices():
    n = 60
    author_ids = pd.Series([f"c{i % 6}" for i in range(n)])
    splits = make_cv_splits(author_ids, n_splits=5)
    all_val = set()
    for _, val_idx in splits:
        all_val.update(val_idx)
    assert all_val == set(range(n))


def test_make_cv_splits_no_overlap_within_fold():
    author_ids = pd.Series([f"c{i % 6}" for i in range(60)])
    for tr_idx, val_idx in make_cv_splits(author_ids):
        assert len(set(tr_idx) & set(val_idx)) == 0


# ------------------------------------------------------------------ #
# Trivial baselines                                                     #
# ------------------------------------------------------------------ #

def test_trivial_zero_rmse_all_zeros():
    y = pd.Series([0.0, 0.0, 0.0])
    assert trivial_zero_rmse(y) == pytest.approx(0.0)


def test_trivial_zero_rmse_nonzero():
    y = pd.Series([1.0, -1.0])
    # sqrt(mean([1, 1])) = 1
    assert trivial_zero_rmse(y) == pytest.approx(1.0)


def test_trivial_creator_mean_perfect():
    """If train mean equals test value for each creator, RMSE = 0."""
    y_train = pd.Series([2.0, 2.0, 3.0, 3.0])
    auth_train = pd.Series(["a", "a", "b", "b"])
    y_test = pd.Series([2.0, 3.0])
    auth_test = pd.Series(["a", "b"])
    rmse_val, preds = trivial_creator_mean(y_train, auth_train, y_test, auth_test)
    assert rmse_val == pytest.approx(0.0)
    assert preds[0] == pytest.approx(2.0)
    assert preds[1] == pytest.approx(3.0)


# ------------------------------------------------------------------ #
# compute_metrics                                                       #
# ------------------------------------------------------------------ #

def test_compute_metrics_returns_all_keys():
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    yp = pd.Series([1.1, 2.1, 2.9, 4.2])
    m = compute_metrics(y, yp)
    for key in ("rmse", "mae", "r2", "spearman", "n"):
        assert key in m, f"Missing key: {key}"


def test_compute_metrics_n():
    y = pd.Series([1.0, 2.0, 3.0])
    yp = pd.Series([1.0, 2.0, 3.0])
    assert compute_metrics(y, yp)["n"] == 3


def test_compute_metrics_perfect():
    y = pd.Series([1.0, 2.0, 3.0])
    m = compute_metrics(y, y)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)


# ------------------------------------------------------------------ #
# compute_per_creator_metrics — n_test invariant                       #
# ------------------------------------------------------------------ #

def test_per_creator_n_test_sums_to_global():
    """Sum of per-creator n_test must equal len(y_test)."""
    rng = np.random.RandomState(7)
    n = 50
    y_true = pd.Series(rng.randn(n))
    y_pred = pd.Series(rng.randn(n))
    author_ids = pd.Series([f"c{i % 5}" for i in range(n)])
    per_creator = compute_per_creator_metrics(y_true, y_pred, author_ids)
    assert per_creator["n_test"].sum() == n


def test_per_creator_covers_same_videos_as_global():
    """The videos used in per-creator rows are exactly the same as the global set."""
    rng = np.random.RandomState(9)
    n = 40
    y_true = pd.Series(rng.randn(n))
    y_pred = pd.Series(rng.randn(n))
    author_ids = pd.Series([f"c{i % 4}" for i in range(n)])
    per_creator = compute_per_creator_metrics(y_true, y_pred, author_ids)
    # RMSE computed globally should match if we could reconstruct — at minimum n counts match
    assert per_creator["n_test"].sum() == len(y_true)
    assert len(per_creator) == 4  # 4 unique creators


def test_per_creator_single_creator():
    y = pd.Series([1.0, 2.0, 3.0])
    yp = pd.Series([1.0, 2.0, 3.0])
    auth = pd.Series(["alice", "alice", "alice"])
    per = compute_per_creator_metrics(y, yp, auth)
    assert len(per) == 1
    assert per.iloc[0]["n_test"] == 3
    assert per.iloc[0]["rmse"] == pytest.approx(0.0)


# ------------------------------------------------------------------ #
# search_random_forest smoke test (tiny data, 2 iterations)            #
# ------------------------------------------------------------------ #

def test_search_random_forest_returns_trained_model():
    X, y, author_ids = _make_tiny_train(n=60, n_creators=6)
    model = search_random_forest(X, y, author_ids, n_splits=3, seed=42, n_iter=2)
    assert isinstance(model, TrainedModel)
    assert model.cv_rmse > 0
    assert model.model_name == "RandomForest"


def test_search_random_forest_predict_shape():
    X, y, author_ids = _make_tiny_train(n=60, n_creators=6)
    model = search_random_forest(X, y, author_ids, n_splits=3, seed=42, n_iter=2)
    preds = model.predict(X)
    assert preds.shape == (len(X),)


# ------------------------------------------------------------------ #
# search_xgboost smoke test (tiny data, 2 iterations)                  #
# ------------------------------------------------------------------ #

def test_search_xgboost_returns_trained_model():
    X, y, author_ids = _make_tiny_train(n=60, n_creators=6)
    model = search_xgboost(X, y, author_ids, n_splits=3, seed=42, n_iter=2)
    assert isinstance(model, TrainedModel)
    assert "XGBoost" in model.model_name


def test_search_xgboost_exclude_col():
    X, y, author_ids = _make_tiny_train(n=60, n_creators=6)
    X["follower_count_log"] = 1.0
    model = search_xgboost(X, y, author_ids, n_splits=3, seed=42, n_iter=2,
                           exclude_cols=["follower_count_log"])
    assert "follower_count_log" not in model.feature_names
    assert "no_follower" in model.model_name


def test_search_xgboost_predict_shape():
    X, y, author_ids = _make_tiny_train(n=60, n_creators=6)
    model = search_xgboost(X, y, author_ids, n_splits=3, seed=42, n_iter=2)
    preds = model.predict(X)
    assert preds.shape == (len(X),)


# ------------------------------------------------------------------ #
# TrainedModel save/load round-trip                                    #
# ------------------------------------------------------------------ #

def test_trained_model_save_load(tmp_path):
    X, y, author_ids = _make_tiny_train(n=60, n_creators=6)
    model = search_xgboost(X, y, author_ids, n_splits=3, seed=42, n_iter=2)
    path = tmp_path / "test_model.joblib"
    model.save(path)
    loaded = TrainedModel.load(path)
    np.testing.assert_allclose(model.predict(X), loaded.predict(X))
