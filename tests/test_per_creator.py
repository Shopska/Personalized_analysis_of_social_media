"""Tests for per-creator model training and prediction.

Key invariants:
- _train_one_creator has no X_test / y_test parameters.
- Each creator model is trained only on that creator's train videos.
- predict_per_creator returns predictions for exactly the same test videos
  as the global model (same author_ids).
- Sum of per-creator n_test equals the total test set size.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from creator_analytics.evaluate import predict_per_creator
from creator_analytics.models import (
    CreatorModel,
    _train_one_creator,
    drop_constant_features,
    search_per_creator_config,
    train_per_creator_models,
)


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _make_creator_data(n_creators: int = 4, n_per_creator: int = 20, seed: int = 0):
    """Synthetic feature + target + author_ids for per-creator tests.

    All features are numeric to match the new one-hot pipeline.
    """
    rng = np.random.RandomState(seed)
    total = n_creators * n_per_creator
    author_ids = pd.Series(
        [f"creator_{i // n_per_creator}" for i in range(total)], name="author_id"
    )
    X = pd.DataFrame({
        "duration_log":  rng.rand(total).astype("float64"),
        "desc_len":      rng.randint(0, 200, total).astype("float64"),
        "publish_hour":  rng.randint(0, 24, total).astype("float64"),
        "ratio_9_16":    rng.randint(0, 2, total).astype("float64"),
        "topic_music":   rng.randint(0, 2, total).astype("float64"),
        "dominant_joy":  rng.randint(0, 2, total).astype("float64"),
    })
    y = pd.Series(rng.randn(total), name="target")
    return X, y, author_ids


# ------------------------------------------------------------------ #
# Signature tests: no test data in training functions                  #
# ------------------------------------------------------------------ #

def test_train_one_creator_no_test_params():
    sig = inspect.signature(_train_one_creator)
    assert "X_test" not in sig.parameters
    assert "y_test" not in sig.parameters


def test_search_per_creator_config_no_test_params():
    sig = inspect.signature(search_per_creator_config)
    assert "X_test" not in sig.parameters
    assert "y_test" not in sig.parameters


def test_train_per_creator_models_no_test_params():
    sig = inspect.signature(train_per_creator_models)
    assert "X_test" not in sig.parameters
    assert "y_test" not in sig.parameters


# ------------------------------------------------------------------ #
# drop_constant_features                                               #
# ------------------------------------------------------------------ #

def test_drop_constant_constant_col():
    df = pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [1.0, 2.0, 3.0]})
    X_clean, dropped = drop_constant_features(df)
    assert "a" in dropped
    assert "b" not in dropped
    assert list(X_clean.columns) == ["b"]


def test_drop_constant_no_change():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    X_clean, dropped = drop_constant_features(df)
    assert dropped == []
    assert len(X_clean.columns) == 2


def test_drop_constant_all_nan_is_constant():
    df = pd.DataFrame({"a": [np.nan, np.nan, np.nan], "b": [1.0, 2.0, 3.0]})
    _, dropped = drop_constant_features(df)
    assert "a" in dropped


def test_drop_constant_single_unique_cat():
    df = pd.DataFrame({"cat": pd.array(["music", "music", "music"], dtype=object),
                       "num": [1.0, 2.0, 3.0]})
    _, dropped = drop_constant_features(df)
    assert "cat" in dropped


# ------------------------------------------------------------------ #
# _train_one_creator — uses only its own data                          #
# ------------------------------------------------------------------ #

def test_train_one_creator_returns_two_models():
    X, y, author_ids = _make_creator_data(n_creators=1, n_per_creator=30)
    rf_cfg = {"n_estimators": 20, "max_depth": 2, "min_samples_leaf": 3, "max_features": 0.5}
    xgb_cfg = {"n_estimators": 20, "max_depth": 2, "min_child_weight": 3,
                "reg_alpha": 0.5, "reg_lambda": 1.5, "subsample": 0.8,
                "colsample_bytree": 0.6, "learning_rate": 0.1}
    rf_m, xgb_m = _train_one_creator("c0", X, y, rf_cfg, xgb_cfg, seed=42)
    assert isinstance(rf_m, CreatorModel)
    assert isinstance(xgb_m, CreatorModel)
    assert rf_m.model_name == "RF"
    assert xgb_m.model_name == "XGB"
    assert rf_m.n_train == 30
    assert xgb_m.n_train == 30


def test_train_one_creator_feature_names_subset():
    """CreatorModel.feature_names must be a subset of X columns."""
    X, y, _ = _make_creator_data(n_creators=1, n_per_creator=30)
    rf_cfg = {"n_estimators": 10, "max_depth": 2, "min_samples_leaf": 3, "max_features": 0.5}
    xgb_cfg = {"n_estimators": 10, "max_depth": 2, "min_child_weight": 3,
                "reg_alpha": 0.5, "reg_lambda": 1.0, "subsample": 0.8,
                "colsample_bytree": 0.6, "learning_rate": 0.1}
    rf_m, xgb_m = _train_one_creator("c0", X, y, rf_cfg, xgb_cfg, seed=0)
    assert set(rf_m.feature_names).issubset(set(X.columns))
    assert set(xgb_m.feature_names).issubset(set(X.columns))


def test_train_one_creator_drops_constants():
    """Constant column in creator's data must be dropped."""
    X, y, _ = _make_creator_data(n_creators=1, n_per_creator=30)
    X["all_same"] = 5.0  # constant feature for this creator
    rf_cfg = {"n_estimators": 10, "max_depth": 2, "min_samples_leaf": 3, "max_features": 0.5}
    xgb_cfg = {"n_estimators": 10, "max_depth": 2, "min_child_weight": 3,
                "reg_alpha": 0.5, "reg_lambda": 1.0, "subsample": 0.8,
                "colsample_bytree": 0.6, "learning_rate": 0.1}
    rf_m, _ = _train_one_creator("c0", X, y, rf_cfg, xgb_cfg, seed=0)
    assert "all_same" not in rf_m.feature_names


# ------------------------------------------------------------------ #
# train_per_creator_models — correct number of models                  #
# ------------------------------------------------------------------ #

def test_train_per_creator_models_count():
    X, y, author_ids = _make_creator_data(n_creators=3, n_per_creator=20)
    rf_cfg = {"n_estimators": 10, "max_depth": 2, "min_samples_leaf": 3, "max_features": 0.5}
    xgb_cfg = {"n_estimators": 10, "max_depth": 2, "min_child_weight": 3,
                "reg_alpha": 0.5, "reg_lambda": 1.0, "subsample": 0.8,
                "colsample_bytree": 0.6, "learning_rate": 0.1}
    rf_models, xgb_models = train_per_creator_models(
        X, y, author_ids, rf_cfg, xgb_cfg, n_jobs=1, seed=42
    )
    assert len(rf_models) == 3
    assert len(xgb_models) == 3
    assert set(rf_models.keys()) == {"creator_0", "creator_1", "creator_2"}


# ------------------------------------------------------------------ #
# predict_per_creator — predictions cover same videos as global        #
# ------------------------------------------------------------------ #

def test_predict_per_creator_all_covered():
    """When all creators are in creator_models, no NaN in predictions."""
    X, y, author_ids = _make_creator_data(n_creators=3, n_per_creator=20)
    rf_cfg = {"n_estimators": 10, "max_depth": 2, "min_samples_leaf": 3, "max_features": 0.5}
    xgb_cfg = {"n_estimators": 10, "max_depth": 2, "min_child_weight": 3,
                "reg_alpha": 0.5, "reg_lambda": 1.0, "subsample": 0.8,
                "colsample_bytree": 0.6, "learning_rate": 0.1}
    rf_models, _ = train_per_creator_models(
        X, y, author_ids, rf_cfg, xgb_cfg, n_jobs=1, seed=0
    )
    # Use same X and author_ids as "test" — all creators present in models
    y_pred = predict_per_creator(X, author_ids, rf_models)
    assert not np.any(np.isnan(y_pred)), "Expected no NaN when all creators are in models"


def test_predict_per_creator_shape():
    X, y, author_ids = _make_creator_data(n_creators=3, n_per_creator=20)
    rf_cfg = {"n_estimators": 10, "max_depth": 2, "min_samples_leaf": 3, "max_features": 0.5}
    xgb_cfg = {"n_estimators": 10, "max_depth": 2, "min_child_weight": 3,
                "reg_alpha": 0.5, "reg_lambda": 1.0, "subsample": 0.8,
                "colsample_bytree": 0.6, "learning_rate": 0.1}
    rf_models, _ = train_per_creator_models(
        X, y, author_ids, rf_cfg, xgb_cfg, n_jobs=1, seed=0
    )
    y_pred = predict_per_creator(X, author_ids, rf_models)
    assert y_pred.shape == (len(X),)


def test_predict_per_creator_missing_creator_is_nan():
    """If a creator is absent from models, their predictions should be NaN."""
    X, y, author_ids = _make_creator_data(n_creators=3, n_per_creator=20)
    rf_cfg = {"n_estimators": 10, "max_depth": 2, "min_samples_leaf": 3, "max_features": 0.5}
    xgb_cfg = {"n_estimators": 10, "max_depth": 2, "min_child_weight": 3,
                "reg_alpha": 0.5, "reg_lambda": 1.0, "subsample": 0.8,
                "colsample_bytree": 0.6, "learning_rate": 0.1}
    rf_models, _ = train_per_creator_models(
        X, y, author_ids, rf_cfg, xgb_cfg, n_jobs=1, seed=0
    )
    # Remove one creator from models
    del rf_models["creator_2"]
    y_pred = predict_per_creator(X, author_ids, rf_models)
    mask_2 = author_ids == "creator_2"
    assert np.all(np.isnan(y_pred[mask_2.values]))
    assert not np.any(np.isnan(y_pred[~mask_2.values]))


# ------------------------------------------------------------------ #
# CreatorModel predict and save/load                                   #
# ------------------------------------------------------------------ #

def test_creator_model_save_load(tmp_path):
    X, y, _ = _make_creator_data(n_creators=1, n_per_creator=30)
    rf_cfg = {"n_estimators": 10, "max_depth": 2, "min_samples_leaf": 3, "max_features": 0.5}
    xgb_cfg = {"n_estimators": 10, "max_depth": 2, "min_child_weight": 3,
                "reg_alpha": 0.5, "reg_lambda": 1.0, "subsample": 0.8,
                "colsample_bytree": 0.6, "learning_rate": 0.1}
    rf_m, _ = _train_one_creator("c0", X, y, rf_cfg, xgb_cfg, seed=0)
    path = tmp_path / "c0.joblib"
    rf_m.save(path)
    loaded = CreatorModel.load(path)
    np.testing.assert_allclose(rf_m.predict(X), loaded.predict(X))
