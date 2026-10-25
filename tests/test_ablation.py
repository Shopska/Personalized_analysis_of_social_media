"""Tests for creator_analytics.ablation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creator_analytics.ablation import (
    build_combinations,
    build_group_map,
    cv_global_rf,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_spec() -> dict:
    """Minimal feature_spec matching real group structure."""
    return {
        "groups": {
            "format": ["f1", "f2"],
            "timing": ["t1", "t2", "t3"],
            "creator_context": ["cc1", "cc2"],
            "vs_creator": ["f1_vs_creator", "t1_vs_creator"],
            "cat_ratio": ["r1"],
            "cat_topic": ["topic1", "topic2"],
        }
    }


def _make_data(n: int = 120, n_creators: int = 6, seed: int = 0):
    rng = np.random.RandomState(seed)
    auth = pd.Series([f"c{i % n_creators}" for i in range(n)], name="author_id")
    X = pd.DataFrame(
        {
            "f1": rng.randn(n),
            "f2": rng.randn(n),
            "t1": rng.randn(n),
            "t2": rng.randn(n),
            "t3": rng.randn(n),
            "cc1": rng.randn(n),
            "cc2": rng.randn(n),
            "f1_vs_creator": rng.randn(n),
            "t1_vs_creator": rng.randn(n),
            "r1": rng.randn(n),
            "topic1": rng.randn(n),
            "topic2": rng.randn(n),
        }
    )
    y = pd.Series(rng.randn(n), name="target_rel")
    return X, y, auth


# ── build_group_map ───────────────────────────────────────────────────────────


def test_build_group_map_merges_cat():
    spec = _make_spec()
    groups = build_group_map(spec)
    assert "cat_*" in groups
    assert "cat_ratio" not in groups
    assert "cat_topic" not in groups
    assert set(groups["cat_*"]) == {"r1", "topic1", "topic2"}


def test_build_group_map_preserves_others():
    spec = _make_spec()
    groups = build_group_map(spec)
    assert "format" in groups
    assert "vs_creator" in groups
    assert groups["format"] == ["f1", "f2"]


# ── build_combinations ────────────────────────────────────────────────────────


@pytest.fixture
def combo_inputs():
    spec = _make_spec()
    groups = build_group_map(spec)
    features_global = [
        "f1",
        "f2",
        "t1",
        "t2",
        "t3",
        "cc1",
        "cc2",
        "f1_vs_creator",
        "t1_vs_creator",
        "r1",
        "topic1",
        "topic2",
    ]
    return features_global, groups


def test_build_combinations_count(combo_inputs):
    fg, groups = combo_inputs
    combos = build_combinations(fg, groups)
    n_groups = len(groups)  # 5: format, timing, creator_context, vs_creator, cat_*
    assert len(combos) == n_groups * 2 + 5, f"Expected {n_groups*2+5} combos, got {len(combos)}"


def test_all_features_combo_present(combo_inputs):
    fg, groups = combo_inputs
    combos = build_combinations(fg, groups)
    names = [c["combination"] for c in combos]
    assert "all_features" in names


def test_loo_removes_group_features(combo_inputs):
    fg, groups = combo_inputs
    combos = build_combinations(fg, groups)
    loo_fmt = next(c for c in combos if c["combination"] == "LOO_format")
    for f in groups["format"]:
        assert f not in loo_fmt["features"]


def test_feature_lists_are_subsets(combo_inputs):
    fg, groups = combo_inputs
    fg_set = set(fg)
    combos = build_combinations(fg, groups)
    for c in combos:
        assert set(c["features"]).issubset(fg_set), c["combination"]


def test_combinations_have_required_keys(combo_inputs):
    fg, groups = combo_inputs
    combos = build_combinations(fg, groups)
    for c in combos:
        for key in ("combination", "type", "groups_label", "features"):
            assert key in c, f"Missing key {key!r} in combo {c.get('combination')!r}"


# ── cv_global_rf ──────────────────────────────────────────────────────────────

RF_PARAMS = {"n_estimators": 10, "max_depth": 3, "max_features": 0.5}


def test_cv_global_rf_returns_five_folds():
    X, y, auth = _make_data()
    feats = ["f1", "f2", "t1"]
    results = cv_global_rf(X, y, auth, feats, RF_PARAMS, n_splits=5, seed=42)
    assert len(results) == 5


def test_cv_global_rf_metric_keys():
    X, y, auth = _make_data()
    feats = ["f1", "t1"]
    results = cv_global_rf(X, y, auth, feats, RF_PARAMS, n_splits=3, seed=42)
    for fold in results:
        assert set(fold.keys()) == {"rmse", "mae", "spearman", "median_creator_spearman"}
        assert fold["rmse"] > 0
        assert fold["mae"] > 0


def test_cv_global_rf_no_test_param():
    """cv_global_rf must not accept test data as a parameter."""
    import inspect

    sig = inspect.signature(cv_global_rf)
    params = set(sig.parameters.keys())
    assert "X_test" not in params
    assert "y_test" not in params
