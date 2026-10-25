"""Feature ablation helpers: group map, combination builder, CV evaluator.

Public API:
    build_group_map     – merge cat_* sub-groups from feature_spec
    build_combinations  – generate all ablation combinations (27 for B1 global RF)
    cv_global_rf        – 5-fold CV for a global RF on a feature subset
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from creator_analytics.evaluate import compute_metrics, compute_per_creator_metrics
from creator_analytics.models import make_cv_splits

_CONTENT_KEYS = ["format", "description", "transcript", "emotions", "music", "hashtags", "cat_*"]
_CONTEXT_KEYS = ["timing", "posting_history", "creator_context", "vs_creator"]


# ── Group map ─────────────────────────────────────────────────────────────────

def build_group_map(feature_spec: dict) -> dict[str, list[str]]:
    """Merge all cat_* sub-groups into one 'cat_*' entry, preserve ordering."""
    cat_feats: list[str] = []
    merged: dict[str, list[str]] = {}
    for name, feats in feature_spec["groups"].items():
        if name.startswith("cat_"):
            cat_feats.extend(feats)
        else:
            merged[name] = list(feats)
    if cat_feats:
        merged["cat_*"] = cat_feats
    return merged


# ── Combination builder ───────────────────────────────────────────────────────

def build_combinations(
    features_global: list[str],
    groups: dict[str, list[str]],
) -> list[dict]:
    """Return the full ablation combination list (27 B1 global RF combos).

    Each entry has:
        combination    – unique name
        type           – "leave_one_out" | "single_group" | "meaningful"
        groups_label   – human-readable description
        features       – feature list for global model
    """
    feat_global_set = set(features_global)
    combos: list[dict] = []

    # ── Leave-one-group-out ──────────────────────────────────────────────────
    for g_name, g_feats in groups.items():
        excl = set(g_feats)
        combos.append({
            "combination":  f"LOO_{g_name}",
            "type":         "leave_one_out",
            "groups_label": f"all−{g_name}",
            "features":     [f for f in features_global if f not in excl],
        })

    # ── Single-group ─────────────────────────────────────────────────────────
    for g_name, g_feats in groups.items():
        fg = [f for f in g_feats if f in feat_global_set]
        combos.append({
            "combination":  f"SINGLE_{g_name}",
            "type":         "single_group",
            "groups_label": g_name,
            "features":     fg,
        })

    # ── Meaningful combinations ───────────────────────────────────────────────
    content_feats = set(f for k in _CONTENT_KEYS for f in groups.get(k, []))
    context_feats = set(f for k in _CONTEXT_KEYS for f in groups.get(k, []))
    creator_ctx   = set(groups.get("creator_context", []))
    vs_creator    = set(groups.get("vs_creator", []))

    combos += [
        {
            "combination":  "content_only",
            "type":         "meaningful",
            "groups_label": ",".join(_CONTENT_KEYS),
            "features":     [f for f in features_global if f in content_feats],
        },
        {
            "combination":  "context_only",
            "type":         "meaningful",
            "groups_label": ",".join(_CONTEXT_KEYS),
            "features":     [f for f in features_global if f in context_feats],
        },
        {
            "combination":  "without_creator_context",
            "type":         "meaningful",
            "groups_label": "all−creator_context",
            "features":     [f for f in features_global if f not in creator_ctx],
        },
        {
            "combination":  "without_vs_creator",
            "type":         "meaningful",
            "groups_label": "all−vs_creator",
            "features":     [f for f in features_global if f not in vs_creator],
        },
        {
            "combination":  "all_features",
            "type":         "meaningful",
            "groups_label": "all",
            "features":     list(features_global),
        },
    ]
    return combos


# ── CV helpers ────────────────────────────────────────────────────────────────

def _fold_metrics(y_vl: pd.Series, preds: np.ndarray, auth_vl: pd.Series) -> dict:
    m  = compute_metrics(y_vl, preds)
    pc = compute_per_creator_metrics(y_vl, pd.Series(preds, index=y_vl.index), auth_vl)
    return {
        "rmse":               m["rmse"],
        "mae":                m["mae"],
        "spearman":           m["spearman"],
        "median_pc_spearman": float(pc["spearman"].median()),
    }


def cv_global_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    auth_train: pd.Series,
    features: list[str],
    rf_params: dict,
    n_splits: int = 5,
    seed: int = 42,
) -> list[dict]:
    """5-fold StratifiedKFold CV for a global RF on ``features``.

    Returns list of 5 per-fold metric dicts (rmse, mae, spearman, median_pc_spearman).
    Test data is never a parameter.
    """
    splits = make_cv_splits(auth_train, n_splits=n_splits, seed=seed)
    results = []
    for train_idx, val_idx in splits:
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("rf",  RandomForestRegressor(**rf_params, n_jobs=-1, random_state=seed)),
        ])
        pipe.fit(X_train.iloc[train_idx][features], y_train.iloc[train_idx])
        preds = pipe.predict(X_train.iloc[val_idx][features])
        results.append(_fold_metrics(y_train.iloc[val_idx], preds, auth_train.iloc[val_idx]))
    return results
