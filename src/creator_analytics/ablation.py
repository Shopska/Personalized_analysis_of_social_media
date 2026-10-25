"""Feature ablation helpers: group map, combination builder, CV evaluators.

Public API:
    build_group_map        – merge cat_* sub-groups from feature_spec
    build_combinations     – generate all ablation combinations (LOO / single / meaningful)
    cv_global_rf           – 5-fold CV for a global RF on a feature subset
    cv_per_creator_rf      – 5-fold OOF CV for per-creator RF on a feature subset
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from creator_analytics.evaluate import compute_metrics, compute_per_creator_metrics
from creator_analytics.models import make_cv_splits

VS_CREATOR_SUFFIX = "_vs_creator"

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
    features_pc: list[str],
    groups: dict[str, list[str]],
) -> list[dict]:
    """Return the full ablation combination list (27 for global, 24 for per-creator).

    Each entry has:
        combination    – unique name
        type           – "leave_one_out" | "single_group" | "meaningful"
        groups_label   – human-readable description
        features_global – feature list for global model
        features_pc     – feature list for per-creator model (no vs_creator)

    For B3 (per-creator), skip combinations where features_pc is empty or
    duplicates "all_features" (i.e. LOO_vs_creator and without_vs_creator).
    """
    feat_global_set = set(features_global)
    combos: list[dict] = []

    # ── Leave-one-group-out ──────────────────────────────────────────────────
    for g_name, g_feats in groups.items():
        excl = set(g_feats)
        combos.append({
            "combination":      f"LOO_{g_name}",
            "type":             "leave_one_out",
            "groups_label":     f"all−{g_name}",
            "features_global":  [f for f in features_global if f not in excl],
            "features_pc":      [f for f in features_pc     if f not in excl],
        })

    # ── Single-group ─────────────────────────────────────────────────────────
    for g_name, g_feats in groups.items():
        fg = [f for f in g_feats if f in feat_global_set]
        combos.append({
            "combination":      f"SINGLE_{g_name}",
            "type":             "single_group",
            "groups_label":     g_name,
            "features_global":  fg,
            "features_pc":      [f for f in fg if not f.endswith(VS_CREATOR_SUFFIX)],
        })

    # ── Meaningful combinations ───────────────────────────────────────────────
    content_feats  = set(f for k in _CONTENT_KEYS for f in groups.get(k, []))
    context_feats  = set(f for k in _CONTEXT_KEYS for f in groups.get(k, []))
    creator_ctx    = set(groups.get("creator_context", []))

    combos += [
        {
            "combination":      "content_only",
            "type":             "meaningful",
            "groups_label":     ",".join(_CONTENT_KEYS),
            "features_global":  [f for f in features_global if f in content_feats],
            "features_pc":      [f for f in features_pc     if f in content_feats],
        },
        {
            "combination":      "context_only",
            "type":             "meaningful",
            "groups_label":     ",".join(_CONTEXT_KEYS),
            "features_global":  [f for f in features_global if f in context_feats],
            "features_pc":      [f for f in features_pc     if f in context_feats],
        },
        {
            "combination":      "without_creator_context",
            "type":             "meaningful",
            "groups_label":     "all−creator_context",
            "features_global":  [f for f in features_global if f not in creator_ctx],
            "features_pc":      [f for f in features_pc     if f not in creator_ctx],
        },
        {
            "combination":      "without_vs_creator",
            "type":             "meaningful",
            "groups_label":     "all−vs_creator (global only)",
            "features_global":  features_pc,  # pc features = global minus vs_creator
            "features_pc":      features_pc,
        },
        {
            "combination":      "all_features",
            "type":             "meaningful",
            "groups_label":     "all",
            "features_global":  features_global,
            "features_pc":      features_pc,
        },
    ]
    return combos


def filter_combinations_b3(combos: list[dict]) -> list[dict]:
    """Return only combinations valid for per-creator models.

    Skips:
      - SINGLE_vs_creator  (features_pc is empty)
      - LOO_vs_creator     (features_pc equals all FEATURES_PC — same as all_features for B3)
      - without_vs_creator (features_pc equals all FEATURES_PC — same as all_features for B3)
    """
    _skip = {"LOO_vs_creator", "SINGLE_vs_creator", "without_vs_creator"}
    return [c for c in combos if c["combination"] not in _skip and len(c["features_pc"]) > 0]


# ── CV helpers ────────────────────────────────────────────────────────────────

def _fold_metrics(y_vl: pd.Series, preds: np.ndarray, auth_vl: pd.Series) -> dict:
    m   = compute_metrics(y_vl, preds)
    pc  = compute_per_creator_metrics(y_vl, pd.Series(preds, index=y_vl.index), auth_vl)
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


def cv_per_creator_rf(
    X_train_pc: pd.DataFrame,
    y_train: pd.Series,
    auth_train: pd.Series,
    features: list[str],
    rf_config: dict,
    n_splits: int = 5,
    seed: int = 42,
) -> list[dict]:
    """5-fold OOF CV for per-creator RF on ``features``.

    For each fold: trains one RF per creator on fold-train data, predicts on fold-val.
    Creators with < 5 fold-train samples fall back to the fold-train mean.
    Returns list of 5 per-fold metric dicts.
    Test data is never a parameter.
    """
    splits = make_cv_splits(auth_train, n_splits=n_splits, seed=seed)
    results = []

    for train_idx, val_idx in splits:
        X_tr    = X_train_pc.iloc[train_idx][features].reset_index(drop=True)
        y_tr    = y_train.iloc[train_idx].reset_index(drop=True)
        auth_tr = auth_train.iloc[train_idx].reset_index(drop=True)
        X_vl    = X_train_pc.iloc[val_idx][features].reset_index(drop=True)
        y_vl    = y_train.iloc[val_idx].reset_index(drop=True)
        auth_vl = auth_train.iloc[val_idx].reset_index(drop=True)
        fold_mean = float(y_tr.mean())

        def _fit_one(a, _X=X_tr, _y=y_tr, _auth=auth_tr,
                     _feats=features, _cfg=rf_config, _seed=seed):
            mask = _auth == a
            Xa, ya = _X[mask], _y[mask]
            if len(Xa) < 5:
                return None
            pipe = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("rf",  RandomForestRegressor(**_cfg, n_jobs=1, random_state=_seed)),
            ])
            pipe.fit(Xa, ya)
            return (a, pipe)

        raw = joblib.Parallel(n_jobs=-1, verbose=0)(
            joblib.delayed(_fit_one)(a) for a in auth_tr.unique()
        )
        models: dict = {}
        for r in raw:
            if r is not None:
                a, pipe = r
                models[a] = pipe

        preds = np.full(len(X_vl), fold_mean)
        for a, idx_vals in auth_vl.groupby(auth_vl).groups.items():
            if a not in models:
                continue
            positions = list(idx_vals)
            preds[positions] = models[a].predict(X_vl.loc[positions])

        results.append(_fold_metrics(y_vl, preds, auth_vl))
    return results
