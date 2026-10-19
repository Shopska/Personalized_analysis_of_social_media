"""Evaluation utilities: scalar metrics, per-creator breakdown, experiments log."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ------------------------------------------------------------------ #
# Scalar metrics                                                       #
# ------------------------------------------------------------------ #

def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    return float(np.sqrt(mean_squared_error(
        np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    )))


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float]:
    """RMSE, MAE, R², Spearman ρ, and n."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    rho, _ = spearmanr(yt, yp)
    return {
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "r2": float(r2_score(yt, yp)),
        "spearman": float(rho),
        "n": int(len(yt)),
    }


# ------------------------------------------------------------------ #
# Per-creator breakdown                                                #
# ------------------------------------------------------------------ #

def compute_per_creator_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    author_ids: pd.Series,
) -> pd.DataFrame:
    """Per-creator RMSE, MAE, Spearman ρ, n_test — all inputs must be aligned.

    The union of per-creator n_test always equals len(y_true).
    """
    rows = []
    for author, idx in author_ids.groupby(author_ids).groups.items():
        yt = y_true.iloc[author_ids.index.get_indexer(idx)] if False else y_true.loc[idx]
        yp = y_pred.loc[idx]
        n = len(yt)
        rho = float(spearmanr(yt, yp).statistic) if n >= 2 else float("nan")
        rows.append({
            "author_id": author,
            "n_test": n,
            "rmse": rmse(yt, yp),
            "mae": float(mean_absolute_error(yt, yp)),
            "spearman": rho,
        })
    return pd.DataFrame(rows).sort_values("author_id").reset_index(drop=True)


# ------------------------------------------------------------------ #
# Experiments log                                                      #
# ------------------------------------------------------------------ #

def log_experiment(
    path: str | Path,
    name: str,
    date: str,
    features: str,
    model: str,
    params: dict | str,
    cv_rmse: float | None,
    test_metrics: dict,
    comment: str = "",
) -> None:
    """Append one row to results/experiments.md."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    params_str = (
        json.dumps(params, ensure_ascii=False)[:100] if isinstance(params, dict) else str(params)[:100]
    )
    cv_str = f"{cv_rmse:.4f}" if cv_rmse is not None else "n/a"
    line = (
        f"| {date} | {name} | {model} | {features} | `{params_str}` "
        f"| {cv_str} | {test_metrics.get('rmse', 'n/a'):.4f} "
        f"| {test_metrics.get('r2', 'n/a'):.4f} "
        f"| {test_metrics.get('spearman', 'n/a'):.4f} | {comment} |\n"
    )
    table_header = (
        "# Experiments Log\n\n"
        "| Date | Name | Model | Features | Params (truncated) "
        "| CV RMSE | Test RMSE | Test R² | Test Spearman | Comment |\n"
        "|------|------|-------|----------|--------------------|"
        "---------|-----------|---------|---------------|---------|"
        "\n"
    )
    if not path.exists() or "| Date |" not in path.read_text(encoding="utf-8"):
        path.write_text(table_header + line, encoding="utf-8")
    else:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)


def predict_per_creator(
    X_test: pd.DataFrame,
    author_ids_test: pd.Series,
    creator_models: dict,
) -> np.ndarray:
    """Return predictions aligned with X_test / author_ids_test.

    For each test video, predicts using its creator's per-creator model.
    Fills NaN for any creator not in creator_models (should be 0 with a full split).
    """
    y_pred = np.full(len(X_test), np.nan)
    for author, idx_vals in author_ids_test.groupby(author_ids_test).groups.items():
        if author not in creator_models:
            continue
        m = creator_models[author]
        positions = X_test.index.get_indexer(idx_vals)
        X_group = X_test.loc[idx_vals, m.feature_names]
        y_pred[positions] = m.model.predict(X_group)
    return y_pred


def save_metrics(metrics: dict, path: str | Path) -> None:
    """Merge new metrics into results/metrics.json (creates or updates)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    existing.update(metrics)
    path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
