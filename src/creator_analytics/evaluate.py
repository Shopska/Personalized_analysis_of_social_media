"""Evaluation utilities: scalar metrics, per-creator breakdown, paired tests, experiments log."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
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
        "rmse":     float(np.sqrt(mean_squared_error(yt, yp))),
        "mae":      float(mean_absolute_error(yt, yp)),
        "r2":       float(r2_score(yt, yp)),
        "spearman": float(rho),
        "n":        int(len(yt)),
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
        yt = y_true.loc[idx]
        yp = y_pred.loc[idx]
        n = len(yt)
        rho = float(spearmanr(yt, yp).statistic) if n >= 2 else float("nan")
        rows.append({
            "author_id": author,
            "n_test":    n,
            "rmse":      rmse(yt, yp),
            "mae":       float(mean_absolute_error(yt, yp)),
            "spearman":  rho,
        })
    return pd.DataFrame(rows).sort_values("author_id").reset_index(drop=True)


# ------------------------------------------------------------------ #
# Paired comparison (Wilcoxon + bootstrap CI)                          #
# ------------------------------------------------------------------ #

def _creator_size_bin(n: int) -> str:
    if n < 50:
        return "24-50"
    if n < 100:
        return "50-100"
    if n < 300:
        return "100-300"
    return "300+"


def paired_comparison(
    y_true: pd.Series,
    pred_a: np.ndarray | pd.Series,
    pred_b: np.ndarray | pd.Series,
    author_ids: pd.Series,
    name_a: str = "A",
    name_b: str = "B",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Paired Wilcoxon signed-rank test + bootstrap 95% CI on RMSE difference.

    Compares model A vs model B per video (squared error difference).
    Also reports stratified win-rate by creator size bucket.

    Returns a dict with:
      rmse_a, rmse_b, rmse_diff (a-b, negative = a is better),
      wilcoxon_stat, wilcoxon_pvalue,
      bootstrap_ci_low, bootstrap_ci_high,
      win_rate_a (fraction of videos where |err_a| < |err_b|),
      stratified_win_rate: dict[size_bin -> win_rate_a]
    """
    yt = np.asarray(y_true, dtype=float)
    pa = np.asarray(pred_a, dtype=float)
    pb = np.asarray(pred_b, dtype=float)

    err_a = (yt - pa) ** 2
    err_b = (yt - pb) ** 2
    diff  = err_a - err_b  # positive → A worse

    r_a = float(np.sqrt(err_a.mean()))
    r_b = float(np.sqrt(err_b.mean()))

    try:
        stat, pval = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        stat, pval = float("nan"), float("nan")

    rng = np.random.RandomState(seed)
    n = len(err_a)
    boot_diffs = np.array([
        np.sqrt(err_a[idx].mean()) - np.sqrt(err_b[idx].mean())
        for _ in range(n_bootstrap)
        for idx in [rng.choice(n, n, replace=True)]
    ])
    ci_low  = float(np.percentile(boot_diffs, 2.5))
    ci_high = float(np.percentile(boot_diffs, 97.5))

    win_rate_a = float((np.abs(yt - pa) < np.abs(yt - pb)).mean())

    n_train_per_creator = author_ids.value_counts()
    strat: dict[str, dict] = {}
    for author, idx in author_ids.groupby(author_ids).groups.items():
        n_c = n_train_per_creator.get(author, 0)
        bucket = _creator_size_bin(n_c)
        if bucket not in strat:
            strat[bucket] = {"wins_a": 0, "total": 0}
        wins = int((np.abs(yt[idx] - pa[idx]) < np.abs(yt[idx] - pb[idx])).sum())
        strat[bucket]["wins_a"] += wins
        strat[bucket]["total"]  += len(idx)

    stratified_win_rate = {
        b: v["wins_a"] / v["total"] for b, v in strat.items() if v["total"] > 0
    }

    return {
        "name_a": name_a, "name_b": name_b,
        "rmse_a": r_a, "rmse_b": r_b,
        "rmse_diff": r_a - r_b,
        "wilcoxon_stat": float(stat), "wilcoxon_pvalue": float(pval),
        "bootstrap_ci_low": ci_low, "bootstrap_ci_high": ci_high,
        "win_rate_a": win_rate_a,
        "stratified_win_rate": stratified_win_rate,
    }


# ------------------------------------------------------------------ #
# Leakage check                                                        #
# ------------------------------------------------------------------ #

def leakage_check(
    model_predict_fn,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    seed: int = 42,
    n_shuffle: int = 5,
) -> dict:
    """Shuffle y_test n_shuffle times; R² on shuffled targets should be ≈ 0.

    Returns {'r2_real': float, 'r2_shuffled_mean': float, 'r2_shuffled_std': float}.
    """
    rng = np.random.RandomState(seed)
    preds = model_predict_fn(X_test)
    r2_real = float(r2_score(y_test.values, preds))

    shuffled_r2s = []
    for _ in range(n_shuffle):
        y_shuf = rng.permutation(y_test.values)
        shuffled_r2s.append(float(r2_score(y_shuf, preds)))

    return {
        "r2_real":          r2_real,
        "r2_shuffled_mean": float(np.mean(shuffled_r2s)),
        "r2_shuffled_std":  float(np.std(shuffled_r2s)),
    }


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
    train_time_s: float | None = None,
    comparison: dict | None = None,
    comment: str = "",
) -> None:
    """Append one row to results/experiments.md.

    comparison: output of paired_comparison() — adds win_rate, p_value, CI columns.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    params_str = (
        json.dumps(params, ensure_ascii=False)[:100]
        if isinstance(params, dict) else str(params)[:100]
    )
    cv_str    = f"{cv_rmse:.4f}" if cv_rmse is not None else "n/a"
    time_str  = f"{train_time_s:.0f}s" if train_time_s is not None else "n/a"
    win_str   = f"{comparison['win_rate_a']:.3f}" if comparison else "n/a"
    pval_str  = f"{comparison['wilcoxon_pvalue']:.4f}" if comparison else "n/a"
    ci_str    = (
        f"[{comparison['bootstrap_ci_low']:.4f}, {comparison['bootstrap_ci_high']:.4f}]"
        if comparison else "n/a"
    )

    line = (
        f"| {date} | {name} | {model} | {features} | `{params_str}` "
        f"| {cv_str} "
        f"| {test_metrics.get('rmse', 'n/a'):.4f} "
        f"| {test_metrics.get('r2', 'n/a'):.4f} "
        f"| {test_metrics.get('spearman', 'n/a'):.4f} "
        f"| {time_str} | {win_str} | {pval_str} | {ci_str} | {comment} |\n"
    )
    table_header = (
        "# Experiments Log\n\n"
        "| Date | Name | Model | Features | Params (truncated) "
        "| CV RMSE | Test RMSE | Test R² | Test Spearman "
        "| Train time | Win rate | p-value | 95% CI RMSE diff | Comment |\n"
        "|------|------|-------|----------|--------------------|"
        "---------|-----------|---------|---------------|"
        "------------|----------|---------|------------------|---------|"
        "\n"
    )
    if not path.exists() or "| Date |" not in path.read_text(encoding="utf-8"):
        path.write_text(table_header + line, encoding="utf-8")
    else:
        existing = path.read_text(encoding="utf-8")
        if f"| {name} |" in existing:
            return  # idempotent: skip if name already logged
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)


def predict_per_creator(
    X_test: pd.DataFrame,
    author_ids_test: pd.Series,
    creator_models: dict,
) -> np.ndarray:
    """Return predictions aligned with X_test / author_ids_test.

    Fills NaN for any creator not in creator_models.
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
