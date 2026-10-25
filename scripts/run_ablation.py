"""Feature ablation CV for B1 (global RF) — saves results/ablation.csv.

Trains B1 across all feature combinations using 5-fold StratifiedKFold
by author_id. Test data is never used.

Run from the project root:
    python scripts/run_ablation.py

Estimated wall time: ~15 min (27 combos × 5-fold CV, 12 cores).
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from creator_analytics import features as fe  # noqa: E402
from creator_analytics.ablation import (  # noqa: E402
    build_combinations,
    build_group_map,
    cv_global_rf,
)
from creator_analytics.config import load_config  # noqa: E402

# ── Load data ─────────────────────────────────────────────────────────────────
cfg = load_config("config/default.yaml")
PROCESSED = Path(cfg["paths"]["processed_dir"])
SEED = cfg["seed"]

print("Loading data...")
data, spec = fe.load_feature_table(PROCESSED)
idx = pd.read_parquet(PROCESSED / "dataset_index.parquet")

FEATURES_GLOBAL = spec["features"]

TRAIN = idx["split"].eq("train")
X_train_global = data.loc[TRAIN.values, FEATURES_GLOBAL].reset_index(drop=True)
y_train = idx.loc[TRAIN, "target_rel"].reset_index(drop=True)
auth_train = idx.loc[TRAIN, "author_id"].reset_index(drop=True)

print(f"Train: {len(X_train_global):,}  Creators: {auth_train.nunique()}")

# ── Configs (fixed, no re-search) ─────────────────────────────────────────────
B1_PARAMS = {
    "n_estimators": 100,
    "min_samples_split": 5,
    "min_samples_leaf": 5,
    "max_features": 0.3,
    "max_depth": None,
}

# ── Combinations ──────────────────────────────────────────────────────────────
GROUPS = build_group_map(spec)
combos = build_combinations(FEATURES_GLOBAL, GROUPS)
print(f"Combinations: B1={len(combos)}")

# ── Helpers ───────────────────────────────────────────────────────────────────


def aggregate_folds(fold_metrics: list[dict]) -> dict:
    keys = fold_metrics[0].keys()
    return {f"cv_{k}_mean": float(np.mean([f[k] for f in fold_metrics])) for k in keys} | {
        f"cv_{k}_std": float(np.std([f[k] for f in fold_metrics], ddof=1)) for k in keys
    }


# ── B1 ablation ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"B1 global RF — {len(combos)} combinations, 5-fold CV")
print("=" * 60)

b1_records = []
t_b1_start = time.time()
for i, combo in enumerate(combos, 1):
    feats = combo["features"]
    if not feats:
        print(f"  [{i:2d}/{len(combos)}] {combo['combination']:30s}  SKIP (0 features)")
        continue
    t0 = time.time()
    fold_metrics = cv_global_rf(
        X_train_global,
        y_train,
        auth_train,
        features=feats,
        rf_params=B1_PARAMS,
        n_splits=5,
        seed=SEED,
    )
    agg = aggregate_folds(fold_metrics)
    elapsed = time.time() - t0
    print(
        f"  [{i:2d}/{len(combos)}] {combo['combination']:30s}  "
        f"RMSE={agg['cv_rmse_mean']:.4f}±{agg['cv_rmse_std']:.4f}  "
        f"Spear={agg['cv_spearman_mean']:.4f}  {elapsed:.0f}s"
    )
    b1_records.append(
        {
            "model": "B1_RF_global",
            "combination": combo["combination"],
            "type": combo["type"],
            "groups_label": combo["groups_label"],
            "n_features": len(feats),
            **agg,
        }
    )

print(f"B1 total: {(time.time()-t_b1_start)/60:.1f} min")

# ── Compute delta vs baseline (all_features) ──────────────────────────────────
df = pd.DataFrame(b1_records)

baseline = df[df["combination"] == "all_features"]
if not baseline.empty:
    base_rmse = baseline["cv_rmse_mean"].iloc[0]
    base_spear = baseline["cv_spearman_mean"].iloc[0]
    df["rmse_vs_baseline"] = df["cv_rmse_mean"] - base_rmse
    df["spearman_vs_baseline"] = df["cv_spearman_mean"] - base_spear

# Test columns — filled later in NB08 for baseline + best combo only
df["test_rmse"] = float("nan")
df["test_mae"] = float("nan")
df["test_spearman"] = float("nan")

# ── Save ──────────────────────────────────────────────────────────────────────
out = Path("results/ablation.csv")
df.to_csv(out, index=False)
print(f"\nSaved {out}  ({len(df)} rows)")
print(df[["combination", "n_features", "cv_rmse_mean", "rmse_vs_baseline"]].to_string())
