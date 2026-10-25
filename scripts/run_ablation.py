"""Feature ablation CV — saves results/ablation.csv.

Trains B1 (global RF, 100 trees) and B3 (per-creator RF) across all feature
combinations using 5-fold StratifiedKFold by author_id. Test data is never used.

Run from the project root:
    python scripts/run_ablation.py

Estimated wall time: ~50 min (B1 ~26 min + B3 ~24 min, 12 cores).
"""
import os, sys, time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from creator_analytics import features as fe
from creator_analytics.config import load_config
from creator_analytics.models import VS_CREATOR_SUFFIX, train_per_creator_models
from creator_analytics.evaluate import compute_metrics, compute_per_creator_metrics, predict_per_creator
from creator_analytics.ablation import (
    build_group_map, build_combinations, filter_combinations_b3,
    cv_global_rf, cv_per_creator_rf,
)

# ── Load data ─────────────────────────────────────────────────────────────────
cfg      = load_config("config/default.yaml")
PROCESSED = Path(cfg["paths"]["processed_dir"])
SEED     = cfg["seed"]

print("Loading data...")
data, spec = fe.load_feature_table(PROCESSED)
idx = pd.read_parquet(PROCESSED / "dataset_index.parquet")

FEATURES_GLOBAL = spec["features"]
FEATURES_PC     = [f for f in FEATURES_GLOBAL if not f.endswith(VS_CREATOR_SUFFIX)]

TRAIN = idx["split"].eq("train")
X_train_global = data.loc[TRAIN.values, FEATURES_GLOBAL].reset_index(drop=True)
X_train_pc     = data.loc[TRAIN.values, FEATURES_PC].reset_index(drop=True)
y_train        = idx.loc[TRAIN, "target_rel"].reset_index(drop=True)
auth_train     = idx.loc[TRAIN, "author_id"].reset_index(drop=True)

print(f"Train: {len(X_train_global):,}  Creators: {auth_train.nunique()}")

# ── Configs (fixed, no re-search) ─────────────────────────────────────────────
# B1: best found params, n_estimators reduced to 100 for ablation speed
# (all other params unchanged; rankings stable at 100 trees)
B1_PARAMS = {
    "n_estimators": 100, "min_samples_split": 5,
    "min_samples_leaf": 5, "max_features": 0.3, "max_depth": None,
}
# B3: fixed per-creator config (same as seed_stability.py)
B3_CONFIG = {
    "n_estimators": 100, "max_depth": 4,
    "min_samples_leaf": 5, "max_features": 0.4,
}

# ── Combinations ──────────────────────────────────────────────────────────────
GROUPS    = build_group_map(spec)
combos    = build_combinations(FEATURES_GLOBAL, FEATURES_PC, GROUPS)
combos_b3 = filter_combinations_b3(combos)
print(f"Combinations: B1={len(combos)}, B3={len(combos_b3)}")

# ── Helpers ───────────────────────────────────────────────────────────────────

def aggregate_folds(fold_metrics: list[dict]) -> dict:
    keys = fold_metrics[0].keys()
    return {
        f"cv_{k}_mean": float(np.mean([f[k] for f in fold_metrics]))
        for k in keys
    } | {
        f"cv_{k}_std": float(np.std([f[k] for f in fold_metrics], ddof=1))
        for k in keys
    }


# ── B1 ablation ───────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"B1 global RF — {len(combos)} combinations, 5-fold CV")
print("="*60)

b1_records = []
t_b1_start = time.time()
for i, combo in enumerate(combos, 1):
    feats = combo["features_global"]
    if not feats:
        print(f"  [{i:2d}/{len(combos)}] {combo['combination']:30s}  SKIP (0 features)")
        continue
    t0 = time.time()
    fold_metrics = cv_global_rf(
        X_train_global, y_train, auth_train,
        features=feats, rf_params=B1_PARAMS, n_splits=5, seed=SEED,
    )
    agg = aggregate_folds(fold_metrics)
    elapsed = time.time() - t0
    print(f"  [{i:2d}/{len(combos)}] {combo['combination']:30s}  "
          f"RMSE={agg['cv_rmse_mean']:.4f}±{agg['cv_rmse_std']:.4f}  "
          f"Spear={agg['cv_spearman_mean']:.4f}  {elapsed:.0f}s")
    b1_records.append({
        "model":        "B1_RF_global",
        "combination":  combo["combination"],
        "type":         combo["type"],
        "groups_label": combo["groups_label"],
        "n_features":   len(feats),
        **agg,
    })

print(f"B1 total: {(time.time()-t_b1_start)/60:.1f} min")

# ── B3 ablation ───────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"B3 per-creator RF — {len(combos_b3)} combinations, 5-fold OOF CV")
print("="*60)

b3_records = []
t_b3_start = time.time()
for i, combo in enumerate(combos_b3, 1):
    feats = combo["features_pc"]
    if not feats:
        print(f"  [{i:2d}/{len(combos_b3)}] {combo['combination']:30s}  SKIP (0 features)")
        continue
    t0 = time.time()
    fold_metrics = cv_per_creator_rf(
        X_train_pc, y_train, auth_train,
        features=feats, rf_config=B3_CONFIG, n_splits=5, seed=SEED,
    )
    agg = aggregate_folds(fold_metrics)
    elapsed = time.time() - t0
    print(f"  [{i:2d}/{len(combos_b3)}] {combo['combination']:30s}  "
          f"RMSE={agg['cv_rmse_mean']:.4f}±{agg['cv_rmse_std']:.4f}  "
          f"Spear={agg['cv_spearman_mean']:.4f}  {elapsed:.0f}s")
    b3_records.append({
        "model":        "B3_RF_pc",
        "combination":  combo["combination"],
        "type":         combo["type"],
        "groups_label": combo["groups_label"],
        "n_features":   len(feats),
        **agg,
    })

print(f"B3 total: {(time.time()-t_b3_start)/60:.1f} min")

# ── Compute delta vs baseline (all_features) ──────────────────────────────────
records = b1_records + b3_records
df = pd.DataFrame(records)

for model_name in df["model"].unique():
    mask = df["model"] == model_name
    baseline = df[mask & (df["combination"] == "all_features")]
    if baseline.empty:
        continue
    base_rmse = baseline["cv_rmse_mean"].iloc[0]
    base_spear = baseline["cv_spearman_mean"].iloc[0]
    df.loc[mask, "rmse_vs_baseline"]    = df.loc[mask, "cv_rmse_mean"] - base_rmse
    df.loc[mask, "spearman_vs_baseline"] = df.loc[mask, "cv_spearman_mean"] - base_spear

# Test columns — filled later in NB08 for baseline + best combo only
df["test_rmse"]               = float("nan")
df["test_mae"]                = float("nan")
df["test_spearman"]           = float("nan")
df["test_median_pc_spearman"] = float("nan")

# ── Save ──────────────────────────────────────────────────────────────────────
out = Path("results/ablation.csv")
df.to_csv(out, index=False)
print(f"\nSaved {out}  ({len(df)} rows)")
print(df[["model", "combination", "n_features", "cv_rmse_mean", "rmse_vs_baseline"]].to_string())
