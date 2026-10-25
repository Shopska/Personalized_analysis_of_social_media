"""Standalone seed stability analysis for NB07.

Trains B3 (RF per-creator), B4 (XGB per-creator), and C5 (two-stage) across
5 seeds and writes results to results/seed_stability.csv.

Run from the project root:
    python scripts/run_seed_stability.py
"""
import os, sys, time
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor

# Project root = this script's parent directory
ROOT = Path(__file__).parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from creator_analytics import features as fe
from creator_analytics.config import load_config
from creator_analytics.models import (
    TrainedModel, train_per_creator_models, train_two_stage,
    VS_CREATOR_SUFFIX,
)
from creator_analytics.evaluate import (
    predict_per_creator, compute_metrics, compute_per_creator_metrics,
)

# ── Load data ─────────────────────────────────────────────────────────────
print("Loading data...")
cfg = load_config("config/default.yaml")
PROCESSED = Path(cfg["paths"]["processed_dir"])
PREDS_DIR = Path("results/predictions")

data, spec = fe.load_feature_table(PROCESSED)
idx = pd.read_parquet(PROCESSED / "dataset_index.parquet")

FEATURES_GLOBAL = spec["features"]
FEATURES_PC = [f for f in FEATURES_GLOBAL if not f.endswith(VS_CREATOR_SUFFIX)]

TRAIN = idx["split"].eq("train")
TEST  = idx["split"].eq("test")

X_train_global = data.loc[TRAIN.values, FEATURES_GLOBAL].reset_index(drop=True)
X_train_pc     = data.loc[TRAIN.values, FEATURES_PC].reset_index(drop=True)
y_train        = idx.loc[TRAIN, "target_rel"].reset_index(drop=True)
auth_train     = idx.loc[TRAIN, "author_id"].reset_index(drop=True)

X_test_global  = data.loc[TEST.values, FEATURES_GLOBAL].reset_index(drop=True)
X_test_pc      = data.loc[TEST.values, FEATURES_PC].reset_index(drop=True)
auth_test      = idx.loc[TEST, "author_id"].reset_index(drop=True)

nb04 = pd.read_parquet(PREDS_DIR / "nb04_predictions.parquet")
y_test = nb04["y_true"]
b1_preds = nb04["B1_RF"].values
b2_preds = nb04["B2_XGB"].values

print(f"Train: {len(X_train_pc):,}  Test: {len(X_test_pc):,}  Creators: {auth_train.nunique()}")

# ── Configs (fixed, no re-search) ─────────────────────────────────────────
RF_CFG  = {"n_estimators": 100, "max_depth": 4, "min_samples_leaf": 5, "max_features": 0.4}
XGB_CFG = {"n_estimators": 100, "max_depth": 3, "min_child_weight": 5,
           "reg_alpha": 1.0, "reg_lambda": 2.0,
           "subsample": 0.7, "colsample_bytree": 0.6, "learning_rate": 0.05}

b2_model = TrainedModel.load("models/B2_global_xgb.joblib")
B2_PARAMS = b2_model.params

print("RF  config:", RF_CFG)
print("XGB config:", XGB_CFG)
print("B2  params:", B2_PARAMS)

# ── Reference predictions ──────────────────────────────────────────────────
REF_PREDS = {
    "B3_RF_pc":    b1_preds,
    "B4_XGB_pc":   b2_preds,
    "C5_two_stage": b2_preds,
}
REF_NAME = {
    "B3_RF_pc": "B1_RF", "B4_XGB_pc": "B2_XGB", "C5_two_stage": "B2_XGB",
}

# ── Seed loop ─────────────────────────────────────────────────────────────
SEEDS = [42, 7, 13, 99, 2025]
records = []

for seed in SEEDS:
    t_seed = time.time()
    print(f"\n{'='*50}")
    print(f"Seed {seed}")
    print('='*50)

    # B3 + B4
    t0 = time.time()
    rf_models_s, xgb_models_s = train_per_creator_models(
        X_train_pc, y_train, auth_train,
        rf_config=RF_CFG, xgb_config=XGB_CFG,
        n_jobs=-1, seed=seed,
    )
    print(f"  B3/B4 trained: {len(rf_models_s)} creators, {time.time()-t0:.0f}s")

    b3_preds = predict_per_creator(X_test_pc, auth_test, rf_models_s)
    b4_preds = predict_per_creator(X_test_pc, auth_test, xgb_models_s)

    # C5: global XGB with B2 params + new seed, then residuals
    t0 = time.time()
    xgb_global = XGBRegressor(**B2_PARAMS, random_state=seed, tree_method="hist", verbosity=0)
    xgb_global.fit(X_train_global.values, y_train.values)
    b2_seeded = TrainedModel(
        model=xgb_global,
        cv_rmse=float("nan"),
        params={**B2_PARAMS, "random_state": seed},
        feature_names=list(X_train_global.columns),
        model_name=f"XGB_seed{seed}",
    )
    c5_model = train_two_stage(
        X_train_pc, y_train, auth_train,
        global_model=b2_seeded,
        seed=seed, n_jobs=-1,
        X_train_global=X_train_global,
    )
    print(f"  C5 trained: {time.time()-t0:.0f}s")

    c5_preds = c5_model.predict(X_test_pc, auth_test, X_global=X_test_global)

    # Metrics
    for name, preds in [("B3_RF_pc", b3_preds), ("B4_XGB_pc", b4_preds),
                        ("C5_two_stage", c5_preds)]:
        m = compute_metrics(y_test, preds)
        pc_a = compute_per_creator_metrics(
            y_test, pd.Series(preds, index=y_test.index), auth_test)
        pc_ref = compute_per_creator_metrics(
            y_test, pd.Series(REF_PREDS[name], index=y_test.index), auth_test)
        merged = (pc_a.set_index("author_id")["rmse"]
                      .to_frame("a")
                      .join(pc_ref.set_index("author_id")["rmse"].rename("b")))
        pc_win = float((merged["a"] < merged["b"]).mean())
        records.append({
            "model":              name,
            "ref_model":          REF_NAME[name],
            "seed":               seed,
            "rmse":               m["rmse"],
            "mae":                m["mae"],
            "spearman":           m["spearman"],
            "median_pc_spearman": float(pc_a["spearman"].median()),
            "pc_win_vs_ref":      round(pc_win, 3),
        })
        print(f"  {name:15s}  RMSE={m['rmse']:.4f}  Spear={m['spearman']:.4f}  "
              f"pc_win={pc_win:.3f}")

    print(f"  Seed {seed} total: {time.time()-t_seed:.0f}s")

# ── Save results ───────────────────────────────────────────────────────────
df = pd.DataFrame(records)
out = Path("results/seed_stability.csv")
df.to_csv(out, index=False)
print(f"\nSaved {out}  ({len(df)} rows)")
print(df.to_string())
