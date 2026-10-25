"""Single entry point: ``python -m creator_analytics <command>``.

Commands
--------
data-summary    Print dataset size before and after filtering.
train           Train baseline models.
evaluate        Evaluate a saved model on the held-out test set.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from creator_analytics.config import load_config
from creator_analytics.data import filter_creators, load_raw_table, train_test_split_by_creator
from creator_analytics.utils import set_seed, setup_logging

logger = logging.getLogger(__name__)


def cmd_data_summary(config: dict) -> int:
    """Print basic dataset stats: videos, creators, before and after filtering."""
    data_cfg = config["data"]
    author_column = data_cfg["author_column"]
    min_videos = data_cfg["min_videos_per_creator"]

    df = load_raw_table(config["paths"]["raw_dir"], "videos")
    filtered = filter_creators(df, author_column, min_videos)

    logger.info("Raw dataset: %d videos, %d creators", len(df), df[author_column].nunique())
    logger.info(
        "With >= %d videos per creator: %d videos, %d creators",
        min_videos, len(filtered), filtered[author_column].nunique(),
    )
    return 0


def _load_train_test(config: dict):
    """Shared data loading for train and evaluate commands."""
    import pandas as pd
    from creator_analytics.features import build_features
    from creator_analytics.models import prepare_X
    from creator_analytics.targets import build_target, get_nonmonotonic_video_ids

    RAW = config["paths"]["raw_dir"]
    min_vids = config["data"]["min_videos_per_creator"]
    seed = config["seed"]

    videos = load_raw_table(RAW, "videos")
    creator_daily = load_raw_table(RAW, "creator_daily")
    engagement = load_raw_table(RAW, "engagement_daily")

    bad_ids = get_nonmonotonic_video_ids(engagement)
    videos_clean = videos[~videos["video_id"].isin(bad_ids)].reset_index(drop=True)
    videos_f = filter_creators(videos_clean, "author_id", min_videos=min_vids)

    train_df, test_df = train_test_split_by_creator(
        videos_f, "author_id", test_size=config["split"]["test_size"], seed=seed
    )
    train_ids = set(train_df["video_id"])
    test_ids = set(test_df["video_id"])

    feat_path = Path(config["paths"].get("processed_features", "data/processed/features.parquet"))
    if feat_path.exists():
        features = pd.read_parquet(feat_path)
    else:
        logger.info("features.parquet not found — building now")
        features = build_features(videos_f, creator_daily, train_video_ids=train_ids)
        feat_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(str(feat_path), index=False)

    rel_t = build_target(engagement, videos_f, day=14, mode="relative",
                         train_video_ids=train_ids)
    data = (features
            .merge(videos_f[["video_id", "author_id"]], on="video_id")
            .merge(rel_t, on="video_id"))

    train_mask = data["video_id"].isin(train_ids)
    test_mask = data["video_id"].isin(test_ids)

    X_all = prepare_X(data.drop(columns=["author_id", "target"]))
    X_train = X_all[train_mask].reset_index(drop=True)
    y_train = data.loc[train_mask, "target"].reset_index(drop=True)
    author_train = data.loc[train_mask, "author_id"].reset_index(drop=True)

    X_test = X_all[test_mask].reset_index(drop=True)
    y_test = data.loc[test_mask, "target"].reset_index(drop=True)
    author_test = data.loc[test_mask, "author_id"].reset_index(drop=True)

    return X_train, y_train, author_train, X_test, y_test, author_test


def cmd_train_global(config: dict) -> int:
    """Train RF and XGBoost (with/without follower); save best XGBoost model."""
    import datetime
    from creator_analytics.evaluate import log_experiment, save_metrics
    from creator_analytics.models import (
        search_random_forest, search_xgboost,
        trivial_zero_rmse, trivial_creator_mean,
    )

    logger.info("Loading data …")
    X_train, y_train, author_train, X_test, y_test, author_test = _load_train_test(config)
    seed = config["seed"]
    n_splits = config.get("cv", {}).get("n_splits", 5)
    today = datetime.date.today().isoformat()
    exp_path = config["paths"].get("experiments_log", "results/experiments.md")
    metrics_path = config["paths"].get("metrics", "results/metrics.json")

    # Trivial baselines (computed on test — but they use no model)
    rmse_zero = trivial_zero_rmse(y_test)
    rmse_cmean, _ = trivial_creator_mean(y_train, author_train, y_test, author_test)
    logger.info("Trivial predict_0 RMSE=%.4f  predict_creator_mean RMSE=%.4f", rmse_zero, rmse_cmean)

    for name, rmse_val in [("predict_0", rmse_zero), ("predict_creator_mean", rmse_cmean)]:
        log_experiment(exp_path, name, today, "none", name, {}, None,
                       {"rmse": rmse_val, "r2": float("nan"), "spearman": float("nan")})

    # RandomForest
    logger.info("Searching RandomForest …")
    rf_model = search_random_forest(X_train, y_train, author_train,
                                    n_splits=n_splits, seed=seed)
    log_experiment(exp_path, "RF_global", today, "all_54", "RandomForest",
                   rf_model.params, rf_model.cv_rmse,
                   {"rmse": float("nan"), "r2": float("nan"), "spearman": float("nan")},
                   "CV only — run evaluate for test metrics")

    # XGBoost with follower
    logger.info("Searching XGBoost (all features) …")
    xgb_with = search_xgboost(X_train, y_train, author_train,
                               n_splits=n_splits, seed=seed)
    log_experiment(exp_path, "XGB_global_with_follower", today, "all_54", "XGBoost",
                   xgb_with.params, xgb_with.cv_rmse,
                   {"rmse": float("nan"), "r2": float("nan"), "spearman": float("nan")})

    # XGBoost without follower
    logger.info("Searching XGBoost (no follower) …")
    xgb_nof = search_xgboost(X_train, y_train, author_train,
                              n_splits=n_splits, seed=seed,
                              exclude_cols=["follower_count_log"])
    log_experiment(exp_path, "XGB_global_no_follower", today, "all_54_no_follower", "XGBoost",
                   xgb_nof.params, xgb_nof.cv_rmse,
                   {"rmse": float("nan"), "r2": float("nan"), "spearman": float("nan")})

    # Save best XGBoost (with follower, usually better)
    best = xgb_with if xgb_with.cv_rmse <= xgb_nof.cv_rmse else xgb_nof
    model_path = "models/global_xgb.joblib"
    best.save(model_path)
    logger.info("Saved best model to %s (CV RMSE=%.4f)", model_path, best.cv_rmse)

    save_metrics({
        "trivial_predict_0_rmse": rmse_zero,
        "trivial_creator_mean_rmse": rmse_cmean,
        "rf_cv_rmse": rf_model.cv_rmse,
        "xgb_with_follower_cv_rmse": xgb_with.cv_rmse,
        "xgb_no_follower_cv_rmse": xgb_nof.cv_rmse,
    }, metrics_path)
    return 0


def cmd_evaluate_global(config: dict) -> int:
    """Load saved model, evaluate on test set, save metrics + creator breakdown."""
    import datetime
    import pandas as pd
    from creator_analytics.evaluate import (
        compute_metrics, compute_per_creator_metrics, log_experiment, save_metrics,
    )
    from creator_analytics.models import TrainedModel

    model_path = "models/global_xgb.joblib"
    if not Path(model_path).exists():
        logger.error("%s not found — run `train` first.", model_path)
        return 1

    logger.info("Loading model from %s", model_path)
    model = TrainedModel.load(model_path)

    logger.info("Loading test data …")
    _, _, _, X_test, y_test, author_test = _load_train_test(config)

    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    logger.info("Test  RMSE=%.4f  MAE=%.4f  R²=%.4f  Spearman=%.4f",
                metrics["rmse"], metrics["mae"], metrics["r2"], metrics["spearman"])

    creator_metrics = compute_per_creator_metrics(y_test, pd.Series(y_pred), author_test)
    zero_rmse = float(y_test.std())
    n_worse = int((creator_metrics["rmse"] > zero_rmse).sum())
    logger.info("%d / %d creators where model > predict_0 RMSE", n_worse, len(creator_metrics))

    today = datetime.date.today().isoformat()
    exp_path = config["paths"].get("experiments_log", "results/experiments.md")
    metrics_path = config["paths"].get("metrics", "results/metrics.json")

    log_experiment(exp_path, f"{model.model_name}_test", today,
                   "all_54" if "no_follower" not in model.model_name else "no_follower",
                   "XGBoost", model.params, model.cv_rmse, metrics)

    save_metrics({
        "global_xgb_test_rmse": metrics["rmse"],
        "global_xgb_test_mae": metrics["mae"],
        "global_xgb_test_r2": metrics["r2"],
        "global_xgb_test_spearman": metrics["spearman"],
    }, metrics_path)

    pred_path = Path(config["paths"].get("predictions_global", "results/predictions_global.parquet"))
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred,
                  "author_id": author_test.values}).to_parquet(str(pred_path), index=False)

    creator_metrics_path = Path("results/creator_metrics_global.parquet")
    creator_metrics.to_parquet(str(creator_metrics_path), index=False)
    logger.info("Saved creator metrics to %s", creator_metrics_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m creator_analytics",
        description="Explainable-ML analysis of TikTok creator performance.",
    )
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--override", action="append", default=[])
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("data-summary")
    subparsers.add_parser("train", help="Train global baseline models")
    subparsers.add_parser("evaluate", help="Evaluate saved model on test")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config, args.override)
    setup_logging(config["logging"]["level"])
    set_seed(config["seed"])

    if args.command == "data-summary":
        sys.exit(cmd_data_summary(config))
    elif args.command == "train":
        sys.exit(cmd_train_global(config))
    elif args.command == "evaluate":
        sys.exit(cmd_evaluate_global(config))
