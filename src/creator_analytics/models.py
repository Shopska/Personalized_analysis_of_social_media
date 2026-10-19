"""Baseline model training — global and per-creator.

Design constraints
------------------
- search_* functions NEVER receive X_test / y_test — test isolation by signature.
- The StratifiedKFold uses author_id labels so that each creator appears in both
  train and val folds (same structure as the train/test split).
- XGBoost uses early stopping evaluated on the val fold (not test).
- The OrdinalEncoder is fitted on full X_train before CV; this is not leakage
  because OrdinalEncoder only learns category names, not statistics of y.
- Per-creator models: one conservative config found via CV on 100 sampled creators,
  applied identically to all 1 439 creators.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

# Categorical columns in the feature matrix (Arrow str dtype from parquet)
CAT_COLS: list[str] = ["ratio", "topic", "music_selected_from", "dominant_emotion"]
# Always drop: ID and zero-variance
DROP_COLS: list[str] = ["video_id", "is_ads"]


# ------------------------------------------------------------------ #
# Data preparation                                                     #
# ------------------------------------------------------------------ #

def prepare_X(features_df: pd.DataFrame) -> pd.DataFrame:
    """Drop ID/zero-variance cols; convert nullable/Arrow types to sklearn-compatible."""
    drop = [c for c in DROP_COLS if c in features_df.columns]
    X = features_df.drop(columns=drop).copy()
    for col in X.columns:
        dtype_str = str(X[col].dtype)
        # Arrow-backed strings → Python object (for OrdinalEncoder)
        if dtype_str in ("str", "large_string") or dtype_str.startswith("string"):
            X[col] = X[col].astype(object)
        # Nullable integers (Int8, Int16, UInt8, …) → float64 to preserve NaN
        elif dtype_str.startswith(("Int", "UInt")):
            X[col] = X[col].astype("float64")
    return X


def make_cv_splits(
    author_ids: pd.Series,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pre-compute StratifiedKFold splits using author_id as the stratum.

    Returns a list of (train_idx, val_idx) numpy arrays.
    Pass this list as the ``cv`` argument to RandomizedSearchCV or use it
    directly for XGBoost's manual search loop.
    """
    author_enc = LabelEncoder().fit_transform(author_ids)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.arange(len(author_ids)), author_enc))


# ------------------------------------------------------------------ #
# TrainedModel — a thin wrapper that bundles preprocessor + model     #
# ------------------------------------------------------------------ #

@dataclass
class TrainedModel:
    """Preprocessor + fitted model with predict() and save/load."""
    preprocessor: Any
    model: Any
    cv_rmse: float
    params: dict
    feature_names: list[str] = field(default_factory=list)
    model_name: str = ""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_aligned = X[self.feature_names]
        X_enc = self.preprocessor.transform(X_aligned)
        return self.model.predict(X_enc)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "TrainedModel":
        return joblib.load(path)


# ------------------------------------------------------------------ #
# Trivial baselines                                                    #
# ------------------------------------------------------------------ #

def trivial_zero_rmse(y_test: pd.Series) -> float:
    """RMSE of always predicting 0 (= predicting per-creator train median on relative target)."""
    return float(np.sqrt(mean_squared_error(y_test, np.zeros(len(y_test)))))


def trivial_creator_mean(
    y_train: pd.Series,
    author_ids_train: pd.Series,
    y_test: pd.Series,
    author_ids_test: pd.Series,
) -> tuple[float, np.ndarray]:
    """RMSE of predicting per-creator mean computed ONLY on train.

    Returns (rmse, predictions array aligned with y_test).
    Uses global train mean as fallback (all creators are in both splits here).
    """
    train_means = y_train.groupby(author_ids_train).mean()
    global_mean = float(y_train.mean())
    y_pred = author_ids_test.map(train_means).fillna(global_mean).values
    rmse_val = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    return rmse_val, y_pred


# ------------------------------------------------------------------ #
# RandomForest hyperparameter search                                   #
# ------------------------------------------------------------------ #

def search_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    author_ids_train: pd.Series,
    n_splits: int = 5,
    seed: int = 42,
    n_iter: int = 30,
) -> TrainedModel:
    """RandomizedSearchCV for RF; test data is NOT a parameter.

    Categorical columns are OrdinalEncoded; numeric NaN is median-imputed.
    """
    cat_cols = [c for c in CAT_COLS if c in X_train.columns]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    ct = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ("num", SimpleImputer(strategy="median"), num_cols),
    ])

    base_pipe = Pipeline([("pre", ct), ("model", RandomForestRegressor(n_jobs=-1, random_state=seed))])

    param_dist = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 8, 15, 25],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 5],
        "model__max_features": [0.3, 0.5, 0.7, "sqrt"],
    }

    cv_splits = make_cv_splits(author_ids_train, n_splits=n_splits, seed=seed)

    search = RandomizedSearchCV(
        base_pipe,
        param_dist,
        n_iter=n_iter,
        cv=cv_splits,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,  # outer parallelism; RF itself uses n_jobs=-1
        random_state=seed,
        refit=True,
        verbose=1,
    )
    search.fit(X_train, y_train)

    best_cv_rmse = float(-search.best_score_)
    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}

    best_pipe = search.best_estimator_
    # Wrap: preprocessor = pipeline.steps[0][1], model = pipeline.steps[1][1]
    fitted_pre = best_pipe.named_steps["pre"]
    fitted_model = best_pipe.named_steps["model"]

    return TrainedModel(
        preprocessor=fitted_pre,
        model=fitted_model,
        cv_rmse=best_cv_rmse,
        params=best_params,
        feature_names=list(X_train.columns),
        model_name="RandomForest",
    )


# ------------------------------------------------------------------ #
# XGBoost hyperparameter search with early stopping                   #
# ------------------------------------------------------------------ #

def search_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    author_ids_train: pd.Series,
    n_splits: int = 5,
    seed: int = 42,
    n_iter: int = 30,
    exclude_cols: list[str] | None = None,
) -> TrainedModel:
    """Random search for XGBoost with per-fold early stopping.

    Early stopping uses the CV validation fold — NOT test data.
    exclude_cols: columns to drop before training (e.g. ['follower_count_log']).
    test data is NOT a parameter — isolation guaranteed by design.
    """
    if exclude_cols:
        X_train = X_train.drop(columns=[c for c in exclude_cols if c in X_train.columns])

    cat_cols = [c for c in CAT_COLS if c in X_train.columns]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    # Fit OrdinalEncoder once on full X_train (no stat leakage — only category names)
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=np.nan
        ), cat_cols),
    ], remainder="passthrough")
    X_enc = pre.fit_transform(X_train)
    y_arr = y_train.values.astype(float)

    cv_splits = make_cv_splits(author_ids_train, n_splits=n_splits, seed=seed)
    rng = np.random.RandomState(seed)

    best_cv_rmse = np.inf
    best_params: dict = {}
    best_n_rounds = 300

    for i in range(n_iter):
        params = {
            "max_depth": int(rng.choice([3, 4, 5, 6])),
            "learning_rate": float(rng.choice([0.05, 0.07, 0.1, 0.15, 0.2])),
            "subsample": float(round(rng.uniform(0.6, 1.0), 2)),
            "colsample_bytree": float(round(rng.uniform(0.5, 1.0), 2)),
            "reg_alpha": float(rng.choice([0.0, 0.1, 0.5, 1.0])),
            "reg_lambda": float(round(rng.uniform(0.5, 2.0), 2)),
            "min_child_weight": int(rng.choice([1, 3, 5, 10])),
        }

        fold_rmses: list[float] = []
        fold_rounds: list[int] = []

        for train_idx, val_idx in cv_splits:
            X_tr, X_val = X_enc[train_idx], X_enc[val_idx]
            y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

            model = XGBRegressor(
                **params,
                n_estimators=1500,
                tree_method="hist",
                early_stopping_rounds=50,
                eval_metric="rmse",
                random_state=seed,
            )
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict(X_val)
            fold_rmses.append(float(np.sqrt(mean_squared_error(y_val, preds))))
            fold_rounds.append(int(model.best_iteration))

        cv_rmse = float(np.mean(fold_rmses))
        mean_rounds = int(np.mean(fold_rounds))
        logger.info(
            "XGB iter %d/%d  RMSE=%.4f  best_rounds=%d  params=%s",
            i + 1, n_iter, cv_rmse, mean_rounds,
            {k: v for k, v in params.items() if k in ("max_depth", "learning_rate")},
        )

        if cv_rmse < best_cv_rmse:
            best_cv_rmse = cv_rmse
            best_params = params.copy()
            best_n_rounds = mean_rounds + 50  # small buffer

    logger.info("Best XGB CV RMSE=%.4f  n_estimators=%d", best_cv_rmse, best_n_rounds)

    # Retrain final model on full train with best params
    final_model = XGBRegressor(
        **best_params,
        n_estimators=best_n_rounds,
        tree_method="hist",
        random_state=seed,
    )
    final_model.fit(X_enc, y_arr)

    model_name = "XGBoost" + ("_no_follower" if exclude_cols else "_with_follower")
    return TrainedModel(
        preprocessor=pre,
        model=final_model,
        cv_rmse=best_cv_rmse,
        params={**best_params, "n_estimators": best_n_rounds},
        feature_names=list(X_train.columns),
        model_name=model_name,
    )


# ================================================================== #
# PER-CREATOR MODELS                                                  #
# ================================================================== #
#
# One conservative config is found via CV over ~100 sampled creators,
# then applied identically to every creator.
# ================================================================== #

# Conservative candidate configs for the per-creator grid search.
# n_estimators=100 for speed during search; same value used for final fit.
_PC_RF_GRID: list[dict] = [
    {"n_estimators": 100, "max_depth": 3, "min_samples_leaf": 5, "max_features": 0.5},
    {"n_estimators": 100, "max_depth": 3, "min_samples_leaf": 10, "max_features": 0.5},
    {"n_estimators": 100, "max_depth": 4, "min_samples_leaf": 5, "max_features": 0.4},
    {"n_estimators": 100, "max_depth": 4, "min_samples_leaf": 10, "max_features": 0.4},
    {"n_estimators": 100, "max_depth": 3, "min_samples_leaf": 3, "max_features": 0.6},
    {"n_estimators": 100, "max_depth": 5, "min_samples_leaf": 10, "max_features": 0.4},
    {"n_estimators": 100, "max_depth": 4, "min_samples_leaf": 5, "max_features": 0.6},
    {"n_estimators": 100, "max_depth": 3, "min_samples_leaf": 7, "max_features": 0.5},
]

_PC_XGB_GRID: list[dict] = [
    {"n_estimators": 100, "max_depth": 3, "min_child_weight": 10, "reg_alpha": 0.5, "reg_lambda": 2.0, "subsample": 0.8, "colsample_bytree": 0.6, "learning_rate": 0.05},
    {"n_estimators": 100, "max_depth": 2, "min_child_weight": 5,  "reg_alpha": 0.1, "reg_lambda": 1.5, "subsample": 0.8, "colsample_bytree": 0.7, "learning_rate": 0.05},
    {"n_estimators": 100, "max_depth": 3, "min_child_weight": 5,  "reg_alpha": 1.0, "reg_lambda": 2.0, "subsample": 0.7, "colsample_bytree": 0.6, "learning_rate": 0.05},
    {"n_estimators": 100, "max_depth": 2, "min_child_weight": 10, "reg_alpha": 1.0, "reg_lambda": 1.0, "subsample": 0.8, "colsample_bytree": 0.5, "learning_rate": 0.07},
    {"n_estimators": 100, "max_depth": 3, "min_child_weight": 3,  "reg_alpha": 0.5, "reg_lambda": 1.5, "subsample": 0.9, "colsample_bytree": 0.7, "learning_rate": 0.07},
    {"n_estimators": 100, "max_depth": 4, "min_child_weight": 5,  "reg_alpha": 0.5, "reg_lambda": 2.0, "subsample": 0.7, "colsample_bytree": 0.5, "learning_rate": 0.05},
    {"n_estimators": 100, "max_depth": 2, "min_child_weight": 3,  "reg_alpha": 0.1, "reg_lambda": 1.0, "subsample": 0.8, "colsample_bytree": 0.8, "learning_rate": 0.10},
    {"n_estimators": 100, "max_depth": 3, "min_child_weight": 10, "reg_alpha": 0.0, "reg_lambda": 2.0, "subsample": 0.9, "colsample_bytree": 0.6, "learning_rate": 0.07},
]


# ------------------------------------------------------------------ #
# CreatorModel — per-creator counterpart of TrainedModel              #
# ------------------------------------------------------------------ #

@dataclass
class CreatorModel:
    """Single-creator model: Pipeline fitted only on that creator's train videos.

    worse_than_zero is True when the CV RMSE exceeds the predict-0 RMSE on the
    creator's train set — useful for analysis, predictions are saved regardless.
    """
    author_id: str
    model: Any          # fitted Pipeline (preprocessor + estimator)
    feature_names: list[str]    # non-constant features for this creator
    cv_rmse: float
    model_name: str     # "RF" or "XGB"
    n_train: int
    n_features: int
    worse_than_zero: bool

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.feature_names])

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "CreatorModel":
        return joblib.load(path)


# ------------------------------------------------------------------ #
# Helpers for per-creator preprocessing                                #
# ------------------------------------------------------------------ #

def drop_constant_features(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns with ≤ 1 unique non-null value in X.

    Returns (X_clean, dropped_col_names).
    """
    dropped = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    return X.drop(columns=dropped), dropped


def _make_creator_pipe(X_ref: pd.DataFrame, model_type: str, params: dict, seed: int) -> Pipeline:
    """Build an unfitted Pipeline for per-creator training.

    X_ref is used only to identify which columns are categorical.
    """
    cat_in = [c for c in CAT_COLS if c in X_ref.columns]
    num_in = [c for c in X_ref.columns if c not in cat_in]

    transformers: list = []
    if cat_in:
        unknown_val = -1 if model_type == "rf" else np.nan
        transformers.append((
            "cat",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=unknown_val),
            cat_in,
        ))
    if num_in:
        if model_type == "rf":
            transformers.append(("num", SimpleImputer(strategy="median"), num_in))
        else:
            transformers.append(("num", "passthrough", num_in))

    ct = ColumnTransformer(transformers, remainder="drop") if transformers else ColumnTransformer([], remainder="passthrough")

    if model_type == "rf":
        mdl = RandomForestRegressor(n_jobs=1, random_state=seed, **params)
    else:
        mdl = XGBRegressor(tree_method="hist", random_state=seed, **params)

    return Pipeline([("pre", ct), ("mdl", mdl)])


# ------------------------------------------------------------------ #
# CV RMSE for a single creator with a single config                    #
# ------------------------------------------------------------------ #

def _creator_cv_rmse(
    X_c: pd.DataFrame,
    y_c: pd.Series,
    model_type: str,
    params: dict,
    seed: int,
) -> float:
    """K-fold CV RMSE on one creator.  Returns nan if too few samples."""
    n = len(X_c)
    k = min(5, max(2, n // 10))
    if n < 6:
        return float("nan")
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    fold_rmses: list[float] = []
    for tr_idx, val_idx in kf.split(X_c):
        X_tr, X_val = X_c.iloc[tr_idx], X_c.iloc[val_idx]
        y_tr, y_val = y_c.iloc[tr_idx], y_c.iloc[val_idx]
        try:
            pipe = _make_creator_pipe(X_tr, model_type, params, seed)
            pipe.fit(X_tr, y_tr.values)
            preds = pipe.predict(X_val)
            fold_rmses.append(float(np.sqrt(mean_squared_error(y_val.values, preds))))
        except Exception:
            continue
    return float(np.mean(fold_rmses)) if fold_rmses else float("nan")


# ------------------------------------------------------------------ #
# Config search over sampled creators                                  #
# ------------------------------------------------------------------ #

def _eval_one_creator_configs(
    X_c: pd.DataFrame,
    y_c: pd.Series,
    seed: int,
) -> tuple[list[float], list[float]]:
    """Evaluate all grid configs on one creator. Returns (rf_rmses, xgb_rmses)."""
    X_clean, _ = drop_constant_features(X_c)
    if len(X_clean.columns) == 0:
        return [float("nan")] * len(_PC_RF_GRID), [float("nan")] * len(_PC_XGB_GRID)
    rf_r = [_creator_cv_rmse(X_clean, y_c, "rf", p, seed) for p in _PC_RF_GRID]
    xgb_r = [_creator_cv_rmse(X_clean, y_c, "xgb", p, seed) for p in _PC_XGB_GRID]
    return rf_r, xgb_r


def search_per_creator_config(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    author_ids: pd.Series,
    sample_n: int = 100,
    seed: int = 42,
    n_jobs: int = -1,
) -> tuple[dict, dict]:
    """Find the best conservative RF and XGBoost configs by averaging CV RMSE
    over `sample_n` randomly sampled creators of varying sizes.

    Returns (best_rf_config, best_xgb_config).
    Test data is NOT a parameter — no leakage by design.
    """
    rng = np.random.RandomState(seed)
    all_authors = author_ids.unique()
    sampled = rng.choice(all_authors, size=min(sample_n, len(all_authors)), replace=False)
    logger.info("Config search: evaluating %d configs × %d creators", len(_PC_RF_GRID) + len(_PC_XGB_GRID), len(sampled))

    def _get_creator_data(a):
        mask = author_ids == a
        return (
            X_train[mask].reset_index(drop=True),
            y_train[mask].reset_index(drop=True),
        )

    results = joblib.Parallel(n_jobs=n_jobs, verbose=0)(
        joblib.delayed(_eval_one_creator_configs)(*_get_creator_data(a), seed)
        for a in sampled
    )

    rf_sums = np.zeros(len(_PC_RF_GRID))
    rf_counts = np.zeros(len(_PC_RF_GRID))
    xgb_sums = np.zeros(len(_PC_XGB_GRID))
    xgb_counts = np.zeros(len(_PC_XGB_GRID))

    for rf_r, xgb_r in results:
        for i, v in enumerate(rf_r):
            if not np.isnan(v):
                rf_sums[i] += v
                rf_counts[i] += 1
        for i, v in enumerate(xgb_r):
            if not np.isnan(v):
                xgb_sums[i] += v
                xgb_counts[i] += 1

    rf_means = np.where(rf_counts > 0, rf_sums / rf_counts, np.inf)
    xgb_means = np.where(xgb_counts > 0, xgb_sums / xgb_counts, np.inf)

    best_rf = _PC_RF_GRID[int(np.argmin(rf_means))].copy()
    best_xgb = _PC_XGB_GRID[int(np.argmin(xgb_means))].copy()

    logger.info(
        "Best RF config (mean CV RMSE=%.4f): %s", float(rf_means.min()),
        {k: v for k, v in best_rf.items() if k != "n_estimators"},
    )
    logger.info(
        "Best XGB config (mean CV RMSE=%.4f): %s", float(xgb_means.min()),
        {k: v for k, v in best_xgb.items() if k not in ("n_estimators", "subsample", "colsample_bytree")},
    )
    return best_rf, best_xgb


# ------------------------------------------------------------------ #
# Training a single creator                                            #
# ------------------------------------------------------------------ #

def _train_one_creator(
    author_id: str,
    X_c: pd.DataFrame,
    y_c: pd.Series,
    rf_config: dict,
    xgb_config: dict,
    seed: int,
) -> tuple["CreatorModel", "CreatorModel"]:
    """Train RF and XGB models for one creator.  No test data parameter.

    Returns (rf_creator_model, xgb_creator_model).
    """
    X_clean, _ = drop_constant_features(X_c)
    n_train = len(X_clean)
    n_features = len(X_clean.columns)

    trivial_rmse = float(np.sqrt(mean_squared_error(y_c.values, np.zeros(n_train))))

    models_out: list[CreatorModel] = []
    for model_type, cfg in [("rf", rf_config), ("xgb", xgb_config)]:
        cv_r = _creator_cv_rmse(X_clean, y_c, model_type, cfg, seed)
        pipe = _make_creator_pipe(X_clean, model_type, cfg, seed)
        pipe.fit(X_clean, y_c.values)
        worse = (not np.isnan(cv_r)) and (cv_r > trivial_rmse)
        models_out.append(CreatorModel(
            author_id=author_id,
            model=pipe,
            feature_names=list(X_clean.columns),
            cv_rmse=cv_r,
            model_name="RF" if model_type == "rf" else "XGB",
            n_train=n_train,
            n_features=n_features,
            worse_than_zero=worse,
        ))
    return models_out[0], models_out[1]


# ------------------------------------------------------------------ #
# Train all creators in parallel                                        #
# ------------------------------------------------------------------ #

def train_per_creator_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    author_ids: pd.Series,
    rf_config: dict,
    xgb_config: dict,
    n_jobs: int = -1,
    seed: int = 42,
) -> tuple[dict[str, "CreatorModel"], dict[str, "CreatorModel"]]:
    """Train one RF model and one XGB model per creator in parallel.

    Test data is NOT a parameter — isolation by design.
    Returns (rf_models, xgb_models), both dicts keyed by author_id.
    """
    unique_authors = list(author_ids.unique())
    logger.info("Training per-creator models: %d creators, n_jobs=%d", len(unique_authors), n_jobs)

    def _do_one(a):
        mask = author_ids == a
        X_c = X_train[mask].reset_index(drop=True)
        y_c = y_train[mask].reset_index(drop=True)
        return _train_one_creator(a, X_c, y_c, rf_config, xgb_config, seed)

    pairs = joblib.Parallel(n_jobs=n_jobs, verbose=5)(
        joblib.delayed(_do_one)(a) for a in unique_authors
    )

    rf_models: dict[str, CreatorModel] = {}
    xgb_models: dict[str, CreatorModel] = {}
    for rf_m, xgb_m in pairs:
        rf_models[rf_m.author_id] = rf_m
        xgb_models[xgb_m.author_id] = xgb_m

    logger.info(
        "Done. RF worse_than_zero: %d/%d  XGB: %d/%d",
        sum(m.worse_than_zero for m in rf_models.values()), len(rf_models),
        sum(m.worse_than_zero for m in xgb_models.values()), len(xgb_models),
    )
    return rf_models, xgb_models
