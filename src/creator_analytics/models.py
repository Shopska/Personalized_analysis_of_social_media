"""Model training — global and per-creator, families A–C7.

Design constraints
------------------
- search_* functions NEVER receive X_test / y_test — test isolation by signature.
- CV uses StratifiedKFold by author_id so each creator appears in both folds.
- XGBoost / LightGBM use early stopping on the CV val fold, not on test data.
- Per-creator models: *_vs_creator features are dropped before training — within
  a single creator these equal the base feature plus a per-creator constant and
  carry no additional information.
- B1 (Global RF): hyperparameter search on a 50k subsample with 3 folds;
  the final model is retrained on the full train set using the best params.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

META_COLS: list[str] = [
    "video_id", "author_id", "split", "create_dt",
    "target_abs", "target_rel", "creator_baseline",
]
VS_CREATOR_SUFFIX = "_vs_creator"


# ------------------------------------------------------------------ #
# Data preparation                                                     #
# ------------------------------------------------------------------ #

def prepare_X(
    df: pd.DataFrame,
    drop_vs_creator: bool = False,
) -> pd.DataFrame:
    """Drop meta columns; optionally drop *_vs_creator features.

    All remaining columns are numeric (float/int) — the new pipeline
    one-hot encodes categoricals and imputes NaN during feature engineering.

    drop_vs_creator=True for per-creator models: within one creator the
    *_vs_creator features are (base_feature - creator_constant), i.e. a
    shifted copy of the base feature that adds no information.
    """
    drop = [c for c in META_COLS if c in df.columns]
    X = df.drop(columns=drop)
    if drop_vs_creator:
        vs_cols = [c for c in X.columns if c.endswith(VS_CREATOR_SUFFIX)]
        X = X.drop(columns=vs_cols)
    return X.copy()


# ------------------------------------------------------------------ #
# CV splits                                                            #
# ------------------------------------------------------------------ #

def make_cv_splits(
    author_ids: pd.Series,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """StratifiedKFold by author_id → (train_idx, val_idx) list."""
    author_enc = LabelEncoder().fit_transform(author_ids)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.arange(len(author_ids)), author_enc))


# ------------------------------------------------------------------ #
# Model containers                                                     #
# ------------------------------------------------------------------ #

@dataclass
class TrainedModel:
    """A fitted model with optional preprocessing step and predict()."""
    model: Any               # sklearn Pipeline or raw estimator
    cv_rmse: float
    params: dict
    feature_names: list[str] = field(default_factory=list)
    model_name: str = ""
    preprocessor: Any = None  # kept for backward compat; None = already in model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xa = X[self.feature_names]
        if self.preprocessor is not None:
            Xa = self.preprocessor.transform(Xa)
            return self.model.predict(Xa)
        return self.model.predict(Xa)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "TrainedModel":
        return joblib.load(path)


@dataclass
class CreatorModel:
    """Single-creator model fitted only on that creator's train videos."""
    author_id: str
    model: Any              # fitted sklearn Pipeline
    feature_names: list[str]
    cv_rmse: float
    model_name: str         # "RF", "XGB", "LGBM", "Ridge", "ResidualXGB", …
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


@dataclass
class ShrinkageBlend:
    """C3: w·per_creator + (1-w)·global, w = n/(n+k)."""
    global_model: TrainedModel
    per_creator_models: dict[str, CreatorModel]
    k: float
    n_train_per_creator: dict[str, int]
    cv_rmse: float
    model_name: str = "ShrinkageBlend"

    def predict(
        self,
        X: pd.DataFrame,
        author_ids: pd.Series,
        X_global: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """X = per-creator features; X_global = full global features for global model.

        If X_global is None, X is used for both (only works if X contains all global features).
        """
        Xg = X_global if X_global is not None else X
        g_pred = self.global_model.predict(Xg)
        blended = g_pred.copy()
        for author, idx_vals in author_ids.groupby(author_ids).groups.items():
            if author not in self.per_creator_models:
                continue
            n = self.n_train_per_creator.get(author, 0)
            w = n / (n + self.k)
            pc_m = self.per_creator_models[author]
            positions = X.index.get_indexer(idx_vals)
            X_pc = X.loc[idx_vals, pc_m.feature_names]
            pc_pred = pc_m.model.predict(X_pc)
            blended[positions] = w * pc_pred + (1 - w) * g_pred[positions]
        return blended


@dataclass
class TwoStageModel:
    """C5: global model + per-creator model trained on residuals."""
    stage1: TrainedModel
    stage2: dict[str, CreatorModel]  # per-creator residual models
    model_name: str = "TwoStage"

    def predict(
        self,
        X: pd.DataFrame,
        author_ids: pd.Series,
        X_global: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """X = per-creator features; X_global = full global features for stage1.

        If X_global is None, X is used for stage1 (only works if X has all global features).
        """
        Xg = X_global if X_global is not None else X
        pred = self.stage1.predict(Xg).copy()
        for author, idx_vals in author_ids.groupby(author_ids).groups.items():
            if author not in self.stage2:
                continue
            m = self.stage2[author]
            positions = X.index.get_indexer(idx_vals)
            X_pc = X.loc[idx_vals, m.feature_names]
            pred[positions] += m.model.predict(X_pc)
        return pred


@dataclass
class ClusterModel:
    """C6: cluster creators, one model per cluster."""
    kmeans: Any             # fitted KMeans on creator embeddings
    cluster_models: dict[int, TrainedModel]
    creator_cluster: dict[str, int]  # author_id → cluster id
    model_name: str = "ClusterModel"
    feature_names: list[str] = field(default_factory=list)

    def predict(self, X: pd.DataFrame, author_ids: pd.Series) -> np.ndarray:
        global_fallback_cluster = 0  # fallback for unseen creators
        pred = np.zeros(len(X))
        for author, idx_vals in author_ids.groupby(author_ids).groups.items():
            cluster_id = self.creator_cluster.get(author, global_fallback_cluster)
            m = self.cluster_models[cluster_id]
            positions = X.index.get_indexer(idx_vals)
            pred[positions] = m.predict(X.loc[idx_vals])
        return pred


# ------------------------------------------------------------------ #
# Trivial baselines (A)                                                #
# ------------------------------------------------------------------ #

def trivial_zero_rmse(y_test: pd.Series) -> float:
    """RMSE of always predicting 0 (= creator train median on relative target)."""
    return float(np.sqrt(mean_squared_error(y_test, np.zeros(len(y_test)))))


def trivial_creator_mean(
    y_train: pd.Series,
    author_ids_train: pd.Series,
    y_test: pd.Series,
    author_ids_test: pd.Series,
) -> tuple[float, np.ndarray]:
    """RMSE of predicting per-creator train mean. Returns (rmse, predictions)."""
    train_means = y_train.groupby(author_ids_train).mean()
    global_mean = float(y_train.mean())
    y_pred = author_ids_test.map(train_means).fillna(global_mean).values
    return float(np.sqrt(mean_squared_error(y_test, y_pred))), y_pred


# ------------------------------------------------------------------ #
# B1: Global Random Forest                                             #
# ------------------------------------------------------------------ #

def search_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    author_ids_train: pd.Series,
    n_splits: int = 5,
    seed: int = 42,
    n_iter: int = 30,
    subsample_n: int | None = 50_000,
    n_folds_search: int | None = 3,
) -> TrainedModel:
    """RandomizedSearchCV for global RF; test data is NOT a parameter.

    Hyperparameter search is run on a random subsample (subsample_n rows, n_folds_search
    CV folds) for speed. The final model is refitted on the full X_train.
    """
    rng = np.random.RandomState(seed)

    if subsample_n is not None and len(X_train) > subsample_n:
        sub_idx = rng.choice(len(X_train), size=subsample_n, replace=False)
        X_search = X_train.iloc[sub_idx].reset_index(drop=True)
        y_search = y_train.iloc[sub_idx].reset_index(drop=True)
        auth_search = author_ids_train.iloc[sub_idx].reset_index(drop=True)
    else:
        X_search, y_search, auth_search = X_train, y_train, author_ids_train

    n_folds = n_folds_search if n_folds_search is not None else n_splits
    cv_splits_search = make_cv_splits(auth_search, n_splits=n_folds, seed=seed)

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(n_jobs=-1, random_state=seed)),
    ])
    param_dist = {
        "rf__n_estimators": [100, 200, 300],
        "rf__max_depth": [None, 8, 15, 25],
        "rf__min_samples_split": [2, 5, 10, 20],
        "rf__min_samples_leaf": [1, 2, 5],
        "rf__max_features": [0.3, 0.5, 0.7, "sqrt"],
    }

    search = RandomizedSearchCV(
        pipe, param_dist,
        n_iter=n_iter,
        cv=cv_splits_search,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
        random_state=seed,
        refit=False,
        verbose=1,
    )
    search.fit(X_search, y_search)
    best_params = search.best_params_
    best_cv_rmse = float(-search.best_score_)

    final_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(
            n_jobs=-1,
            random_state=seed,
            **{k.replace("rf__", ""): v for k, v in best_params.items()},
        )),
    ])
    final_pipe.fit(X_train, y_train)

    return TrainedModel(
        model=final_pipe,
        cv_rmse=best_cv_rmse,
        params={k.replace("rf__", ""): v for k, v in best_params.items()},
        feature_names=list(X_train.columns),
        model_name="RandomForest",
    )


# ------------------------------------------------------------------ #
# B2: Global XGBoost                                                   #
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
    """Random search for global XGBoost with per-fold early stopping.

    Early stopping uses the CV val fold — NOT test data.
    """
    if exclude_cols:
        X_train = X_train.drop(columns=[c for c in exclude_cols if c in X_train.columns])

    X_arr = X_train.values.astype(float)
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
        for tr_idx, val_idx in cv_splits:
            mdl = XGBRegressor(
                **params,
                n_estimators=1500,
                tree_method="hist",
                early_stopping_rounds=50,
                eval_metric="rmse",
                random_state=seed,
            )
            mdl.fit(
                X_arr[tr_idx], y_arr[tr_idx],
                eval_set=[(X_arr[val_idx], y_arr[val_idx])],
                verbose=False,
            )
            preds = mdl.predict(X_arr[val_idx])
            fold_rmses.append(float(np.sqrt(mean_squared_error(y_arr[val_idx], preds))))
            fold_rounds.append(int(mdl.best_iteration))

        cv_rmse = float(np.mean(fold_rmses))
        mean_rounds = int(np.mean(fold_rounds))
        logger.info(
            "XGB iter %d/%d  RMSE=%.4f  rounds=%d",
            i + 1, n_iter, cv_rmse, mean_rounds,
        )
        if cv_rmse < best_cv_rmse:
            best_cv_rmse = cv_rmse
            best_params = params.copy()
            best_n_rounds = mean_rounds + 50

    final_model = XGBRegressor(
        **best_params,
        n_estimators=best_n_rounds,
        tree_method="hist",
        random_state=seed,
    )
    final_model.fit(X_arr, y_arr)

    suffix = "_no_follower" if exclude_cols else "_with_follower"
    return TrainedModel(
        model=final_model,
        cv_rmse=best_cv_rmse,
        params={**best_params, "n_estimators": best_n_rounds},
        feature_names=list(X_train.columns),
        model_name="XGBoost" + suffix,
    )


# ================================================================== #
# PER-CREATOR HELPERS (B3, B4 and per-creator parts of C1, C2)        #
# ================================================================== #

_PC_RF_GRID: list[dict] = [
    {"n_estimators": 100, "max_depth": 3, "min_samples_leaf": 5,  "max_features": 0.5},
    {"n_estimators": 100, "max_depth": 3, "min_samples_leaf": 10, "max_features": 0.5},
    {"n_estimators": 100, "max_depth": 4, "min_samples_leaf": 5,  "max_features": 0.4},
    {"n_estimators": 100, "max_depth": 4, "min_samples_leaf": 10, "max_features": 0.4},
    {"n_estimators": 100, "max_depth": 3, "min_samples_leaf": 3,  "max_features": 0.6},
    {"n_estimators": 100, "max_depth": 5, "min_samples_leaf": 10, "max_features": 0.4},
    {"n_estimators": 100, "max_depth": 4, "min_samples_leaf": 5,  "max_features": 0.6},
    {"n_estimators": 100, "max_depth": 3, "min_samples_leaf": 7,  "max_features": 0.5},
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


def drop_constant_features(
    X: pd.DataFrame,
    threshold: float = 0.95,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop features constant in ≥ threshold fraction of rows (≤1 unique non-null value).

    threshold=0.95 → used for global constant check.
    Use threshold=1.0 for strict constancy (per-creator use case).
    """
    dropped = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    return X.drop(columns=dropped), dropped


def _make_creator_pipe(
    X_ref: pd.DataFrame,
    model_type: str,
    params: dict,
    seed: int,
) -> Pipeline:
    """Build an unfitted Pipeline for per-creator training.

    All features are already numeric; RF uses median imputer, XGB/LGBM handle
    NaN natively (no imputer).
    """
    if model_type == "rf":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("mdl", RandomForestRegressor(n_jobs=1, random_state=seed, **params)),
        ])
    if model_type == "xgb":
        return Pipeline([
            ("mdl", XGBRegressor(tree_method="hist", random_state=seed, **params)),
        ])
    if model_type == "lgbm":
        from lightgbm import LGBMRegressor
        return Pipeline([
            ("mdl", LGBMRegressor(random_state=seed, verbose=-1, **params)),
        ])
    if model_type == "ridge":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("mdl", Ridge(**params)),
        ])
    raise ValueError(f"Unknown model_type: {model_type!r}")


def _creator_cv_rmse(
    X_c: pd.DataFrame,
    y_c: pd.Series,
    model_type: str,
    params: dict,
    seed: int,
) -> float:
    """K-fold RMSE on one creator; returns nan if too few samples."""
    n = len(X_c)
    if n < 6:
        return float("nan")
    k = min(5, max(2, n // 10))
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    fold_rmses: list[float] = []
    for tr_idx, val_idx in kf.split(X_c):
        try:
            pipe = _make_creator_pipe(X_c, model_type, params, seed)
            pipe.fit(X_c.iloc[tr_idx], y_c.iloc[tr_idx].values)
            preds = pipe.predict(X_c.iloc[val_idx])
            fold_rmses.append(float(np.sqrt(mean_squared_error(
                y_c.iloc[val_idx].values, preds))))
        except Exception:
            continue
    return float(np.mean(fold_rmses)) if fold_rmses else float("nan")


def _eval_one_creator_configs(
    X_c: pd.DataFrame,
    y_c: pd.Series,
    seed: int,
) -> tuple[list[float], list[float]]:
    """Evaluate all grid configs on one creator. Returns (rf_rmses, xgb_rmses)."""
    X_clean, _ = drop_constant_features(X_c, threshold=1.0)
    if len(X_clean.columns) == 0:
        return [float("nan")] * len(_PC_RF_GRID), [float("nan")] * len(_PC_XGB_GRID)
    rf_r  = [_creator_cv_rmse(X_clean, y_c, "rf",  p, seed) for p in _PC_RF_GRID]
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
    """Best conservative RF and XGB configs averaged over sample_n creators.

    X_train must already have *_vs_creator features dropped.
    Returns (best_rf_config, best_xgb_config).
    """
    rng = np.random.RandomState(seed)
    all_authors = author_ids.unique()
    sampled = rng.choice(all_authors, size=min(sample_n, len(all_authors)), replace=False)

    def _get(a):
        mask = author_ids == a
        return (
            X_train[mask].reset_index(drop=True),
            y_train[mask].reset_index(drop=True),
        )

    results = joblib.Parallel(n_jobs=n_jobs, verbose=0)(
        joblib.delayed(_eval_one_creator_configs)(*_get(a), seed)
        for a in sampled
    )

    rf_sums  = np.zeros(len(_PC_RF_GRID))
    rf_cnt   = np.zeros(len(_PC_RF_GRID))
    xgb_sums = np.zeros(len(_PC_XGB_GRID))
    xgb_cnt  = np.zeros(len(_PC_XGB_GRID))

    for rf_r, xgb_r in results:
        for i, v in enumerate(rf_r):
            if not np.isnan(v):
                rf_sums[i] += v; rf_cnt[i] += 1
        for i, v in enumerate(xgb_r):
            if not np.isnan(v):
                xgb_sums[i] += v; xgb_cnt[i] += 1

    rf_means  = np.where(rf_cnt  > 0, rf_sums  / rf_cnt,  np.inf)
    xgb_means = np.where(xgb_cnt > 0, xgb_sums / xgb_cnt, np.inf)

    best_rf  = _PC_RF_GRID[int(np.argmin(rf_means))].copy()
    best_xgb = _PC_XGB_GRID[int(np.argmin(xgb_means))].copy()

    logger.info("Best RF CV RMSE=%.4f  config=%s", rf_means.min(), best_rf)
    logger.info("Best XGB CV RMSE=%.4f  config=%s", xgb_means.min(), best_xgb)
    return best_rf, best_xgb


def _train_one_creator(
    author_id: str,
    X_c: pd.DataFrame,
    y_c: pd.Series,
    rf_config: dict,
    xgb_config: dict,
    seed: int,
) -> tuple[CreatorModel, CreatorModel]:
    """Train RF and XGB for one creator. No test data parameter."""
    X_clean, _ = drop_constant_features(X_c, threshold=1.0)
    n_train = len(X_clean)
    trivial = float(np.sqrt(mean_squared_error(y_c.values, np.zeros(n_train))))

    out: list[CreatorModel] = []
    for mtype, cfg in [("rf", rf_config), ("xgb", xgb_config)]:
        cv_r = _creator_cv_rmse(X_clean, y_c, mtype, cfg, seed)
        pipe = _make_creator_pipe(X_clean, mtype, cfg, seed)
        pipe.fit(X_clean, y_c.values)
        out.append(CreatorModel(
            author_id=author_id,
            model=pipe,
            feature_names=list(X_clean.columns),
            cv_rmse=cv_r,
            model_name="RF" if mtype == "rf" else "XGB",
            n_train=n_train,
            n_features=len(X_clean.columns),
            worse_than_zero=(not np.isnan(cv_r)) and (cv_r > trivial),
        ))
    return out[0], out[1]


def train_per_creator_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    author_ids: pd.Series,
    rf_config: dict,
    xgb_config: dict,
    n_jobs: int = -1,
    seed: int = 42,
) -> tuple[dict[str, CreatorModel], dict[str, CreatorModel]]:
    """Train one RF + one XGB per creator in parallel.

    X_train must already have *_vs_creator features dropped.
    Returns (rf_models, xgb_models) keyed by author_id.
    """
    authors = list(author_ids.unique())

    def _do(a):
        mask = author_ids == a
        return _train_one_creator(
            a,
            X_train[mask].reset_index(drop=True),
            y_train[mask].reset_index(drop=True),
            rf_config, xgb_config, seed,
        )

    pairs = joblib.Parallel(n_jobs=n_jobs, verbose=5)(
        joblib.delayed(_do)(a) for a in authors
    )

    rf_dict: dict[str, CreatorModel] = {}
    xgb_dict: dict[str, CreatorModel] = {}
    for rf_m, xgb_m in pairs:
        rf_dict[rf_m.author_id] = rf_m
        xgb_dict[xgb_m.author_id] = xgb_m

    logger.info(
        "RF worse_than_zero: %d/%d  XGB: %d/%d",
        sum(m.worse_than_zero for m in rf_dict.values()), len(rf_dict),
        sum(m.worse_than_zero for m in xgb_dict.values()), len(xgb_dict),
    )
    return rf_dict, xgb_dict


# ================================================================== #
# C1: LightGBM (global + per-creator)                                 #
# ================================================================== #

def search_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    author_ids_train: pd.Series,
    n_splits: int = 5,
    seed: int = 42,
    n_iter: int = 30,
) -> TrainedModel:
    """Random search for global LightGBM with per-fold early stopping."""
    from lightgbm import LGBMRegressor

    X_arr = X_train.values.astype(float)
    y_arr = y_train.values.astype(float)
    cv_splits = make_cv_splits(author_ids_train, n_splits=n_splits, seed=seed)
    rng = np.random.RandomState(seed)

    best_cv_rmse = np.inf
    best_params: dict = {}
    best_n_rounds = 300

    for i in range(n_iter):
        params = {
            "num_leaves":    int(rng.choice([31, 63, 127, 255])),
            "max_depth":     int(rng.choice([-1, 4, 6, 8])),
            "learning_rate": float(rng.choice([0.05, 0.07, 0.1, 0.15])),
            "subsample":     float(round(rng.uniform(0.6, 1.0), 2)),
            "colsample_bytree": float(round(rng.uniform(0.5, 1.0), 2)),
            "reg_alpha":     float(rng.choice([0.0, 0.1, 0.5, 1.0])),
            "reg_lambda":    float(round(rng.uniform(0.5, 2.0), 2)),
            "min_child_samples": int(rng.choice([5, 10, 20, 50])),
        }
        fold_rmses: list[float] = []
        fold_rounds: list[int] = []
        for tr_idx, val_idx in cv_splits:
            mdl = LGBMRegressor(
                **params,
                n_estimators=1500,
                random_state=seed,
                verbose=-1,
            )
            mdl.fit(
                X_arr[tr_idx], y_arr[tr_idx],
                eval_set=[(X_arr[val_idx], y_arr[val_idx])],
                callbacks=[
                    __import__("lightgbm").early_stopping(50, verbose=False),
                    __import__("lightgbm").log_evaluation(period=-1),
                ],
            )
            preds = mdl.predict(X_arr[val_idx])
            fold_rmses.append(float(np.sqrt(mean_squared_error(y_arr[val_idx], preds))))
            fold_rounds.append(int(mdl.best_iteration_))

        cv_rmse = float(np.mean(fold_rmses))
        mean_rounds = int(np.mean(fold_rounds))
        logger.info("LGBM iter %d/%d  RMSE=%.4f  rounds=%d", i + 1, n_iter, cv_rmse, mean_rounds)
        if cv_rmse < best_cv_rmse:
            best_cv_rmse = cv_rmse
            best_params = params.copy()
            best_n_rounds = mean_rounds + 50

    final_model = LGBMRegressor(
        **best_params,
        n_estimators=best_n_rounds,
        random_state=seed,
        verbose=-1,
    )
    final_model.fit(X_arr, y_arr)
    return TrainedModel(
        model=final_model,
        cv_rmse=best_cv_rmse,
        params={**best_params, "n_estimators": best_n_rounds},
        feature_names=list(X_train.columns),
        model_name="LightGBM",
    )


def _train_one_creator_lgbm(
    author_id: str,
    X_c: pd.DataFrame,
    y_c: pd.Series,
    params: dict,
    seed: int,
) -> CreatorModel:
    from lightgbm import LGBMRegressor
    X_clean, _ = drop_constant_features(X_c, threshold=1.0)
    n_train = len(X_clean)
    trivial = float(np.sqrt(mean_squared_error(y_c.values, np.zeros(n_train))))
    cv_r = _creator_cv_rmse(X_clean, y_c, "lgbm", params, seed)
    pipe = _make_creator_pipe(X_clean, "lgbm", params, seed)
    pipe.fit(X_clean, y_c.values)
    return CreatorModel(
        author_id=author_id,
        model=pipe,
        feature_names=list(X_clean.columns),
        cv_rmse=cv_r,
        model_name="LGBM",
        n_train=n_train,
        n_features=len(X_clean.columns),
        worse_than_zero=(not np.isnan(cv_r)) and (cv_r > trivial),
    )


def train_per_creator_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    author_ids: pd.Series,
    params: dict,
    n_jobs: int = -1,
    seed: int = 42,
) -> dict[str, CreatorModel]:
    """Train one LightGBM per creator in parallel.

    X_train must already have *_vs_creator features dropped.
    """
    authors = list(author_ids.unique())

    def _do(a):
        mask = author_ids == a
        return _train_one_creator_lgbm(
            a,
            X_train[mask].reset_index(drop=True),
            y_train[mask].reset_index(drop=True),
            params, seed,
        )

    models = joblib.Parallel(n_jobs=n_jobs, verbose=5)(
        joblib.delayed(_do)(a) for a in authors
    )
    return {m.author_id: m for m in models}


# ================================================================== #
# C2: Ridge (global + per-creator)                                    #
# ================================================================== #

def train_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    author_ids_train: pd.Series,
    n_splits: int = 5,
    seed: int = 42,
    alphas: list[float] | None = None,
) -> TrainedModel:
    """Global Ridge with CV alpha selection."""
    from sklearn.linear_model import RidgeCV

    if alphas is None:
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    cv_splits = make_cv_splits(author_ids_train, n_splits=n_splits, seed=seed)

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("ridge", RidgeCV(alphas=alphas, cv=cv_splits, scoring="neg_mean_squared_error")),
    ])
    pipe.fit(X_train, y_train)

    ridge_cv = pipe.named_steps["ridge"]
    best_alpha = float(ridge_cv.alpha_)
    cv_rmse = float(np.sqrt(-ridge_cv.best_score_))

    return TrainedModel(
        model=pipe,
        cv_rmse=cv_rmse,
        params={"alpha": best_alpha},
        feature_names=list(X_train.columns),
        model_name="Ridge",
    )


def _train_one_creator_ridge(
    author_id: str,
    X_c: pd.DataFrame,
    y_c: pd.Series,
    seed: int,
) -> CreatorModel:
    X_clean, _ = drop_constant_features(X_c, threshold=1.0)
    n_train = len(X_clean)
    trivial = float(np.sqrt(mean_squared_error(y_c.values, np.zeros(n_train))))
    cv_r = _creator_cv_rmse(X_clean, y_c, "ridge", {"alpha": 1.0}, seed)
    pipe = _make_creator_pipe(X_clean, "ridge", {"alpha": 1.0}, seed)
    pipe.fit(X_clean, y_c.values)
    return CreatorModel(
        author_id=author_id,
        model=pipe,
        feature_names=list(X_clean.columns),
        cv_rmse=cv_r,
        model_name="Ridge",
        n_train=n_train,
        n_features=len(X_clean.columns),
        worse_than_zero=(not np.isnan(cv_r)) and (cv_r > trivial),
    )


def train_per_creator_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    author_ids: pd.Series,
    n_jobs: int = -1,
    seed: int = 42,
) -> dict[str, CreatorModel]:
    """Train one Ridge per creator in parallel.

    X_train must already have *_vs_creator features dropped.
    """
    authors = list(author_ids.unique())

    def _do(a):
        mask = author_ids == a
        return _train_one_creator_ridge(
            a,
            X_train[mask].reset_index(drop=True),
            y_train[mask].reset_index(drop=True),
            seed,
        )

    models = joblib.Parallel(n_jobs=n_jobs, verbose=5)(
        joblib.delayed(_do)(a) for a in authors
    )
    return {m.author_id: m for m in models}


# ================================================================== #
# C3: Shrinkage blend                                                  #
# ================================================================== #

def find_shrinkage_k(
    per_creator_models: dict[str, CreatorModel],
    global_model: TrainedModel,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    author_ids_val: pd.Series,
    k_grid: list[float] | None = None,
    X_val_global: pd.DataFrame | None = None,
) -> float:
    """Find k that minimises blended RMSE on a validation set.

    X_val = per-creator features; X_val_global = global features for global model.
    w_i = n_train_i / (n_train_i + k); higher k → more shrinkage toward global.
    """
    if k_grid is None:
        k_grid = [1, 5, 10, 20, 50, 100, 200, 500, 1000, 5000]

    n_train_pc = {a: m.n_train for a, m in per_creator_models.items()}
    Xvg = X_val_global if X_val_global is not None else X_val
    g_pred = global_model.predict(Xvg)

    best_k, best_rmse = k_grid[0], np.inf
    for k in k_grid:
        blended = g_pred.copy()
        for author, idx_vals in author_ids_val.groupby(author_ids_val).groups.items():
            if author not in per_creator_models:
                continue
            n = n_train_pc.get(author, 0)
            w = n / (n + k)
            m = per_creator_models[author]
            positions = X_val.index.get_indexer(idx_vals)
            X_pc = X_val.loc[idx_vals, m.feature_names]
            pc_pred = m.model.predict(X_pc)
            blended[positions] = w * pc_pred + (1 - w) * g_pred[positions]

        rmse_k = float(np.sqrt(mean_squared_error(y_val.values, blended)))
        logger.info("Shrinkage k=%-6g  RMSE=%.4f", k, rmse_k)
        if rmse_k < best_rmse:
            best_rmse = rmse_k
            best_k = k

    logger.info("Best shrinkage k=%g  RMSE=%.4f", best_k, best_rmse)
    return float(best_k)


def train_shrinkage_blend(
    per_creator_models: dict[str, CreatorModel],
    global_model: TrainedModel,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    author_ids_train: pd.Series,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    X_train_global: pd.DataFrame | None = None,
) -> ShrinkageBlend:
    """Find k by CV on train, return a ShrinkageBlend using all train per-creator models.

    X_train = per-creator features; X_train_global = full global features for global model.
    """
    from collections import Counter
    k_per_fold: list[float] = []

    for tr_idx, val_idx in cv_splits:
        X_tr  = X_train.iloc[tr_idx].reset_index(drop=True)
        y_tr  = y_train.iloc[tr_idx].reset_index(drop=True)
        auth_tr = author_ids_train.iloc[tr_idx].reset_index(drop=True)
        X_val  = X_train.iloc[val_idx]
        y_val  = y_train.iloc[val_idx]
        auth_val = author_ids_train.iloc[val_idx]
        X_val_global = X_train_global.iloc[val_idx] if X_train_global is not None else None

        # Build fold-level per-creator models (fast proxy — first grid config)
        fold_pc: dict[str, CreatorModel] = {}
        for a in auth_tr.unique():
            mask = auth_tr == a
            Xc = X_tr[mask].reset_index(drop=True)
            yc = y_tr[mask].reset_index(drop=True)
            Xc, _ = drop_constant_features(Xc, threshold=1.0)
            if len(Xc) < 5:
                continue
            pipe = _make_creator_pipe(Xc, "xgb", _PC_XGB_GRID[0], seed=42)
            pipe.fit(Xc, yc.values)
            fold_pc[a] = CreatorModel(
                author_id=a, model=pipe, feature_names=list(Xc.columns),
                cv_rmse=float("nan"), model_name="XGB_fold",
                n_train=len(Xc), n_features=len(Xc.columns), worse_than_zero=False,
            )

        k_val = find_shrinkage_k(
            fold_pc, global_model, X_val, y_val, auth_val,
            X_val_global=X_val_global,
        )
        k_per_fold.append(k_val)

    # Majority vote: use the k chosen most often across folds
    best_k = float(Counter(k_per_fold).most_common(1)[0][0])

    n_train_pc = {a: m.n_train for a, m in per_creator_models.items()}
    Xg = X_train_global if X_train_global is not None else X_train
    g_pred = global_model.predict(Xg)
    blended = g_pred.copy()
    for author, idx_vals in author_ids_train.groupby(author_ids_train).groups.items():
        if author not in per_creator_models:
            continue
        n = n_train_pc.get(author, 0)
        w = n / (n + best_k)
        m = per_creator_models[author]
        positions = X_train.index.get_indexer(idx_vals)
        X_pc = X_train.loc[idx_vals, m.feature_names]
        pc_pred = m.model.predict(X_pc)
        blended[positions] = w * pc_pred + (1 - w) * g_pred[positions]

    cv_rmse = float(np.sqrt(mean_squared_error(y_train.values, blended)))
    return ShrinkageBlend(
        global_model=global_model,
        per_creator_models=per_creator_models,
        k=best_k,
        n_train_per_creator=n_train_pc,
        cv_rmse=cv_rmse,
    )


# ================================================================== #
# C4: Global → per-creator fine-tuning                                #
# ================================================================== #

def train_finetune_per_creator(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    author_ids: pd.Series,
    global_model: TrainedModel,
    model_type: str = "xgb",
    n_jobs: int = -1,
    seed: int = 42,
) -> dict[str, CreatorModel]:
    """Fine-tune a global XGB or LGBM model on each creator's data.

    XGB: pass xgb_model=global_model.model to continue training.
    LGBM: pass init_model=global_model.model to continue training.
    X_train must already have *_vs_creator features dropped.
    """
    authors = list(author_ids.unique())
    global_estimator = global_model.model  # raw XGBRegressor or LGBMRegressor
    global_feature_names = global_model.feature_names

    def _do_xgb(a):
        mask = author_ids == a
        Xc = X_train[mask][global_feature_names].reset_index(drop=True)
        yc = y_train[mask].reset_index(drop=True)
        Xc, dropped = drop_constant_features(Xc, threshold=1.0)
        if len(Xc) < 5:
            return None
        fine_tuned = XGBRegressor(
            n_estimators=50,
            learning_rate=0.05,
            max_depth=3,
            tree_method="hist",
            random_state=seed,
        )
        fine_tuned.fit(Xc, yc.values, xgb_model=global_estimator)
        trivial = float(np.sqrt(mean_squared_error(yc.values, np.zeros(len(yc)))))
        preds = fine_tuned.predict(Xc)
        cv_r = float(np.sqrt(mean_squared_error(yc.values, preds)))
        return CreatorModel(
            author_id=a, model=fine_tuned,
            feature_names=list(Xc.columns),
            cv_rmse=cv_r, model_name="XGB_finetune",
            n_train=len(Xc), n_features=len(Xc.columns),
            worse_than_zero=cv_r > trivial,
        )

    def _do_lgbm(a):
        from lightgbm import LGBMRegressor
        mask = author_ids == a
        Xc = X_train[mask][global_feature_names].reset_index(drop=True)
        yc = y_train[mask].reset_index(drop=True)
        Xc, _ = drop_constant_features(Xc, threshold=1.0)
        if len(Xc) < 5:
            return None
        fine_tuned = LGBMRegressor(
            n_estimators=50,
            learning_rate=0.05,
            num_leaves=31,
            random_state=seed,
            verbose=-1,
        )
        fine_tuned.fit(Xc, yc.values, init_model=global_estimator)
        trivial = float(np.sqrt(mean_squared_error(yc.values, np.zeros(len(yc)))))
        preds = fine_tuned.predict(Xc)
        cv_r = float(np.sqrt(mean_squared_error(yc.values, preds)))
        return CreatorModel(
            author_id=a, model=fine_tuned,
            feature_names=list(Xc.columns),
            cv_rmse=cv_r, model_name="LGBM_finetune",
            n_train=len(Xc), n_features=len(Xc.columns),
            worse_than_zero=cv_r > trivial,
        )

    _do = _do_xgb if model_type == "xgb" else _do_lgbm
    results = joblib.Parallel(n_jobs=n_jobs, verbose=5)(
        joblib.delayed(_do)(a) for a in authors
    )
    return {m.author_id: m for m in results if m is not None}


# ================================================================== #
# C5: Two-stage (global + per-creator residuals)                      #
# ================================================================== #

def train_two_stage(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    author_ids: pd.Series,
    global_model: TrainedModel,
    seed: int = 42,
    n_jobs: int = -1,
    X_train_global: pd.DataFrame | None = None,
) -> TwoStageModel:
    """Stage 1 = global model predictions; stage 2 = per-creator on residuals.

    X_train = per-creator features (no *_vs_creator).
    X_train_global = full global features for stage-1 predictions; if None, uses X_train.
    """
    Xg = X_train_global if X_train_global is not None else X_train
    residuals = pd.Series(
        y_train.values - global_model.predict(Xg),
        index=y_train.index,
        name="residual",
    )

    residual_xgb_config = {
        "n_estimators": 100, "max_depth": 2, "min_child_weight": 10,
        "reg_alpha": 1.0, "reg_lambda": 2.0,
        "subsample": 0.8, "colsample_bytree": 0.5, "learning_rate": 0.05,
    }
    authors = list(author_ids.unique())

    def _do(a):
        mask = author_ids == a
        Xc = X_train[mask].reset_index(drop=True)
        rc = residuals[mask].reset_index(drop=True)
        Xc, _ = drop_constant_features(Xc, threshold=1.0)
        if len(Xc) < 5:
            return None
        trivial = float(np.sqrt(mean_squared_error(rc.values, np.zeros(len(rc)))))
        cv_r = _creator_cv_rmse(Xc, rc, "xgb", residual_xgb_config, seed)
        pipe = _make_creator_pipe(Xc, "xgb", residual_xgb_config, seed)
        pipe.fit(Xc, rc.values)
        return CreatorModel(
            author_id=a, model=pipe, feature_names=list(Xc.columns),
            cv_rmse=cv_r, model_name="XGB_residual",
            n_train=len(Xc), n_features=len(Xc.columns),
            worse_than_zero=(not np.isnan(cv_r)) and (cv_r > trivial),
        )

    stage2 = joblib.Parallel(n_jobs=n_jobs, verbose=5)(
        joblib.delayed(_do)(a) for a in authors
    )
    stage2_dict = {m.author_id: m for m in stage2 if m is not None}
    return TwoStageModel(stage1=global_model, stage2=stage2_dict)


# ================================================================== #
# C6: Cluster models                                                   #
# ================================================================== #

def train_cluster_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    author_ids: pd.Series,
    n_clusters: int = 10,
    seed: int = 42,
    n_jobs: int = -1,
) -> ClusterModel:
    """Cluster creators by their mean feature vector; train one XGB per cluster."""
    creator_means = X_train.groupby(author_ids).mean()
    imp = SimpleImputer(strategy="median")
    creator_emb = imp.fit_transform(creator_means)

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    km.fit(creator_emb)

    creator_cluster = dict(zip(creator_means.index, km.labels_))
    cluster_ids = pd.Series(author_ids.map(creator_cluster), index=author_ids.index)

    cluster_models: dict[int, TrainedModel] = {}
    feature_names = list(X_train.columns)

    def _fit_cluster(cid):
        mask = cluster_ids == cid
        Xc = X_train[mask].reset_index(drop=True)
        yc = y_train[mask].reset_index(drop=True)
        auth_c = author_ids[mask].reset_index(drop=True)
        n_splits = min(5, auth_c.nunique())
        if n_splits < 2:
            n_splits = 2

        xgb_params = {
            "n_estimators": 200, "max_depth": 4, "learning_rate": 0.1,
            "subsample": 0.8, "colsample_bytree": 0.7,
            "reg_alpha": 0.1, "reg_lambda": 1.0, "min_child_weight": 5,
            "tree_method": "hist", "random_state": seed,
        }
        mdl = XGBRegressor(**xgb_params)
        mdl.fit(Xc.values.astype(float), yc.values.astype(float))
        return cid, TrainedModel(
            model=mdl, cv_rmse=float("nan"),
            params=xgb_params, feature_names=feature_names,
            model_name=f"ClusterXGB_{cid}",
        )

    pairs = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_fit_cluster)(cid) for cid in range(n_clusters)
    )
    cluster_models = {cid: m for cid, m in pairs}

    return ClusterModel(
        kmeans=km,
        cluster_models=cluster_models,
        creator_cluster=creator_cluster,
        feature_names=feature_names,
    )


# ================================================================== #
# C7: Mixed-effects (statsmodels MixedLM)                             #
# ================================================================== #

def train_mixed_effects(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    author_ids: pd.Series,
    random_slopes: list[str] | None = None,
    max_time_seconds: int = 1800,
) -> dict:
    """Simplified MixedLM: fixed effects for all features + random intercept per
    creator, with optional random slopes for 1–2 key features.

    Returns a dict with keys: 'result' (fitted MixedLMResults or None),
    'feature_names', 'group_col', 'converged', 'elapsed_seconds', 'skip_reason'.

    Skips and documents reason if training exceeds max_time_seconds or fails
    to converge. Caller should log the skip_reason to experiments.md.
    """
    import statsmodels.formula.api as smf
    import threading

    feature_names = list(X_train.columns)
    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(
        imp.fit_transform(X_train), columns=feature_names, index=X_train.index
    )
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_imp), columns=feature_names, index=X_train.index
    )

    df_fit = X_scaled.copy()
    df_fit["__target__"] = y_train.values
    df_fit["__group__"] = author_ids.values

    exog_re_cols = random_slopes if random_slopes else []
    exog_re_cols = [c for c in exog_re_cols if c in feature_names]

    result_holder: dict[str, Any] = {"result": None, "error": None}

    def _fit():
        try:
            if exog_re_cols:
                exog_re = df_fit[exog_re_cols]
                mdl = smf.mixedlm(
                    "__target__ ~ " + " + ".join(feature_names),
                    data=df_fit,
                    groups=df_fit["__group__"],
                    exog_re=exog_re,
                )
            else:
                mdl = smf.mixedlm(
                    "__target__ ~ " + " + ".join(feature_names),
                    data=df_fit,
                    groups=df_fit["__group__"],
                )
            result_holder["result"] = mdl.fit(reml=True, method="lbfgs", maxiter=200)
        except Exception as exc:
            result_holder["error"] = str(exc)

    t0 = time.time()
    thread = threading.Thread(target=_fit, daemon=True)
    thread.start()
    thread.join(timeout=max_time_seconds)
    elapsed = time.time() - t0

    if thread.is_alive():
        skip_reason = f"MixedLM exceeded {max_time_seconds}s time limit ({elapsed:.0f}s elapsed)"
        logger.warning(skip_reason)
        return {
            "result": None, "feature_names": feature_names,
            "group_col": "__group__", "converged": False,
            "elapsed_seconds": elapsed, "skip_reason": skip_reason,
            "imputer": imp, "scaler": scaler,
        }

    if result_holder["error"]:
        skip_reason = f"MixedLM failed: {result_holder['error']}"
        logger.warning(skip_reason)
        return {
            "result": None, "feature_names": feature_names,
            "group_col": "__group__", "converged": False,
            "elapsed_seconds": elapsed, "skip_reason": skip_reason,
            "imputer": imp, "scaler": scaler,
        }

    fitted = result_holder["result"]
    converged = bool(getattr(fitted, "converged", True))
    return {
        "result": fitted, "feature_names": feature_names,
        "group_col": "__group__", "converged": converged,
        "elapsed_seconds": elapsed, "skip_reason": None,
        "imputer": imp, "scaler": scaler,
    }


def predict_mixed_effects(
    me_result: dict,
    X_test: pd.DataFrame,
    author_ids_test: pd.Series,
) -> np.ndarray:
    """Predict using a fitted MixedLM result dict.

    Falls back to fixed-effects-only predictions for creators not in training.
    """
    if me_result["result"] is None:
        raise ValueError("MixedLM did not converge or was skipped.")

    feature_names = me_result["feature_names"]
    imp: SimpleImputer = me_result["imputer"]
    scaler: StandardScaler = me_result["scaler"]
    fitted = me_result["result"]

    X_imp = pd.DataFrame(
        imp.transform(X_test[feature_names]), columns=feature_names, index=X_test.index
    )
    X_scaled = pd.DataFrame(
        scaler.transform(X_imp), columns=feature_names, index=X_test.index
    )

    import statsmodels.api as sm
    X_with_const = sm.add_constant(X_scaled, has_constant="add")
    fe_pred = X_with_const.values @ fitted.fe_params.values

    pred = fe_pred.copy()
    re_dict = fitted.random_effects
    for author, idx_vals in author_ids_test.groupby(author_ids_test).groups.items():
        if author not in re_dict:
            continue
        re = re_dict[author]
        positions = X_test.index.get_indexer(idx_vals)
        re_val = float(re.get("Intercept", re.iloc[0]) if hasattr(re, "get") else re.values[0])
        pred[positions] += re_val

    return pred
