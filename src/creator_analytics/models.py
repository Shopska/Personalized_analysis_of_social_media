"""Model training — global families A–C2.

Design constraints
------------------
- search_* functions NEVER receive X_test / y_test — test isolation by signature.
- CV uses StratifiedKFold by author_id so each creator appears in both folds.
- XGBoost / LightGBM use early stopping on the CV val fold, not on test data.
- B1 (Global RF): hyperparameter search on a 50k subsample with 3 folds;
  the final model is retrained on the full train set using the best params.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

META_COLS: list[str] = [
    "video_id",
    "author_id",
    "split",
    "create_dt",
    "target_abs",
    "target_rel",
    "creator_baseline",
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

    model: Any  # sklearn Pipeline or raw estimator
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

    pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("rf", RandomForestRegressor(n_jobs=-1, random_state=seed)),
        ]
    )
    param_dist = {
        "rf__n_estimators": [100, 200, 300],
        "rf__max_depth": [None, 8, 15, 25],
        "rf__min_samples_split": [2, 5, 10, 20],
        "rf__min_samples_leaf": [1, 2, 5],
        "rf__max_features": [0.3, 0.5, 0.7, "sqrt"],
    }

    search = RandomizedSearchCV(
        pipe,
        param_dist,
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

    final_pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestRegressor(
                    n_jobs=-1,
                    random_state=seed,
                    **{k.replace("rf__", ""): v for k, v in best_params.items()},
                ),
            ),
        ]
    )
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
                X_arr[tr_idx],
                y_arr[tr_idx],
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
            i + 1,
            n_iter,
            cv_rmse,
            mean_rounds,
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


# ------------------------------------------------------------------ #
# C1: Global LightGBM                                                  #
# ------------------------------------------------------------------ #


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
            "num_leaves": int(rng.choice([31, 63, 127, 255])),
            "max_depth": int(rng.choice([-1, 4, 6, 8])),
            "learning_rate": float(rng.choice([0.05, 0.07, 0.1, 0.15])),
            "subsample": float(round(rng.uniform(0.6, 1.0), 2)),
            "colsample_bytree": float(round(rng.uniform(0.5, 1.0), 2)),
            "reg_alpha": float(rng.choice([0.0, 0.1, 0.5, 1.0])),
            "reg_lambda": float(round(rng.uniform(0.5, 2.0), 2)),
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
                X_arr[tr_idx],
                y_arr[tr_idx],
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


# ------------------------------------------------------------------ #
# C2: Global Ridge                                                      #
# ------------------------------------------------------------------ #


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

    pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("ridge", RidgeCV(alphas=alphas, cv=cv_splits, scoring="neg_mean_squared_error")),
        ]
    )
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
