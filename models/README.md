# models/

Trained models are **not committed to git** (they are large and can be re-created). Only this README is tracked; everything else in this folder is ignored by `.gitignore`.

## What appears here after running the notebooks

| File | Model | Created by |
|---|---|---|
| `B1_global_rf.joblib` | B1 · global Random Forest | `notebooks/04-baseline-models.ipynb` |
| `B2_global_xgb.joblib` | B2 · global XGBoost | `notebooks/04-baseline-models.ipynb` |
| `global_xgb.joblib` | Global XGBoost used by the CLI and the Streamlit demo | `python -m creator_analytics train` |
| `alternative/C1_global_lgbm.joblib` | C1 · global LightGBM | `notebooks/05-alternative-models.ipynb` |
| `alternative/C2_global_ridge.joblib` | C2 · global Ridge | `notebooks/05-alternative-models.ipynb` |
| `alternative/training_times.json` | Training time of C1 and C2 | `notebooks/05-alternative-models.ipynb` |

All models are trained on the `train` split only and predict the **relative target** (`log1p(views at day 14)` minus the creator's train median).

Per-creator models will be added here after the midterm (see [`docs/plan.md`](../docs/plan.md)).

## Working in Colab

`/content` is wiped when the session ends, so save trained models to Google Drive if you don't want to re-train them (notebook 04 takes ~30 minutes).
