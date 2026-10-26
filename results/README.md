# results/

Everything here is produced by the notebooks and **is committed**, so results can be checked without re-running the code. Don't edit these files by hand: re-run the notebook instead.

| File | What's inside | Created by |
|---|---|---|
| `leaderboard.csv` | Test metrics of all models (RMSE, MAE, R², Spearman) + comparison with B2 | NB06 |
| `metrics.json` | Key test and CV RMSE values, used for slides and the README | `make reproduce` (CLI) |
| `experiments.md` | Log of all experiments: model, parameters, CV and test metrics, comments | NB04, NB05 |
| `ablation.csv` | CV (and partly test) metrics for 27 feature group combinations | `scripts/run_ablation.py`, NB07 |
| `predictions/nb04_predictions.parquet` | Test predictions of A1, A2, B1, B2 | NB04 |
| `predictions/nb05_predictions.parquet` | Test predictions of C1, C2 | NB05 |
| `predictions_global.parquet` | Test predictions of the global XGBoost (`y_true`, `y_pred`, `author_id`) | `make reproduce` (CLI) |
| `shap_global_test.parquet` | SHAP values on the test set, used by the Streamlit demo | old pipeline, to be regenerated |
| `figures/` | All figures from the notebooks; the prefix shows the source (`fig1–6` = NB02, `nb01_`, `nb04_`, `nb06_`, `nb07_`) | NB01–NB07 |

All metrics are computed on the **relative target** on the held-out test split (40,124 videos). The test split is used only for final evaluation, never for tuning.
