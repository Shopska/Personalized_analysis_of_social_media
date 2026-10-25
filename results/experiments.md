# Experiments Log

| Date | Name | Model | Features | Params (truncated) | CV RMSE | Test RMSE | Test R² | Test Spearman | Train time | Win rate vs B2 | p-value | 95% CI RMSE diff vs B2 | Comment |
|------|------|-------|----------|--------------------|---------|-----------|---------|---------------|------------|----------------|---------|------------------------|---------|
| 2026-10-25 | A1_predict_zero | trivial_zero | 106 global | `{}` | n/a | 1.0005 | -0.0215 | nan | 0s | n/a | n/a | n/a | NB04 trivial baseline |
| 2026-10-25 | A2_creator_mean | trivial_creator_mean | 106 global | `{}` | n/a | 0.9690 | 0.0418 | 0.0759 | 0s | n/a | n/a | n/a | NB04 trivial baseline |
| 2026-10-25 | B1_global_RF | RandomForest | 106 global | `{"n_estimators": 300, "min_samples_split": 5, "min_samples_leaf": 5, "max_features": 0.3, "max_depth": null}` | 0.9376 | 0.9113 | 0.1525 | 0.3264 | 845s | n/a (vs predict-0: CI=[-0.093,-0.084]) | 0.0000 | n/a | NB04; search on 50k subsample 3-fold |
| 2026-10-25 | B2_global_XGB | XGBoost | 106 global | `{"max_depth": 6, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.84, "n_estimators": 1408}` | 0.9137 | 0.9127 | 0.1499 | 0.3101 | ~900s | n/a (vs predict-0: CI=[-0.092,-0.082]) | 0.0000 | n/a | NB04; full 160k 5-fold 30-iter |
| 2026-10-25 | C1_global_LGBM | LightGBM | 106 global | `{}` | n/a | 0.9104 | 0.1541 | 0.3198 | n/a | 0.507 | 0.0027 | [-0.0045, 0.0002] | NB05; vs B2 global XGB |
| 2026-10-25 | C2_global_Ridge | Ridge | 106 global | `{}` | n/a | 0.9718 | 0.0362 | 0.1416 | n/a | 0.471 | 0.0000 | [0.0545, 0.0633] | NB05; vs B2 global XGB |
