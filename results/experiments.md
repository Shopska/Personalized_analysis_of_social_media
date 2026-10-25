# Experiments Log

| Date | Name | Model | Features | Params (truncated) | CV RMSE | Test RMSE | Test R² | Test Spearman | Train time | Win rate vs 0 | p-value | 95% CI RMSE diff vs 0 | Comment |
|------|------|-------|----------|--------------------|---------|-----------|---------|---------------|------------|---------------|---------|----------------------|---------|
| 2026-10-19 | per_creator_RF | RF_per_creator | non_constant_per_creator | `{"n_estimators": 100, "max_depth": 4, "min_samples_leaf": 5, "max_features": 0.4}` | n/a | 0.8931 | 0.1860 | 0.3776 | n/a | n/a | n/a | n/a | old pipeline; worse_than_zero_cv=158/1437 |
| 2026-10-19 | per_creator_XGB | XGB_per_creator | non_constant_per_creator | `{"n_estimators": 100, "max_depth": 3, "min_child_weight": 5, "reg_alpha": 1.0}` | n/a | 0.9023 | 0.1691 | 0.3838 | n/a | n/a | n/a | n/a | old pipeline; worse_than_zero_cv=294/1437 |
| 2026-10-19 | RF_global | RandomForest | all_54 | `{"n_estimators": 200, "min_samples_split": 10, "min_samples_leaf": 2, "max_features": 0.5}` | 0.9307 | 0.9311 | 0.1152 | 0.2759 | n/a | n/a | n/a | n/a | old pipeline |
| 2026-10-19 | XGB_with_follower | XGBoost | all_54 | `{"max_depth": 6, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.84}` | 0.9323 | 0.9321 | 0.1133 | 0.2615 | n/a | n/a | n/a | n/a | old pipeline |
| 2026-10-25 | A1_predict_zero | trivial_zero | 106 global | `{}` | n/a | 1.0005 | -0.0215 | nan | 0s | n/a | n/a | n/a | NB04 trivial baseline |
| 2026-10-25 | A2_creator_mean | trivial_creator_mean | 106 global | `{}` | n/a | 0.9690 | 0.0418 | 0.0759 | 0s | n/a | n/a | n/a | NB04 trivial baseline |
| 2026-10-25 | B1_global_RF | RandomForest | 106 global | `{"n_estimators": 300, "min_samples_split": 5, "min_samples_leaf": 5, "max_features": 0.3, "max_depth": null}` | 0.9376 | 0.9113 | 0.1525 | 0.3264 | 845s | 0.495 | 0.0000 | [-0.0932, -0.0844] | NB04; search on 50k subsample 3-fold; final fit on 160k |
| 2026-10-25 | B2_global_XGB | XGBoost | 106 global | `{"max_depth": 6, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.84, "reg_alpha": 0.0, "n_estimators": 1408}` | 0.9137 | 0.9127 | 0.1499 | 0.3101 | 34344s | 0.488 | 0.0000 | [-0.0924, -0.0825] | NB04; search on full 160k 5-fold 30-iter; slow — add subsampling next run |
