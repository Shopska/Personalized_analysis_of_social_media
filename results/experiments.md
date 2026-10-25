# Experiments Log

| Date | Name | Model | Features | Params (truncated) | CV RMSE | Test RMSE | Test R² | Test Spearman | Train time | Win rate vs B2 | p-value | 95% CI RMSE diff vs B2 | Comment |
|------|------|-------|----------|--------------------|---------|-----------|---------|---------------|------------|----------------|---------|------------------------|---------|
| 2026-10-19 | per_creator_RF | RF_per_creator | non_constant_per_creator | `{"n_estimators": 100, "max_depth": 4, "min_samples_leaf": 5, "max_features": 0.4}` | n/a | 0.8931 | 0.1860 | 0.3776 | n/a | n/a | n/a | n/a | old pipeline; worse_than_zero_cv=158/1437 |
| 2026-10-19 | per_creator_XGB | XGB_per_creator | non_constant_per_creator | `{"n_estimators": 100, "max_depth": 3, "min_child_weight": 5, "reg_alpha": 1.0}` | n/a | 0.9023 | 0.1691 | 0.3838 | n/a | n/a | n/a | n/a | old pipeline; worse_than_zero_cv=294/1437 |
| 2026-10-19 | RF_global | RandomForest | all_54 | `{"n_estimators": 200, "min_samples_split": 10, "min_samples_leaf": 2, "max_features": 0.5}` | 0.9307 | 0.9311 | 0.1152 | 0.2759 | n/a | n/a | n/a | n/a | old pipeline |
| 2026-10-19 | XGB_with_follower | XGBoost | all_54 | `{"max_depth": 6, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.84}` | 0.9323 | 0.9321 | 0.1133 | 0.2615 | n/a | n/a | n/a | n/a | old pipeline |
| 2026-10-25 | A1_predict_zero | trivial_zero | 106 global | `{}` | n/a | 1.0005 | -0.0215 | nan | 0s | n/a | n/a | n/a | NB04 trivial baseline |
| 2026-10-25 | A2_creator_mean | trivial_creator_mean | 106 global | `{}` | n/a | 0.9690 | 0.0418 | 0.0759 | 0s | n/a | n/a | n/a | NB04 trivial baseline |
| 2026-10-25 | B1_global_RF | RandomForest | 106 global | `{"n_estimators": 300, "min_samples_split": 5, "min_samples_leaf": 5, "max_features": 0.3, "max_depth": null}` | 0.9376 | 0.9113 | 0.1525 | 0.3264 | 845s | n/a (vs predict-0: CI=[-0.093,-0.084]) | 0.0000 | n/a | NB04; search on 50k subsample 3-fold |
| 2026-10-25 | B2_global_XGB | XGBoost | 106 global | `{"max_depth": 6, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.84, "n_estimators": 1408}` | 0.9137 | 0.9127 | 0.1499 | 0.3101 | ~900s | n/a (vs predict-0: CI=[-0.092,-0.082]) | 0.0000 | n/a | NB04; full 160k 5-fold 30-iter |
| 2026-10-25 | B3_perCreator_RF | RF per-creator | 99 pc (no *_vs_creator) | `{"n_estimators": 100, "max_depth": 4, "min_samples_leaf": 5, "max_features": 0.4}` | n/a | 0.8872 | 0.1968 | 0.3925 | 68s | 0.542 | 0.0000 | [-0.0296, -0.0215] | NB05; 155/1437 worse-than-zero |
| 2026-10-25 | B4_perCreator_XGB | XGB per-creator | 99 pc (no *_vs_creator) | `{"n_estimators": 100, "max_depth": 3, "min_child_weight": 5, "reg_alpha": 1.0}` | n/a | 0.8967 | 0.1794 | 0.3897 | 68s | 0.534 | 0.0000 | [-0.0207, -0.0110] | NB05; 266/1437 worse-than-zero |
| 2026-10-25 | C1_global_LGBM | C1_global_LGBM | 106 global / 99 pc | `{}` | n/a | 0.9104 | 0.1541 | 0.3198 | 0s | 0.507 | 0.0027 | [-0.0045, 0.0002] | NB06; vs B2 global XGB |
| 2026-10-25 | C2_global_Ridge | C2_global_Ridge | 106 global / 99 pc | `{}` | n/a | 0.9718 | 0.0362 | 0.1416 | 0s | 0.471 | 0.0000 | [0.0545, 0.0633] | NB06; vs B2 global XGB |
| 2026-10-25 | C1_perCreator_LGBM | C1_perCreator_LGBM | 106 global / 99 pc | `{}` | n/a | 0.9219 | 0.1327 | 0.3553 | 0s | 0.498 | 0.0023 | [0.0039, 0.0148] | NB06; vs B2 global XGB |
| 2026-10-25 | C2_perCreator_Ridge | C2_perCreator_Ridge | 106 global / 99 pc | `{}` | n/a | 2.5268 | -5.5161 | 0.2712 | 0s | 0.402 | 0.0000 | [0.8812, 2.4584] | NB06; vs B2 global XGB |
| 2026-10-25 | C3_shrinkage | C3_shrinkage | 106 global / 99 pc | `{}` | n/a | 0.9089 | 0.1569 | 0.3208 | 0s | 0.617 | 0.0000 | [-0.0040, -0.0035] | NB06; vs B2 global XGB |
| 2026-10-25 | C4_finetune_XGB | C4_finetune_XGB | 106 global / 99 pc | `{}` | n/a | 0.9046 | 0.1649 | 0.3687 | 0s | 0.484 | 0.0000 | [-0.0124, -0.0042] | NB06; vs B2 global XGB |
| 2026-10-25 | C5_two_stage | C5_two_stage | 106 global / 99 pc | `{}` | n/a | 0.8905 | 0.1907 | 0.3985 | 0s | 0.533 | 0.0000 | [-0.0258, -0.0187] | NB06; vs B2 global XGB |
| 2026-10-25 | C6_cluster | C6_cluster | 106 global / 99 pc | `{}` | n/a | 0.9121 | 0.1509 | 0.3020 | 0s | 0.502 | 0.8196 | [-0.0031, 0.0020] | NB06; vs B2 global XGB |
| 2026-10-25 | C7_mixed_effects | C7_mixed_effects | 106 global / 99 pc | `{}` | n/a | 0.9570 | 0.0653 | 0.1631 | 18s | 0.451 | 0.0000 | [0.0408, 0.0480] | NB06; vs B2 global XGB |
