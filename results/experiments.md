# Experiments Log

| Date | Name | Model | Features | Params (truncated) | CV RMSE | Test RMSE | Test R² | Test Spearman | Comment |
|------|------|-------|----------|--------------------|---------|-----------|---------|---------------|---------|
| 2026-10-19 | per_creator_RF | RF_per_creator | non_constant_per_creator | `{"n_estimators": 100, "max_depth": 4, "min_samples_leaf": 5, "max_features": 0.4}` | n/a | 0.8931 | 0.1860 | 0.3776 | worse_than_zero_cv=158/1437 |
| 2026-10-19 | per_creator_XGB | XGB_per_creator | non_constant_per_creator | `{"n_estimators": 100, "max_depth": 3, "min_child_weight": 5, "reg_alpha": 1.0, "reg_lambda": 2.0, "s` | n/a | 0.9023 | 0.1691 | 0.3838 | worse_than_zero_cv=294/1437 |
| 2026-10-19 | predict_0 | predict_0 | none | `{}` | n/a | 1.0005 | -0.0215 | nan |  |
| 2026-10-19 | predict_creator_mean | predict_creator_mean | none | `{}` | n/a | 0.9690 | -0.0215 | nan |  |
| 2026-10-19 | RF_global | RandomForest | all_54 | `{"n_estimators": 200, "min_samples_split": 10, "min_samples_leaf": 2, "max_features": 0.5, "max_dept` | 0.9307 | 0.9311 | 0.1152 | 0.2759 |  |
| 2026-10-19 | XGB_with_follower | XGBoost | all_54 | `{"max_depth": 6, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.84, "reg_alpha": 0.` | 0.9323 | 0.9321 | 0.1133 | 0.2615 |  |
| 2026-10-19 | XGB_no_follower | XGBoost | all_54_no_follower | `{"max_depth": 6, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.84, "reg_alpha": 0.` | 0.9461 | 0.9468 | 0.0852 | 0.2381 |  |
| 2026-10-19 | comparison_global_vs_percreator | comparison | all_test_videos | `{}` | n/a | 0.8931 | 0.1860 | 0.3776 | global_rmse=0.9321  wilcoxon_p=0.0000  CI=[0.0255,0.0531]  pc_wins=933/1437 |
| 2026-10-19 | SHAP_explainability | XGBoost_SHAP | all_53 | `{"shap_version": "0.52.0", "n_test": 40124}` | n/a | 0.0000 | 0.0000 | 0.0000 | global_top3=follower_count_log,sticker_text_len,duration_log; median_rank_corr=0.513; n_creators_shap=1437 |
| 2026-10-19 | SHAP_explainability | XGBoost_SHAP | all_53 | `{"shap_version": "0.52.0", "n_test": 40124}` | n/a | 0.0000 | 0.0000 | 0.0000 | global_top3=follower_count_log,sticker_text_len,duration_log; median_rank_corr=0.513; n_creators_shap=1437 |
| 2026-10-19 | per_creator_RF | RF_per_creator | non_constant_per_creator | `{"n_estimators": 100, "max_depth": 4, "min_samples_leaf": 5, "max_features": 0.4}` | nan | 0.8931 | 0.1860 | 0.3776 | worse_than_zero_cv=158/1437 |
| 2026-10-19 | per_creator_XGB | XGB_per_creator | non_constant_per_creator | `{"n_estimators": 100, "max_depth": 3, "min_child_weight": 5, "reg_alpha": 1.0, "reg_lambda": 2.0, "s` | nan | 0.9023 | 0.1691 | 0.3838 | worse_than_zero_cv=294/1437 |
