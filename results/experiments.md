# Experiments Log

| Date | Name | Model | Features | Params (truncated) | CV RMSE | Test RMSE | Test R² | Test Spearman | Comment |
|------|------|-------|----------|--------------------|---------|-----------|---------|---------------|---------|
| 2026-10-19 | per_creator_RF | RF_per_creator | non_constant_per_creator | `{"n_estimators": 100, "max_depth": 4, "min_samples_leaf": 5, "max_features": 0.4}` | n/a | 0.8931 | 0.1860 | 0.3776 | worse_than_zero_cv=158/1437 |
| 2026-10-19 | per_creator_XGB | XGB_per_creator | non_constant_per_creator | `{"n_estimators": 100, "max_depth": 3, "min_child_weight": 5, "reg_alpha": 1.0, "reg_lambda": 2.0, "s` | n/a | 0.9023 | 0.1691 | 0.3838 | worse_than_zero_cv=294/1437 |
