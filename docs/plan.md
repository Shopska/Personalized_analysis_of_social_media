# Plan to the Final Defense

## Where we are (midterm)

Done: data loading and cleaning, EDA, target selection (relative target at day 14), feature engineering (106 features), global baseline models (RF, XGBoost, LightGBM, Ridge), model comparison, feature group ablation.

Best global models reach test RMSE ≈ 0.91 and Spearman ≈ 0.32, beating the trivial baselines by ~0.09 RMSE. These global models are the **baseline** for the next stage.

## Stages until the final defense

Owners: **OK** = Olha Kaplysh, **AS** = Anastasiia Shopska.

### Must-have

| # | Task | Owner | Done when |
|---|---|---|---|
| 1 | **Leakage audit of `creator_context`.** Check that account snapshots taken before publication are not affected by the video itself. Re-run the best model with and without this group. | OK | Written conclusion in a notebook; feature set fixed. |
| 2 | **Final feature set.** Apply ablation results (e.g. drop `emotions`, which hurt test RMSE) and freeze the feature list. | OK | `feature_spec.json` updated; global baseline re-trained on it. |
| 3 | **Per-creator models.** Train RF and XGBoost separately for each creator on that creator's train videos only. Use the same train/test split as the global model. | AS | Predictions for all creators on the test split. |
| 4 | **Per-creator vs global comparison.** Per creator: RMSE and Spearman for both approaches; overall: share of creators where per-creator wins, paired significance test, results by creator size. | OK | Comparison notebook + table in README. |
| 5 | **SHAP analysis.** Global SHAP vs per-creator SHAP: do the top factors differ between creators? Qualitative check on 5–10 creators of different niches. | OK + AS | Figures and conclusions in a notebook. |
| 6 | **Final documentation.** README, slides, poster. | AS + OK | Files in `docs/`. |

### Nice-to-have

| Task | Why it's optional |
|---|---|
| **Hybrid model** (global model + per-creator correction) | Useful if per-creator models overfit on small creators; only after task 4 shows where they fail. |
| **LLM layer** turning SHAP output into plain-language advice | Improves presentation, but the core result is the SHAP analysis itself. |
| **Updated Streamlit demo** on the final models | Good for the defense, not required by the research question. |
| More hyperparameter tuning | Tree models are already within 0.002 RMSE of each other; expected gain is small. |

## Risks and plan B

| Risk | Impact | What we do |
|---|---|---|
| **Too few videos for some creators** (filter is ≥ 30 videos, so the smallest creators have ~24 training videos) | Per-creator models overfit | Simpler models (shallow trees, strong regularization); report results by creator size; raise the minimum threshold; hybrid model as plan B. |
| **Per-creator models lose to the global model** | Main hypothesis not confirmed | Still a valid result. We report where and for whom personalization helps (e.g. only large creators) and use the hybrid approach. |
| **`creator_context` features leak information** | Global results are overestimated | Task 1. If leakage is found, drop or rebuild the group and re-train all baselines. |
| **Low R² (~0.15)** | Weak predictions look unconvincing | Focus on ranking quality (Spearman) and on explanations; compare against trivial baselines, not against an absolute bar. |
| **Unstable SHAP for small creators** | Misleading advice | Show a stability warning for creators with few videos; check top-feature stability across bootstrap samples or seeds. |
