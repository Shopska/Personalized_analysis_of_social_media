# Notebook conventions

Naming: `NN-kebab-case-name.ipynb`, numbered in the order you'd actually read them (`NN` is two digits: `01`, `02`, ...).

1. **A notebook is a draft and a narrative, not the source of truth.** Working code that's used more than once belongs in `src/creator_analytics/`, and the notebook imports it — the rule of thumb is: if a function is needed twice, it belongs in `src/`, not copy-pasted between notebooks.
2. **Every notebook opens with a markdown cell**: Goal, Input, Conclusion. The Conclusion is written **after** the analysis, once you know what the results actually say — not drafted in advance and left unchanged.
3. **`Restart & Run All` before committing.** A notebook that doesn't run cleanly top to bottom doesn't get committed — a notebook full of out-of-order cell numbers is a common source of "works on my machine, mysteriously not on yours."
4. **Outputs are committed.** This course's convention is to keep cell outputs in the `.ipynb` file (not strip them), so plots and tables are visible directly on GitHub without anyone having to re-run the notebook first. The tradeoff is merge conflicts (see rule 5) and file size — keep figures reasonably sized, don't `df.head(10000)` into a cell.
   - `.pre-commit-config.yaml` ships an `nbstripout` hook that would strip outputs on commit — it's **disabled by default** to match this convention. If your team decides the conflict pain outweighs the benefit, there are instructions in that file for turning it on.
5. **Notebooks and git merges don't mix well.** The underlying JSON makes conflicts nearly unreadable, and git can't meaningfully merge two people's changes to the same notebook. Practical rule: one notebook, one author, at a time. If two people need to work on the same analysis, split it into two notebooks or hand it off explicitly rather than editing in parallel.
6. **Colab compatibility.** Each notebook carries an "Open in Colab" badge and a guarded first code cell (`if "google.colab" in sys.modules: ...`) that clones the repo and installs dependencies — safe to leave in when running locally, since the block is a no-op outside Colab.

## Notebooks in this project

All seven notebooks are fully executed and committed with outputs. Run them in order:

| # | Notebook | Goal | Key output |
|---|----------|------|-----------|
| 01 | `01-data-exploration.ipynb` | Audit three raw tables; characterise nulls, distributions, join integrity | Dataset overview |
| 02 | `02-target-selection.ipynb` | Compare absolute vs relative play-count target; confirm follower-count leakage | Relative target chosen |
| 03 | `03-feature-engineering.ipynb` | Build leakage-safe features (106 features, 14 groups) with `creator_analytics.features`; drop text stats that are a collection artifact | `data/processed/features.parquet`, `dataset_index.parquet`, `feature_spec.json` |
| 04 | `04-baseline-model.ipynb` | Train global RF + XGBoost; CV and test evaluation | `models/global_xgb.joblib`, `results/predictions_global.parquet` |
| 05 | `05-per-creator-models.ipynb` | Train 1 437 personalised RF models; compare with global | `models/per_creator/RF/`, `results/predictions_per_creator.parquet` |
| 06 | `06-model-comparison.ipynb` | Paired Wilcoxon test + bootstrap CI; personalisation wins for 65 % of creators | `results/figures/nb06_*.png` |
| 07 | `07-explainability.ipynb` | SHAP for global model + all per-creator models; stability analysis | `results/shap_*.parquet`, `results/figures/nb07_*.png` |

To regenerate all notebooks from source:
```bash
python scripts/make_notebooks.py   # writes .ipynb files from make_notebooks.py strings
# then execute each with:
.venv/bin/jupyter-nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=creator_analytics_venv \
    notebooks/07-explainability.ipynb
```
