# notebooks/

The analysis, step by step. All notebooks are committed **with outputs**, so results are visible on GitHub without re-running anything.

## Notebooks

Run them in order: each one uses the outputs of the previous ones.

| # | Notebook | Goal | Main output |
|---|---|---|---|
| 01 | `01-data-exploration.ipynb` | Audit the three raw tables: size, nulls, distributions, join integrity | Dataset overview |
| 02 | `02-target-selection.ipynb` | Choose the measurement day (14) and the relative target | Target decision |
| 03 | `03-feature-engineering.ipynb` | Build 106 leakage-safe features in 14 groups | `data/processed/features.parquet`, `dataset_index.parquet`, `feature_spec.json` |
| 04 | `04-baseline-models.ipynb` | Trivial baselines (A1, A2), global Random Forest (B1) and XGBoost (B2) | `models/B1_global_rf.joblib`, `models/B2_global_xgb.joblib`, `results/predictions/nb04_predictions.parquet` |
| 05 | `05-alternative-models.ipynb` | Global LightGBM (C1) and Ridge (C2) | `models/alternative/`, `results/predictions/nb05_predictions.parquet` |
| 06 | `06-model-comparison.ipynb` | Leaderboard of all models + significance tests vs B2 | `results/leaderboard.csv`, `results/figures/nb06_leaderboard.png` |
| 07 | `07-feature-ablation.ipynb` | Which feature groups matter (27 combinations, B1 model) | `results/ablation.csv`, `results/figures/nb07_loo_contribution.png` |

`data/processed/` and `models/` are not in git, so notebooks 04–07 need notebook 03 to be run first in the same environment.

## How to run

**Google Colab.** Open a notebook with its Colab badge (in the root README). Notebooks 01–03 start with a setup cell that clones the repo and installs the package; run it first. Things to keep in mind:
- Everything in `/content` is deleted when the session ends. Commit and push before closing, and keep large files (raw data, models) on Google Drive.
- Notebook 04 trains for ~30 minutes. Keep the tab active, or the free runtime may disconnect.
- To save an executed notebook to the repo: *File → Save a copy in GitHub*, same path in `notebooks/`.

**Locally** (after `make setup`):

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks/04-baseline-models.ipynb
```

Replace the file name to run any other notebook.

## Conventions

1. **Code that is used more than once lives in `src/creator_analytics/`**, notebooks import it.
2. **Every notebook starts with Goal, Input and Conclusion.** The conclusion is written after the analysis.
3. **Every result cell is followed by a short Result / Takeaway** explaining what the output means.
4. **Restart & Run All before committing**, so cell numbers go in order and the notebook runs from top to bottom.
5. **One notebook, one author at a time.** Git can't merge two people's changes to the same notebook.
