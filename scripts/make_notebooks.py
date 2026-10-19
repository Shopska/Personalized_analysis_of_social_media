#!/usr/bin/env python
"""Generate all project notebooks (01–06) from source strings.

Run once to (re)create the notebooks from scratch:
    python scripts/make_notebooks.py

Then execute them:
    jupyter nbconvert --to notebook --execute --inplace notebooks/01-data-exploration.ipynb
    jupyter nbconvert --to notebook --execute --inplace notebooks/02-target-selection.ipynb
    jupyter nbconvert --to notebook --execute --inplace notebooks/03-feature-engineering.ipynb
    jupyter nbconvert --to notebook --execute --inplace notebooks/04-baseline-model.ipynb
    jupyter nbconvert --to notebook --execute --inplace notebooks/05-per-creator-models.ipynb
    jupyter nbconvert --to notebook --execute --inplace notebooks/06-model-comparison.ipynb
"""

from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

NB_DIR = Path("notebooks")

# ------------------------------------------------------------------ #
# Shared Colab guard (same for both notebooks)                        #
# ------------------------------------------------------------------ #
COLAB_GUARD = """\
import sys
if "google.colab" in sys.modules:
    import subprocess, os
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/Shopska/Personalized_analysis_of_social_media.git", "repo"],
        check=True,
    )
    os.chdir("repo")
    subprocess.run(
        ["pip", "install", "-q", "-r", "requirements.txt", "-r", "requirements-dev.txt"],
        check=True,
    )
    subprocess.run(["pip", "install", "-q", "-e", "."], check=True)
    subprocess.run(["python", "scripts/download_data.py"], check=True)
"""

SETUP_COMMON = """\
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Ensure working directory is the project root when opened from notebooks/
if Path.cwd().name == "notebooks":
    os.chdir(Path.cwd().parent)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from IPython.display import display
%matplotlib inline
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                     "axes.spines.right": False})
"""

# ================================================================== #
# NOTEBOOK 01: Data Exploration                                       #
# ================================================================== #

NB01_GOAL = """\
# 01 — Data Exploration

**Goal** – Audit the three raw tables (`videos`, `creator_daily`,
`engagement_daily`): characterise rows, columns, nulls, date ranges,
creator distribution, join integrity, and engagement-metric distributions.

**Input** – `data/raw/*.parquet` downloaded by `scripts/download_data.py`.

**Conclusion** –
Three tables downloaded cleanly:
`videos` 209 543 rows × 33 cols (175.1 MB, dates 2024-06-24–2024-11-09),
`creator_daily` 278 433 × 7 (3.1 MB),
`engagement_daily` 6 068 955 × 10 (31.8 MB, **0 nulls anywhere**).
1 872 unique creators; median 69 videos per creator (min 5, max 2 218).
Join integrity is perfect: every `video_id` appears in both `videos` and
`engagement_daily`; every `author_id` in both `videos` and `creator_daily`.
Notable nulls: `transcript` 60.3 %, `sticker_text` 52.8 % (use `gpt_summary`
which is 0 % null); `enterprise_verified` in `creator_daily` is 100 % null.
`play_count` is highly right-skewed (median 570, max 44 442 900) —
log-scale transformation is required before modelling.
"""

NB01_LOAD = """\
from creator_analytics.config import load_config
from creator_analytics.data import load_raw_table

cfg    = load_config("config/default.yaml")
RAW    = cfg["paths"]["raw_dir"]
FIGS   = Path("results/figures")
FIGS.mkdir(parents=True, exist_ok=True)

videos        = load_raw_table(RAW, "videos")
creator_daily = load_raw_table(RAW, "creator_daily")
engagement    = load_raw_table(RAW, "engagement_daily")

def _mb(t: str) -> float:
    return Path(RAW).joinpath(f"{t}.parquet").stat().st_size / 1_048_576

for t, df in [("videos", videos), ("creator_daily", creator_daily),
               ("engagement_daily", engagement)]:
    print(f"{t:<22} {len(df):>9,} rows  {len(df.columns):>2} cols  {_mb(t):>7.1f} MB")
"""

NB01_SCHEMA_MD = "## Table schemas and null rates"

NB01_SCHEMA = """\
def _schema(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        try:
            nu = df[col].nunique()
        except TypeError:
            nu = None          # unhashable type (e.g. list/array column)
        rows.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "null_%": round(df[col].isna().mean() * 100, 1),
            "n_unique": nu,
        })
    return pd.DataFrame(rows).set_index("column")

print("### videos")
display(_schema(videos))
print("### creator_daily")
display(_schema(creator_daily))
print("### engagement_daily")
display(_schema(engagement))
"""

NB01_DATERANGE_MD = "## Date ranges"

NB01_DATERANGE = """\
for name, df, col in [
    ("videos",        videos,        "create_date"),
    ("creator_daily", creator_daily, "date"),
    ("engagement",    engagement,    "date"),
]:
    s = pd.to_datetime(df[col], errors="coerce")
    print(f"{name:<22} {s.min().date()} – {s.max().date()}")
"""

NB01_CREATORS_MD = "## Videos per creator"

NB01_CREATORS = """\
vc = videos["author_id"].value_counts()
print(f"Unique creators : {len(vc):,}")
print(f"Min videos      : {vc.min()}")
print(f"Median videos   : {vc.median():.0f}")
print(f"Max videos      : {vc.max()}")
"""

NB01_CREATORS_FIG = """\
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Histogram (log y)
axes[0].hist(vc.values, bins=60, color="steelblue", edgecolor="none", log=True)
axes[0].set_xlabel("Videos per creator")
axes[0].set_ylabel("Number of creators (log)")
axes[0].set_title("Distribution of videos per creator")

# CDF
sorted_vc = np.sort(vc.values)
axes[1].plot(sorted_vc, np.linspace(0, 1, len(sorted_vc)), color="steelblue")
axes[1].set_xlabel("Videos per creator")
axes[1].set_ylabel("Cumulative fraction of creators")
axes[1].set_title("CDF of videos per creator")
for thresh in [10, 20, 30, 50]:
    pct = (vc >= thresh).mean()
    axes[1].axvline(thresh, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    axes[1].text(thresh + 2, pct - 0.03, f"{thresh}", fontsize=7, color="gray")

fig.tight_layout()
fig.savefig(FIGS / "nb01_creator_distribution.png", dpi=120)
plt.show()
"""

NB01_THRESHOLD = """\
rows = [(t, int((vc >= t).sum()), int(vc[vc >= t].sum())) for t in [10, 20, 30, 50]]
display(
    pd.DataFrame(rows, columns=["min_videos_per_creator", "creators_kept", "videos_kept"])
    .set_index("min_videos_per_creator")
)
"""

NB01_JOIN_MD = "## Join integrity"

NB01_JOIN = """\
v_ids = set(videos["video_id"])
e_ids = set(engagement["video_id"])
a_ids = set(videos["author_id"])
c_ids = set(creator_daily["author_id"])

print(f"video_id in videos   only (not in engagement_daily): {len(v_ids - e_ids):,}")
print(f"video_id in eng.     only (not in videos)          : {len(e_ids - v_ids):,}")
print(f"author_id in videos  only (not in creator_daily)   : {len(a_ids - c_ids):,}")
print(f"author_id in creator only (not in videos)          : {len(c_ids - a_ids):,}")
"""

NB01_ENG_MD = "## `engagement_daily`: structure and metric distributions"

NB01_ENG_DSP = """\
dsp = engagement["days_since_post"]
print("days_since_post distribution across all rows:")
print(f"  min={dsp.min()}  p25={dsp.quantile(0.25):.0f}  "
      f"median={dsp.median():.0f}  p75={dsp.quantile(0.75):.0f}  max={dsp.max()}")

fig, ax = plt.subplots(figsize=(8, 3))
ax.hist(dsp.values, bins=range(0, 32), color="steelblue", edgecolor="white", linewidth=0.3)
ax.set_xlabel("days_since_post")
ax.set_ylabel("Row count")
ax.set_title("Row count by days_since_post (all 6 M rows)")
ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
fig.tight_layout()
plt.show()
"""

NB01_ENG_METRICS = """\
metric_cols = ["play_count", "like_count", "comment_count",
               "share_count", "collect_count", "download_count"]

# Show stats at day 14 (full coverage)
eng14 = engagement[engagement["days_since_post"] == 14]
stats = (
    eng14[metric_cols]
    .agg(["min", "median", "mean", "max"])
    .T.astype(int)
    .rename(columns=str.title)
)
display(stats)
"""

NB01_ENG_FIG = """\
eng14 = engagement[engagement["days_since_post"] == 14]
metric_cols = ["play_count", "like_count", "comment_count",
               "share_count", "collect_count", "download_count"]

fig, axes = plt.subplots(2, 3, figsize=(13, 6))
for ax, col in zip(axes.flat, metric_cols):
    data = eng14[col].clip(lower=1)          # shift 0→1 so log1p is fine
    ax.hist(np.log1p(data), bins=60, color="steelblue", edgecolor="none")
    ax.set_xlabel(f"log1p({col})")
    ax.set_ylabel("Videos")
    ax.set_title(col)

fig.suptitle("Engagement metric distributions at day 14 (log scale)", fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(FIGS / "nb01_engagement_distributions.png", dpi=120, bbox_inches="tight")
plt.show()
"""

# ================================================================== #
# NOTEBOOK 02: Target Selection                                       #
# ================================================================== #

NB02_GOAL = """\
# 02 — Target Selection

**Goal** – Decide between `log1p(play_count)` at day 7, 14, or 30, and
between an *absolute* and *relative* (creator-centred) target formulation.
Five questions are answered in sequence.

**Input** – The three raw tables + `config/default.yaml`.

**Conclusion** –
**Day 14** is the right measurement point: 100 % exact coverage (day 7 = 98.4 %;
day 30 = 35.3 %), and videos already hold a median **95.2 %** of their
day-30 play_count by then.
Day-30 under-coverage is time-based, not popularity-based (day-14 medians 623 vs
587 for covered/uncovered groups).
1 240 videos (0.59 %) have a monotonicity violation; when present the median
drop is 96 % of peak — a data-collection reset artefact.
**Relative target is strongly preferred**: 73.2 % of the variance in the
absolute `log1p(play_count_day14)` is explained by `author_id` alone (test
R² = 0.732); the relative target (subtract per-creator train median) reduces
this to 4.4 %, isolating content-driven variance.
`follower_count` at publish time (as-of join, 100 % coverage) has Spearman
ρ = 0.639 with the absolute target — it is a powerful feature, not a target
component, and confirms the scale of the creator effect.
"""

NB02_LOAD = """\
from creator_analytics.config import load_config
from creator_analytics.data import filter_creators, load_raw_table, train_test_split_by_creator
from creator_analytics.features import follower_asof_join
from creator_analytics.targets import build_target

cfg    = load_config("config/default.yaml")
RAW    = cfg["paths"]["raw_dir"]
FIGS   = Path("results/figures")
FIGS.mkdir(parents=True, exist_ok=True)

videos        = load_raw_table(RAW, "videos")
creator_daily = load_raw_table(RAW, "creator_daily")
engagement    = load_raw_table(RAW, "engagement_daily")
print(f"videos {len(videos):,}  creator_daily {len(creator_daily):,}  engagement {len(engagement):,}")
"""

NB02_COV_MD = "## 1. Coverage by `days_since_post`"

NB02_COV = """\
total = videos["video_id"].nunique()
rows = []
for day in [0, 7, 14, 21, 28, 30]:
    exact = engagement[engagement["days_since_post"] == day]["video_id"].nunique()
    geq   = engagement[engagement["days_since_post"] >= day]["video_id"].nunique()
    rows.append({"day": day,
                 "exact_n": exact, "exact_%": round(exact / total * 100, 1),
                 "≥day_n": geq,    "≥day_%":  round(geq   / total * 100, 1)})

display(pd.DataFrame(rows).set_index("day"))

# tracking window per video
max_day = engagement.groupby("video_id")["days_since_post"].max()
n_days  = engagement.groupby("video_id")["days_since_post"].nunique()
print(f"\\nDistinct days tracked per video: "
      f"median={n_days.median():.0f}  p5={n_days.quantile(0.05):.0f}  p95={n_days.quantile(0.95):.0f}")
"""

NB02_COV_NOTE = """\
> **Key observation**: day 14 is the *only* day with 100 % exact coverage.
> Day 30 is 35.3 %; day 7 is 98.4 %.  Use the table above to compare.
"""

NB02_COV_FIG = """\
cov_df   = pd.DataFrame(rows)
sel_days = [7, 14, 21, 28, 30]
sub      = cov_df[cov_df["day"].isin(sel_days)].reset_index(drop=True)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, col, title, color in [
    (axes[0], "exact_%", "Exact coverage (== day)", "steelblue"),
    (axes[1], "≥day_%",  "Cumulative coverage (≥ day)", "darkorange"),
]:
    bars = ax.bar([str(d) for d in sel_days], sub[col], color=color)
    ax.set_ylim(0, 115); ax.set_xlabel("days_since_post"); ax.set_ylabel("% of videos")
    ax.set_title(title)
    for bar, v in zip(bars, sub[col]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.0f}%",
                ha="center", fontsize=8)

fig.suptitle("Engagement record coverage by measurement day", fontweight="bold")
fig.tight_layout()
fig.savefig(FIGS / "fig1_coverage_by_day.png", dpi=120)
plt.show()
"""

NB02_MONO_MD = "## 2. Monotonicity of `play_count`"

NB02_MONO = """\
eng_s = engagement.sort_values(["video_id", "days_since_post"]).copy()
eng_s["play_diff"] = eng_s.groupby("video_id")["play_count"].diff()

viols = eng_s[eng_s["play_diff"] < 0]
n_viol = viols["video_id"].nunique()
print(f"Videos with ≥1 decrease in play_count : {n_viol:,}  "
      f"({n_viol / len(videos) * 100:.2f}%)")
print(f"Total violation rows                    : {len(viols):,}")

if n_viol:
    max_drop = viols.groupby("video_id")["play_diff"].min().abs()
    peak     = engagement.groupby("video_id")["play_count"].max()
    sev      = (max_drop / peak * 100).dropna()
    print(f"Max drop as % of peak — median={sev.median():.1f}%  p95={sev.quantile(0.95):.1f}%")
    print("→ Violations are severe but rare; likely a data-collection reset artefact.")
"""

NB02_RATIO_MD = "## 3. Day-14 as a fraction of day-30 play_count"

NB02_RATIO = """\
def _exact_plays(day: int) -> pd.Series:
    return engagement[engagement["days_since_post"] == day].set_index("video_id")["play_count"]

pc7, pc14, pc30 = _exact_plays(7), _exact_plays(14), _exact_plays(30)
common = pc7.index.intersection(pc14.index).intersection(pc30.index)
print(f"Videos with exact records at days 7, 14, 30: {len(common):,}  "
      f"({len(common)/len(videos)*100:.1f}%)")

ratio14 = pc14.loc[common] / pc30.loc[common].replace(0, np.nan)
ratio7  = pc7.loc[common]  / pc30.loc[common].replace(0, np.nan)

for label, r in [("day 7 / day 30", ratio7), ("day 14 / day 30", ratio14)]:
    q = r.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
    print(f"\\n{label}  (n={r.notna().sum():,})")
    print(f"  p5={q[0.05]:.3f}  p25={q[0.25]:.3f}  median={q[0.50]:.3f}"
          f"  p75={q[0.75]:.3f}  p95={q[0.95]:.3f}")
    print(f"  Videos ≥ 90% of day-30 already by this day: {(r>=0.9).mean()*100:.1f}%")
"""

NB02_BIAS = """\
# Bias check: do day-30 videos differ systematically?
has30  = set(pc30.index)
v30    = videos[ videos["video_id"].isin(has30)].copy()
v_no30 = videos[~videos["video_id"].isin(has30)].copy()

for grp, label in [(v30, "has day-30"), (v_no30, "no day-30")]:
    s = pd.to_datetime(grp["create_date"], errors="coerce")
    pc14_grp = pc14.reindex(grp["video_id"]).dropna()
    print(f"{label} ({len(grp):,} videos): median create_date={s.median().date()}  "
          f"play_count(day14) median={pc14_grp.median():.0f}  mean={pc14_grp.mean():.0f}")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# CDF of ratios
for r, label, color in [(ratio7, "day 7 / day 30", "steelblue"),
                         (ratio14, "day 14 / day 30", "darkorange")]:
    rc = r.dropna().clip(0, 1)
    axes[0].plot(np.sort(rc), np.linspace(0, 1, len(rc)), label=label, color=color)
axes[0].axvline(0.9, color="gray", linestyle="--", linewidth=0.8, label="90% threshold")
axes[0].set_xlabel("Fraction of day-30 play_count"); axes[0].set_ylabel("CDF")
axes[0].set_title("CDF: day-7 and day-14 vs day-30"); axes[0].legend(fontsize=8)

# Bias: create_date
for grp, label, c in [(v30, "has day-30", "darkorange"), (v_no30, "no day-30", "steelblue")]:
    s = pd.to_datetime(grp["create_date"], errors="coerce")
    axes[1].hist(s.dt.dayofyear, bins=40, alpha=0.55, label=label, color=c, density=True)
axes[1].set_xlabel("Day of year (2024)"); axes[1].set_ylabel("Density")
axes[1].set_title("Publication date: day-30 vs not"); axes[1].legend(fontsize=8)

# Bias: play_count(day14)
for grp, label, c in [(v30, "has day-30", "darkorange"), (v_no30, "no day-30", "steelblue")]:
    pc_g = pc14.reindex(grp["video_id"]).dropna().clip(lower=1)
    axes[2].hist(np.log1p(pc_g), bins=50, alpha=0.55, label=label, color=c, density=True)
axes[2].set_xlabel("log1p(play_count at day 14)"); axes[2].set_ylabel("Density")
axes[2].set_title("play_count(day14): day-30 vs not"); axes[2].legend(fontsize=8)

fig.suptitle("Day-30 bias check and fraction of day-30 engagement captured earlier",
             fontweight="bold")
fig.tight_layout()
fig.savefig(FIGS / "fig2_day30_ratio_and_bias.png", dpi=120)
plt.show()
"""

NB02_TARGET_MD = "## 4. Absolute vs relative target"

NB02_SPLIT = """\
min_vids = cfg["data"]["min_videos_per_creator"]
seed     = cfg["seed"]

vids_f   = filter_creators(videos, "author_id", min_videos=min_vids)
train_df, test_df = train_test_split_by_creator(vids_f, "author_id", test_size=0.2, seed=seed)
train_ids = set(train_df["video_id"])
test_ids  = set(test_df["video_id"])
print(f"Filtered (≥{min_vids} videos): {len(vids_f):,} videos  {vids_f['author_id'].nunique():,} creators")
print(f"Train: {len(train_df):,}  |  Test: {len(test_df):,}")
"""

NB02_BUILD_TARGET = """\
abs_t = build_target(engagement, vids_f, day=14, mode="absolute")
rel_t = build_target(engagement, vids_f, day=14, mode="relative",
                     train_video_ids=train_ids)

for label, t in [("Absolute", abs_t), ("Relative", rel_t)]:
    print(f"{label}: n={len(t):,}  mean={t['target'].mean():.3f}"
          f"  std={t['target'].std():.3f}"
          f"  min={t['target'].min():.2f}  max={t['target'].max():.2f}")
"""

NB02_TARGET_FIG = """\
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, t, color, title in [
    (axes[0], abs_t, "steelblue",
     "Absolute: log1p(play_count) at day 14"),
    (axes[1], rel_t, "darkorange",
     "Relative: log1p(play_count) − creator median (train)"),
]:
    ax.hist(t["target"], bins=80, color=color, edgecolor="none", alpha=0.85)
    ax.axvline(t["target"].mean(), color="red", linewidth=1.2, linestyle="--",
               label=f"mean={t['target'].mean():.2f}")
    if color == "darkorange":
        ax.axvline(0, color="black", linewidth=0.8, linestyle=":", label="baseline=0")
    ax.set_xlabel("target value"); ax.set_ylabel("Videos")
    ax.set_title(title); ax.legend(fontsize=8)

fig.suptitle("Target distributions at day 14", fontweight="bold")
fig.tight_layout()
fig.savefig(FIGS / "fig3_target_distributions.png", dpi=120)
plt.show()
"""

NB02_R2 = """\
def _author_r2(target_df: pd.DataFrame) -> tuple[float, float]:
    \"\"\"R² of 'predict per-author train mean' model on test set.\"\"\"
    td = target_df.merge(vids_f[["video_id", "author_id"]], on="video_id", how="inner")
    tr = td[td["video_id"].isin(train_ids)]
    te = td[td["video_id"].isin(test_ids)].copy()
    author_means = tr.groupby("author_id")["target"].mean()
    te["pred"]   = te["author_id"].map(author_means).fillna(tr["target"].mean())
    ss_res = ((te["target"] - te["pred"]) ** 2).sum()
    ss_tot = ((te["target"] - te["target"].mean()) ** 2).sum()
    tr_pred = tr.copy()
    tr_pred["pred"] = tr_pred["author_id"].map(author_means)
    ss_res_tr = ((tr_pred["target"] - tr_pred["pred"]) ** 2).sum()
    ss_tot_tr = ((tr_pred["target"] - tr_pred["target"].mean()) ** 2).sum()
    return float(1 - ss_res_tr / ss_tot_tr), float(1 - ss_res / ss_tot)

r2_abs_tr, r2_abs_te = _author_r2(abs_t)
r2_rel_tr, r2_rel_te = _author_r2(rel_t)
print(f"R² (per-author mean model)  |  train  |  test")
print(f"  Absolute target            |  {r2_abs_tr:.3f}  |  {r2_abs_te:.3f}")
print(f"  Relative target            |  {r2_rel_tr:.3f}  |  {r2_rel_te:.3f}")
print(f"\\n→ {r2_abs_te:.1%} of absolute-target variance is author_id alone.")
print(f"→ Relative target reduces this to {max(r2_rel_te,0):.1%}.")
"""

NB02_R2_FIG = """\
fig, ax = plt.subplots(figsize=(5, 4))
labels  = ["Absolute\\n(test)", "Relative\\n(test)"]
heights = [r2_abs_te, max(r2_rel_te, 0)]
bars    = ax.bar(labels, heights, color=["steelblue", "darkorange"], width=0.4)
for bar, h in zip(bars, heights):
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}",
            ha="center", fontsize=11)
ax.set_ylim(0, max(heights) * 1.3)
ax.set_ylabel("R²")
ax.set_title('R² of "predict per-author mean" model on test set\\n'
             '(what fraction of variance is creator-level?)', fontsize=9)
fig.tight_layout()
fig.savefig(FIGS / "fig4_author_r2.png", dpi=120)
plt.show()
"""

NB02_FOLLOWER_MD = "## 5. Follower count at publish time (as-of join)"

NB02_FOLLOWER = """\
videos_with_fc = follower_asof_join(videos, creator_daily)
cov = videos_with_fc["follower_count_at_publish"].notna().mean()
print(f"Coverage: {cov*100:.1f}%  ({videos_with_fc['follower_count_at_publish'].notna().sum():,} videos)")

joined = (abs_t
          .merge(videos_with_fc[["video_id", "follower_count_at_publish"]], on="video_id")
          .dropna(subset=["follower_count_at_publish"]))
joined["log_fc"] = np.log1p(joined["follower_count_at_publish"].clip(0))
pearson  = joined["log_fc"].corr(joined["target"])
spearman = joined["log_fc"].corr(joined["target"], method="spearman")
print(f"Pearson r  = {pearson:.3f}  (n={len(joined):,})")
print(f"Spearman ρ = {spearman:.3f}")
"""

NB02_FOLLOWER_FIG = """\
sample = joined.sample(min(30_000, len(joined)), random_state=42)
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(sample["log_fc"], sample["target"],
           alpha=0.05, s=4, color="steelblue", rasterized=True)
ax.set_xlabel("log1p(follower_count at publish date)")
ax.set_ylabel("log1p(play_count at day 14)")
ax.set_title(f"Followers vs day-14 play_count\\n"
             f"Pearson r={pearson:.3f}  Spearman ρ={spearman:.3f}  n={len(joined):,}")
fig.tight_layout()
fig.savefig(FIGS / "fig5_follower_vs_target.png", dpi=120)
plt.show()
"""


# ================================================================== #
# Assemble and write                                                  #
# ================================================================== #

META = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.12.7",
    },
}


def make_nb(cells: list) -> nbformat.NotebookNode:
    nb = new_notebook(cells=cells)
    nb["metadata"] = META
    return nb


nb01 = make_nb([
    new_markdown_cell(NB01_GOAL),
    new_code_cell(COLAB_GUARD),
    new_code_cell(SETUP_COMMON),
    new_code_cell(NB01_LOAD),
    new_markdown_cell(NB01_SCHEMA_MD),
    new_code_cell(NB01_SCHEMA),
    new_markdown_cell(NB01_DATERANGE_MD),
    new_code_cell(NB01_DATERANGE),
    new_markdown_cell(NB01_CREATORS_MD),
    new_code_cell(NB01_CREATORS),
    new_code_cell(NB01_CREATORS_FIG),
    new_code_cell(NB01_THRESHOLD),
    new_markdown_cell(NB01_JOIN_MD),
    new_code_cell(NB01_JOIN),
    new_markdown_cell(NB01_ENG_MD),
    new_code_cell(NB01_ENG_DSP),
    new_code_cell(NB01_ENG_METRICS),
    new_code_cell(NB01_ENG_FIG),
])

nb02 = make_nb([
    new_markdown_cell(NB02_GOAL),
    new_code_cell(COLAB_GUARD),
    new_code_cell(SETUP_COMMON),
    new_code_cell(NB02_LOAD),
    new_markdown_cell(NB02_COV_MD),
    new_code_cell(NB02_COV),
    new_markdown_cell(NB02_COV_NOTE),
    new_code_cell(NB02_COV_FIG),
    new_markdown_cell(NB02_MONO_MD),
    new_code_cell(NB02_MONO),
    new_markdown_cell(NB02_RATIO_MD),
    new_code_cell(NB02_RATIO),
    new_code_cell(NB02_BIAS),
    new_markdown_cell(NB02_TARGET_MD),
    new_code_cell(NB02_SPLIT),
    new_code_cell(NB02_BUILD_TARGET),
    new_code_cell(NB02_TARGET_FIG),
    new_code_cell(NB02_R2),
    new_code_cell(NB02_R2_FIG),
    new_markdown_cell(NB02_FOLLOWER_MD),
    new_code_cell(NB02_FOLLOWER),
    new_code_cell(NB02_FOLLOWER_FIG),
])

nbformat.write(nb01, NB_DIR / "01-data-exploration.ipynb")
nbformat.write(nb02, NB_DIR / "02-target-selection.ipynb")
print("Written notebooks/01-data-exploration.ipynb")
print("Written notebooks/02-target-selection.ipynb")

# ================================================================== #
# NOTEBOOK 03: Feature Engineering                                    #
# ================================================================== #

NB03_GOAL = """\
# 03 — Feature Engineering

**Goal** – Build a single feature matrix for the TikTok engagement prediction task,
analyse each feature's distribution and null rate, identify near-duplicate feature
pairs (|r| > 0.8), compute a Spearman-correlation sanity check against the relative
target (train only), and flag features that are (nearly) constant within creators
— those are uninformative for per-creator models even if useful globally.

**Input** – `data/raw/*.parquet` + `config/default.yaml`.
Non-monotonic videos and creators below the min-threshold are excluded before
building targets and features.

**Conclusion** –
Starting with 209 543 videos, 1 240 non-monotonic videos are removed (data-reset
artefacts), then 1 687 videos from creators with < 30 videos are dropped, leaving
**200 616 videos × 1 437 creators**. The feature matrix has **54 columns** (50
numeric, 4 categorical: `ratio`, `topic`, `music_selected_from`, `dominant_emotion`).

*Null rates*: Hook features (`hook_first_sentence_*`) and `transcript_n_words` have
~59.8 % null (matches `has_transcript = 0`). `speaking_rate` has 37.7 % null.
All other features are ≤ 1.4 % null. Nulls are not imputed — XGBoost handles them.

*Near-duplicate pairs* (|Pearson r| > 0.8):
`transcript_n_words ↔ hook_first_sentence_len` (r = 0.94),
`desc_len ↔ desc_n_words` (r = 0.93),
`has_transcript ↔ speaking_rate` (r = 0.90),
`duration_log ↔ transcript_n_words` (r = 0.82).
Recommendation: keep one from each pair; drop `hook_first_sentence_len` (already
noted as unreliable), `desc_n_words`, and use `has_transcript` as the binary flag
while keeping `speaking_rate` for videos that have it.

*Per-creator constancy* (features useless for per-creator model):
`is_ads` (always 0 — zero variance globally too) and
`hook_first_sentence_is_question` (100 % of creators have constant value — mostly
NaN). All top-20 hashtag indicators have 75–89 % of creators constant (hashtags
are per-creator niche, not within-creator signal). `created_by_ai` is constant for
86 % of creators. These features are valuable for the **global** model but should be
excluded from per-creator models.
"""

NB03_LOAD = """\
from creator_analytics.config import load_config
from creator_analytics.data import (
    filter_creators, load_raw_table, train_test_split_by_creator,
)
from creator_analytics.features import (
    FEATURE_DESCRIPTIONS, build_features, follower_asof_join,
)
from creator_analytics.targets import build_target, get_nonmonotonic_video_ids

cfg    = load_config("config/default.yaml")
RAW    = cfg["paths"]["raw_dir"]
FIGS   = Path("results/figures")
FIGS.mkdir(parents=True, exist_ok=True)

videos        = load_raw_table(RAW, "videos")
creator_daily = load_raw_table(RAW, "creator_daily")
engagement    = load_raw_table(RAW, "engagement_daily")
print(f"videos {len(videos):,}  creator_daily {len(creator_daily):,}  engagement {len(engagement):,}")
"""

NB03_STEP0_MD = "## Step 0 — Data quality checks and pre-filtering"

NB03_STEP0 = """\
# --- Fix 1: non-monotonic videos ---
bad_ids  = get_nonmonotonic_video_ids(engagement)
print(f"Non-monotonic videos (play_count drops): {len(bad_ids):,}")
videos_clean = videos[~videos["video_id"].isin(bad_ids)].reset_index(drop=True)
print(f"Videos after removing non-monotonic: {len(videos_clean):,}")

# --- Fix 2: filter creators ---
min_vids = cfg["data"]["min_videos_per_creator"]
videos_f = filter_creators(videos_clean, "author_id", min_videos=min_vids)
print(f"Videos after min={min_vids} filter: {len(videos_f):,}  "
      f"creators: {videos_f['author_id'].nunique():,}")

# --- Fix 3: build_target scope check ---
abs_t_all   = build_target(engagement, videos,   day=14, mode="absolute")
abs_t_filt  = build_target(engagement, videos_f, day=14, mode="absolute")
print(f"\\nbuild_target scope check:")
print(f"  all videos     → {len(abs_t_all):,} rows")
print(f"  filtered videos → {len(abs_t_filt):,} rows  (must equal {len(videos_f):,})")
assert len(abs_t_filt) == len(videos_f), "build_target returned more rows than videos passed!"
"""

NB03_FOLLOWER_MD = "## Step 0b — follower_asof_join: as-of direction and coverage"

NB03_FOLLOWER_DIAG = """\
import pandas as pd

vids_diag = videos_f[["video_id", "author_id", "create_date"]].copy()
vids_diag["_vdt"] = pd.to_datetime(vids_diag["create_date"])

crd_diag  = creator_daily[["author_id", "date", "follower_count"]].copy()
crd_diag["_cdt"] = pd.to_datetime(crd_diag["date"])

vids_s = vids_diag.sort_values("_vdt")
crd_s  = crd_diag.dropna(subset=["_cdt"]).sort_values("_cdt")

merged_diag = pd.merge_asof(
    vids_s,
    crd_s[["author_id", "_cdt", "follower_count"]],
    left_on="_vdt", right_on="_cdt",
    by="author_id", direction="backward",
)
merged_diag["days_diff"] = (merged_diag["_vdt"] - merged_diag["_cdt"]).dt.days

no_prior = merged_diag["follower_count"].isna().sum()
print(f"Videos with no creator_daily record before publish: {no_prior:,}  "
      f"({no_prior / len(merged_diag) * 100:.2f}%)")
print(f"\\nDays between publish date and matched creator_daily record:")
dd = merged_diag["days_diff"].dropna()
for q in [0, 0.25, 0.5, 0.75, 0.95, 1.0]:
    print(f"  p{q:.0%} = {dd.quantile(q):.0f} days")
"""

NB03_FOLLOWER_FIG = """\
fig, ax = plt.subplots(figsize=(8, 3))
ax.hist(merged_diag["days_diff"].dropna().clip(upper=90), bins=60,
        color="steelblue", edgecolor="none")
ax.set_xlabel("Days between publish date and matched creator_daily snapshot (capped at 90)")
ax.set_ylabel("Videos")
ax.set_title("as-of join: staleness of follower snapshot at publish time")
ax.axvline(merged_diag["days_diff"].median(), color="red", linestyle="--",
           linewidth=1.2, label=f"median={merged_diag['days_diff'].median():.0f}d")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGS / "nb03_follower_join_staleness.png", dpi=120)
plt.show()
"""

NB03_SPLIT_MD = "## Split — train / test"

NB03_SPLIT = """\
seed = cfg["seed"]
train_df, test_df = train_test_split_by_creator(videos_f, "author_id",
                                                test_size=0.2, seed=seed)
train_ids = set(train_df["video_id"])
test_ids  = set(test_df["video_id"])
print(f"Train: {len(train_df):,}  Test: {len(test_df):,}")
"""

NB03_BUILD_MD = "## Build feature matrix"

NB03_BUILD = """\
features = build_features(videos_f, creator_daily, train_video_ids=train_ids)
print(f"Feature matrix: {features.shape[0]:,} rows × {features.shape[1]} columns")

# Save to data/processed/
out_path = Path("data/processed/features.parquet")
out_path.parent.mkdir(parents=True, exist_ok=True)
features.to_parquet(str(out_path), index=False)
print(f"Saved → {out_path}")
"""

NB03_TARGET = """\
rel_t = build_target(engagement, videos_f, day=14, mode="relative",
                     train_video_ids=train_ids)
print(f"Relative target: {len(rel_t):,} rows  mean={rel_t['target'].mean():.4f}")
"""

NB03_PROFILE_MD = "## Feature profiles: type, null %, distribution"

NB03_PROFILE = """\
feat_cols = [c for c in features.columns if c != "video_id"]
# Use select_dtypes so Arrow-backed string columns are correctly classified
num_cols = features[feat_cols].select_dtypes(include=np.number).columns.tolist()
cat_cols = [c for c in feat_cols if c not in num_cols]

profile_rows = []
for col in feat_cols:
    s = features[col]
    profile_rows.append({
        "feature": col,
        "dtype": str(s.dtype),
        "null_%": round(s.isna().mean() * 100, 1),
        "n_unique": s.nunique() if s.dtype != object else s.nunique(),
        "description": FEATURE_DESCRIPTIONS.get(col, ""),
    })
profile = pd.DataFrame(profile_rows).set_index("feature")
print(f"Total features: {len(feat_cols)}")
print(f"  Numeric: {len(num_cols)}")
print(f"  Categorical (object): {len(cat_cols)}")
print(f"  Features with >50% nulls:")
print(profile[profile["null_%"] > 50][["null_%", "dtype"]].to_string())
"""

NB03_PROFILE_FIG = """\
null_series = profile["null_%"].sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(14, max(4, len(null_series) * 0.18)))
colors = ["tomato" if v > 50 else "steelblue" for v in null_series]
ax.barh(null_series.index, null_series.values, color=colors, height=0.8)
ax.set_xlabel("Null % per feature")
ax.set_title("Feature null rates  (red = >50%)")
ax.axvline(50, color="red", linestyle="--", linewidth=0.8)
fig.tight_layout()
fig.savefig(FIGS / "nb03_null_rates.png", dpi=120)
plt.show()
"""

NB03_DIST_FIG = """\
# Distribution histograms for numeric features (up to 30)
num_show = [c for c in num_cols if features[c].notna().sum() > 10][:30]
n = len(num_show)
ncols = 5
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.5))
for ax, col in zip(axes.flat, num_show):
    data = features[col].dropna()
    ax.hist(data, bins=40, color="steelblue", edgecolor="none")
    ax.set_title(col, fontsize=7)
    ax.tick_params(labelsize=6)
for ax in axes.flat[len(num_show):]:
    ax.set_visible(False)
fig.suptitle("Numeric feature distributions", fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(FIGS / "nb03_numeric_distributions.png", dpi=120, bbox_inches="tight")
plt.show()
"""

NB03_CORR_MD = "## Correlation matrix — near-duplicate features (|r| > 0.8)"

NB03_CORR = """\
feat_num = features[num_cols].select_dtypes(include=[np.number])
corr = feat_num.corr(method="pearson", min_periods=100)

# Find pairs with |r| > 0.8 (excluding diagonal)
mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
high_corr = (
    corr.where(mask)
    .stack()
    .reset_index()
    .rename(columns={"level_0": "feature_a", "level_1": "feature_b", 0: "pearson_r"})
)
high_corr = high_corr[high_corr["pearson_r"].abs() > 0.8].sort_values("pearson_r", ascending=False)
print(f"Feature pairs with |Pearson r| > 0.8: {len(high_corr)}")
if len(high_corr):
    display(high_corr.to_string(index=False))
"""

NB03_CORR_FIG = """\
import matplotlib.colors as mcolors

n_num = min(len(num_cols), 40)
sub_cols = num_cols[:n_num]
sub_corr = features[sub_cols].select_dtypes(include=[np.number]).corr(min_periods=50)

fig, ax = plt.subplots(figsize=(n_num * 0.45 + 1, n_num * 0.45 + 1))
im = ax.imshow(sub_corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
ax.set_xticks(range(len(sub_cols))); ax.set_xticklabels(sub_cols, rotation=90, fontsize=6)
ax.set_yticks(range(len(sub_cols))); ax.set_yticklabels(sub_cols, fontsize=6)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_title("Pearson correlation — numeric features (first 40)")
fig.tight_layout()
fig.savefig(FIGS / "nb03_correlation_matrix.png", dpi=120, bbox_inches="tight")
plt.show()
"""

NB03_SPEARMAN_MD = "## Spearman correlation with relative target (train only — sanity check)"

NB03_SPEARMAN = """\
merged_train = (features[features["video_id"].isin(train_ids)]
                .merge(rel_t, on="video_id"))

spearman_rows = []
for col in num_cols:
    sub = merged_train[["target", col]].dropna()
    if len(sub) < 50:
        continue
    r = sub["target"].corr(sub[col], method="spearman")
    spearman_rows.append({"feature": col, "spearman_r": round(r, 4)})

spearman_df = (pd.DataFrame(spearman_rows)
               .sort_values("spearman_r", key=abs, ascending=False)
               .reset_index(drop=True))

print("Top features by |Spearman r| with relative target (train):")
display(spearman_df.head(20).to_string(index=False))
print(f"\\nTotal numeric features with >=50 train rows: {len(spearman_df)}")
"""

NB03_SPEARMAN_FIG = """\
top_n = min(30, len(spearman_df))
fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
sp_top = spearman_df.head(top_n).sort_values("spearman_r")
colors = ["steelblue" if v >= 0 else "tomato" for v in sp_top["spearman_r"]]
ax.barh(sp_top["feature"], sp_top["spearman_r"], color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Spearman ρ with relative target")
ax.set_title(f"Top {top_n} features by |Spearman ρ| — train set only")
fig.tight_layout()
fig.savefig(FIGS / "nb03_spearman_vs_target.png", dpi=120)
plt.show()
"""

NB03_CREATOR_MD = "## Per-creator constancy — which features don't vary within a creator?"

NB03_CREATOR = """\
# For each feature, compute the fraction of creators where the feature
# is (nearly) constant — std == 0 or only 1 unique non-null value.
# A feature constant within a creator is useless for a per-creator model.

feat_w_author = features.merge(videos_f[["video_id", "author_id"]], on="video_id")

constancy_rows = []
for col in feat_cols:
    if col == "video_id":
        continue
    null_pct = features[col].isna().mean() * 100
    if features[col].dtype == object:
        # categorical: constant if all values the same within creator
        per_creator_nunique = feat_w_author.groupby("author_id")[col].nunique()
    else:
        per_creator_nunique = (
            feat_w_author.groupby("author_id")[col]
            .apply(lambda s: s.dropna().nunique() if s.notna().any() else 0)
        )
    frac_constant = (per_creator_nunique <= 1).mean()
    constancy_rows.append({
        "feature": col,
        "null_%": round(null_pct, 1),
        "frac_creators_constant": round(frac_constant, 3),
    })

const_df = (pd.DataFrame(constancy_rows)
            .sort_values("frac_creators_constant", ascending=False)
            .reset_index(drop=True))

print("Features sorted by fraction of creators with constant value:")
display(const_df.to_string(index=False))
"""

NB03_CREATOR_FIG = """\
fig, ax = plt.subplots(figsize=(14, max(4, len(const_df) * 0.18)))
colors = ["tomato" if v > 0.9 else ("orange" if v > 0.5 else "steelblue")
          for v in const_df["frac_creators_constant"]]
ax.barh(const_df["feature"], const_df["frac_creators_constant"], color=colors, height=0.8)
ax.axvline(0.9, color="red", linestyle="--", linewidth=0.8, label=">90% creators constant (useless for per-creator)")
ax.axvline(0.5, color="orange", linestyle="--", linewidth=0.8, label=">50%")
ax.set_xlabel("Fraction of creators where feature is constant")
ax.set_title("Per-creator feature constancy  (red = >90%: useless for per-creator model)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGS / "nb03_creator_constancy.png", dpi=120)
plt.show()
"""

NB03_SUMMARY_MD = "## Summary table: null % and per-creator constancy"

NB03_SUMMARY = """\
summary = (profile[["null_%", "dtype"]]
           .join(const_df.set_index("feature")[["frac_creators_constant"]])
           .sort_values("frac_creators_constant", ascending=False))
print("Full feature audit table:")
display(summary)
"""


nb03 = make_nb([
    new_markdown_cell(NB03_GOAL),
    new_code_cell(COLAB_GUARD),
    new_code_cell(SETUP_COMMON),
    new_code_cell(NB03_LOAD),
    new_markdown_cell(NB03_STEP0_MD),
    new_code_cell(NB03_STEP0),
    new_markdown_cell(NB03_FOLLOWER_MD),
    new_code_cell(NB03_FOLLOWER_DIAG),
    new_code_cell(NB03_FOLLOWER_FIG),
    new_markdown_cell(NB03_SPLIT_MD),
    new_code_cell(NB03_SPLIT),
    new_markdown_cell(NB03_BUILD_MD),
    new_code_cell(NB03_BUILD),
    new_code_cell(NB03_TARGET),
    new_markdown_cell(NB03_PROFILE_MD),
    new_code_cell(NB03_PROFILE),
    new_code_cell(NB03_PROFILE_FIG),
    new_code_cell(NB03_DIST_FIG),
    new_markdown_cell(NB03_CORR_MD),
    new_code_cell(NB03_CORR),
    new_code_cell(NB03_CORR_FIG),
    new_markdown_cell(NB03_SPEARMAN_MD),
    new_code_cell(NB03_SPEARMAN),
    new_code_cell(NB03_SPEARMAN_FIG),
    new_markdown_cell(NB03_CREATOR_MD),
    new_code_cell(NB03_CREATOR),
    new_code_cell(NB03_CREATOR_FIG),
    new_markdown_cell(NB03_SUMMARY_MD),
    new_code_cell(NB03_SUMMARY),
])

nbformat.write(nb03, NB_DIR / "03-feature-engineering.ipynb")
print("Written notebooks/03-feature-engineering.ipynb")


# ================================================================== #
# NOTEBOOK 04: Baseline Model                                         #
# ================================================================== #

NB04_GOAL = """\
# 04 — Baseline Global Model

**Goal** – Train and evaluate a global baseline model predicting
per-creator-relative video engagement at day 14.

Steps:
1. Load train/test splits and the pre-built feature matrix.
2. Compute trivial baselines (predict 0, predict creator mean).
3. Hyperparameter search: RandomForest and XGBoost (with / without follower count).
4. Final evaluation on the held-out test set.
5. Summary: metrics table, predicted vs actual, error distribution, per-creator RMSE.
"""

NB04_LOAD = """\
import pandas as pd
import numpy as np

from creator_analytics.config import load_config
from creator_analytics.data import filter_creators, load_raw_table, train_test_split_by_creator
from creator_analytics.features import build_features
from creator_analytics.models import prepare_X
from creator_analytics.targets import build_target, get_nonmonotonic_video_ids
from creator_analytics.utils import set_seed

config = load_config("config/default.yaml")
set_seed(config["seed"])

RAW = config["paths"]["raw_dir"]
min_vids = config["data"]["min_videos_per_creator"]
SEED = config["seed"]

videos = load_raw_table(RAW, "videos")
creator_daily = load_raw_table(RAW, "creator_daily")
engagement = load_raw_table(RAW, "engagement_daily")

bad_ids = get_nonmonotonic_video_ids(engagement)
videos_clean = videos[~videos["video_id"].isin(bad_ids)].reset_index(drop=True)
videos_f = filter_creators(videos_clean, "author_id", min_videos=min_vids)

train_df, test_df = train_test_split_by_creator(
    videos_f, "author_id", test_size=config["split"]["test_size"], seed=SEED
)
train_ids = set(train_df["video_id"])
test_ids  = set(test_df["video_id"])

from pathlib import Path
feat_path = Path("data/processed/features.parquet")
if feat_path.exists():
    features = pd.read_parquet(feat_path)
else:
    features = build_features(videos_f, creator_daily, train_video_ids=train_ids)
    feat_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(str(feat_path), index=False)

rel_t = build_target(engagement, videos_f, day=14, mode="relative", train_video_ids=train_ids)
data = (features
        .merge(videos_f[["video_id", "author_id"]], on="video_id")
        .merge(rel_t, on="video_id"))

train_mask = data["video_id"].isin(train_ids)
test_mask  = data["video_id"].isin(test_ids)

X_all = prepare_X(data.drop(columns=["author_id", "target"]))
X_train = X_all[train_mask].reset_index(drop=True)
y_train = data.loc[train_mask, "target"].reset_index(drop=True)
author_train = data.loc[train_mask, "author_id"].reset_index(drop=True)

X_test = X_all[test_mask].reset_index(drop=True)
y_test = data.loc[test_mask, "target"].reset_index(drop=True)
author_test = data.loc[test_mask, "author_id"].reset_index(drop=True)

print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows | Features: {X_train.shape[1]}")
"""

NB04_TRIVIAL_MD = """\
## 1. Trivial Baselines

Two baselines that require no ML:

| Model | Prediction |
|-------|-----------|
| **predict_0** | Always 0 (= per-creator train median on relative target) |
| **predict_creator_mean** | Per-creator train mean |
"""

NB04_TRIVIAL = """\
from creator_analytics.models import trivial_zero_rmse, trivial_creator_mean

rmse_zero = trivial_zero_rmse(y_test)
rmse_cmean, preds_cmean = trivial_creator_mean(y_train, author_train, y_test, author_test)
print(f"predict_0 RMSE:            {rmse_zero:.4f}")
print(f"predict_creator_mean RMSE: {rmse_cmean:.4f}")
"""

NB04_RF_MD = """\
## 2. RandomForest — Hyperparameter Search

StratifiedKFold (k=5) by author_id, 30 random iterations, RMSE.
"""

NB04_RF = """\
from creator_analytics.models import search_random_forest

n_splits = config["cv"]["n_splits"]
rf_model = search_random_forest(X_train, y_train, author_train,
                                 n_splits=n_splits, seed=SEED, n_iter=30)
print(f"RF best CV RMSE: {rf_model.cv_rmse:.4f}")
print(f"Best params: {rf_model.params}")
"""

NB04_XGB_MD = """\
## 3. XGBoost — Hyperparameter Search (with and without follower count)

Manual random search with per-fold early stopping (early_stopping_rounds=50) on
the CV validation fold — **test set is never seen during search**.
"""

NB04_XGB = """\
from creator_analytics.models import search_xgboost

xgb_with = search_xgboost(X_train, y_train, author_train,
                            n_splits=n_splits, seed=SEED, n_iter=30)
xgb_nof  = search_xgboost(X_train, y_train, author_train,
                            n_splits=n_splits, seed=SEED, n_iter=30,
                            exclude_cols=["follower_count_log"])

print(f"XGB (with follower) CV RMSE: {xgb_with.cv_rmse:.4f}")
print(f"XGB (no  follower ) CV RMSE: {xgb_nof.cv_rmse:.4f}")
follower_delta = xgb_with.cv_rmse - xgb_nof.cv_rmse
print(f"Follower impact on CV RMSE: {follower_delta:+.4f}")

best_xgb = xgb_with if xgb_with.cv_rmse <= xgb_nof.cv_rmse else xgb_nof
"""

NB04_CV_MD = """\
## 4. CV Summary — All Models
"""

NB04_CV_TABLE = """\
import pandas as pd

cv_summary = pd.DataFrame([
    {"model": "predict_0",               "cv_rmse": None,            "test_rmse": rmse_zero},
    {"model": "predict_creator_mean",    "cv_rmse": None,            "test_rmse": rmse_cmean},
    {"model": "RandomForest",            "cv_rmse": rf_model.cv_rmse, "test_rmse": None},
    {"model": "XGB_with_follower",       "cv_rmse": xgb_with.cv_rmse,"test_rmse": None},
    {"model": "XGB_no_follower",         "cv_rmse": xgb_nof.cv_rmse, "test_rmse": None},
]).set_index("model")
display(cv_summary.style.format({"cv_rmse": "{:.4f}", "test_rmse": "{:.4f}"}, na_rep="—"))
"""

NB04_TEST_MD = """\
## 5. Final Test Evaluation

The test set is touched **exactly once** — here.
"""

NB04_TEST = """\
from creator_analytics.evaluate import compute_metrics, compute_per_creator_metrics

# Trivial baselines
m_zero  = compute_metrics(y_test, np.zeros(len(y_test)))
m_cmean = compute_metrics(y_test, preds_cmean)

# RF
y_pred_rf = rf_model.predict(X_test)
m_rf = compute_metrics(y_test, y_pred_rf)

# XGB with follower
y_pred_xgb_with = xgb_with.predict(X_test)
m_xgb_with = compute_metrics(y_test, y_pred_xgb_with)

# XGB without follower
y_pred_xgb_nof = xgb_nof.predict(X_test)
m_xgb_nof = compute_metrics(y_test, y_pred_xgb_nof)

# Best model
y_pred_best = best_xgb.predict(X_test)
m_best = compute_metrics(y_test, y_pred_best)

results_df = pd.DataFrame({
    "predict_0":            m_zero,
    "predict_creator_mean": m_cmean,
    "RandomForest":         m_rf,
    "XGB_with_follower":    m_xgb_with,
    "XGB_no_follower":      m_xgb_nof,
}).T[["rmse","mae","r2","spearman","n"]]

display(results_df.style.format({"rmse":"{:.4f}","mae":"{:.4f}","r2":"{:.4f}","spearman":"{:.4f}"}))
"""

NB04_FIGURES_MD = """\
## 6. Diagnostics
"""

NB04_FIG_PRED_VS_ACTUAL = """\
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (name, y_pred) in zip(axes, [
    ("RF", y_pred_rf),
    ("XGB (best)", y_pred_best),
]):
    ax.scatter(y_test, y_pred, alpha=0.15, s=8, linewidths=0)
    lim = max(abs(y_test.min()), abs(y_test.max()))
    ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1)
    ax.set_xlabel("Actual (relative target)")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{name}: predicted vs actual")

plt.tight_layout()
plt.show()
"""

NB04_FIG_ERROR = """\
fig, ax = plt.subplots(figsize=(9, 4))
errors_rf   = y_test.values - y_pred_rf
errors_best = y_test.values - y_pred_best
ax.hist(errors_rf,   bins=80, alpha=0.5, label="RF",       density=True)
ax.hist(errors_best, bins=80, alpha=0.5, label="XGB best", density=True)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Residual (actual − predicted)")
ax.set_ylabel("Density")
ax.set_title("Residual distribution")
ax.legend()
plt.tight_layout()
plt.show()
"""

NB04_PER_CREATOR_MD = """\
## 7. Per-Creator Analysis
"""

NB04_PER_CREATOR = """\
per_creator = compute_per_creator_metrics(y_test, pd.Series(y_pred_best), author_test)
print(f"Creators in test: {len(per_creator)}")
print(f"\\nTop 5 worst (highest RMSE):")
display(per_creator.nlargest(5, "rmse"))
print(f"\\nTop 5 best (lowest RMSE):")
display(per_creator.nsmallest(5, "rmse"))
"""

NB04_PER_CREATOR_FIG = """\
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(per_creator["rmse"], bins=40, edgecolor="none")
ax.axvline(m_best["rmse"], color="red", linestyle="--", linewidth=1.2, label="Global RMSE")
ax.axvline(rmse_zero, color="orange", linestyle=":", linewidth=1.2, label="predict_0 RMSE")
ax.set_xlabel("Per-creator test RMSE")
ax.set_ylabel("Number of creators")
ax.set_title("Distribution of per-creator RMSE (XGB best)")
ax.legend()
plt.tight_layout()
plt.show()

n_worse = (per_creator["rmse"] > rmse_zero).sum()
print(f"{n_worse} / {len(per_creator)} creators where model is worse than predict_0")
"""

NB04_SAVE_MD = """\
## 8. Save Artefacts
"""

NB04_SAVE = """\
from pathlib import Path
from creator_analytics.evaluate import log_experiment, save_metrics
import datetime

today = datetime.date.today().isoformat()
exp_path = config["paths"].get("experiments_log", "results/experiments.md")
metrics_path = config["paths"].get("metrics", "results/metrics.json")

for name, m_base in [("predict_0", m_zero), ("predict_creator_mean", m_cmean)]:
    log_experiment(exp_path, name, today, "none", name, {}, None,
                   {"rmse": m_base["rmse"], "r2": m_base["r2"],
                    "spearman": m_base["spearman"]})

log_experiment(exp_path, "RF_global", today, "all_54", "RandomForest",
               rf_model.params, rf_model.cv_rmse, m_rf)
log_experiment(exp_path, "XGB_with_follower", today, "all_54", "XGBoost",
               xgb_with.params, xgb_with.cv_rmse, m_xgb_with)
log_experiment(exp_path, "XGB_no_follower", today, "all_54_no_follower", "XGBoost",
               xgb_nof.params, xgb_nof.cv_rmse, m_xgb_nof)

save_metrics({
    "trivial_predict_0_rmse":       m_zero["rmse"],
    "trivial_creator_mean_rmse":    m_cmean["rmse"],
    "rf_cv_rmse":                   rf_model.cv_rmse,
    "rf_test_rmse":                 m_rf["rmse"],
    "xgb_with_follower_cv_rmse":    xgb_with.cv_rmse,
    "xgb_with_follower_test_rmse":  m_xgb_with["rmse"],
    "xgb_no_follower_cv_rmse":      xgb_nof.cv_rmse,
    "xgb_no_follower_test_rmse":    m_xgb_nof["rmse"],
}, metrics_path)

# Save best model
best_xgb.model_name = "global_xgb_best"
best_xgb.save("models/global_xgb.joblib")

# Save predictions
pred_path = Path("results/predictions_global.parquet")
pred_path.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame({
    "y_true": y_test.values, "y_pred": y_pred_best, "author_id": author_test.values
}).to_parquet(str(pred_path), index=False)

per_creator.to_parquet("results/per_creator_metrics_global.parquet", index=False)

print("Saved: models/global_xgb.joblib")
print("Saved: results/metrics.json")
print("Saved: results/predictions_global.parquet")
print("Saved: results/per_creator_metrics_global.parquet")
print("Saved: results/experiments.md")
"""

NB04_SUMMARY_MD = """\
## Summary

| Model | CV RMSE | Test RMSE | Test MAE | Test R² | Test Spearman |
|-------|---------|-----------|----------|---------|---------------|
"""

NB04_SUMMARY = """\
summary_rows = [
    ("predict_0",            None,             m_zero["rmse"],    m_zero["mae"],    m_zero["r2"],    float("nan")),
    ("predict_creator_mean", None,             m_cmean["rmse"],   m_cmean["mae"],   m_cmean["r2"],   float("nan")),
    ("RandomForest",         rf_model.cv_rmse, m_rf["rmse"],      m_rf["mae"],      m_rf["r2"],      m_rf["spearman"]),
    ("XGB_with_follower",    xgb_with.cv_rmse, m_xgb_with["rmse"],m_xgb_with["mae"],m_xgb_with["r2"],m_xgb_with["spearman"]),
    ("XGB_no_follower",      xgb_nof.cv_rmse,  m_xgb_nof["rmse"], m_xgb_nof["mae"], m_xgb_nof["r2"], m_xgb_nof["spearman"]),
]
summary_df = pd.DataFrame(summary_rows,
    columns=["model", "cv_rmse", "test_rmse", "test_mae", "test_r2", "test_spearman"]
).set_index("model")
display(summary_df.style.format({
    "cv_rmse":      "{:.4f}", "test_rmse": "{:.4f}", "test_mae": "{:.4f}",
    "test_r2":      "{:.4f}", "test_spearman": "{:.4f}",
}, na_rep="—"))

follower_impact = m_xgb_with["rmse"] - m_xgb_nof["rmse"]
print(f"\\nFollower count impact on test RMSE: {follower_impact:+.4f}")
print(f"(negative = model WITH follower is better)")

n_worse = (per_creator["rmse"] > rmse_zero).sum()
print(f"\\nCreators where model > predict_0 RMSE: {n_worse} / {len(per_creator)}")
print(f"Best hyperparams (XGB best):")
for k, v in best_xgb.params.items():
    print(f"  {k}: {v}")
"""

nb04 = make_nb([
    new_markdown_cell(NB04_GOAL),
    new_code_cell(COLAB_GUARD),
    new_code_cell(SETUP_COMMON),
    new_markdown_cell("## 0. Load Data and Build Splits"),
    new_code_cell(NB04_LOAD),
    new_markdown_cell(NB04_TRIVIAL_MD),
    new_code_cell(NB04_TRIVIAL),
    new_markdown_cell(NB04_RF_MD),
    new_code_cell(NB04_RF),
    new_markdown_cell(NB04_XGB_MD),
    new_code_cell(NB04_XGB),
    new_markdown_cell(NB04_CV_MD),
    new_code_cell(NB04_CV_TABLE),
    new_markdown_cell(NB04_TEST_MD),
    new_code_cell(NB04_TEST),
    new_markdown_cell(NB04_FIGURES_MD),
    new_code_cell(NB04_FIG_PRED_VS_ACTUAL),
    new_code_cell(NB04_FIG_ERROR),
    new_markdown_cell(NB04_PER_CREATOR_MD),
    new_code_cell(NB04_PER_CREATOR),
    new_code_cell(NB04_PER_CREATOR_FIG),
    new_markdown_cell(NB04_SAVE_MD),
    new_code_cell(NB04_SAVE),
    new_markdown_cell(NB04_SUMMARY_MD),
    new_code_cell(NB04_SUMMARY),
])

nbformat.write(nb04, NB_DIR / "04-baseline-model.ipynb")
print("Written notebooks/04-baseline-model.ipynb")


# ================================================================== #
# NOTEBOOK 05: Per-Creator Models                                     #
# ================================================================== #

NB05_GOAL = """\
# 05 — Per-Creator Models

**Goal** – Train one personalized model per creator (1 439 models) and compare
with the global baseline.

Steps:
1. Load the same train/test split and feature matrix as NB04.
2. Choose one conservative config via CV on 100 sampled creators.
3. Train RF and XGBoost per-creator (in parallel).
4. Evaluate on the held-out test set.
5. Compare with global model.
6. Analyse: RMSE distribution, quality vs. creator size, flag worse-than-zero.
"""

NB05_LOAD = """\
import pandas as pd
import numpy as np
from pathlib import Path

from creator_analytics.config import load_config
from creator_analytics.data import filter_creators, load_raw_table, train_test_split_by_creator
from creator_analytics.features import build_features
from creator_analytics.models import prepare_X
from creator_analytics.targets import build_target, get_nonmonotonic_video_ids
from creator_analytics.utils import set_seed

config = load_config("config/default.yaml")
set_seed(config["seed"])
SEED   = config["seed"]
RAW    = config["paths"]["raw_dir"]
N_JOBS = -1

videos        = load_raw_table(RAW, "videos")
creator_daily = load_raw_table(RAW, "creator_daily")
engagement    = load_raw_table(RAW, "engagement_daily")

bad_ids  = get_nonmonotonic_video_ids(engagement)
videos_f = filter_creators(
    videos[~videos["video_id"].isin(bad_ids)].reset_index(drop=True),
    "author_id", config["data"]["min_videos_per_creator"]
)

train_df, test_df = train_test_split_by_creator(
    videos_f, "author_id", test_size=config["split"]["test_size"], seed=SEED
)
train_ids = set(train_df["video_id"])
test_ids  = set(test_df["video_id"])

feat_path = Path("data/processed/features.parquet")
features  = pd.read_parquet(feat_path) if feat_path.exists() else build_features(
    videos_f, creator_daily, train_video_ids=train_ids
)

rel_t = build_target(engagement, videos_f, day=14, mode="relative", train_video_ids=train_ids)
data  = (features
         .merge(videos_f[["video_id", "author_id"]], on="video_id")
         .merge(rel_t, on="video_id"))

train_mask = data["video_id"].isin(train_ids)
test_mask  = data["video_id"].isin(test_ids)

X_all = prepare_X(data.drop(columns=["author_id", "target"]))

X_train      = X_all[train_mask].reset_index(drop=True)
y_train      = data.loc[train_mask, "target"].reset_index(drop=True)
author_train = data.loc[train_mask, "author_id"].reset_index(drop=True)

X_test       = X_all[test_mask].reset_index(drop=True)
y_test       = data.loc[test_mask, "target"].reset_index(drop=True)
author_test  = data.loc[test_mask, "author_id"].reset_index(drop=True)

n_creators = author_train.nunique()
print(f"Train: {len(X_train):,} rows  Test: {len(X_test):,} rows  Creators: {n_creators}")
"""

NB05_CONFIG_MD = """\
## 1. Choose Conservative Configuration

Grid-search 8 RF configs × 8 XGBoost configs, averaged over 100 randomly
sampled creators of varying sizes.  Pick the config with the lowest mean CV RMSE.
"""

NB05_CONFIG = """\
from creator_analytics.models import search_per_creator_config, _PC_RF_GRID, _PC_XGB_GRID

print(f"RF  grid: {len(_PC_RF_GRID)} configs")
print(f"XGB grid: {len(_PC_XGB_GRID)} configs")

rf_cfg, xgb_cfg = search_per_creator_config(
    X_train, y_train, author_train, sample_n=100, seed=SEED, n_jobs=N_JOBS
)
print("\\nBest RF config:")
for k, v in rf_cfg.items():
    print(f"  {k}: {v}")
print("\\nBest XGB config:")
for k, v in xgb_cfg.items():
    print(f"  {k}: {v}")
"""

NB05_TRAIN_MD = """\
## 2. Train Per-Creator Models

Train one RF and one XGBoost model per creator in parallel.
Features that are constant within a creator's train videos are dropped.
"""

NB05_TRAIN = """\
from creator_analytics.models import train_per_creator_models

rf_models, xgb_models = train_per_creator_models(
    X_train, y_train, author_train,
    rf_config=rf_cfg, xgb_config=xgb_cfg,
    n_jobs=N_JOBS, seed=SEED,
)
print(f"Trained {len(rf_models)} RF models, {len(xgb_models)} XGB models")

# Feature count statistics
n_features_rf  = [m.n_features for m in rf_models.values()]
n_features_xgb = [m.n_features for m in xgb_models.values()]
print(f"RF  features per creator: min={min(n_features_rf)}  median={np.median(n_features_rf):.0f}  max={max(n_features_rf)}")
print(f"XGB features per creator: min={min(n_features_xgb)}  median={np.median(n_features_xgb):.0f}  max={max(n_features_xgb)}")

n_worse_rf  = sum(m.worse_than_zero for m in rf_models.values())
n_worse_xgb = sum(m.worse_than_zero for m in xgb_models.values())
print(f"\\nWorse than predict_0 on train CV:")
print(f"  RF : {n_worse_rf} / {len(rf_models)}")
print(f"  XGB: {n_worse_xgb} / {len(xgb_models)}")
"""

NB05_EVAL_MD = """\
## 3. Test Evaluation

Test set is touched **once** — here.
"""

NB05_EVAL = """\
from creator_analytics.evaluate import (
    compute_metrics, compute_per_creator_metrics,
    predict_per_creator,
)
from creator_analytics.models import trivial_zero_rmse, trivial_creator_mean

rmse_zero = trivial_zero_rmse(y_test)
rmse_cmean, _ = trivial_creator_mean(y_train, author_train, y_test, author_test)

y_pred_rf  = predict_per_creator(X_test, author_test, rf_models)
y_pred_xgb = predict_per_creator(X_test, author_test, xgb_models)

mask_valid = ~np.isnan(y_pred_rf) & ~np.isnan(y_pred_xgb)
m_rf  = compute_metrics(y_test[mask_valid], y_pred_rf[mask_valid])
m_xgb = compute_metrics(y_test[mask_valid], y_pred_xgb[mask_valid])

print(f"predict_0          RMSE={rmse_zero:.4f}")
print(f"predict_creator_mean RMSE={rmse_cmean:.4f}")
print(f"RF  per-creator    RMSE={m_rf['rmse']:.4f}  R²={m_rf['r2']:.4f}  Spearman={m_rf['spearman']:.4f}")
print(f"XGB per-creator    RMSE={m_xgb['rmse']:.4f}  R²={m_xgb['r2']:.4f}  Spearman={m_xgb['spearman']:.4f}")

best_name = "RF" if m_rf["rmse"] <= m_xgb["rmse"] else "XGB"
y_pred_best = y_pred_rf if best_name == "RF" else y_pred_xgb
m_best = m_rf if best_name == "RF" else m_xgb
best_models = rf_models if best_name == "RF" else xgb_models
print(f"\\nWinner: {best_name}")
"""

NB05_COMPARE_MD = """\
## 4. Comparison: Global vs Per-Creator
"""

NB05_COMPARE = """\
global_preds_path = Path("results/predictions_global.parquet")
if global_preds_path.exists():
    global_preds = pd.read_parquet(global_preds_path)
    m_global = compute_metrics(global_preds["y_true"], global_preds["y_pred"])
else:
    m_global = {"rmse": float("nan"), "r2": float("nan"), "spearman": float("nan")}

summary = pd.DataFrame({
    "predict_0":            {"rmse": rmse_zero,     "r2": float("nan"), "spearman": float("nan")},
    "predict_creator_mean": {"rmse": rmse_cmean,    "r2": float("nan"), "spearman": float("nan")},
    "global_XGB":           m_global,
    f"per_creator_RF":       m_rf,
    f"per_creator_XGB":      m_xgb,
}).T[["rmse", "r2", "spearman"]]

display(summary.style.format({"rmse": "{:.4f}", "r2": "{:.4f}", "spearman": "{:.4f}"}, na_rep="—"))
"""

NB05_DIST_MD = """\
## 5. RMSE Distribution Across Creators
"""

NB05_DIST = """\
per_creator_pc = compute_per_creator_metrics(
    y_test[mask_valid], pd.Series(y_pred_best[mask_valid]), author_test[mask_valid]
)

# Also load global per-creator breakdown if available
pc_global_path = Path("results/per_creator_metrics_global.parquet")
if pc_global_path.exists():
    per_creator_global = pd.read_parquet(pc_global_path)
    merged = per_creator_pc.merge(per_creator_global[["author_id","rmse"]],
                                   on="author_id", suffixes=("_percreator", "_global"))
else:
    merged = per_creator_pc.rename(columns={"rmse": "rmse_percreator"})

print(f"Per-creator {best_name}: median RMSE={per_creator_pc['rmse'].median():.4f}  "
      f"mean={per_creator_pc['rmse'].mean():.4f}")

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(per_creator_pc["rmse"], bins=50, alpha=0.7, label=f"per-creator {best_name}", edgecolor="none")
if "rmse_global" in merged.columns:
    ax.hist(merged["rmse_global"], bins=50, alpha=0.5, label="global XGB", edgecolor="none")
ax.axvline(rmse_zero, color="orange", linestyle=":", linewidth=1.5, label="predict_0")
ax.axvline(per_creator_pc["rmse"].median(), color="blue", linestyle="--", linewidth=1, label="per-creator median")
ax.set_xlabel("Per-creator test RMSE")
ax.set_ylabel("Number of creators")
ax.set_title("RMSE distribution: per-creator models vs global")
ax.legend()
plt.tight_layout()
plt.show()

n_worse_test = (per_creator_pc["rmse"] > rmse_zero).sum()
print(f"Creators where per-creator model > predict_0 on test: {n_worse_test} / {len(per_creator_pc)}")
"""

NB05_SIZE_MD = """\
## 6. Quality vs Creator Size
"""

NB05_SIZE = """\
# Attach n_train and worse_than_zero flag to per-creator metrics
n_train_map = {m.author_id: m.n_train for m in best_models.values()}
worse_map   = {m.author_id: m.worse_than_zero for m in best_models.values()}
per_creator_pc["n_train"]       = per_creator_pc["author_id"].map(n_train_map)
per_creator_pc["worse_than_zero"] = per_creator_pc["author_id"].map(worse_map)

fig, ax = plt.subplots(figsize=(9, 5))
colors = per_creator_pc["worse_than_zero"].map({True: "red", False: "steelblue"})
ax.scatter(per_creator_pc["n_train"], per_creator_pc["rmse"],
           alpha=0.3, s=10, c=colors)
ax.axhline(rmse_zero, color="orange", linestyle=":", linewidth=1.5, label="predict_0")
ax.set_xlabel("Number of train videos")
ax.set_ylabel("Test RMSE")
ax.set_title(f"Test RMSE vs creator train size ({best_name} per-creator)")
ax.legend()
plt.tight_layout()
plt.show()

# Bin by creator size
bins = [0, 50, 100, 200, 500, 10000]
labels = ["24-50", "51-100", "101-200", "201-500", "500+"]
per_creator_pc["size_bin"] = pd.cut(per_creator_pc["n_train"], bins=bins, labels=labels)
size_summary = per_creator_pc.groupby("size_bin", observed=True)["rmse"].agg(["mean", "median", "count"])
display(size_summary.style.format({"mean": "{:.4f}", "median": "{:.4f}"}))
"""

NB05_SAVE_MD = """\
## 7. Save Artefacts
"""

NB05_SAVE = """\
import datetime
from creator_analytics.evaluate import log_experiment, save_metrics

today = datetime.date.today().isoformat()
exp_path = config["paths"].get("experiments_log", "results/experiments.md")
metrics_path = config["paths"].get("metrics", "results/metrics.json")

n_worse_rf_cv  = sum(m.worse_than_zero for m in rf_models.values())
n_worse_xgb_cv = sum(m.worse_than_zero for m in xgb_models.values())

log_experiment(exp_path, "per_creator_RF",  today, "non_constant_per_creator",
               "RF_per_creator",  rf_cfg,  float("nan"), m_rf,
               f"worse_than_zero_cv={n_worse_rf_cv}/{len(rf_models)}")
log_experiment(exp_path, "per_creator_XGB", today, "non_constant_per_creator",
               "XGB_per_creator", xgb_cfg, float("nan"), m_xgb,
               f"worse_than_zero_cv={n_worse_xgb_cv}/{len(xgb_models)}")

save_metrics({
    "per_creator_rf_test_rmse":           m_rf["rmse"],
    "per_creator_rf_test_r2":             m_rf["r2"],
    "per_creator_rf_worse_than_zero_cv":  n_worse_rf_cv,
    "per_creator_xgb_test_rmse":          m_xgb["rmse"],
    "per_creator_xgb_test_r2":            m_xgb["r2"],
    "per_creator_xgb_worse_than_zero_cv": n_worse_xgb_cv,
    "per_creator_winner":                 best_name,
}, metrics_path)

# Save predictions from winner
pred_path = Path("results/predictions_per_creator.parquet")
pred_path.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame({
    "y_true":    y_test.values,
    "y_pred":    y_pred_best,
    "author_id": author_test.values,
}).to_parquet(str(pred_path), index=False)

per_creator_pc.to_parquet("results/per_creator_metrics_percreator.parquet", index=False)

# Save models
for author_id, m in rf_models.items():
    m.save(f"models/per_creator/RF/{author_id}.joblib")
for author_id, m in xgb_models.items():
    m.save(f"models/per_creator/XGB/{author_id}.joblib")

print(f"Saved {len(rf_models)} RF + {len(xgb_models)} XGB creator models")
print(f"Saved: results/predictions_per_creator.parquet")
print(f"Saved: results/experiments.md")
"""

NB05_SUMMARY_MD = """\
## Summary

### Chosen configuration

**Why conservative?**  Creators have 24–1 700 train videos; on small sets a
deep unconstrained model overfits immediately.  The config is chosen by
averaging CV RMSE over 100 diverse creators — short trees, strong leaf
regularisation.

### Global metrics summary

| Model | Test RMSE | Test R² | Test Spearman |
|-------|-----------|---------|---------------|
"""

NB05_SUMMARY = """\
print("=== Chosen configs ===")
print("RF :", {k: v for k, v in rf_cfg.items() if k != "n_estimators"})
print("XGB:", {k: v for k, v in xgb_cfg.items() if k not in ("n_estimators", "subsample", "colsample_bytree")})
print()

final_table = pd.DataFrame({
    "predict_0":            {"test_rmse": rmse_zero,   "test_r2": float("nan"), "test_spearman": float("nan")},
    "predict_creator_mean": {"test_rmse": rmse_cmean,  "test_r2": float("nan"), "test_spearman": float("nan")},
    "global_XGB":           {"test_rmse": m_global.get("rmse", float("nan")),
                              "test_r2":   m_global.get("r2",   float("nan")),
                              "test_spearman": m_global.get("spearman", float("nan"))},
    "per_creator_RF":       {"test_rmse": m_rf["rmse"],  "test_r2": m_rf["r2"],  "test_spearman": m_rf["spearman"]},
    "per_creator_XGB":      {"test_rmse": m_xgb["rmse"], "test_r2": m_xgb["r2"], "test_spearman": m_xgb["spearman"]},
}).T
display(final_table.style.format("{:.4f}", na_rep="—"))

n_worse_test = (per_creator_pc["rmse"] > rmse_zero).sum()
print(f"\\nWinner: {best_name}")
print(f"Creators with test RMSE > predict_0: {n_worse_test} / {len(per_creator_pc)}")
print(f"Creators with CV RMSE > predict_0: RF={n_worse_rf_cv} / {len(rf_models)},  XGB={n_worse_xgb_cv} / {len(xgb_models)}")
"""

nb05 = make_nb([
    new_markdown_cell(NB05_GOAL),
    new_code_cell(COLAB_GUARD),
    new_code_cell(SETUP_COMMON),
    new_markdown_cell("## 0. Load Data"),
    new_code_cell(NB05_LOAD),
    new_markdown_cell(NB05_CONFIG_MD),
    new_code_cell(NB05_CONFIG),
    new_markdown_cell(NB05_TRAIN_MD),
    new_code_cell(NB05_TRAIN),
    new_markdown_cell(NB05_EVAL_MD),
    new_code_cell(NB05_EVAL),
    new_markdown_cell(NB05_COMPARE_MD),
    new_code_cell(NB05_COMPARE),
    new_markdown_cell(NB05_DIST_MD),
    new_code_cell(NB05_DIST),
    new_markdown_cell(NB05_SIZE_MD),
    new_code_cell(NB05_SIZE),
    new_markdown_cell(NB05_SAVE_MD),
    new_code_cell(NB05_SAVE),
    new_markdown_cell(NB05_SUMMARY_MD),
    new_code_cell(NB05_SUMMARY),
])

nbformat.write(nb05, NB_DIR / "05-per-creator-models.ipynb")
print("Written notebooks/05-per-creator-models.ipynb")


# ================================================================== #
# NOTEBOOK 06: Model Comparison — Per-Creator vs Global               #
# ================================================================== #

NB06_GOAL = """\
# 06 — Model Comparison: Per-Creator vs Global

**Goal** – Compare the per-creator and global models on the same held-out test
videos.  All analysis uses *saved prediction parquets only* — no retraining.

Analyses:
1. Overall metrics on all test videos.
2. Paired per-creator comparison: Wilcoxon signed-rank test + bootstrap 95% CI.
3. Moderation: does personalisation help more for large creators? High-variance targets?

> **Note on noise**: median creator has only 19 test videos; per-creator RMSE
> estimates are therefore noisy.  We report CIs throughout and avoid strong
> claims about individual creators.
"""

NB06_LOAD = """\
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── sanity check ────────────────────────────────────────────────────
PRED_GLOBAL = Path("results/predictions_global.parquet")
PRED_PC     = Path("results/predictions_per_creator.parquet")
PC_META     = Path("results/per_creator_metrics_percreator.parquet")

if not PRED_GLOBAL.exists():
    raise FileNotFoundError(
        f"{PRED_GLOBAL} not found. "
        "Run notebooks/04-baseline-model.ipynb first."
    )

pred_g  = pd.read_parquet(PRED_GLOBAL)   # y_true, y_pred, author_id
pred_pc = pd.read_parquet(PRED_PC)       # y_true, y_pred, author_id
meta_pc = pd.read_parquet(PC_META)       # author_id, n_test, rmse, n_train, …

print(f"Global predictions :  {len(pred_g):,} rows,  "
      f"NaN y_pred={pred_g.y_pred.isna().sum()}")
print(f"Per-creator preds  :  {len(pred_pc):,} rows,  "
      f"NaN y_pred={pred_pc.y_pred.isna().sum()}")

# align on common creators + verify y_true matches
common = set(pred_g.author_id.unique()) & set(pred_pc.author_id.unique())
print(f"Common creators    :  {len(common)}")
pred_g  = pred_g[pred_g.author_id.isin(common)].reset_index(drop=True)
pred_pc = pred_pc[pred_pc.author_id.isin(common)].reset_index(drop=True)

# merge to compare y_true column (must be identical)
merged = pred_g.rename(columns={"y_pred":"y_pred_g"}).merge(
    pred_pc[["author_id", "y_pred"]].rename(columns={"y_pred":"y_pred_pc"}),
    left_index=True, right_index=True
)
# verify y_true alignment by recomputing from both parquets
yt_g  = pred_g.sort_values(["author_id","y_true"])["y_true"].values
yt_pc = pred_pc.sort_values(["author_id","y_true"])["y_true"].values
assert np.allclose(yt_g, yt_pc, atol=1e-9), "y_true mismatch between prediction files!"
print("y_true alignment   :  OK")
"""

NB06_HELPERS = """\
# ── helper functions ────────────────────────────────────────────────

def _rmse(yt, yp):
    return float(np.sqrt(mean_squared_error(yt, yp)))

def compute_global_metrics(df):
    yt, yp = df.y_true.values, df.y_pred.values
    rho, _ = spearmanr(yt, yp)
    return {"rmse": _rmse(yt, yp),
            "mae":  float(mean_absolute_error(yt, yp)),
            "r2":   float(r2_score(yt, yp)),
            "spearman": float(rho),
            "n":    len(yt)}

def compute_per_creator_metrics(df):
    rows = []
    for author, g in df.groupby("author_id"):
        yt, yp = g.y_true.values, g.y_pred.values
        n = len(yt)
        rho = float(spearmanr(yt, yp).statistic) if n >= 2 else float("nan")
        rows.append({"author_id": author, "n_test": n,
                     "rmse": _rmse(yt, yp),
                     "spearman": rho})
    return pd.DataFrame(rows)

def bootstrap_ci(values, stat_fn=np.mean, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    boots = [stat_fn(rng.choice(values, size=len(values), replace=True))
             for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)

print("Helper functions defined.")
"""

NB06_TRIVIAL = """\
# ── trivial baselines (computed from y_true only) ───────────────────

# predict_0: always predict 0
# predict_creator_mean: per-creator mean of y_true in the *same test set*
# (conservative — overestimates creator-mean baseline skill since we use test mean)
# For a fair estimate, we would need train means; for comparison we note this caveat.

y_true_all = pred_g["y_true"].values

# predict_0
pred_zero = np.zeros(len(y_true_all))

# predict_creator_test_mean (uses test mean — slight overestimate of a real baseline)
test_means = pred_g.groupby("author_id")["y_true"].transform("mean")
pred_creator_mean_test = test_means.values

m_zero = {"name": "predict_0",
          "rmse": _rmse(y_true_all, pred_zero),
          "mae":  float(mean_absolute_error(y_true_all, pred_zero)),
          "r2":   float(r2_score(y_true_all, pred_zero)),
          "spearman": float("nan")}

m_cmean = {"name": "predict_creator_mean (test)",
           "rmse": _rmse(y_true_all, pred_creator_mean_test),
           "mae":  float(mean_absolute_error(y_true_all, pred_creator_mean_test)),
           "r2":   float(r2_score(y_true_all, pred_creator_mean_test)),
           "spearman": float("nan")}

print(f"predict_0            RMSE={m_zero['rmse']:.4f}  MAE={m_zero['mae']:.4f}  R²={m_zero['r2']:.4f}")
print(f"predict_creator_mean RMSE={m_cmean['rmse']:.4f}  MAE={m_cmean['mae']:.4f}  R²={m_cmean['r2']:.4f}")
print("(creator-mean baseline uses test mean — slightly optimistic)")
"""

NB06_OVERALL_MD = """\
## 1. Overall Metrics
"""

NB06_OVERALL = """\
m_global = compute_global_metrics(pred_g.rename(columns={"y_pred":"y_pred"}).assign(y_pred=pred_g.y_pred))
m_pc     = compute_global_metrics(pred_pc.rename(columns={"y_pred":"y_pred"}).assign(y_pred=pred_pc.y_pred))

overall = pd.DataFrame([
    {"model": "predict_0",                 **{k: m_zero[k] for k in ["rmse","mae","r2","spearman"]}},
    {"model": "predict_creator_mean(test)",**{k: m_cmean[k] for k in ["rmse","mae","r2","spearman"]}},
    {"model": "global XGB",               **{k: m_global[k] for k in ["rmse","mae","r2","spearman"]}},
    {"model": "per-creator RF",           **{k: m_pc[k] for k in ["rmse","mae","r2","spearman"]}},
]).set_index("model")

display(overall.style
        .format({"rmse":"{:.4f}","mae":"{:.4f}","r2":"{:.4f}","spearman":"{:.4f}"}, na_rep="—")
        .highlight_min(subset=["rmse","mae"], color="#d4f1c2")
        .highlight_max(subset=["r2","spearman"], color="#d4f1c2"))

rmse_improvement = (m_global["rmse"] - m_pc["rmse"]) / m_global["rmse"] * 100
print(f"\\nPer-creator vs global: RMSE improvement {rmse_improvement:+.1f}%")
"""

NB06_PAIRED_MD = """\
## 2. Paired Per-Creator Comparison

We compute per-creator RMSE and Spearman for **each model** on each creator's
test videos, then take the **difference** (global − per-creator).  A positive
difference means the per-creator model is better for that creator.

Each creator appears once in each model → naturally paired observations.
"""

NB06_PAIRED = """\
pc_g  = compute_per_creator_metrics(pred_g)
pc_pc = compute_per_creator_metrics(pred_pc)

cmp = pc_g.merge(pc_pc, on="author_id", suffixes=("_g","_pc"))
cmp = cmp.merge(meta_pc[["author_id","n_train","size_bin","worse_than_zero"]], on="author_id", how="left")
cmp["target_std"] = pred_g.groupby("author_id")["y_true"].std().reindex(cmp["author_id"]).values

# positive  = per-creator wins
cmp["rmse_diff"]     = cmp["rmse_g"]     - cmp["rmse_pc"]
cmp["spearman_diff"] = cmp["spearman_pc"] - cmp["spearman_g"]

n_total = len(cmp)
n_pc_wins_rmse     = (cmp["rmse_diff"] > 0).sum()
n_pc_wins_spearman = (cmp["spearman_diff"] > 0).sum()

print(f"Total creators in comparison: {n_total}")
print(f"Per-creator wins on RMSE    : {n_pc_wins_rmse} / {n_total}  ({100*n_pc_wins_rmse/n_total:.1f}%)")
print(f"Per-creator wins on Spearman: {n_pc_wins_spearman} / {n_total}  ({100*n_pc_wins_spearman/n_total:.1f}%)")
print(f"\\nMean RMSE diff (global - pc): {cmp.rmse_diff.mean():.4f}")
print(f"Median RMSE diff            : {cmp.rmse_diff.median():.4f}")
ci_lo, ci_hi = bootstrap_ci(cmp["rmse_diff"].values)
print(f"Bootstrap 95% CI (2 000 resamples, creator-level): [{ci_lo:.4f}, {ci_hi:.4f}]")
"""

NB06_WILCOXON_MD = """\
## 3. Wilcoxon Signed-Rank Test

**Why a paired test?**  Each creator contributes *one RMSE estimate per model*
(not an independent sample per model).  The paired design removes between-creator
variance and focuses on the within-creator improvement.

**Why Wilcoxon and not a paired t-test?**  RMSE differences across creators are
unlikely to follow a normal distribution — the distribution is bounded below,
right-skewed, and contains a few creators with very high RMSE.  The Wilcoxon
signed-rank test makes no normality assumption; it tests whether the distribution
of signed differences is symmetric around zero.

$H_0$: the median RMSE difference (global − per-creator) is zero.
$H_1$: the median is positive (per-creator model is better).
"""

NB06_WILCOXON = """\
diffs = cmp["rmse_diff"].dropna().values

stat, pval_two = wilcoxon(diffs)                              # two-sided
stat_g, pval_one = wilcoxon(diffs, alternative="greater")    # one-sided: pc better

print(f"Wilcoxon signed-rank test on RMSE differences (n={len(diffs)} creator pairs)")
print(f"  Two-sided  : W={stat:.1f}   p={pval_two:.4f}")
print(f"  One-sided  : W={stat_g:.1f}  p={pval_one:.4f}  (H1: per-creator RMSE < global)")
print()
if pval_two < 0.05:
    print("✓ Significant at α=0.05 — the per-creator model differs from global in a "
          "consistent direction.")
else:
    print("✗ Not significant at α=0.05 — no consistent per-creator advantage detected "
          "across the full creator population.")

print()
print("Bootstrap 95% CI for mean RMSE diff (global − per-creator):")
ci_lo, ci_hi = bootstrap_ci(diffs)
print(f"  [{ci_lo:.4f}, {ci_hi:.4f}]")
if ci_lo > 0:
    print("  → CI entirely above 0: per-creator model reliably reduces RMSE on average.")
elif ci_hi < 0:
    print("  → CI entirely below 0: global model reliably better on average.")
else:
    print("  → CI straddles 0: no reliable direction of average difference.")
"""

NB06_DIST_FIG = """\
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

ax = axes[0]
ax.hist(cmp["rmse_diff"], bins=60, edgecolor="none", color="steelblue", alpha=0.8)
ax.axvline(0, color="black", linewidth=0.8, linestyle="--", label="no difference")
ax.axvline(cmp["rmse_diff"].mean(), color="red", linewidth=1.2, linestyle="-",
           label=f"mean={cmp['rmse_diff'].mean():.3f}")
ax.axvline(cmp["rmse_diff"].median(), color="orange", linewidth=1.2, linestyle="-.",
           label=f"median={cmp['rmse_diff'].median():.3f}")
ax.set_xlabel("RMSE_global − RMSE_per_creator")
ax.set_ylabel("Number of creators")
ax.set_title("Distribution of per-creator RMSE improvement")
ax.legend(fontsize=8)

ax = axes[1]
ax.hist(cmp["spearman_diff"].dropna(), bins=60, edgecolor="none", color="darkorange", alpha=0.8)
ax.axvline(0, color="black", linewidth=0.8, linestyle="--", label="no difference")
ax.axvline(cmp["spearman_diff"].mean(), color="red", linewidth=1.2,
           label=f"mean={cmp['spearman_diff'].mean():.3f}")
ax.set_xlabel("Spearman_per_creator − Spearman_global")
ax.set_ylabel("Number of creators")
ax.set_title("Distribution of Spearman improvement")
ax.legend(fontsize=8)

plt.tight_layout()
plt.show()

print("Positive tail = per-creator model better for that creator.")
"""

NB06_MOD_MD = """\
## 4. Moderation Analysis

Our primary hypothesis: *personalisation gains more for creators with a larger
training history*, because the per-creator model can learn creator-specific
patterns only when enough data exist.

We also check whether creators with **more variable content** (higher target std)
benefit differently.

> Caution: creators with 6–10 test videos have very noisy RMSE estimates.
> We show group-level bootstrap CIs and avoid per-creator conclusions.
"""

NB06_MOD_SIZE = """\
# ── by creator size (n_train bins) ──────────────────────────────────

bins   = [0, 50, 100, 300, 10000]
labels = ["24-50", "51-100", "101-300", "300+"]
cmp["size_group"] = pd.cut(cmp["n_train"], bins=bins, labels=labels)

group_stats = []
for grp, sub in cmp.groupby("size_group", observed=True):
    d = sub["rmse_diff"].dropna().values
    ci_lo, ci_hi = bootstrap_ci(d) if len(d) >= 10 else (float("nan"), float("nan"))
    group_stats.append({
        "size_group": grp,
        "n_creators": len(d),
        "mean_rmse_diff": np.mean(d),
        "median_rmse_diff": np.median(d),
        "ci_lo": ci_lo, "ci_hi": ci_hi,
        "pct_pc_wins": 100 * (d > 0).mean(),
    })

size_df = pd.DataFrame(group_stats).set_index("size_group")
display(size_df.style.format({
    "mean_rmse_diff":   "{:.4f}",
    "median_rmse_diff": "{:.4f}",
    "ci_lo":            "{:.4f}",
    "ci_hi":            "{:.4f}",
    "pct_pc_wins":      "{:.1f}%",
}))
"""

NB06_MOD_SIZE_FIG = """\
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Left: mean RMSE diff ± 95% CI by size group
ax = axes[0]
grps = size_df.index.tolist()
x    = np.arange(len(grps))
means = size_df["mean_rmse_diff"].values
lo    = means - size_df["ci_lo"].values
hi    = size_df["ci_hi"].values - means
ax.bar(x, means, color="steelblue", alpha=0.7)
ax.errorbar(x, means, yerr=[lo, hi], fmt="none", color="black", capsize=4, linewidth=1.5)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xticks(x); ax.set_xticklabels(grps)
ax.set_xlabel("Creator train size (n_train videos)")
ax.set_ylabel("Mean RMSE diff (global − per-creator)\\n(positive = per-creator wins)")
ax.set_title("Per-creator improvement by creator size\\n(bars = bootstrap 95% CI)")

# Right: violin of RMSE diff by group
ax = axes[1]
data_by_group = [cmp[cmp["size_group"] == g]["rmse_diff"].dropna().values for g in grps]
vp = ax.violinplot(data_by_group, positions=x, showmedians=True, showextrema=False)
for body in vp["bodies"]:
    body.set_alpha(0.6); body.set_facecolor("steelblue")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xticks(x); ax.set_xticklabels(grps)
ax.set_xlabel("Creator train size")
ax.set_ylabel("RMSE diff (global − per-creator)")
ax.set_title("Distribution of RMSE improvement by creator size")

plt.tight_layout()
plt.show()
"""

NB06_MOD_VAR_MD = """\
### Moderation by Target Variance

Creators whose video performance is more heterogeneous (higher std of relative
target) might benefit more from personalisation — a per-creator model can
capture that creator's unique spread.
"""

NB06_MOD_VAR = """\
# scatter: RMSE diff vs target_std (colour by size group)
fig, ax = plt.subplots(figsize=(8, 5))
colors = {"24-50": "#aec6e8", "51-100": "#4c96d7", "101-300": "#1a5fa3", "300+": "#0a2f6b"}
for grp, sub in cmp.groupby("size_group", observed=True):
    ax.scatter(sub["target_std"], sub["rmse_diff"],
               alpha=0.3, s=12, label=grp, color=colors.get(str(grp), "grey"))
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("Target std (per creator test videos)")
ax.set_ylabel("RMSE diff (global − per-creator)")
ax.set_title("Per-creator improvement vs target variability")
ax.legend(title="Train size", fontsize=8)
plt.tight_layout()
plt.show()

# correlation: does higher target_std predict more gain?
valid = cmp[["rmse_diff","target_std","n_train"]].dropna()
rho_var,  pval_var  = spearmanr(valid["target_std"],  valid["rmse_diff"])
rho_size, pval_size = spearmanr(valid["n_train"],     valid["rmse_diff"])
print(f"Spearman(target_std,  rmse_diff) = {rho_var:.3f}  p={pval_var:.4f}")
print(f"Spearman(n_train,     rmse_diff) = {rho_size:.3f}  p={pval_size:.4f}")
"""

NB06_NOISE_MD = """\
## 5. Noise Caveat: Small-Creator Metrics

The median creator has only 19 test videos. With so few observations,
per-creator RMSE is a noisy estimator.  Below we show how RMSE uncertainty
scales with test-set size (using prediction-level standard deviation as a proxy).
"""

NB06_NOISE = """\
# CI width for per-creator RMSE as a function of n_test
# Proxy: for each creator, compute std of (y_true - y_pred)^2 residuals → std / sqrt(n)
def rmse_se_approx(df_creator):
    resid2 = (df_creator.y_true - df_creator.y_pred) ** 2
    return resid2.std() / (2 * _rmse(df_creator.y_true, df_creator.y_pred) * np.sqrt(len(resid2)))

se_data = []
for a, g in pred_pc.groupby("author_id"):
    if len(g) >= 4:
        se_data.append({"author_id": a, "n_test": len(g), "rmse_se": rmse_se_approx(g)})
se_df = pd.DataFrame(se_data)

fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(se_df["n_test"], se_df["rmse_se"], alpha=0.3, s=8, color="grey")
ax.axvline(10, color="orange", linestyle=":", linewidth=1, label="n=10")
ax.axvline(30, color="steelblue", linestyle=":", linewidth=1, label="n=30")
ax.set_xscale("log")
ax.set_xlabel("Number of test videos (log scale)")
ax.set_ylabel("Approx. RMSE standard error")
ax.set_title("RMSE uncertainty decreases with creator test-set size")
ax.legend()
plt.tight_layout()
plt.show()

pct_small = (se_df.n_test < 10).mean() * 100
print(f"{pct_small:.0f}% of creators have < 10 test videos → their individual RMSE is unreliable.")
print("Group-level statistics (bootstrap CIs above) are valid because they average over many creators.")
"""

NB06_SAVE_MD = """\
## 6. Save Results
"""

NB06_SAVE = """\
import datetime
from creator_analytics.evaluate import log_experiment, save_metrics
from creator_analytics.config import load_config

config = load_config("config/default.yaml")
today  = datetime.date.today().isoformat()
exp_path  = config["paths"].get("experiments_log", "results/experiments.md")
metrics_path = config["paths"].get("metrics", "results/metrics.json")

log_experiment(
    exp_path, "comparison_global_vs_percreator", today,
    "all_test_videos", "comparison", {},
    cv_rmse=None,
    test_metrics={"rmse": m_pc["rmse"], "r2": m_pc["r2"], "spearman": m_pc["spearman"]},
    comment=(f"global_rmse={m_global['rmse']:.4f}  "
             f"wilcoxon_p={pval_two:.4f}  "
             f"CI=[{ci_lo:.4f},{ci_hi:.4f}]  "
             f"pc_wins={n_pc_wins_rmse}/{n_total}")
)

save_metrics({
    "comparison_global_rmse": m_global["rmse"],
    "comparison_pc_rmse":     m_pc["rmse"],
    "comparison_rmse_diff_mean":   float(cmp.rmse_diff.mean()),
    "comparison_rmse_diff_ci_lo":  ci_lo,
    "comparison_rmse_diff_ci_hi":  ci_hi,
    "comparison_wilcoxon_p":       pval_two,
    "comparison_pct_pc_wins_rmse": float(n_pc_wins_rmse / n_total * 100),
}, metrics_path)

print("Saved to experiments.md and metrics.json")
"""

NB06_SUMMARY_MD = """\
## Summary

### What we found
"""

NB06_SUMMARY = """\
# ── print cautious conclusions ──────────────────────────────────────────────

print("=" * 65)
print("MODEL COMPARISON SUMMARY")
print("=" * 65)
print(f"  Test videos: {len(pred_g):,}  |  Creators: {n_total}")
print()
print("Overall RMSE:")
print(f"  predict_0            : {m_zero['rmse']:.4f}")
print(f"  global XGB           : {m_global['rmse']:.4f}")
print(f"  per-creator RF       : {m_pc['rmse']:.4f}")
print(f"  per-creator vs global: {m_global['rmse'] - m_pc['rmse']:+.4f}  "
      f"({(m_global['rmse'] - m_pc['rmse'])/m_global['rmse']*100:+.1f}%)")
print()
print("Paired per-creator test (Wilcoxon signed-rank):")
print(f"  p-value (two-sided)  : {pval_two:.4f}")
print(f"  Bootstrap 95% CI     : [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"  Creators where per-creator wins (RMSE): "
      f"{n_pc_wins_rmse}/{n_total} ({100*n_pc_wins_rmse/n_total:.0f}%)")
print()
print("Moderation (Spearman correlations with RMSE gain):")
print(f"  n_train   vs gain    : ρ={rho_size:.3f}  p={pval_size:.4f}")
print(f"  target_std vs gain   : ρ={rho_var:.3f}   p={pval_var:.4f}")
print()
print("Conclusion (cautious):")

conclusions = []

if ci_lo > 0:
    conclusions.append(
        "The 95% bootstrap CI lies entirely above 0: on average, the per-creator "
        "RF model reduces RMSE compared with the global XGB model."
    )
elif ci_hi < 0:
    conclusions.append(
        "The 95% bootstrap CI lies entirely below 0: on average, the global model "
        "outperforms the per-creator model."
    )
else:
    conclusions.append(
        "The 95% bootstrap CI straddles 0: the direction of average improvement "
        "is uncertain."
    )

if pval_two < 0.05:
    conclusions.append(
        f"The Wilcoxon test is significant (p={pval_two:.4f}): the per-creator and "
        "global models differ systematically across creators."
    )
else:
    conclusions.append(
        f"The Wilcoxon test is not significant (p={pval_two:.4f}): no systematic "
        "directional difference detected at α=0.05."
    )

if abs(rho_size) > 0.1 and pval_size < 0.05:
    direction = "more" if rho_size > 0 else "less"
    conclusions.append(
        f"Larger creators benefit {direction} from personalisation "
        f"(Spearman ρ={rho_size:.3f}, p={pval_size:.4f})."
    )
else:
    conclusions.append(
        f"Creator size is not a strong predictor of personalisation gain "
        f"(Spearman ρ={rho_size:.3f}, p={pval_size:.4f})."
    )

for i, c in enumerate(conclusions, 1):
    print(f"  {i}. {c}")
"""

nb06 = make_nb([
    new_markdown_cell(NB06_GOAL),
    new_code_cell(COLAB_GUARD),
    new_code_cell(SETUP_COMMON),
    new_markdown_cell("## 0. Load Predictions"),
    new_code_cell(NB06_LOAD),
    new_code_cell(NB06_HELPERS),
    new_code_cell(NB06_TRIVIAL),
    new_markdown_cell(NB06_OVERALL_MD),
    new_code_cell(NB06_OVERALL),
    new_markdown_cell(NB06_PAIRED_MD),
    new_code_cell(NB06_PAIRED),
    new_markdown_cell(NB06_WILCOXON_MD),
    new_code_cell(NB06_WILCOXON),
    new_code_cell(NB06_DIST_FIG),
    new_markdown_cell(NB06_MOD_MD),
    new_code_cell(NB06_MOD_SIZE),
    new_code_cell(NB06_MOD_SIZE_FIG),
    new_markdown_cell(NB06_MOD_VAR_MD),
    new_code_cell(NB06_MOD_VAR),
    new_markdown_cell(NB06_NOISE_MD),
    new_code_cell(NB06_NOISE),
    new_markdown_cell(NB06_SAVE_MD),
    new_code_cell(NB06_SAVE),
    new_markdown_cell(NB06_SUMMARY_MD),
    new_code_cell(NB06_SUMMARY),
])

nbformat.write(nb06, NB_DIR / "06-model-comparison.ipynb")
print("Written notebooks/06-model-comparison.ipynb")
