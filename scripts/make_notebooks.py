#!/usr/bin/env python
"""Generate notebooks/01-data-exploration.ipynb and 02-target-selection.ipynb.

Run once to (re)create the notebooks from scratch:
    python scripts/make_notebooks.py

Then execute them:
    jupyter nbconvert --to notebook --execute --inplace notebooks/01-data-exploration.ipynb
    jupyter nbconvert --to notebook --execute --inplace notebooks/02-target-selection.ipynb
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
