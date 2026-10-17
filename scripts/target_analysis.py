#!/usr/bin/env python
"""Target-variable analysis for the per-creator modelling pipeline.

Answers five questions to support choosing between absolute and relative
log1p(play_count) targets at different measurement days.

Run: python scripts/target_analysis.py
Figures are saved to results/figures/.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=FutureWarning)

RAW = Path("data/raw")
FIGS = Path("results/figures")
FIGS.mkdir(parents=True, exist_ok=True)

SEP = "=" * 70

# ------------------------------------------------------------------ #
# Load data                                                           #
# ------------------------------------------------------------------ #
print(SEP)
print("Loading tables …")
engagement = pd.read_parquet(RAW / "engagement_daily.parquet")
videos = pd.read_parquet(RAW / "videos.parquet")
creator_daily = pd.read_parquet(RAW / "creator_daily.parquet")
print(f"  engagement_daily : {len(engagement):,} rows")
print(f"  videos           : {len(videos):,} rows")
print(f"  creator_daily    : {len(creator_daily):,} rows")

# ================================================================== #
# PART 1 — Coverage: exact vs >= at each day                          #
# ================================================================== #
print(f"\n{SEP}")
print("PART 1 — Coverage by days_since_post")
print(SEP)

total_videos = videos["video_id"].nunique()
print(f"  Total unique videos: {total_videos:,}")

coverage_rows = []
for day in [0, 7, 14, 21, 28, 29, 30]:
    exact = engagement[engagement["days_since_post"] == day]["video_id"].nunique()
    geq   = engagement[engagement["days_since_post"] >= day]["video_id"].nunique()
    coverage_rows.append({
        "day": day,
        "exact_n": exact, "exact_pct": exact / total_videos * 100,
        "geq_n":   geq,   "geq_pct":   geq / total_videos * 100,
    })
    print(f"  day == {day:2d}: {exact:,} videos ({exact/total_videos*100:.1f}%)  |"
          f"  day >= {day:2d}: {geq:,} ({geq/total_videos*100:.1f}%)")

# Tracking window per video
max_day_per_video = engagement.groupby("video_id")["days_since_post"].max()
n_days_per_video  = engagement.groupby("video_id")["days_since_post"].nunique()
print(f"\n  max(days_since_post) per video:")
for v in [0, 7, 14, 21, 28, 30]:
    n = (max_day_per_video == v).sum()
    print(f"    max == {v:2d}: {n:,} videos ({n/total_videos*100:.1f}%)")
print(f"  median distinct days per video: {n_days_per_video.median():.0f}")
print(f"  p5 / p95 distinct days per video: {n_days_per_video.quantile(0.05):.0f} / "
      f"{n_days_per_video.quantile(0.95):.0f}")

# Videos without an exact day-14 record: what is the nearest available day?
has_exact_14 = set(engagement[engagement["days_since_post"] == 14]["video_id"])
missing_14 = videos[~videos["video_id"].isin(has_exact_14)]["video_id"]
if len(missing_14):
    nearest = (
        engagement[engagement["video_id"].isin(missing_14)]
        .groupby("video_id")["days_since_post"]
        .apply(lambda s: s.iloc[(s - 14).abs().argsort()].iloc[0])
    )
    print(f"\n  Videos with NO exact day-14 record: {len(missing_14):,}")
    print(f"  Nearest available day (for those): {nearest.value_counts().head(5).to_dict()}")
else:
    print("\n  All videos have an exact day-14 record. ✓")

# -- Figure 1 --
cov_df = pd.DataFrame(coverage_rows)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
days_plot = [7, 14, 21, 28, 30]
cov_plot = cov_df[cov_df["day"].isin(days_plot)]
axes[0].bar([str(d) for d in days_plot], cov_plot["exact_pct"], color="steelblue")
axes[0].set_title("Exact coverage (days_since_post == day)")
axes[0].set_xlabel("days_since_post"); axes[0].set_ylabel("% of videos")
axes[0].set_ylim(0, 105)
for i, row in cov_plot.reset_index(drop=True).iterrows():
    axes[0].text(i, row["exact_pct"] + 1, f"{row['exact_pct']:.1f}%", ha="center", fontsize=8)

axes[1].bar([str(d) for d in days_plot], cov_plot["geq_pct"], color="darkorange")
axes[1].set_title("Cumulative coverage (days_since_post >= day)")
axes[1].set_xlabel("days_since_post"); axes[1].set_ylabel("% of videos")
axes[1].set_ylim(0, 105)
for i, row in cov_plot.reset_index(drop=True).iterrows():
    axes[1].text(i, row["geq_pct"] + 1, f"{row['geq_pct']:.1f}%", ha="center", fontsize=8)

fig.suptitle("Engagement record coverage by measurement day", fontweight="bold")
fig.tight_layout()
fig.savefig(FIGS / "fig1_coverage_by_day.png", dpi=120)
plt.close(fig)
print(f"\n  → Saved fig1_coverage_by_day.png")

# ================================================================== #
# PART 2 — Monotonicity of play_count                                 #
# ================================================================== #
print(f"\n{SEP}")
print("PART 2 — Monotonicity check (play_count should be non-decreasing)")
print(SEP)

eng_sorted = engagement.sort_values(["video_id", "days_since_post"])
eng_sorted = eng_sorted.copy()
eng_sorted["play_diff"] = eng_sorted.groupby("video_id")["play_count"].diff()

violations = eng_sorted[eng_sorted["play_diff"] < 0]
n_violating_videos = violations["video_id"].nunique()
n_violation_rows   = len(violations)

print(f"  Videos with at least one decrease: {n_violating_videos:,}  "
      f"({n_violating_videos/total_videos*100:.2f}%)")
print(f"  Total rows with a negative diff  : {n_violation_rows:,}")

if n_violating_videos > 0:
    max_drop_per_video = violations.groupby("video_id")["play_diff"].min()  # most negative
    severity_pct = (max_drop_per_video.abs() /
                    engagement.groupby("video_id")["play_count"].max() * 100)
    print(f"  Max drop as % of video peak play_count:")
    print(f"    median: {severity_pct.median():.2f}%   p95: {severity_pct.quantile(0.95):.2f}%"
          f"   max: {severity_pct.max():.2f}%")
    print("  → Small violations are likely data collection artefacts, not real drops.")
else:
    print("  No violations found — play_count is strictly non-decreasing. ✓")

# ================================================================== #
# PART 3 — Day 14 (and 7) as fraction of day 30                       #
# ================================================================== #
print(f"\n{SEP}")
print("PART 3 — How much of day-30 play_count is already accumulated by day 7 and day 14")
print(SEP)

# Extract exact-day snapshots
def exact_day_plays(day: int) -> pd.Series:
    """Return a Series video_id → play_count at exactly `day`."""
    return (
        engagement[engagement["days_since_post"] == day]
        .set_index("video_id")["play_count"]
    )

pc7  = exact_day_plays(7)
pc14 = exact_day_plays(14)
pc30 = exact_day_plays(30)

# Videos with all three snapshots
common = pc14.index.intersection(pc30.index).intersection(pc7.index)
print(f"  Videos with exact records at days 7, 14 AND 30: {len(common):,}  "
      f"({len(common)/total_videos*100:.1f}% of all videos)")

ratio14 = pc14.loc[common] / pc30.loc[common].replace(0, np.nan)
ratio7  = pc7.loc[common]  / pc30.loc[common].replace(0, np.nan)

for label, ratio in [("day 7 / day 30", ratio7), ("day 14 / day 30", ratio14)]:
    q = ratio.quantile([0.05, 0.25, 0.50, 0.75, 0.95]).round(3)
    pct_above_09 = (ratio >= 0.9).mean() * 100
    print(f"\n  {label}  (n={ratio.notna().sum():,})")
    print(f"    p5={q[0.05]:.3f}  p25={q[0.25]:.3f}  median={q[0.50]:.3f}"
          f"  p75={q[0.75]:.3f}  p95={q[0.95]:.3f}")
    print(f"    >= 0.90 (90% of day-30 already in): {pct_above_09:.1f}% of videos")

# Bias check: are day-30 videos systematically different?
print(f"\n  --- Bias check: day-30 coverage vs not ---")
video30_ids  = set(pc30.index)
video_no30   = videos[~videos["video_id"].isin(video30_ids)].copy()
video_yes30  = videos[ videos["video_id"].isin(video30_ids)].copy()

# Publish date
video_no30["create_date_dt"]  = pd.to_datetime(video_no30["create_date"], errors="coerce")
video_yes30["create_date_dt"] = pd.to_datetime(video_yes30["create_date"], errors="coerce")
print(f"  Has day-30  ({len(video_yes30):,} videos): "
      f"median create_date = {video_yes30['create_date_dt'].dt.date.astype(str).mode()[0]}")
print(f"  No day-30   ({len(video_no30):,} videos): "
      f"median create_date = {video_no30['create_date_dt'].dt.date.astype(str).mode()[0]}")

# play_count at day 14 for both groups
pc14_yes30 = pc14.loc[pc14.index.isin(video30_ids)]
pc14_no30  = pc14.loc[~pc14.index.isin(video30_ids)]
print(f"  play_count(day14)  has day-30: median={pc14_yes30.median():.0f}  "
      f"mean={pc14_yes30.mean():.0f}")
print(f"  play_count(day14)  no day-30:  median={pc14_no30.median():.0f}  "
      f"mean={pc14_no30.mean():.0f}")

# -- Figure 2 --
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Ratio CDFs
for ratio, label, color in [
    (ratio7, "day 7 / day 30", "steelblue"),
    (ratio14, "day 14 / day 30", "darkorange"),
]:
    r_clean = ratio.dropna().clip(0, 1)
    sorted_r = np.sort(r_clean)
    axes[0].plot(sorted_r, np.linspace(0, 1, len(sorted_r)), label=label, color=color)
axes[0].axvline(0.9, color="gray", linestyle="--", alpha=0.6, label="90% threshold")
axes[0].set_xlabel("Ratio to day-30 play_count")
axes[0].set_ylabel("CDF")
axes[0].set_title("CDF of day7/day30 and day14/day30 ratios")
axes[0].legend(fontsize=8)

# Bias: create_date distributions
for grp, label, color in [
    (video_yes30, "has day-30", "darkorange"),
    (video_no30,  "no day-30",  "steelblue"),
]:
    axes[1].hist(grp["create_date_dt"].dt.dayofyear, bins=40, alpha=0.5,
                 label=label, color=color, density=True)
axes[1].set_xlabel("Day of year (2024)")
axes[1].set_ylabel("Density")
axes[1].set_title("Create date: has day-30 vs not")
axes[1].legend(fontsize=8)

# Bias: play_count(day14) distributions
for pc, label, color in [
    (pc14_yes30, "has day-30", "darkorange"),
    (pc14_no30,  "no day-30",  "steelblue"),
]:
    axes[2].hist(np.log1p(pc.clip(0)), bins=50, alpha=0.5,
                 label=label, color=color, density=True)
axes[2].set_xlabel("log1p(play_count at day 14)")
axes[2].set_ylabel("Density")
axes[2].set_title("play_count(day14): has day-30 vs not")
axes[2].legend(fontsize=8)

fig.suptitle("Day-30 bias check and ratio to day-30 engagement", fontweight="bold")
fig.tight_layout()
fig.savefig(FIGS / "fig2_day30_ratio_and_bias.png", dpi=120)
plt.close(fig)
print(f"\n  → Saved fig2_day30_ratio_and_bias.png")

# ================================================================== #
# PART 4 — Absolute vs relative target at day 14                      #
# ================================================================== #
print(f"\n{SEP}")
print("PART 4 — Absolute vs relative target (day 14), R² from author_id baseline")
print(SEP)

from creator_analytics.config import load_config
from creator_analytics.data import filter_creators, train_test_split_by_creator
from creator_analytics.targets import build_target

cfg = load_config("config/default.yaml")
min_vids = cfg["data"]["min_videos_per_creator"]
seed = cfg["seed"]

videos_filtered = filter_creators(videos, "author_id", min_videos=min_vids)
train_df, test_df = train_test_split_by_creator(
    videos_filtered, "author_id", test_size=0.2, seed=seed
)
train_ids = set(train_df["video_id"])
test_ids  = set(test_df["video_id"])
print(f"  After filter (>= {min_vids} videos): {len(videos_filtered):,} videos, "
      f"{videos_filtered['author_id'].nunique():,} creators")
print(f"  Train: {len(train_df):,} videos  |  Test: {len(test_df):,} videos")

# Build absolute and relative targets (day 14)
abs_target = build_target(engagement, videos_filtered, day=14, mode="absolute")
rel_target = build_target(
    engagement, videos_filtered, day=14, mode="relative", train_video_ids=train_ids
)
print(f"\n  Absolute target (day 14): n={len(abs_target):,}")
print(f"    mean={abs_target['target'].mean():.3f}  std={abs_target['target'].std():.3f}"
      f"  min={abs_target['target'].min():.2f}  max={abs_target['target'].max():.2f}")
print(f"\n  Relative target (day 14): n={len(rel_target):,}")
print(f"    mean={rel_target['target'].mean():.3f}  std={rel_target['target'].std():.3f}"
      f"  min={rel_target['target'].min():.2f}  max={rel_target['target'].max():.2f}")

# R² from author_id baseline model (train mean → predict test)
def author_r2(target_df: pd.DataFrame) -> tuple[float, float]:
    """Return (train_R2, test_R2) for the 'predict per-author mean' model."""
    td = target_df.merge(videos_filtered[["video_id", "author_id"]], on="video_id", how="inner")
    td_train = td[td["video_id"].isin(train_ids)]
    td_test  = td[td["video_id"].isin(test_ids)]

    author_means = td_train.groupby("author_id")["target"].mean()
    global_mean  = float(td_train["target"].mean())

    td_test = td_test.copy()
    td_test["pred"] = td_test["author_id"].map(author_means).fillna(global_mean)

    ss_res   = ((td_test["target"] - td_test["pred"]) ** 2).sum()
    ss_tot   = ((td_test["target"] - td_test["target"].mean()) ** 2).sum()
    r2_test  = float(1 - ss_res / ss_tot)

    ss_res_tr  = ((td_train["target"] - td_train["author_id"].map(author_means)) ** 2).sum()
    ss_tot_tr  = ((td_train["target"] - td_train["target"].mean()) ** 2).sum()
    r2_train   = float(1 - ss_res_tr / ss_tot_tr)
    return r2_train, r2_test

r2_abs_train, r2_abs_test = author_r2(abs_target)
r2_rel_train, r2_rel_test = author_r2(rel_target)
print(f"\n  R² of 'per-author mean' model (absolute target):  "
      f"train={r2_abs_train:.3f}  test={r2_abs_test:.3f}")
print(f"  R² of 'per-author mean' model (relative target):  "
      f"train={r2_rel_train:.3f}  test={r2_rel_test:.3f}")
print(f"\n  Interpretation: absolute target is ~{r2_abs_test:.0%} explained by author_id alone.")
print(f"  Relative target reduces this to ~{max(r2_rel_test, 0):.0%},"
      f" leaving content-driven variance.")

# -- Figure 3: distributions --
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(abs_target["target"], bins=80, color="steelblue", alpha=0.8, edgecolor="none")
axes[0].axvline(abs_target["target"].mean(), color="red", linestyle="--", linewidth=1.2,
                label=f"mean={abs_target['target'].mean():.2f}")
axes[0].set_title("Absolute target: log1p(play_count) at day 14")
axes[0].set_xlabel("log1p(play_count)"); axes[0].set_ylabel("Count")
axes[0].legend(fontsize=8)

axes[1].hist(rel_target["target"], bins=80, color="darkorange", alpha=0.8, edgecolor="none")
axes[1].axvline(rel_target["target"].mean(), color="red", linestyle="--", linewidth=1.2,
                label=f"mean={rel_target['target'].mean():.2f}")
axes[1].axvline(0, color="black", linestyle=":", linewidth=1, label="baseline=0")
axes[1].set_title("Relative target: log1p(play_count) − creator median (train) at day 14")
axes[1].set_xlabel("Relative log1p(play_count)"); axes[1].set_ylabel("Count")
axes[1].legend(fontsize=8)

fig.suptitle("Target distributions — absolute vs relative (day 14)", fontweight="bold")
fig.tight_layout()
fig.savefig(FIGS / "fig3_target_distributions.png", dpi=120)
plt.close(fig)

# -- Figure 4: R² comparison --
fig, ax = plt.subplots(figsize=(6, 4))
labels  = ["Absolute\n(test)", "Relative\n(test)"]
heights = [r2_abs_test, max(r2_rel_test, 0)]
colors  = ["steelblue", "darkorange"]
bars = ax.bar(labels, heights, color=colors, width=0.4)
for bar, h in zip(bars, heights):
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}", ha="center", fontsize=11)
ax.set_ylim(0, max(heights) * 1.25)
ax.set_ylabel("R²")
ax.set_title('R² of "predict per-author mean" model on test set\n'
             '(measures how much variance is author-level, not content-level)',
             fontsize=9)
fig.tight_layout()
fig.savefig(FIGS / "fig4_author_r2.png", dpi=120)
plt.close(fig)
print(f"\n  → Saved fig3_target_distributions.png, fig4_author_r2.png")

# ================================================================== #
# PART 5 — Follower count: as-of join coverage and correlation        #
# ================================================================== #
print(f"\n{SEP}")
print("PART 5 — Follower count from creator_daily (as-of join, no look-ahead)")
print(SEP)

vids_join = videos[["video_id", "author_id", "create_date"]].copy()
vids_join["create_date_dt"] = pd.to_datetime(vids_join["create_date"], errors="coerce")

crd_join = creator_daily[["author_id", "date", "follower_count"]].copy()
crd_join["crd_date_dt"] = pd.to_datetime(crd_join["date"], errors="coerce")
crd_join = crd_join.dropna(subset=["crd_date_dt"])

# As-of join: for each video, the last creator_daily row where date <= create_date
vids_sorted = vids_join.sort_values("create_date_dt")
crd_sorted  = crd_join.sort_values("crd_date_dt")

merged = pd.merge_asof(
    vids_sorted,
    crd_sorted.rename(columns={"crd_date_dt": "crd_dt"})[["author_id", "crd_dt", "follower_count"]],
    left_on="create_date_dt",
    right_on="crd_dt",
    by="author_id",
    direction="backward",
)
coverage_asof = merged["follower_count"].notna().mean()
print(f"  Coverage of as-of follower_count join: {coverage_asof*100:.1f}%  "
      f"({merged['follower_count'].notna().sum():,} / {len(merged):,} videos)")

# Correlation with absolute target at day 14
abs_for_corr = abs_target.merge(merged[["video_id", "follower_count"]], on="video_id", how="inner")
abs_for_corr = abs_for_corr.dropna(subset=["follower_count"])
abs_for_corr["log_followers"] = np.log1p(abs_for_corr["follower_count"].clip(0))

pearson_r  = abs_for_corr["log_followers"].corr(abs_for_corr["target"])
spearman_r = abs_for_corr["log_followers"].corr(abs_for_corr["target"], method="spearman")
print(f"\n  Correlation log1p(follower_count) vs absolute log1p(play_count day14):")
print(f"    Pearson r  = {pearson_r:.3f}")
print(f"    Spearman ρ = {spearman_r:.3f}")
print(f"    n          = {len(abs_for_corr):,}")

# -- Figure 5: scatter --
fig, ax = plt.subplots(figsize=(7, 5))
sample = abs_for_corr.sample(min(30_000, len(abs_for_corr)), random_state=42)
ax.scatter(sample["log_followers"], sample["target"],
           alpha=0.05, s=4, color="steelblue", rasterized=True)
ax.set_xlabel("log1p(follower_count at create_date)")
ax.set_ylabel("log1p(play_count at day 14)")
ax.set_title(f"Followers vs day-14 play_count\n"
             f"Pearson r = {pearson_r:.3f}, Spearman ρ = {spearman_r:.3f}  (n={len(abs_for_corr):,})")
fig.tight_layout()
fig.savefig(FIGS / "fig5_follower_vs_target.png", dpi=120)
plt.close(fig)
print(f"\n  → Saved fig5_follower_vs_target.png")

# ================================================================== #
# SUMMARY                                                             #
# ================================================================== #
print(f"\n{SEP}")
print("SUMMARY")
print(SEP)
cov_d7  = cov_df[cov_df["day"] == 7]["exact_pct"].iloc[0]
cov_d14 = cov_df[cov_df["day"] == 14]["exact_pct"].iloc[0]
cov_d30 = cov_df[cov_df["day"] == 30]["exact_pct"].iloc[0]
print(f"  Exact coverage   day 7: {cov_d7:.1f}%   day 14: {cov_d14:.1f}%   day 30: {cov_d30:.1f}%")
print(f"  Monotonicity violations: {n_violating_videos:,} videos ({n_violating_videos/total_videos*100:.2f}%)")
print(f"  Median day-14/day-30 ratio: {ratio14.median():.3f}  (day-7/day-30: {ratio7.median():.3f})")
print(f"  R² absolute (author_id model): {r2_abs_test:.3f}")
print(f"  R² relative (author_id model): {r2_rel_test:.3f}")
print(f"  Follower coverage (as-of join): {coverage_asof*100:.1f}%")
print(f"  Follower–target Spearman ρ: {spearman_r:.3f}")
print()
print("  Figures saved to results/figures/:")
for fig_name in sorted(FIGS.glob("fig*.png")):
    print(f"    {fig_name.name}")
