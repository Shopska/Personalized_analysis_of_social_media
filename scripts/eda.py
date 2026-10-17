#!/usr/bin/env python
"""Quick EDA over the three raw tables.

Prints all stats to stdout; no files written.
Run: python scripts/eda.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
TABLES = ["videos", "creator_daily", "engagement_daily"]


def file_mb(table: str) -> float:
    return (RAW_DIR / f"{table}.parquet").stat().st_size / 1_048_576


def null_frac(df: pd.DataFrame) -> pd.Series:
    return (df.isna().sum() / len(df)).round(4)


def date_range(df: pd.DataFrame) -> str:
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return f"{df[col].min().date()} – {df[col].max().date()}"
        if "date" in col.lower() or "time" in col.lower() or "day" in col.lower():
            try:
                s = pd.to_datetime(df[col], errors="coerce")
                if s.notna().sum() > len(df) * 0.5:
                    return f"{s.min().date()} – {s.max().date()}"
            except Exception:
                pass
    return "n/a"


print("=" * 70)
print("TABLE OVERVIEW")
print("=" * 70)

dfs: dict[str, pd.DataFrame] = {}
for t in TABLES:
    df = pd.read_parquet(RAW_DIR / f"{t}.parquet")
    dfs[t] = df
    print(f"\n--- {t} ({len(df):,} rows × {len(df.columns)} cols, {file_mb(t):.1f} MB) ---")
    print(f"  Date range: {date_range(df)}")
    print("  Columns:")
    for col in df.columns:
        dtype = str(df[col].dtype)
        nf = df[col].isna().mean()
        null_str = f"  {nf:.1%} null" if nf > 0 else ""
        print(f"    {col:<35} {dtype:<15}{null_str}")
    nonzero_null = null_frac(df)
    nonzero_null = nonzero_null[nonzero_null > 0]
    if nonzero_null.empty:
        print("  Nulls: none")

# ---- videos: distribution of videos per creator ---
print("\n" + "=" * 70)
print("VIDEOS PER CREATOR (from videos table)")
print("=" * 70)
vids = dfs["videos"]
counts = vids["author_id"].value_counts()
print(f"  Total creators : {counts.shape[0]:,}")
print(f"  Min videos     : {counts.min()}")
print(f"  Median videos  : {counts.median():.0f}")
print(f"  Max videos     : {counts.max()}")
print()
for threshold in [10, 20, 30, 50]:
    n_creators = (counts >= threshold).sum()
    n_videos = counts[counts >= threshold].sum()
    print(f"  min_videos >= {threshold:2d}: {n_creators:4d} creators, {n_videos:,} videos")

# ---- join integrity ---
print("\n" + "=" * 70)
print("JOIN INTEGRITY")
print("=" * 70)
eng = dfs["engagement_daily"]
crd = dfs["creator_daily"]

videos_ids = set(vids["video_id"])
eng_ids = set(eng["video_id"])
missing_in_eng = videos_ids - eng_ids
missing_in_vid = eng_ids - videos_ids
print(f"  video_id in videos but NOT in engagement_daily : {len(missing_in_eng):,}")
print(f"  video_id in engagement_daily but NOT in videos  : {len(missing_in_vid):,}")

video_authors = set(vids["author_id"])
crd_authors = set(crd["author_id"])
missing_authors = video_authors - crd_authors
print(f"  author_id in videos but NOT in creator_daily    : {len(missing_authors):,}")
extra_authors = crd_authors - video_authors
print(f"  author_id in creator_daily but NOT in videos    : {len(extra_authors):,}")

# ---- engagement_daily: days_since_post distribution ---
print("\n" + "=" * 70)
print("ENGAGEMENT_DAILY: days_since_post")
print("=" * 70)
dsp = eng["days_since_post"]
print(f"  min  : {dsp.min()}")
print(f"  p25  : {dsp.quantile(0.25):.0f}")
print(f"  p50  : {dsp.median():.0f}")
print(f"  p75  : {dsp.quantile(0.75):.0f}")
print(f"  max  : {dsp.max()}")
print()
total_videos_eng = eng["video_id"].nunique()
print(f"  Unique videos with any engagement record: {total_videos_eng:,}")
for day in [7, 14, 30]:
    n = eng[eng["days_since_post"] >= day]["video_id"].nunique()
    pct = n / total_videos_eng * 100
    print(f"  Videos with data at day >= {day:2d}: {n:,}  ({pct:.1f}% of videos with any record)")

# ---- engagement columns quick look ---
print("\n" + "=" * 70)
print("ENGAGEMENT_DAILY: metric columns (non-null rows sample stats)")
print("=" * 70)
metric_cols = [c for c in eng.columns if c not in ("video_id", "author_id", "days_since_post", "date", "collect_date")]
for col in metric_cols:
    s = eng[col].dropna()
    print(f"  {col:<30} min={s.min():.0f}  median={s.median():.0f}  max={s.max():.0f}")

print()
