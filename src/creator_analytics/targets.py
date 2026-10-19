"""Build the prediction target from engagement_daily.

Usage
-----
>>> from creator_analytics.targets import build_target
>>> target_df = build_target(engagement, videos, day=14, mode="absolute")
>>> # or, leakage-safe relative:
>>> target_df = build_target(
...     engagement, videos, day=14, mode="relative", train_video_ids=train_ids
... )
"""

from __future__ import annotations

from collections.abc import Collection

import numpy as np
import pandas as pd


def get_nonmonotonic_video_ids(engagement: pd.DataFrame) -> frozenset[str]:
    """Return video_ids that have at least one decrease in play_count over time.

    These videos (1,240 in the full dataset) likely experienced a data-collection
    reset and should be excluded before building the target.
    """
    eng_s = engagement.sort_values(["video_id", "days_since_post"])
    diffs = eng_s.groupby("video_id")["play_count"].diff()
    return frozenset(eng_s.loc[diffs < 0, "video_id"].unique())


def build_target(
    engagement: pd.DataFrame,
    videos: pd.DataFrame,
    day: int,
    mode: str = "absolute",
    train_video_ids: Collection[str] | None = None,
) -> pd.DataFrame:
    """Build ``log1p(play_count)`` at ``day`` days after posting.

    Parameters
    ----------
    engagement :
        engagement_daily table; must have columns ``video_id``,
        ``days_since_post``, ``play_count``.
    videos :
        videos table; must have columns ``video_id``, ``author_id``.
        Only needed for mode="relative".
    day :
        Measurement point. Only videos with a record at *exactly*
        ``days_since_post == day`` are returned.  Call sites must decide
        what to do with videos that have no record at this day.
    mode :
        ``"absolute"``  — ``log1p(play_count)`` unchanged.
        ``"relative"``  — ``log1p(play_count)`` minus the per-creator
        median of ``log1p(play_count)`` in *training* videos only.
        Subtracting a creator-level baseline removes scale differences
        between big and small creators, leaving content-driven variance.
    train_video_ids :
        Required when ``mode="relative"``.  The set of ``video_id``
        values that belong to the training split.  Creator baselines are
        computed on this subset exclusively; applying them to test videos
        prevents target leakage from test set into the baseline.

    Returns
    -------
    pd.DataFrame
        Columns: ``video_id``, ``target``.  Index is reset.
        Only videos with a record at exactly ``day`` days_since_post
        appear in the output.

    Raises
    ------
    ValueError
        If ``mode`` is not ``"absolute"`` or ``"relative"``, or if
        ``mode="relative"`` is requested without ``train_video_ids``.
    """
    if mode not in ("absolute", "relative"):
        raise ValueError(f"mode must be 'absolute' or 'relative', got {mode!r}")
    if mode == "relative" and train_video_ids is None:
        raise ValueError(
            "train_video_ids is required for mode='relative' to prevent leakage: "
            "the per-creator baseline must be computed on training data only."
        )

    # Restrict to video_ids present in the videos table that was passed in.
    # Without this, engagement_daily rows for videos excluded by earlier filters
    # (e.g. min_videos_per_creator, non-monotonic exclusion) would appear in the output.
    video_ids_in_scope: frozenset[str] = frozenset(videos["video_id"])

    eng_day = (
        engagement.loc[
            (engagement["days_since_post"] == day)
            & engagement["video_id"].isin(video_ids_in_scope),
            ["video_id", "play_count"],
        ]
        .copy()
        .reset_index(drop=True)
    )
    eng_day["target"] = np.log1p(eng_day["play_count"].astype(float))

    if mode == "absolute":
        return eng_day[["video_id", "target"]]

    # --- relative: subtract per-creator median computed on train only ---
    train_ids_set: frozenset[str] = frozenset(train_video_ids)  # type: ignore[arg-type]
    train_eng = eng_day[eng_day["video_id"].isin(train_ids_set)].merge(
        videos[["video_id", "author_id"]], on="video_id", how="inner"
    )
    creator_medians: pd.Series = train_eng.groupby("author_id")["target"].median()
    # Fallback for any creator that has no training video at this day (rare edge case)
    global_fallback: float = float(train_eng["target"].median())

    eng_day = eng_day.merge(videos[["video_id", "author_id"]], on="video_id", how="inner")
    eng_day["creator_median"] = eng_day["author_id"].map(creator_medians).fillna(global_fallback)
    eng_day["target"] = eng_day["target"] - eng_day["creator_median"]

    return eng_day[["video_id", "target"]].reset_index(drop=True)
