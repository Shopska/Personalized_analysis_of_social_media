"""Feature-engineering utilities: joins, transformations, aggregations."""

from __future__ import annotations

import pandas as pd


def follower_asof_join(
    videos: pd.DataFrame,
    creator_daily: pd.DataFrame,
) -> pd.DataFrame:
    """Add ``follower_count_at_publish`` to ``videos`` via a look-ahead-free join.

    For each video, finds the most recent ``creator_daily`` row where
    ``date <= create_date`` (backward as-of join).  No future data is used.

    Parameters
    ----------
    videos :
        Must have columns ``video_id``, ``author_id``, ``create_date``
        (date string parseable by ``pd.to_datetime``).
    creator_daily :
        Must have columns ``author_id``, ``date``, ``follower_count``.

    Returns
    -------
    pd.DataFrame
        ``videos`` with one added column ``follower_count_at_publish``.
        Videos without any matching creator_daily row (rare) receive NaN.
    """
    vids = videos[["video_id", "author_id", "create_date"]].copy()
    vids["_create_dt"] = pd.to_datetime(vids["create_date"], errors="coerce")
    vids_sorted = vids.dropna(subset=["_create_dt"]).sort_values("_create_dt")

    crd = creator_daily[["author_id", "date", "follower_count"]].copy()
    crd["_date_dt"] = pd.to_datetime(crd["date"], errors="coerce")
    crd_sorted = crd.dropna(subset=["_date_dt"]).sort_values("_date_dt")

    merged = pd.merge_asof(
        vids_sorted,
        crd_sorted[["author_id", "_date_dt", "follower_count"]],
        left_on="_create_dt",
        right_on="_date_dt",
        by="author_id",
        direction="backward",
    )
    return videos.merge(
        merged[["video_id", "follower_count"]].rename(
            columns={"follower_count": "follower_count_at_publish"}
        ),
        on="video_id",
        how="left",
    )
