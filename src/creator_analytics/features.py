"""Feature-engineering for TikTok creator analytics.

Public API
----------
build_features(videos, creator_daily, train_video_ids) -> pd.DataFrame
    Builds all feature groups and joins them on video_id.

follower_asof_join(videos, creator_daily) -> pd.DataFrame
    Adds follower_count_at_publish via a look-ahead-free as-of join
    (kept for backward compatibility with notebook 02).

FEATURE_DESCRIPTIONS
    Dict mapping feature name -> human-readable Ukrainian description.
    Used downstream to translate SHAP values into creator recommendations.

Leakage rules enforced here
----------------------------
- No columns from engagement_daily appear in any feature function.
- Anything computed from data (top-N hashtags, rare-category grouping) uses
  only train_video_ids and is then applied to the full dataset.
- Anything that depends on time (sound popularity, follower count, video count,
  "previous video" stats) uses strictly past data relative to each video's
  publish date.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Collection
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ #
# Emoji regex — covers the most common Unicode emoji ranges           #
# ------------------------------------------------------------------ #
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F004-\U0001F0CF"
    "\U0001F910-\U0001F9FF"
    "]",
    re.UNICODE,
)

# ------------------------------------------------------------------ #
# Feature descriptions (Ukrainian) for SHAP → recommendation layer    #
# ------------------------------------------------------------------ #
FEATURE_DESCRIPTIONS: dict[str, str] = {
    # Group 1 — Format
    "duration_log": "Тривалість відео у логарифмованих секундах (log1p)",
    "ratio": "Роздільна здатність відео (540p, 720p тощо)",
    "created_by_ai": "Чи позначено відео як AI-згенероване (0/1)",
    "is_ads": "Рекламне відео (0/1); у датасеті завжди 0 — нульова варіативність",
    # Group 2 — Timing  [create_time likely UTC; no tz marker in data]
    "publish_hour": "Година публікації (ймовірно UTC, tz невідомий)",
    "publish_dow": "День тижня публікації (0=Пн, 6=Нд; ймовірно UTC)",
    "days_since_prev_video": "Кількість днів з попереднього відео цього ж автора",
    "videos_last_7d": "Кількість відео автора за 7 днів до публікації (не включаючи день публікації)",
    # Group 3 — Description
    "desc_len": "Довжина опису відео у символах (0 якщо опис відсутній)",
    "desc_n_words": "Кількість слів в описі (пробільна токенізація)",
    "desc_n_emojis": "Кількість емоджі в описі",
    "desc_n_questions": "Кількість знаків питання в описі",
    "n_hashtags": "Кількість хештегів (з масиву hashtags, без пропусків)",
    "has_sticker_text": "Чи є текст на екрані (sticker text) (0/1)",
    "sticker_text_len": "Довжина тексту на екрані у символах (0 якщо відсутній)",
    "is_english": "Чи визначено мову опису як англійську (0/1)",
    # Group 4 — Speech
    "has_transcript": (
        "Чи є транскрипт мовлення; відсутність = швидше за все відео без голосу"
    ),
    "speaking_rate": "Швидкість мовлення (слів на хвилину); NaN якщо немає транскрипту",
    "transcript_n_words": "Кількість слів у транскрипті; NaN якщо немає транскрипту",
    # Group 5 — Hook (proxy; noisy for unformatted transcripts)
    "hook_first_sentence_len": (
        "Довжина першого речення транскрипту; NaN якщо немає транскрипту. "
        "НЕНАДІЙНО: усні транскрипти часто без пунктуації"
    ),
    "hook_first_sentence_is_question": (
        "Чи закінчується перше речення транскрипту на '?'; NaN якщо немає транскрипту. "
        "НЕНАДІЙНО: та сама причина"
    ),
    # Group 6 — Emotions
    "anger": "Частка гніву в емоційній моделі",
    "joy": "Частка радості в емоційній моделі",
    "surprise": "Частка здивування в емоційній моделі",
    "sadness": "Частка суму в емоційній моделі",
    "disgust": "Частка огиди в емоційній моделі",
    "fear": "Частка страху в емоційній моделі",
    "dominant_emotion": "Домінуюча емоція (назва стовпця з максимальним значенням)",
    # Group 7 — Topic
    "topic": "Тематична категорія відео (рядок-категорія)",
    # Group 8 — Music
    "music_selected_from": "Звідки обрано музику (проксі шаблону монтажу)",
    "original_sound": "Чи є звук оригінальним (music_owner_id == author_id) (0/1)",
    "sound_popularity": (
        "Кількість відео з тим же music_id, опублікованих СТРОГО ДО цього відео "
        "(тільки минуле — немає leakage)"
    ),
    # Group 9 — Hashtag indicators (keys added dynamically: hashtag_{name})
    # Group 10 — Creator context
    "follower_count_log": "Кількість підписників на момент публікації (log1p, as-of join)",
    "video_count_at_publish": "Загальна кількість відео автора на момент публікації (as-of join)",
}


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _extract_hashtag_names(hval) -> list[str]:
    """Return lowercase hashtag_name strings from one cell of the hashtags column."""
    if hval is None:
        return []
    if not hasattr(hval, "__iter__") or isinstance(hval, str):
        return []
    out = []
    for item in hval:
        if isinstance(item, dict):
            name = item.get("hashtag_name", "")
            if name:
                out.append(str(name).lower())
    return out


# ------------------------------------------------------------------ #
# Group 1 — Format                                                     #
# ------------------------------------------------------------------ #

def format_features(videos: pd.DataFrame) -> pd.DataFrame:
    """1. Format: log1p(duration), ratio, created_by_ai, is_ads."""
    out = videos[["video_id"]].copy()
    out["duration_log"] = np.log1p(videos["duration"].astype(float))
    out["ratio"] = videos["ratio"]
    out["created_by_ai"] = videos["created_by_ai"].fillna(0).astype(np.int8)
    # is_ads is always 0 in this dataset; included for completeness
    out["is_ads"] = pd.to_numeric(videos["is_ads"], errors="coerce").fillna(0).astype(np.int8)
    return out


# ------------------------------------------------------------------ #
# Group 2 — Timing                                                     #
# ------------------------------------------------------------------ #

def timing_features(videos: pd.DataFrame) -> pd.DataFrame:
    """2. Timing: hour, day-of-week, days_since_prev_video, videos_last_7d.

    create_time contains no timezone marker; values are treated as-is (likely UTC).
    days_since_prev_video and videos_last_7d use only strictly past data per creator.
    """
    out = videos[["video_id", "author_id"]].copy()
    dt = pd.to_datetime(videos["create_time"], errors="coerce")
    out["publish_hour"] = dt.dt.hour.astype("Int8")
    out["publish_dow"] = dt.dt.dayofweek.astype("Int8")
    out["_dt"] = pd.to_datetime(videos["create_date"], errors="coerce")

    # Sort by (author_id, date) for temporal ops
    out = out.sort_values(["author_id", "_dt"]).reset_index(drop=True)

    # days_since_prev_video — shift(1) within each author group
    prev_dt = out.groupby("author_id")["_dt"].shift(1)
    out["days_since_prev_video"] = (out["_dt"] - prev_dt).dt.days

    # videos_last_7d — count author's videos published in [dt-7d, dt)
    def _last7d(group: pd.DataFrame) -> pd.Series:
        dt_arr = group["_dt"].values.astype("datetime64[D]")
        window = np.timedelta64(7, "D")
        counts = np.empty(len(dt_arr), dtype=np.int32)
        for i in range(len(dt_arr)):
            left = np.searchsorted(dt_arr, dt_arr[i] - window, side="left")
            counts[i] = i - left  # i videos before index i within the window
        return pd.Series(counts, index=group.index)

    out["videos_last_7d"] = out.groupby("author_id", group_keys=False).apply(_last7d)

    return out[["video_id", "publish_hour", "publish_dow",
                "days_since_prev_video", "videos_last_7d"]]


# ------------------------------------------------------------------ #
# Group 3 — Description                                                #
# ------------------------------------------------------------------ #

def description_features(videos: pd.DataFrame) -> pd.DataFrame:
    """3. Description: desc_len, n_words, n_emojis, n_questions, n_hashtags,
    has_sticker_text, sticker_text_len, is_english.
    All computed fresh; no imputed nulls (null desc treated as empty string).
    """
    desc = videos["desc"].fillna("").astype(str)
    sticker = videos["sticker_text"]

    out = videos[["video_id"]].copy()
    out["desc_len"] = desc.str.len().astype(np.int32)
    out["desc_n_words"] = desc.str.split().str.len().fillna(0).astype(np.int32)
    out["desc_n_emojis"] = desc.apply(lambda t: len(_EMOJI_RE.findall(t))).astype(np.int16)
    out["desc_n_questions"] = desc.str.count(r"\?").astype(np.int16)
    out["n_hashtags"] = (
        videos["hashtags"]
        .apply(lambda h: len(h) if hasattr(h, "__len__") and not isinstance(h, str) else 0)
        .astype(np.int16)
    )
    out["has_sticker_text"] = sticker.notna().astype(np.int8)
    out["sticker_text_len"] = (
        sticker.fillna("").str.len().astype(np.int32)
    )
    out["is_english"] = videos["is_english"].fillna(0).astype(np.int8)
    return out


# ------------------------------------------------------------------ #
# Group 4 — Speech                                                     #
# ------------------------------------------------------------------ #

def speech_features(videos: pd.DataFrame) -> pd.DataFrame:
    """4. Speech: has_transcript, speaking_rate, transcript_n_words.
    Nulls preserved — XGBoost handles them natively.
    """
    transcript = videos["transcript"]
    out = videos[["video_id"]].copy()
    out["has_transcript"] = transcript.notna().astype(np.int8)
    out["speaking_rate"] = pd.to_numeric(videos["speaking_rate"], errors="coerce")
    out["transcript_n_words"] = transcript.str.split().str.len()
    return out


# ------------------------------------------------------------------ #
# Group 5 — Hook (proxy; noisy for unpunctuated transcripts)           #
# ------------------------------------------------------------------ #

_SENT_SPLIT_RE = re.compile(r"[.!?]+")


def hook_features(videos: pd.DataFrame) -> pd.DataFrame:
    """5. Hook: length and question-flag for first transcript sentence.

    Reliability caveat: spoken transcripts typically lack punctuation, so
    "sentence" splitting is coarse — the entire transcript may appear as one
    sentence. Treat these features as weak proxies.
    """
    def _first_sent(text: str | float) -> str | None:
        if not isinstance(text, str) or not text.strip():
            return None
        parts = _SENT_SPLIT_RE.split(text.strip())
        return parts[0].strip() if parts else None

    first = videos["transcript"].apply(_first_sent)
    out = videos[["video_id"]].copy()
    out["hook_first_sentence_len"] = first.str.len()
    out["hook_first_sentence_is_question"] = (
        videos["transcript"]
        .apply(lambda t: int(bool(isinstance(t, str) and t.strip().endswith("?")))
               if isinstance(t, str) else None)
    )
    return out


# ------------------------------------------------------------------ #
# Group 6 — Emotions                                                   #
# ------------------------------------------------------------------ #

_EMOTION_COLS = ["anger", "joy", "surprise", "sadness", "disgust", "fear"]


def emotion_features(videos: pd.DataFrame) -> pd.DataFrame:
    """6. Emotions: 6 scores (existing) + dominant_emotion (argmax of scores)."""
    out = videos[["video_id"]].copy()
    for col in _EMOTION_COLS:
        out[col] = pd.to_numeric(videos[col], errors="coerce")
    em_vals = out[_EMOTION_COLS].to_numpy(dtype=float, na_value=np.nan)
    null_rows = np.all(np.isnan(em_vals), axis=1)
    # Replace NaN with -inf so argmax ignores them; null_rows flagged separately
    em_safe = np.where(np.isnan(em_vals), -np.inf, em_vals)
    dom_idx = np.argmax(em_safe, axis=1)
    dom = np.array([_EMOTION_COLS[i] for i in dom_idx], dtype=object)
    dom[null_rows] = None
    out["dominant_emotion"] = dom
    return out


# ------------------------------------------------------------------ #
# Group 7 — Topic                                                      #
# ------------------------------------------------------------------ #

def topic_features(videos: pd.DataFrame) -> pd.DataFrame:
    """7. Topic: the pre-extracted topic category string."""
    return videos[["video_id", "topic"]].copy()


# ------------------------------------------------------------------ #
# Group 8 — Music                                                      #
# ------------------------------------------------------------------ #

def music_features(videos: pd.DataFrame) -> pd.DataFrame:
    """8. Music: music_selected_from, original_sound, sound_popularity.

    sound_popularity: number of videos in this dataset that used the same
    music_id STRICTLY BEFORE this video's publish date (no same-day counts).
    """
    out = videos[["video_id"]].copy()
    out["music_selected_from"] = videos["music_selected_from"]

    # original_sound: 1 when music_owner_id == author_id (creator's own sound)
    own_music = videos["music_owner_id"].fillna("") == videos["author_id"].fillna("_na_")
    out["original_sound"] = own_music.astype(np.int8)

    # sound_popularity: cumulative past usage of the same music_id
    sounds = videos[["video_id", "music_id", "create_date"]].copy()
    sounds["_dt"] = pd.to_datetime(sounds["create_date"], errors="coerce")

    music_daily = (
        sounds.dropna(subset=["_dt", "music_id"])
        .groupby(["music_id", "_dt"])
        .size()
        .reset_index(name="_n")
        .sort_values(["music_id", "_dt"])
    )
    music_daily["_cum"] = music_daily.groupby("music_id")["_n"].cumsum()
    # popularity before this day = cumsum_through_today - n_today
    music_daily["popularity_before"] = music_daily["_cum"] - music_daily["_n"]

    sounds = sounds.merge(
        music_daily[["music_id", "_dt", "popularity_before"]],
        on=["music_id", "_dt"],
        how="left",
    )
    out["sound_popularity"] = sounds["popularity_before"].fillna(0).astype(np.int32).values
    return out


# ------------------------------------------------------------------ #
# Group 9 — Hashtag indicators (train-only top-20)                    #
# ------------------------------------------------------------------ #

def hashtag_indicator_features(
    videos: pd.DataFrame,
    train_video_ids: Collection[str] | None = None,
) -> pd.DataFrame:
    """9. Binary indicators for top-20 hashtags determined on train only.

    If train_video_ids is None, top-20 are computed on all videos (for use
    without a train/test split, e.g. diagnostics).
    """
    if train_video_ids is not None:
        train_set = frozenset(train_video_ids)
        train_videos = videos[videos["video_id"].isin(train_set)]
    else:
        train_videos = videos

    all_tags: list[str] = []
    for hval in train_videos["hashtags"]:
        all_tags.extend(_extract_hashtag_names(hval))
    top20 = [tag for tag, _ in Counter(all_tags).most_common(20)]

    video_tags = videos["hashtags"].apply(_extract_hashtag_names)
    out = videos[["video_id"]].copy()
    for tag in top20:
        out[f"hashtag_{tag}"] = video_tags.apply(lambda ts, t=tag: int(t in ts)).astype(np.int8)

    # Store chosen tags for downstream use (not a column — just metadata)
    out.attrs["top_hashtags"] = top20
    return out


# ------------------------------------------------------------------ #
# Group 10 — Creator context (as-of join, no look-ahead)              #
# ------------------------------------------------------------------ #

def creator_context_features(
    videos: pd.DataFrame,
    creator_daily: pd.DataFrame,
) -> pd.DataFrame:
    """10. log1p(follower_count_at_publish) and video_count_at_publish.

    Both use a backward as-of join on creator_daily: for each video, finds
    the most recent snapshot dated on or before the video's publish date.
    No future data is used.
    """
    vids = videos[["video_id", "author_id", "create_date"]].copy()
    vids["_vdt"] = pd.to_datetime(vids["create_date"], errors="coerce")
    vids_s = vids.dropna(subset=["_vdt"]).sort_values("_vdt")

    crd = creator_daily[["author_id", "date", "follower_count", "video_count"]].copy()
    crd["_cdt"] = pd.to_datetime(crd["date"], errors="coerce")
    crd_s = crd.dropna(subset=["_cdt"]).sort_values("_cdt")

    merged = pd.merge_asof(
        vids_s,
        crd_s[["author_id", "_cdt", "follower_count", "video_count"]],
        left_on="_vdt",
        right_on="_cdt",
        by="author_id",
        direction="backward",
    )
    merged["follower_count_log"] = np.log1p(
        pd.to_numeric(merged["follower_count"], errors="coerce").clip(lower=0)
    )
    merged["video_count_at_publish"] = pd.to_numeric(
        merged["video_count"], errors="coerce"
    )
    result = videos[["video_id"]].merge(
        merged[["video_id", "follower_count_log", "video_count_at_publish"]],
        on="video_id",
        how="left",
    )
    return result


# ------------------------------------------------------------------ #
# follower_asof_join — kept for backward compatibility (notebook 02)  #
# ------------------------------------------------------------------ #

def follower_asof_join(
    videos: pd.DataFrame,
    creator_daily: pd.DataFrame,
) -> pd.DataFrame:
    """Add ``follower_count_at_publish`` to ``videos`` via a look-ahead-free join.

    For each video, finds the most recent ``creator_daily`` row where
    ``date <= create_date`` (backward as-of join).  No future data is used.
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


# ------------------------------------------------------------------ #
# Main entry point                                                     #
# ------------------------------------------------------------------ #

def build_features(
    videos: pd.DataFrame,
    creator_daily: pd.DataFrame,
    train_video_ids: Collection[str] | None = None,
) -> pd.DataFrame:
    """Build all feature groups and return a single DataFrame indexed by video_id.

    Parameters
    ----------
    videos :
        Filtered videos table (after min_videos_per_creator and non-monotonic
        exclusion).  Must NOT contain engagement_daily columns.
    creator_daily :
        Full creator_daily table (as-of join uses all historical rows).
    train_video_ids :
        video_ids in the training split.  Required for leakage-safe features
        (top-hashtag indicators).  If None, all videos are used as proxy train.

    Returns
    -------
    pd.DataFrame
        One row per video_id; columns are features only (no raw text, no IDs
        other than video_id).
    """
    groups = [
        format_features(videos),
        timing_features(videos),
        description_features(videos),
        speech_features(videos),
        hook_features(videos),
        emotion_features(videos),
        topic_features(videos),
        music_features(videos),
        hashtag_indicator_features(videos, train_video_ids),
        creator_context_features(videos, creator_daily),
    ]
    result = videos[["video_id"]].copy()
    for df in groups:
        result = result.merge(df, on="video_id", how="left")

    # Dynamically update FEATURE_DESCRIPTIONS for hashtag indicators
    top_tags = groups[8].attrs.get("top_hashtags", [])
    for tag in top_tags:
        key = f"hashtag_{tag}"
        if key not in FEATURE_DESCRIPTIONS:
            FEATURE_DESCRIPTIONS[key] = (
                f"Відео використовує хештег #{tag} (0/1; топ-20 хештеги тільки з train)"
            )
    return result
