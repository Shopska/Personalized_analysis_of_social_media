# data/ — data card

> Numbers come from `notebooks/01-data-exploration.ipynb` after downloading the data
> with `python scripts/download_data.py`.

## Name and source

**TikTok Video Engagement 200k** (`lingbow/tiktok-video-engagement-200k`), hosted on
Hugging Face Datasets.

- Dataset page: <https://huggingface.co/datasets/lingbow/tiktok-video-engagement-200k>
- Three subsets: `videos`, `creator_daily`, `engagement_daily`, each with a single
  split called `"train"`.
- 1,872 unique creators; `author_id` is the join key across all three tables.
- Content features were pre-extracted by the dataset authors using AI tools
  (transcripts, generated summaries, emotion scores, topic labels). We use these
  as-is instead of running our own video analysis.

## License and terms of use

**CC-BY-NC-4.0** (Creative Commons Attribution-NonCommercial 4.0).

What this means for this project:

- **Allowed**: use for non-commercial purposes, such as this university course project,
  as long as the source is credited.
- **Not allowed**: any commercial use of the data, or of anything derived from it.
- **Models and results inherit this restriction.** Trained models and per-creator
  insights are derived from NC data, so they are also non-commercial only — even
  though the code in this repository is MIT-licensed (see the root `LICENSE`).
- We do **not** redistribute the data: it is downloaded on demand by
  `scripts/download_data.py` and never committed to git.

## Size and format

Three Parquet files in `data/raw/`, downloaded by `scripts/download_data.py`.

| Table              | Rows       | Cols | File size |
|--------------------|------------|------|-----------|
| `videos`           | 209,543    | 33   | 175.1 MB  |
| `creator_daily`    | 278,433    | 7    | 3.1 MB    |
| `engagement_daily` | 6,068,955  | 10   | 31.8 MB   |

Date range covered: **2024-06-24 – 2024-11-09** (videos); creator_daily and
engagement_daily extend to 2024-12-09 (data collected after the last video was posted).

## Schema

### `videos` — one row per video

| Column | Type | Null % | Notes |
|---|---|---|---|
| `video_id` | str | 0 | Primary key |
| `author_id` | str | 0 | Join key to creator_daily and engagement_daily |
| `create_time` | str | 0 | Timestamp as string |
| `create_date` | str | 0 | Date portion of create_time |
| `duration` | float64 | 0 | Video duration in seconds |
| `ratio` | str | 0 | Aspect ratio |
| `desc_language` | str | 0 | Detected language of description |
| `is_english` | int64 | 0 | 1 if desc_language == en |
| `desc` | str | 9.1% | Video caption/description |
| `sticker_text` | str | 52.8% | Text overlaid as stickers |
| `hashtags` | object | 0 | List of hashtag strings |
| `created_by_ai` | float64 | ~0 | AI-generated flag |
| `is_ads` | str | 0 | Ad / non-ad label |
| `music_selected_from` | str | 1.3% | Source of background music |
| `music_album` | str | 77.6% | Album name (mostly null) |
| `music_author` | str | 0.2% | Music artist |
| `music_owner_id` | str | 24.5% | Music owner identifier |
| `music_id` | str | ~0 | Music track identifier |
| `music_title` | str | 0.1% | Music track title |
| `transcript` | str | 60.3% | Speech-to-text transcript (missing ~60%) |
| `word_count` | float64 | 37.9% | Word count of transcript |
| `emoji_count` | float64 | 37.9% | Emoji count in transcript |
| `question_count` | float64 | 37.9% | Question count in transcript |
| `hashtag_count` | float64 | 37.9% | Hashtag count |
| `speaking_rate` | float64 | 37.9% | Words per minute |
| `gpt_summary` | str | 0 | AI-generated summary (always present) |
| `topic` | str | 0 | Topic category |
| `anger` | float64 | 0.5% | Emotion score |
| `joy` | float64 | 0.5% | Emotion score |
| `surprise` | float64 | 0.5% | Emotion score |
| `sadness` | float64 | 0.5% | Emotion score |
| `disgust` | float64 | 0.5% | Emotion score |
| `fear` | float64 | 0.5% | Emotion score |

**Notable:** `transcript` is missing for 60% of videos; `gpt_summary` is always present
and will be the primary text feature. The transcript-derived columns (word_count, etc.)
have 37.9% null — not 60%, because some are imputed or computed from other sources.
**This table has NO engagement metrics** — views/likes are only in `engagement_daily`.

### `creator_daily` — one row per creator per day

| Column | Type | Null % | Notes |
|---|---|---|---|
| `author_id` | str | 0 | Join key |
| `date` | str | 0 | Snapshot date |
| `follower_count` | float64 | 0.1% | Followers on that date |
| `following_count` | float64 | 0.1% | Following on that date |
| `total_favorited` | int64 | 0 | Total likes received (cumulative) |
| `video_count` | float64 | ~0 | Total videos published (cumulative) |
| `enterprise_verified` | str | 100% | **Entirely null — unusable** |

Date range: 2024-06-24 – 2024-12-09

### `engagement_daily` — one row per video per day since posting

| Column | Type | Null % | Notes |
|---|---|---|---|
| `video_id` | str | 0 | Join key to videos |
| `date` | str | 0 | Calendar date of measurement |
| `days_since_post` | int64 | 0 | 0 = day of posting, max = 30 |
| `play_count` | int64 | 0 | Cumulative plays as of this day |
| `like_count` | int64 | 0 | Cumulative likes |
| `comment_count` | int64 | 0 | Cumulative comments |
| `share_count` | int64 | 0 | Cumulative shares |
| `collect_count` | int64 | 0 | Cumulative collects/saves |
| `download_count` | int64 | 0 | Cumulative downloads |
| `whatsapp_share_count` | int64 | 0 | Cumulative WhatsApp shares |

**No nulls anywhere.** All metrics are cumulative counts from day of posting.
`days_since_post` ranges 0–30 (p25=7, p50=14, p75=22).

Engagement metric ranges (across all rows):

| Metric | Median | Max |
|---|---|---|
| play_count | 570 | 44,442,900 |
| like_count | 42 | 3,044,931 |
| comment_count | 3 | 67,276 |
| share_count | 0 | 991,356 |
| collect_count | 2 | 242,201 |
| download_count | 0 | 42,972 |
| whatsapp_share_count | 0 | 14,159 |

## Join keys and integrity

| Join | Key | Coverage |
|---|---|---|
| videos ↔ engagement_daily | `video_id` | **100%** — every video has engagement records; every engagement record has a matching video |
| videos ↔ creator_daily | `author_id` | **100%** — every author in videos has daily creator stats and vice versa |

The three tables form a clean star schema: no orphan rows.

## `data/raw/` vs. `data/interim/` vs. `data/processed/`

- `data/raw/*.parquet` — the tables exactly as downloaded from Hugging Face.
- `data/interim/` — reserved for intermediate files (currently unused; cleaning and
  filtering happen in memory in `creator_analytics.data`).
- `data/processed/` — model-ready features produced by `src/creator_analytics/features.py`.

None of these files are committed to git; only this README and `data/raw/CHECKSUMS.txt` are.

## How to get it

```bash
python scripts/download_data.py          # all three tables
python scripts/download_data.py --table videos   # one table only
```

This downloads all three subsets from Hugging Face into `data/raw/` and writes sha256
checksums to `data/raw/CHECKSUMS.txt`. Running it again is a no-op if checksums already
match; use `--force` to re-download.

## Distribution of videos across creators

| | Value |
|---|---|
| Total creators (videos table) | 1,872 |
| Min videos per creator | 5 |
| Median videos per creator | 69 |
| Max videos per creator | 2,218 |

Creators and videos remaining at each min_videos_per_creator threshold:

| Threshold | Creators | Videos |
|---|---|---|
| ≥ 10 | 1,802 | 209,051 |
| ≥ 20 | 1,629 | 206,519 |
| ≥ 30 | 1,439 | 201,832 |
| ≥ 50 | 1,143 | 190,260 |

**Chosen threshold: ≥ 30 videos** (`min_videos_per_creator` in `config/default.yaml`). After also removing 1,240 videos with a decreasing `play_count` (a data-collection artefact, see notebook 02), the modelling set has **200,616 videos from 1,437 creators**.

## Coverage of days_since_post in engagement_daily

| Day threshold | Videos with data | Coverage |
|---|---|---|
| ≥ 7 | 209,543 | 100% |
| ≥ 14 | 209,543 | 100% |
| ≥ 30 | 73,940 | 35.3% |

Every video has at least one record at or after day 7 and day 14.
Only 35.3% of videos have a record at day 30 (likely because they were posted too recently
relative to the data collection window ending on 2024-12-09).

## Splits

- Creators with fewer than 30 videos (`min_videos_per_creator`) are removed first.
- Then an 80/20 train/test split, **stratified by `author_id`**, with `seed: 42`
  (`config/default.yaml`), computed at runtime in
  `creator_analytics.data.train_test_split_by_creator`.

Why stratify by creator: after the midterm, the project compares per-creator models
with one global model. Stratifying guarantees that every creator has videos in both train and test, so
both approaches are evaluated on exactly the same videos.

## Personal data

The dataset consists of publicly posted TikTok videos, but it is still content created
by real people:

- `author_id` identifies a creator; transcripts may contain names or other personal
  details spoken in the videos.
- We do not try to link `author_id` to real accounts or identities.
- In reports, slides and the demo we refer to creators by anonymized labels (e.g.
  "Creator A"), not by their IDs, and we do not quote transcripts at length.

## Known limitations and biases

- **Uneven distribution across creators** (expected): many creators may have too few
  videos for a per-creator model; filtering them out biases the study towards more
  active creators.
- **transcript missing for 60%** of videos; text-based features will rely primarily
  on `gpt_summary`, which is always present.
- **enterprise_verified entirely null** — this column in creator_daily cannot be used.
- **day-30 coverage is only 35.3%**: targets that require a day-30 reading exclude ~65%
  of videos; day-7 and day-14 targets have full coverage.
- **Highly skewed metrics**: median play_count is 570, max is 44 M — log-scaling will be
  essential before modelling.
- **AI-extracted features may be noisy**: transcripts, summaries, emotion scores and
  topics were generated automatically and may contain errors.
- **No retention data**: the exact moment viewers stop watching is private and not
  available, so it is out of scope.

## Ethical risks

- Per-creator insights could be misread as definitive advice about a real person's
  content; we present them as model explanations with uncertainty, not as guarantees.
- The NC license must be respected in everything derived from the data, including any
  published model.
- Results describe patterns in this dataset only and should not be generalized to all
  TikTok creators or other platforms.

## Version and checksum

`data/raw/CHECKSUMS.txt` (committed to git, unlike the data itself) records the sha256
of the exact files this project was built and tested against. Regenerate and compare
with `python scripts/download_data.py`.
