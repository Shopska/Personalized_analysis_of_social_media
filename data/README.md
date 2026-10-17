# data/ — data card

> Status: partially filled before EDA. Sections marked **TODO (after EDA)** will be
> completed once the dataset is downloaded and explored in `notebooks/01-eda.ipynb`.

## Name and source

**TikTok Video Engagement 200k** (`lingbow/tiktok-video-engagement-200k`), hosted on
Hugging Face Datasets.

- Dataset page: <https://huggingface.co/datasets/lingbow/tiktok-video-engagement-200k>
- Approximately 210,000 TikTok videos, each linked to its creator via `author_id`,
  which is what makes per-creator modeling possible.
- Besides the main video-level table, the dataset includes two auxiliary tables:
  `creator_daily` and `engagement_daily`.
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

**Citation**: TODO — check the dataset page for a preferred citation; until then, cite
the dataset by its Hugging Face id and URL above.

## Size and format

- ~210,000 rows (one row per video) in the main table
- Stored locally as one Parquet file: `data/raw/videos.parquet`. Parquet rather than CSV
  because of the long free-text columns (`transcript`, `gpt_summary`) and the row count.
- File size on disk: **TODO (after download)**
- No train/test files on disk — the split is computed at runtime from a fixed seed
  (see "Splits" below).

## Schema

Columns we plan to use, based on the dataset description. Types, exact column names
and the full column list will be confirmed after download (`download_data.py` prints
every column name).

| Column | Type | Description | Used as |
|---|---|---|---|
| `author_id` | TODO | Creator identifier | Grouping key for per-creator models and for the split |
| `transcript` | text | Speech transcript of the video | Source for text features |
| `gpt_summary` | text | AI-generated summary of the video | Source for text features |
| emotion scores | numeric | Emotional tone of the video (exact columns TODO) | Features |
| `topic` | categorical | Topic category of the video | Feature |
| `music_selected_from` | categorical | Where the music was selected from | Proxy for the editing template type |
| performance metric(s) | numeric | Views / likes / comments / shares (exact columns TODO) | Target variable — **to be chosen after EDA** |

**TODO (after EDA)**: full column list with types and units; description of
`creator_daily` and `engagement_daily` and whether we use them.

## `data/raw/` vs. `data/interim/` vs. `data/processed/`

- `data/raw/videos.parquet` — the dataset exactly as downloaded from Hugging Face.
- `data/interim/` — the dataset after cleaning and after filtering out creators with
  too few videos (`min_videos_per_creator` in `config/default.yaml`).
- `data/processed/` — model-ready features produced by `src/creator_analytics/features.py`.

None of these files are committed to git; only this README and `data/raw/CHECKSUMS.txt` are.

## How to get it

```bash
python scripts/download_data.py
```

This downloads the dataset from Hugging Face into `data/raw/videos.parquet` and writes
its sha256 to `data/raw/CHECKSUMS.txt`. Running it again does nothing if the checksum
already matches; use `--force` to re-download. The dataset id, subset and split are
set in the `data` section of `config/default.yaml`.

## Splits

- Creators with fewer than `min_videos_per_creator` videos are removed first
  (default 30, final value to be chosen after EDA).
- Then an 80/20 train/test split, **stratified by `author_id`**, with `seed: 42`
  (`config/default.yaml`), computed at runtime in
  `creator_analytics.data.train_test_split_by_creator`.

Why stratify by creator: the project compares per-creator models with one global
model. Stratifying guarantees that every creator has videos in both train and test, so
both approaches are evaluated on exactly the same videos.

## Distribution of videos across creators

**TODO (after EDA)**: number of unique creators, videos per creator (min / median /
max, histogram), and how many creators remain at different values of
`min_videos_per_creator`.

## Personal data

The dataset consists of publicly posted TikTok videos, but it is still content created
by real people:

- `author_id` identifies a creator; transcripts may contain names or other personal
  details spoken in the videos.
- We do not try to link `author_id` to real accounts or identities.
- In reports, slides and the demo we refer to creators by anonymized labels (e.g.
  "Creator A"), not by their IDs, and we do not quote transcripts at length.

## Scraping

Not applicable to us — we did not collect this data; we use the published dataset.
**TODO**: describe how the dataset authors collected it, based on the dataset page.

## Known limitations and biases

- **Uneven distribution across creators** (expected): many creators may have too few
  videos for a per-creator model; filtering them out biases the study towards more
  active creators.
- **AI-extracted features may be noisy**: transcripts, summaries, emotion scores and
  topics were generated automatically and may contain errors.
- **Proxy features**: `music_selected_from` is only an approximation of the editing
  template; we do not analyze the video itself.
- **No retention data**: the exact moment viewers stop watching is private and not
  available, so it is out of scope.
- **TODO (after EDA)**: missing values, outliers in engagement metrics, the time period
  covered, and when engagement was measured relative to posting.

## Ethical risks

- Per-creator insights could be misread as definitive advice about a real person's
  content; we present them as model explanations with uncertainty, not as guarantees.
- The NC license must be respected in everything derived from the data, including any
  published model.
- Results describe patterns in this dataset only and should not be generalized to all
  TikTok creators or other platforms.

## Version and checksum

`data/raw/CHECKSUMS.txt` (committed to git, unlike the data itself) records the sha256
of the exact file this project was built and tested against. Regenerate and compare
with `python scripts/download_data.py`.