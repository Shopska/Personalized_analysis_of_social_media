"""Single entry point for this project: ``python -m creator_analytics <command>``.

Commands:

- ``data-summary`` — load the downloaded dataset and print how many videos and creators
  there are, before and after filtering by ``min_videos_per_creator``. Works now.
- ``train`` / ``evaluate`` — not implemented yet; they will train and evaluate the
  baseline (global) and per-creator models.

One CLI with subcommands (rather than separate scripts) means there's one place that
knows how to load config, set the seed, and set up logging.
"""

from __future__ import annotations

import argparse
import logging
import sys

from creator_analytics.config import load_config
from creator_analytics.data import filter_creators, load_raw_data
from creator_analytics.utils import set_seed, setup_logging

logger = logging.getLogger(__name__)


def cmd_data_summary(config: dict) -> int:
    """Print basic dataset stats: videos, creators, and what survives the filter."""
    data_cfg = config["data"]
    author_column = data_cfg["author_column"]
    min_videos = data_cfg["min_videos_per_creator"]

    df = load_raw_data(config["paths"]["raw_data"])
    filtered = filter_creators(df, author_column, min_videos)

    logger.info("Raw dataset: %d videos, %d creators", len(df), df[author_column].nunique())
    logger.info(
        "With >= %d videos per creator: %d videos, %d creators",
        min_videos,
        len(filtered),
        filtered[author_column].nunique(),
    )
    return 0


def cmd_not_implemented(name: str) -> int:
    logger.error("`%s` is not implemented yet — it comes with the baseline model.", name)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m creator_analytics",
        description="Personalized explainable-ML analysis of TikTok creator performance.",
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to a YAML config file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override a config value, e.g. --override data.min_videos_per_creator=50",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("data-summary", help="Show dataset size before and after filtering")
    subparsers.add_parser("train", help="Train models (not implemented yet)")
    subparsers.add_parser("evaluate", help="Evaluate models (not implemented yet)")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config, args.override)
    setup_logging(config["logging"]["level"])
    set_seed(config["seed"])

    if args.command == "data-summary":
        sys.exit(cmd_data_summary(config))
    sys.exit(cmd_not_implemented(args.command))
