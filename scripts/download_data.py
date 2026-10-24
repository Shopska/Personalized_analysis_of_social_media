#!/usr/bin/env python
"""Download the TikTok engagement dataset tables to data/raw/ and record their checksums.

Downloads three HF subsets: videos, creator_daily, engagement_daily.
Each lands at data/raw/<table>.parquet; sha256 entries are written to
data/raw/CHECKSUMS.txt (one line per file).

Idempotent: a table is skipped when its file already exists and the recorded
sha256 matches. Use --force to re-download unconditionally.
CHECKSUMS.txt IS committed to git (unlike the data itself).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

from creator_analytics.config import load_config
from creator_analytics.data import materialize_raw_table, raw_table_path

logger = logging.getLogger(__name__)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_recorded_checksum(checksums_file: Path, filename: str) -> str | None:
    if not checksums_file.exists():
        return None
    for line in checksums_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[1] == filename:
            return parts[0]
    return None


def update_checksum_entry(checksums_file: Path, filename: str, digest: str) -> None:
    """Add or replace the checksum line for ``filename`` in CHECKSUMS.txt."""
    checksums_file.parent.mkdir(parents=True, exist_ok=True)
    entries: dict[str, str] = {}
    if checksums_file.exists():
        for line in checksums_file.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split(maxsplit=1)
            if len(parts) == 2:
                entries[parts[1]] = parts[0]
    entries[filename] = digest
    header = (
        "# sha256 checksums for files in data/raw/, so reproducibility can be checked\n"
        "# without the data itself being committed. Regenerate by rerunning this script.\n"
    )
    body = "".join(f"{d}  {n}\n" for n, d in sorted(entries.items()))
    checksums_file.write_text(header + body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if checksums already match"
    )
    parser.add_argument("--table", help="Download only this table (default: all tables in config)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(args.config)
    raw_dir = Path(config["paths"]["raw_dir"])
    checksums_file = Path(config["paths"]["checksums"])
    data_cfg = config["data"]

    tables: list[str] = data_cfg["tables"]
    if args.table:
        if args.table not in tables:
            logger.error("Unknown table %r. Available: %s", args.table, ", ".join(tables))
            raise SystemExit(1)
        tables = [args.table]

    for table in tables:
        raw_path = raw_table_path(raw_dir, table)
        recorded = read_recorded_checksum(checksums_file, raw_path.name)
        if not args.force and raw_path.exists() and recorded is not None:
            if sha256_of(raw_path) == recorded:
                logger.info(
                    "%s already up to date (sha256 matches CHECKSUMS.txt), skipping", raw_path
                )
                continue
            logger.warning("%s exists but checksum doesn't match — re-downloading", raw_path)

        logger.info("Downloading %s from %s …", table, data_cfg["hf_repo"])
        materialize_raw_table(
            raw_dir,
            table,
            hf_repo=data_cfg["hf_repo"],
            hf_split=data_cfg.get("hf_split", "train"),
        )
        digest = sha256_of(raw_path)
        update_checksum_entry(checksums_file, raw_path.name, digest)
        logger.info("Wrote %s (sha256 %s)", raw_path, digest)


if __name__ == "__main__":
    main()
