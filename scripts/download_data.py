#!/usr/bin/env python
"""Download the TikTok engagement dataset to data/raw/ and record its checksum.

Idempotent: if the file already exists and its sha256 matches what's on record in
data/raw/CHECKSUMS.txt, nothing is re-downloaded. CHECKSUMS.txt IS committed to git
(unlike the data itself): it lets anyone check they got exactly the same data.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

from creator_analytics.config import load_config
from creator_analytics.data import materialize_raw_data

logger = logging.getLogger(__name__)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):  # 1 MB at a time
            digest.update(chunk)
    return digest.hexdigest()


def read_recorded_checksum(checksums_file: Path, filename: str) -> str | None:
    if not checksums_file.exists():
        return None
    for line in checksums_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, recorded_name = line.split(maxsplit=1)
        if recorded_name == filename:
            return digest
    return None


def write_checksum(checksums_file: Path, filename: str, digest: str) -> None:
    checksums_file.parent.mkdir(parents=True, exist_ok=True)
    checksums_file.write_text(
        "# sha256 checksums for files in data/raw/, so reproducibility can be checked\n"
        "# without the data itself being committed. Regenerate by rerunning this "
        "script.\n"
        f"{digest}  {filename}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if checksums already match"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(args.config)
    raw_path = Path(config["paths"]["raw_data"])
    checksums_file = Path(config["paths"]["checksums"])
    data_cfg = config["data"]

    recorded = read_recorded_checksum(checksums_file, raw_path.name)
    if not args.force and raw_path.exists() and recorded is not None:
        if sha256_of(raw_path) == recorded:
            logger.info("%s already up to date (sha256 matches CHECKSUMS.txt), skipping", raw_path)
            return
        logger.warning("%s exists but checksum doesn't match — re-downloading", raw_path)

    df = materialize_raw_data(
        raw_path,
        hf_repo=data_cfg["hf_repo"],
        hf_config=data_cfg.get("hf_config"),
        hf_split=data_cfg.get("hf_split", "train"),
    )
    digest = sha256_of(raw_path)
    write_checksum(checksums_file, raw_path.name, digest)
    logger.info("Wrote %s: %d rows, %d columns (sha256 %s)", raw_path, *df.shape, digest)
    logger.info("Columns: %s", ", ".join(df.columns))


if __name__ == "__main__":
    main()
