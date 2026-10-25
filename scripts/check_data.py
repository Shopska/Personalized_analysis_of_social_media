#!/usr/bin/env python
"""Sanity-check the downloaded data and the repo in one go.

    python scripts/check_data.py

Checks, independently of whoever downloaded/processed the data:
1. each table in data/raw/ has exactly the row/column count published on Hugging Face;
2. each file's sha256 matches data/raw/CHECKSUMS.txt;
3. no data files are staged/tracked by git (only CHECKSUMS.txt should be);
4. the test suite passes;
5. prints creator stats, so you can compare them with data/README.md by eye.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
# Source: https://datasets-server.huggingface.co/info?dataset=lingbow/tiktok-video-engagement-200k
EXPECTED_SHAPES = {
    "videos": (209_543, 33),
    "creator_daily": (278_433, 7),
    "engagement_daily": (6_068_955, 10),
}

failures: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(("  OK   " if ok else "  FAIL ") + msg)
    if not ok:
        failures.append(msg)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


print("1. Розміри таблиць vs Hugging Face")
tables = {}
for name, expected in EXPECTED_SHAPES.items():
    path = RAW / f"{name}.parquet"
    if not path.exists():
        check(False, f"{path} не існує — запусти scripts/download_data.py")
        continue
    tables[name] = df = pd.read_parquet(path)
    check(df.shape == expected, f"{name}: {df.shape}, очікувалось {expected}")

print("2. Чексуми")
checksums_file = RAW / "CHECKSUMS.txt"
recorded = {}
if checksums_file.exists():
    for line in checksums_file.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            digest, filename = line.split(maxsplit=1)
            recorded[filename] = digest
for name in EXPECTED_SHAPES:
    path = RAW / f"{name}.parquet"
    if path.exists():
        check(recorded.get(path.name) == sha256_of(path), f"{path.name} збігається з CHECKSUMS.txt")

print("3. Дані не потрапляють у git")
tracked = subprocess.run(
    ["git", "ls-files", "--cached", "data/"], capture_output=True, text=True
).stdout.split()
leaked = [f for f in tracked if f.endswith(".parquet")]
check(not leaked, "жодного .parquet у git" + (f" (знайдено: {leaked})" if leaked else ""))

print("4. Тести")
result = subprocess.run([sys.executable, "-m", "pytest", "-q"], capture_output=True, text=True)
summary = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else result.stderr
check(result.returncode == 0, f"pytest: {summary}")

if "videos" in tables:
    print("5. Креатори (звір з data/README.md)")
    counts = tables["videos"]["author_id"].value_counts()
    print(
        f"  креаторів: {len(counts)}, відео на креатора "
        f"min/median/max: {counts.min()} / {counts.median():.0f} / {counts.max()}"
    )
    for threshold in (10, 20, 30, 50):
        kept = counts[counts >= threshold]
        print(f"  >= {threshold} відео: {len(kept)} креаторів, {kept.sum()} відео")

print()
if failures:
    print(f"Є проблеми ({len(failures)}):")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("Усе ок")
