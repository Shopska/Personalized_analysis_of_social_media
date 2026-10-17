#!/usr/bin/env python
"""Compare the global (baseline) model with per-creator models. Not implemented yet.

Planned: train Random Forest and XGBoost in two modes — one global model on all
creators and one model per creator — evaluate both on the same test videos with RMSE,
MAE and R², and write the comparison to results/metrics.json and
results/experiments.md.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.error("Not implemented yet: the model comparison comes after the baseline model.")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
