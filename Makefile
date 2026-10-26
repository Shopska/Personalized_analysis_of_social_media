# Deliberately minimal: two targets — "get me set up" and "prove it works".
# Everything else (training, evaluating, the demo, linting, tests) is run directly;
# see README.md's Quick start / Development sections for the exact commands.

VENV := .venv

ifeq ($(OS),Windows_NT)
    PYTHON ?= python
    BIN    := $(VENV)/Scripts
else
    PYTHON ?= python3
    BIN    := $(VENV)/bin
endif

.PHONY: setup reproduce test lint

setup: ## Create .venv, install runtime+dev deps and this package, install pre-commit hooks
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -r requirements.txt -r requirements-dev.txt
	$(BIN)/python -m pip install -e .
	$(BIN)/pre-commit install

reproduce: ## data -> global baseline models -> evaluation
	$(BIN)/python scripts/download_data.py
	$(BIN)/python -m creator_analytics train
	$(BIN)/python -m creator_analytics evaluate

test: ## Run pytest
	$(BIN)/pytest tests/ -v

lint: ## Run pre-commit on all files
	$(BIN)/pre-commit run --all-files