# Fata Cognita — Developer Guide

## Project Overview
Neural Actuarial Trajectory Model: VAE + Transformer decoder for longitudinal life-trajectory modeling from NLSY survey data.

## Commands
- `make dev` — install dependencies with uv
- `make test` — run all tests
- `make test-unit` — run unit tests only
- `make lint` — check linting and formatting
- `make format` — auto-fix lint and format
- `make train` — run model training
- `make serve` — start FastAPI server
- `make dashboard` — start Streamlit dashboard

## Architecture
- `src/fata_cognita/` — main package
  - `config.py` — centralized Pydantic config (loads `config/default.yaml`)
  - `device.py` — GPU/CPU device detection
  - `data/` — data pipeline (NLSY loading, feature engineering, sequences, synthetic generator)
  - `model/` — VAE architecture (encoder, transformer decoder, loss, beta schedule)
  - `training/` — training loop, metrics, callbacks
  - `archetypes/` — GMM extraction, profiling, visualization
  - `inference/` — prediction, Monte Carlo simulation, sensitivity analysis
  - `api/` — FastAPI backend
  - `dashboard/` — Streamlit frontend
- `config/default.yaml` — all hyperparameters
- `scripts/` — CLI entry points

## Conventions
- Python 3.14+, type hints everywhere, docstrings on all public functions
- Ruff for linting/formatting (line length 100)
- All config via `config/default.yaml` + env var overrides
- Tests use synthetic data by default (no NLSY data required)
