# Contributing to Fata Cognita

Thank you for your interest in contributing. This document covers the conventions and workflow for the project.

## Development Setup

```bash
# Clone and install
git clone https://github.com/<your-org>/fata-cognita.git
cd fata-cognita
make dev
```

This installs all runtime and dev dependencies via [uv](https://docs.astral.sh/uv/).

## Code Standards

### Python

- **Python 3.14+** -- use modern syntax freely (union types `X | Y`, `Self`, etc.)
- **Type hints on all function signatures** -- no untyped public functions.
- **Docstrings on all public functions and classes** -- use Google-style format:
  ```python
  def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
      """Encode input features to latent distribution parameters.

      Args:
          x: Static features of shape (batch_size, num_features).

      Returns:
          Tuple of (mu, log_var), each of shape (batch_size, latent_dim).
      """
  ```
- **Line length: 100 characters** (enforced by Ruff).
- Private helpers use `_underscore_prefix` and do not require docstrings (but benefit from inline comments if non-obvious).

### Formatting and Linting

We use [Ruff](https://docs.astral.sh/ruff/) for both linting and formatting:

```bash
make lint     # Check without modifying
make format   # Auto-fix
```

Ruff rules enabled: `E`, `F`, `W`, `I` (isort), `UP` (pyupgrade), `B` (bugbear), `SIM` (simplify), `TCH` (type-checking imports).

### Configuration

All hyperparameters and paths live in `config/default.yaml`. Do not scatter magic numbers through source code. If you add a new tunable parameter:

1. Add it to `config/default.yaml` with a descriptive comment.
2. Add a corresponding field in the appropriate config dataclass in `src/fata_cognita/config.py`.
3. Reference it via the `Config` object -- never hardcode.

## Testing

```bash
make test          # Run all tests
make test-unit     # Unit tests only
make test-cov      # Tests with coverage
```

### Testing Conventions

- Tests live in `tests/unit/` and `tests/integration/`.
- All tests run on **synthetic data only** -- no NLSY files required.
- Use the `tiny_config` and `synthetic_batch` fixtures from `tests/conftest.py` for small, fast model tests.
- Name test files `test_<module>.py` and test functions `test_<behavior>`.
- New modules should have corresponding test files covering happy paths and edge cases.

### What to Test

- **Model components**: tensor shapes, gradient flow, deterministic vs stochastic modes.
- **Data pipeline**: correct transformations, sentinel handling, edge cases.
- **API endpoints**: status codes, response schemas, error handling.
- **Do not test**: trivial getters, framework internals, or third-party library behavior.

## Project Layout

```
src/fata_cognita/     # Main package
  config.py           # Pydantic config (frozen dataclasses)
  device.py           # Hardware detection
  data/               # Data loading and preprocessing
  model/              # VAE architecture
  training/           # Training loop and metrics
  archetypes/         # GMM clustering and profiling
  inference/          # Prediction, simulation, sensitivity
  api/                # FastAPI backend
  dashboard/          # Streamlit frontend
scripts/              # CLI entry points
tests/                # Unit and integration tests
config/               # YAML configuration files
```

## Git Workflow

1. Create a feature branch from `main`: `git checkout -b feat/my-feature`
2. Make small, focused commits with descriptive messages.
3. Ensure `make lint` and `make test` pass before pushing.
4. Open a pull request against `main`.
5. CI must pass (lint + tests + Docker build) before merge.

### Commit Message Style

Use imperative mood, lowercase, no period:

```
add Monte Carlo confidence intervals to simulator
fix sentinel value handling for NLSY97 round mapping
update decoder cross-attention to use multi-head
```

Prefix with scope when helpful: `data: ...`, `model: ...`, `api: ...`, `docs: ...`

## Architecture Decisions

### Why VAE + Transformer (not RNN/LSTM)?

Transformers handle variable-length sequences with parallel training and learn long-range dependencies better than RNNs. The VAE latent space enables meaningful clustering and generative sampling.

### Why Cyclical Beta Annealing?

Standard linear annealing often causes KL vanishing. Cyclical annealing (Fu et al., 2019) periodically resets the KL weight, giving the encoder multiple chances to learn informative representations.

### Why Learned Uncertainty Weighting?

The three loss terms (classification, income regression, satisfaction regression) operate on different scales. Learned log-variance parameters (Kendall et al., 2018) let the model automatically balance them.

## Questions?

Open an issue on GitHub or start a discussion.
