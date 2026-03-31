# Fata Cognita

**Neural Actuarial Trajectory Model** -- a deep generative model that learns latent life-trajectory archetypes from longitudinal survey data and produces probabilistic futures.

> *Fata Cognita* (Latin): "known fates" -- discovering the hidden structure of life trajectories.

## What It Does

Fata Cognita encodes individuals from the [National Longitudinal Survey of Youth](https://www.bls.gov/nls/) (NLSY79/NLSY97) into a compact latent space using a Variational Autoencoder with a Transformer decoder. From that space it:

1. **Discovers population archetypes** via Gaussian Mixture Model clustering in the latent space.
2. **Generates probabilistic life trajectories** -- career states, income, and life satisfaction from age 14 to 75.
3. **Runs Monte Carlo simulations** to produce distributional futures with percentile bands.
4. **Performs sensitivity analysis** to identify inflection points where a single life decision has outsized downstream effects.

## Architecture

```
Static Features (14-dim)
        |
   MLP Encoder
    /       \
  mu      log_var     -->  z ~ N(mu, sigma^2)   [16-dim latent]
                               |
                    Transformer Decoder (4 layers, cross-attention on z)
                     /         |           \
            Life States    Income     Satisfaction
            (9-class)    (continuous)  (continuous)
```

**Key design choices:**
- **Cyclical beta annealing** (Fu et al., 2019) prevents KL vanishing / posterior collapse.
- **Learned uncertainty weighting** (Kendall et al., 2018) balances the multi-task loss automatically.
- **Causal masking** in the Transformer decoder enforces autoregressive trajectory generation.
- **Cross-attention conditioning** injects the latent vector z into every decoder layer.

## Quick Start

### Prerequisites

- **Python 3.14+**
- **[uv](https://docs.astral.sh/uv/)** for dependency management

### Install

```bash
git clone https://github.com/<your-org>/fata-cognita.git
cd fata-cognita
make dev          # installs all dependencies via uv
```

### Run on Synthetic Data (No NLSY Files Needed)

```bash
make pipeline-synthetic
```

This runs the full pipeline end-to-end:
1. Generates 500 synthetic individuals with Markov-based life trajectories
2. Trains the VAE for 200 epochs (~4 min on CPU)
3. Extracts GMM archetypes with BIC-optimal cluster selection

### Run with Real NLSY Data

1. Download your NLSY79 and/or NLSY97 extracts from [NLS Investigator](https://www.nlsinfo.org/investigator/).
2. Place the CSV files at:
   ```
   data/raw/nlsy79.csv
   data/raw/nlsy97.csv
   ```
3. Run the pipeline:
   ```bash
   make preprocess
   make train
   make archetypes
   ```

See [Data Requirements](#data-requirements) for the expected column naming conventions.

### Serve

```bash
# Terminal 1: API server on :8000
make serve

# Terminal 2: Streamlit dashboard on :8501
make dashboard
```

Or with Docker:

```bash
make docker-up
```

## Project Structure

```
fata-cognita/
+-- config/
|   +-- default.yaml             # All hyperparameters (single source of truth)
+-- src/fata_cognita/
|   +-- config.py                # Pydantic config loader
|   +-- device.py                # GPU/CPU detection (ROCm, CUDA, CPU)
|   +-- data/
|   |   +-- synthetic.py         # Markov-based synthetic data generator
|   |   +-- nlsy_loader.py       # NLSY CSV loading and sentinel cleaning
|   |   +-- feature_engineer.py  # Life-state derivation, CPI adjustment
|   |   +-- sequence_builder.py  # Age-aligned tensor construction
|   |   +-- dataset.py           # PyTorch Dataset and DataLoader factory
|   |   +-- scaler.py            # StandardScaler wrapper for features/income
|   +-- model/
|   |   +-- encoder.py           # MLP encoder -> mu, log_var
|   |   +-- decoder.py           # Transformer decoder with 3 output heads
|   |   +-- vae.py               # Full VAE (encoder + reparameterize + decoder)
|   |   +-- loss.py              # Multi-task loss with uncertainty weighting
|   |   +-- beta_schedule.py     # Cyclical beta annealing for KL term
|   +-- training/
|   |   +-- trainer.py           # Training loop with AdamW + cosine restarts
|   |   +-- metrics.py           # Accuracy, F1, MAE, active latent units
|   |   +-- callbacks.py         # EarlyStopping, TrainingLog
|   +-- archetypes/
|   |   +-- extractor.py         # GMM fitting with BIC selection
|   |   +-- profiler.py          # Per-archetype demographic profiles
|   |   +-- visualizer.py        # t-SNE/UMAP latent space plots
|   +-- inference/
|   |   +-- predictor.py         # Single-individual trajectory prediction
|   |   +-- simulator.py         # Monte Carlo trajectory simulation
|   |   +-- sensitivity.py       # Counterfactual sensitivity analysis
|   +-- api/
|   |   +-- main.py              # FastAPI app factory with lifespan
|   |   +-- deps.py              # Dependency injection (model, GMM, scaler)
|   |   +-- routes/              # POST /predict, /simulate, /inflection-points
|   |   +-- schemas/             # Pydantic request/response models
|   +-- dashboard/
|       +-- app.py               # Streamlit 3-page app
+-- scripts/
|   +-- preprocess.py            # Data preprocessing CLI
|   +-- train.py                 # Model training CLI
|   +-- extract_archetypes.py    # Archetype extraction CLI
|   +-- serve.py                 # FastAPI server launcher
|   +-- download_nlsy.py         # NLSY download instructions
+-- tests/
|   +-- unit/                    # 65 unit tests
|   +-- integration/             # 7 integration tests
|   +-- conftest.py              # Shared fixtures
+-- Dockerfile                   # Multi-stage build (API + Dashboard)
+-- docker-compose.yml           # Orchestrates API + Dashboard services
+-- .github/workflows/ci.yml    # Lint, test, Docker build
```

## Configuration

All hyperparameters live in `config/default.yaml`. Override with a custom file:

```bash
FC_CONFIG_PATH=config/experiment.yaml make train
```

Key sections:

| Section    | Controls                                                        |
|------------|-----------------------------------------------------------------|
| `data`     | Paths, sequence length (62 steps), train/val/test split ratios  |
| `model`    | Latent dim (16), Transformer layers (4), heads (4), d_model     |
| `training` | Epochs (200), batch size (256), LR (1e-3), beta annealing       |
| `api`      | Host, port, CORS, checkpoint paths                              |
| `dashboard`| API URL for the Streamlit frontend                              |

## API Endpoints

| Method | Path                             | Description                                  |
|--------|----------------------------------|----------------------------------------------|
| GET    | `/health`                        | Health check                                 |
| POST   | `/api/v1/predict`                | Deterministic trajectory for one individual  |
| POST   | `/api/v1/simulate`               | Monte Carlo simulation with percentile bands |
| POST   | `/api/v1/inflection-points`      | Sensitivity analysis and inflection detection|
| GET    | `/api/v1/archetypes`             | List all discovered archetypes               |
| GET    | `/api/v1/archetypes/{id}/trajectory` | Canonical trajectory for an archetype    |

## Life States

The model tracks 9 mutually exclusive labor/life states at each age step:

| Code | State           | Description                     |
|------|-----------------|---------------------------------|
| 0    | EMPLOYED_FT     | Full-time employment            |
| 1    | EMPLOYED_PT     | Part-time employment            |
| 2    | SELF_EMPLOYED   | Self-employment                 |
| 3    | UNEMPLOYED      | Actively seeking work           |
| 4    | OUT_OF_LABOR    | Out of labor force              |
| 5    | STUDENT         | Enrolled in education           |
| 6    | MILITARY        | Active military service         |
| 7    | RETIRED         | Retired                         |
| 8    | DISABLED        | Unable to work due to disability|

## Data Requirements

### NLSY CSV Format

Place NLSY extracts (from [NLS Investigator](https://www.nlsinfo.org/investigator/)) at:

- `data/raw/nlsy79.csv` -- NLSY79 cohort (born 1957-1964)
- `data/raw/nlsy97.csv` -- NLSY97 cohort (born 1980-1984)

**Required variable families** (use the default NLS Investigator variable names):

| Variable Family              | Example Column Pattern     | Used For           |
|------------------------------|----------------------------|--------------------|
| Case ID                      | `CASEID`                   | Individual linking  |
| Sex                          | `SEX`                      | Static feature      |
| Race/Ethnicity               | `RACE`, `ETHNICITY`        | Static features     |
| Date of Birth                | `DOB_YR`, `DOB_MO`         | Age computation     |
| Employment Status (per round)| `EMP_STATUS_*`             | Life state          |
| Hourly Rate / Income         | `HOURLY_RATE_*`, `INCOME_*`| Income trajectory   |
| Job Satisfaction              | `JOB_SAT_*`               | Satisfaction        |
| Education                    | `HIGHEST_GRADE_*`          | Static feature      |
| Region                       | `REGION_*`                 | Static features     |
| AFQT Score                   | `AFQT_*`                   | Static feature      |

The loader automatically handles NLSY sentinel values (-1 through -5) and maps wide-format round columns to long-format year-indexed rows.

### Synthetic Data

No external data needed. The synthetic generator creates Markov-chain-based life trajectories with age-dependent transition probabilities, log-normal income distributions, and beta-distributed satisfaction scores.

```bash
make preprocess-synthetic
```

## GPU Support

The model automatically detects available hardware:

| Priority | Backend | Hardware              | Detection              |
|----------|---------|-----------------------|------------------------|
| 1        | ROCm    | AMD GPUs (RDNA/CDNA)  | `torch.cuda.is_available()` with ROCm build |
| 2        | CUDA    | NVIDIA GPUs           | `torch.cuda.is_available()`               |
| 3        | CPU     | Any                   | Fallback                                   |

Override with the `FC_DEVICE` environment variable:

```bash
FC_DEVICE=cpu make train    # Force CPU even if GPU available
```

## Development

```bash
make dev              # Install dependencies
make test             # Run all 72 tests
make test-unit        # Unit tests only
make test-cov         # Tests with coverage report
make lint             # Check code style (ruff)
make format           # Auto-fix formatting
make typecheck        # Static type checking (pyright)
```

### Testing Philosophy

- All tests run against **synthetic data by default** -- no NLSY files required.
- Integration tests exercise the full API with a real model forward pass.
- The `tiny_config` fixture uses minimal dimensions for fast test execution.

## License

MIT License. See [LICENSE](LICENSE) for details.
