.PHONY: dev test test-unit test-integration test-cov lint format typecheck \
       preprocess train archetypes serve dashboard docker-up docker-down clean help

# ──────────────────────────────────────────────────────────────── Setup
dev:  ## Install all dependencies (including dev extras)
	uv sync --dev

# ──────────────────────────────────────────────────────────────── Quality
test:  ## Run full test suite
	uv run pytest tests/

test-unit:  ## Run unit tests only
	uv run pytest tests/unit/

test-integration:  ## Run integration tests only
	uv run pytest tests/integration/

test-cov:  ## Run tests with coverage report
	uv run pytest tests/ --cov=src/fata_cognita --cov-report=term-missing --cov-report=html

lint:  ## Check linting and formatting (no changes)
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

format:  ## Auto-fix lint issues and reformat
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/

typecheck:  ## Run static type checking
	uv run pyright src/

# ──────────────────────────────────────────────────────────────── Pipeline
preprocess:  ## Run data preprocessing (add --synthetic for synthetic data)
	uv run python scripts/preprocess.py

preprocess-synthetic:  ## Preprocess with synthetic data (no NLSY files needed)
	uv run python scripts/preprocess.py --synthetic

train:  ## Train the VAE model
	uv run python scripts/train.py

train-synthetic:  ## Train on synthetic data (no NLSY files needed)
	uv run python scripts/train.py --synthetic

archetypes:  ## Extract archetypes from trained model
	uv run python scripts/extract_archetypes.py

archetypes-synthetic:  ## Extract archetypes using synthetic data
	uv run python scripts/extract_archetypes.py --synthetic

pipeline-synthetic:  ## Run full pipeline end-to-end on synthetic data
	$(MAKE) preprocess-synthetic
	$(MAKE) train-synthetic
	$(MAKE) archetypes-synthetic

# ──────────────────────────────────────────────────────────────── Serving
serve:  ## Start FastAPI server on port 8000
	uv run python scripts/serve.py

dashboard:  ## Start Streamlit dashboard on port 8501
	uv run streamlit run src/fata_cognita/dashboard/app.py

# ──────────────────────────────────────────────────────────────── Docker
docker-up:  ## Build and start all services with Docker Compose
	docker-compose up --build

docker-down:  ## Stop all Docker Compose services
	docker-compose down

# ──────────────────────────────────────────────────────────────── Cleanup
clean:  ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info/ htmlcov/ .coverage .ruff_cache/ .mypy_cache/

# ──────────────────────────────────────────────────────────────── Help
help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
