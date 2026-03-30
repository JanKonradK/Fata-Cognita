.PHONY: dev test lint format train archetypes serve dashboard docker-up clean

dev:
	uv sync --dev

test:
	uv run pytest tests/

test-unit:
	uv run pytest tests/unit/

test-integration:
	uv run pytest tests/integration/

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

format:
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/

preprocess:
	uv run python scripts/preprocess.py

train:
	uv run python scripts/train.py

archetypes:
	uv run python scripts/extract_archetypes.py

serve:
	uv run python scripts/serve.py

dashboard:
	uv run streamlit run src/fata_cognita/dashboard/app.py

docker-up:
	docker-compose up --build

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info/
