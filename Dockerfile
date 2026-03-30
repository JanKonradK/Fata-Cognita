# Multi-stage Dockerfile for Fata Cognita

# === Base stage: Python + dependencies ===
FROM python:3.14-slim AS base

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml ./
RUN uv pip install --system --no-cache -r pyproject.toml

# Copy source code
COPY src/ src/
COPY config/ config/
COPY scripts/ scripts/

# === API server stage ===
FROM base AS api

EXPOSE 8000

CMD ["python", "scripts/serve.py"]

# === Dashboard stage ===
FROM base AS dashboard

EXPOSE 8501

CMD ["streamlit", "run", "src/fata_cognita/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
