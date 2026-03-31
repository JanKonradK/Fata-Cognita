"""FastAPI backend for serving predictions and archetypes."""

from fata_cognita.api.deps import AppState, get_gmm, get_model, get_scaler, load_artifacts
from fata_cognita.api.main import create_app

__all__ = [
    "AppState",
    "create_app",
    "get_gmm",
    "get_model",
    "get_scaler",
    "load_artifacts",
]
