"""Dependency injection for FastAPI: model, GMM, and scaler singletons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from fastapi import HTTPException

from fata_cognita.archetypes.extractor import load_gmm
from fata_cognita.config import Config, load_config
from fata_cognita.data.scaler import FeatureScaler
from fata_cognita.device import get_device
from fata_cognita.model.vae import TrajectoryVAE

if TYPE_CHECKING:
    from fastapi import Request
    from sklearn.mixture import GaussianMixture


class AppState:
    """Holds loaded model artifacts for the API."""

    def __init__(self) -> None:
        self.model: TrajectoryVAE | None = None
        self.gmm: GaussianMixture | None = None
        self.scaler: FeatureScaler | None = None
        self.config: Config | None = None
        self.device: torch.device = torch.device("cpu")
        self.feature_names: list[str] = []
        self.archetype_profiles: list[dict] | None = None


_state = AppState()


def load_artifacts(config: Config | None = None) -> AppState:
    """Load all model artifacts into the app state.

    Args:
        config: Optional config override.

    Returns:
        Populated AppState.
    """
    if config is None:
        config = load_config()

    _state.config = config
    _state.device = get_device()
    _state.feature_names = config.data.static_features

    # Load model
    num_features = config.data.num_static_features
    model = TrajectoryVAE(num_features, config)

    checkpoint = torch.load(
        config.api.model_checkpoint,
        map_location=_state.device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(_state.device)
    model.eval()
    _state.model = model

    # Load GMM
    _state.gmm = load_gmm(config.api.gmm_path)

    # Load scaler
    _state.scaler = FeatureScaler.load(config.api.scaler_path)

    # Load archetype profiles
    profiles_path = Path(config.api.archetype_profiles_path)
    if profiles_path.exists():
        with open(profiles_path) as f:
            _state.archetype_profiles = json.load(f)

    return _state


def get_state() -> AppState:
    """Get the current app state."""
    return _state


def get_model(request: Request) -> TrajectoryVAE:
    """FastAPI dependency: get the VAE model."""
    state: AppState = request.app.state.app_state
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return state.model


def get_gmm(request: Request) -> GaussianMixture:
    """FastAPI dependency: get the fitted GMM."""
    state: AppState = request.app.state.app_state
    if state.gmm is None:
        raise HTTPException(status_code=503, detail="GMM not loaded")
    return state.gmm


def get_scaler(request: Request) -> FeatureScaler:
    """FastAPI dependency: get the feature scaler."""
    state: AppState = request.app.state.app_state
    if state.scaler is None:
        raise HTTPException(status_code=503, detail="Scaler not loaded")
    return state.scaler
