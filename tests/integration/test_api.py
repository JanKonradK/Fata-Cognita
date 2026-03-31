"""Integration tests for the FastAPI API."""

from __future__ import annotations

import numpy as np
import torch
from fastapi.testclient import TestClient
from sklearn.mixture import GaussianMixture

from fata_cognita.api.deps import AppState
from fata_cognita.api.main import create_app
from fata_cognita.config import Config, DataConfig, ModelConfig, SyntheticConfig
from fata_cognita.data.scaler import FeatureScaler
from fata_cognita.model.vae import TrajectoryVAE


def _make_test_state() -> AppState:
    """Create a minimal AppState with an untrained model for testing."""
    config = Config(
        data=DataConfig(
            max_seq_len=10,
            synthetic=SyntheticConfig(n_individuals=32),
        ),
        model=ModelConfig(
            latent_dim=4,
            encoder_hidden_dims=[16, 8],
            d_model=16,
            nhead=2,
            num_decoder_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        ),
    )

    state = AppState()
    state.config = config
    state.device = torch.device("cpu")
    state.feature_names = config.data.static_features

    # Create model
    n_features = config.data.num_static_features
    model = TrajectoryVAE(n_features, config)
    model.eval()
    state.model = model

    # Create dummy GMM
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(np.random.randn(30, config.model.latent_dim))
    state.gmm = gmm

    # Identity scaler
    state.scaler = FeatureScaler(fit_static=False, fit_income=False)

    # Dummy archetype profiles
    state.archetype_profiles = [
        {
            "archetype_id": 0,
            "prevalence": 0.5,
            "member_count": 15,
            "feature_means": {"sex": 0.5},
            "canonical_trajectory": {
                "life_states": [5] * 10,
                "income": [7.0] * 10,
                "satisfaction": [0.5] * 10,
            },
        },
        {
            "archetype_id": 1,
            "prevalence": 0.3,
            "member_count": 10,
            "feature_means": {"sex": 0.6},
            "canonical_trajectory": {
                "life_states": [0] * 10,
                "income": [10.0] * 10,
                "satisfaction": [0.7] * 10,
            },
        },
    ]

    return state


def _make_features(config: Config) -> dict[str, float]:
    return {name: 0.0 for name in config.data.static_features}


def _make_client() -> tuple[TestClient, AppState]:
    """Create a TestClient with pre-loaded state."""
    state = _make_test_state()
    app = create_app(state=state)
    # Ensure lifespan runs by using context manager
    client = TestClient(app, raise_server_exceptions=True)
    return client, state


class TestHealthCheck:
    def test_health(self):
        state = _make_test_state()
        app = create_app(state=state)
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"


class TestPredictEndpoint:
    def test_predict_200(self):
        state = _make_test_state()
        app = create_app(state=state)
        with TestClient(app) as client:
            features = _make_features(state.config)
            resp = client.post(
                "/api/v1/predict",
                json={
                    "static_features": features,
                    "deterministic": True,
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["trajectory"]) == 10
            assert "archetype_id" in data
            assert len(data["latent_vector"]) == 4

    def test_predict_422_missing_body(self):
        state = _make_test_state()
        app = create_app(state=state)
        with TestClient(app) as client:
            resp = client.post("/api/v1/predict", json={})
            assert resp.status_code == 422


class TestSimulateEndpoint:
    def test_simulate_200(self):
        state = _make_test_state()
        app = create_app(state=state)
        with TestClient(app) as client:
            features = _make_features(state.config)
            resp = client.post(
                "/api/v1/simulate",
                json={
                    "static_features": features,
                    "n_simulations": 100,
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "percentile_bands" in data
            assert len(data["percentile_bands"]["age"]) == 10


class TestArchetypesEndpoint:
    def test_list_archetypes(self):
        state = _make_test_state()
        app = create_app(state=state)
        with TestClient(app) as client:
            resp = client.get("/api/v1/archetypes")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["archetypes"]) == 2
            assert data["k_selected"] == 2

    def test_archetype_trajectory(self):
        state = _make_test_state()
        app = create_app(state=state)
        with TestClient(app) as client:
            resp = client.get("/api/v1/archetypes/0/trajectory")
            assert resp.status_code == 200
            data = resp.json()
            assert data["archetype_id"] == 0
            assert len(data["canonical_trajectory"]) == 10

    def test_archetype_not_found(self):
        state = _make_test_state()
        app = create_app(state=state)
        with TestClient(app) as client:
            resp = client.get("/api/v1/archetypes/999/trajectory")
            assert resp.status_code == 404
