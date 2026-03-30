"""Tests for trajectory prediction."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from fata_cognita.config import Config
from fata_cognita.data.scaler import FeatureScaler
from fata_cognita.inference.predictor import predict_trajectory
from fata_cognita.model.vae import TrajectoryVAE


def test_prediction_output(tiny_config: Config):
    """Prediction returns trajectory with correct structure."""
    n_features = tiny_config.data.num_static_features
    model = TrajectoryVAE(n_features, tiny_config)
    device = torch.device("cpu")

    # Dummy scaler (identity transform)
    scaler = FeatureScaler(fit_static=False, fit_income=False)

    # Dummy GMM
    gmm = GaussianMixture(n_components=3, random_state=42)
    dummy_z = np.random.randn(30, tiny_config.model.latent_dim)
    gmm.fit(dummy_z)

    features = {name: 0.0 for name in tiny_config.data.static_features}
    result = predict_trajectory(
        static_features=features,
        feature_names=tiny_config.data.static_features,
        model=model,
        gmm=gmm,
        scaler=scaler,
        device=device,
    )

    assert len(result.trajectory) == tiny_config.data.max_seq_len
    assert result.trajectory[0].age == 14
    assert all(t.life_state in [s.name for s in __import__('fata_cognita.data.synthetic', fromlist=['LifeState']).LifeState] for t in result.trajectory)
    assert len(result.latent_vector) == tiny_config.model.latent_dim
    assert isinstance(result.archetype_id, int)
