"""Tests for sensitivity analysis."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from fata_cognita.config import Config
from fata_cognita.data.scaler import FeatureScaler
from fata_cognita.inference.sensitivity import run_sensitivity_analysis
from fata_cognita.model.vae import TrajectoryVAE


def test_sensitivity_produces_deltas(tiny_config: Config):
    """Sensitivity analysis produces non-empty deltas."""
    n_features = tiny_config.data.num_static_features
    model = TrajectoryVAE(n_features, tiny_config)
    device = torch.device("cpu")

    scaler = FeatureScaler(fit_static=False, fit_income=False)

    gmm = GaussianMixture(n_components=3, random_state=42)
    dummy_z = np.random.randn(30, tiny_config.model.latent_dim)
    gmm.fit(dummy_z)

    features = {name: 0.0 for name in tiny_config.data.static_features}

    result = run_sensitivity_analysis(
        static_features=features,
        perturb_variable="birth_year",
        perturb_value=2.0,
        feature_names=tiny_config.data.static_features,
        model=model,
        gmm=gmm,
        scaler=scaler,
        device=device,
        n_simulations=50,
    )

    assert len(result.deltas_by_age) == tiny_config.data.max_seq_len
    assert result.perturb_variable == "birth_year"
    assert result.perturb_value == 2.0
    assert isinstance(result.overall_effect_size, float)
    assert isinstance(result.base_archetype, int)
    assert isinstance(result.perturbed_archetype, int)
