"""Tests for Monte Carlo trajectory simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from fata_cognita.data.scaler import FeatureScaler
from fata_cognita.inference.simulator import simulate_trajectories
from fata_cognita.model.vae import TrajectoryVAE

if TYPE_CHECKING:
    from fata_cognita.config import Config


def test_simulation_shapes(tiny_config: Config):
    """Simulation returns correct shapes and percentile ordering."""
    n_features = tiny_config.data.num_static_features
    model = TrajectoryVAE(n_features, tiny_config)
    device = torch.device("cpu")

    scaler = FeatureScaler(fit_static=False, fit_income=False)

    features = {name: 0.0 for name in tiny_config.data.static_features}
    result = simulate_trajectories(
        static_features=features,
        feature_names=tiny_config.data.static_features,
        model=model,
        scaler=scaler,
        device=device,
        n_simulations=100,
    )

    seq_len = tiny_config.data.max_seq_len
    assert len(result.ages) == seq_len
    assert len(result.income_percentiles["p10"]) == seq_len
    assert len(result.income_percentiles["p90"]) == seq_len
    assert len(result.state_distribution) == seq_len
    assert result.n_simulations == 100


def test_percentile_ordering(tiny_config: Config):
    """P10 <= P25 <= P50 <= P75 <= P90 at every age."""
    n_features = tiny_config.data.num_static_features
    model = TrajectoryVAE(n_features, tiny_config)
    device = torch.device("cpu")

    scaler = FeatureScaler(fit_static=False, fit_income=False)

    features = {name: 0.0 for name in tiny_config.data.static_features}
    result = simulate_trajectories(
        static_features=features,
        feature_names=tiny_config.data.static_features,
        model=model,
        scaler=scaler,
        device=device,
        n_simulations=200,
    )

    for t in range(len(result.ages)):
        p10 = result.income_percentiles["p10"][t]
        p25 = result.income_percentiles["p25"][t]
        p50 = result.income_percentiles["p50"][t]
        p75 = result.income_percentiles["p75"][t]
        p90 = result.income_percentiles["p90"][t]
        assert p10 <= p25 <= p50 <= p75 <= p90, f"Ordering violated at age {result.ages[t]}"
