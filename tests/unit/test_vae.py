"""Tests for the full VAE model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from fata_cognita.model.vae import TrajectoryVAE

if TYPE_CHECKING:
    from fata_cognita.config import Config


def test_vae_forward_keys(tiny_config: Config):
    """VAE forward returns all expected keys."""
    n_features = tiny_config.data.num_static_features
    model = TrajectoryVAE(n_features, tiny_config)

    x = torch.randn(8, n_features)
    out = model(x)

    expected_keys = {"life_state_logits", "income", "satisfaction", "mu", "log_var", "z"}
    assert set(out.keys()) == expected_keys


def test_vae_output_shapes(tiny_config: Config):
    """VAE outputs have correct shapes."""
    n_features = tiny_config.data.num_static_features
    model = TrajectoryVAE(n_features, tiny_config)

    x = torch.randn(8, n_features)
    out = model(x)

    seq_len = tiny_config.data.max_seq_len
    latent_dim = tiny_config.model.latent_dim

    assert out["life_state_logits"].shape == (8, seq_len, 9)
    assert out["income"].shape == (8, seq_len)
    assert out["satisfaction"].shape == (8, seq_len)
    assert out["mu"].shape == (8, latent_dim)
    assert out["log_var"].shape == (8, latent_dim)
    assert out["z"].shape == (8, latent_dim)


def test_vae_deterministic_mode(tiny_config: Config):
    """Deterministic mode produces identical z = mu on repeated calls."""
    n_features = tiny_config.data.num_static_features
    model = TrajectoryVAE(n_features, tiny_config)
    model.eval()

    x = torch.randn(4, n_features)
    out1 = model(x, deterministic=True)
    out2 = model(x, deterministic=True)

    assert torch.equal(out1["z"], out1["mu"])
    assert torch.allclose(out1["income"], out2["income"])


def test_vae_stochastic_mode(tiny_config: Config):
    """Stochastic mode produces different z on repeated calls."""
    n_features = tiny_config.data.num_static_features
    model = TrajectoryVAE(n_features, tiny_config)
    model.eval()

    x = torch.randn(4, n_features)
    out1 = model(x, deterministic=False)
    out2 = model(x, deterministic=False)

    # z should differ (with high probability)
    assert not torch.equal(out1["z"], out2["z"])
