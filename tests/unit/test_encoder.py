"""Tests for the MLP encoder."""

from __future__ import annotations

import torch

from fata_cognita.config import Config
from fata_cognita.model.encoder import Encoder


def test_encoder_output_shapes(tiny_config: Config):
    """Encoder produces mu and log_var with correct shapes."""
    n_features = tiny_config.data.num_static_features
    enc = Encoder(n_features, tiny_config.model)

    x = torch.randn(8, n_features)
    mu, log_var = enc(x)

    assert mu.shape == (8, tiny_config.model.latent_dim)
    assert log_var.shape == (8, tiny_config.model.latent_dim)


def test_encoder_gradient_flow(tiny_config: Config):
    """Gradients flow through the encoder."""
    n_features = tiny_config.data.num_static_features
    enc = Encoder(n_features, tiny_config.model)

    x = torch.randn(4, n_features, requires_grad=True)
    mu, log_var = enc(x)
    loss = mu.sum() + log_var.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_encoder_different_inputs_different_outputs(tiny_config: Config):
    """Different inputs produce different outputs."""
    n_features = tiny_config.data.num_static_features
    enc = Encoder(n_features, tiny_config.model)

    x1 = torch.randn(4, n_features)
    x2 = torch.randn(4, n_features)
    mu1, _ = enc(x1)
    mu2, _ = enc(x2)

    assert not torch.allclose(mu1, mu2)
