"""Shared test fixtures for Fata Cognita."""

from __future__ import annotations

import pytest
import torch

from fata_cognita.config import (
    Config,
    DataConfig,
    ModelConfig,
    SyntheticConfig,
    TrainingConfig,
)
from fata_cognita.data.synthetic import generate_synthetic_data


@pytest.fixture()
def tiny_config() -> Config:
    """A minimal config for fast tests."""
    return Config(
        data=DataConfig(
            max_seq_len=10,
            synthetic=SyntheticConfig(n_individuals=32, missing_rate=0.2, seed=123),
        ),
        model=ModelConfig(
            latent_dim=4,
            encoder_hidden_dims=[32, 16],
            d_model=32,
            nhead=2,
            num_decoder_layers=1,
            dim_feedforward=64,
            dropout=0.0,
        ),
        training=TrainingConfig(
            max_epochs=3,
            batch_size=8,
            lr=1e-3,
            patience=2,
            seed=123,
        ),
    )


@pytest.fixture()
def synthetic_batch(tiny_config: Config) -> dict[str, torch.Tensor]:
    """A small synthetic dataset for testing."""
    return generate_synthetic_data(tiny_config)


@pytest.fixture()
def device() -> torch.device:
    """Available compute device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
