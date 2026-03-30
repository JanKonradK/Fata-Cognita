"""Centralized configuration for Fata Cognita.

Loads from YAML config file with environment variable overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import yaml


@dataclass(frozen=True)
class SyntheticConfig:
    """Configuration for the synthetic data generator."""

    n_individuals: int = 500
    missing_rate: float = 0.2
    seed: int = 42


@dataclass(frozen=True)
class DataConfig:
    """Configuration for data loading and preprocessing."""

    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    tensor_dir: str = "data/tensors"
    max_seq_len: int = 62
    num_life_states: int = 9
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    static_features: list[str] = field(default_factory=lambda: [
        "sex", "race_hispanic", "race_black", "race_other",
        "birth_year", "parent_education", "family_income_14",
        "region_northeast", "region_north_central", "region_south", "region_west",
        "afqt_score", "afqt_available", "cohort",
    ])
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)

    @property
    def num_static_features(self) -> int:
        """Number of static input features for the encoder."""
        return len(self.static_features)


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the VAE-Transformer model."""

    latent_dim: int = 16
    encoder_hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    d_model: int = 128
    nhead: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for the training loop."""

    max_epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    patience: int = 20
    beta_cycles: int = 4
    beta_ratio: float = 0.5
    seed: int = 42
    checkpoint_dir: str = "checkpoints"


@dataclass(frozen=True)
class APIConfig:
    """Configuration for the FastAPI server."""

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    model_checkpoint: str = "checkpoints/best_model.pt"
    gmm_path: str = "checkpoints/gmm.pkl"
    scaler_path: str = "checkpoints/scalers.pkl"
    archetype_profiles_path: str = "checkpoints/archetype_profiles.json"


@dataclass(frozen=True)
class DashboardConfig:
    """Configuration for the Streamlit dashboard."""

    api_url: str = "http://localhost:8000/api/v1"


@dataclass(frozen=True)
class Config:
    """Root configuration for the entire project."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            A Config instance populated from the YAML file.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, raw: dict) -> Self:
        """Build Config from a nested dictionary."""
        data_raw = raw.get("data", {})
        synthetic_raw = data_raw.pop("synthetic", {})

        return cls(
            data=DataConfig(
                **{k: v for k, v in data_raw.items() if k != "synthetic"},
                synthetic=SyntheticConfig(**synthetic_raw) if synthetic_raw else SyntheticConfig(),
            ),
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
            api=APIConfig(**raw.get("api", {})),
            dashboard=DashboardConfig(**raw.get("dashboard", {})),
        )


def load_config(path: str | Path | None = None) -> Config:
    """Load config from the given path or the FC_CONFIG_PATH env var.

    Falls back to ``config/default.yaml`` relative to the project root.

    Args:
        path: Optional explicit path to the config YAML.

    Returns:
        A fully populated Config instance.
    """
    if path is None:
        path = os.environ.get("FC_CONFIG_PATH", "config/default.yaml")
    return Config.from_yaml(path)
