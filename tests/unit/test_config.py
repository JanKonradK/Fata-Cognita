"""Tests for the configuration system."""

from __future__ import annotations

from pathlib import Path

import pytest

from fata_cognita.config import Config, load_config


def test_default_config_loads():
    """Default Config() should have sensible defaults."""
    cfg = Config()
    assert cfg.model.latent_dim == 16
    assert cfg.data.max_seq_len == 62
    assert cfg.data.num_life_states == 9
    assert cfg.training.max_epochs == 200
    assert cfg.data.num_static_features == 14


def test_from_yaml(tmp_path: Path):
    """Config loads correctly from a YAML file."""
    yaml_content = """
model:
  latent_dim: 8
  d_model: 64
training:
  max_epochs: 50
  batch_size: 128
data:
  max_seq_len: 30
"""
    cfg_file = tmp_path / "test_config.yaml"
    cfg_file.write_text(yaml_content)

    cfg = Config.from_yaml(cfg_file)
    assert cfg.model.latent_dim == 8
    assert cfg.model.d_model == 64
    assert cfg.training.max_epochs == 50
    assert cfg.training.batch_size == 128
    assert cfg.data.max_seq_len == 30


def test_missing_yaml_raises():
    """Loading a nonexistent YAML file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        Config.from_yaml("nonexistent_file.yaml")


def test_load_config_default():
    """load_config with the project default.yaml works."""
    cfg = load_config("config/default.yaml")
    assert cfg.model.latent_dim == 16
    assert cfg.data.synthetic.n_individuals == 500


def test_num_static_features():
    """num_static_features property matches the feature list length."""
    cfg = Config()
    assert cfg.data.num_static_features == len(cfg.data.static_features)
