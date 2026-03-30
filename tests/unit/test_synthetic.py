"""Tests for the synthetic data generator."""

from __future__ import annotations

import torch

from fata_cognita.config import Config
from fata_cognita.data.synthetic import LifeState, generate_synthetic_data


def test_output_keys(synthetic_batch: dict[str, torch.Tensor]):
    """Generator returns all expected tensor keys."""
    expected = {"static_features", "life_states", "income", "satisfaction", "masks"}
    assert set(synthetic_batch.keys()) == expected


def test_static_features_shape(synthetic_batch: dict[str, torch.Tensor], tiny_config: Config):
    """Static features have shape (N, num_static_features)."""
    sf = synthetic_batch["static_features"]
    assert sf.shape == (32, tiny_config.data.num_static_features)
    assert sf.dtype == torch.float32


def test_life_states_shape_and_range(synthetic_batch: dict[str, torch.Tensor]):
    """Life states have correct shape and valid range [0, 8]."""
    ls = synthetic_batch["life_states"]
    assert ls.shape == (32, 10)
    assert ls.dtype == torch.int64
    assert ls.min() >= 0
    assert ls.max() <= 8


def test_income_positive(synthetic_batch: dict[str, torch.Tensor]):
    """Income values are non-negative."""
    inc = synthetic_batch["income"]
    assert inc.shape == (32, 10)
    assert inc.dtype == torch.float32
    assert (inc >= 0).all()


def test_satisfaction_bounded(synthetic_batch: dict[str, torch.Tensor]):
    """Satisfaction values are in [0, 1]."""
    sat = synthetic_batch["satisfaction"]
    assert sat.shape == (32, 10)
    assert sat.dtype == torch.float32
    assert (sat >= 0).all()
    assert (sat <= 1).all()


def test_masks_boolean(synthetic_batch: dict[str, torch.Tensor]):
    """Masks are boolean with expected missingness."""
    m = synthetic_batch["masks"]
    assert m.shape == (32, 10)
    assert m.dtype == torch.bool
    # Not all True (some missingness) and not all False
    assert not m.all()
    assert m.any()


def test_reproducibility():
    """Same seed produces identical data."""
    cfg = Config()
    d1 = generate_synthetic_data(cfg)
    d2 = generate_synthetic_data(cfg)
    assert torch.equal(d1["life_states"], d2["life_states"])
    assert torch.equal(d1["static_features"], d2["static_features"])


def test_life_state_enum():
    """LifeState enum covers all 9 states."""
    assert len(LifeState) == 9
    assert LifeState.EMPLOYED_FT == 0
    assert LifeState.DISABLED == 8
