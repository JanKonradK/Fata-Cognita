"""Tests for sequence builder."""

from __future__ import annotations

import pandas as pd
import torch

from fata_cognita.config import DataConfig
from fata_cognita.data.sequence_builder import build_sequences, split_by_caseid


def test_build_sequences_shapes():
    """Output tensors have correct shapes for the given config."""
    config = DataConfig(max_seq_len=10)

    df = pd.DataFrame({
        "caseid": [1, 1, 1, 2, 2],
        "age": [14, 15, 16, 14, 15],
        "life_state": [5, 5, 0, 5, 1],
        "log_income": [7.0, 7.5, 10.0, 7.0, 9.0],
        "satisfaction": [0.5, 0.6, 0.7, 0.4, 0.5],
    })

    result = build_sequences(df, config)

    assert result["life_states"].shape == (2, 10)
    assert result["income"].shape == (2, 10)
    assert result["satisfaction"].shape == (2, 10)
    assert result["masks"].shape == (2, 10)


def test_build_sequences_alignment():
    """Values are placed at the correct age-aligned positions."""
    config = DataConfig(max_seq_len=10)

    df = pd.DataFrame({
        "caseid": [1, 1],
        "age": [14, 16],  # age 14 → index 0, age 16 → index 2
        "life_state": [5, 0],
        "log_income": [7.0, 10.0],
        "satisfaction": [0.5, 0.7],
    })

    result = build_sequences(df, config)

    # Position 0 (age 14) should be set
    assert result["life_states"][0, 0] == 5
    assert result["masks"][0, 0]

    # Position 1 (age 15) should be unset
    assert not result["masks"][0, 1]

    # Position 2 (age 16) should be set
    assert result["life_states"][0, 2] == 0
    assert result["masks"][0, 2]


def test_build_sequences_padding():
    """Unobserved positions are zero-padded with False masks."""
    config = DataConfig(max_seq_len=10)

    df = pd.DataFrame({
        "caseid": [1],
        "age": [14],
        "life_state": [5],
        "log_income": [7.0],
        "satisfaction": [0.5],
    })

    result = build_sequences(df, config)

    # Only position 0 should be masked True
    assert result["masks"][0, 0]
    assert not result["masks"][0, 1:].any()

    # Padded positions should be 0
    assert result["life_states"][0, 1:].sum() == 0


def test_split_by_caseid():
    """Split produces non-overlapping train/val/test sets."""
    caseids = torch.arange(100)
    splits = split_by_caseid(caseids, train_frac=0.8, val_frac=0.1, seed=42)

    assert len(splits["train"]) == 80
    assert len(splits["val"]) == 10
    assert len(splits["test"]) == 10

    # No overlap
    all_indices = splits["train"] + splits["val"] + splits["test"]
    assert len(set(all_indices)) == 100
