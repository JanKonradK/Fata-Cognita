"""Tests for validation metrics."""

from __future__ import annotations

import torch

from fata_cognita.training.metrics import (
    compute_accuracy,
    compute_active_units,
    compute_f1_macro,
    compute_mae,
)


def test_accuracy_perfect():
    """Accuracy is 1.0 when all predictions correct."""
    logits = torch.zeros(2, 5, 3)
    logits[:, :, 0] = 10.0  # strongly predict class 0
    targets = torch.zeros(2, 5, dtype=torch.long)
    masks = torch.ones(2, 5, dtype=torch.bool)

    assert compute_accuracy(logits, targets, masks) == 1.0


def test_accuracy_zero():
    """Accuracy is 0 when all predictions wrong."""
    logits = torch.zeros(2, 5, 3)
    logits[:, :, 0] = 10.0  # predict class 0
    targets = torch.ones(2, 5, dtype=torch.long)  # actual class 1
    masks = torch.ones(2, 5, dtype=torch.bool)

    assert compute_accuracy(logits, targets, masks) == 0.0


def test_mae_zero_when_matching():
    """MAE is 0 when predictions exactly match targets."""
    pred = torch.tensor([[1.0, 2.0, 3.0]])
    target = torch.tensor([[1.0, 2.0, 3.0]])
    masks = torch.ones(1, 3, dtype=torch.bool)

    assert compute_mae(pred, target, masks) == 0.0


def test_mae_value():
    """MAE computes correctly for known values."""
    pred = torch.tensor([[1.0, 2.0, 3.0]])
    target = torch.tensor([[2.0, 2.0, 5.0]])
    masks = torch.ones(1, 3, dtype=torch.bool)

    # |1-2| + |2-2| + |3-5| = 1 + 0 + 2 = 3, mean = 1.0
    assert abs(compute_mae(pred, target, masks) - 1.0) < 1e-5


def test_active_units():
    """Active units counts dimensions with variance above threshold."""
    # 4 dimensions, only first 2 have significant variance
    mu = torch.tensor(
        [
            [1.0, -1.0, 0.01, 0.0],
            [-1.0, 1.0, -0.01, 0.0],
            [0.5, -0.5, 0.005, 0.0],
        ]
    )
    assert compute_active_units(mu, threshold=0.01) == 2


def test_f1_macro_perfect():
    """F1 is 1.0 for perfect predictions."""
    logits = torch.zeros(2, 5, 3)
    targets = torch.zeros(2, 5, dtype=torch.long)
    for i in range(2):
        for j in range(5):
            logits[i, j, targets[i, j]] = 10.0
    masks = torch.ones(2, 5, dtype=torch.bool)

    assert compute_f1_macro(logits, targets, masks, num_classes=3) == 1.0
