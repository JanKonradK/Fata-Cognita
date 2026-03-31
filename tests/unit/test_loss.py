"""Tests for the multi-task loss function."""

from __future__ import annotations

import torch

from fata_cognita.model.loss import MultiTaskLoss


def test_loss_positive_for_random():
    """Loss is positive for random predictions."""
    loss_fn = MultiTaskLoss()

    logits = torch.randn(4, 10, 9)
    targets = torch.randint(0, 9, (4, 10))
    income_pred = torch.randn(4, 10)
    income_target = torch.randn(4, 10)
    satis_pred = torch.sigmoid(torch.randn(4, 10))
    satis_target = torch.rand(4, 10)
    mu = torch.randn(4, 16)
    log_var = torch.randn(4, 16)
    masks = torch.ones(4, 10, dtype=torch.bool)

    result = loss_fn(
        logits,
        targets,
        income_pred,
        income_target,
        satis_pred,
        satis_target,
        mu,
        log_var,
        masks,
        beta=1.0,
    )

    assert result.total.item() > 0


def test_kl_zero_at_prior():
    """KL divergence is zero when mu=0 and log_var=0 (matching prior)."""
    loss_fn = MultiTaskLoss()

    logits = torch.zeros(2, 5, 9)
    targets = torch.zeros(2, 5, dtype=torch.long)
    income_pred = torch.zeros(2, 5)
    income_target = torch.zeros(2, 5)
    satis_pred = torch.full((2, 5), 0.5)
    satis_target = torch.full((2, 5), 0.5)
    mu = torch.zeros(2, 8)
    log_var = torch.zeros(2, 8)
    masks = torch.ones(2, 5, dtype=torch.bool)

    result = loss_fn(
        logits,
        targets,
        income_pred,
        income_target,
        satis_pred,
        satis_target,
        mu,
        log_var,
        masks,
        beta=1.0,
    )

    assert abs(result.kl.item()) < 1e-5


def test_mask_respected():
    """Loss should be zero contribution from masked positions."""
    loss_fn = MultiTaskLoss()

    # All positions masked out
    logits = torch.randn(2, 5, 9)
    targets = torch.randint(0, 9, (2, 5))
    income_pred = torch.randn(2, 5)
    income_target = torch.randn(2, 5)
    satis_pred = torch.sigmoid(torch.randn(2, 5))
    satis_target = torch.rand(2, 5)
    mu = torch.zeros(2, 8)
    log_var = torch.zeros(2, 8)
    masks = torch.zeros(2, 5, dtype=torch.bool)  # all masked

    result = loss_fn(
        logits,
        targets,
        income_pred,
        income_target,
        satis_pred,
        satis_target,
        mu,
        log_var,
        masks,
        beta=0.0,
    )

    # With beta=0 and all masked, reconstruction should be ~0 (just log_var terms)
    assert result.life_state_ce.item() == 0.0
    assert result.income_huber.item() == 0.0
    assert result.satisfaction_mse.item() == 0.0


def test_uncertainty_weights_learnable():
    """Uncertainty weight parameters have requires_grad=True."""
    loss_fn = MultiTaskLoss()
    assert loss_fn.log_var_state.requires_grad
    assert loss_fn.log_var_income.requires_grad
    assert loss_fn.log_var_satis.requires_grad
