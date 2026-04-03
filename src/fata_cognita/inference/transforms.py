"""Shared transforms for inference output post-processing."""

from __future__ import annotations

import torch


def inverse_income_to_nominal(log_income: torch.Tensor) -> torch.Tensor:
    """Convert log-scale income to nominal dollars.

    Clamps negative log-income to zero before applying expm1 to avoid
    negative dollar values.

    Args:
        log_income: Tensor of log(1 + income) values.

    Returns:
        Tensor of nominal income values (non-negative).
    """
    return torch.expm1(log_income.clamp(min=0.0))
