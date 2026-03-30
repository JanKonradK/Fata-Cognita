"""Multi-task loss with learned uncertainty weighting and KL divergence.

Implements the loss from Kendall et al. 2018 for heterogeneous multi-task
learning, plus the VAE KL divergence term.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossComponents:
    """Breakdown of individual loss terms."""

    total: torch.Tensor
    reconstruction: torch.Tensor
    kl: torch.Tensor
    life_state_ce: torch.Tensor
    income_huber: torch.Tensor
    satisfaction_mse: torch.Tensor


class MultiTaskLoss(nn.Module):
    """Multi-task loss with learned uncertainty weighting.

    Learns per-task log-variance parameters that automatically balance
    the contribution of each loss component.

    Args:
        initial_log_var: Initial value for log-variance parameters.
    """

    def __init__(self, initial_log_var: float = 0.0) -> None:
        super().__init__()
        self.log_var_state = nn.Parameter(torch.tensor(initial_log_var))
        self.log_var_income = nn.Parameter(torch.tensor(initial_log_var))
        self.log_var_satis = nn.Parameter(torch.tensor(initial_log_var))

    def forward(
        self,
        life_state_logits: torch.Tensor,
        life_state_targets: torch.Tensor,
        income_pred: torch.Tensor,
        income_targets: torch.Tensor,
        satisfaction_pred: torch.Tensor,
        satisfaction_targets: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        masks: torch.Tensor,
        beta: float = 1.0,
    ) -> LossComponents:
        """Compute the full VAE multi-task loss.

        Args:
            life_state_logits: Predicted logits, shape (B, S, C).
            life_state_targets: Target labels, shape (B, S).
            income_pred: Predicted income, shape (B, S).
            income_targets: Target income, shape (B, S).
            satisfaction_pred: Predicted satisfaction, shape (B, S).
            satisfaction_targets: Target satisfaction, shape (B, S).
            mu: Latent mean, shape (B, latent_dim).
            log_var: Latent log-variance, shape (B, latent_dim).
            masks: Observation masks, shape (B, S). True = observed.
            beta: KL divergence weight (annealed during training).

        Returns:
            LossComponents with total loss and individual terms.
        """
        # Flatten to observed positions only
        mask_flat = masks.reshape(-1)

        # Life state cross-entropy (masked)
        logits_flat = life_state_logits.reshape(-1, life_state_logits.size(-1))
        targets_flat = life_state_targets.reshape(-1)
        if mask_flat.any():
            ce_loss = F.cross_entropy(
                logits_flat[mask_flat], targets_flat[mask_flat], reduction="mean"
            )
        else:
            ce_loss = torch.tensor(0.0, device=mu.device)

        # Income Huber loss (masked)
        income_flat = income_pred.reshape(-1)
        income_target_flat = income_targets.reshape(-1)
        if mask_flat.any():
            huber_loss = F.smooth_l1_loss(
                income_flat[mask_flat], income_target_flat[mask_flat], reduction="mean"
            )
        else:
            huber_loss = torch.tensor(0.0, device=mu.device)

        # Satisfaction MSE loss (masked)
        satis_flat = satisfaction_pred.reshape(-1)
        satis_target_flat = satisfaction_targets.reshape(-1)
        if mask_flat.any():
            mse_loss = F.mse_loss(
                satis_flat[mask_flat], satis_target_flat[mask_flat], reduction="mean"
            )
        else:
            mse_loss = torch.tensor(0.0, device=mu.device)

        # Uncertainty-weighted reconstruction loss
        precision_state = torch.exp(-self.log_var_state)
        precision_income = torch.exp(-self.log_var_income)
        precision_satis = torch.exp(-self.log_var_satis)

        recon_loss = (
            0.5 * precision_state * ce_loss + 0.5 * self.log_var_state
            + 0.5 * precision_income * huber_loss + 0.5 * self.log_var_income
            + 0.5 * precision_satis * mse_loss + 0.5 * self.log_var_satis
        )

        # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        )

        total = recon_loss + beta * kl_loss

        return LossComponents(
            total=total,
            reconstruction=recon_loss,
            kl=kl_loss,
            life_state_ce=ce_loss,
            income_huber=huber_loss,
            satisfaction_mse=mse_loss,
        )
