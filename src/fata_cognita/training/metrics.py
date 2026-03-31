"""Validation metrics for trajectory model evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass
class ValidationMetrics:
    """Container for validation metric values."""

    total_loss: float
    life_state_accuracy: float
    life_state_f1_macro: float
    income_mae: float
    satisfaction_mae: float
    kl_divergence: float
    active_units: int


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor) -> float:
    """Compute top-1 accuracy on masked positions.

    Args:
        logits: Predicted logits, shape (B, S, C).
        targets: Target labels, shape (B, S).
        masks: Observation masks, shape (B, S).

    Returns:
        Accuracy as a float in [0, 1].
    """
    if not masks.any():
        return 0.0

    preds = logits.argmax(dim=-1)  # (B, S)
    correct = (preds == targets) & masks
    return (correct.sum().float() / masks.sum().float()).item()


def compute_f1_macro(
    logits: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int = 9,
) -> float:
    """Compute macro F1 score across all classes on masked positions.

    Args:
        logits: Predicted logits, shape (B, S, C).
        targets: Target labels, shape (B, S).
        masks: Observation masks, shape (B, S).
        num_classes: Number of classes.

    Returns:
        Macro F1 score.
    """
    if not masks.any():
        return 0.0

    preds = logits.argmax(dim=-1)[masks]
    tgts = targets[masks]

    f1_sum = 0.0
    n_classes_present = 0

    for c in range(num_classes):
        tp = ((preds == c) & (tgts == c)).sum().float()
        fp = ((preds == c) & (tgts != c)).sum().float()
        fn = ((preds != c) & (tgts == c)).sum().float()

        if tp + fn == 0:
            continue  # class not present in targets

        n_classes_present += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn)
        if precision + recall > 0:
            f1_sum += 2 * precision * recall / (precision + recall)

    return f1_sum / max(n_classes_present, 1)


def compute_mae(pred: torch.Tensor, target: torch.Tensor, masks: torch.Tensor) -> float:
    """Compute mean absolute error on masked positions.

    Args:
        pred: Predictions, shape (B, S).
        target: Targets, shape (B, S).
        masks: Observation masks, shape (B, S).

    Returns:
        MAE as a float.
    """
    if not masks.any():
        return 0.0

    return (pred[masks] - target[masks]).abs().mean().item()


def compute_active_units(mu: torch.Tensor, threshold: float = 0.01) -> int:
    """Count latent dimensions with variance above a threshold.

    Active units indicate the model is using the latent space effectively.
    Low count suggests KL collapse.

    Args:
        mu: Latent means from the entire dataset, shape (N, latent_dim).
        threshold: Minimum variance to count as active.

    Returns:
        Number of active latent dimensions.
    """
    var_per_dim = mu.var(dim=0)  # (latent_dim,)
    return int((var_per_dim > threshold).sum().item())
