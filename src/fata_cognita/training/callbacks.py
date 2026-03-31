"""Training callbacks for logging, LR scheduling, and beta updates."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TrainingLog:
    """Stores training history for all epochs."""

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    betas: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    kl_values: list[float] = field(default_factory=list)
    life_state_accuracies: list[float] = field(default_factory=list)
    income_maes: list[float] = field(default_factory=list)
    satisfaction_maes: list[float] = field(default_factory=list)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        beta: float,
        lr: float,
        kl: float,
        accuracy: float,
        income_mae: float,
        satis_mae: float,
    ) -> None:
        """Record and log metrics for one epoch.

        Args:
            epoch: Current epoch number.
            train_loss: Average training loss.
            val_loss: Average validation loss.
            beta: Current beta value.
            lr: Current learning rate.
            kl: KL divergence.
            accuracy: Life state accuracy.
            income_mae: Income MAE.
            satis_mae: Satisfaction MAE.
        """
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.betas.append(beta)
        self.learning_rates.append(lr)
        self.kl_values.append(kl)
        self.life_state_accuracies.append(accuracy)
        self.income_maes.append(income_mae)
        self.satisfaction_maes.append(satis_mae)

        logger.info(
            "Epoch %3d | train_loss=%.4f | val_loss=%.4f | acc=%.3f | "
            "income_mae=%.4f | satis_mae=%.4f | kl=%.4f | beta=%.3f | lr=%.2e",
            epoch,
            train_loss,
            val_loss,
            accuracy,
            income_mae,
            satis_mae,
            kl,
            beta,
            lr,
        )


class EarlyStopping:
    """Stop training when validation loss stops improving.

    Args:
        patience: Number of epochs to wait after last improvement.
        min_delta: Minimum change to qualify as an improvement.
    """

    def __init__(self, patience: int = 20, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float | None = None
        self.counter = 0

    def should_stop(self, val_loss: float) -> bool:
        """Check if training should stop.

        Args:
            val_loss: Current validation loss.

        Returns:
            True if patience exceeded.
        """
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False

        self.counter += 1
        if self.counter >= self.patience:
            logger.info(
                "Early stopping triggered after %d epochs without improvement",
                self.counter,
            )
            return True
        return False

    @property
    def improved(self) -> bool:
        """Whether the last call to should_stop saw an improvement."""
        return self.counter == 0
