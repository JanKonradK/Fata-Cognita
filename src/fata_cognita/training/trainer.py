"""Training loop for the trajectory VAE model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from fata_cognita.model.beta_schedule import CyclicalBetaSchedule
from fata_cognita.model.loss import LossComponents, MultiTaskLoss
from fata_cognita.training.callbacks import EarlyStopping, TrainingLog
from fata_cognita.training.metrics import (
    compute_accuracy,
    compute_mae,
)

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from fata_cognita.config import Config
    from fata_cognita.model.vae import TrajectoryVAE

logger = logging.getLogger(__name__)


class Trainer:
    """Manages the full training loop for a TrajectoryVAE.

    Args:
        model: The VAE model to train.
        config: Project configuration.
        device: Compute device.
    """

    def __init__(
        self,
        model: TrajectoryVAE,
        config: Config,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.train_cfg = config.training

        self.loss_fn = MultiTaskLoss().to(device)

        # Optimizer includes both model params and loss uncertainty params
        all_params = list(model.parameters()) + list(self.loss_fn.parameters())
        self.optimizer = AdamW(
            all_params,
            lr=self.train_cfg.lr,
            weight_decay=self.train_cfg.weight_decay,
        )

        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=25, T_mult=2, eta_min=1e-6)

        self.early_stopping = EarlyStopping(patience=self.train_cfg.patience)
        self.log = TrainingLog()

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> TrainingLog:
        """Run the full training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.

        Returns:
            TrainingLog with full history.
        """
        total_steps = self.train_cfg.max_epochs * len(train_loader)
        beta_schedule = CyclicalBetaSchedule(
            total_steps=max(total_steps, 1),
            n_cycles=self.train_cfg.beta_cycles,
            ratio=self.train_cfg.beta_ratio,
        )

        best_val_loss = float("inf")

        for epoch in range(1, self.train_cfg.max_epochs + 1):
            # Train
            train_loss, kl_val = self._train_epoch(train_loader, beta_schedule)

            # Validate
            val_loss, accuracy, income_mae, satis_mae, val_kl = self._validate(
                val_loader, beta_schedule.get_beta()
            )

            # LR scheduler step
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Log
            self.log.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                beta=beta_schedule.get_beta(),
                lr=current_lr,
                kl=val_kl,
                accuracy=accuracy,
                income_mae=income_mae,
                satis_mae=satis_mae,
            )

            # Checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss)

            # Early stopping
            if self.early_stopping.should_stop(val_loss):
                break

        return self.log

    def _train_epoch(
        self,
        loader: DataLoader,
        beta_schedule: CyclicalBetaSchedule,
    ) -> tuple[float, float]:
        """Run one training epoch.

        Args:
            loader: Training DataLoader.
            beta_schedule: Beta annealing schedule.

        Returns:
            Tuple of (average loss, average KL).
        """
        self.model.train()
        total_loss = 0.0
        total_kl = 0.0
        n_batches = 0

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            beta = beta_schedule.get_beta()

            self.optimizer.zero_grad()

            outputs = self.model(batch["static_features"])
            loss_result: LossComponents = self.loss_fn(
                life_state_logits=outputs["life_state_logits"],
                life_state_targets=batch["life_states"],
                income_pred=outputs["income"],
                income_targets=batch["income"],
                satisfaction_pred=outputs["satisfaction"],
                satisfaction_targets=batch["satisfaction"],
                mu=outputs["mu"],
                log_var=outputs["log_var"],
                masks=batch["masks"],
                beta=beta,
            )

            loss_result.total.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.grad_clip_norm)
            self.optimizer.step()
            beta_schedule.step()

            total_loss += loss_result.total.item()
            total_kl += loss_result.kl.item()
            n_batches += 1

        return total_loss / max(n_batches, 1), total_kl / max(n_batches, 1)

    @torch.no_grad()
    def _validate(
        self,
        loader: DataLoader,
        beta: float,
    ) -> tuple[float, float, float, float, float]:
        """Run validation.

        Args:
            loader: Validation DataLoader.
            beta: Current beta value.

        Returns:
            Tuple of (val_loss, accuracy, income_mae, satis_mae, kl).
        """
        self.model.eval()
        total_loss = 0.0
        total_kl = 0.0
        total_acc = 0.0
        total_income_mae = 0.0
        total_satis_mae = 0.0
        n_batches = 0

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(batch["static_features"])

            loss_result = self.loss_fn(
                life_state_logits=outputs["life_state_logits"],
                life_state_targets=batch["life_states"],
                income_pred=outputs["income"],
                income_targets=batch["income"],
                satisfaction_pred=outputs["satisfaction"],
                satisfaction_targets=batch["satisfaction"],
                mu=outputs["mu"],
                log_var=outputs["log_var"],
                masks=batch["masks"],
                beta=beta,
            )

            total_loss += loss_result.total.item()
            total_kl += loss_result.kl.item()
            total_acc += compute_accuracy(
                outputs["life_state_logits"], batch["life_states"], batch["masks"]
            )
            total_income_mae += compute_mae(outputs["income"], batch["income"], batch["masks"])
            total_satis_mae += compute_mae(
                outputs["satisfaction"], batch["satisfaction"], batch["masks"]
            )
            n_batches += 1

        n = max(n_batches, 1)
        return (
            total_loss / n,
            total_acc / n,
            total_income_mae / n,
            total_satis_mae / n,
            total_kl / n,
        )

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """Save model checkpoint.

        Args:
            epoch: Current epoch number.
            val_loss: Validation loss at this checkpoint.
        """
        ckpt_dir = Path(self.train_cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss_fn_state_dict": self.loss_fn.state_dict(),
        }
        torch.save(checkpoint, ckpt_dir / "best_model.pt")
        logger.info("Checkpoint saved at epoch %d (val_loss=%.4f)", epoch, val_loss)
