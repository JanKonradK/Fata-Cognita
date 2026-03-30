"""Integration test: training loop on synthetic data."""

from __future__ import annotations

from pathlib import Path

import torch

from fata_cognita.config import Config
from fata_cognita.data.dataset import create_dataloaders
from fata_cognita.data.sequence_builder import split_by_caseid
from fata_cognita.data.synthetic import generate_synthetic_data
from fata_cognita.model.vae import TrajectoryVAE
from fata_cognita.training.trainer import Trainer


def test_training_loss_decreases(tmp_path: Path):
    """Training for 3 epochs on synthetic data should decrease loss."""
    config = Config()
    config = Config(
        data=config.data,
        model=config.model,
        training=config.training.__class__(
            max_epochs=3,
            batch_size=64,
            lr=1e-3,
            patience=10,
            seed=42,
            checkpoint_dir=str(tmp_path / "checkpoints"),
        ),
    )

    # Generate synthetic data
    data = generate_synthetic_data(config)

    # Split
    n = data["static_features"].shape[0]
    indices = list(range(n))
    splits = {
        "train": indices[:400],
        "val": indices[400:450],
        "test": indices[450:],
    }

    loaders = create_dataloaders(data, splits, batch_size=64)

    # Create model
    num_features = data["static_features"].shape[1]
    model = TrajectoryVAE(num_features, config)
    device = torch.device("cpu")

    # Train
    trainer = Trainer(model, config, device)
    log = trainer.train(loaders["train"], loaders["val"])

    # Verify
    assert len(log.train_losses) == 3
    assert all(not torch.isnan(torch.tensor(l)) for l in log.train_losses)

    # Loss should generally decrease (epoch 3 < epoch 1)
    assert log.train_losses[-1] < log.train_losses[0], (
        f"Loss did not decrease: {log.train_losses[0]:.4f} → {log.train_losses[-1]:.4f}"
    )

    # Checkpoint should exist
    assert (tmp_path / "checkpoints" / "best_model.pt").exists()
