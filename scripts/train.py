"""CLI entry point for model training."""

from __future__ import annotations

import argparse
import logging

import torch

from fata_cognita.config import load_config
from fata_cognita.data.dataset import create_dataloaders
from fata_cognita.data.scaler import FeatureScaler
from fata_cognita.data.sequence_builder import split_by_caseid
from fata_cognita.data.synthetic import generate_synthetic_data
from fata_cognita.device import get_device
from fata_cognita.model.vae import TrajectoryVAE
from fata_cognita.training.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Run training pipeline."""
    parser = argparse.ArgumentParser(description="Train the Fata Cognita trajectory model")
    parser.add_argument("--config", default="config/default.yaml", help="Path to config YAML")
    parser.add_argument("--synthetic", action="store_true", default=True, help="Use synthetic data")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()
    logger.info("Device: %s", device)

    # Load data
    if args.synthetic:
        logger.info("Generating synthetic data...")
        data = generate_synthetic_data(config)
    else:
        raise NotImplementedError("Real NLSY data loading not yet implemented. Use --synthetic.")

    # Scale
    scaler = FeatureScaler()
    n = data["static_features"].shape[0]
    splits = split_by_caseid(torch.arange(n), config.data.train_fraction, config.data.val_fraction, config.training.seed)
    train_idx = splits["train"]

    scaler.fit(
        static_features=data["static_features"][train_idx],
        income=data["income"][train_idx],
        masks=data["masks"][train_idx],
    )
    data["static_features"] = scaler.transform_static(data["static_features"])
    data["income"] = scaler.transform_income(data["income"])
    scaler.save(config.api.scaler_path)

    # Create data loaders
    loaders = create_dataloaders(data, splits, config.training.batch_size)

    # Create model
    num_features = data["static_features"].shape[1]
    model = TrajectoryVAE(num_features, config)
    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    # Train
    trainer = Trainer(model, config, device)
    log = trainer.train(loaders["train"], loaders["val"])

    logger.info("Training complete. Best val loss: %.4f", min(log.val_losses))
    logger.info("Final accuracy: %.3f", log.life_state_accuracies[-1])


if __name__ == "__main__":
    main()
