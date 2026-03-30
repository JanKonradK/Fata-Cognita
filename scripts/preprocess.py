"""CLI entry point for data preprocessing pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from fata_cognita.config import load_config
from fata_cognita.data.synthetic import generate_synthetic_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Run data preprocessing pipeline."""
    parser = argparse.ArgumentParser(description="Preprocess data for training")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--synthetic", action="store_true", default=True)
    args = parser.parse_args()

    config = load_config(args.config)
    tensor_dir = Path(config.data.tensor_dir)
    tensor_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        logger.info("Generating synthetic data...")
        data = generate_synthetic_data(config)
    else:
        raise NotImplementedError("Real NLSY preprocessing not yet implemented.")

    # Save tensors
    for key, tensor in data.items():
        path = tensor_dir / f"{key}.pt"
        torch.save(tensor, path)
        logger.info("Saved %s: shape=%s → %s", key, tensor.shape, path)

    logger.info("Preprocessing complete. Tensors saved to %s", tensor_dir)


if __name__ == "__main__":
    main()
