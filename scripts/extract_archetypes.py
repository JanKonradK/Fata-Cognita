"""CLI entry point for post-training archetype extraction."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from fata_cognita.archetypes.extractor import encode_all, fit_gmm_with_bic, save_gmm
from fata_cognita.archetypes.profiler import profile_archetypes
from fata_cognita.config import load_config
from fata_cognita.data.synthetic import generate_synthetic_data
from fata_cognita.device import get_device
from fata_cognita.model.vae import TrajectoryVAE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Extract archetypes from a trained model."""
    parser = argparse.ArgumentParser(description="Extract archetypes from trained model")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--synthetic", action="store_true", default=True)
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()

    # Load model
    num_features = config.data.num_static_features
    model = TrajectoryVAE(num_features, config)
    checkpoint = torch.load(config.api.model_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load data
    if args.synthetic:
        data = generate_synthetic_data(config)
    else:
        raise NotImplementedError("Real data loading not yet implemented.")

    static_features = data["static_features"]

    # Encode all
    logger.info("Encoding %d individuals...", len(static_features))
    z = encode_all(model, static_features, device)

    # Fit GMM
    logger.info("Fitting GMM with BIC selection...")
    gmm, k_best, bic_scores = fit_gmm_with_bic(z, k_range=(3, 15))
    save_gmm(gmm, config.api.gmm_path)
    logger.info("Saved GMM (k=%d) to %s", k_best, config.api.gmm_path)

    # Profile archetypes
    labels = gmm.predict(z)
    profiles = profile_archetypes(
        labels=labels,
        static_features=static_features.numpy(),
        feature_names=config.data.static_features,
        gmm_means=gmm.means_,
        model=model,
        device=device,
    )

    # Save profiles
    profiles_path = Path(config.training.checkpoint_dir) / "archetype_profiles.json"
    profiles_data = []
    for p in profiles:
        profiles_data.append({
            "archetype_id": p.archetype_id,
            "prevalence": p.prevalence,
            "member_count": p.member_count,
            "feature_means": p.feature_means,
            "canonical_trajectory": p.canonical_trajectory,
            "cohens_d": p.cohens_d,
        })
    with open(profiles_path, "w") as f:
        json.dump(profiles_data, f, indent=2)
    logger.info("Saved profiles to %s", profiles_path)

    # Log summary
    for p in profiles:
        logger.info(
            "Archetype %d: prevalence=%.1f%%, members=%d",
            p.archetype_id, p.prevalence * 100, p.member_count,
        )


if __name__ == "__main__":
    main()
