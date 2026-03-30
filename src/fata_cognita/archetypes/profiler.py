"""Archetype profiler: compute per-archetype statistics and profiles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from fata_cognita.model.vae import TrajectoryVAE


@dataclass
class ArchetypeProfile:
    """Statistical profile for a single archetype."""

    archetype_id: int
    prevalence: float
    member_count: int
    feature_means: dict[str, float]
    feature_stds: dict[str, float]
    canonical_trajectory: dict[str, list[float]]
    cohens_d: dict[str, float]


def profile_archetypes(
    labels: np.ndarray,
    static_features: np.ndarray,
    feature_names: list[str],
    gmm_means: np.ndarray,
    model: TrajectoryVAE,
    device: torch.device,
) -> list[ArchetypeProfile]:
    """Compute profiles for all discovered archetypes.

    Args:
        labels: Hard archetype assignments, shape (N,).
        static_features: Static features, shape (N, F).
        feature_names: Names of static features.
        gmm_means: GMM component means, shape (k, latent_dim).
        model: Trained VAE for decoding canonical trajectories.
        device: Compute device.

    Returns:
        List of ArchetypeProfile, one per archetype.
    """
    n_total = len(labels)
    unique_labels = sorted(set(labels))
    pop_mean = static_features.mean(axis=0)
    pop_std = static_features.std(axis=0) + 1e-8

    profiles = []
    for aid in unique_labels:
        mask = labels == aid
        member_features = static_features[mask]
        count = int(mask.sum())

        # Feature means and stds
        means = {name: float(member_features[:, i].mean()) for i, name in enumerate(feature_names)}
        stds = {name: float(member_features[:, i].std()) for i, name in enumerate(feature_names)}

        # Cohen's d vs population
        member_mean = member_features.mean(axis=0)
        cohens = {
            name: float((member_mean[i] - pop_mean[i]) / pop_std[i])
            for i, name in enumerate(feature_names)
        }

        # Canonical trajectory: decode from GMM mean
        z_mean = torch.from_numpy(gmm_means[aid].astype(np.float32)).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            out = model.decode(z_mean)

        canonical = {
            "life_states": out["life_state_logits"].argmax(dim=-1)[0].cpu().tolist(),
            "income": out["income"][0].cpu().tolist(),
            "satisfaction": out["satisfaction"][0].cpu().tolist(),
        }

        profiles.append(ArchetypeProfile(
            archetype_id=int(aid),
            prevalence=count / n_total,
            member_count=count,
            feature_means=means,
            feature_stds=stds,
            canonical_trajectory=canonical,
            cohens_d=cohens,
        ))

    return profiles
