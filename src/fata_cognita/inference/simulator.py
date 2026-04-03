"""Monte Carlo trajectory simulation with percentile bands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from fata_cognita.data.synthetic import LifeState
from fata_cognita.inference.transforms import inverse_income_to_nominal

if TYPE_CHECKING:
    from fata_cognita.data.scaler import FeatureScaler
    from fata_cognita.model.vae import TrajectoryVAE


@dataclass
class SimulationResult:
    """Result of Monte Carlo trajectory simulation."""

    ages: list[int]
    income_percentiles: dict[str, list[float]]
    satisfaction_percentiles: dict[str, list[float]]
    state_distribution: list[dict[str, float]]
    archetype_id: int
    n_simulations: int


def simulate_trajectories(
    static_features: dict[str, float],
    feature_names: list[str],
    model: TrajectoryVAE,
    scaler: FeatureScaler,
    device: torch.device,
    n_simulations: int = 1000,
    percentiles: list[int] | None = None,
    min_age: int = 14,
) -> SimulationResult:
    """Run Monte Carlo simulation for an individual.

    Samples N latent vectors from the posterior and decodes each into a
    trajectory. Computes percentile bands and state distributions.

    Args:
        static_features: Dict mapping feature names to values.
        feature_names: Ordered list of feature names.
        model: Trained VAE model.
        scaler: Fitted feature scaler.
        device: Compute device.
        n_simulations: Number of trajectories to sample.
        percentiles: Percentiles to compute (default: [10, 25, 50, 75, 90]).
        min_age: Starting age.

    Returns:
        SimulationResult with percentile bands and state distributions.
    """
    if percentiles is None:
        percentiles = [10, 25, 50, 75, 90]

    # Build feature vector
    x = torch.tensor(
        [[static_features.get(name, 0.0) for name in feature_names]],
        dtype=torch.float32,
    )
    x = scaler.transform_static(x).to(device)

    model.eval()
    with torch.no_grad():
        # Encode once
        mu, log_var = model.encode(x)

        # Sample N z vectors
        std = torch.exp(0.5 * log_var)
        eps = torch.randn(n_simulations, mu.size(-1), device=device)
        z_samples = mu + std * eps  # (N, D)

        # Decode all at once
        outputs = model.decode(z_samples)

    seq_len = outputs["income"].shape[1]
    ages = list(range(min_age, min_age + seq_len))
    state_names = [s.name for s in LifeState]

    # Income: inverse-scale and convert from log scale to nominal dollars
    income_all = inverse_income_to_nominal(scaler.inverse_income(outputs["income"].cpu())).numpy()

    # Satisfaction
    satis_all = outputs["satisfaction"].cpu().numpy()

    # Life state distributions via softmax sampling
    logits = outputs["life_state_logits"].cpu()
    probs = F.softmax(logits, dim=-1)  # (N, S, C)

    # Percentile bands
    income_pcts: dict[str, list[float]] = {}
    satis_pcts: dict[str, list[float]] = {}
    for p in percentiles:
        key = f"p{p}"
        income_pcts[key] = np.percentile(income_all, p, axis=0).tolist()
        satis_pcts[key] = np.percentile(satis_all, p, axis=0).tolist()

    # State distribution: average probabilities across simulations
    avg_probs = probs.mean(dim=0)  # (S, C)
    state_dist = []
    for t in range(seq_len):
        step_dist = {state_names[c]: float(avg_probs[t, c]) for c in range(len(state_names))}
        state_dist.append(step_dist)

    # Archetype: use the deterministic mu
    archetype_id = 0  # will be set by caller with GMM

    return SimulationResult(
        ages=ages,
        income_percentiles=income_pcts,
        satisfaction_percentiles=satis_pcts,
        state_distribution=state_dist,
        archetype_id=archetype_id,
        n_simulations=n_simulations,
    )
