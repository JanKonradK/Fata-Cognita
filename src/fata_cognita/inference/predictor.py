"""Single-individual trajectory prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from fata_cognita.data.synthetic import LifeState

if TYPE_CHECKING:
    from sklearn.mixture import GaussianMixture

    from fata_cognita.data.scaler import FeatureScaler
    from fata_cognita.model.vae import TrajectoryVAE


@dataclass
class TrajectoryPoint:
    """A single time-step in a predicted trajectory."""

    age: int
    life_state: str
    life_state_probs: dict[str, float]
    income: float
    satisfaction: float


@dataclass
class PredictionResult:
    """Full prediction result for an individual."""

    trajectory: list[TrajectoryPoint]
    archetype_id: int
    archetype_membership: dict[int, float]
    latent_vector: list[float]


def predict_trajectory(
    static_features: dict[str, float],
    feature_names: list[str],
    model: TrajectoryVAE,
    gmm: GaussianMixture,
    scaler: FeatureScaler,
    device: torch.device,
    deterministic: bool = True,
    min_age: int = 14,
) -> PredictionResult:
    """Generate a predicted life trajectory for one individual.

    Args:
        static_features: Dict mapping feature names to values.
        feature_names: Ordered list of feature names matching model input.
        model: Trained VAE model.
        gmm: Fitted GMM for archetype assignment.
        scaler: Fitted feature scaler.
        device: Compute device.
        deterministic: If True, use mu instead of sampling.
        min_age: Starting age for the trajectory.

    Returns:
        PredictionResult with trajectory and archetype info.
    """
    # Build feature vector
    x = torch.tensor(
        [[static_features.get(name, 0.0) for name in feature_names]],
        dtype=torch.float32,
    )
    x = scaler.transform_static(x).to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(x, deterministic=deterministic)

    logits = outputs["life_state_logits"][0]  # (S, C)
    income = outputs["income"][0]  # (S,)
    satisfaction = outputs["satisfaction"][0]  # (S,)
    mu = outputs["mu"][0].cpu().numpy()  # (D,)
    z = outputs["z"][0].cpu().tolist()

    # Inverse-scale income
    income_original = scaler.inverse_income(income.unsqueeze(0).cpu())[0]

    # Archetype assignment
    hard_label = int(gmm.predict(mu.reshape(1, -1))[0])
    soft_probs = gmm.predict_proba(mu.reshape(1, -1))[0]
    membership = {i: float(p) for i, p in enumerate(soft_probs)}

    # Build trajectory
    state_names = [s.name for s in LifeState]
    probs = F.softmax(logits, dim=-1).cpu()

    trajectory = []
    for t in range(logits.shape[0]):
        age = min_age + t
        state_idx = int(logits[t].argmax())
        state_prob_dict = {state_names[c]: float(probs[t, c]) for c in range(len(state_names))}

        trajectory.append(
            TrajectoryPoint(
                age=age,
                life_state=state_names[state_idx],
                life_state_probs=state_prob_dict,
                income=float(np.expm1(max(0.0, income_original[t].item()))),
                satisfaction=float(satisfaction[t].item()),
            )
        )

    return PredictionResult(
        trajectory=trajectory,
        archetype_id=hard_label,
        archetype_membership=membership,
        latent_vector=z,
    )
