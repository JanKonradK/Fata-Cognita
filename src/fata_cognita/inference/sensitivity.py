"""Counterfactual sensitivity / inflection-point analysis.

Perturbs one decision variable and compares Monte Carlo trajectories
to identify which decisions have the largest impact on life outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fata_cognita.inference.simulator import SimulationResult, simulate_trajectories


@dataclass
class InflectionPoint:
    """A detected inflection point where a perturbation has maximum impact."""

    age: int
    delta_income: float
    delta_satisfaction: float
    significance: float


@dataclass
class SensitivityResult:
    """Result of a counterfactual sensitivity analysis."""

    perturb_variable: str
    perturb_value: float
    deltas_by_age: list[dict[str, float]]
    inflection_points: list[InflectionPoint]
    overall_effect_size: float
    base_archetype: int
    perturbed_archetype: int


def run_sensitivity_analysis(
    static_features: dict[str, float],
    perturb_variable: str,
    perturb_value: float,
    feature_names: list[str],
    model,
    gmm,
    scaler,
    device,
    n_simulations: int = 10000,
    min_age: int = 14,
) -> SensitivityResult:
    """Run counterfactual sensitivity analysis.

    Simulates trajectories for base and perturbed scenarios, computes
    deltas, and identifies inflection points.

    Args:
        static_features: Base person features.
        perturb_variable: Feature name to perturb.
        perturb_value: New value for the perturbed feature.
        feature_names: Ordered feature names.
        model: Trained VAE.
        gmm: Fitted GMM.
        scaler: Fitted scaler.
        device: Compute device.
        n_simulations: Number of MC trajectories per scenario.
        min_age: Starting age.

    Returns:
        SensitivityResult with deltas and inflection points.
    """
    # Base scenario
    base_sim = simulate_trajectories(
        static_features=static_features,
        feature_names=feature_names,
        model=model,
        scaler=scaler,
        device=device,
        n_simulations=n_simulations,
        min_age=min_age,
    )

    # Perturbed scenario
    perturbed_features = dict(static_features)
    perturbed_features[perturb_variable] = perturb_value

    perturbed_sim = simulate_trajectories(
        static_features=perturbed_features,
        feature_names=feature_names,
        model=model,
        scaler=scaler,
        device=device,
        n_simulations=n_simulations,
        min_age=min_age,
    )

    # Compute deltas
    base_income_p50 = base_sim.income_percentiles["p50"]
    pert_income_p50 = perturbed_sim.income_percentiles["p50"]
    base_satis_p50 = base_sim.satisfaction_percentiles["p50"]
    pert_satis_p50 = perturbed_sim.satisfaction_percentiles["p50"]

    deltas = []
    for i, age in enumerate(base_sim.ages):
        delta_inc = pert_income_p50[i] - base_income_p50[i]
        delta_sat = pert_satis_p50[i] - base_satis_p50[i]
        deltas.append({
            "age": age,
            "delta_income": delta_inc,
            "delta_satisfaction": delta_sat,
        })

    # Identify inflection points: top 3 ages by absolute income delta
    income_deltas = np.array([d["delta_income"] for d in deltas])
    abs_deltas = np.abs(income_deltas)
    top_indices = abs_deltas.argsort()[-3:][::-1]

    inflection_points = []
    for idx in top_indices:
        if abs_deltas[idx] > 0:
            inflection_points.append(InflectionPoint(
                age=base_sim.ages[idx],
                delta_income=deltas[idx]["delta_income"],
                delta_satisfaction=deltas[idx]["delta_satisfaction"],
                significance=float(abs_deltas[idx] / (abs_deltas.std() + 1e-8)),
            ))

    # Overall effect size: mean absolute income delta
    overall_effect = float(abs_deltas.mean())

    # Archetype assignments
    import torch
    x_base = torch.tensor(
        [[static_features.get(n, 0.0) for n in feature_names]], dtype=torch.float32
    )
    x_pert = torch.tensor(
        [[perturbed_features.get(n, 0.0) for n in feature_names]], dtype=torch.float32
    )
    x_base = scaler.transform_static(x_base).to(device)
    x_pert = scaler.transform_static(x_pert).to(device)

    model.eval()
    with torch.no_grad():
        mu_base, _ = model.encode(x_base)
        mu_pert, _ = model.encode(x_pert)

    base_arch = int(gmm.predict(mu_base.cpu().numpy())[0])
    pert_arch = int(gmm.predict(mu_pert.cpu().numpy())[0])

    return SensitivityResult(
        perturb_variable=perturb_variable,
        perturb_value=perturb_value,
        deltas_by_age=deltas,
        inflection_points=inflection_points,
        overall_effect_size=overall_effect,
        base_archetype=base_arch,
        perturbed_archetype=pert_arch,
    )
