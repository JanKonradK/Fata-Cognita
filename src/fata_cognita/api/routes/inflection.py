"""POST /inflection-points route."""

from __future__ import annotations

from fastapi import APIRouter, Request

from fata_cognita.api.schemas.inflection import (
    InflectionPointSchema,
    InflectionRequest,
    InflectionResponse,
)
from fata_cognita.inference.sensitivity import run_sensitivity_analysis

router = APIRouter()


@router.post("/inflection-points", response_model=InflectionResponse)
def inflection_points(body: InflectionRequest, request: Request) -> InflectionResponse:
    """Run counterfactual sensitivity analysis."""
    state = request.app.state.app_state

    result = run_sensitivity_analysis(
        static_features=body.static_features,
        perturb_variable=body.perturb_variable,
        perturb_value=body.perturb_value,
        feature_names=state.feature_names,
        model=state.model,
        gmm=state.gmm,
        scaler=state.scaler,
        device=state.device,
        n_simulations=body.n_simulations,
    )

    inflection_pts = [
        InflectionPointSchema(
            age=ip.age,
            delta_income=ip.delta_income,
            delta_satisfaction=ip.delta_satisfaction,
            significance=ip.significance,
        )
        for ip in result.inflection_points
    ]

    return InflectionResponse(
        perturb_variable=result.perturb_variable,
        perturb_value=result.perturb_value,
        deltas_by_age=result.deltas_by_age,
        inflection_points=inflection_pts,
        overall_effect_size=result.overall_effect_size,
        base_archetype=result.base_archetype,
        perturbed_archetype=result.perturbed_archetype,
    )
