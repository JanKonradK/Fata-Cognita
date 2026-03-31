"""POST /simulate route."""

from __future__ import annotations

from fastapi import APIRouter, Request

from fata_cognita.api.schemas.simulate import (
    PercentileBandsSchema,
    SimulateRequest,
    SimulateResponse,
    StateDistributionSchema,
)
from fata_cognita.inference.simulator import simulate_trajectories

router = APIRouter()


@router.post("/simulate", response_model=SimulateResponse)
def simulate(body: SimulateRequest, request: Request) -> SimulateResponse:
    """Run Monte Carlo trajectory simulation."""
    state = request.app.state.app_state

    result = simulate_trajectories(
        static_features=body.static_features,
        feature_names=state.feature_names,
        model=state.model,
        scaler=state.scaler,
        device=state.device,
        n_simulations=body.n_simulations,
        percentiles=body.percentiles,
    )

    # Assign archetype
    import torch

    x = torch.tensor(
        [[body.static_features.get(n, 0.0) for n in state.feature_names]],
        dtype=torch.float32,
    )
    x = state.scaler.transform_static(x).to(state.device)
    with torch.no_grad():
        mu, _ = state.model.encode(x)
    archetype_id = int(state.gmm.predict(mu.cpu().numpy())[0])

    return SimulateResponse(
        percentile_bands=PercentileBandsSchema(
            age=result.ages,
            income=result.income_percentiles,
            satisfaction=result.satisfaction_percentiles,
        ),
        state_distribution=StateDistributionSchema(
            age=result.ages,
            probabilities=result.state_distribution,
        ),
        archetype_id=archetype_id,
        n_simulations=result.n_simulations,
    )
