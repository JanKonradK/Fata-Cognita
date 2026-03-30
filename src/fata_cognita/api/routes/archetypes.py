"""GET /archetypes and GET /archetypes/{id}/trajectory routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from fata_cognita.api.schemas.archetypes import (
    ArchetypeListResponse,
    ArchetypeSummary,
    ArchetypeTrajectoryResponse,
)
from fata_cognita.api.schemas.predict import TrajectoryPointSchema

router = APIRouter()


@router.get("/archetypes", response_model=ArchetypeListResponse)
def list_archetypes(request: Request) -> ArchetypeListResponse:
    """List all discovered archetypes with profiles."""
    state = request.app.state.app_state
    profiles = state.archetype_profiles or []

    summaries = []
    for p in profiles:
        traj = p.get("canonical_trajectory", {})
        income_vals = traj.get("income", [])
        peak_income = max(income_vals) if income_vals else 0.0

        states = traj.get("life_states", [])
        if states:
            from collections import Counter
            dominant = Counter(states).most_common(1)[0][0]
            from fata_cognita.data.synthetic import LifeState
            dominant_name = LifeState(dominant).name
        else:
            dominant_name = "UNKNOWN"

        summaries.append(ArchetypeSummary(
            id=p["archetype_id"],
            prevalence=p["prevalence"],
            member_count=p["member_count"],
            demographic_profile=p.get("feature_means", {}),
            median_peak_income=peak_income,
            dominant_life_state=dominant_name,
        ))

    return ArchetypeListResponse(
        archetypes=summaries,
        total_individuals=sum(p["member_count"] for p in profiles),
        k_selected=len(profiles),
    )


@router.get("/archetypes/{archetype_id}/trajectory", response_model=ArchetypeTrajectoryResponse)
def archetype_trajectory(archetype_id: int, request: Request) -> ArchetypeTrajectoryResponse:
    """Get canonical trajectory for a specific archetype."""
    state = request.app.state.app_state
    profiles = state.archetype_profiles or []

    profile = None
    for p in profiles:
        if p["archetype_id"] == archetype_id:
            profile = p
            break

    if profile is None:
        raise HTTPException(status_code=404, detail=f"Archetype {archetype_id} not found")

    traj = profile["canonical_trajectory"]
    min_age = 14
    from fata_cognita.data.synthetic import LifeState

    trajectory_points = []
    for t in range(len(traj["income"])):
        state_idx = traj["life_states"][t]
        trajectory_points.append(TrajectoryPointSchema(
            age=min_age + t,
            life_state=LifeState(state_idx).name,
            life_state_probs={},
            income=traj["income"][t],
            satisfaction=traj["satisfaction"][t],
        ))

    return ArchetypeTrajectoryResponse(
        archetype_id=archetype_id,
        canonical_trajectory=trajectory_points,
        income_percentiles={},
        member_count=profile["member_count"],
    )
