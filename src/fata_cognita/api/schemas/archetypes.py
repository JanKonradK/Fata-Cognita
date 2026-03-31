"""Pydantic schemas for the /archetypes endpoints."""

from __future__ import annotations

from pydantic import BaseModel

from fata_cognita.api.schemas.predict import TrajectoryPointSchema  # noqa: TC001


class ArchetypeSummary(BaseModel):
    """Summary info for one archetype."""

    id: int
    prevalence: float
    member_count: int
    demographic_profile: dict[str, float]
    median_peak_income: float
    dominant_life_state: str


class ArchetypeListResponse(BaseModel):
    """Response body for GET /archetypes."""

    archetypes: list[ArchetypeSummary]
    total_individuals: int
    k_selected: int


class ArchetypeTrajectoryResponse(BaseModel):
    """Response body for GET /archetypes/{id}/trajectory."""

    archetype_id: int
    canonical_trajectory: list[TrajectoryPointSchema]
    income_percentiles: dict[str, list[float]]
    member_count: int
