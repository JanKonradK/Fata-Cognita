"""Pydantic schemas for the /inflection-points endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class InflectionRequest(BaseModel):
    """Request body for inflection-point analysis."""

    static_features: dict[str, float] = Field(..., description="Base person features")
    perturb_variable: str = Field(..., description="Feature name to perturb")
    perturb_value: float = Field(..., description="New value for the perturbed feature")
    n_simulations: int = Field(10000, ge=100, le=50000, description="MC trajectories per scenario")


class InflectionPointSchema(BaseModel):
    """A detected inflection point."""

    age: int
    delta_income: float
    delta_satisfaction: float
    significance: float


class InflectionResponse(BaseModel):
    """Response body for inflection-point analysis."""

    perturb_variable: str
    perturb_value: float
    deltas_by_age: list[dict[str, float]]
    inflection_points: list[InflectionPointSchema]
    overall_effect_size: float
    base_archetype: int
    perturbed_archetype: int
