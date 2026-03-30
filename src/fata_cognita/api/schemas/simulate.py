"""Pydantic schemas for the /simulate endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SimulateRequest(BaseModel):
    """Request body for Monte Carlo simulation."""

    static_features: dict[str, float] = Field(
        ..., description="Mapping of feature names to values"
    )
    n_simulations: int = Field(
        1000, ge=100, le=10000, description="Number of trajectories to sample"
    )
    percentiles: list[int] = Field(
        default=[10, 25, 50, 75, 90], description="Percentiles to compute"
    )


class PercentileBandsSchema(BaseModel):
    """Percentile bands for a continuous outcome."""

    age: list[int]
    income: dict[str, list[float]]
    satisfaction: dict[str, list[float]]


class StateDistributionSchema(BaseModel):
    """Per-age probability distribution over life states."""

    age: list[int]
    probabilities: list[dict[str, float]]


class SimulateResponse(BaseModel):
    """Response body for Monte Carlo simulation."""

    percentile_bands: PercentileBandsSchema
    state_distribution: StateDistributionSchema
    archetype_id: int
    n_simulations: int
