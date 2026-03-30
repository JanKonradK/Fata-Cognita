"""Pydantic schemas for the /predict endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request body for trajectory prediction."""

    static_features: dict[str, float] = Field(
        ..., description="Mapping of feature names to values"
    )
    deterministic: bool = Field(
        True, description="Use deterministic (mu) or stochastic (sampled) prediction"
    )


class TrajectoryPointSchema(BaseModel):
    """A single time-step in a trajectory."""

    age: int
    life_state: str
    life_state_probs: dict[str, float]
    income: float
    satisfaction: float


class PredictResponse(BaseModel):
    """Response body for trajectory prediction."""

    trajectory: list[TrajectoryPointSchema]
    archetype_id: int
    archetype_membership: dict[int, float]
    latent_vector: list[float]
