"""Pydantic schemas for the /inflection-points endpoint."""

from __future__ import annotations

import math

from pydantic import BaseModel, Field, model_validator


class InflectionRequest(BaseModel):
    """Request body for inflection-point analysis."""

    static_features: dict[str, float] = Field(..., description="Base person features")
    perturb_variable: str = Field(..., description="Feature name to perturb")
    perturb_value: float = Field(..., description="New value for the perturbed feature")
    n_simulations: int = Field(10000, ge=100, le=10000, description="MC trajectories per scenario")

    @model_validator(mode="after")
    def validate_feature_values(self) -> InflectionRequest:
        """Ensure all feature values are finite numbers."""
        for name, value in self.static_features.items():
            if not math.isfinite(value):
                msg = f"Feature '{name}' has non-finite value: {value}"
                raise ValueError(msg)
        if not math.isfinite(self.perturb_value):
            msg = f"perturb_value must be finite, got {self.perturb_value}"
            raise ValueError(msg)
        return self


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
