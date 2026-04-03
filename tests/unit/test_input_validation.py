"""Tests for API input validation (Pydantic schema validators)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fata_cognita.api.schemas.inflection import InflectionRequest
from fata_cognita.api.schemas.predict import PredictRequest
from fata_cognita.api.schemas.simulate import SimulateRequest


class TestPredictRequestValidation:
    def test_valid_features(self):
        req = PredictRequest(static_features={"sex": 1.0, "age": 25.0})
        assert req.static_features["sex"] == 1.0

    def test_nan_feature_rejected(self):
        with pytest.raises(ValidationError, match="non-finite"):
            PredictRequest(static_features={"sex": float("nan")})

    def test_inf_feature_rejected(self):
        with pytest.raises(ValidationError, match="non-finite"):
            PredictRequest(static_features={"sex": float("inf")})

    def test_neg_inf_feature_rejected(self):
        with pytest.raises(ValidationError, match="non-finite"):
            PredictRequest(static_features={"sex": float("-inf")})


class TestSimulateRequestValidation:
    def test_valid_request(self):
        req = SimulateRequest(static_features={"sex": 1.0}, n_simulations=500)
        assert req.n_simulations == 500

    def test_nan_feature_rejected(self):
        with pytest.raises(ValidationError, match="non-finite"):
            SimulateRequest(static_features={"x": float("nan")})

    def test_simulations_below_min(self):
        with pytest.raises(ValidationError):
            SimulateRequest(static_features={"x": 1.0}, n_simulations=10)

    def test_simulations_above_max(self):
        with pytest.raises(ValidationError):
            SimulateRequest(static_features={"x": 1.0}, n_simulations=100000)


class TestInflectionRequestValidation:
    def test_valid_request(self):
        req = InflectionRequest(
            static_features={"sex": 1.0},
            perturb_variable="sex",
            perturb_value=0.0,
        )
        assert req.perturb_variable == "sex"

    def test_nan_perturb_value_rejected(self):
        with pytest.raises(ValidationError, match="finite"):
            InflectionRequest(
                static_features={"sex": 1.0},
                perturb_variable="sex",
                perturb_value=float("nan"),
            )

    def test_nan_feature_rejected(self):
        with pytest.raises(ValidationError, match="non-finite"):
            InflectionRequest(
                static_features={"sex": float("inf")},
                perturb_variable="sex",
                perturb_value=0.0,
            )

    def test_simulations_capped_at_10k(self):
        with pytest.raises(ValidationError):
            InflectionRequest(
                static_features={"x": 1.0},
                perturb_variable="x",
                perturb_value=0.0,
                n_simulations=50000,
            )
