"""Integration tests for API error handling."""

from __future__ import annotations

from fastapi.testclient import TestClient

from fata_cognita.api.deps import AppState
from fata_cognita.api.main import create_app


class TestArtifactNotLoaded:
    def test_predict_500_when_model_missing(self):
        """API returns error when model artifacts are not loaded."""
        state = AppState()  # empty state — no model loaded
        app = create_app(state=state)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/api/v1/predict",
                json={"static_features": {"sex": 1.0}},
            )
            assert resp.status_code >= 500


class TestNaNInputRejected:
    def test_predict_422_nan_feature(self):
        """API returns 422 when NaN feature value is submitted."""
        state = AppState()
        app = create_app(state=state)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/api/v1/predict",
                json={"static_features": {"sex": "NaN"}},
            )
            # NaN as string won't parse as float, or if it does, validator catches it
            assert resp.status_code == 422
