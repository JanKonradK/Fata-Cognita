"""POST /predict route."""

from __future__ import annotations

from fastapi import APIRouter, Request

from fata_cognita.api.schemas.predict import PredictRequest, PredictResponse, TrajectoryPointSchema
from fata_cognita.inference.predictor import predict_trajectory

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict(
    body: PredictRequest,
    request: Request,
) -> PredictResponse:
    """Generate a predicted life trajectory for an individual."""
    state = request.app.state.app_state
    model = state.model
    gmm = state.gmm
    scaler = state.scaler
    device = state.device
    feature_names = state.feature_names

    result = predict_trajectory(
        static_features=body.static_features,
        feature_names=feature_names,
        model=model,
        gmm=gmm,
        scaler=scaler,
        device=device,
        deterministic=body.deterministic,
    )

    trajectory = [
        TrajectoryPointSchema(
            age=t.age,
            life_state=t.life_state,
            life_state_probs=t.life_state_probs,
            income=t.income,
            satisfaction=t.satisfaction,
        )
        for t in result.trajectory
    ]

    return PredictResponse(
        trajectory=trajectory,
        archetype_id=result.archetype_id,
        archetype_membership=result.archetype_membership,
        latent_vector=result.latent_vector,
    )
