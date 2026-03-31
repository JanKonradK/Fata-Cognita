"""Inference: prediction, Monte Carlo simulation, and sensitivity analysis."""

from fata_cognita.inference.predictor import PredictionResult, TrajectoryPoint, predict_trajectory
from fata_cognita.inference.sensitivity import (
    InflectionPoint,
    SensitivityResult,
    run_sensitivity_analysis,
)
from fata_cognita.inference.simulator import SimulationResult, simulate_trajectories

__all__ = [
    "InflectionPoint",
    "PredictionResult",
    "SensitivityResult",
    "SimulationResult",
    "TrajectoryPoint",
    "predict_trajectory",
    "run_sensitivity_analysis",
    "simulate_trajectories",
]
