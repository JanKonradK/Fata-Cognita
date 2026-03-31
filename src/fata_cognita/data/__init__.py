"""Data loading, preprocessing, and synthetic generation pipeline."""

from fata_cognita.data.dataset import TrajectoryDataset, create_dataloaders
from fata_cognita.data.feature_engineer import (
    adjust_income_for_inflation,
    derive_life_state,
    extract_static_features,
    log_income,
    scale_satisfaction,
)
from fata_cognita.data.nlsy_loader import clean_sentinels, compute_age, load_nlsy_csv, wide_to_long
from fata_cognita.data.scaler import FeatureScaler
from fata_cognita.data.sequence_builder import build_sequences, split_by_caseid
from fata_cognita.data.synthetic import LifeState, generate_synthetic_data

__all__ = [
    "FeatureScaler",
    "LifeState",
    "TrajectoryDataset",
    "adjust_income_for_inflation",
    "build_sequences",
    "clean_sentinels",
    "compute_age",
    "create_dataloaders",
    "derive_life_state",
    "extract_static_features",
    "generate_synthetic_data",
    "load_nlsy_csv",
    "log_income",
    "scale_satisfaction",
    "split_by_caseid",
    "wide_to_long",
]
