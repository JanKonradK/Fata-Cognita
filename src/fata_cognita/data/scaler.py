"""Feature scaling utilities for income and static features.

Wraps scikit-learn scalers with save/load for reproducibility.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """Manages scaling for static features and income.

    Fits on training data only and applies transforms consistently
    to validation and test sets.

    Args:
        fit_static: Whether to scale static features.
        fit_income: Whether to scale log income.
    """

    def __init__(self, fit_static: bool = True, fit_income: bool = True) -> None:
        self.fit_static = fit_static
        self.fit_income = fit_income
        self._static_scaler: StandardScaler | None = None
        self._income_scaler: StandardScaler | None = None

    def fit(
        self,
        static_features: torch.Tensor | None = None,
        income: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
    ) -> None:
        """Fit scalers on training data.

        Args:
            static_features: Static features tensor (N, F).
            income: Log income tensor (N, S).
            masks: Observation masks (N, S). Only observed values used for fitting.
        """
        if self.fit_static and static_features is not None:
            self._static_scaler = StandardScaler()
            self._static_scaler.fit(static_features.detach().cpu().numpy())

        if self.fit_income and income is not None:
            self._income_scaler = StandardScaler()
            if masks is not None:
                observed = income[masks].detach().cpu().numpy().reshape(-1, 1)
            else:
                observed = income.detach().cpu().numpy().reshape(-1, 1)
            self._income_scaler.fit(observed)

    def transform_static(self, static_features: torch.Tensor) -> torch.Tensor:
        """Scale static features using the fitted scaler.

        Args:
            static_features: Tensor (N, F).

        Returns:
            Scaled tensor (N, F).
        """
        if self._static_scaler is None:
            return static_features
        scaled = self._static_scaler.transform(static_features.detach().cpu().numpy())
        return torch.from_numpy(scaled.astype(np.float32))

    def transform_income(self, income: torch.Tensor) -> torch.Tensor:
        """Scale log income values.

        Args:
            income: Tensor (N, S).

        Returns:
            Scaled tensor (N, S).
        """
        if self._income_scaler is None:
            return income
        shape = income.shape
        flat = income.detach().cpu().numpy().reshape(-1, 1)
        scaled = self._income_scaler.transform(flat).reshape(shape)
        return torch.from_numpy(scaled.astype(np.float32))

    def inverse_income(self, income: torch.Tensor) -> torch.Tensor:
        """Inverse-transform scaled income back to log scale.

        Args:
            income: Scaled tensor.

        Returns:
            Original-scale log income tensor.
        """
        if self._income_scaler is None:
            return income
        shape = income.shape
        flat = income.detach().cpu().numpy().reshape(-1, 1)
        original = self._income_scaler.inverse_transform(flat).reshape(shape)
        return torch.from_numpy(original.astype(np.float32))

    def save(self, path: str | Path) -> None:
        """Save fitted scalers to a pickle file.

        Args:
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "static_scaler": self._static_scaler,
            "income_scaler": self._income_scaler,
            "fit_static": self.fit_static,
            "fit_income": self.fit_income,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str | Path) -> FeatureScaler:
        """Load a previously saved FeatureScaler.

        Args:
            path: Path to the pickle file.

        Returns:
            FeatureScaler with restored state.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)  # noqa: S301
        scaler = cls(fit_static=state["fit_static"], fit_income=state["fit_income"])
        scaler._static_scaler = state["static_scaler"]
        scaler._income_scaler = state["income_scaler"]
        return scaler
