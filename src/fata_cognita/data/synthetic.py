"""Synthetic data generator for end-to-end pipeline testing.

Produces fake longitudinal data matching the NLSY-derived tensor schema,
using Markov chains for life-state transitions and correlated continuous
outcomes. No real data required.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import torch

from fata_cognita.config import Config, DataConfig, SyntheticConfig


class LifeState(IntEnum):
    """Enumeration of possible life states."""

    EMPLOYED_FT = 0
    EMPLOYED_PT = 1
    SELF_EMPLOYED = 2
    UNEMPLOYED = 3
    OUT_OF_LABOR = 4
    STUDENT = 5
    MILITARY = 6
    RETIRED = 7
    DISABLED = 8


# Transition matrix: rows = from-state, cols = to-state.
# Probabilities are age-independent defaults; age-dependent adjustments applied in code.
_BASE_TRANSITION = np.array([
    # FT    PT    SE    UE    OLF   STU   MIL   RET   DIS
    [0.85, 0.04, 0.02, 0.04, 0.02, 0.01, 0.00, 0.01, 0.01],  # EMPLOYED_FT
    [0.10, 0.65, 0.02, 0.08, 0.08, 0.04, 0.00, 0.02, 0.01],  # EMPLOYED_PT
    [0.05, 0.03, 0.80, 0.05, 0.03, 0.01, 0.00, 0.02, 0.01],  # SELF_EMPLOYED
    [0.25, 0.10, 0.02, 0.40, 0.15, 0.05, 0.01, 0.01, 0.01],  # UNEMPLOYED
    [0.10, 0.10, 0.02, 0.10, 0.55, 0.05, 0.01, 0.05, 0.02],  # OUT_OF_LABOR
    [0.30, 0.15, 0.02, 0.08, 0.05, 0.35, 0.02, 0.01, 0.02],  # STUDENT
    [0.30, 0.05, 0.02, 0.05, 0.05, 0.03, 0.45, 0.03, 0.02],  # MILITARY
    [0.02, 0.02, 0.01, 0.01, 0.04, 0.00, 0.00, 0.88, 0.02],  # RETIRED
    [0.03, 0.02, 0.01, 0.02, 0.05, 0.01, 0.00, 0.02, 0.84],  # DISABLED
], dtype=np.float64)


def _age_adjusted_transition(age: int) -> np.ndarray:
    """Adjust transition probabilities based on age.

    Args:
        age: Current age of the individual.

    Returns:
        A (9, 9) transition matrix adjusted for the given age.
    """
    tm = _BASE_TRANSITION.copy()

    # Young people more likely to be students
    if age < 22:
        tm[:, LifeState.STUDENT] *= 2.0
        tm[:, LifeState.RETIRED] *= 0.01

    # Retirement increases after 55
    if age >= 55:
        tm[:, LifeState.RETIRED] *= 1.0 + (age - 55) * 0.15
        tm[:, LifeState.STUDENT] *= 0.1

    # Normalize rows to sum to 1
    row_sums = tm.sum(axis=1, keepdims=True)
    tm = tm / row_sums
    return tm


def _income_for_state(
    state: int, age: int, education: float, rng: np.random.Generator
) -> float:
    """Generate a plausible income given life state, age, and education.

    Args:
        state: Life state integer.
        age: Current age.
        education: Years of education (normalized ~0-1 scale).
        rng: NumPy random generator.

    Returns:
        Log1p income value (pre-scaling).
    """
    base_by_state = {
        LifeState.EMPLOYED_FT: 10.5,
        LifeState.EMPLOYED_PT: 9.8,
        LifeState.SELF_EMPLOYED: 10.2,
        LifeState.UNEMPLOYED: 7.0,
        LifeState.OUT_OF_LABOR: 6.5,
        LifeState.STUDENT: 7.5,
        LifeState.MILITARY: 10.0,
        LifeState.RETIRED: 9.5,
        LifeState.DISABLED: 8.0,
    }
    base = base_by_state.get(state, 9.0)

    # Age effect: income peaks around 50
    age_effect = 0.03 * (age - 18) - 0.0005 * (age - 18) ** 2
    edu_effect = 0.8 * education
    noise = rng.normal(0, 0.3)

    return max(0.0, base + age_effect + edu_effect + noise)


def _satisfaction_for_state(
    state: int, income: float, rng: np.random.Generator
) -> float:
    """Generate a plausible life satisfaction score.

    Args:
        state: Life state integer.
        income: Log income value.
        rng: NumPy random generator.

    Returns:
        Satisfaction score in [0, 1].
    """
    base_by_state = {
        LifeState.EMPLOYED_FT: 0.65,
        LifeState.EMPLOYED_PT: 0.55,
        LifeState.SELF_EMPLOYED: 0.60,
        LifeState.UNEMPLOYED: 0.30,
        LifeState.OUT_OF_LABOR: 0.40,
        LifeState.STUDENT: 0.55,
        LifeState.MILITARY: 0.50,
        LifeState.RETIRED: 0.60,
        LifeState.DISABLED: 0.35,
    }
    base = base_by_state.get(state, 0.5)

    # Income has a small positive effect on satisfaction
    income_effect = 0.02 * (income - 9.0)
    noise = rng.normal(0, 0.08)

    return float(np.clip(base + income_effect + noise, 0.0, 1.0))


def generate_synthetic_data(config: Config | None = None) -> dict[str, torch.Tensor]:
    """Generate a complete synthetic dataset matching the NLSY tensor schema.

    Args:
        config: Optional Config object. Uses defaults if None.

    Returns:
        Dictionary with keys: static_features, life_states, income,
        satisfaction, masks — all as PyTorch tensors with shapes matching
        the real data pipeline output.
    """
    if config is None:
        config = Config()

    data_cfg: DataConfig = config.data
    syn_cfg: SyntheticConfig = data_cfg.synthetic
    rng = np.random.default_rng(syn_cfg.seed)

    n = syn_cfg.n_individuals
    seq_len = data_cfg.max_seq_len
    n_features = data_cfg.num_static_features

    # --- Static features ---
    static = np.zeros((n, n_features), dtype=np.float32)
    for i in range(n):
        col = 0
        # sex (binary)
        static[i, col] = rng.choice([0.0, 1.0])
        col += 1
        # race one-hot (3 categories: hispanic, black, other)
        race = rng.choice(3)
        static[i, col + race] = 1.0
        col += 3
        # birth_year (normalized)
        static[i, col] = rng.normal(0.0, 1.0)
        col += 1
        # parent_education (normalized)
        static[i, col] = rng.normal(0.0, 1.0)
        col += 1
        # family_income_14 (normalized)
        static[i, col] = rng.normal(0.0, 1.0)
        col += 1
        # region one-hot (4 regions)
        region = rng.choice(4)
        static[i, col + region] = 1.0
        col += 4
        # afqt_score (normalized)
        static[i, col] = rng.normal(0.0, 1.0)
        col += 1
        # afqt_available (binary indicator)
        static[i, col] = rng.choice([0.0, 1.0], p=[0.3, 0.7])
        col += 1
        # cohort (binary)
        static[i, col] = rng.choice([0.0, 1.0])

    # --- Sequences ---
    life_states = np.zeros((n, seq_len), dtype=np.int64)
    income = np.zeros((n, seq_len), dtype=np.float32)
    satisfaction = np.zeros((n, seq_len), dtype=np.float32)
    masks = np.ones((n, seq_len), dtype=bool)

    for i in range(n):
        education = float(np.clip(static[i, 5] * 0.5 + 0.5, 0.0, 1.0))  # proxy

        # Initial state: most start as students at age 14
        current_state = LifeState.STUDENT

        for t in range(seq_len):
            age = 14 + t

            # Transition
            if t > 0:
                tm = _age_adjusted_transition(age)
                probs = tm[current_state]
                current_state = LifeState(rng.choice(len(LifeState), p=probs))

            life_states[i, t] = int(current_state)
            inc = _income_for_state(int(current_state), age, education, rng)
            income[i, t] = inc
            satisfaction[i, t] = _satisfaction_for_state(int(current_state), inc, rng)

        # Apply random missingness
        n_missing = int(seq_len * syn_cfg.missing_rate)
        missing_indices = rng.choice(seq_len, size=n_missing, replace=False)
        masks[i, missing_indices] = False

    return {
        "static_features": torch.from_numpy(static),
        "life_states": torch.from_numpy(life_states),
        "income": torch.from_numpy(income),
        "satisfaction": torch.from_numpy(satisfaction),
        "masks": torch.from_numpy(masks),
    }
