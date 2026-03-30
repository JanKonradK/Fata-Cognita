"""Sequence builder: construct age-aligned trajectory tensors.

Converts long-format DataFrames into fixed-length tensor sequences
aligned by age, with padding and observation masks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from fata_cognita.config import DataConfig


def build_sequences(
    df: pd.DataFrame,
    config: DataConfig,
    life_state_col: str = "life_state",
    income_col: str = "log_income",
    satisfaction_col: str = "satisfaction",
    age_col: str = "age",
) -> dict[str, torch.Tensor]:
    """Build age-aligned tensor sequences from long-format data.

    Args:
        df: Long-format DataFrame with caseid, age, and outcome columns.
        config: DataConfig with max_seq_len and related settings.
        life_state_col: Column name for life state integers.
        income_col: Column name for log income.
        satisfaction_col: Column name for satisfaction (0-1).
        age_col: Column name for age.

    Returns:
        Dictionary with tensors: life_states (N, S), income (N, S),
        satisfaction (N, S), masks (N, S), caseids (N,).
    """
    min_age = 14
    max_age = min_age + config.max_seq_len - 1
    seq_len = config.max_seq_len

    caseids = sorted(df["caseid"].unique())
    n = len(caseids)
    caseid_to_idx = {cid: i for i, cid in enumerate(caseids)}

    life_states = np.zeros((n, seq_len), dtype=np.int64)
    income = np.zeros((n, seq_len), dtype=np.float32)
    satisfaction = np.zeros((n, seq_len), dtype=np.float32)
    masks = np.zeros((n, seq_len), dtype=bool)

    for _, row in df.iterrows():
        cid = row["caseid"]
        age = row.get(age_col)
        if age is None or np.isnan(age):
            continue

        age = int(age)
        if age < min_age or age > max_age:
            continue

        idx = caseid_to_idx[cid]
        t = age - min_age

        # Life state
        ls = row.get(life_state_col)
        if ls is not None and not (isinstance(ls, float) and np.isnan(ls)):
            life_states[idx, t] = int(ls)
            masks[idx, t] = True

        # Income
        inc = row.get(income_col)
        if inc is not None and not (isinstance(inc, float) and np.isnan(inc)):
            income[idx, t] = float(inc)

        # Satisfaction
        sat = row.get(satisfaction_col)
        if sat is not None and not (isinstance(sat, float) and np.isnan(sat)):
            satisfaction[idx, t] = float(sat)

    return {
        "life_states": torch.from_numpy(life_states),
        "income": torch.from_numpy(income),
        "satisfaction": torch.from_numpy(satisfaction),
        "masks": torch.from_numpy(masks),
        "caseids": torch.tensor(caseids, dtype=torch.int64),
    }


def split_by_caseid(
    caseids: torch.Tensor,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Split case IDs into train/val/test sets.

    Args:
        caseids: Tensor of unique case IDs.
        train_frac: Fraction for training.
        val_frac: Fraction for validation.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with 'train', 'val', 'test' keys mapping to index lists.
    """
    rng = np.random.default_rng(seed)
    n = len(caseids)
    indices = rng.permutation(n)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    return {
        "train": indices[:n_train].tolist(),
        "val": indices[n_train:n_train + n_val].tolist(),
        "test": indices[n_train + n_val:].tolist(),
    }
