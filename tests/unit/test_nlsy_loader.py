"""Tests for the NLSY data loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fata_cognita.data.nlsy_loader import (
    clean_sentinels,
    load_nlsy_csv,
    wide_to_long,
)


@pytest.fixture()
def tiny_csv_path() -> Path:
    return Path("tests/fixtures/tiny_nlsy_sample.csv")


def test_load_nlsy_csv(tiny_csv_path: Path):
    """CSV loads with CASEID as index."""
    df = load_nlsy_csv(tiny_csv_path)
    assert df.index.name == "caseid"
    assert len(df) == 4


def test_clean_sentinels(tiny_csv_path: Path):
    """Negative sentinels (-1 through -5) are replaced with NaN."""
    df = load_nlsy_csv(tiny_csv_path)
    df = clean_sentinels(df)

    # CASEID 2 had income.1 = -1, should now be NaN
    assert np.isnan(df.loc[2, "income.1"])
    # CASEID 3 had income.2 = -4, should now be NaN
    assert np.isnan(df.loc[3, "income.2"])
    # Valid values should be preserved
    assert df.loc[1, "income.1"] == 15000


def test_wide_to_long(tiny_csv_path: Path):
    """Wide-to-long reshaping produces correct structure."""
    df = load_nlsy_csv(tiny_csv_path)
    df = clean_sentinels(df)
    long_df = wide_to_long(df, cohort="nlsy79")

    # Should have caseid, year, round columns
    assert "caseid" in long_df.columns
    assert "year" in long_df.columns
    assert "round" in long_df.columns

    # Should have variable columns (without round suffix)
    assert "income" in long_df.columns

    # 4 individuals * num_rounds rows
    assert len(long_df) > 0
    assert long_df["caseid"].nunique() == 4


def test_missing_file_raises():
    """Loading nonexistent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_nlsy_csv("nonexistent.csv")
