"""Feature engineering: derive life states, scale income, encode categoricals.

Transforms cleaned long-format NLSY data into model-ready features.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import pandas as pd


class LifeState(IntEnum):
    """Life state categories derived from NLSY employment/activity data."""

    EMPLOYED_FT = 0
    EMPLOYED_PT = 1
    SELF_EMPLOYED = 2
    UNEMPLOYED = 3
    OUT_OF_LABOR = 4
    STUDENT = 5
    MILITARY = 6
    RETIRED = 7
    DISABLED = 8


# CPI-U annual averages for deflating to 2022 dollars (selected years)
_CPI_2022 = 292.655
_CPI_TABLE: dict[int, float] = {
    1979: 72.6, 1980: 82.4, 1985: 107.6, 1990: 130.7, 1995: 152.4,
    2000: 172.2, 2005: 195.3, 2010: 218.1, 2015: 237.0, 2020: 258.8,
    2022: 292.655,
}


def _interpolate_cpi(year: int) -> float:
    """Get CPI for a given year via linear interpolation of the CPI table.

    Args:
        year: Calendar year.

    Returns:
        Approximate CPI-U value for that year.
    """
    years = sorted(_CPI_TABLE.keys())
    if year <= years[0]:
        return _CPI_TABLE[years[0]]
    if year >= years[-1]:
        return _CPI_TABLE[years[-1]]

    for i in range(len(years) - 1):
        if years[i] <= year <= years[i + 1]:
            frac = (year - years[i]) / (years[i + 1] - years[i])
            return _CPI_TABLE[years[i]] + frac * (_CPI_TABLE[years[i + 1]] - _CPI_TABLE[years[i]])
    return _CPI_2022


def derive_life_state(
    employment_status: float | None,
    hours_per_week: float | None,
    class_of_worker: float | None,
    enrolled_in_school: float | None,
    in_military: float | None,
    health_limitation: float | None,
    age: float | None,
    in_labor_force: float | None,
) -> int | None:
    """Derive a life state from NLSY indicator variables.

    Args:
        employment_status: 1=employed, 0=not employed, NaN=missing.
        hours_per_week: Weekly work hours.
        class_of_worker: 1=private, 2=govt, 3=self-employed, etc.
        enrolled_in_school: 1=enrolled, 0=not.
        in_military: 1=active military.
        health_limitation: 1=health limits work.
        age: Current age.
        in_labor_force: 1=in LF, 0=not.

    Returns:
        LifeState integer or None if insufficient data.
    """
    if not _is_missing(in_military) and in_military == 1:
        return LifeState.MILITARY

    if not _is_missing(health_limitation) and health_limitation == 1:
        return LifeState.DISABLED

    if not _is_missing(enrolled_in_school) and enrolled_in_school == 1:
        return LifeState.STUDENT

    if not _is_missing(employment_status) and employment_status == 1:
        if not _is_missing(class_of_worker) and class_of_worker == 3:
            return LifeState.SELF_EMPLOYED
        if not _is_missing(hours_per_week) and hours_per_week >= 35:
            return LifeState.EMPLOYED_FT
        return LifeState.EMPLOYED_PT

    # Not employed
    if not _is_missing(in_labor_force) and in_labor_force == 1:
        return LifeState.UNEMPLOYED

    if age is not None and age >= 55:
        return LifeState.RETIRED

    return LifeState.OUT_OF_LABOR


def _is_missing(val: float | None) -> bool:
    """Check if a value is missing (None or NaN).

    Args:
        val: The value to check.

    Returns:
        True if the value is None or NaN.
    """
    if val is None:
        return True
    try:
        return np.isnan(val)
    except (TypeError, ValueError):
        return False


def adjust_income_for_inflation(income: float, year: int) -> float:
    """Convert nominal income to 2022 real dollars.

    Args:
        income: Nominal income value.
        year: Year the income was reported.

    Returns:
        Income in 2022 dollars.
    """
    if _is_missing(income):
        return np.nan
    cpi_year = _interpolate_cpi(year)
    return income * (_CPI_2022 / cpi_year)


def log_income(income: float) -> float:
    """Compute log1p of income for normalization.

    Args:
        income: Real income value (2022 dollars).

    Returns:
        log1p(income), or NaN if input is NaN.
    """
    if _is_missing(income):
        return np.nan
    return float(np.log1p(max(0.0, income)))


def scale_satisfaction(raw_satisfaction: float, scale_min: int = 1, scale_max: int = 4) -> float:
    """Rescale satisfaction from NLSY integer scale to [0, 1].

    Args:
        raw_satisfaction: Raw NLSY satisfaction value.
        scale_min: Minimum value on the original scale.
        scale_max: Maximum value on the original scale.

    Returns:
        Satisfaction in [0, 1], or NaN if input is NaN.
    """
    if _is_missing(raw_satisfaction):
        return np.nan
    return float((raw_satisfaction - scale_min) / (scale_max - scale_min))


def extract_static_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract per-person static features from the long-format DataFrame.

    Takes the first non-NaN value for each person for time-invariant features,
    and applies one-hot encoding for categoricals.

    Args:
        df: Long-format DataFrame with caseid, year, and feature columns.

    Returns:
        DataFrame indexed by caseid with one row per person and columns
        matching the static feature list.
    """
    grouped = df.groupby("caseid")

    static = pd.DataFrame(index=grouped.groups.keys())
    static.index.name = "caseid"

    # Sex
    if "sex" in df.columns:
        static["sex"] = grouped["sex"].first()
    else:
        static["sex"] = 0.0

    # Race (one-hot)
    if "race" in df.columns:
        race = grouped["race"].first()
        static["race_hispanic"] = (race == 1).astype(float)
        static["race_black"] = (race == 2).astype(float)
        static["race_other"] = (race == 3).astype(float)
    else:
        static["race_hispanic"] = 0.0
        static["race_black"] = 0.0
        static["race_other"] = 0.0

    # Birth year (will be normalized later)
    if "birth_year" in df.columns:
        static["birth_year"] = grouped["birth_year"].first()
    else:
        static["birth_year"] = 0.0

    # Parent education
    if "parent_education" in df.columns:
        static["parent_education"] = grouped["parent_education"].first()
    else:
        static["parent_education"] = 0.0

    # Family income at 14
    if "family_income_14" in df.columns:
        static["family_income_14"] = grouped["family_income_14"].first()
    else:
        static["family_income_14"] = 0.0

    # Region (one-hot, 4 categories)
    if "region" in df.columns:
        region = grouped["region"].first()
        static["region_northeast"] = (region == 1).astype(float)
        static["region_north_central"] = (region == 2).astype(float)
        static["region_south"] = (region == 3).astype(float)
        static["region_west"] = (region == 4).astype(float)
    else:
        for r in ["region_northeast", "region_north_central", "region_south", "region_west"]:
            static[r] = 0.0

    # AFQT score
    if "afqt_score" in df.columns:
        afqt = grouped["afqt_score"].first()
        static["afqt_score"] = afqt.fillna(0.0)
        static["afqt_available"] = afqt.notna().astype(float)
    else:
        static["afqt_score"] = 0.0
        static["afqt_available"] = 0.0

    # Cohort
    if "cohort" in df.columns:
        static["cohort"] = grouped["cohort"].first()
    else:
        static["cohort"] = 0.0

    return static.fillna(0.0)
