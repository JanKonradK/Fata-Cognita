"""NLSY data loader: raw CSV ingestion, wide-to-long reshaping, and cleaning.

Handles both NLSY79 and NLSY97 data exported from the NLS Investigator tool.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# NLSY uses negative sentinel values for missing data
MISSING_SENTINELS = {-1, -2, -3, -4, -5}

# Round-to-year mappings
NLSY79_ROUND_YEARS: dict[int, int] = {
    **{r: 1978 + r for r in range(1, 17)},  # rounds 1-16: 1979-1994 (annual)
    **{r: 1994 + (r - 16) * 2 for r in range(17, 31)},  # rounds 17-30: 1996-2022 (biennial)
}

NLSY97_ROUND_YEARS: dict[int, int] = {
    **{r: 1996 + r for r in range(1, 16)},  # rounds 1-15: 1997-2011 (annual)
    **{r: 2011 + (r - 15) * 2 for r in range(16, 21)},  # rounds 16-20: 2013-2021 (biennial)
}


def _detect_cohort(columns: list[str]) -> str:
    """Heuristic to detect whether data is NLSY79 or NLSY97.

    Args:
        columns: Column names from the CSV.

    Returns:
        'nlsy79' or 'nlsy97'.
    """
    col_str = " ".join(columns).upper()
    if "NLSY97" in col_str or "Y97" in col_str:
        return "nlsy97"
    return "nlsy79"


def _get_round_years(cohort: str) -> dict[int, int]:
    """Return the round-to-year mapping for the given cohort.

    Args:
        cohort: Either 'nlsy79' or 'nlsy97'.

    Returns:
        Dictionary mapping round numbers to survey years.
    """
    if cohort == "nlsy97":
        return NLSY97_ROUND_YEARS
    return NLSY79_ROUND_YEARS


def load_nlsy_csv(path: str | Path) -> pd.DataFrame:
    """Load a raw NLSY CSV file and return a cleaned wide-format DataFrame.

    Args:
        path: Path to the CSV file from NLS Investigator.

    Returns:
        DataFrame with CASEID as index and original columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NLSY data file not found: {path}")

    df = pd.read_csv(path)

    # Normalize column names: strip whitespace, uppercase
    df.columns = [c.strip() for c in df.columns]

    # Set CASEID as index if present
    caseid_col = [c for c in df.columns if c.upper() in ("CASEID", "CASE_ID", "R0000100")]
    if caseid_col:
        df = df.set_index(caseid_col[0])
        df.index.name = "caseid"

    return df


def clean_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NLSY negative sentinel values with NaN.

    Args:
        df: DataFrame potentially containing sentinel values (-1 through -5).

    Returns:
        DataFrame with sentinels replaced by NaN.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].where(~df[col].isin(MISSING_SENTINELS), other=np.nan)
    return df


def wide_to_long(
    df: pd.DataFrame,
    variable_map: dict[str, list[str]] | None = None,
    cohort: str | None = None,
) -> pd.DataFrame:
    """Reshape wide NLSY data to long format (one row per person-year).

    Expects columns in the pattern ``VARNAME.ROUND`` or explicitly mapped.

    Args:
        df: Wide-format NLSY DataFrame with CASEID as index.
        variable_map: Optional dict mapping target variable names to lists
            of column names (one per round). If None, infers from column
            naming patterns.
        cohort: 'nlsy79' or 'nlsy97'. Auto-detected if None.

    Returns:
        Long-format DataFrame with columns: caseid, year, round, and
        one column per variable.
    """
    if cohort is None:
        cohort = _detect_cohort(list(df.columns))

    round_years = _get_round_years(cohort)

    if variable_map is not None:
        return _reshape_with_map(df, variable_map, round_years)

    return _reshape_by_pattern(df, round_years)


def _reshape_by_pattern(df: pd.DataFrame, round_years: dict[int, int]) -> pd.DataFrame:
    """Reshape by detecting ``VARNAME.ROUND`` column patterns.

    Args:
        df: Wide DataFrame.
        round_years: Round-to-year mapping.

    Returns:
        Long-format DataFrame.
    """
    # Group columns by variable name (part before last dot)
    var_rounds: dict[str, dict[int, str]] = {}
    static_cols: list[str] = []

    for col in df.columns:
        if "." in col:
            parts = col.rsplit(".", 1)
            var_name = parts[0]
            try:
                round_num = int(parts[1])
                var_rounds.setdefault(var_name, {})[round_num] = col
            except ValueError:
                static_cols.append(col)
        else:
            static_cols.append(col)

    if not var_rounds:
        raise ValueError(
            "No round-varying columns found (expected pattern VARNAME.ROUND). "
            "Provide an explicit variable_map."
        )

    records = []
    for caseid in df.index:
        for round_num, year in round_years.items():
            row: dict = {"caseid": caseid, "year": year, "round": round_num}
            for var_name, round_cols in var_rounds.items():
                if round_num in round_cols:
                    row[var_name] = df.at[caseid, round_cols[round_num]]
                else:
                    row[var_name] = np.nan
            records.append(row)

    long_df = pd.DataFrame(records)
    return long_df


def _reshape_with_map(
    df: pd.DataFrame,
    variable_map: dict[str, list[str]],
    round_years: dict[int, int],
) -> pd.DataFrame:
    """Reshape using an explicit variable-to-columns mapping.

    Args:
        df: Wide DataFrame.
        variable_map: Dict mapping variable names to ordered lists of column names.
        round_years: Round-to-year mapping.

    Returns:
        Long-format DataFrame.
    """
    rounds_sorted = sorted(round_years.keys())

    records = []
    for caseid in df.index:
        for i, round_num in enumerate(rounds_sorted):
            year = round_years[round_num]
            row: dict = {"caseid": caseid, "year": year, "round": round_num}
            for var_name, col_list in variable_map.items():
                if i < len(col_list) and col_list[i] in df.columns:
                    row[var_name] = df.at[caseid, col_list[i]]
                else:
                    row[var_name] = np.nan
            records.append(row)

    return pd.DataFrame(records)


def compute_age(df: pd.DataFrame, birth_year_col: str = "birth_year") -> pd.DataFrame:
    """Add an 'age' column computed from year and birth_year.

    Args:
        df: Long-format DataFrame with 'year' and birth_year columns.
        birth_year_col: Name of the birth year column.

    Returns:
        DataFrame with 'age' column added.
    """
    if birth_year_col not in df.columns:
        raise ValueError(f"Column '{birth_year_col}' not found in DataFrame")

    df = df.copy()
    df["age"] = df["year"] - df[birth_year_col]
    return df
