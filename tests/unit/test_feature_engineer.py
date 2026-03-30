"""Tests for feature engineering."""

from __future__ import annotations

import numpy as np

from fata_cognita.data.feature_engineer import (
    LifeState,
    adjust_income_for_inflation,
    derive_life_state,
    log_income,
    scale_satisfaction,
)


def test_employed_ft():
    """Full-time employment: employed + hours >= 35."""
    state = derive_life_state(
        employment_status=1, hours_per_week=40, class_of_worker=1,
        enrolled_in_school=0, in_military=0, health_limitation=0,
        age=30, in_labor_force=1,
    )
    assert state == LifeState.EMPLOYED_FT


def test_employed_pt():
    """Part-time employment: employed + hours < 35."""
    state = derive_life_state(
        employment_status=1, hours_per_week=20, class_of_worker=1,
        enrolled_in_school=0, in_military=0, health_limitation=0,
        age=30, in_labor_force=1,
    )
    assert state == LifeState.EMPLOYED_PT


def test_self_employed():
    """Self-employment: class_of_worker == 3."""
    state = derive_life_state(
        employment_status=1, hours_per_week=45, class_of_worker=3,
        enrolled_in_school=0, in_military=0, health_limitation=0,
        age=35, in_labor_force=1,
    )
    assert state == LifeState.SELF_EMPLOYED


def test_student():
    """Student: enrolled_in_school == 1."""
    state = derive_life_state(
        employment_status=0, hours_per_week=0, class_of_worker=None,
        enrolled_in_school=1, in_military=0, health_limitation=0,
        age=20, in_labor_force=0,
    )
    assert state == LifeState.STUDENT


def test_military():
    """Military takes priority."""
    state = derive_life_state(
        employment_status=1, hours_per_week=40, class_of_worker=1,
        enrolled_in_school=0, in_military=1, health_limitation=0,
        age=22, in_labor_force=1,
    )
    assert state == LifeState.MILITARY


def test_retired():
    """Age >= 55 and not employed → retired."""
    state = derive_life_state(
        employment_status=0, hours_per_week=0, class_of_worker=None,
        enrolled_in_school=0, in_military=0, health_limitation=0,
        age=60, in_labor_force=0,
    )
    assert state == LifeState.RETIRED


def test_disabled():
    """Health limitation → disabled."""
    state = derive_life_state(
        employment_status=0, hours_per_week=0, class_of_worker=None,
        enrolled_in_school=0, in_military=0, health_limitation=1,
        age=40, in_labor_force=0,
    )
    assert state == LifeState.DISABLED


def test_log_income():
    """log_income returns log1p of positive values."""
    assert log_income(0) == 0.0
    assert log_income(50000) > 0
    assert np.isnan(log_income(np.nan))


def test_scale_satisfaction():
    """Satisfaction rescaled from 1-4 to 0-1."""
    assert scale_satisfaction(1) == 0.0
    assert scale_satisfaction(4) == 1.0
    assert abs(scale_satisfaction(2.5) - 0.5) < 1e-6
    assert np.isnan(scale_satisfaction(np.nan))


def test_adjust_income_inflation():
    """CPI adjustment produces reasonable values."""
    # 2022 income should stay roughly the same
    adjusted = adjust_income_for_inflation(50000, 2022)
    assert abs(adjusted - 50000) < 1.0

    # 1979 income should be inflated significantly
    adjusted_79 = adjust_income_for_inflation(10000, 1979)
    assert adjusted_79 > 10000
