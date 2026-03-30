"""Tests for the cyclical beta annealing schedule."""

from __future__ import annotations

import pytest

from fata_cognita.model.beta_schedule import CyclicalBetaSchedule


def test_beta_starts_at_zero():
    """Beta is 0 at step 0."""
    sched = CyclicalBetaSchedule(total_steps=100, n_cycles=4, ratio=0.5)
    assert sched.get_beta() == 0.0


def test_beta_reaches_one():
    """Beta reaches 1.0 at the end of the ramp-up phase."""
    sched = CyclicalBetaSchedule(total_steps=100, n_cycles=4, ratio=0.5)
    # Each cycle = 25 steps, ramp-up = 12.5 steps
    # At step 12 (within ramp), beta should be close to but not quite 1
    # At step 13, beta should be >= 1 (or very close)
    for _ in range(13):
        sched.step()
    assert sched.get_beta() >= 0.95


def test_beta_holds_after_ramp():
    """Beta stays at 1.0 after ramp-up within a cycle."""
    sched = CyclicalBetaSchedule(total_steps=100, n_cycles=4, ratio=0.5)
    # Advance to step 20 (well past ramp in first cycle)
    for _ in range(20):
        sched.step()
    assert sched.get_beta() == 1.0


def test_beta_resets_at_new_cycle():
    """Beta resets to near-zero at the start of a new cycle."""
    sched = CyclicalBetaSchedule(total_steps=100, n_cycles=4, ratio=0.5)
    # Advance to step 25 (start of second cycle)
    for _ in range(25):
        sched.step()
    # At the boundary, beta should reset
    assert sched.get_beta() < 0.1


def test_invalid_params():
    """Invalid parameters raise ValueError."""
    with pytest.raises(ValueError):
        CyclicalBetaSchedule(total_steps=0)
    with pytest.raises(ValueError):
        CyclicalBetaSchedule(total_steps=100, n_cycles=0)
    with pytest.raises(ValueError):
        CyclicalBetaSchedule(total_steps=100, ratio=0.0)


def test_reset():
    """Reset brings schedule back to step 0."""
    sched = CyclicalBetaSchedule(total_steps=100)
    for _ in range(50):
        sched.step()
    sched.reset()
    assert sched.current_step == 0
    assert sched.get_beta() == 0.0
