"""Tests for inference transform utilities."""

from __future__ import annotations

import torch

from fata_cognita.inference.transforms import inverse_income_to_nominal


class TestInverseIncomeToNominal:
    def test_zero_input(self):
        result = inverse_income_to_nominal(torch.tensor([0.0]))
        assert result.item() == 0.0

    def test_positive_log_income(self):
        log_val = torch.tensor([1.0])
        result = inverse_income_to_nominal(log_val)
        expected = torch.expm1(log_val).item()
        assert abs(result.item() - expected) < 1e-6

    def test_negative_clamped_to_zero(self):
        result = inverse_income_to_nominal(torch.tensor([-5.0]))
        assert result.item() == 0.0

    def test_batch(self):
        batch = torch.tensor([-1.0, 0.0, 1.0, 5.0])
        result = inverse_income_to_nominal(batch)
        assert result.shape == (4,)
        assert result[0].item() == 0.0
        assert result[1].item() == 0.0
        assert result[2].item() > 0.0
        assert result[3].item() > result[2].item()

    def test_non_negative_output(self):
        inputs = torch.randn(100)
        result = inverse_income_to_nominal(inputs)
        assert (result >= 0).all()
