"""Tests for FLX thermal system — ThermalEstimator."""

import torch
import pytest

from flx.thermal import ThermalEstimator, count_active_flops


class TestThermalEstimator:
    def test_forward(self):
        estimator = ThermalEstimator(d_model=64)
        x = torch.randn(2, 10, 64)
        tau = estimator(x)
        assert tau.shape == (2,)
        assert (tau >= 0).all() and (tau <= 1).all()

    def test_history_tracking(self):
        estimator = ThermalEstimator(d_model=64)
        x = torch.randn(2, 10, 64)

        estimator(x)
        estimator(x)
        estimator(x)

        history = estimator.get_history()
        assert len(history) == 3

    def test_history_reset(self):
        estimator = ThermalEstimator(d_model=64)
        x = torch.randn(2, 10, 64)
        estimator(x)
        estimator.reset_history()
        assert len(estimator.get_history()) == 0

    def test_history_persistence(self):
        estimator = ThermalEstimator(d_model=64)
        x = torch.randn(2, 10, 64)
        estimator(x)
        estimator(x)

        history = estimator.get_history()
        estimator.reset_history()
        estimator.set_history(history)
        assert len(estimator.get_history()) == len(history)


class TestCountActiveFlops:
    def test_low_compute(self):
        cost = count_active_flops(tau=0.1, num_strata_active=1, num_bridges_active=0, num_loops=0)
        assert cost == 1.0

    def test_high_compute(self):
        cost = count_active_flops(tau=0.9, num_strata_active=4, num_bridges_active=3, num_loops=2)
        assert cost > 10
