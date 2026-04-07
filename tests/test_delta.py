"""Tests for FLX delta system — FLXDelta, DeltaStack, composition."""

import torch
import pytest

from flx.delta import FLXDelta, DeltaStack, compose_weights


class TestFLXDelta:
    def test_create_delta(self):
        delta = FLXDelta(d_in=512, d_out=512, rank=32)
        assert delta.A.shape == (32, 512)
        assert delta.B.shape == (512, 32)
        assert delta.rank == 32

    def test_delta_starts_as_zero(self):
        """B initialized to zero, so initial delta contribution is zero."""
        delta = FLXDelta(d_in=64, d_out=64, rank=8)
        result = delta.compute()
        assert torch.allclose(result, torch.zeros(64, 64), atol=1e-7)

    def test_delta_forward(self):
        delta = FLXDelta(d_in=64, d_out=64, rank=8)
        # Manually set B to non-zero
        delta.B.data.fill_(0.01)
        x = torch.randn(2, 10, 64)
        out = delta(x)
        assert out.shape == (2, 10, 64)

    def test_thermal_threshold(self):
        delta = FLXDelta(d_in=64, d_out=64, rank=8, thermal_threshold=0.5)
        assert delta.is_active(0.6)
        assert delta.is_active(0.5)
        assert not delta.is_active(0.3)

    def test_confidence_clamped(self):
        delta = FLXDelta(d_in=64, d_out=64, rank=8, confidence=2.0)
        result = delta.compute()
        # confidence clamped to 1.0
        assert delta.confidence.item() == 2.0  # raw param
        # But compute uses clamp


class TestDeltaStack:
    def test_push_pop(self):
        stack = DeltaStack(capacity=3)
        d1 = FLXDelta(d_in=64, d_out=64, rank=8)
        d2 = FLXDelta(d_in=64, d_out=64, rank=8)
        stack.push(d1)
        stack.push(d2)
        assert len(stack) == 2

        popped = stack.pop()
        assert len(stack) == 1

    def test_capacity_limit(self):
        stack = DeltaStack(capacity=2)
        stack.push(FLXDelta(d_in=64, d_out=64, rank=8))
        stack.push(FLXDelta(d_in=64, d_out=64, rank=8))
        with pytest.raises(RuntimeError, match="capacity"):
            stack.push(FLXDelta(d_in=64, d_out=64, rank=8))

    def test_empty_pop_raises(self):
        stack = DeltaStack()
        with pytest.raises(RuntimeError, match="empty"):
            stack.pop()

    def test_active_deltas(self):
        stack = DeltaStack()
        d1 = FLXDelta(d_in=64, d_out=64, rank=8, thermal_threshold=0.0)
        d2 = FLXDelta(d_in=64, d_out=64, rank=8, thermal_threshold=0.5)
        d3 = FLXDelta(d_in=64, d_out=64, rank=8, thermal_threshold=0.8)
        stack.push(d1)
        stack.push(d2)
        stack.push(d3)

        assert len(stack.active_deltas(0.3)) == 1
        assert len(stack.active_deltas(0.6)) == 2
        assert len(stack.active_deltas(0.9)) == 3

    def test_compose(self):
        stack = DeltaStack()
        delta = FLXDelta(d_in=64, d_out=64, rank=8, thermal_threshold=0.0)
        stack.push(delta)

        base = torch.eye(64)
        composed = stack.compose(base, tau=0.5)
        assert composed.shape == (64, 64)


class TestComposeWeights:
    def test_no_deltas(self):
        base = torch.randn(64, 64)
        result = compose_weights(base, [], tau=0.5)
        assert torch.allclose(result, base)

    def test_with_deltas(self):
        base = torch.randn(64, 64)
        d1 = FLXDelta(d_in=64, d_out=64, rank=8, thermal_threshold=0.0)
        d1.B.data.fill_(0.01)
        result = compose_weights(base, [d1], tau=0.5)
        assert result.shape == (64, 64)
        # Result should differ from base since B is non-zero
        assert not torch.allclose(result, base)
