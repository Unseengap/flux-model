"""Tests for FLX kernels (PyTorch fallback) and autograd bridge."""

import torch
import pytest

from flx.kernels import triton_delta_compose, HAS_TRITON
from flx.autograd_bridge import DeltaCompose, delta_compose_autograd
from flx.delta import FLXDelta


class TestTritonDeltaCompose:
    """Tests run against whichever backend is available (Triton or PyTorch fallback)."""

    def test_no_deltas_returns_clone(self):
        W = torch.randn(64, 64)
        result = triton_delta_compose(W, [], [], [], [])
        assert torch.allclose(result, W)

    def test_single_delta(self):
        W = torch.eye(64)
        A = [torch.randn(8, 64)]
        B = [torch.randn(64, 8)]
        conf = [torch.tensor(1.0)]
        scale = [0.5]
        result = triton_delta_compose(W, A, B, conf, scale)
        expected = W + conf[0] * scale[0] * (B[0] @ A[0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_multiple_deltas(self):
        W = torch.eye(32)
        A = [torch.randn(4, 32), torch.randn(4, 32)]
        B = [torch.randn(32, 4), torch.randn(32, 4)]
        conf = [torch.tensor(0.8), torch.tensor(0.5)]
        scale = [0.25, 0.25]
        result = triton_delta_compose(W, A, B, conf, scale)
        expected = W.clone()
        for a, b, c, s in zip(A, B, conf, scale):
            expected = expected + c.clamp(0, 1) * s * (b @ a)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_zero_confidence_no_effect(self):
        W = torch.eye(32)
        A = [torch.randn(4, 32)]
        B = [torch.randn(32, 4)]
        conf = [torch.tensor(0.0)]
        scale = [1.0]
        result = triton_delta_compose(W, A, B, conf, scale)
        assert torch.allclose(result, W, atol=1e-6)


class TestDeltaComposeAutograd:
    def test_forward_matches_reference(self):
        W = torch.eye(32)
        d1 = FLXDelta(d_in=32, d_out=32, rank=4, thermal_threshold=0.0, confidence=0.8)
        d1.B.data.normal_(0, 0.1)

        result = delta_compose_autograd(W, [d1], tau=0.5)
        expected = W + d1.compute()
        assert torch.allclose(result, expected, atol=1e-5)

    def test_inactive_delta_skipped(self):
        W = torch.eye(32)
        d1 = FLXDelta(d_in=32, d_out=32, rank=4, thermal_threshold=0.8)
        d1.B.data.normal_(0, 0.1)

        result = delta_compose_autograd(W, [d1], tau=0.3)
        assert torch.allclose(result, W)

    def test_gradient_to_A(self):
        W = torch.eye(32, requires_grad=False)
        d1 = FLXDelta(d_in=32, d_out=32, rank=4, thermal_threshold=0.0)
        d1.B.data.normal_(0, 0.1)

        result = delta_compose_autograd(W, [d1], tau=0.5)
        loss = result.sum()
        loss.backward()
        assert d1.A.grad is not None

    def test_gradient_to_B(self):
        W = torch.eye(32, requires_grad=False)
        d1 = FLXDelta(d_in=32, d_out=32, rank=4, thermal_threshold=0.0)
        d1.B.data.normal_(0, 0.1)

        result = delta_compose_autograd(W, [d1], tau=0.5)
        loss = result.sum()
        loss.backward()
        assert d1.B.grad is not None

    def test_gradient_to_confidence(self):
        W = torch.eye(32, requires_grad=False)
        d1 = FLXDelta(d_in=32, d_out=32, rank=4, thermal_threshold=0.0)
        d1.B.data.normal_(0, 0.1)

        result = delta_compose_autograd(W, [d1], tau=0.5)
        loss = result.sum()
        loss.backward()
        assert d1.confidence.grad is not None

    def test_gradient_to_W_base(self):
        W = torch.eye(32, requires_grad=True)
        d1 = FLXDelta(d_in=32, d_out=32, rank=4, thermal_threshold=0.0)
        d1.B.data.normal_(0, 0.1)

        result = delta_compose_autograd(W, [d1], tau=0.5)
        loss = result.sum()
        loss.backward()
        assert W.grad is not None

    def test_multiple_deltas_gradients(self):
        W = torch.eye(32, requires_grad=True)
        d1 = FLXDelta(d_in=32, d_out=32, rank=4, thermal_threshold=0.0)
        d2 = FLXDelta(d_in=32, d_out=32, rank=4, thermal_threshold=0.0)
        d1.B.data.normal_(0, 0.1)
        d2.B.data.normal_(0, 0.1)

        result = delta_compose_autograd(W, [d1, d2], tau=0.5)
        loss = result.sum()
        loss.backward()
        assert d1.A.grad is not None
        assert d2.A.grad is not None
        assert W.grad is not None

    def test_empty_deltas_returns_base(self):
        W = torch.eye(32)
        result = delta_compose_autograd(W, [], tau=0.5)
        assert torch.allclose(result, W)
