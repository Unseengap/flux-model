"""Tests for FLX cross-cortical bridges."""

import torch
import pytest

from flx.bridges import CrossCorticalBridge, build_bridges


class TestCrossCorticalBridge:
    def test_forward_shape(self):
        bridge = CrossCorticalBridge(d_model=64, source_cortex="math", target_cortex="code")
        x = torch.randn(2, 10, 64)
        out = bridge(x, tau=0.5)
        assert out.shape == (2, 10, 64)

    def test_below_tau_min_near_zero(self):
        """Bridge output should be near-zero below tau_min."""
        bridge = CrossCorticalBridge(d_model=64, tau_min=0.3)
        x = torch.randn(2, 10, 64)
        out = bridge(x, tau=0.01)
        # sigmoid((0.01 - 0.3) * 10) ≈ 0.054, so output is heavily gated
        assert out.abs().mean() < 0.5

    def test_high_tau_passes_signal(self):
        """Bridge should pass meaningful signal at mid-range τ."""
        bridge = CrossCorticalBridge(d_model=64, tau_min=0.3, tau_max=1.0)
        x = torch.ones(1, 5, 64)
        out_low = bridge(x, tau=0.01)
        out_high = bridge(x, tau=0.6)
        # High-tau output should be stronger
        assert out_high.abs().mean() > out_low.abs().mean()

    def test_above_tau_max_gated(self):
        """Bridge with narrow τ window goes quiet above tau_max."""
        bridge = CrossCorticalBridge(d_model=64, tau_min=0.3, tau_max=0.5)
        x = torch.ones(1, 5, 64)
        out = bridge(x, tau=0.95)
        # sigmoid((0.5 - 0.95) * 10) ≈ 0.01, heavily gated
        assert out.abs().mean() < 0.5

    def test_bandwidth_and_compatibility_learnable(self):
        bridge = CrossCorticalBridge(d_model=64)
        assert bridge.bandwidth.requires_grad
        assert bridge.compatibility.requires_grad

    def test_gradient_flows(self):
        bridge = CrossCorticalBridge(d_model=64)
        x = torch.randn(2, 10, 64, requires_grad=True)
        out = bridge(x, tau=0.5)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert bridge.proj.weight.grad is not None


class TestBuildBridges:
    def test_correct_count(self):
        """N cortices → N*(N-1) bridges (both directions)."""
        names = ["math", "code", "language"]
        bridges = build_bridges(names, d_model=64)
        # 3 cortices → 3*2 = 6 bridges
        assert len(bridges) == 6

    def test_five_cortices(self):
        names = ["language", "math", "code", "science", "reasoning"]
        bridges = build_bridges(names, d_model=64)
        # 5 cortices → 5*4 = 20 bridges
        assert len(bridges) == 20

    def test_bridge_keys_are_source_target(self):
        names = ["math", "code"]
        bridges = build_bridges(names, d_model=64)
        assert "math→code" in bridges
        assert "code→math" in bridges

    def test_bridges_are_modules(self):
        names = ["math", "code"]
        bridges = build_bridges(names, d_model=64)
        for key, bridge in bridges.items():
            assert isinstance(bridge, CrossCorticalBridge)

    def test_all_bridges_forward(self):
        names = ["math", "code", "language"]
        bridges = build_bridges(names, d_model=64)
        x = torch.randn(2, 10, 64)
        for key, bridge in bridges.items():
            out = bridge(x, tau=0.5)
            assert out.shape == (2, 10, 64), f"Bridge {key} output shape wrong"

    def test_parameter_count(self):
        names = ["math", "code"]
        bridges = build_bridges(names, d_model=64)
        total = sum(p.numel() for p in bridges.parameters())
        assert total > 0
