"""Tests for FLX cortex system — Stratum, DomainCortex, model assembly."""

import torch
import pytest

from flx.model import (
    Stratum,
    DomainCortex,
    SharedTrunk,
    CortexMerger,
    Decoder,
    FLXNano,
)


class TestStratum:
    def test_basic_forward(self):
        stratum = Stratum(d_model=64, nhead=4, num_layers=1, depth="intermediate")
        x = torch.randn(2, 10, 64)
        out = stratum(x, tau=0.5)
        assert out.shape == (2, 10, 64)

    def test_below_threshold_returns_zeros(self):
        stratum = Stratum(d_model=64, nhead=4, num_layers=1, depth="expert", tau_min=0.5)
        x = torch.randn(2, 10, 64)
        out = stratum(x, tau=0.3)
        assert torch.allclose(out, torch.zeros_like(out))

    def test_frontier_high_threshold(self):
        stratum = Stratum(d_model=64, nhead=4, num_layers=1, depth="frontier")
        assert stratum.tau_min == 0.7
        assert stratum.confidence.item() == pytest.approx(0.3)


class TestDomainCortex:
    def test_forward(self):
        cortex = DomainCortex(
            domain_id="math", d_model=64, nhead=4,
            layers_per_stratum=1, dim_feedforward=128,
        )
        x = torch.randn(2, 10, 64)
        out = cortex(x, tau=0.5)
        assert out.shape == (2, 10, 64)

    def test_strata_names(self):
        cortex = DomainCortex(domain_id="code", d_model=64, nhead=4)
        assert set(cortex.stratum_names()) == {"intermediate", "expert", "frontier"}

    def test_low_tau_limits_strata(self):
        cortex = DomainCortex(
            domain_id="math", d_model=64, nhead=4,
            layers_per_stratum=1, dim_feedforward=128,
        )
        x = torch.randn(2, 10, 64)
        # At tau=0.1, only intermediate stratum (tau_min=0.25) might be close
        # but expert (0.5) and frontier (0.7) should not fire
        out_low = cortex(x, tau=0.1)
        out_high = cortex(x, tau=0.9)
        # Both should be valid shapes
        assert out_low.shape == out_high.shape == (2, 10, 64)


class TestSharedTrunk:
    def test_forward(self):
        trunk = SharedTrunk(
            vocab_size=1000, d_model=64, nhead=4,
            num_layers=2, max_seq_len=128, dim_feedforward=128,
        )
        input_ids = torch.randint(0, 1000, (2, 20))
        out = trunk(input_ids)
        assert out.shape == (2, 20, 64)


class TestCortexMerger:
    def test_merge(self):
        merger = CortexMerger(d_model=64)
        trunk_output = torch.randn(2, 10, 64)
        cortex_outputs = {
            "math": torch.randn(2, 10, 64),
            "code": torch.randn(2, 10, 64),
        }
        domain_scores = {
            "math": torch.tensor([0.9, 0.4]),
            "code": torch.tensor([0.3, 0.8]),
        }
        out = merger(cortex_outputs, domain_scores, trunk_output)
        assert out.shape == (2, 10, 64)


class TestDecoder:
    def test_forward(self):
        decoder = Decoder(d_model=64, vocab_size=1000)
        x = torch.randn(2, 10, 64)
        logits = decoder(x)
        assert logits.shape == (2, 10, 1000)


class TestFLXNano:
    def _make_nano(self, **kwargs):
        defaults = dict(
            vocab_size=1000, d_model=64, nhead=4,
            trunk_layers=2, layers_per_stratum=1,
            delta_rank=8, max_seq_len=128,
            dim_feedforward=128,
        )
        defaults.update(kwargs)
        return FLXNano(**defaults)

    def test_forward_no_components(self):
        """Forward pass works even without router/thermal/bridges/memory."""
        model = self._make_nano()
        input_ids = torch.randint(0, 1000, (2, 20))
        logits = model(input_ids)
        assert logits.shape == (2, 20, 1000)

    def test_with_router(self):
        from flx.router import ThalamicRouter

        model = self._make_nano()
        router = ThalamicRouter(d_model=64, cortex_names=model.cortex_names)
        model.attach_router(router)

        input_ids = torch.randint(0, 1000, (2, 20))
        logits = model(input_ids)
        assert logits.shape == (2, 20, 1000)

    def test_with_thermal(self):
        from flx.thermal import ThermalEstimator

        model = self._make_nano()
        thermal = ThermalEstimator(d_model=64)
        model.attach_thermal(thermal)

        input_ids = torch.randint(0, 1000, (2, 20))
        logits = model(input_ids)
        assert logits.shape == (2, 20, 1000)

    def test_count_parameters(self):
        model = self._make_nano()
        counts = model.count_parameters()
        assert "shared_trunk" in counts
        assert "total" in counts
        assert counts["total"] > 0

    def test_forward_deterministic(self):
        model = self._make_nano()
        model.eval()
        input_ids = torch.randint(0, 1000, (1, 10))
        with torch.no_grad():
            out1 = model(input_ids)
            out2 = model(input_ids)
        assert torch.allclose(out1, out2)
