"""Tests for FLX MetaDeltaGenerator."""

import torch
import pytest

from flx.meta_gen import MetaDeltaGenerator
from flx.delta import FLXDelta


class TestMetaDeltaGenerator:
    def _make_gen(self, d_model=64, rank=8):
        return MetaDeltaGenerator(
            d_model=d_model, delta_rank=rank,
            num_cortices=3, num_strata=3,
            nhead=4, num_layers=1,
        )

    def test_forward_shapes(self):
        gen = self._make_gen()
        errors = torch.randn(2, 10, 64)
        A, B, meta = gen(errors)
        assert A.shape == (2, 8, 64)
        assert B.shape == (2, 64, 8)
        assert meta["cortex_logits"].shape == (2, 3)
        assert meta["stratum_logits"].shape == (2, 3)
        assert meta["threshold"].shape == (2,)

    def test_threshold_in_01(self):
        gen = self._make_gen()
        errors = torch.randn(1, 5, 64)
        _, _, meta = gen(errors)
        t = meta["threshold"]
        assert (t >= 0).all() and (t <= 1).all()

    def test_with_stack_summary(self):
        gen = self._make_gen()
        errors = torch.randn(2, 10, 64)
        stack_summary = torch.randn(2, 64)
        A, B, meta = gen(errors, stack_summary)
        assert A.shape == (2, 8, 64)

    def test_without_stack_summary(self):
        gen = self._make_gen()
        errors = torch.randn(2, 10, 64)
        A, B, meta = gen(errors, stack_summary=None)
        assert A.shape == (2, 8, 64)

    def test_output_matrices_small_scale(self):
        """Generated A/B should be small (0.01 scaling)."""
        gen = self._make_gen()
        errors = torch.randn(1, 10, 64)
        A, B, _ = gen(errors)
        # 0.01 scaling means values should be small
        assert A.abs().mean() < 1.0
        assert B.abs().mean() < 1.0

    def test_gradient_flows(self):
        gen = self._make_gen()
        errors = torch.randn(1, 5, 64)
        A, B, meta = gen(errors)
        # Include all outputs so all heads get gradients
        loss = A.sum() + B.sum() + meta["threshold"].sum() + meta["cortex_logits"].sum() + meta["stratum_logits"].sum()
        loss.backward()
        assert gen.A_head.weight.grad is not None
        assert gen.cortex_head.weight.grad is not None
        assert gen.stratum_head.weight.grad is not None


class TestGenerateDelta:
    def _make_gen(self, d_model=64, rank=8):
        return MetaDeltaGenerator(
            d_model=d_model, delta_rank=rank,
            num_cortices=3, num_strata=3,
            nhead=4, num_layers=1,
        )

    def test_returns_flx_delta(self):
        gen = self._make_gen()
        errors = torch.randn(5, 64)  # unbatched
        delta = gen.generate_delta(errors)
        assert isinstance(delta, FLXDelta)
        assert delta.d_in == 64
        assert delta.d_out == 64
        assert delta.rank == 8

    def test_probationary_confidence(self):
        gen = self._make_gen()
        errors = torch.randn(5, 64)
        delta = gen.generate_delta(errors)
        assert delta.confidence.item() == pytest.approx(0.1)

    def test_metadata_populated(self):
        gen = self._make_gen()
        cortex_names = ["math", "code", "language"]
        stratum_names = ["intermediate", "expert", "frontier"]
        errors = torch.randn(5, 64)
        delta = gen.generate_delta(
            errors, cortex_names=cortex_names, stratum_names=stratum_names,
        )
        assert delta.metadata.source == "meta_gen"
        assert delta.metadata.target_cortex in cortex_names
        assert delta.metadata.target_stratum in stratum_names

    def test_batched_input(self):
        gen = self._make_gen()
        errors = torch.randn(1, 5, 64)  # already batched
        delta = gen.generate_delta(errors)
        assert isinstance(delta, FLXDelta)

    def test_with_stack_summary(self):
        gen = self._make_gen()
        errors = torch.randn(5, 64)
        stack_summary = torch.randn(64)
        delta = gen.generate_delta(errors, stack_summary=stack_summary)
        assert isinstance(delta, FLXDelta)
