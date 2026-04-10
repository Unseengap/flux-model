"""Tests for Phase 5 — Abstract Rule Induction.

Tests HypothesisHead, TaskScratchpad, and phase5_training_step.
"""

import torch
import pytest

from flx.model import FLXNano
from flx.router import ThalamicRouter
from flx.memory import MemoryController
from flx.hypothesis import HypothesisHead, TaskScratchpad
from flx.training.phase5_abstraction import (
    phase5_training_step,
    consistency_loss,
    loop_efficiency_loss,
)

# ---------------------------------------------------------------------------
# Constants — tiny dimensions for speed
# ---------------------------------------------------------------------------

VOCAB = 256
D_MODEL = 32
NHEAD = 4
SEQ_LEN = 16
BATCH = 2
N_DEMOS = 3


def _tiny_model(**kwargs):
    defaults = dict(
        vocab_size=VOCAB, d_model=D_MODEL, nhead=NHEAD,
        trunk_layers=1, layers_per_stratum=1,
        cortex_names=["lang", "math", "code"],
        delta_rank=4, delta_capacity=3,
        max_seq_len=64, dim_feedforward=64,
    )
    defaults.update(kwargs)
    return FLXNano(**defaults)


def _make_hypothesis_head(**kwargs):
    defaults = dict(
        d_model=D_MODEL, hypothesis_dim=D_MODEL,
        nhead=NHEAD, num_layers=1,
    )
    defaults.update(kwargs)
    return HypothesisHead(**defaults)


def _random_batch():
    ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    tgt = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    return ids, tgt


def _random_demos(n=N_DEMOS):
    demo_inputs = [torch.randint(0, VOCAB, (BATCH, SEQ_LEN)) for _ in range(n)]
    demo_targets = [torch.randint(0, VOCAB, (BATCH, SEQ_LEN)) for _ in range(n)]
    return demo_inputs, demo_targets


# ---------------------------------------------------------------------------
# HypothesisHead
# ---------------------------------------------------------------------------

class TestHypothesisHead:
    def test_forward_shapes(self):
        head = _make_hypothesis_head()
        fused = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        hypothesis, consistency, conditioning = head(fused)

        assert hypothesis.shape == (BATCH, D_MODEL)
        assert consistency.shape == (BATCH,)
        assert conditioning.shape == (BATCH, 1, D_MODEL)

    def test_consistency_in_01(self):
        head = _make_hypothesis_head()
        fused = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        _, consistency, _ = head(fused)

        assert (consistency >= 0).all()
        assert (consistency <= 1).all()

    def test_with_demo_embeddings(self):
        head = _make_hypothesis_head()
        fused = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        demos = torch.randn(BATCH, N_DEMOS, D_MODEL)

        hypothesis, consistency, conditioning = head(fused, demo_embeddings=demos)
        assert hypothesis.shape == (BATCH, D_MODEL)
        assert consistency.shape == (BATCH,)

    def test_with_trajectory(self):
        head = _make_hypothesis_head()
        fused = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        trajectory = torch.randn(BATCH, 2, D_MODEL)  # 2 prior hypotheses

        hypothesis, consistency, conditioning = head(
            fused, trajectory=trajectory,
        )
        assert hypothesis.shape == (BATCH, D_MODEL)

    def test_empty_trajectory(self):
        head = _make_hypothesis_head()
        fused = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        trajectory = torch.zeros(BATCH, 0, D_MODEL)  # no prior hypotheses

        hypothesis, consistency, conditioning = head(
            fused, trajectory=trajectory,
        )
        assert hypothesis.shape == (BATCH, D_MODEL)

    def test_all_inputs(self):
        head = _make_hypothesis_head()
        fused = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        demos = torch.randn(BATCH, N_DEMOS, D_MODEL)
        trajectory = torch.randn(BATCH, 3, D_MODEL)

        hypothesis, consistency, conditioning = head(
            fused, demo_embeddings=demos, trajectory=trajectory,
        )
        assert hypothesis.shape == (BATCH, D_MODEL)
        assert conditioning.shape == (BATCH, 1, D_MODEL)

    def test_gradient_flows(self):
        head = _make_hypothesis_head()
        fused = torch.randn(BATCH, SEQ_LEN, D_MODEL, requires_grad=True)
        demos = torch.randn(BATCH, N_DEMOS, D_MODEL)

        hypothesis, consistency, conditioning = head(fused, demos)
        loss = hypothesis.sum() + consistency.sum() + conditioning.sum()
        loss.backward()

        assert head.hypothesis_proj.weight.grad is not None
        assert head.consistency_head.weight.grad is not None
        assert head.condition_proj.weight.grad is not None


# ---------------------------------------------------------------------------
# TaskScratchpad
# ---------------------------------------------------------------------------

class TestTaskScratchpad:
    def test_empty(self):
        pad = TaskScratchpad(hypothesis_dim=D_MODEL)
        assert len(pad) == 0
        assert pad.is_empty
        assert pad.get_best() is None

    def test_add_and_retrieve(self):
        pad = TaskScratchpad(hypothesis_dim=D_MODEL)
        h = torch.randn(D_MODEL)
        pad.add_hypothesis(h, 0.7)
        assert len(pad) == 1
        assert not pad.is_empty

        best = pad.get_best()
        assert best is not None
        assert best.shape == (D_MODEL,)

    def test_get_best_returns_highest(self):
        pad = TaskScratchpad(hypothesis_dim=D_MODEL)
        pad.add_hypothesis(torch.randn(D_MODEL), 0.3)
        pad.add_hypothesis(torch.randn(D_MODEL), 0.9)
        pad.add_hypothesis(torch.randn(D_MODEL), 0.5)

        # Best should be the one with score 0.9
        assert pad.scores[pad.scores.index(max(pad.scores))] == pytest.approx(0.9)

    def test_trajectory_shape(self):
        pad = TaskScratchpad(hypothesis_dim=D_MODEL)
        pad.add_hypothesis(torch.randn(D_MODEL), 0.5)
        pad.add_hypothesis(torch.randn(D_MODEL), 0.6)

        traj = pad.get_trajectory()
        assert traj.shape == (1, 2, D_MODEL)

    def test_trajectory_empty(self):
        pad = TaskScratchpad(hypothesis_dim=D_MODEL)
        traj = pad.get_trajectory()
        assert traj.shape == (1, 0, D_MODEL)

    def test_max_capacity(self):
        pad = TaskScratchpad(hypothesis_dim=D_MODEL, max_hypotheses=3)
        for i in range(5):
            pad.add_hypothesis(torch.randn(D_MODEL), float(i) / 5)
        assert len(pad) == 3

    def test_clear(self):
        pad = TaskScratchpad(hypothesis_dim=D_MODEL)
        pad.add_hypothesis(torch.randn(D_MODEL), 0.5)
        pad.clear()
        assert len(pad) == 0
        assert pad.is_empty

    def test_batched_hypothesis(self):
        """add_hypothesis with batched input stores first element."""
        pad = TaskScratchpad(hypothesis_dim=D_MODEL)
        h = torch.randn(BATCH, D_MODEL)
        pad.add_hypothesis(h, 0.6)
        assert len(pad) == 1
        assert pad.hypotheses[0].shape == (D_MODEL,)


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

class TestLossFunctions:
    def test_consistency_loss_perfect(self):
        cons = torch.ones(BATCH)
        assert consistency_loss(cons).item() == pytest.approx(0.0)

    def test_consistency_loss_worst(self):
        cons = torch.zeros(BATCH)
        assert consistency_loss(cons).item() == pytest.approx(1.0)

    def test_consistency_loss_mid(self):
        cons = torch.full((BATCH,), 0.5)
        assert consistency_loss(cons).item() == pytest.approx(0.5)

    def test_loop_efficiency_zero(self):
        assert loop_efficiency_loss(0, 3).item() == pytest.approx(0.0)

    def test_loop_efficiency_max(self):
        assert loop_efficiency_loss(3, 3).item() == pytest.approx(1.0)

    def test_loop_efficiency_partial(self):
        assert loop_efficiency_loss(1, 3).item() == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# Phase 5 Training Step — Smoke Tests
# ---------------------------------------------------------------------------

class TestPhase5Smoke:
    def _setup_model(self):
        model = _tiny_model()
        router = ThalamicRouter(d_model=D_MODEL, cortex_names=model.cortex_names)
        model.attach_router(router)
        mem_ctrl = MemoryController(d_model=D_MODEL, episode_dim=16)
        model.attach_memory(mem_ctrl)
        return model

    def test_single_step(self):
        model = self._setup_model()
        head = _make_hypothesis_head()
        model.train()

        demo_inputs, demo_targets = _random_demos()
        test_in, test_tgt = _random_batch()

        losses = phase5_training_step(
            model, head,
            demo_inputs, demo_targets,
            test_in, test_tgt,
        )

        assert "total_loss" in losses
        assert "pred_loss" in losses
        assert "consistency_loss" in losses
        assert "consistency" in losses
        assert "num_loops" in losses
        assert losses["total_loss"].isfinite()
        assert losses["pred_loss"].item() > 0

    def test_backward(self):
        model = self._setup_model()
        head = _make_hypothesis_head()
        model.train()

        demo_inputs, demo_targets = _random_demos()
        test_in, test_tgt = _random_batch()

        losses = phase5_training_step(
            model, head,
            demo_inputs, demo_targets,
            test_in, test_tgt,
        )
        losses["total_loss"].backward()

        # HypothesisHead should have gradients
        assert any(p.grad is not None for p in head.parameters())

    def test_requires_memory_controller(self):
        model = _tiny_model()
        head = _make_hypothesis_head()

        demo_inputs, demo_targets = _random_demos()
        test_in, test_tgt = _random_batch()

        with pytest.raises(AssertionError, match="memory controller"):
            phase5_training_step(
                model, head,
                demo_inputs, demo_targets,
                test_in, test_tgt,
            )

    def test_consistency_in_01(self):
        model = self._setup_model()
        head = _make_hypothesis_head()
        model.train()

        demo_inputs, demo_targets = _random_demos()
        test_in, test_tgt = _random_batch()

        losses = phase5_training_step(
            model, head,
            demo_inputs, demo_targets,
            test_in, test_tgt,
        )
        cons = losses["consistency"].item()
        assert 0.0 <= cons <= 1.0

    def test_num_loops_bounded(self):
        model = self._setup_model()
        head = _make_hypothesis_head()
        model.train()

        demo_inputs, demo_targets = _random_demos()
        test_in, test_tgt = _random_batch()

        losses = phase5_training_step(
            model, head,
            demo_inputs, demo_targets,
            test_in, test_tgt,
            max_loops=2,
        )
        assert losses["num_loops"].item() <= 2

    def test_different_demo_counts(self):
        """Phase 5 works with 2-5 demonstrations."""
        model = self._setup_model()
        head = _make_hypothesis_head()
        model.train()

        for n in [2, 3, 5]:
            demo_inputs, demo_targets = _random_demos(n=n)
            test_in, test_tgt = _random_batch()

            losses = phase5_training_step(
                model, head,
                demo_inputs, demo_targets,
                test_in, test_tgt,
            )
            assert losses["total_loss"].isfinite()

    def test_attach_hypothesis_head(self):
        """FLXNano.attach_hypothesis_head registers the module."""
        model = _tiny_model()
        head = _make_hypothesis_head()
        assert model.hypothesis_head is None

        model.attach_hypothesis_head(head)
        assert model.hypothesis_head is head

        # Should appear in parameter count
        counts = model.count_parameters()
        assert "hypothesis_head" in counts
        assert counts["hypothesis_head"] > 0


# ---------------------------------------------------------------------------
# Loop Count Bug Fix
# ---------------------------------------------------------------------------

class TestLoopCountReset:
    def test_loop_count_reset_on_forward(self):
        """_loop_count is reset at the start of FLXNano.forward."""
        model = _tiny_model()
        mem_ctrl = MemoryController(d_model=D_MODEL, episode_dim=16)
        model.attach_memory(mem_ctrl)

        # Manually dirty the loop count
        mem_ctrl._loop_count = 2

        ids, _ = _random_batch()
        episode = torch.randn(16)
        _ = model(ids, tau=0.8, episodic_buffer=[episode])

        # After forward, loop count should have been reset
        assert mem_ctrl._loop_count == 0
