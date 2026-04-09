"""Smoke tests for all 5 training phases.

Each test creates a tiny model, runs 1-2 training steps, and verifies
the step completes without error and produces valid loss values.
These catch regressions in the 1,087 lines of training code.
"""

import torch
import pytest

from flx.model import FLXNano
from flx.router import ThalamicRouter
from flx.thermal import ThermalEstimator
from flx.bridges import build_bridges
from flx.memory import EpisodicCompressor, MemoryController
from flx.meta_gen import MetaDeltaGenerator
from flx.training.phase0_cortex import phase0_training_step
from flx.training.phase1_delta import phase1_training_step, train_phase1, _init_delta_pool
from flx.training.phase2_thermal import phase2_training_step
from flx.training.phase3_memory import phase3_training_step
from flx.training.phase4_meta import phase4_training_step, ErrorBuffer


# ---------------------------------------------------------------------------
# Tiny model factory — shared across all phase tests
# ---------------------------------------------------------------------------

VOCAB = 256
D_MODEL = 32
NHEAD = 4
SEQ_LEN = 16
BATCH = 2


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


def _random_batch():
    ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    tgt = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    return ids, tgt


# ---------------------------------------------------------------------------
# Phase 0 — Cortex Specialization
# ---------------------------------------------------------------------------

class TestPhase0Smoke:
    def test_single_step(self):
        model = _tiny_model()
        router = ThalamicRouter(d_model=D_MODEL, cortex_names=model.cortex_names)
        model.attach_router(router)
        model.train()

        ids, tgt = _random_batch()
        losses = phase0_training_step(model, ids, tgt)

        assert "total_loss" in losses
        assert "pred_loss" in losses
        assert "div_loss" in losses
        assert "bal_loss" in losses
        assert losses["total_loss"].isfinite()
        assert losses["pred_loss"].item() > 0

    def test_backward(self):
        model = _tiny_model()
        router = ThalamicRouter(d_model=D_MODEL, cortex_names=model.cortex_names)
        model.attach_router(router)
        model.train()

        ids, tgt = _random_batch()
        losses = phase0_training_step(model, ids, tgt)
        losses["total_loss"].backward()

        # Router should have gradients
        assert any(p.grad is not None for p in router.parameters())

    def test_requires_router(self):
        model = _tiny_model()
        ids, tgt = _random_batch()
        with pytest.raises(AssertionError, match="thalamic router"):
            phase0_training_step(model, ids, tgt)

    def test_dropout_routing_no_crash(self):
        """Dropout routing with prob=1.0 should still work."""
        model = _tiny_model()
        router = ThalamicRouter(d_model=D_MODEL, cortex_names=model.cortex_names)
        model.attach_router(router)
        model.train()

        ids, tgt = _random_batch()
        losses = phase0_training_step(model, ids, tgt, dropout_top_prob=1.0)
        assert losses["total_loss"].isfinite()


# ---------------------------------------------------------------------------
# Phase 1 — Delta-Receptive Pretraining
# ---------------------------------------------------------------------------

class TestPhase1Smoke:
    def test_single_step(self):
        model = _tiny_model()
        router = ThalamicRouter(d_model=D_MODEL, cortex_names=model.cortex_names)
        model.attach_router(router)
        _init_delta_pool(model, pool_size=2)
        model.train()

        ids, tgt = _random_batch()
        losses = phase1_training_step(model, ids, tgt, tau=0.5)

        assert "pred_loss" in losses
        assert losses["pred_loss"].isfinite()
        assert losses["pred_loss"].item() > 0

    def test_backward(self):
        model = _tiny_model()
        _init_delta_pool(model, pool_size=2)
        model.train()

        ids, tgt = _random_batch()
        losses = phase1_training_step(model, ids, tgt, tau=0.5)
        losses["total_loss"].backward()

        # Cortex weights should have gradients
        assert any(
            p.grad is not None
            for cortex in model.cortices.values()
            for p in cortex.parameters()
        )

    def test_variable_tau(self):
        """Phase 1 trains across different τ levels."""
        model = _tiny_model()
        _init_delta_pool(model, pool_size=2)
        model.train()

        ids, tgt = _random_batch()
        for tau in [0.1, 0.5, 0.9]:
            losses = phase1_training_step(model, ids, tgt, tau=tau)
            assert losses["pred_loss"].isfinite()

    def test_amp_step(self):
        """Phase 1 step works under AMP autocast (CPU fallback)."""
        model = _tiny_model()
        _init_delta_pool(model, pool_size=2)
        model.train()

        ids, tgt = _random_batch()
        with torch.amp.autocast("cpu", enabled=True):
            losses = phase1_training_step(model, ids, tgt, tau=0.5)
        assert losses["pred_loss"].isfinite()

    def test_max_steps(self):
        """train_phase1 respects max_steps cap."""
        model = _tiny_model()
        router = ThalamicRouter(d_model=D_MODEL, cortex_names=model.cortex_names)
        model.attach_router(router)

        dataset = torch.utils.data.TensorDataset(
            torch.randint(0, VOCAB, (20, SEQ_LEN)),
            torch.randint(0, VOCAB, (20, SEQ_LEN)),
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH)

        history = train_phase1(
            model, loader,
            num_epochs=10, max_steps=5,
            log_every=999, use_amp=False,
        )
        assert len(history) == 5


# ---------------------------------------------------------------------------
# Phase 2 — Thermal Routing
# ---------------------------------------------------------------------------

class TestPhase2Smoke:
    def test_single_step(self):
        model = _tiny_model()
        thermal = ThermalEstimator(d_model=D_MODEL)
        model.attach_thermal(thermal)
        bridges = build_bridges(model.cortex_names, d_model=D_MODEL)
        model.attach_bridges(bridges)
        model.train()

        ids, tgt = _random_batch()
        losses = phase2_training_step(model, ids, tgt)

        assert "pred_loss" in losses
        assert "compute_cost" in losses
        assert "tau" in losses
        assert losses["total_loss"].isfinite()

    def test_backward(self):
        model = _tiny_model()
        thermal = ThermalEstimator(d_model=D_MODEL)
        model.attach_thermal(thermal)
        bridges = build_bridges(model.cortex_names, d_model=D_MODEL)
        model.attach_bridges(bridges)
        model.train()

        ids, tgt = _random_batch()
        losses = phase2_training_step(model, ids, tgt)
        losses["total_loss"].backward()

        # Thermal estimator should have gradients
        assert any(p.grad is not None for p in thermal.parameters())

    def test_requires_thermal(self):
        model = _tiny_model()
        ids, tgt = _random_batch()
        with pytest.raises(AssertionError, match="thermal estimator"):
            phase2_training_step(model, ids, tgt)

    def test_compute_cost_positive(self):
        model = _tiny_model()
        thermal = ThermalEstimator(d_model=D_MODEL)
        model.attach_thermal(thermal)
        model.train()

        ids, tgt = _random_batch()
        losses = phase2_training_step(model, ids, tgt)
        assert losses["compute_cost"].item() >= 0


# ---------------------------------------------------------------------------
# Phase 3 — Memory System
# ---------------------------------------------------------------------------

class TestPhase3Smoke:
    def test_single_conversation(self):
        model = _tiny_model()
        mem_ctrl = MemoryController(d_model=D_MODEL, episode_dim=16)
        model.attach_memory(mem_ctrl)
        compressor = EpisodicCompressor(d_model=D_MODEL, episode_dim=16, nhead=4, num_layers=1)
        model.train()

        # 3-turn conversation
        chain = [_random_batch() for _ in range(3)]
        losses = phase3_training_step(model, compressor, chain)

        assert "total_loss" in losses
        assert losses["total_loss"].isfinite()
        assert losses["num_turns"].item() == 3

    def test_backward(self):
        model = _tiny_model()
        mem_ctrl = MemoryController(d_model=D_MODEL, episode_dim=16)
        model.attach_memory(mem_ctrl)
        compressor = EpisodicCompressor(d_model=D_MODEL, episode_dim=16, nhead=4, num_layers=1)
        model.train()

        chain = [_random_batch() for _ in range(2)]
        losses = phase3_training_step(model, compressor, chain)
        losses["total_loss"].backward()

        # Memory controller should have gradients
        assert any(p.grad is not None for p in mem_ctrl.parameters())

    def test_requires_memory_controller(self):
        model = _tiny_model()
        compressor = EpisodicCompressor(d_model=D_MODEL, episode_dim=16, nhead=4, num_layers=1)
        chain = [_random_batch()]
        with pytest.raises(AssertionError, match="memory controller"):
            phase3_training_step(model, compressor, chain)

    def test_episodes_accumulate(self):
        model = _tiny_model()
        mem_ctrl = MemoryController(d_model=D_MODEL, episode_dim=16)
        model.attach_memory(mem_ctrl)
        compressor = EpisodicCompressor(d_model=D_MODEL, episode_dim=16, nhead=4, num_layers=1)
        model.train()

        chain = [_random_batch() for _ in range(5)]
        losses = phase3_training_step(model, compressor, chain)
        assert losses["num_episodes"].item() >= 1


# ---------------------------------------------------------------------------
# Phase 4 — Meta-Delta Generation
# ---------------------------------------------------------------------------

class TestPhase4Smoke:
    def test_single_step(self):
        model = _tiny_model()
        router = ThalamicRouter(d_model=D_MODEL, cortex_names=model.cortex_names)
        model.attach_router(router)
        meta_gen = MetaDeltaGenerator(
            d_model=D_MODEL, delta_rank=4,
            num_cortices=len(model.cortex_names), num_strata=3,
            nhead=4, num_layers=1,
        )
        model.train()

        # Fill error buffer
        error_buf = ErrorBuffer(max_size=64, d_model=D_MODEL)
        for _ in range(20):
            error_buf.add(torch.randn(D_MODEL), torch.randn(D_MODEL))

        ids, tgt = _random_batch()
        losses = phase4_training_step(meta_gen, model, error_buf, ids, tgt)

        assert "meta_loss" in losses
        assert "improvement" in losses
        assert "accepted" in losses
        assert losses["meta_loss"].isfinite()

    def test_backward(self):
        model = _tiny_model()
        router = ThalamicRouter(d_model=D_MODEL, cortex_names=model.cortex_names)
        model.attach_router(router)
        meta_gen = MetaDeltaGenerator(
            d_model=D_MODEL, delta_rank=4,
            num_cortices=len(model.cortex_names), num_strata=3,
            nhead=4, num_layers=1,
        )
        model.train()

        error_buf = ErrorBuffer(max_size=64, d_model=D_MODEL)
        for _ in range(20):
            error_buf.add(torch.randn(D_MODEL), torch.randn(D_MODEL))

        ids, tgt = _random_batch()
        losses = phase4_training_step(meta_gen, model, error_buf, ids, tgt)
        losses["meta_loss"].backward()

        # Meta-gen should have gradients
        assert any(p.grad is not None for p in meta_gen.parameters())

    def test_rejected_delta_rolled_back(self):
        """If delta hurts performance, it gets popped (clean rollback)."""
        model = _tiny_model()
        router = ThalamicRouter(d_model=D_MODEL, cortex_names=model.cortex_names)
        model.attach_router(router)
        meta_gen = MetaDeltaGenerator(
            d_model=D_MODEL, delta_rank=4,
            num_cortices=len(model.cortex_names), num_strata=3,
            nhead=4, num_layers=1,
        )

        # Count deltas before
        delta_count_before = sum(
            len(s.delta_stack)
            for c in model.cortices.values()
            for s in c.strata.values()
        )

        error_buf = ErrorBuffer(max_size=64, d_model=D_MODEL)
        for _ in range(20):
            error_buf.add(torch.randn(D_MODEL), torch.randn(D_MODEL))

        ids, tgt = _random_batch()
        losses = phase4_training_step(meta_gen, model, error_buf, ids, tgt)

        delta_count_after = sum(
            len(s.delta_stack)
            for c in model.cortices.values()
            for s in c.strata.values()
        )

        if losses["accepted"].item() == 0:
            # Rejected → rolled back, count unchanged
            assert delta_count_after == delta_count_before
        else:
            # Accepted → exactly one more delta
            assert delta_count_after == delta_count_before + 1


# ---------------------------------------------------------------------------
# Error Buffer
# ---------------------------------------------------------------------------

class TestErrorBuffer:
    def test_add_and_len(self):
        buf = ErrorBuffer(max_size=10, d_model=32)
        for _ in range(5):
            buf.add(torch.randn(32), torch.randn(32))
        assert len(buf) == 5

    def test_max_size(self):
        buf = ErrorBuffer(max_size=5, d_model=32)
        for _ in range(10):
            buf.add(torch.randn(32), torch.randn(32))
        assert len(buf) == 5

    def test_ready(self):
        buf = ErrorBuffer(max_size=64, d_model=32)
        assert not buf.ready
        for _ in range(16):
            buf.add(torch.randn(32), torch.randn(32))
        assert buf.ready

    def test_get_buffer_shape(self):
        buf = ErrorBuffer(max_size=64, d_model=32)
        for _ in range(10):
            buf.add(torch.randn(32), torch.randn(32))
        t = buf.get_buffer()
        assert t.shape == (1, 10, 32)

    def test_clear(self):
        buf = ErrorBuffer(max_size=64, d_model=32)
        for _ in range(10):
            buf.add(torch.randn(32), torch.randn(32))
        buf.clear()
        assert len(buf) == 0
