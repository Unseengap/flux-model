"""Tests for FLX serialization — .flx save/load round-trip."""

import tempfile
import torch
import pytest

from flx.model import FLXNano
from flx.router import ThalamicRouter
from flx.thermal import ThermalEstimator
from flx.bridges import build_bridges
from flx.memory import EpisodicBuffer, MemoryController
from flx.delta import FLXDelta
from flx.serialization import save_flx, load_flx


def _make_model(d_model=64, nhead=4):
    """Create a small FLXNano for test purposes."""
    model = FLXNano(
        vocab_size=1000, d_model=d_model, nhead=nhead,
        trunk_layers=2, layers_per_stratum=1,
        delta_rank=8, max_seq_len=128, dim_feedforward=128,
    )
    # Attach all components
    router = ThalamicRouter(d_model=d_model, cortex_names=model.cortex_names)
    model.attach_router(router)

    thermal = ThermalEstimator(d_model=d_model)
    model.attach_thermal(thermal)

    bridges = build_bridges(model.cortex_names, d_model=d_model)
    model.attach_bridges(bridges)

    mem_ctrl = MemoryController(d_model=d_model, episode_dim=32)
    model.attach_memory(mem_ctrl)

    return model


class TestSerialization:
    def test_save_and_load(self):
        model = _make_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/test.flx"
            save_flx(model, path)
            loaded_model, ep_buf, act_hist = load_flx(path)

        # Check basic structure
        assert set(loaded_model.cortex_names) == set(model.cortex_names)
        assert loaded_model.d_model == model.d_model

    def test_round_trip_inference(self):
        """Model produces same output after save/load (no bridges for exact match)."""
        model = FLXNano(
            vocab_size=1000, d_model=64, nhead=4,
            trunk_layers=2, layers_per_stratum=1,
            delta_rank=8, max_seq_len=128, dim_feedforward=128,
        )
        router = ThalamicRouter(d_model=64, cortex_names=model.cortex_names)
        model.attach_router(router)
        model.eval()

        input_ids = torch.randint(0, 1000, (1, 10))
        with torch.no_grad():
            original_out = model(input_ids, tau=0.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/test.flx"
            save_flx(model, path)
            loaded_model, _, _ = load_flx(path)

        loaded_model.eval()
        with torch.no_grad():
            loaded_out = loaded_model(input_ids, tau=0.5)

        assert torch.allclose(original_out, loaded_out, atol=1e-5)

    def test_episodic_buffer_persistence(self):
        """Episodic buffer is saved and restored."""
        model = _make_model()
        ep_buf = EpisodicBuffer()
        ep_buf.add(torch.randn(32))
        ep_buf.add(torch.randn(32))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/test.flx"
            save_flx(model, path, episodic_buffer=ep_buf)
            _, loaded_buf, _ = load_flx(path)

        assert loaded_buf is not None
        assert len(loaded_buf) == 2

    def test_thermal_history_persistence(self):
        """Thermal history survives save/load."""
        model = _make_model()
        # Generate some thermal history
        x = torch.randint(0, 1000, (1, 10))
        model.eval()
        with torch.no_grad():
            model(x)
            model(x)

        history_before = model.thermal_estimator.get_history()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/test.flx"
            save_flx(model, path)
            loaded_model, _, _ = load_flx(path)

        history_after = loaded_model.thermal_estimator.get_history()
        assert len(history_after) == len(history_before)

    def test_delta_persistence(self):
        """Deltas in strata survive save/load."""
        model = _make_model()
        # Add a delta to math cortex, expert stratum
        delta = FLXDelta(d_in=64, d_out=64, rank=8, thermal_threshold=0.5, confidence=0.7)
        delta.B.data.fill_(0.01)
        model.cortices["math"].strata["expert"].delta_stack.push(delta)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/test.flx"
            save_flx(model, path)
            loaded_model, _, _ = load_flx(path)

        loaded_stack = loaded_model.cortices["math"].strata["expert"].delta_stack
        assert len(loaded_stack) == 1
        loaded_delta = list(loaded_stack.deltas)[0]
        assert loaded_delta.thermal_threshold == 0.5
        assert abs(loaded_delta.confidence.item() - 0.7) < 0.01
