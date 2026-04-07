"""Tests for FLX memory subsystem — EpisodicCompressor, MemoryController."""

import torch
import pytest

from flx.memory import EpisodicCompressor, EpisodicBuffer, MemoryController


class TestEpisodicCompressor:
    def test_compress(self):
        compressor = EpisodicCompressor(d_model=64, episode_dim=32, nhead=4, num_layers=1)
        kv_chunk = torch.randn(2, 20, 64)
        episode = compressor(kv_chunk)
        assert episode.shape == (2, 32)

    def test_compress_unbatched(self):
        compressor = EpisodicCompressor(d_model=64, episode_dim=32, nhead=4, num_layers=1)
        kv_chunk = torch.randn(20, 64)
        episode = compressor(kv_chunk)
        assert episode.shape == (32,)


class TestEpisodicBuffer:
    def test_add_and_get(self):
        buffer = EpisodicBuffer(max_episodes=10)
        buffer.add(torch.randn(32))
        buffer.add(torch.randn(32))
        assert len(buffer) == 2
        assert len(buffer.get_all()) == 2

    def test_fifo_eviction(self):
        buffer = EpisodicBuffer(max_episodes=3)
        for i in range(5):
            buffer.add(torch.full((32,), float(i)))
        assert len(buffer) == 3
        # First two should have been evicted
        assert buffer.get_all()[0][0].item() == 2.0

    def test_clear(self):
        buffer = EpisodicBuffer()
        buffer.add(torch.randn(32))
        buffer.clear()
        assert buffer.is_empty()


class TestMemoryController:
    def test_below_threshold_passthrough(self):
        controller = MemoryController(d_model=64, episode_dim=32)
        x = torch.randn(2, 10, 64)
        episodes = [torch.randn(32)]
        out, should_loop = controller(x, episodes, tau=0.3)
        assert out.shape == (2, 10, 64)
        assert torch.allclose(out, x)
        assert not should_loop

    def test_retrieval_at_high_tau(self):
        controller = MemoryController(d_model=64, episode_dim=32)
        x = torch.randn(2, 10, 64)
        episodes = [torch.randn(32) for _ in range(5)]
        out, should_loop = controller(x, episodes, tau=0.6)
        assert out.shape == (2, 10, 64)
        # Should differ from input (memory was fused)
        assert not torch.allclose(out, x)

    def test_empty_buffer_passthrough(self):
        controller = MemoryController(d_model=64, episode_dim=32)
        x = torch.randn(2, 10, 64)
        out, should_loop = controller(x, [], tau=0.8)
        assert torch.allclose(out, x)
        assert not should_loop

    def test_loop_gating(self):
        controller = MemoryController(d_model=64, episode_dim=32, loop_tau_min=0.7)
        x = torch.randn(2, 10, 64)
        episodes = [torch.randn(32)]
        controller.reset_loop_count()
        _, should_loop = controller(x, episodes, tau=0.8)
        assert should_loop  # tau > 0.7 and loop_count < 3
