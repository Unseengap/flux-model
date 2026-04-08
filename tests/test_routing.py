"""Tests for FLX thalamic router — routing, diversity loss, load balance."""

import torch
import pytest

from flx.router import ThalamicRouter, diversity_loss, load_balance_loss


class TestThalamicRouter:
    def test_forward_returns_dict(self):
        router = ThalamicRouter(d_model=64, cortex_names=["math", "code", "language"])
        x = torch.randn(2, 20, 64)
        scores = router(x)
        assert isinstance(scores, dict)
        for name, score in scores.items():
            assert name in ["math", "code", "language"]
            assert score.shape == (2,)

    def test_forward_raw_returns_tensor(self):
        router = ThalamicRouter(d_model=64, cortex_names=["math", "code"])
        x = torch.randn(4, 20, 64)
        scores = router.forward_raw(x)
        assert scores.shape == (4, 2)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_chunk_routing(self):
        """With long sequences, router uses chunk-level scoring."""
        router = ThalamicRouter(d_model=64, chunk_size=16)
        x = torch.randn(2, 100, 64)  # longer than chunk_size
        scores = router(x)
        assert isinstance(scores, dict)

    def test_single_chunk(self):
        """Short sequences use a single chunk."""
        router = ThalamicRouter(d_model=64, chunk_size=64)
        x = torch.randn(2, 10, 64)  # shorter than chunk_size
        scores = router(x)
        assert isinstance(scores, dict)


class TestDiversityLoss:
    def test_identical_scores_high_loss(self):
        """All cortices activating equally → high diversity loss."""
        scores = torch.ones(32, 5) * 0.8
        loss = diversity_loss(scores)
        assert loss.item() > 0.9  # uniform scores → near-max entropy

    def test_orthogonal_scores_low_loss(self):
        """Each cortex activating on different inputs → low diversity loss."""
        scores = torch.eye(5).repeat(1, 1)  # 5 inputs, 5 cortices
        loss = diversity_loss(scores)
        assert loss.item() < 0.1


class TestLoadBalanceLoss:
    def test_balanced_routing(self):
        """Equal routing → minimal loss."""
        scores = torch.ones(32, 5) / 5.0
        loss = load_balance_loss(scores, num_cortices=5)
        assert loss.item() >= 0

    def test_imbalanced_routing(self):
        """One cortex dominates → higher loss."""
        scores = torch.zeros(32, 5)
        scores[:, 0] = 1.0
        loss_imbalanced = load_balance_loss(scores, num_cortices=5)

        scores_balanced = torch.ones(32, 5) * 0.2
        loss_balanced = load_balance_loss(scores_balanced, num_cortices=5)

        assert loss_imbalanced > loss_balanced
