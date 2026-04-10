"""FLX Hypothesis System — rule induction for few-shot tasks.

HypothesisHead encodes candidate transformation rules as dense vectors.
TaskScratchpad tracks hypothesis attempts within a single problem.

Trained in Phase 5 after all other components are stable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class HypothesisHead(nn.Module):
    """Encodes a candidate transformation rule as a dense vector.

    Sits between MemoryController output and the refinement loop.
    Each loop iteration refines the hypothesis based on demonstration
    feedback before the final prediction.

    The hypothesis vector conditions the decoder: "given this rule,
    predict the test output."  The consistency score is a self-assessment
    of how well the current hypothesis explains all demonstrations.

    Args:
        d_model: Model dimension (must match pipeline).
        hypothesis_dim: Dimension of hypothesis vectors.
        nhead: Attention heads in the hypothesis encoder.
        num_layers: Transformer layers in the hypothesis encoder.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 512,
        hypothesis_dim: int = 512,
        nhead: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.hypothesis_dim = hypothesis_dim

        # Encode fused representation into hypothesis space
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.hypothesis_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )
        self.hypothesis_proj = nn.Linear(d_model, hypothesis_dim)
        self.hypothesis_norm = nn.LayerNorm(hypothesis_dim)

        # Self-consistency scorer: how well does hypothesis explain demos
        self.consistency_head = nn.Linear(hypothesis_dim, 1)

        # Demo integration: cross-attend over demonstration embeddings
        self.demo_proj = nn.Linear(d_model, hypothesis_dim)
        self.demo_attn = nn.MultiheadAttention(
            embed_dim=hypothesis_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.demo_norm = nn.LayerNorm(hypothesis_dim)

        # Scratchpad integration: attend over prior hypotheses
        self.trajectory_proj = nn.Linear(hypothesis_dim, hypothesis_dim)
        self.trajectory_gate = nn.Sequential(
            nn.Linear(hypothesis_dim * 2, hypothesis_dim),
            nn.GELU(),
            nn.Linear(hypothesis_dim, hypothesis_dim),
        )

        # Conditioning projection: hypothesis → d_model for decoder
        self.condition_proj = nn.Linear(hypothesis_dim, d_model)

    def forward(
        self,
        fused_repr: Tensor,
        demo_embeddings: Tensor | None = None,
        trajectory: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Produce a hypothesis vector and consistency score.

        Args:
            fused_repr: [batch, seq, d_model] from memory controller.
            demo_embeddings: [batch, N, d_model] encoded demonstrations.
                If None, hypothesis is based only on fused representation.
            trajectory: [batch, num_prior, hypothesis_dim] prior hypothesis
                vectors from the scratchpad.  If None, no trajectory context.

        Returns:
            hypothesis: [batch, hypothesis_dim] — rule representation.
            consistency: [batch] — self-assessed quality in (0, 1).
            conditioning: [batch, 1, d_model] — additive bias for decoder.
        """
        # Encode fused representation
        encoded = self.hypothesis_encoder(fused_repr)
        pooled = encoded.mean(dim=1)  # [batch, d_model]
        hypothesis = self.hypothesis_norm(
            self.hypothesis_proj(pooled)
        )  # [batch, hypothesis_dim]

        # Integrate demonstration context via cross-attention
        if demo_embeddings is not None:
            demo_keys = self.demo_proj(demo_embeddings)  # [batch, N, hypothesis_dim]
            query = hypothesis.unsqueeze(1)  # [batch, 1, hypothesis_dim]
            attended, _ = self.demo_attn(query, demo_keys, demo_keys)
            attended = attended.squeeze(1)  # [batch, hypothesis_dim]
            hypothesis = self.demo_norm(hypothesis + attended)

        # Integrate scratchpad trajectory (avoid repeating failed hypotheses)
        if trajectory is not None and trajectory.shape[1] > 0:
            traj_repr = self.trajectory_proj(
                trajectory.mean(dim=1)
            )  # [batch, hypothesis_dim]
            combined = torch.cat([hypothesis, traj_repr], dim=-1)
            hypothesis = hypothesis + self.trajectory_gate(combined)

        # Consistency score: sigmoid → (0, 1)
        consistency = torch.sigmoid(
            self.consistency_head(hypothesis).squeeze(-1)
        )  # [batch]

        # Decoder conditioning: project hypothesis back to d_model
        conditioning = self.condition_proj(hypothesis).unsqueeze(1)  # [batch, 1, d_model]

        return hypothesis, consistency, conditioning


class TaskScratchpad:
    """Task-scoped working memory for hypothesis tracking.

    Created at the start of each few-shot task, discarded after.
    Not an nn.Module — pure state container with no learnable parameters.

    Tracks previous hypothesis vectors and their self-assessed quality
    so the refinement loop can avoid repeating failed approaches.

    Args:
        hypothesis_dim: Dimension of hypothesis vectors.
        max_hypotheses: Maximum hypotheses to store before FIFO eviction.
    """

    def __init__(self, hypothesis_dim: int = 512, max_hypotheses: int = 8):
        self.hypothesis_dim = hypothesis_dim
        self.max_hypotheses = max_hypotheses
        self.hypotheses: list[Tensor] = []
        self.scores: list[float] = []

    def add_hypothesis(self, hypothesis: Tensor, score: float) -> None:
        """Record a hypothesis attempt and its consistency score.

        Args:
            hypothesis: [hypothesis_dim] or [batch, hypothesis_dim].
            score: Consistency score in (0, 1).
        """
        h = hypothesis.detach()
        if h.dim() > 1:
            h = h[0]  # store first batch element as representative
        self.hypotheses.append(h)
        self.scores.append(score)
        if len(self.hypotheses) > self.max_hypotheses:
            self.hypotheses.pop(0)
            self.scores.pop(0)

    def get_best(self) -> Tensor | None:
        """Return the highest-scoring hypothesis, or None if empty."""
        if not self.hypotheses:
            return None
        best_idx = max(range(len(self.scores)), key=lambda i: self.scores[i])
        return self.hypotheses[best_idx]

    def get_trajectory(self, device: str | torch.device = "cpu") -> Tensor:
        """Return all prior hypotheses as a trajectory tensor.

        Returns:
            [1, num_hypotheses, hypothesis_dim] or [1, 0, hypothesis_dim]
                if no hypotheses stored yet.
        """
        if not self.hypotheses:
            return torch.zeros(1, 0, self.hypothesis_dim, device=device)
        stacked = torch.stack(self.hypotheses).unsqueeze(0)  # [1, N, hypothesis_dim]
        return stacked.to(device)

    def clear(self) -> None:
        self.hypotheses.clear()
        self.scores.clear()

    def __len__(self) -> int:
        return len(self.hypotheses)

    @property
    def is_empty(self) -> bool:
        return len(self.hypotheses) == 0
