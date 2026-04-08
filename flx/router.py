"""FLX Thalamic Router — domain classification + multi-hot dispatch.

Routes input to relevant domain cortices using chunk-level routing
with max aggregation (see spec 10 — reality checks).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ThalamicRouter(nn.Module):
    """Routes input to relevant domain cortices via chunk-level multi-hot scoring.

    Like the brain's thalamus — a small learned network that detects which
    knowledge domains the input requires. Outputs a multi-hot domain score
    vector. Multiple cortices can activate simultaneously.

    Uses chunk-level routing (spec 10 fix): input is segmented into fixed-size
    chunks, each chunk is scored independently, and sequence-level scores are
    the max across chunks (union of domain needs).

    Args:
        d_model: Model dimension.
        cortex_names: List of domain cortex identifiers.
        chunk_size: Token window size for chunk-level routing.
        activation_threshold: Minimum score to activate a cortex.
    """

    def __init__(
        self,
        d_model: int = 512,
        cortex_names: list[str] | None = None,
        chunk_size: int = 64,
        activation_threshold: float = 0.2,
    ):
        super().__init__()
        self.cortex_names = cortex_names or [
            "language", "math", "code", "science", "reasoning"
        ]
        self.chunk_size = chunk_size
        self.activation_threshold = activation_threshold
        self.num_cortices = len(self.cortex_names)

        # Domain classifier: projects chunk mean to domain scores
        self.domain_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, self.num_cortices),
        )

    def forward(self, embedded_input: Tensor) -> dict[str, Tensor]:
        """Compute multi-hot domain scores via chunk-level routing.

        Args:
            embedded_input: [batch, seq_len, d_model] from shared trunk.

        Returns:
            Dict mapping cortex names to scores: {name: [batch] tensor}.
            Only cortices above activation_threshold are included.
        """
        batch_size, seq_len, d_model = embedded_input.shape

        # Segment into chunks
        if seq_len <= self.chunk_size:
            chunks = [embedded_input]
        else:
            chunks = []
            for start in range(0, seq_len, self.chunk_size):
                end = min(start + self.chunk_size, seq_len)
                chunks.append(embedded_input[:, start:end, :])

        # Score each chunk independently
        chunk_scores = []
        for chunk in chunks:
            chunk_mean = chunk.mean(dim=1)  # [batch, d_model]
            scores = torch.sigmoid(self.domain_classifier(chunk_mean))  # [batch, num_cortices]
            chunk_scores.append(scores)

        # Max aggregation: sequence needs every cortex that any chunk needs
        all_scores = torch.stack(chunk_scores, dim=0)  # [num_chunks, batch, num_cortices]
        sequence_scores = all_scores.max(dim=0).values  # [batch, num_cortices]

        # Build output dict with only active cortices
        result = {}
        for i, name in enumerate(self.cortex_names):
            score = sequence_scores[:, i]  # [batch]
            if (score > self.activation_threshold).any():
                result[name] = score

        return result

    def forward_raw(self, embedded_input: Tensor) -> Tensor:
        """Return raw score tensor (useful for loss computation).

        Args:
            embedded_input: [batch, seq_len, d_model]

        Returns:
            domain_scores: [batch, num_cortices] in [0, 1]
        """
        batch_size, seq_len, d_model = embedded_input.shape

        if seq_len <= self.chunk_size:
            chunks = [embedded_input]
        else:
            chunks = []
            for start in range(0, seq_len, self.chunk_size):
                end = min(start + self.chunk_size, seq_len)
                chunks.append(embedded_input[:, start:end, :])

        chunk_scores = []
        for chunk in chunks:
            chunk_mean = chunk.mean(dim=1)
            scores = torch.sigmoid(self.domain_classifier(chunk_mean))
            chunk_scores.append(scores)

        all_scores = torch.stack(chunk_scores, dim=0)
        sequence_scores = all_scores.max(dim=0).values

        return sequence_scores


def diversity_loss(domain_scores: Tensor) -> Tensor:
    """Penalize routing collapse where all samples route to the same cortex(es).

    Measures batch-level cortex utilization uniformity. We want different
    samples to prefer DIFFERENT cortices, producing specialization.
    All samples → one cortex → loss ≈ 1.0 (collapsed).
    Samples distributed across cortices → loss ≈ 0.0 (diverse).

    Args:
        domain_scores: [batch, num_cortices] activation scores in [0, 1].

    Returns:
        Scalar diversity loss normalized to [0, 1].
    """
    K = domain_scores.shape[1]
    # Sharpen scores to get soft assignment (which cortex does each sample prefer?)
    assignment = torch.softmax(domain_scores * 5.0, dim=-1)  # [batch, K]
    # Average across batch: what fraction of the batch uses each cortex?
    utilization = assignment.mean(dim=0)  # [K]
    # Entropy of utilization: high entropy = uniform usage = good
    entropy = -(utilization * (utilization + 1e-8).log()).sum()
    max_entropy = torch.tensor(float(K), device=domain_scores.device).log()
    # Loss: 1 - normalized_entropy. Uniform → 0, collapsed → 1.
    return 1.0 - entropy / (max_entropy + 1e-8)


def load_balance_loss(domain_scores: Tensor, num_cortices: int) -> Tensor:
    """Switch Transformer-style load balancing loss.

    Prevents routing collapse where one cortex dominates.
    See spec 10 — reality checks.

    Args:
        domain_scores: [batch, num_cortices] activation scores.
        num_cortices: Number of cortices (K).

    Returns:
        Scalar balance loss.
    """
    # Fraction of tokens routed to each cortex
    fraction_routed = domain_scores.mean(dim=0)  # [num_cortices]

    # Fraction of routing probability assigned to each cortex
    fraction_prob = domain_scores.sum(dim=0) / (domain_scores.sum() + 1e-8)

    # Penalize imbalance: want fraction_routed ≈ 1/K
    balance_loss = num_cortices * (fraction_routed * fraction_prob).sum()

    return balance_loss
