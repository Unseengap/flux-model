"""FLX Thalamic Router — domain classification + multi-hot dispatch.

Routes input to relevant domain cortices using chunk-level routing
with max aggregation (see spec 10 — reality checks).
"""

from __future__ import annotations

import math

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
        sequence_scores = self.forward_raw(embedded_input)  # [batch, num_cortices]

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
            # Single chunk — fast path, no padding/reshaping needed
            chunk_mean = embedded_input.mean(dim=1)  # [batch, d_model]
            return torch.sigmoid(self.domain_classifier(chunk_mean))

        # Pad to even multiple of chunk_size, reshape, batch through classifier
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        padded_len = num_chunks * self.chunk_size
        if padded_len > seq_len:
            padding = embedded_input.new_zeros(batch_size, padded_len - seq_len, d_model)
            padded = torch.cat([embedded_input, padding], dim=1)
        else:
            padded = embedded_input

        # [batch, num_chunks, chunk_size, d_model]
        chunks = padded.view(batch_size, num_chunks, self.chunk_size, d_model)
        chunk_means = chunks.mean(dim=2)  # [batch, num_chunks, d_model]

        # Batch all chunks through classifier at once
        # Reshape to [batch * num_chunks, d_model], run classifier, reshape back
        flat_means = chunk_means.reshape(batch_size * num_chunks, d_model)
        flat_scores = torch.sigmoid(self.domain_classifier(flat_means))
        all_scores = flat_scores.view(batch_size, num_chunks, self.num_cortices)

        # Max aggregation across chunks
        sequence_scores = all_scores.max(dim=1).values  # [batch, num_cortices]
        return sequence_scores


def diversity_loss(domain_scores: Tensor) -> Tensor:
    """Penalize routing collapse — both score-to-zero bypass and all-to-same.

    Two complementary terms:

    - **Spikiness**: Each sample's top cortex score should be high (near 1.0),
      preventing the all-scores-to-zero collapse where the model bypasses
      cortices via the merger's residual gate.  ``max()`` gradient flows to
      the argmax element per sample — different samples naturally pick
      different cortices, bootstrapping diversity.  Critically, this term
      has **non-zero gradient even when all sigmoid outputs are uniform**
      (~0.5), which the earlier entropy-only formulation lacked.

    - **Spread**: Batch-level utilization across cortices should be uniform,
      preventing all-to-one-cortex collapse.  Only activates once spikiness
      has broken the uniform-sigmoid plateau.

    Args:
        domain_scores: [batch, num_cortices] activation scores in [0, 1].

    Returns:
        Scalar diversity loss.  Near 0 when routing is diverse and confident;
        near 2 when fully collapsed.
    """
    K = domain_scores.shape[1]

    # Spikiness: push each sample's top cortex score toward 1.0.
    # At init (all ~0.5): loss ≈ 0.5 with gradient −1/B on each sample's
    # argmax element — the first signal that breaks the uniform plateau.
    max_scores = domain_scores.max(dim=-1).values  # [batch]
    spikiness = (1.0 - max_scores).mean()

    # Spread: batch utilization entropy (Attempt 3 logic, still valid once
    # spikiness has broken the uniform plateau and scores diverge).
    assignment = torch.softmax(domain_scores * 5.0, dim=-1)  # [batch, K]
    utilization = assignment.mean(dim=0)  # [K]
    log_util = (utilization + 1e-8).log()
    entropy = -(utilization * log_util).sum()
    max_entropy = math.log(K)
    spread = 1.0 - entropy / (max_entropy + 1e-8)

    return spikiness + spread


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
