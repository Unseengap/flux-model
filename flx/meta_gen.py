"""FLX Meta-Delta Generator — online self-improvement.

A small network that takes accumulated prediction errors + current delta stack
state and produces new delta A/B matrices targeting the correct cortex + stratum.
Trained in Phase 4 with meta-learning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .delta import DeltaMetadata, FLXDelta


class MetaDeltaGenerator(nn.Module):
    """Generates delta A/B matrices from accumulated error signals.

    Takes (error_summary, current_delta_stack_state) as input and produces
    the A and B matrices for a new delta in a single forward pass.

    Args:
        d_model: Model dimension (matches cortex dimension).
        delta_rank: Rank for generated deltas.
        num_cortices: Number of domain cortices.
        num_strata: Number of strata per cortex (typically 3: intermediate, expert, frontier).
        nhead: Attention heads in the error encoder.
        num_layers: Layers in the error encoder.
    """

    def __init__(
        self,
        d_model: int = 512,
        delta_rank: int = 32,
        num_cortices: int = 5,
        num_strata: int = 3,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.delta_rank = delta_rank
        self.num_cortices = num_cortices
        self.num_strata = num_strata

        # Error encoder: processes accumulated error signals
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.error_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Stack encoder: encodes current delta stack state
        self.stack_encoder = nn.Linear(d_model, d_model)

        # Delta matrix generators
        self.A_head = nn.Linear(d_model, delta_rank * d_model)
        self.B_head = nn.Linear(d_model, d_model * delta_rank)

        # Metadata heads: predict target cortex, stratum, and thermal threshold
        self.cortex_head = nn.Linear(d_model, num_cortices)
        self.stratum_head = nn.Linear(d_model, num_strata)
        self.threshold_head = nn.Linear(d_model, 1)

    def forward(
        self,
        error_buffer: Tensor,
        stack_summary: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, dict]:
        """Generate a candidate delta from error signals.

        Args:
            error_buffer: [batch, num_errors, d_model] accumulated error embeddings.
            stack_summary: [batch, d_model] summary of current delta stack state.
                If None, uses zero vector.

        Returns:
            (A, B, metadata) where:
                A: [batch, rank, d_model]
                B: [batch, d_model, rank]
                metadata: dict with cortex_logits, stratum_logits, threshold
        """
        batch_size = error_buffer.shape[0]
        device = error_buffer.device

        # Encode errors
        err_repr = self.error_encoder(error_buffer)
        err_summary = err_repr.mean(dim=1)  # [batch, d_model]

        # Encode stack state
        if stack_summary is not None:
            stk_repr = self.stack_encoder(stack_summary)
            combined = err_summary + stk_repr
        else:
            combined = err_summary

        # Generate A and B matrices
        A = self.A_head(combined).reshape(batch_size, self.delta_rank, self.d_model)
        B = self.B_head(combined).reshape(batch_size, self.d_model, self.delta_rank)

        # Scale initialization: start small
        A = A * 0.01
        B = B * 0.01

        # Predict target metadata
        cortex_logits = self.cortex_head(combined)  # [batch, num_cortices]
        stratum_logits = self.stratum_head(combined)  # [batch, num_strata]
        threshold = torch.sigmoid(self.threshold_head(combined))  # [batch, 1]

        metadata = {
            "cortex_logits": cortex_logits,
            "stratum_logits": stratum_logits,
            "threshold": threshold.squeeze(-1),
        }

        return A, B, metadata

    def generate_delta(
        self,
        error_buffer: Tensor,
        stack_summary: Tensor | None = None,
        cortex_names: list[str] | None = None,
        stratum_names: list[str] | None = None,
    ) -> FLXDelta:
        """Generate a complete FLXDelta object from error signals.

        Convenience method that wraps forward() and creates an FLXDelta.
        Uses batch dimension 0 (single example).

        Args:
            error_buffer: [1, num_errors, d_model] or [num_errors, d_model]
            stack_summary: [1, d_model] or [d_model] or None
            cortex_names: Names of cortices for metadata.
            stratum_names: Names of strata for metadata.

        Returns:
            FLXDelta with probationary confidence (0.1).
        """
        if error_buffer.dim() == 2:
            error_buffer = error_buffer.unsqueeze(0)
        if stack_summary is not None and stack_summary.dim() == 1:
            stack_summary = stack_summary.unsqueeze(0)

        A, B, metadata = self.forward(error_buffer, stack_summary)

        # Create delta
        delta = FLXDelta(
            d_in=self.d_model,
            d_out=self.d_model,
            rank=self.delta_rank,
            thermal_threshold=metadata["threshold"][0].item(),
            confidence=0.1,  # probationary
        )

        # Set A and B from generated matrices
        with torch.no_grad():
            delta.A.copy_(A[0])
            delta.B.copy_(B[0])

        # Set metadata
        cortex_names = cortex_names or [f"cortex_{i}" for i in range(self.num_cortices)]
        stratum_names = stratum_names or ["intermediate", "expert", "frontier"]

        cortex_idx = metadata["cortex_logits"][0].argmax().item()
        stratum_idx = metadata["stratum_logits"][0].argmax().item()

        delta.metadata = DeltaMetadata(
            source="meta_gen",
            target_cortex=cortex_names[cortex_idx] if cortex_idx < len(cortex_names) else "unknown",
            target_stratum=stratum_names[stratum_idx] if stratum_idx < len(stratum_names) else "unknown",
        )

        return delta
