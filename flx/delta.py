"""FLX Delta — The foundational weight primitive.

Low-rank delta matrices (A, B) that compose onto base weights.
W_effective = W_base + Σ(confidence * scale * B @ A)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class DeltaMetadata:
    """Provenance and lifecycle metadata for a delta."""

    name: str = ""
    source: str = ""  # "phase1", "meta_gen", "manual", etc.
    target_cortex: str = ""
    target_stratum: str = ""
    created_at: str = ""
    description: str = ""


class FLXDelta(nn.Module):
    """A single low-rank delta: W_delta = confidence * scale * B @ A.

    Args:
        d_in: Input dimension of the weight matrix this delta modifies.
        d_out: Output dimension of the weight matrix this delta modifies.
        rank: Rank of the low-rank decomposition.
        thermal_threshold: Minimum τ required to activate this delta.
        confidence: Learned confidence scalar (0-1).
        scale: Fixed scaling factor (typically alpha/rank).
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        rank: int = 32,
        thermal_threshold: float = 0.0,
        confidence: float = 1.0,
        scale: Optional[float] = None,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank
        self.thermal_threshold = thermal_threshold

        # A: [rank, d_in], B: [d_out, rank]
        self.A = nn.Parameter(torch.empty(rank, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, rank))
        self.confidence = nn.Parameter(torch.tensor(confidence))

        # Scale factor: alpha / rank (like LoRA)
        self._scale = scale if scale is not None else 1.0 / rank

        self.metadata = DeltaMetadata()

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        # B initialized to zero so delta starts as identity
        nn.init.zeros_(self.B)

    @property
    def scale(self) -> float:
        return self._scale

    def compute(self) -> Tensor:
        """Compute the full delta matrix: confidence * scale * B @ A."""
        return self.confidence.clamp(0, 1) * self._scale * (self.B @ self.A)

    def forward(self, x: Tensor) -> Tensor:
        """Apply delta to input: x @ (B @ A)^T * confidence * scale."""
        # x: [batch, seq, d_in] -> [batch, seq, d_out]
        return self.confidence.clamp(0, 1) * self._scale * (x @ self.A.T @ self.B.T)

    def is_active(self, tau: float) -> bool:
        """Whether this delta should activate at the given thermal level."""
        return tau >= self.thermal_threshold


class DeltaStack(nn.Module):
    """An ordered stack of deltas with push/pop/rollback semantics.

    Deltas compose additively onto base weights:
    W_effective = W_base + Σ_i delta_i.compute()  for active deltas
    """

    def __init__(self, capacity: int = 16):
        super().__init__()
        self.deltas = nn.ModuleList()
        self.capacity = capacity

    def push(self, delta: FLXDelta) -> None:
        """Push a delta onto the stack."""
        if len(self.deltas) >= self.capacity:
            raise RuntimeError(
                f"Delta stack at capacity ({self.capacity}). "
                "Consolidate or increase capacity."
            )
        self.deltas.append(delta)

    def pop(self) -> FLXDelta:
        """Pop the most recent delta (clean rollback)."""
        if len(self.deltas) == 0:
            raise RuntimeError("Delta stack is empty.")
        delta = self.deltas[-1]
        self.deltas = nn.ModuleList(list(self.deltas)[:-1])
        return delta

    def active_deltas(self, tau: float) -> list[FLXDelta]:
        """Return deltas whose thermal threshold is met."""
        return [d for d in self.deltas if d.is_active(tau)]

    def compose(self, base_weight: Tensor, tau: float) -> Tensor:
        """Compose active deltas onto a base weight matrix.

        W_effective = W_base + Σ(confidence * scale * B @ A)
        """
        W = base_weight
        for delta in self.active_deltas(tau):
            W = W + delta.compute()
        return W

    def compose_input(self, x: Tensor, tau: float) -> Tensor:
        """Apply all active deltas to an input tensor (additive).

        Returns the sum of delta contributions: Σ delta_i(x).
        This is added to the base layer's output.
        """
        out = torch.zeros_like(x[..., : self.deltas[0].d_out]) if len(self.deltas) > 0 else None
        for delta in self.active_deltas(tau):
            contribution = delta(x)
            if out is None:
                out = contribution
            else:
                out = out + contribution
        return out

    def consolidate(self, base_weight: Tensor, tau: float = 0.0) -> Tensor:
        """Merge high-confidence deltas into base weights and clear them.

        Returns the new base weight with consolidated deltas baked in.
        Only consolidates deltas with confidence > 0.8.
        """
        keep = []
        consolidated = base_weight.clone()
        for delta in self.deltas:
            if delta.confidence.item() > 0.8:
                consolidated = consolidated + delta.compute().detach()
            else:
                keep.append(delta)
        self.deltas = nn.ModuleList(keep)
        return consolidated

    def __len__(self) -> int:
        return len(self.deltas)

    def __iter__(self):
        return iter(self.deltas)


def compose_weights(
    base_weights: Tensor,
    deltas: list[FLXDelta],
    tau: float,
) -> Tensor:
    """Functional interface for delta composition.

    W_effective = W_base + Σ(confidence * scale * B @ A) for active deltas.
    """
    W = base_weights
    for delta in deltas:
        if delta.thermal_threshold <= tau:
            W = W + delta.compute()
    return W
