"""FLX Model — Core architecture components.

SharedTrunk, Stratum, DomainCortex, CortexMerger, and FLXNano model.

Architecture follows "thick trunk, thin branches" from spec 10:
- SharedTrunk (~100M): embedder + 6 transformer layers serving as basic stratum
- DomainCortex branches (~10M each): intermediate, expert, frontier strata only
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .delta import DeltaStack, FLXDelta


# ---------------------------------------------------------------------------
# Stratum — one difficulty layer within a cortex
# ---------------------------------------------------------------------------

class Stratum(nn.Module):
    """One difficulty layer within a domain cortex.

    Each stratum has its own transformer layers, confidence score,
    thermal threshold, and delta stack.

    Args:
        d_model: Model dimension.
        nhead: Number of attention heads.
        num_layers: Number of transformer encoder layers.
        depth: Stratum name — "intermediate", "expert", or "frontier".
        tau_min: Minimum τ to activate this stratum.
        initial_confidence: Starting confidence value.
        delta_capacity: Max number of deltas in this stratum's stack.
    """

    DEPTH_DEFAULTS = {
        "intermediate": {"tau_min": 0.25, "confidence": 0.7},
        "expert": {"tau_min": 0.5, "confidence": 0.5},
        "frontier": {"tau_min": 0.7, "confidence": 0.3},
    }

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        num_layers: int = 2,
        depth: str = "intermediate",
        tau_min: Optional[float] = None,
        initial_confidence: Optional[float] = None,
        delta_capacity: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.depth = depth
        defaults = self.DEPTH_DEFAULTS.get(depth, {"tau_min": 0.25, "confidence": 0.5})
        self.tau_min = tau_min if tau_min is not None else defaults["tau_min"]
        self.confidence = nn.Parameter(
            torch.tensor(initial_confidence if initial_confidence is not None else defaults["confidence"])
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.delta_stack = DeltaStack(capacity=delta_capacity)

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """Return a cached causal mask, regenerating only on size/device change."""
        if (
            not hasattr(self, "_causal_mask_cache")
            or self._causal_mask_cache.shape[0] != seq_len
            or self._causal_mask_cache.device != device
        ):
            self._causal_mask_cache = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=device),
                diagonal=1,
            )
        return self._causal_mask_cache

    def forward(self, x: Tensor, tau: float) -> Tensor:
        """Forward pass through stratum layers + delta contributions.

        Args:
            x: [batch, seq, d_model]
            tau: Current thermal level.

        Returns:
            Stratum output scaled by confidence, or zeros if below tau_min.
        """
        if tau < self.tau_min:
            return torch.zeros_like(x)

        causal_mask = self._get_causal_mask(x.shape[1], x.device)
        out = self.layers(x, mask=causal_mask)

        # Add delta contributions
        if len(self.delta_stack) > 0:
            delta_out = self.delta_stack.compose_input(x, tau)
            if delta_out is not None:
                out = out + delta_out

        return self.confidence.clamp(0, 1) * out

    def is_saturated(self, threshold: int = None) -> bool:
        """Check if delta stack is near capacity."""
        threshold = threshold or self.delta_stack.capacity
        return len(self.delta_stack) >= threshold


# ---------------------------------------------------------------------------
# SharedTrunk — thick trunk serving as the basic stratum for all cortices
# ---------------------------------------------------------------------------

class SharedTrunk(nn.Module):
    """Shared basic processing trunk (~100M params for Nano).

    Includes the canonizer, embedder, and basic transformer layers.
    Serves as the "basic stratum" shared across all cortices.

    Args:
        vocab_size: Vocabulary size.
        d_model: Model dimension.
        nhead: Number of attention heads.
        num_layers: Number of transformer encoder layers in the shared trunk.
        max_seq_len: Maximum sequence length for positional encoding.
        dim_feedforward: Feedforward dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        max_seq_len: int = 2048,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Canonizer: input normalization (lightweight)
        self.canonizer = nn.LayerNorm(d_model)

        # Embedder: token + positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.embed_norm = nn.LayerNorm(d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # Shared basic transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.trunk_layers = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device, _cache: dict = {}) -> Tensor:
        """Create a cached upper-triangular causal mask for autoregressive attention."""
        key = (seq_len, device)
        if key not in _cache:
            _cache[key] = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=device),
                diagonal=1,
            )
        return _cache[key]

    def forward(self, input_ids: Tensor) -> Tensor:
        """Process input tokens through embedder + shared trunk layers.

        Args:
            input_ids: [batch, seq_len] token IDs.

        Returns:
            Hidden states: [batch, seq_len, d_model]
        """
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        # Embed
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = x * math.sqrt(self.d_model)
        x = self.embed_norm(x)
        x = self.embed_dropout(x)

        # Canonize
        x = self.canonizer(x)

        # Shared trunk (basic stratum) — causal mask for autoregressive generation
        causal_mask = self._causal_mask(seq_len, x.device)
        x = self.trunk_layers(x, mask=causal_mask)

        return x


# ---------------------------------------------------------------------------
# DomainCortex — specialized brain region
# ---------------------------------------------------------------------------

class DomainCortex(nn.Module):
    """One domain-specialized brain region (e.g., Math, Language, Code).

    Contains intermediate, expert, and frontier strata.
    Basic stratum is the SharedTrunk (not duplicated per cortex).

    When internal_dim differs from d_model, adapter projections translate
    between the trunk's dimension and the cortex's native dimension.

    Args:
        domain_id: Domain name identifier.
        d_model: Model dimension (trunk output dimension).
        internal_dim: Internal dimension for this cortex's strata.
            If None or equal to d_model, no adapters are created.
        nhead: Number of attention heads.
        layers_per_stratum: Transformer layers per stratum.
        delta_capacity: Delta slots per stratum.
        dim_feedforward: Feedforward dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        domain_id: str,
        d_model: int = 512,
        internal_dim: int | None = None,
        nhead: int = 8,
        layers_per_stratum: int = 2,
        delta_capacity: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.domain_id = domain_id
        self.d_model = d_model
        self.internal_dim = internal_dim or d_model

        # Adapter projections (only if dimensions differ)
        if self.internal_dim != d_model:
            self.proj_in = nn.Linear(d_model, self.internal_dim)
            self.proj_out = nn.Linear(self.internal_dim, d_model)
        else:
            self.proj_in = nn.Identity()
            self.proj_out = nn.Identity()

        self.strata = nn.ModuleDict({
            "intermediate": Stratum(
                self.internal_dim, nhead, layers_per_stratum,
                depth="intermediate", delta_capacity=delta_capacity,
                dim_feedforward=dim_feedforward, dropout=dropout,
            ),
            "expert": Stratum(
                self.internal_dim, nhead, layers_per_stratum,
                depth="expert", delta_capacity=delta_capacity,
                dim_feedforward=dim_feedforward, dropout=dropout,
            ),
            "frontier": Stratum(
                self.internal_dim, nhead, layers_per_stratum,
                depth="frontier", delta_capacity=delta_capacity,
                dim_feedforward=dim_feedforward, dropout=dropout,
            ),
        })

        # Difficulty gate: determines stratum weighting given input
        self.difficulty_gate = nn.Linear(self.internal_dim, len(self.strata))

    def forward(self, x: Tensor, tau: float) -> Tensor:
        """Forward through active strata, weighted by difficulty gate.

        Args:
            x: [batch, seq, d_model] — output from shared trunk.
            tau: Current thermal level.

        Returns:
            Cortex output: [batch, seq, d_model]
        """
        # Project to internal dimension
        x_internal = self.proj_in(x)  # [batch, seq, internal_dim]

        # Compute stratum weights from input mean
        stratum_weights = F.softmax(self.difficulty_gate(x_internal.mean(dim=1)), dim=-1)

        out = torch.zeros_like(x_internal)
        for i, (name, stratum) in enumerate(self.strata.items()):
            weight = stratum_weights[:, i]  # [batch]
            if tau >= stratum.tau_min and (weight > 0.1).any():
                stratum_out = stratum(x_internal, tau)  # [batch, seq, internal_dim]
                out = out + weight.unsqueeze(-1).unsqueeze(-1) * stratum_out

        # Project back to d_model
        return self.proj_out(out)

    def stratum_names(self) -> list[str]:
        return list(self.strata.keys())


# ---------------------------------------------------------------------------
# CortexMerger — weighted combination of cortical outputs
# ---------------------------------------------------------------------------

class CortexMerger(nn.Module):
    """Weighted combination of all active cortex outputs + residual gate.

    Args:
        d_model: Model dimension.
    """

    def __init__(self, d_model: int = 512):
        super().__init__()
        self.residual_gate = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        cortex_outputs: dict[str, Tensor],
        domain_scores: dict[str, Tensor],
        trunk_output: Tensor,
    ) -> Tensor:
        """Merge cortex outputs weighted by domain scores.

        Args:
            cortex_outputs: {domain_name: [batch, seq, d_model]}
            domain_scores: {domain_name: [batch] scores in [0, 1]}
            trunk_output: [batch, seq, d_model] from shared trunk (residual).

        Returns:
            Merged output: [batch, seq, d_model]
        """
        merged = torch.zeros_like(trunk_output)
        total_weight = torch.zeros(trunk_output.shape[0], 1, 1, device=trunk_output.device)

        for name, output in cortex_outputs.items():
            if name in domain_scores:
                score = domain_scores[name]  # [batch]
                weight = score.unsqueeze(-1).unsqueeze(-1)  # [batch, 1, 1]
                merged = merged + weight * output
                total_weight = total_weight + weight

        # Normalize by total weight to avoid scaling issues
        total_weight = total_weight.clamp(min=1e-8)
        merged = merged / total_weight

        # Residual gate: preserve trunk information
        combined = torch.cat([merged, trunk_output], dim=-1)
        merged = self.residual_gate(combined)
        merged = self.norm(merged)

        return merged


# ---------------------------------------------------------------------------
# Decoder — final projection to logits
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """Representation → vocabulary logits.

    Args:
        d_model: Model dimension.
        vocab_size: Vocabulary size.
    """

    def __init__(self, d_model: int = 512, vocab_size: int = 32000):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Project hidden states to logits.

        Args:
            x: [batch, seq, d_model]

        Returns:
            Logits: [batch, seq, vocab_size]
        """
        return self.proj(self.norm(x))


# ---------------------------------------------------------------------------
# FLXNano — the complete model assembly
# ---------------------------------------------------------------------------

DEFAULT_CORTEX_NAMES = ["language", "math", "code", "science", "reasoning"]


class FLXNano(nn.Module):
    """FLX-Nano: ~145M parameter cortical, delta-native, thermally-routed LLM.

    Architecture:
        input_ids → SharedTrunk → ThalamicRouter → DomainCortices → Bridges
        → CortexMerger → MemoryController → Decoder → logits

    The thermal signal τ gates everything: strata activation, bridge bandwidth,
    memory retrieval, refinement loops.

    Args:
        vocab_size: Vocabulary size.
        d_model: Model dimension.
        nhead: Number of attention heads.
        trunk_layers: Transformer layers in shared trunk.
        layers_per_stratum: Transformer layers per cortex stratum.
        cortex_names: Domain cortex identifiers.
        cortex_dims: Per-cortex internal dimensions, e.g. {"math": 1536}.
            Cortices not listed use d_model. Enables dimension-agnostic
            cortices for transplanting donor model layers.
        delta_rank: Rank for delta decomposition.
        delta_capacity: Delta slots per stratum.
        max_seq_len: Maximum sequence length.
        dim_feedforward: Feedforward dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        nhead: int = 8,
        trunk_layers: int = 6,
        layers_per_stratum: int = 2,
        cortex_names: list[str] | None = None,
        cortex_dims: dict[str, int] | None = None,
        delta_rank: int = 32,
        delta_capacity: int = 8,
        max_seq_len: int = 2048,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.delta_rank = delta_rank
        self.cortex_names = cortex_names or DEFAULT_CORTEX_NAMES
        self.cortex_dims = cortex_dims or {}

        # Shared trunk (canonizer + embedder + basic stratum)
        self.shared_trunk = SharedTrunk(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=trunk_layers,
            max_seq_len=max_seq_len,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Domain cortices (intermediate, expert, frontier strata each)
        self.cortices = nn.ModuleDict({
            name: DomainCortex(
                domain_id=name,
                d_model=d_model,
                internal_dim=self.cortex_dims.get(name),
                nhead=nhead,
                layers_per_stratum=layers_per_stratum,
                delta_capacity=delta_capacity,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for name in self.cortex_names
        })

        # These components are set by calling attach_*() methods or
        # are initialized externally. They live as separate modules to
        # support phased training where not all components exist yet.
        self.thalamic_router: nn.Module | None = None
        self.thermal_estimator: nn.Module | None = None
        self.bridges: nn.ModuleDict | None = None
        self.memory_controller: nn.Module | None = None
        self.meta_generator: nn.Module | None = None
        self.hypothesis_head: nn.Module | None = None

        # Cortex merger + decoder
        self.cortex_merger = CortexMerger(d_model)
        self.decoder = Decoder(d_model, vocab_size)

    def attach_router(self, router: nn.Module) -> None:
        self.thalamic_router = router

    def attach_thermal(self, thermal: nn.Module) -> None:
        self.thermal_estimator = thermal

    def attach_bridges(self, bridges: nn.ModuleDict) -> None:
        self.bridges = bridges

    def attach_memory(self, memory: nn.Module) -> None:
        self.memory_controller = memory

    def attach_meta_generator(self, meta_gen: nn.Module) -> None:
        self.meta_generator = meta_gen

    def attach_hypothesis_head(self, head: nn.Module) -> None:
        self.hypothesis_head = head

    def forward(
        self,
        input_ids: Tensor,
        tau: Optional[float] = None,
        domain_scores: Optional[dict[str, Tensor]] = None,
        episodic_buffer: Optional[list[Tensor]] = None,
    ) -> Tensor:
        """Full forward pass through the cortical graph.

        Args:
            input_ids: [batch, seq_len] token IDs.
            tau: Thermal level. If None and thermal_estimator attached, computed.
            domain_scores: Pre-computed routing scores. If None, uses router.
            episodic_buffer: List of episode vectors for memory retrieval.

        Returns:
            Logits: [batch, seq_len, vocab_size]
        """
        # 1. Shared trunk: embed + basic processing
        trunk_output = self.shared_trunk(input_ids)

        # 2. Compute τ if not provided
        if tau is None and self.thermal_estimator is not None:
            tau = self.thermal_estimator(trunk_output).mean().item()
        elif tau is None:
            tau = 0.5  # default mid-range

        # 3. Route to cortices
        if domain_scores is None and self.thalamic_router is not None:
            domain_scores = self.thalamic_router(trunk_output)
        elif domain_scores is None:
            # Equal routing to all cortices (pre-Phase 0)
            batch_size = input_ids.shape[0]
            device = input_ids.device
            score = 1.0 / len(self.cortex_names)
            domain_scores = {
                name: torch.full((batch_size,), score, device=device)
                for name in self.cortex_names
            }

        # 4. Forward through active cortices
        cortex_outputs = {}
        for name, cortex in self.cortices.items():
            if name in domain_scores:
                score = domain_scores[name]
                if (score > 0.2).any():
                    cortex_outputs[name] = cortex(trunk_output, tau)

        # 5. Apply cross-cortical bridges
        if self.bridges is not None and tau >= 0.3:
            bridge_contributions = self._apply_bridges(cortex_outputs, tau)
            for name, contrib in bridge_contributions.items():
                if name in cortex_outputs:
                    cortex_outputs[name] = cortex_outputs[name] + contrib

        # 6. Merge cortex outputs
        merged = self.cortex_merger(cortex_outputs, domain_scores, trunk_output)

        # 7. Memory controller (retrieval + potential loops)
        if self.memory_controller is not None and episodic_buffer:
            self.memory_controller.reset_loop_count()
            merged, should_loop = self.memory_controller(
                merged, episodic_buffer, tau
            )
            # Refinement loops (max 3, gated by τ > 0.7)
            loop_count = 0
            while should_loop and loop_count < 3:
                # Re-merge with memory-fused representation
                merged = self.cortex_merger(cortex_outputs, domain_scores, merged)
                merged, should_loop = self.memory_controller(
                    merged, episodic_buffer, tau
                )
                loop_count += 1

        # 8. Decode to logits
        logits = self.decoder(merged)
        return logits

    def _apply_bridges(
        self,
        cortex_outputs: dict[str, Tensor],
        tau: float,
    ) -> dict[str, Tensor]:
        """Apply cross-cortical bridges between active cortices."""
        contributions: dict[str, Tensor] = {}
        if self.bridges is None:
            return contributions

        active_names = list(cortex_outputs.keys())
        for bridge_key, bridge in self.bridges.items():
            # Bridge keys use → delimiter: "source→target"
            parts = bridge_key.split("→")
            if len(parts) != 2:
                continue
            source, target = parts
            if source in active_names and target in active_names:
                contrib = bridge(cortex_outputs[source], tau)
                if target not in contributions:
                    contributions[target] = contrib
                else:
                    contributions[target] = contributions[target] + contrib

        return contributions

    def count_parameters(self) -> dict[str, int]:
        """Count parameters by component."""
        counts = {}
        counts["shared_trunk"] = sum(p.numel() for p in self.shared_trunk.parameters())
        for name, cortex in self.cortices.items():
            counts[f"cortex_{name}"] = sum(p.numel() for p in cortex.parameters())
        counts["cortex_merger"] = sum(p.numel() for p in self.cortex_merger.parameters())
        counts["decoder"] = sum(p.numel() for p in self.decoder.parameters())
        if self.thalamic_router is not None:
            counts["thalamic_router"] = sum(p.numel() for p in self.thalamic_router.parameters())
        if self.thermal_estimator is not None:
            counts["thermal_estimator"] = sum(p.numel() for p in self.thermal_estimator.parameters())
        if self.bridges is not None:
            counts["bridges"] = sum(p.numel() for p in self.bridges.parameters())
        if self.memory_controller is not None:
            counts["memory_controller"] = sum(p.numel() for p in self.memory_controller.parameters())
        if self.meta_generator is not None:
            counts["meta_generator"] = sum(p.numel() for p in self.meta_generator.parameters())
        if self.hypothesis_head is not None:
            counts["hypothesis_head"] = sum(p.numel() for p in self.hypothesis_head.parameters())
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts
