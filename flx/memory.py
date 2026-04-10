"""FLX Memory Subsystem — native persistence, not bolted-on RAG.

Three-tier memory:
1. Working memory — current KV cache (handled by PyTorch natively)
2. Episodic buffer — compressed prior sessions
3. Thermal history — τ trajectory (handled by ThermalEstimator)

The EpisodicCompressor and MemoryController are trained end-to-end.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class EpisodicCompressor(nn.Module):
    """Compress a KV cache chunk into a fixed-size episode vector.

    Each episode ≈ episode_dim floats. 1000 sessions fit in ~500KB.

    Args:
        d_model: Model dimension (input KV dimension).
        episode_dim: Dimension of compressed episode vectors.
        nhead: Attention heads in the encoder.
        num_layers: Encoder layers.
    """

    def __init__(
        self,
        d_model: int = 512,
        episode_dim: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.episode_dim = episode_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(d_model, episode_dim)
        self.norm = nn.LayerNorm(episode_dim)

    def forward(self, kv_chunk: Tensor) -> Tensor:
        """Compress a KV cache chunk into an episode vector.

        Args:
            kv_chunk: [batch, seq_len, d_model] or [seq_len, d_model]

        Returns:
            episode: [batch, episode_dim] or [episode_dim]
        """
        squeezed = False
        if kv_chunk.dim() == 2:
            kv_chunk = kv_chunk.unsqueeze(0)
            squeezed = True

        encoded = self.encoder(kv_chunk)
        # Mean pool over sequence dimension
        pooled = encoded.mean(dim=1)  # [batch, d_model]
        episode = self.norm(self.proj(pooled))  # [batch, episode_dim]

        if squeezed:
            episode = episode.squeeze(0)
        return episode


class EpisodicBuffer:
    """Storage for compressed episode vectors.

    Simple list-based buffer with max capacity and FIFO eviction.
    """

    def __init__(self, max_episodes: int = 1000):
        self.max_episodes = max_episodes
        self.episodes: list[Tensor] = []

    def add(self, episode: Tensor) -> None:
        """Add an episode vector to the buffer."""
        self.episodes.append(episode.detach())
        if len(self.episodes) > self.max_episodes:
            self.episodes.pop(0)

    def get_all(self) -> list[Tensor]:
        """Return all stored episodes."""
        return self.episodes

    def clear(self) -> None:
        self.episodes.clear()

    def __len__(self) -> int:
        return len(self.episodes)

    def is_empty(self) -> bool:
        return len(self.episodes) == 0


class MemoryController(nn.Module):
    """Retrieves from episodic buffer and gates refinement loops.

    The only component that can trigger a cycle back to the cortex merger.
    τ must be > 0.5 for retrieval and > 0.7 for loops.

    Args:
        d_model: Model dimension.
        episode_dim: Dimension of episode vectors.
        max_loops: Maximum refinement loops allowed.
        retrieval_tau_min: Minimum τ for episodic retrieval.
        loop_tau_min: Minimum τ for refinement loops.
    """

    def __init__(
        self,
        d_model: int = 512,
        episode_dim: int = 256,
        max_loops: int = 3,
        retrieval_tau_min: float = 0.5,
        loop_tau_min: float = 0.7,
    ):
        super().__init__()
        self.max_loops = max_loops
        self.retrieval_tau_min = retrieval_tau_min
        self.loop_tau_min = loop_tau_min

        # Query and key projections for attention over episodes
        self.query_head = nn.Linear(d_model, episode_dim)
        self.key_head = nn.Linear(episode_dim, episode_dim)

        # Project episode back to d_model for fusion
        self.episode_proj = nn.Linear(episode_dim, d_model)

        # Fuse gate: merge retrieved context with current representation
        self.fuse_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.fuse_norm = nn.LayerNorm(d_model)

        # Loop decision head
        self.loop_head = nn.Linear(d_model, 1)

        self._loop_count = 0

    def forward(
        self,
        merger_output: Tensor,
        episodic_buffer: list[Tensor],
        tau: float | Tensor,
    ) -> tuple[Tensor, bool]:
        """Retrieve from episodes and determine if a loop is needed.

        Args:
            merger_output: [batch, seq, d_model] from cortex merger.
            episodic_buffer: List of [episode_dim] or [batch, episode_dim] tensors.
            tau: Thermal level (scalar or [batch]).

        Returns:
            (fused_output, should_loop): fused representation and loop decision.
        """
        tau_val = tau.mean().item() if isinstance(tau, Tensor) else tau

        # Loop count is reset by the caller (FLXNano.forward) at the
        # start of each top-level forward pass via reset_loop_count().
        # Within a refinement loop, repeated forward() calls increment
        # _loop_count so the max-loop cap is respected.

        # Below retrieval threshold — pass through
        if tau_val < self.retrieval_tau_min or len(episodic_buffer) == 0:
            self._loop_count = 0
            return merger_output, False

        # Attend over episodes to find relevant memories
        query = self.query_head(merger_output.mean(dim=1))  # [batch, episode_dim]

        # Stack episodes and compute keys
        episodes = torch.stack(episodic_buffer, dim=0)  # [num_episodes, episode_dim]
        if episodes.dim() == 2:
            episodes = episodes.unsqueeze(0).expand(query.shape[0], -1, -1)
            # [batch, num_episodes, episode_dim]

        keys = self.key_head(episodes)  # [batch, num_episodes, episode_dim]
        d_k = keys.shape[-1]

        # Scaled dot-product attention over episodes
        attn_scores = torch.bmm(
            query.unsqueeze(1), keys.transpose(1, 2)
        ) / math.sqrt(d_k)  # [batch, 1, num_episodes]
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Retrieved context
        retrieved = torch.bmm(attn_weights, episodes).squeeze(1)  # [batch, episode_dim]
        retrieved_proj = self.episode_proj(retrieved)  # [batch, d_model]

        # Fuse retrieved context with current representation
        # Expand retrieved to sequence dimension
        retrieved_expanded = retrieved_proj.unsqueeze(1).expand_as(merger_output)
        combined = torch.cat([merger_output, retrieved_expanded], dim=-1)
        fused = self.fuse_gate(combined)
        fused = self.fuse_norm(fused + merger_output)  # residual

        # Loop decision
        should_loop = (
            tau_val > self.loop_tau_min
            and self._loop_count < self.max_loops
        )
        if should_loop:
            self._loop_count += 1
        else:
            self._loop_count = 0

        return fused, should_loop

    def reset_loop_count(self) -> None:
        self._loop_count = 0
