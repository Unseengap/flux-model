"""FLX Thermal System — the τ signal.

τ ∈ (0, 1) is a differentiable arousal signal computed by a learned network.
It gates: strata activation, bridge bandwidth, memory retrieval, loop count.
"""

from __future__ import annotations

from collections import deque

import torch
import torch.nn as nn
from torch import Tensor


class ThermalEstimator(nn.Module):
    """Computes τ from input hidden state + thermal history.

    τ = blend(surprise, context_novelty, history)

    Args:
        d_model: Model dimension.
        history_len: Number of past τ values to track.
    """

    def __init__(self, d_model: int = 512, history_len: int = 32):
        super().__init__()
        self.surprise_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )
        self.context_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )
        self.history_weight = nn.Parameter(torch.tensor(0.3))
        self.history_len = history_len
        self._tau_history: deque[float] = deque(maxlen=history_len)

    def forward(self, hidden_state: Tensor) -> Tensor:
        """Compute τ from hidden state.

        Args:
            hidden_state: [batch, seq_len, d_model] from shared trunk.

        Returns:
            tau: [batch] scalar in (0, 1).
        """
        # Surprise: how unexpected is this input?
        surprise = torch.sigmoid(
            self.surprise_head(hidden_state.mean(dim=1))
        ).squeeze(-1)  # [batch]

        # Context novelty: from first token representation
        context_novelty = torch.sigmoid(
            self.context_head(hidden_state[:, 0])
        ).squeeze(-1)  # [batch]

        # Blend surprise and context
        raw_tau = 0.6 * surprise + 0.4 * context_novelty

        # Smooth with history
        hw = torch.sigmoid(self.history_weight)
        if len(self._tau_history) > 0:
            hist_mean = sum(self._tau_history) / len(self._tau_history)
            tau = (1 - hw) * raw_tau + hw * hist_mean
        else:
            tau = raw_tau

        # Update history (use batch mean for tracking)
        self._tau_history.append(tau.mean().item())

        return tau

    def reset_history(self) -> None:
        """Clear thermal history (e.g., new session)."""
        self._tau_history.clear()

    def get_history(self) -> list[float]:
        """Return thermal history as a list."""
        return list(self._tau_history)

    def set_history(self, history: list[float]) -> None:
        """Restore thermal history from saved state."""
        self._tau_history.clear()
        for val in history[-self.history_len :]:
            self._tau_history.append(val)


def count_active_flops(
    tau: float,
    num_strata_active: int,
    num_bridges_active: int,
    num_loops: int,
    base_flops_per_stratum: int = 1,
) -> float:
    """Estimate relative compute cost for the thermal efficiency objective.

    Used in Phase 2 training: loss = pred_loss + λ * compute_cost.

    Args:
        tau: Current thermal level.
        num_strata_active: Number of strata that fired.
        num_bridges_active: Number of bridges that opened.
        num_loops: Number of refinement loops.
        base_flops_per_stratum: Normalized cost per stratum.

    Returns:
        Normalized compute cost scalar.
    """
    stratum_cost = num_strata_active * base_flops_per_stratum
    bridge_cost = num_bridges_active * 0.5 * base_flops_per_stratum
    loop_cost = num_loops * (stratum_cost + bridge_cost)
    return stratum_cost + bridge_cost + loop_cost
