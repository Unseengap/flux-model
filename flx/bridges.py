"""FLX Cross-Cortical Bridges — multi-domain reasoning pathways.

Learned linear projections with bandwidth gates between cortex pairs.
Thermally gated: bridges open at higher τ for cross-domain reasoning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class CrossCorticalBridge(nn.Module):
    """Learned communication channel between two cortices.

    Bandwidth-gated and τ-responsive. The model learns which cortex pairs
    need strong bridges and which are rarely co-activated.

    Args:
        d_model: Model dimension.
        source_cortex: Name of the source cortex.
        target_cortex: Name of the target cortex.
        tau_min: Minimum τ to activate bridge.
        tau_max: Maximum τ for bridge activity window.
    """

    def __init__(
        self,
        d_model: int = 512,
        source_cortex: str = "",
        target_cortex: str = "",
        tau_min: float = 0.3,
        tau_max: float = 1.0,
    ):
        super().__init__()
        self.source_cortex = source_cortex
        self.target_cortex = target_cortex
        self.tau_min = tau_min
        self.tau_max = tau_max

        self.proj = nn.Linear(d_model, d_model)
        self.bandwidth = nn.Parameter(torch.tensor(0.5))
        self.compatibility = nn.Parameter(torch.tensor(0.5))

    def forward(self, source_output: Tensor, tau: float) -> Tensor:
        """Apply bridge projection with thermal gating.

        Args:
            source_output: [batch, seq, d_model] from source cortex.
            tau: Current thermal level.

        Returns:
            Bridge contribution: [batch, seq, d_model]
        """
        # Thermal gating: bridge inactive outside its tau range
        gate = (
            torch.sigmoid(torch.tensor((tau - self.tau_min) * 10.0, device=source_output.device))
            * torch.sigmoid(torch.tensor((self.tau_max - tau) * 10.0, device=source_output.device))
        )
        bw = torch.sigmoid(self.bandwidth)
        compat = torch.sigmoid(self.compatibility)
        return gate * bw * compat * self.proj(source_output)


def build_bridges(
    cortex_names: list[str],
    d_model: int = 512,
    tau_min: float = 0.3,
) -> nn.ModuleDict:
    """Build all pairwise cross-cortical bridges.

    For N cortices, creates N*(N-1)/2 bidirectional bridges
    (stored as source_target pairs in both directions).

    Args:
        cortex_names: List of cortex domain names.
        d_model: Model dimension.
        tau_min: Minimum τ for bridge activation.

    Returns:
        ModuleDict of bridges keyed as "source_target".
    """
    bridges = {}
    for i, src in enumerate(cortex_names):
        for j, tgt in enumerate(cortex_names):
            if i < j:
                # Forward bridge
                key_fwd = f"{src}→{tgt}"
                bridges[key_fwd] = CrossCorticalBridge(
                    d_model=d_model,
                    source_cortex=src,
                    target_cortex=tgt,
                    tau_min=tau_min,
                )
                # Reverse bridge
                key_rev = f"{tgt}→{src}"
                bridges[key_rev] = CrossCorticalBridge(
                    d_model=d_model,
                    source_cortex=tgt,
                    target_cortex=src,
                    tau_min=tau_min,
                )
    return nn.ModuleDict(bridges)
