"""FLX Triton Kernels — fused operations for delta composition.

Triton kernels for performance-critical operations. These are optional:
the pure PyTorch path (delta.py) is the reference implementation.
Only use these after profiling shows bottlenecks.

Requires: pip install triton
"""

from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _delta_compose_kernel(
        W_base_ptr,
        W_out_ptr,
        A_ptr,
        B_ptr,
        conf_ptr,
        scale_ptr,
        n_deltas,
        d_model: tl.constexpr,
        rank: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused delta composition: W_out = W_base + Σ(conf * scale * B @ A).

        Operates on blocks of the output weight matrix.
        Each program processes one block of rows.
        """
        pid = tl.program_id(0)
        row_start = pid * BLOCK_SIZE
        row_offsets = row_start + tl.arange(0, BLOCK_SIZE)
        row_mask = row_offsets < d_model

        for col in range(d_model):
            # Load base weight element
            idx = row_offsets * d_model + col
            w = tl.load(W_base_ptr + idx, mask=row_mask, other=0.0)

            # Accumulate delta contributions
            for i in range(n_deltas):
                conf = tl.load(conf_ptr + i)
                scale = tl.load(scale_ptr + i)

                # Compute (B @ A)[row, col] for this delta
                # B[row, :rank] @ A[:rank, col]
                acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
                for r in range(rank):
                    b_val = tl.load(
                        B_ptr + i * d_model * rank + row_offsets * rank + r,
                        mask=row_mask,
                        other=0.0,
                    )
                    a_val = tl.load(A_ptr + i * rank * d_model + r * d_model + col)
                    acc += b_val * a_val

                w += conf * scale * acc

            tl.store(W_out_ptr + idx, w, mask=row_mask)

    def triton_delta_compose(
        W_base: Tensor,
        A_list: list[Tensor],
        B_list: list[Tensor],
        conf_list: list[Tensor],
        scale_list: list[float],
    ) -> Tensor:
        """Fused delta composition using Triton kernel.

        Args:
            W_base: [d_out, d_in] base weight matrix.
            A_list: List of [rank, d_in] matrices.
            B_list: List of [d_out, rank] matrices.
            conf_list: List of scalar confidence tensors.
            scale_list: List of scale floats.

        Returns:
            W_out: [d_out, d_in] composed weight matrix.
        """
        n_deltas = len(A_list)
        if n_deltas == 0:
            return W_base.clone()

        d_out, d_in = W_base.shape
        rank = A_list[0].shape[0]

        # Stack inputs for kernel
        A_stacked = torch.stack(A_list).contiguous()  # [n_deltas, rank, d_in]
        B_stacked = torch.stack(B_list).contiguous()  # [n_deltas, d_out, rank]
        conf_stacked = torch.stack(conf_list).contiguous()  # [n_deltas]
        scale_tensor = torch.tensor(scale_list, device=W_base.device, dtype=W_base.dtype)

        W_out = torch.empty_like(W_base)
        BLOCK_SIZE = 128

        grid = ((d_out + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        _delta_compose_kernel[grid](
            W_base, W_out,
            A_stacked, B_stacked,
            conf_stacked, scale_tensor,
            n_deltas, d_out, rank, BLOCK_SIZE,
        )

        return W_out

else:

    def triton_delta_compose(
        W_base: Tensor,
        A_list: list[Tensor],
        B_list: list[Tensor],
        conf_list: list[Tensor],
        scale_list: list[float],
    ) -> Tensor:
        """Fallback pure PyTorch delta composition (Triton not available)."""
        W = W_base.clone()
        for A, B, conf, scale in zip(A_list, B_list, conf_list, scale_list):
            W = W + conf.clamp(0, 1) * scale * (B @ A)
        return W
