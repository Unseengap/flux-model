"""FLX Autograd Bridge — torch.autograd.Function wrappers for Triton kernels.

Enables Triton kernels to participate in PyTorch's backward pass.
Without this, Triton kernels are inference-only.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .kernels import triton_delta_compose


class DeltaCompose(torch.autograd.Function):
    """Autograd-compatible delta composition.

    Forward: uses Triton kernel (if available) for fused delta composition.
    Backward: computes gradients for base weights, A, B, and confidence.

    W_out = W_base + Σ_i (conf_i * scale_i * B_i @ A_i)

    Gradients:
        ∂L/∂W_base = grad_output
        ∂L/∂A_i = conf_i * scale_i * B_i^T @ grad_output
        ∂L/∂B_i = conf_i * scale_i * grad_output @ A_i^T
        ∂L/∂conf_i = scale_i * (grad_output ⊙ (B_i @ A_i)).sum()
    """

    @staticmethod
    def forward(
        ctx,
        W_base: Tensor,
        n_deltas: int,
        *args,
    ) -> Tensor:
        """Forward pass with fused Triton kernel.

        Args packed in *args as: A_0, B_0, conf_0, scale_0, A_1, B_1, conf_1, scale_1, ...
        """
        A_list = []
        B_list = []
        conf_list = []
        scale_list = []

        for i in range(n_deltas):
            base = i * 4
            A_list.append(args[base])
            B_list.append(args[base + 1])
            conf_list.append(args[base + 2])
            scale_list.append(args[base + 3].item())

        # Save for backward
        ctx.save_for_backward(W_base, *[t for t in args[:n_deltas * 4]])
        ctx.n_deltas = n_deltas
        ctx.scale_list = scale_list

        # Use Triton kernel for forward
        W_out = triton_delta_compose(W_base, A_list, B_list, conf_list, scale_list)
        return W_out

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        saved = ctx.saved_tensors
        n_deltas = ctx.n_deltas
        scale_list = ctx.scale_list

        W_base = saved[0]

        # Gradient for W_base: identity
        grad_W_base = grad_output.clone()

        grads = [grad_W_base, None]  # W_base, n_deltas

        for i in range(n_deltas):
            base = 1 + i * 4
            A = saved[base]
            B = saved[base + 1]
            conf = saved[base + 2]
            scale = scale_list[i]

            # ∂L/∂A_i = conf_i * scale_i * B_i^T @ grad_output
            grad_A = conf.clamp(0, 1) * scale * (B.T @ grad_output)

            # ∂L/∂B_i = conf_i * scale_i * grad_output @ A_i^T
            grad_B = conf.clamp(0, 1) * scale * (grad_output @ A.T)

            # ∂L/∂conf_i = scale_i * (grad_output ⊙ (B_i @ A_i)).sum()
            grad_conf = torch.tensor(
                scale * (grad_output * (B @ A)).sum().item(),
                device=conf.device,
            )

            grads.extend([grad_A, grad_B, grad_conf, None])  # None for scale (not a tensor)

        return tuple(grads)


def delta_compose_autograd(
    W_base: Tensor,
    deltas: list,
    tau: float,
) -> Tensor:
    """Autograd-compatible delta composition via DeltaCompose.

    Replaces compose_weights() when Triton acceleration is desired
    with full backward pass support.

    Args:
        W_base: [d_out, d_in] base weight matrix.
        deltas: List of FLXDelta modules.
        tau: Current thermal level.

    Returns:
        W_effective: [d_out, d_in] composed weight matrix.
    """
    active = [d for d in deltas if d.thermal_threshold <= tau]
    if not active:
        return W_base

    args = []
    for d in active:
        args.extend([
            d.A, d.B, d.confidence,
            torch.tensor(d.scale, device=W_base.device),
        ])

    return DeltaCompose.apply(W_base, len(active), *args)
