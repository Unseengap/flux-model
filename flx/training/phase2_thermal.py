"""Phase 2 — Thermal Routing + Bridge Training.

Train τ end-to-end as a differentiable signal with dual objective:
- Minimize prediction loss
- Minimize active compute (efficiency pressure)

The model learns to think harder on hard problems and faster on easy ones.
Cortex bases are frozen; train τ estimator, bridges, strata gates.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from ..bridges import build_bridges
from ..model import FLXNano
from ..thermal import ThermalEstimator, count_active_flops
from .utils import EarlyStopState, evaluate_val_loss, save_checkpoint


def phase2_training_step(
    model: FLXNano,
    batch_input_ids: Tensor,
    batch_targets: Tensor,
    lambda_compute: float = 0.01,
) -> dict[str, Tensor]:
    """One Phase 2 training step: thermal routing.

    Args:
        model: FLXNano with thermal estimator and bridges.
        batch_input_ids: [batch, seq_len]
        batch_targets: [batch, seq_len]
        lambda_compute: Compute cost coefficient.

    Returns:
        Dict with pred_loss, compute_cost, total_loss, tau.
    """
    assert model.thermal_estimator is not None, "Phase 2 requires thermal estimator"

    # 1. Shared trunk
    trunk_output = model.shared_trunk(batch_input_ids)

    # 2. Compute τ from input
    tau_tensor = model.thermal_estimator(trunk_output)  # [batch]
    tau = tau_tensor.mean().item()

    # 3. Route
    if model.thalamic_router is not None:
        domain_scores = model.thalamic_router(trunk_output)
    else:
        domain_scores = {
            name: torch.ones(batch_input_ids.shape[0], device=batch_input_ids.device)
            / len(model.cortex_names)
            for name in model.cortex_names
        }

    # 4. Forward through cortices (gated by τ)
    cortex_outputs = {}
    num_strata_active = 0
    for name, cortex in model.cortices.items():
        if name in domain_scores and (domain_scores[name] > 0.2).any():
            cortex_outputs[name] = cortex(trunk_output, tau)
            # Count active strata
            for stratum in cortex.strata.values():
                if tau >= stratum.tau_min:
                    num_strata_active += 1

    # 5. Apply bridges (gated by τ)
    num_bridges_active = 0
    if model.bridges is not None and tau >= 0.3:
        active_names = list(cortex_outputs.keys())
        for bridge_key, bridge in model.bridges.items():
            parts = bridge_key.split("_", 1)
            if len(parts) == 2:
                src, tgt = parts
                if src in active_names and tgt in active_names:
                    contrib = bridge(cortex_outputs[src], tau)
                    cortex_outputs[tgt] = cortex_outputs[tgt] + contrib
                    num_bridges_active += 1

    # 6. Merge
    merged = model.cortex_merger(cortex_outputs, domain_scores, trunk_output)

    # 7. Memory loop check (if controller attached)
    num_loops = 0
    if model.memory_controller is not None and tau > 0.7:
        num_loops = min(int((tau - 0.7) / 0.1), 3)

    # 8. Decode
    logits = model.decoder(merged)

    # 9. Prediction loss
    pred_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        batch_targets.view(-1),
        ignore_index=-100,
    )

    # 10. Compute cost (differentiable proxy via τ)
    # τ is differentiable, so compute_cost must depend on tau_tensor
    compute_cost = tau_tensor.mean() * (num_strata_active + 0.5 * num_bridges_active)

    # 11. Dual objective
    total_loss = pred_loss + lambda_compute * compute_cost

    return {
        "pred_loss": pred_loss,
        "compute_cost": compute_cost,
        "total_loss": total_loss,
        "tau": tau_tensor.mean(),
        "num_strata_active": torch.tensor(float(num_strata_active)),
        "num_bridges_active": torch.tensor(float(num_bridges_active)),
    }


def train_phase2(
    model: FLXNano,
    dataloader: DataLoader,
    val_dataloader: DataLoader | None = None,
    num_epochs: int = 5,
    lr: float = 3e-5,
    lambda_compute: float = 0.01,
    patience: int = 3,
    checkpoint_dir: str | None = None,
    device: str = "cpu",
    log_every: int = 100,
) -> list[dict[str, float]]:
    """Full Phase 2 training loop.

    Trains thermal estimator and bridges. Cortex bases frozen.

    Args:
        model: FLXNano with thermal estimator and bridges attached.
        dataloader: Difficulty-diverse training data.
        val_dataloader: Validation data for early stopping. If None, uses train loss.
        num_epochs: Training epochs.
        lr: Learning rate.
        lambda_compute: Compute efficiency pressure.
        patience: Early stop after N epochs without improvement on val loss.
        checkpoint_dir: If set, save per-epoch checkpoints here.
        device: Device.
        log_every: Log interval.

    Returns:
        Per-step loss history.
    """
    model = model.to(device)
    model.train()

    # Freeze cortex base weights (only train thermal + bridges + strata gates)
    for p in model.shared_trunk.parameters():
        p.requires_grad = False
    for cortex in model.cortices.values():
        for stratum in cortex.strata.values():
            for p in stratum.layers.parameters():
                p.requires_grad = False
            # Keep confidence and difficulty_gate trainable
        cortex.difficulty_gate.requires_grad_(True)

    trainable_params = []
    if model.thermal_estimator is not None:
        trainable_params.extend(model.thermal_estimator.parameters())
    if model.bridges is not None:
        trainable_params.extend(model.bridges.parameters())
    # Cortex difficulty gates and stratum confidences
    for cortex in model.cortices.values():
        trainable_params.extend(cortex.difficulty_gate.parameters())
        for stratum in cortex.strata.values():
            trainable_params.append(stratum.confidence)

    optimizer = torch.optim.AdamW(
        [p for p in trainable_params if p.requires_grad], lr=lr
    )

    early_stop = EarlyStopState(patience=patience, mode="min")
    history = []
    step = 0
    for epoch in range(num_epochs):
        epoch_pred_sum = 0.0
        epoch_steps = 0

        for batch in dataloader:
            input_ids = batch[0].to(device)
            targets = batch[1].to(device)

            optimizer.zero_grad()
            losses = phase2_training_step(
                model, input_ids, targets, lambda_compute=lambda_compute
            )
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in trainable_params if p.requires_grad], 1.0
            )
            optimizer.step()

            record = {k: v.item() if isinstance(v, Tensor) else v for k, v in losses.items()}
            record["epoch"] = epoch
            record["step"] = step
            history.append(record)
            epoch_pred_sum += record["pred_loss"]
            epoch_steps += 1

            if step % log_every == 0:
                print(
                    f"Phase 2 | epoch={epoch} step={step} | "
                    f"pred={record['pred_loss']:.4f} "
                    f"compute={record['compute_cost']:.4f} "
                    f"τ={record['tau']:.3f} "
                    f"strata={record['num_strata_active']:.0f} "
                    f"bridges={record['num_bridges_active']:.0f}"
                )

            step += 1

        # End of epoch — checkpoint + early stopping
        epoch_avg_pred = epoch_pred_sum / max(epoch_steps, 1)

        if val_dataloader is not None:
            val_loss = evaluate_val_loss(model, val_dataloader, device=device)
            stop_metric = val_loss
            print(f"Phase 2 | epoch {epoch} train_pred={epoch_avg_pred:.4f} val_loss={val_loss:.4f}")
        else:
            stop_metric = epoch_avg_pred
            print(f"Phase 2 | epoch {epoch} avg pred_loss={epoch_avg_pred:.4f}")

        if checkpoint_dir:
            save_checkpoint(model, f"{checkpoint_dir}/phase2_epoch{epoch}.pt", epoch,
                            {"avg_pred_loss": epoch_avg_pred})

        if early_stop.check(stop_metric, epoch, model):
            print(f"Phase 2 | Early stop at epoch {epoch} (patience={patience})")
            break

    early_stop.restore_best(model)

    # Unfreeze everything for future phases
    for p in model.parameters():
        p.requires_grad = True

    return history
