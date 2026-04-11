"""Phase 2 — Thermal Routing + Bridge Training.

Train τ end-to-end as a differentiable signal with dual objective:
- Minimize prediction loss
- Minimize active compute (efficiency pressure)

The model learns to think harder on hard problems and faster on easy ones.
Cortex bases are frozen; train τ estimator, bridges, strata gates.
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from ..bridges import build_bridges
from ..model import FLXNano
from ..thermal import ThermalEstimator, count_active_flops
from .utils import EarlyStopState, configure_gpu, evaluate_val_loss, save_checkpoint


def phase2_training_step(
    model: FLXNano,
    batch_input_ids: Tensor,
    batch_targets: Tensor,
    lambda_compute: float = 0.01,
    tau_floor: float = 0.3,
    pred_loss_ema: float = 0.0,
) -> dict[str, Tensor]:
    """One Phase 2 training step: thermal routing.

    Args:
        model: FLXNano with thermal estimator and bridges.
        batch_input_ids: [batch, seq_len]
        batch_targets: [batch, seq_len]
        lambda_compute: Compute cost coefficient.
        tau_floor: Minimum τ for forward pass. Prevents τ collapse by ensuring
            intermediate stratum always fires and pred_loss gradient flows.
        pred_loss_ema: Running mean of pred_loss from the training loop.
            Used to compute per-batch difficulty for the bidirectional τ target.
            Pass 0.0 on the first step (uses default target of 0.5).

    Returns:
        Dict with pred_loss, compute_cost, total_loss, tau.
    """
    assert model.thermal_estimator is not None, "Phase 2 requires thermal estimator"

    # 1. Shared trunk
    trunk_output = model.shared_trunk(batch_input_ids)

    # 2. Compute τ from input
    tau_tensor = model.thermal_estimator(trunk_output)  # [batch]

    # Soft clamp: floor τ so intermediate stratum always fires.
    # Uses softplus for smooth gradient near the floor (not hard clamp
    # which kills gradient when τ < floor).
    tau_floored = tau_floor + F.softplus(tau_tensor - tau_floor, beta=5.0)
    tau = tau_floored.mean().item()

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
            src = bridge.source_cortex
            tgt = bridge.target_cortex
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

    # 10. Difficulty-responsive compute cost (bidirectional τ target).
    # v1: tau * num_strata → zero-gradient dead zone when strata=0.
    # v2: tau_floored alone → unidirectional, drives τ_raw → 0.
    # v3 (current): τ target tracks input difficulty via pred_loss vs EMA.
    # Hard batches (pred_loss > EMA) → high target → pushes τ UP.
    # Easy batches (pred_loss < EMA) → low target → pushes τ DOWN.
    if pred_loss_ema > 0:
        difficulty = torch.sigmoid(5.0 * (pred_loss.detach() - pred_loss_ema))
    else:
        difficulty = torch.tensor(0.5, device=pred_loss.device)
    tau_target = 0.3 + 0.4 * difficulty  # range [0.3, 0.7]
    compute_cost = (tau_floored.mean() - tau_target) ** 2

    # 11. Dual objective
    total_loss = pred_loss + lambda_compute * compute_cost

    return {
        "pred_loss": pred_loss,
        "compute_cost": compute_cost,
        "total_loss": total_loss,
        "tau": tau_floored.mean(),
        "tau_raw": tau_tensor.mean(),
        "tau_target": tau_target,
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
    weight_decay: float = 0.01,
    patience: int = 3,
    checkpoint_dir: str | None = None,
    checkpoint_every: int = 10_000,
    max_steps: int = 0,
    resume_from_checkpoint: str | None = None,
    warmup_steps: int = 500,
    gradient_accumulation_steps: int = 1,
    loss_spike_threshold: float = 3.0,
    loss_spike_patience: int = 50,
    device: str = "cpu",
    log_every: int = 100,
    use_amp: bool = True,
) -> list[dict[str, float]]:
    """Full Phase 2 training loop.

    Trains thermal estimator and bridges. Cortex bases frozen.

    Args:
        model: FLXNano with thermal estimator and bridges attached.
        dataloader: Difficulty-diverse training data.
        val_dataloader: Validation data for early stopping. If None, uses train loss.
        num_epochs: Training epochs.
        lr: Peak learning rate (after warmup).
        lambda_compute: Compute efficiency pressure.
        weight_decay: AdamW weight decay coefficient.
        patience: Early stop after N epochs without improvement on val loss.
        checkpoint_dir: If set, save per-epoch checkpoints here.
        checkpoint_every: Save a checkpoint every N steps (default 10,000).
        max_steps: Hard cap on total training steps (0 = no cap, use epochs).
        resume_from_checkpoint: Path to a step checkpoint to resume from.
        warmup_steps: Linear LR warmup steps before cosine decay.
        gradient_accumulation_steps: Accumulate gradients over N micro-batches
            before stepping the optimizer. Effective batch size = batch_size × N.
        loss_spike_threshold: Halt if pred_loss exceeds this multiple of the
            running average.
        loss_spike_patience: Number of consecutive spike steps before halting.
        device: Device.
        log_every: Log interval.
        use_amp: Enable automatic mixed precision (float16) for GPU training.

    Returns:
        Per-step loss history.
    """
    configure_gpu()
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
        [p for p in trainable_params if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    # Mixed precision: only use on CUDA devices
    amp_enabled = use_amp and device != "cpu" and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    early_stop = EarlyStopState(patience=patience, mode="min")
    total_steps = num_epochs * len(dataloader)
    if max_steps > 0:
        total_steps = min(total_steps, max_steps)

    # Cosine LR schedule with linear warmup
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(warmup_steps, 1)
        progress = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume from checkpoint
    start_step = 0
    start_epoch = 0
    if resume_from_checkpoint is not None:
        ckpt = torch.load(resume_from_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0)
        start_epoch = ckpt.get("epoch", 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda, last_epoch=start_step
            )
        print(f"Phase 2 | Resumed from {resume_from_checkpoint} "
              f"(epoch={start_epoch}, step={start_step})")

    # Loss spike detection state
    pred_loss_ema = 0.0
    spike_counter = 0
    history: list[dict[str, float]] = []

    step = start_step
    for epoch in range(start_epoch, num_epochs):
        epoch_pred_sum = 0.0
        epoch_steps = 0

        for batch_idx, batch in enumerate(dataloader):
            # Skip already-processed batches when resuming mid-epoch
            if epoch == start_epoch and batch_idx < (start_step % len(dataloader)) and resume_from_checkpoint is not None:
                continue
            input_ids = batch[0].to(device, non_blocking=True)
            targets = batch[1].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                losses = phase2_training_step(
                    model, input_ids, targets,
                    lambda_compute=lambda_compute,
                    pred_loss_ema=pred_loss_ema,
                )

            scaled_loss = losses["total_loss"] / gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in trainable_params if p.requires_grad], 1.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            record = {k: v.item() if isinstance(v, Tensor) else v for k, v in losses.items()}
            record["lr"] = scheduler.get_last_lr()[0]
            record["epoch"] = epoch
            record["step"] = step
            history.append(record)
            epoch_pred_sum += record["pred_loss"]
            epoch_steps += 1

            # Loss spike detection
            pred_val = record["pred_loss"]
            if step == start_step:
                pred_loss_ema = pred_val
            else:
                pred_loss_ema = 0.99 * pred_loss_ema + 0.01 * pred_val

            if pred_loss_ema > 0 and pred_val > loss_spike_threshold * pred_loss_ema:
                spike_counter += 1
                if spike_counter >= loss_spike_patience:
                    print(
                        f"Phase 2 | HALT: loss spike at step {step} "
                        f"(pred={pred_val:.4f}, ema={pred_loss_ema:.4f}, "
                        f"ratio={pred_val / pred_loss_ema:.1f}x). "
                        f"Saving emergency checkpoint."
                    )
                    if checkpoint_dir:
                        save_checkpoint(
                            model, f"{checkpoint_dir}/phase2_spike_step{step}.pt", epoch,
                            {
                                "step": step,
                                "avg_pred_loss": epoch_pred_sum / max(epoch_steps, 1),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "spike_detected": True,
                            },
                        )
                    early_stop.restore_best(model)
                    for p in model.parameters():
                        p.requires_grad = True
                    return history
            else:
                spike_counter = 0

            if step % log_every == 0:
                print(
                    f"Phase 2 | epoch={epoch} step={step} | "
                    f"pred={record['pred_loss']:.4f} "
                    f"compute={record['compute_cost']:.4f} "
                    f"τ={record['tau']:.3f} "
                    f"τ_raw={record.get('tau_raw', 0):.3f} "
                    f"τ_tgt={record.get('tau_target', 0):.3f} "
                    f"strata={record['num_strata_active']:.0f} "
                    f"bridges={record['num_bridges_active']:.0f}"
                )

            if checkpoint_dir and checkpoint_every > 0 and step > 0 and step % checkpoint_every == 0:
                save_checkpoint(
                    model, f"{checkpoint_dir}/phase2_step{step}.pt", epoch,
                    {
                        "step": step,
                        "avg_pred_loss": epoch_pred_sum / max(epoch_steps, 1),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                )

            step += 1

            if max_steps > 0 and step >= max_steps:
                print(f"Phase 2 | Reached max_steps={max_steps}, stopping.")
                break

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

        if max_steps > 0 and step >= max_steps:
            break

    early_stop.restore_best(model)

    # Unfreeze everything for future phases
    for p in model.parameters():
        p.requires_grad = True

    return history
