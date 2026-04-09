"""Phase 1 — Delta-Receptive Pretraining Within Cortices.

Trains each cortex's base weights to be a compositional substrate for deltas.
Standard pretraining optimizes "predict next token with fixed weights."
FLX pretraining optimizes "predict next token given dynamically composed weights."

Thalamic router is frozen from Phase 0.
"""

from __future__ import annotations

import math
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from ..delta import FLXDelta
from ..model import FLXNano
from .utils import EarlyStopState, evaluate_val_loss, save_checkpoint


def _sample_random_deltas(model: FLXNano, tau: float = 0.5) -> None:
    """For each stratum in each cortex, randomly activate a subset of deltas.

    During Phase 1, each stratum gets a pool of training deltas.
    We randomly sample K of them per step to train delta-receptive bases.
    """
    for cortex in model.cortices.values():
        for stratum in cortex.strata.values():
            if len(stratum.delta_stack) == 0:
                continue
            # Randomly activate a subset
            all_deltas = list(stratum.delta_stack.deltas)
            k = random.randint(0, len(all_deltas))
            active = random.sample(all_deltas, k)
            # Set confidence to 1.0 for active, 0.0 for inactive
            for delta in all_deltas:
                delta.confidence.data.fill_(1.0 if delta in active else 0.0)


def _init_delta_pool(model: FLXNano, pool_size: int = 3) -> None:
    """Initialize random delta pools for each stratum if empty.

    Creates pool_size random deltas per stratum for delta-receptive training.
    """
    for cortex in model.cortices.values():
        for stratum_name, stratum in cortex.strata.items():
            if len(stratum.delta_stack) == 0:
                for _ in range(pool_size):
                    delta = FLXDelta(
                        d_in=model.d_model,
                        d_out=model.d_model,
                        rank=model.delta_rank,
                        thermal_threshold=stratum.tau_min,
                    )
                    stratum.delta_stack.push(delta)


def phase1_training_step(
    model: FLXNano,
    batch_input_ids: Tensor,
    batch_targets: Tensor,
    tau: float = 0.5,
) -> dict[str, Tensor]:
    """One Phase 1 training step: delta-receptive pretraining.

    Args:
        model: FLXNano with frozen router.
        batch_input_ids: [batch, seq_len] input token IDs.
        batch_targets: [batch, seq_len] target token IDs.
        tau: Thermal level (varied during Phase 1).

    Returns:
        Dict with pred_loss.
    """
    # 1. Sample random delta activations per stratum
    _sample_random_deltas(model, tau)

    # 2. Forward through router (frozen) to get domain scores
    trunk_output = model.shared_trunk(batch_input_ids)

    if model.thalamic_router is not None:
        with torch.no_grad():
            domain_scores = model.thalamic_router(trunk_output)
    else:
        domain_scores = {
            name: torch.ones(batch_input_ids.shape[0], device=batch_input_ids.device)
            / len(model.cortex_names)
            for name in model.cortex_names
        }

    # 3. Forward through cortices with sampled deltas
    cortex_outputs = {}
    for name, cortex in model.cortices.items():
        if name in domain_scores and (domain_scores[name] > 0.2).any():
            cortex_outputs[name] = cortex(trunk_output, tau)

    # 4. Merge and decode
    merged = model.cortex_merger(cortex_outputs, domain_scores, trunk_output)
    logits = model.decoder(merged)

    # 5. Prediction loss
    pred_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        batch_targets.view(-1),
        ignore_index=-100,
    )

    return {"pred_loss": pred_loss, "total_loss": pred_loss}


def train_phase1(
    model: FLXNano,
    dataloader: DataLoader,
    val_dataloader: DataLoader | None = None,
    num_epochs: int = 10,
    lr: float = 5e-5,
    delta_pool_size: int = 3,
    weight_decay: float = 0.01,
    patience: int = 3,
    checkpoint_dir: str | None = None,
    checkpoint_every: int = 10_000,
    max_steps: int = 0,
    resume_from_checkpoint: str | None = None,
    warmup_steps: int = 500,
    loss_spike_threshold: float = 3.0,
    loss_spike_patience: int = 50,
    device: str = "cpu",
    log_every: int = 100,
    use_amp: bool = True,
) -> list[dict[str, float]]:
    """Full Phase 1 training loop.

    Cortex bases and delta pools train jointly.
    Thalamic router is frozen.

    Args:
        model: FLXNano with trained router (frozen).
        dataloader: Training data.
        val_dataloader: Validation data for early stopping. If None, uses train loss.
        num_epochs: Number of epochs.
        lr: Peak learning rate (after warmup).
        delta_pool_size: Deltas per stratum.
        weight_decay: AdamW weight decay coefficient.
        patience: Early stop after N epochs without improvement on val loss.
        checkpoint_dir: If set, save per-epoch checkpoints here.
        checkpoint_every: Save a checkpoint every N steps (default 10,000).
        max_steps: Hard cap on total training steps (0 = no cap, use epochs).
        resume_from_checkpoint: Path to a step checkpoint to resume from.
        warmup_steps: Linear LR warmup steps before cosine decay.
        loss_spike_threshold: Halt if pred_loss exceeds this multiple of the
            running average (e.g. 3.0 = 3x the recent average).
        loss_spike_patience: Number of consecutive spike steps before halting.
        device: Training device.
        log_every: Log interval.
        use_amp: Enable automatic mixed precision (float16) for GPU training.

    Returns:
        List of per-step loss dicts.
    """
    model = model.to(device)
    model.train()

    # Initialize delta pools
    _init_delta_pool(model, pool_size=delta_pool_size)

    # Freeze router
    if model.thalamic_router is not None:
        for p in model.thalamic_router.parameters():
            p.requires_grad = False

    # Train cortex bases, deltas, trunk, merger, decoder
    optimizer = torch.optim.AdamW(
        [
            {"params": model.shared_trunk.parameters()},
            {"params": model.cortices.parameters()},
            {"params": model.cortex_merger.parameters()},
            {"params": model.decoder.parameters()},
        ],
        lr=lr,
        weight_decay=weight_decay,
    )

    # Mixed precision: only use on CUDA devices
    amp_enabled = use_amp and device != "cpu" and torch.cuda.is_available()
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

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
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda, last_epoch=start_step
        )
        print(f"Phase 1 | Resumed from {resume_from_checkpoint} "
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
            input_ids = batch[0].to(device)
            targets = batch[1].to(device)

            # Vary τ to train across thermal regimes
            tau = random.uniform(0.1, 0.9)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                losses = phase1_training_step(model, input_ids, targets, tau=tau)

            scaler.scale(losses["total_loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            record = {k: v.item() for k, v in losses.items()}
            record["tau"] = tau
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
                        f"Phase 1 | HALT: loss spike at step {step} "
                        f"(pred={pred_val:.4f}, ema={pred_loss_ema:.4f}, "
                        f"ratio={pred_val / pred_loss_ema:.1f}x). "
                        f"Saving emergency checkpoint."
                    )
                    if checkpoint_dir:
                        save_checkpoint(
                            model, f"{checkpoint_dir}/phase1_spike_step{step}.pt", epoch,
                            {
                                "step": step,
                                "avg_pred_loss": epoch_pred_sum / max(epoch_steps, 1),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "spike_detected": True,
                            },
                        )
                    early_stop.restore_best(model)
                    return history
            else:
                spike_counter = 0

            if step % log_every == 0:
                print(
                    f"Phase 1 | epoch={epoch} step={step} | "
                    f"pred={record['pred_loss']:.4f} τ={tau:.3f}"
                )

            if checkpoint_dir and checkpoint_every > 0 and step > 0 and step % checkpoint_every == 0:
                save_checkpoint(
                    model, f"{checkpoint_dir}/phase1_step{step}.pt", epoch,
                    {
                        "step": step,
                        "avg_pred_loss": epoch_pred_sum / max(epoch_steps, 1),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                )

            step += 1

            if max_steps > 0 and step >= max_steps:
                print(f"Phase 1 | Reached max_steps={max_steps}, stopping.")
                break

        # End of epoch — checkpoint + early stopping
        epoch_avg_pred = epoch_pred_sum / max(epoch_steps, 1)

        if val_dataloader is not None:
            val_loss = evaluate_val_loss(model, val_dataloader, device=device)
            stop_metric = val_loss
            print(f"Phase 1 | epoch {epoch} train_pred={epoch_avg_pred:.4f} val_loss={val_loss:.4f}")
        else:
            stop_metric = epoch_avg_pred
            print(f"Phase 1 | epoch {epoch} avg pred_loss={epoch_avg_pred:.4f}")

        if checkpoint_dir:
            save_checkpoint(model, f"{checkpoint_dir}/phase1_epoch{epoch}.pt", epoch,
                            {"avg_pred_loss": epoch_avg_pred})

        if early_stop.check(stop_metric, epoch, model):
            print(f"Phase 1 | Early stop at epoch {epoch} (patience={patience})")
            break

        if max_steps > 0 and step >= max_steps:
            break

    early_stop.restore_best(model)

    # Unfreeze router for future phases
    if model.thalamic_router is not None:
        for p in model.thalamic_router.parameters():
            p.requires_grad = True

    return history
