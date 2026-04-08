"""Phase 1 — Delta-Receptive Pretraining Within Cortices.

Trains each cortex's base weights to be a compositional substrate for deltas.
Standard pretraining optimizes "predict next token with fixed weights."
FLX pretraining optimizes "predict next token given dynamically composed weights."

Thalamic router is frozen from Phase 0.
"""

from __future__ import annotations

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
    patience: int = 3,
    checkpoint_dir: str | None = None,
    device: str = "cpu",
    log_every: int = 100,
) -> list[dict[str, float]]:
    """Full Phase 1 training loop.

    Cortex bases and delta pools train jointly.
    Thalamic router is frozen.

    Args:
        model: FLXNano with trained router (frozen).
        dataloader: Training data.
        val_dataloader: Validation data for early stopping. If None, uses train loss.
        num_epochs: Number of epochs.
        lr: Learning rate.
        delta_pool_size: Deltas per stratum.
        patience: Early stop after N epochs without improvement on val loss.
        checkpoint_dir: If set, save per-epoch checkpoints here.
        device: Training device.
        log_every: Log interval.

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
    trainable_params = []
    trainable_params.extend(model.shared_trunk.parameters())
    trainable_params.extend(model.cortices.parameters())
    trainable_params.extend(model.cortex_merger.parameters())
    trainable_params.extend(model.decoder.parameters())

    optimizer = torch.optim.AdamW(
        [p for p in trainable_params if p.requires_grad],
        lr=lr,
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

            # Vary τ to train across thermal regimes
            tau = random.uniform(0.1, 0.9)

            optimizer.zero_grad()
            losses = phase1_training_step(model, input_ids, targets, tau=tau)
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in trainable_params if p.requires_grad], 1.0
            )
            optimizer.step()

            record = {k: v.item() for k, v in losses.items()}
            record["tau"] = tau
            record["epoch"] = epoch
            record["step"] = step
            history.append(record)
            epoch_pred_sum += record["pred_loss"]
            epoch_steps += 1

            if step % log_every == 0:
                print(
                    f"Phase 1 | epoch={epoch} step={step} | "
                    f"pred={record['pred_loss']:.4f} τ={tau:.3f}"
                )

            step += 1

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

    early_stop.restore_best(model)

    # Unfreeze router for future phases
    if model.thalamic_router is not None:
        for p in model.thalamic_router.parameters():
            p.requires_grad = True

    return history
