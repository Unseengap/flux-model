"""Phase 3 — Memory System Training.

Train on conversation chains (not isolated sequences).
The model learns to compress old context into episodic vectors
and retrieve them when relevant. Cross-session memory becomes native.
"""

from __future__ import annotations

import math
import random
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from ..memory import EpisodicBuffer, EpisodicCompressor, MemoryController
from ..model import FLXNano
from .utils import EarlyStopState, configure_gpu, evaluate_val_loss, save_checkpoint


def phase3_training_step(
    model: FLXNano,
    compressor: EpisodicCompressor,
    conversation_chain: list[tuple[Tensor, Tensor]],
    compress_threshold: int = 512,
) -> dict[str, Tensor]:
    """One Phase 3 training step on a multi-turn conversation chain.

    Args:
        model: FLXNano with memory controller.
        compressor: Episodic compressor module.
        conversation_chain: List of (input_ids, target_ids) turns.
        compress_threshold: Sequence length threshold for compression.

    Returns:
        Dict with total loss across all turns.
    """
    assert model.memory_controller is not None, "Phase 3 requires memory controller"

    episodic_buffer = EpisodicBuffer()
    total_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    total_turns = 0
    running_context: Tensor | None = None

    for turn_input, turn_target in conversation_chain:
        # Build episodic list for the model
        episodes = episodic_buffer.get_all()

        # Forward with memory
        trunk_output = model.shared_trunk(turn_input)

        # Compute τ
        if model.thermal_estimator is not None:
            tau_tensor = model.thermal_estimator(trunk_output)
            tau = tau_tensor.mean().item()
        else:
            tau = 0.5

        # Route
        if model.thalamic_router is not None:
            domain_scores = model.thalamic_router(trunk_output)
        else:
            domain_scores = {
                name: torch.ones(turn_input.shape[0], device=turn_input.device)
                / len(model.cortex_names)
                for name in model.cortex_names
            }

        # Cortex forward
        cortex_outputs = {}
        for name, cortex in model.cortices.items():
            if name in domain_scores and (domain_scores[name] > 0.2).any():
                cortex_outputs[name] = cortex(trunk_output, tau)

        # Merge
        merged = model.cortex_merger(cortex_outputs, domain_scores, trunk_output)

        # Memory controller: retrieve + fuse + loop
        if len(episodes) > 0:
            model.memory_controller.reset_loop_count()
            merged, should_loop = model.memory_controller(
                merged, episodes, tau
            )
            loop_count = 0
            while should_loop and loop_count < 3:
                merged = model.cortex_merger(cortex_outputs, domain_scores, merged)
                merged, should_loop = model.memory_controller(
                    merged, episodes, tau
                )
                loop_count += 1

        # Decode
        logits = model.decoder(merged)

        # Loss for this turn
        turn_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            turn_target.view(-1),
            ignore_index=-100,
        )
        total_loss = total_loss + turn_loss
        total_turns += 1

        # Compress context into episodic vector if long enough
        if trunk_output.shape[1] >= compress_threshold or True:
            # Always compress each turn into an episode for training purposes
            episode = compressor(trunk_output.detach())
            if episode.dim() == 2:
                episode = episode[0]  # Take first in batch
            episodic_buffer.add(episode)

        # Track running context
        running_context = trunk_output.detach()

    avg_loss = total_loss / max(total_turns, 1)
    return {
        "total_loss": avg_loss,
        "pred_loss": avg_loss,
        "num_turns": torch.tensor(float(total_turns)),
        "num_episodes": torch.tensor(float(len(episodic_buffer))),
    }


def train_phase3(
    model: FLXNano,
    compressor: EpisodicCompressor,
    conversation_data: list[list[tuple[Tensor, Tensor]]],
    val_dataloader: DataLoader | None = None,
    num_epochs: int = 5,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    patience: int = 3,
    checkpoint_dir: str | None = None,
    checkpoint_every: int = 5_000,
    max_steps: int = 0,
    resume_from_checkpoint: str | None = None,
    warmup_steps: int = 500,
    gradient_accumulation_steps: int = 1,
    loss_spike_threshold: float = 3.0,
    loss_spike_patience: int = 50,
    device: str = "cpu",
    log_every: int = 10,
    use_amp: bool = True,
) -> list[dict[str, float]]:
    """Full Phase 3 training loop.

    Args:
        model: FLXNano with memory controller.
        compressor: Episodic compressor (trained jointly).
        conversation_data: List of conversations, each a list of (input, target) turns.
        val_dataloader: Validation data for early stopping. If None, uses train loss.
        num_epochs: Training epochs.
        lr: Peak learning rate (after warmup).
        weight_decay: AdamW weight decay coefficient.
        patience: Early stop after N epochs without improvement on val loss.
        checkpoint_dir: If set, save per-epoch checkpoints here.
        checkpoint_every: Save a checkpoint every N steps (default 5,000).
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
    compressor = compressor.to(device)
    model.train()
    compressor.train()

    # Freeze trunk, cortices, and decoder — Phase 0-2 weights should not drift.
    # Only train memory controller, compressor, and merger (merger needs to learn
    # to integrate memory-fused outputs).
    for p in model.shared_trunk.parameters():
        p.requires_grad = False
    for cortex in model.cortices.values():
        for p in cortex.parameters():
            p.requires_grad = False
    for p in model.decoder.parameters():
        p.requires_grad = False
    if model.thermal_estimator is not None:
        for p in model.thermal_estimator.parameters():
            p.requires_grad = False
    if model.thalamic_router is not None:
        for p in model.thalamic_router.parameters():
            p.requires_grad = False
    if model.bridges is not None:
        for p in model.bridges.parameters():
            p.requires_grad = False

    optimizer = torch.optim.AdamW(
        [
            {"params": model.memory_controller.parameters(), "lr": lr},
            {"params": compressor.parameters(), "lr": lr},
            {"params": model.cortex_merger.parameters(), "lr": lr * 0.1},
        ],
        weight_decay=weight_decay,
    )

    # Mixed precision: only use on CUDA devices
    amp_enabled = use_amp and device != "cpu" and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    early_stop = EarlyStopState(patience=patience, mode="min")
    total_steps = num_epochs * len(conversation_data)
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
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda, last_epoch=start_step
        )
        print(f"Phase 3 | Resumed from {resume_from_checkpoint} "
              f"(epoch={start_epoch}, step={start_step})")

    # Loss spike detection state
    pred_loss_ema = 0.0
    spike_counter = 0
    history: list[dict[str, float]] = []

    step = start_step
    for epoch in range(start_epoch, num_epochs):
        epoch_pred_sum = 0.0
        epoch_steps = 0

        # Shuffle conversations each epoch to avoid ordering bias
        epoch_order = list(range(len(conversation_data)))
        random.shuffle(epoch_order)

        for conv_idx_in_epoch, data_idx in enumerate(epoch_order):
            conversation = conversation_data[data_idx]
            conv_idx = conv_idx_in_epoch
            # Skip already-processed conversations when resuming mid-epoch
            if epoch == start_epoch and conv_idx < (start_step % len(conversation_data)) and resume_from_checkpoint is not None:
                continue
            # Move conversation to device
            conversation_device = [
                (inp.to(device, non_blocking=True), tgt.to(device, non_blocking=True)) for inp, tgt in conversation
            ]

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                losses = phase3_training_step(model, compressor, conversation_device)

            scaled_loss = losses["total_loss"] / gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()

            if (conv_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(compressor.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            record = {k: v.item() if isinstance(v, Tensor) else v for k, v in losses.items()}
            record["lr"] = scheduler.get_last_lr()[0]
            record["epoch"] = epoch
            record["conv_idx"] = conv_idx
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
                        f"Phase 3 | HALT: loss spike at step {step} "
                        f"(pred={pred_val:.4f}, ema={pred_loss_ema:.4f}, "
                        f"ratio={pred_val / pred_loss_ema:.1f}x). "
                        f"Saving emergency checkpoint."
                    )
                    if checkpoint_dir:
                        save_checkpoint(
                            model, f"{checkpoint_dir}/phase3_spike_step{step}.pt", epoch,
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
                    f"Phase 3 | epoch={epoch} conv={conv_idx} step={step} | "
                    f"loss={record['pred_loss']:.4f} "
                    f"turns={record['num_turns']:.0f} "
                    f"episodes={record['num_episodes']:.0f}"
                )

            if checkpoint_dir and checkpoint_every > 0 and step > 0 and step % checkpoint_every == 0:
                save_checkpoint(
                    model, f"{checkpoint_dir}/phase3_step{step}.pt", epoch,
                    {
                        "step": step,
                        "avg_pred_loss": epoch_pred_sum / max(epoch_steps, 1),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                )

            step += 1

            if max_steps > 0 and step >= max_steps:
                print(f"Phase 3 | Reached max_steps={max_steps}, stopping.")
                break

        # End of epoch — checkpoint + early stopping
        epoch_avg_pred = epoch_pred_sum / max(epoch_steps, 1)

        if val_dataloader is not None:
            val_loss = evaluate_val_loss(model, val_dataloader, device=device)
            stop_metric = val_loss
            print(f"Phase 3 | epoch {epoch} train_pred={epoch_avg_pred:.4f} val_loss={val_loss:.4f}")
        else:
            stop_metric = epoch_avg_pred
            print(f"Phase 3 | epoch {epoch} avg pred_loss={epoch_avg_pred:.4f}")

        if checkpoint_dir:
            save_checkpoint(model, f"{checkpoint_dir}/phase3_epoch{epoch}.pt", epoch,
                            {"avg_pred_loss": epoch_avg_pred})

        if early_stop.check(stop_metric, epoch, model):
            print(f"Phase 3 | Early stop at epoch {epoch} (patience={patience})")
            break

        if max_steps > 0 and step >= max_steps:
            break

    early_stop.restore_best(model)
    # Unfreeze all params before returning
    for p in model.parameters():
        p.requires_grad = True
    return history
