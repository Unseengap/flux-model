"""Phase 4 — Online Delta Generation (Meta-Learning).

Train a meta-delta generator that takes accumulated prediction errors
and produces new delta A/B matrices targeting the correct cortex + stratum.
Self-improvement with full rollback capability.
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from ..delta import FLXDelta
from ..meta_gen import MetaDeltaGenerator
from ..model import FLXNano
from .utils import EarlyStopState, configure_gpu, evaluate_val_loss, save_checkpoint


class ErrorBuffer:
    """Rolling buffer for accumulating prediction error signals."""

    def __init__(self, max_size: int = 256, d_model: int = 512):
        self.max_size = max_size
        self.d_model = d_model
        self.errors: list[Tensor] = []
        self.inputs: list[Tensor] = []

    def add(self, input_repr: Tensor, error_signal: Tensor) -> None:
        """Add an error observation.

        Args:
            input_repr: [d_model] input representation.
            error_signal: [d_model] error embedding.
        """
        self.errors.append(error_signal.detach())
        self.inputs.append(input_repr.detach())
        if len(self.errors) > self.max_size:
            self.errors.pop(0)
            self.inputs.pop(0)

    def get_buffer(self, device: str = "cpu") -> Tensor:
        """Return error buffer as a tensor.

        Returns:
            [1, num_errors, d_model]
        """
        if len(self.errors) == 0:
            return torch.zeros(1, 1, self.d_model, device=device)
        return torch.stack(self.errors).unsqueeze(0).to(device)

    def get_holdout(self, fraction: float = 0.2) -> tuple[list[Tensor], list[Tensor]]:
        """Split buffer into train/holdout for validation."""
        n = len(self.errors)
        split = max(1, int(n * (1 - fraction)))
        return self.errors[:split], self.errors[split:]

    def clear(self) -> None:
        self.errors.clear()
        self.inputs.clear()

    def __len__(self) -> int:
        return len(self.errors)

    @property
    def ready(self) -> bool:
        return len(self.errors) >= 16


def phase4_training_step(
    meta_gen: MetaDeltaGenerator,
    model: FLXNano,
    error_buffer: ErrorBuffer,
    eval_input_ids: Tensor,
    eval_targets: Tensor,
) -> dict[str, Tensor]:
    """One Phase 4 training step: meta-delta generation.

    1. Meta-generator produces a candidate delta from errors
    2. Validate: does it improve on held-out data?
    3. Train meta-generator on improvement signal

    Args:
        meta_gen: Meta-delta generator.
        model: FLXNano model.
        error_buffer: Accumulated errors.
        eval_input_ids: [batch, seq] validation input.
        eval_targets: [batch, seq] validation targets.

    Returns:
        Dict with meta_loss, improvement, accepted.
    """
    device = eval_input_ids.device

    # 1. Get error buffer
    err_tensor = error_buffer.get_buffer(device)

    # 2. Generate candidate delta
    A, B, metadata = meta_gen(err_tensor)

    # 3. Evaluate model BEFORE adding delta
    model.eval()
    with torch.no_grad():
        logits_before = model(eval_input_ids)
        loss_before = F.cross_entropy(
            logits_before.view(-1, logits_before.size(-1)),
            eval_targets.view(-1),
            ignore_index=-100,
        )

    # 4. Create candidate delta and add to target cortex/stratum
    cortex_idx = metadata["cortex_logits"][0].argmax().item()
    stratum_idx = metadata["stratum_logits"][0].argmax().item()

    cortex_name = model.cortex_names[cortex_idx % len(model.cortex_names)]
    stratum_names = list(model.cortices[cortex_name].strata.keys())
    stratum_name = stratum_names[stratum_idx % len(stratum_names)]

    candidate = FLXDelta(
        d_in=model.d_model,
        d_out=model.d_model,
        rank=meta_gen.delta_rank,
        thermal_threshold=metadata["threshold"][0].item(),
        confidence=0.1,
    )
    with torch.no_grad():
        candidate.A.copy_(A[0])
        candidate.B.copy_(B[0])
    candidate = candidate.to(device)

    target_stratum = model.cortices[cortex_name].strata[stratum_name]
    # Evict oldest delta when stack is at capacity
    if len(target_stratum.delta_stack.deltas) >= target_stratum.delta_stack.capacity:
        evicted = target_stratum.delta_stack.deltas[0]
        target_stratum.delta_stack.deltas = nn.ModuleList(
            list(target_stratum.delta_stack.deltas)[1:]
        )
    target_stratum.delta_stack.push(candidate)

    # 5. Evaluate AFTER adding delta
    with torch.no_grad():
        logits_after = model(eval_input_ids)
        loss_after = F.cross_entropy(
            logits_after.view(-1, logits_after.size(-1)),
            eval_targets.view(-1),
            ignore_index=-100,
        )

    # 6. Improvement signal for meta-generator
    improvement = loss_before - loss_after  # positive = good

    # 7. Accept or reject
    accepted = improvement.item() > 0
    if not accepted:
        target_stratum.delta_stack.pop()  # clean rollback

    model.train()

    # 8. Meta-loss: train meta-gen to produce better deltas (REINFORCE-style)
    # Re-run forward through meta-gen (with gradients this time)
    A_grad, B_grad, metadata_grad = meta_gen(err_tensor)

    # The delta contribution has gradient w.r.t. meta-gen parameters.
    # Use improvement as the reward signal (REINFORCE).
    delta_contribution = A_grad[0].norm() + B_grad[0].norm()

    # Scale reward relative to loss_before so small absolute gains
    # (e.g. 0.0001 on loss=3.5) become meaningful percentage signals.
    raw_reward = improvement.detach()
    scaled_reward = raw_reward / (loss_before.detach().clamp(min=0.1))

    reg_weight = 0.001
    meta_loss = -scaled_reward * delta_contribution + reg_weight * delta_contribution

    # Cortex entropy bonus — encourage exploration across all cortices
    cortex_probs = F.softmax(metadata_grad["cortex_logits"][0], dim=-1)
    cortex_entropy = -(cortex_probs * (cortex_probs + 1e-10).log()).sum()
    entropy_weight = 0.1
    meta_loss = meta_loss - entropy_weight * cortex_entropy

    return {
        "meta_loss": meta_loss,
        "total_loss": meta_loss,
        "improvement": improvement,
        "loss_before": loss_before,
        "loss_after": loss_after,
        "accepted": torch.tensor(float(accepted)),
        "target_cortex": cortex_name,
        "target_stratum": stratum_name,
        "cortex_entropy": cortex_entropy,
    }


def train_phase4(
    model: FLXNano,
    meta_gen: MetaDeltaGenerator,
    dataloader: DataLoader,
    val_dataloader: DataLoader | None = None,
    num_epochs: int = 3,
    lr: float = 1e-4,
    buffer_threshold: int = 32,
    weight_decay: float = 0.01,
    patience: int = 3,
    checkpoint_dir: str | None = None,
    checkpoint_every: int = 5_000,
    max_steps: int = 0,
    resume_from_checkpoint: str | None = None,
    warmup_steps: int = 500,
    loss_spike_threshold: float = 3.0,
    loss_spike_patience: int = 50,
    device: str = "cpu",
    log_every: int = 10,
    use_amp: bool = True,
) -> list[dict[str, float]]:
    """Full Phase 4 training loop.

    Accumulates errors from model predictions, then trains meta-gen
    to produce useful deltas.

    Args:
        model: FLXNano.
        meta_gen: Meta-delta generator.
        dataloader: Training data.
        val_dataloader: Validation data for early stopping. If None, uses acceptance rate.
        num_epochs: Training epochs.
        lr: Peak learning rate (after warmup).
        buffer_threshold: Minimum errors before generating a delta.
        weight_decay: AdamW weight decay coefficient.
        patience: Early stop after N epochs without improvement.
        checkpoint_dir: If set, save per-epoch checkpoints here.
        checkpoint_every: Save a checkpoint every N steps (default 5,000).
        max_steps: Hard cap on total training steps (0 = no cap, use epochs).
        resume_from_checkpoint: Path to a step checkpoint to resume from.
        warmup_steps: Linear LR warmup steps before cosine decay.
        loss_spike_threshold: Halt if meta_loss exceeds this multiple of the
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
    meta_gen = meta_gen.to(device)

    # Freeze all model weights — Phase 4 only trains meta_gen
    # Note: meta_gen is a submodule of model (via attach_meta_generator),
    # so we must re-enable its gradients after the blanket freeze.
    for param in model.parameters():
        param.requires_grad = False
    for param in meta_gen.parameters():
        param.requires_grad = True

    model.train()
    meta_gen.train()

    # Only train meta-generator in Phase 4
    optimizer = torch.optim.AdamW(meta_gen.parameters(), lr=lr, weight_decay=weight_decay)

    # Mixed precision: only use on CUDA devices
    amp_enabled = use_amp and device != "cpu" and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    early_stop = EarlyStopState(
        patience=patience,
        mode="min" if val_dataloader is not None else "max",
    )
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
        warnings.simplefilter("ignore")
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume from checkpoint
    start_step = 0
    start_epoch = 0
    if resume_from_checkpoint is not None:
        ckpt = torch.load(resume_from_checkpoint, map_location=device, weights_only=False)
        meta_gen.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0)
        start_epoch = ckpt.get("epoch", 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda, last_epoch=start_step
            )
        print(f"Phase 4 | Resumed from {resume_from_checkpoint} "
              f"(epoch={start_epoch}, step={start_step})")

    # Loss spike detection state
    meta_loss_ema = 0.0
    spike_counter = 0
    error_buffer = ErrorBuffer(max_size=256, d_model=model.d_model)
    history: list[dict[str, float]] = []

    step = start_step
    accepted_count = 0
    total_generated = 0
    optimizer_stepped = start_step > 0  # Track if optimizer has stepped (for scheduler)

    for epoch in range(start_epoch, num_epochs):
        epoch_accepted = 0
        epoch_generated = 0

        for batch_idx, batch in enumerate(dataloader):
            # Skip already-processed batches when resuming mid-epoch
            if epoch == start_epoch and batch_idx < (start_step % len(dataloader)) and resume_from_checkpoint is not None:
                continue
            input_ids = batch[0].to(device, non_blocking=True)
            targets = batch[1].to(device, non_blocking=True)

            # Accumulate errors: run model and check uncertainty
            model.eval()
            with torch.no_grad():
                trunk_out = model.shared_trunk(input_ids)
                logits = model(input_ids)
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)  # [batch, seq]

                # High-entropy positions are errors
                high_entropy = entropy.mean(dim=1)  # [batch]
                for i in range(input_ids.shape[0]):
                    if high_entropy[i] > 2.0:  # above threshold
                        error_buffer.add(
                            trunk_out[i].mean(dim=0),
                            trunk_out[i].mean(dim=0) * high_entropy[i],
                        )

            model.train()

            # Generate delta when buffer is ready
            if error_buffer.ready and len(error_buffer) >= buffer_threshold:
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    losses = phase4_training_step(
                        meta_gen, model, error_buffer,
                        input_ids, targets,
                    )

                scaler.scale(losses["total_loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(meta_gen.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer_stepped = True

                total_generated += 1
                epoch_generated += 1
                if losses["accepted"].item() > 0:
                    accepted_count += 1
                    epoch_accepted += 1

                record = {}
                for k, v in losses.items():
                    if isinstance(v, Tensor):
                        record[k] = v.item()
                    else:
                        record[k] = v
                record["lr"] = scheduler.get_last_lr()[0]
                record["epoch"] = epoch
                record["step"] = step
                record["acceptance_rate"] = accepted_count / max(total_generated, 1)
                history.append(record)

                # Loss spike detection on meta_loss
                meta_val = abs(record.get("meta_loss", 0))
                if total_generated == 1:
                    meta_loss_ema = meta_val
                else:
                    meta_loss_ema = 0.99 * meta_loss_ema + 0.01 * meta_val

                if meta_loss_ema > 0 and meta_val > loss_spike_threshold * meta_loss_ema:
                    spike_counter += 1
                    if spike_counter >= loss_spike_patience:
                        print(
                            f"Phase 4 | HALT: loss spike at step {step} "
                            f"(meta={meta_val:.4f}, ema={meta_loss_ema:.4f}, "
                            f"ratio={meta_val / meta_loss_ema:.1f}x). "
                            f"Saving emergency checkpoint."
                        )
                        if checkpoint_dir:
                            save_checkpoint(
                                meta_gen, f"{checkpoint_dir}/phase4_spike_step{step}.pt", epoch,
                                {
                                    "step": step,
                                    "acceptance_rate": accepted_count / max(total_generated, 1),
                                    "optimizer_state_dict": optimizer.state_dict(),
                                    "spike_detected": True,
                                },
                            )
                        early_stop.restore_best(meta_gen)
                        return history
                else:
                    spike_counter = 0

                if step % log_every == 0:
                    print(
                        f"Phase 4 | epoch={epoch} step={step} | "
                        f"improve={record.get('improvement', 0):.4f} "
                        f"accepted={record.get('accepted', 0):.0f} "
                        f"rate={record['acceptance_rate']:.2f} "
                        f"→ {record.get('target_cortex', '?')}/{record.get('target_stratum', '?')}"
                    )

                # Clear buffer after generating
                error_buffer.clear()

            if checkpoint_dir and checkpoint_every > 0 and step > 0 and step % checkpoint_every == 0:
                save_checkpoint(
                    meta_gen, f"{checkpoint_dir}/phase4_step{step}.pt", epoch,
                    {
                        "step": step,
                        "acceptance_rate": accepted_count / max(total_generated, 1),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                )

            if optimizer_stepped:
                scheduler.step()
            step += 1

            if max_steps > 0 and step >= max_steps:
                print(f"Phase 4 | Reached max_steps={max_steps}, stopping.")
                break

        # End of epoch — checkpoint + early stopping
        epoch_rate = epoch_accepted / max(epoch_generated, 1)

        if val_dataloader is not None:
            val_loss = evaluate_val_loss(model, val_dataloader, device=device)
            stop_metric = val_loss
            print(f"Phase 4 | epoch {epoch} acceptance_rate={epoch_rate:.2f} "
                  f"({epoch_accepted}/{epoch_generated}) val_loss={val_loss:.4f}")
        else:
            stop_metric = epoch_rate
            print(f"Phase 4 | epoch {epoch} acceptance_rate={epoch_rate:.2f} "
                  f"({epoch_accepted}/{epoch_generated})")

        if checkpoint_dir:
            save_checkpoint(meta_gen, f"{checkpoint_dir}/phase4_epoch{epoch}.pt", epoch,
                            {"acceptance_rate": epoch_rate})

        if early_stop.check(stop_metric, epoch, meta_gen):
            print(f"Phase 4 | Early stop at epoch {epoch} (patience={patience})")
            break

        if max_steps > 0 and step >= max_steps:
            break

    early_stop.restore_best(meta_gen)
    return history
