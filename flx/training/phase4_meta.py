"""Phase 4 — Online Delta Generation (Meta-Learning).

Train a meta-delta generator that takes accumulated prediction errors
and produces new delta A/B matrices targeting the correct cortex + stratum.
Self-improvement with full rollback capability.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from ..delta import FLXDelta
from ..meta_gen import MetaDeltaGenerator
from ..model import FLXNano
from .utils import EarlyStopState, evaluate_val_loss, save_checkpoint


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

    # 8. Meta-loss: train meta-gen to produce better deltas
    # Re-run forward through meta-gen (with gradients this time)
    A_grad, B_grad, metadata_grad = meta_gen(err_tensor)

    # The meta-loss encourages deltas that reduce prediction loss
    # Use the composed delta contribution as a proxy
    delta_norm = (A_grad[0].norm() + B_grad[0].norm()) * 0.001
    meta_loss = -improvement.detach() * delta_norm + delta_norm  # regularize

    return {
        "meta_loss": meta_loss,
        "total_loss": meta_loss,
        "improvement": improvement,
        "loss_before": loss_before,
        "loss_after": loss_after,
        "accepted": torch.tensor(float(accepted)),
        "target_cortex": cortex_name,
        "target_stratum": stratum_name,
    }


def train_phase4(
    model: FLXNano,
    meta_gen: MetaDeltaGenerator,
    dataloader: DataLoader,
    val_dataloader: DataLoader | None = None,
    num_epochs: int = 3,
    lr: float = 1e-4,
    buffer_threshold: int = 32,
    patience: int = 3,
    checkpoint_dir: str | None = None,
    device: str = "cpu",
    log_every: int = 10,
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
        lr: Learning rate for meta-gen.
        buffer_threshold: Minimum errors before generating a delta.
        patience: Early stop after N epochs without improvement.
        checkpoint_dir: If set, save per-epoch checkpoints here.
        device: Device.
        log_every: Log interval.

    Returns:
        Per-step loss history.
    """
    model = model.to(device)
    meta_gen = meta_gen.to(device)
    model.train()
    meta_gen.train()

    # Only train meta-generator in Phase 4
    optimizer = torch.optim.AdamW(meta_gen.parameters(), lr=lr)

    early_stop = EarlyStopState(
        patience=patience,
        mode="min" if val_dataloader is not None else "max",
    )
    error_buffer = ErrorBuffer(max_size=256, d_model=model.d_model)
    history = []
    step = 0
    accepted_count = 0
    total_generated = 0

    for epoch in range(num_epochs):
        epoch_accepted = 0
        epoch_generated = 0

        for batch in dataloader:
            input_ids = batch[0].to(device)
            targets = batch[1].to(device)

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
                optimizer.zero_grad()

                losses = phase4_training_step(
                    meta_gen, model, error_buffer,
                    input_ids, targets,
                )

                losses["total_loss"].backward()
                torch.nn.utils.clip_grad_norm_(meta_gen.parameters(), 1.0)
                optimizer.step()

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
                record["epoch"] = epoch
                record["step"] = step
                record["acceptance_rate"] = accepted_count / max(total_generated, 1)
                history.append(record)

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

            step += 1

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

    early_stop.restore_best(meta_gen)
    return history
