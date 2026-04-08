"""Phase 0 — Cortex Specialization Pretraining.

Forces cortices to differentiate into domain specialists via:
- Next-token prediction loss
- Diversity pressure (penalizes cortex activation overlap)
- Load-balancer loss (prevents routing collapse)
- Dropout routing (safeguard against hard collapse)

After Phase 0, each cortex has a distinct "receptive field" for knowledge domains.
"""

from __future__ import annotations

import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from ..model import FLXNano
from ..router import ThalamicRouter, diversity_loss, load_balance_loss
from .utils import EarlyStopState, evaluate_val_loss, save_checkpoint


def phase0_training_step(
    model: FLXNano,
    batch_input_ids: Tensor,
    batch_targets: Tensor,
    tau: float = 0.5,
    lambda_div: float = 0.1,
    lambda_bal: float = 0.5,
    dropout_top_prob: float = 0.1,
) -> dict[str, Tensor]:
    """One Phase 0 training step: cortex specialization.

    Args:
        model: FLXNano with thalamic router attached.
        batch_input_ids: [batch, seq_len] input token IDs.
        batch_targets: [batch, seq_len] target token IDs.
        tau: Fixed thermal level (0.5 during Phase 0).
        lambda_div: Diversity loss coefficient.
        lambda_bal: Load balance loss coefficient.
        dropout_top_prob: Probability of dropping the top-scoring cortex.

    Returns:
        Dict with loss components: pred_loss, div_loss, bal_loss, total_loss.
    """
    assert model.thalamic_router is not None, "Phase 0 requires thalamic router"

    # 1. Forward through shared trunk
    trunk_output = model.shared_trunk(batch_input_ids)

    # 2. Route via thalamic router (get raw scores for loss computation)
    domain_scores_raw = model.thalamic_router.forward_raw(trunk_output)  # [batch, K]

    # 3. Dropout routing: randomly drop top-scoring cortex
    domain_scores_gated = domain_scores_raw.clone()
    if model.training and random.random() < dropout_top_prob:
        top_cortex = domain_scores_raw.mean(dim=0).argmax()
        domain_scores_gated[:, top_cortex] = 0.0

    # 4. Build domain_scores dict for forward
    #    Phase 0: activate ALL cortices so they all receive gradients.
    #    Threshold-based gating is for inference; during specialization
    #    training every cortex must see data to differentiate.
    domain_scores = {}
    for i, name in enumerate(model.cortex_names):
        domain_scores[name] = domain_scores_gated[:, i]

    # 5. Forward through ALL cortices (no threshold during Phase 0)
    cortex_outputs = {}
    for name, cortex in model.cortices.items():
        cortex_outputs[name] = cortex(trunk_output, tau)

    # 6. Merge and decode
    merged = model.cortex_merger(cortex_outputs, domain_scores, trunk_output)
    logits = model.decoder(merged)

    # 7. Prediction loss
    pred_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        batch_targets.view(-1),
        ignore_index=-100,
    )

    # 8. Diversity loss — force cortices to specialize
    # Scale diversity loss: higher for expert/frontier strata
    div_loss = diversity_loss(domain_scores_raw)

    # 9. Load balance loss — prevent routing collapse
    bal_loss = load_balance_loss(domain_scores_raw, len(model.cortex_names))

    # 10. Combined objective
    total_loss = pred_loss + lambda_div * div_loss + lambda_bal * bal_loss

    return {
        "pred_loss": pred_loss,
        "div_loss": div_loss,
        "bal_loss": bal_loss,
        "total_loss": total_loss,
    }


def train_phase0(
    model: FLXNano,
    dataloader: DataLoader,
    val_dataloader: DataLoader | None = None,
    num_epochs: int = 10,
    lr: float = 1e-4,
    lambda_div: float = 1.0,
    lambda_bal_start: float = 0.5,
    lambda_bal_end: float = 0.05,
    dropout_top_prob: float = 0.1,
    weight_decay: float = 0.01,
    patience: int = 3,
    checkpoint_dir: str | None = None,
    checkpoint_every: int = 10_000,
    max_steps: int = 0,
    device: str = "cpu",
    log_every: int = 100,
    use_amp: bool = True,
) -> list[dict[str, float]]:
    """Full Phase 0 training loop.

    λ_bal anneals from high (force equal routing) to low (let natural
    specialization emerge).

    Args:
        model: FLXNano with thalamic router attached.
        dataloader: Training data yielding (input_ids, targets) batches.
        val_dataloader: Validation data for early stopping. If None, uses train loss.
        num_epochs: Number of training epochs.
        lr: Learning rate.
        lambda_div: Diversity loss coefficient.
        lambda_bal_start: Initial load balance coefficient.
        lambda_bal_end: Final load balance coefficient.
        dropout_top_prob: Top-cortex dropout probability.
        weight_decay: AdamW weight decay coefficient.
        patience: Early stop after N epochs without improvement on val loss.
        checkpoint_dir: If set, save per-epoch checkpoints here.
        checkpoint_every: Save a checkpoint every N steps (default 10,000).
        max_steps: Hard cap on total training steps (0 = no cap, use epochs).
        device: Training device.
        log_every: Log metrics every N steps.
        use_amp: Enable automatic mixed precision (float16) for GPU training.

    Returns:
        List of per-step loss dicts.
    """
    model = model.to(device)
    model.train()

    # Only train trunk, router, cortices, merger, decoder in Phase 0
    optimizer = torch.optim.AdamW(
        [
            {"params": model.shared_trunk.parameters()},
            {"params": model.thalamic_router.parameters()},
            {"params": model.cortices.parameters()},
            {"params": model.cortex_merger.parameters()},
            {"params": model.decoder.parameters()},
        ],
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
    history = []

    step = 0
    for epoch in range(num_epochs):
        epoch_pred_sum = 0.0
        epoch_steps = 0

        for batch in dataloader:
            input_ids = batch[0].to(device)
            targets = batch[1].to(device)

            # Anneal λ_bal
            progress = step / max(total_steps - 1, 1)
            lambda_bal = lambda_bal_start + (lambda_bal_end - lambda_bal_start) * progress

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                losses = phase0_training_step(
                    model, input_ids, targets,
                    tau=0.5,
                    lambda_div=lambda_div,
                    lambda_bal=lambda_bal,
                    dropout_top_prob=dropout_top_prob,
                )

            scaler.scale(losses["total_loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            record = {k: v.item() for k, v in losses.items()}
            record["lambda_bal"] = lambda_bal
            record["epoch"] = epoch
            record["step"] = step
            history.append(record)
            epoch_pred_sum += record["pred_loss"]
            epoch_steps += 1

            if step % log_every == 0:
                print(
                    f"Phase 0 | epoch={epoch} step={step} | "
                    f"pred={record['pred_loss']:.4f} "
                    f"div={record['div_loss']:.4f} "
                    f"bal={record['bal_loss']:.4f} "
                    f"total={record['total_loss']:.4f} "
                    f"λ_bal={lambda_bal:.4f}"
                )

            if checkpoint_dir and checkpoint_every > 0 and step > 0 and step % checkpoint_every == 0:
                save_checkpoint(
                    model, f"{checkpoint_dir}/phase0_step{step}.pt", epoch,
                    {
                        "step": step,
                        "avg_pred_loss": epoch_pred_sum / max(epoch_steps, 1),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "lambda_bal": lambda_bal,
                    },
                )

            step += 1

            if max_steps > 0 and step >= max_steps:
                print(f"Phase 0 | Reached max_steps={max_steps}, stopping.")
                break

        # End of epoch — checkpoint + early stopping
        epoch_avg_pred = epoch_pred_sum / max(epoch_steps, 1)

        # Use val loss for early stopping if available, else train loss
        if val_dataloader is not None:
            val_loss = evaluate_val_loss(model, val_dataloader, device=device)
            stop_metric = val_loss
            print(f"Phase 0 | epoch {epoch} train_pred={epoch_avg_pred:.4f} val_loss={val_loss:.4f}")
        else:
            stop_metric = epoch_avg_pred
            print(f"Phase 0 | epoch {epoch} avg pred_loss={epoch_avg_pred:.4f}")

        if checkpoint_dir:
            save_checkpoint(model, f"{checkpoint_dir}/phase0_epoch{epoch}.pt", epoch,
                            {"avg_pred_loss": epoch_avg_pred})

        if early_stop.check(stop_metric, epoch, model):
            print(f"Phase 0 | Early stop at epoch {epoch} (patience={patience})")
            break

        if max_steps > 0 and step >= max_steps:
            break

    early_stop.restore_best(model)
    return history
