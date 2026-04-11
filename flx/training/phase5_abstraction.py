"""Phase 5 — Abstract Rule Induction (Few-Shot Learning).

Train a hypothesis head to induce transformation rules from minimal
demonstration pairs and apply them to novel test inputs.  Extends the
existing refinement loop with a task-scoped scratchpad for hypothesis
tracking.

Requires Phases 0–4 complete.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from ..hypothesis import HypothesisHead, TaskScratchpad
from ..meta_gen import MetaDeltaGenerator
from ..model import FLXNano
from .utils import EarlyStopState, evaluate_val_loss, save_checkpoint


# ---------------------------------------------------------------------------
# Loss functions (standalone, not class methods)
# ---------------------------------------------------------------------------

def consistency_loss(consistency: Tensor) -> Tensor:
    """Penalize low self-assessed hypothesis quality.

    Args:
        consistency: [batch] scores in (0, 1) from HypothesisHead.

    Returns:
        Scalar loss.  0 when consistency = 1, 1 when consistency = 0.
    """
    return (1.0 - consistency).mean()


def loop_efficiency_loss(num_loops: int, max_loops: int = 3) -> Tensor:
    """Penalize excessive refinement loops.

    Args:
        num_loops: Number of loops used.
        max_loops: Maximum loops allowed.

    Returns:
        Scalar in [0, 1] — fraction of max loops consumed.
    """
    return torch.tensor(num_loops / max(max_loops, 1), dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def phase5_training_step(
    model: FLXNano,
    hypothesis_head: HypothesisHead,
    demo_inputs: list[Tensor],
    demo_targets: list[Tensor],
    test_input: Tensor,
    test_target: Tensor,
    tau: float = 0.8,
    max_loops: int = 3,
    min_loops: int = 1,
    consistency_threshold: float = 0.85,
    lambda_cons: float = 0.15,
    lambda_loop: float = 0.01,
    lambda_cal: float = 0.05,
) -> dict[str, Tensor]:
    """One Phase 5 training step: few-shot rule induction.

    1. Encode demonstration pairs through shared trunk
    2. Forward test input through full pipeline at high τ
    3. Hypothesis refinement loop with scratchpad
    4. Hypothesis-conditioned prediction

    Args:
        model: FLXNano with all Phase 0-4 components attached.
        hypothesis_head: HypothesisHead to train.
        demo_inputs: List of N tensors, each [batch, seq].
        demo_targets: List of N tensors, each [batch, seq].
        test_input: [batch, seq] test input.
        test_target: [batch, seq] test target.
        tau: Thermal level (fixed high for few-shot tasks).
        max_loops: Maximum hypothesis refinement loops.
        min_loops: Minimum loops before consistency can stop early.
        consistency_threshold: Stop looping above this consistency.
        lambda_cons: Weight for consistency loss.
        lambda_loop: Weight for loop efficiency loss.
        lambda_cal: Weight for calibration loss (penalise overconfidence).

    Returns:
        Dict with pred_loss, consistency_loss, calibration_loss,
        total_loss, consistency, num_loops.
    """
    assert model.memory_controller is not None, (
        "Phase 5 requires memory controller (Phase 3 must be complete)"
    )

    device = test_input.device

    # 1. Encode demonstrations through shared trunk
    demo_embeddings = []
    with torch.no_grad():
        for demo_in in demo_inputs:
            trunk_out = model.shared_trunk(demo_in)
            demo_embeddings.append(trunk_out.mean(dim=1))  # [batch, d_model]

    demo_emb = torch.stack(demo_embeddings, dim=1)  # [batch, N, d_model]

    # 2. Forward test input through trunk → cortices → bridges → merger
    trunk_output = model.shared_trunk(test_input)

    # Route
    if model.thalamic_router is not None:
        domain_scores = model.thalamic_router(trunk_output)
    else:
        batch_size = test_input.shape[0]
        score = 1.0 / len(model.cortex_names)
        domain_scores = {
            name: torch.full((batch_size,), score, device=device)
            for name in model.cortex_names
        }

    # Cortex forward
    cortex_outputs = {}
    for name, cortex in model.cortices.items():
        if name in domain_scores and (domain_scores[name] > 0.2).any():
            cortex_outputs[name] = cortex(trunk_output, tau)

    # Bridges
    if model.bridges is not None and tau >= 0.3:
        bridge_contribs = model._apply_bridges(cortex_outputs, tau)
        for name, contrib in bridge_contribs.items():
            if name in cortex_outputs:
                cortex_outputs[name] = cortex_outputs[name] + contrib

    # Merge
    merged = model.cortex_merger(cortex_outputs, domain_scores, trunk_output)

    # 3. Hypothesis refinement loop
    scratchpad = TaskScratchpad(
        hypothesis_dim=hypothesis_head.hypothesis_dim,
    )
    model.memory_controller.reset_loop_count()

    # Build episodic buffer from demo embeddings for memory controller
    # Project demo embeddings to episode_dim via the query_head's input dim
    # The memory controller expects episode vectors of size episode_dim
    episode_dim = model.memory_controller.query_head.out_features
    demo_proj = torch.zeros(len(demo_embeddings), episode_dim, device=device)
    for i, emb in enumerate(demo_embeddings):
        # Mean-pool batch dim, then truncate/pad to episode_dim
        flat = emb.mean(dim=0)  # [d_model]
        dim = min(flat.shape[0], episode_dim)
        demo_proj[i, :dim] = flat[:dim].detach()
    episodic_buffer = [demo_proj[i] for i in range(demo_proj.shape[0])]

    num_loops = 0
    final_consistency = torch.zeros(test_input.shape[0], device=device)
    conditioning = torch.zeros(
        test_input.shape[0], 1, model.d_model, device=device,
    )

    for loop_idx in range(max_loops + 1):
        # Memory retrieval
        fused, should_loop = model.memory_controller(
            merged, episodic_buffer, tau,
        )

        # Hypothesis generation
        trajectory = scratchpad.get_trajectory(device=device)
        # Expand trajectory to batch dimension
        if trajectory.shape[0] == 1 and test_input.shape[0] > 1:
            trajectory = trajectory.expand(test_input.shape[0], -1, -1)

        hypothesis, cons_score, cond = hypothesis_head(
            fused, demo_emb, trajectory,
        )

        final_consistency = cons_score
        conditioning = cond

        # Record in scratchpad
        scratchpad.add_hypothesis(hypothesis, cons_score.mean().item())

        # Check stopping condition (enforce min_loops first)
        if loop_idx >= min_loops:
            if cons_score.mean().item() >= consistency_threshold:
                break
            if not should_loop:
                break
        if loop_idx >= max_loops:
            break

        # Re-merge for next iteration (with conditioning from hypothesis)
        merged = model.cortex_merger(
            cortex_outputs, domain_scores, fused + conditioning,
        )
        num_loops += 1

    # 4. Hypothesis-conditioned decoding
    conditioned = merged + conditioning.expand_as(merged)
    logits = model.decoder(conditioned)

    # 5. Compute losses
    pred_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        test_target.view(-1),
        ignore_index=-100,
    )
    cons_loss = consistency_loss(final_consistency)
    eff_loss = loop_efficiency_loss(num_loops, max_loops)

    # Calibration: penalise high consistency when prediction is poor.
    # Gradient flows only through consistency (pred_loss is detached).
    cal_loss = (final_consistency * pred_loss.detach()).mean()

    total_loss = (
        pred_loss
        + lambda_cons * cons_loss
        + lambda_loop * eff_loss
        + lambda_cal * cal_loss
    )

    return {
        "pred_loss": pred_loss,
        "consistency_loss": cons_loss,
        "calibration_loss": cal_loss,
        "loop_efficiency_loss": eff_loss,
        "total_loss": total_loss,
        "consistency": final_consistency.mean().detach(),
        "num_loops": torch.tensor(float(num_loops)),
    }


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def train_phase5(
    model: FLXNano,
    hypothesis_head: HypothesisHead,
    dataloader: DataLoader,
    val_dataloader: DataLoader | None = None,
    meta_gen: MetaDeltaGenerator | None = None,
    num_epochs: int = 5,
    lr: float = 3e-4,
    finetune_lr: float = 1e-5,
    tau: float = 0.8,
    max_loops: int = 3,
    min_loops: int = 1,
    consistency_threshold: float = 0.85,
    lambda_cons: float = 0.15,
    lambda_loop: float = 0.01,
    lambda_cal: float = 0.05,
    patience: int = 5,
    checkpoint_dir: str | None = None,
    device: str = "cpu",
    log_every: int = 10,
) -> list[dict[str, float]]:
    """Full Phase 5 training loop.

    HypothesisHead trains at full LR.  Cortices, bridges, memory controller,
    and decoder fine-tune at low LR.  Trunk, router, and merger are frozen.

    Args:
        model: FLXNano with Phases 0-4 complete.
        hypothesis_head: HypothesisHead to train.
        dataloader: Yields (demo_inputs, demo_targets, test_input, test_target).
            demo_inputs and demo_targets are lists of N tensors.
        val_dataloader: Validation data for early stopping.
        meta_gen: Optional meta-generator for rule-encoding delta extension.
        num_epochs: Training epochs.
        lr: Learning rate for HypothesisHead.
        finetune_lr: Learning rate for fine-tuned components.
        tau: Fixed thermal level.
        max_loops: Maximum refinement loops.
        consistency_threshold: Early-stop consistency for loops.
        lambda_cons: Consistency loss weight.
        lambda_loop: Loop efficiency loss weight.
        patience: Early stop patience.
        checkpoint_dir: Checkpoint directory.
        device: Device.
        log_every: Log interval.

    Returns:
        Per-step loss history.
    """
    model = model.to(device)
    hypothesis_head = hypothesis_head.to(device)
    model.train()
    hypothesis_head.train()

    # Freeze trunk, router, merger
    for p in model.shared_trunk.parameters():
        p.requires_grad = False
    if model.thalamic_router is not None:
        for p in model.thalamic_router.parameters():
            p.requires_grad = False
    for p in model.cortex_merger.parameters():
        p.requires_grad = False

    # Build optimizer with two param groups
    param_groups = [
        {"params": hypothesis_head.parameters(), "lr": lr},
    ]

    # Fine-tune components at low LR
    finetune_params = []
    for cortex in model.cortices.values():
        finetune_params.extend(cortex.parameters())
    if model.bridges is not None:
        finetune_params.extend(model.bridges.parameters())
    if model.memory_controller is not None:
        finetune_params.extend(model.memory_controller.parameters())
    finetune_params.extend(model.decoder.parameters())
    if meta_gen is not None:
        meta_gen = meta_gen.to(device)
        meta_gen.train()
        finetune_params.extend(meta_gen.parameters())

    if finetune_params:
        param_groups.append({"params": finetune_params, "lr": finetune_lr})

    optimizer = torch.optim.AdamW(param_groups)
    early_stop = EarlyStopState(patience=patience, mode="min")
    history: list[dict[str, float]] = []
    step = 0

    for epoch in range(num_epochs):
        epoch_pred_loss = 0.0
        epoch_cons = 0.0
        epoch_batches = 0

        for batch in dataloader:
            demo_inputs, demo_targets, test_input, test_target = batch
            # Move to device
            demo_inputs = [d.to(device) for d in demo_inputs]
            demo_targets = [d.to(device) for d in demo_targets]
            test_input = test_input.to(device)
            test_target = test_target.to(device)

            optimizer.zero_grad()

            losses = phase5_training_step(
                model, hypothesis_head,
                demo_inputs, demo_targets,
                test_input, test_target,
                tau=tau,
                max_loops=max_loops,
                min_loops=min_loops,
                consistency_threshold=consistency_threshold,
                lambda_cons=lambda_cons,
                lambda_loop=lambda_loop,
                lambda_cal=lambda_cal,
            )

            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                list(hypothesis_head.parameters())
                + [p for p in finetune_params if p.requires_grad],
                1.0,
            )
            optimizer.step()

            # Record
            record = {}
            for k, v in losses.items():
                if isinstance(v, Tensor):
                    record[k] = v.item()
                else:
                    record[k] = v
            record["epoch"] = epoch
            record["step"] = step
            history.append(record)

            epoch_pred_loss += record["pred_loss"]
            epoch_cons += record["consistency"]
            epoch_batches += 1

            if step % log_every == 0:
                print(
                    f"Phase 5 | epoch={epoch} step={step} | "
                    f"pred={record['pred_loss']:.4f} "
                    f"cons={record['consistency']:.3f} "
                    f"loops={record['num_loops']:.0f}"
                )

            step += 1

        # End of epoch
        avg_pred = epoch_pred_loss / max(epoch_batches, 1)
        avg_cons = epoch_cons / max(epoch_batches, 1)

        if val_dataloader is not None:
            val_loss = _evaluate_phase5(
                model, hypothesis_head, val_dataloader,
                tau=tau, device=device,
            )
            stop_metric = val_loss
            print(
                f"Phase 5 | epoch {epoch} pred_loss={avg_pred:.4f} "
                f"consistency={avg_cons:.3f} val_loss={val_loss:.4f}"
            )
        else:
            stop_metric = avg_pred
            print(
                f"Phase 5 | epoch {epoch} pred_loss={avg_pred:.4f} "
                f"consistency={avg_cons:.3f}"
            )

        if checkpoint_dir:
            save_checkpoint(
                hypothesis_head,
                f"{checkpoint_dir}/phase5_epoch{epoch}.pt",
                epoch,
                {"pred_loss": avg_pred, "consistency": avg_cons},
            )

        if early_stop.check(stop_metric, epoch, hypothesis_head):
            print(f"Phase 5 | Early stop at epoch {epoch} (patience={patience})")
            break

    early_stop.restore_best(hypothesis_head)

    # Unfreeze everything after Phase 5 training
    for p in model.shared_trunk.parameters():
        p.requires_grad = True
    if model.thalamic_router is not None:
        for p in model.thalamic_router.parameters():
            p.requires_grad = True
    for p in model.cortex_merger.parameters():
        p.requires_grad = True

    return history


@torch.no_grad()
def _evaluate_phase5(
    model: FLXNano,
    hypothesis_head: HypothesisHead,
    val_dataloader: DataLoader,
    tau: float = 0.8,
    device: str = "cpu",
    max_batches: int = 0,
) -> float:
    """Compute average prediction loss on a Phase 5 validation set.

    Args:
        model: FLXNano.
        hypothesis_head: HypothesisHead.
        val_dataloader: Yields (demo_inputs, demo_targets, test_input, test_target).
        tau: Fixed thermal level.
        device: Device.
        max_batches: Cap evaluation batches (0 = all).

    Returns:
        Average prediction loss.
    """
    model.eval()
    hypothesis_head.eval()
    total_loss = 0.0
    total_batches = 0

    for batch in val_dataloader:
        demo_inputs, demo_targets, test_input, test_target = batch
        demo_inputs = [d.to(device) for d in demo_inputs]
        demo_targets = [d.to(device) for d in demo_targets]
        test_input = test_input.to(device)
        test_target = test_target.to(device)

        losses = phase5_training_step(
            model, hypothesis_head,
            demo_inputs, demo_targets,
            test_input, test_target,
            tau=tau,
        )
        total_loss += losses["pred_loss"].item()
        total_batches += 1
        if max_batches > 0 and total_batches >= max_batches:
            break

    model.train()
    hypothesis_head.train()
    return total_loss / max(total_batches, 1)
