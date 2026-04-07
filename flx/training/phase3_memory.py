"""Phase 3 — Memory System Training.

Train on conversation chains (not isolated sequences).
The model learns to compress old context into episodic vectors
and retrieve them when relevant. Cross-session memory becomes native.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..memory import EpisodicBuffer, EpisodicCompressor, MemoryController
from ..model import FLXNano


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
    num_epochs: int = 5,
    lr: float = 2e-5,
    device: str = "cpu",
    log_every: int = 10,
) -> list[dict[str, float]]:
    """Full Phase 3 training loop.

    Args:
        model: FLXNano with memory controller.
        compressor: Episodic compressor (trained jointly).
        conversation_data: List of conversations, each a list of (input, target) turns.
        num_epochs: Training epochs.
        lr: Learning rate.
        device: Device.
        log_every: Log interval.

    Returns:
        Per-step loss history.
    """
    model = model.to(device)
    compressor = compressor.to(device)
    model.train()
    compressor.train()

    optimizer = torch.optim.AdamW(
        [
            {"params": model.memory_controller.parameters(), "lr": lr},
            {"params": compressor.parameters(), "lr": lr},
            # Fine-tune other components at lower LR
            {"params": model.shared_trunk.parameters(), "lr": lr * 0.1},
            {"params": model.cortices.parameters(), "lr": lr * 0.1},
            {"params": model.cortex_merger.parameters(), "lr": lr * 0.1},
            {"params": model.decoder.parameters(), "lr": lr * 0.1},
        ],
    )

    history = []
    step = 0

    for epoch in range(num_epochs):
        for conv_idx, conversation in enumerate(conversation_data):
            # Move conversation to device
            conversation_device = [
                (inp.to(device), tgt.to(device)) for inp, tgt in conversation
            ]

            optimizer.zero_grad()
            losses = phase3_training_step(model, compressor, conversation_device)
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(compressor.parameters(), 1.0)
            optimizer.step()

            record = {k: v.item() if isinstance(v, Tensor) else v for k, v in losses.items()}
            record["epoch"] = epoch
            record["conv_idx"] = conv_idx
            record["step"] = step
            history.append(record)

            if step % log_every == 0:
                print(
                    f"Phase 3 | epoch={epoch} conv={conv_idx} step={step} | "
                    f"loss={record['pred_loss']:.4f} "
                    f"turns={record['num_turns']:.0f} "
                    f"episodes={record['num_episodes']:.0f}"
                )

            step += 1

    return history
