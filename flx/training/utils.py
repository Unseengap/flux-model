"""Training utilities — checkpointing, early stopping, validation, GPU config."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split


def configure_gpu() -> None:
    """Enable GPU-specific optimizations. Call once before training.

    - TF32 matmul/cuDNN: ~3x faster on Ampere+ (A100, L4) with negligible precision loss.
    - cuDNN benchmark: autoselects fastest convolution algorithm for fixed input sizes.
    """
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


@dataclass
class EarlyStopState:
    """Tracks best metric and patience counter for early stopping.

    Args:
        patience: Number of epochs without improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
        mode: 'min' (lower is better, e.g. loss) or 'max' (higher is better).
    """
    patience: int = 3
    min_delta: float = 1e-4
    mode: str = "min"  # 'min' for loss, 'max' for improvement/acceptance rate
    best_value: float = field(init=False, default=float("inf"))
    best_epoch: int = field(init=False, default=0)
    counter: int = field(init=False, default=0)
    best_state: Optional[dict] = field(init=False, default=None, repr=False)
    stopped: bool = field(init=False, default=False)

    def __post_init__(self):
        if self.mode == "max":
            self.best_value = float("-inf")

    def check(self, value: float, epoch: int, model: nn.Module) -> bool:
        """Check if training should stop. Returns True if should stop.

        Saves best model state_dict when a new best is found.
        """
        improved = False
        if self.mode == "min":
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.stopped = True
            return True
        return False

    def restore_best(self, model: nn.Module) -> None:
        """Restore model to the best checkpoint."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            print(f"  Restored best model from epoch {self.best_epoch} "
                  f"(best={self.best_value:.4f})")


def save_checkpoint(model: nn.Module, path: str, epoch: int,
                    extras: Optional[dict] = None) -> None:
    """Save a training checkpoint to disk."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    }
    if extras:
        checkpoint.update(extras)
    torch.save(checkpoint, path)
    print(f"  Checkpoint saved → {path} (epoch {epoch})")


@torch.no_grad()
def evaluate_val_loss(
    model: nn.Module,
    val_dataloader: DataLoader,
    device: str = "cpu",
    max_batches: int = 0,
    use_amp: bool = True,
) -> float:
    """Compute average cross-entropy loss on a validation set.

    Args:
        model: FLXNano model (must have a forward that returns logits).
        val_dataloader: Validation DataLoader yielding (input_ids, targets).
        device: Device.
        max_batches: Cap evaluation to this many batches (0 = all).
        use_amp: Use automatic mixed precision during validation.

    Returns:
        Average cross-entropy loss over the validation set.
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0
    amp_enabled = use_amp and device != "cpu" and torch.cuda.is_available()

    for batch in val_dataloader:
        input_ids = batch[0].to(device, non_blocking=True)
        targets = batch[1].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        total_loss += loss.item()
        total_batches += 1
        if max_batches > 0 and total_batches >= max_batches:
            break

    model.train()
    return total_loss / max(total_batches, 1)


def make_train_val_split(
    dataset: Dataset,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """Split a dataset into train and validation subsets.

    Args:
        dataset: Full dataset to split.
        val_fraction: Fraction of data for validation (default 10%).
        seed: Random seed for reproducibility.

    Returns:
        (train_dataset, val_dataset) tuple.
    """
    n = len(dataset)
    val_size = max(1, int(n * val_fraction))
    train_size = n - val_size
    gen = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=gen)
