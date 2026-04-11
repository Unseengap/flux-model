# Phase 3 Debugging — Memory System

## Goal

Phase 3 trains the episodic compressor and memory controller on multi-turn conversations. The model learns to compress past turn context into episode vectors and retrieve relevant memories for later turns. Trunk, cortices, router, thermal estimator, bridges, and decoder are frozen from Phases 0–2.

## What "working" looks like

- `loss`: Starts ~4.5–4.7 (higher than Phase 2 due to conversation data + freshly initialized memory). Should decline to **3.5–4.0** as memory retrieval improves later-turn predictions.
- `turns`: 2–14 per conversation (from UltraChat). Higher-turn convos benefit more from memory.
- `episodes`: Should equal `turns` (each turn is compressed into one episode).
- Later turns in a conversation should have progressively lower per-turn loss (more context available).

## Problem History

### Attempt 1: Trunk/cortex fine-tuning + unshuffled data (FAILED — loss regression)

**File**: `flx/training/phase3_memory.py` — `train_phase3()`

Original optimizer included trunk, cortices, merger, and decoder at `lr * 0.1`:
```python
optimizer = torch.optim.AdamW([
    {"params": model.memory_controller.parameters(), "lr": lr},
    {"params": compressor.parameters(), "lr": lr},
    {"params": model.shared_trunk.parameters(), "lr": lr * 0.1},  # BAD
    {"params": model.cortices.parameters(), "lr": lr * 0.1},      # BAD
    {"params": model.cortex_merger.parameters(), "lr": lr * 0.1},
    {"params": model.decoder.parameters(), "lr": lr * 0.1},       # BAD
])
```

Also, `conversation_data` was iterated in UltraChat's original dataset order — no shuffling between epochs.

**Training evidence** (loss dips then regresses):
```
step=0    | loss=4.7033 turns=8
step=300  | loss=3.9776 turns=6   ← initial improvement
step=1100 | loss=3.8586 turns=8   ← best point
step=1400 | loss=4.6101 turns=4   ← regression starts
step=1900 | loss=4.9019 turns=4   ← worse than step 0
```

**Two root causes**:

1. **Trunk/cortex weight drift**: Even at `lr * 0.1 = 2e-6`, the Phase 0–2 learned language model weights were drifting over ~2000 steps. Phase 2 explicitly froze trunk and cortex bases for this exact reason. Phase 3 unfroze them, and the conversation-style training data (UltraChat) pushed the weights away from the diverse Phase 0/1 distribution. The initial improvement was the memory controller learning; the regression was trunk degradation overtaking it.

2. **Unshuffled data**: UltraChat conversations were iterated in dataset order. Any ordering bias (e.g., topic clustering, difficulty progression) appeared as training trends rather than being averaged out. Each epoch saw the same order.

### Attempt 2: Frozen trunk + shuffled data (CONVERGED — val_loss=3.31)

**File**: `flx/training/phase3_memory.py` — `train_phase3()`

Two changes:

**1. Freeze everything except memory controller, compressor, and merger:**
```python
# Frozen: trunk, cortices, decoder, thermal estimator, router, bridges
for p in model.shared_trunk.parameters():
    p.requires_grad = False
# ... (same for cortices, decoder, thermal, router, bridges)

# Trainable: memory_controller (full lr), compressor (full lr), merger (lr * 0.1)
optimizer = torch.optim.AdamW([
    {"params": model.memory_controller.parameters(), "lr": lr},
    {"params": compressor.parameters(), "lr": lr},
    {"params": model.cortex_merger.parameters(), "lr": lr * 0.1},
])
```

**Why merger at low LR**: The merger needs to learn to integrate memory-fused outputs (it now receives output from the memory controller's `fuse_gate`). But it was already trained in Phase 2 and shouldn't change much.

**2. Shuffle conversations each epoch:**
```python
epoch_order = list(range(len(conversation_data)))
random.shuffle(epoch_order)
for conv_idx_in_epoch, data_idx in enumerate(epoch_order):
    conversation = conversation_data[data_idx]
```

**Expected behavior after fix**:
- Loss should monotonically decrease (no regression) since frozen weights can't drift
- Shuffling removes any dataset ordering artifacts
- Memory controller gets clean gradient signal: it either helps or not, the trunk doesn't change under it

## Training Outcome — CONVERGED at 20k steps (1 epoch)

**Config**: `num_epochs=5, lr=2e-5, patience=3, max_steps=20_000, checkpoint_every=5_000`

Hit `max_steps=20_000` during epoch 0 (20K conversations out of 20K dataset = full pass).

**Final metrics**:
- `train_pred`: 3.93 (epoch average)
- `val_loss`: **3.31** (Phase 2 cached data, 23,759 samples)
- `final step loss`: 3.63

**Loss trajectory** (500-step moving average):
```
steps 0–2000:    ~4.1  (memory controller learning from scratch)
steps 2000–5000: ~3.9  (retrieval starting to help)
steps 5000–10000: ~3.85 (steady improvement)
steps 10000–15000: ~3.80 (diminishing returns)
steps 15000–20000: ~3.78 (plateau — convergence)
```

**No regression** — the frozen trunk fix worked. Loss monotonically decreased (smoothed).

**Why val_loss (3.31) < train avg (3.93)**: The val loader uses Phase 2 difficulty-stratified data (standard single-sequence prediction), which the model handles well from Phases 0–2. Conversation data is harder because UltraChat contains diverse multi-turn dialogue.

**Could benefit from longer training**: With 20K conversations at ~6 turns average, the model saw ~120K turn-level training examples in one epoch. At 100K–200K steps (5–10 epochs with reshuffling), the memory controller would see each conversation multiple times with different random orderings, likely pushing loss below 3.5. This is acceptable for Phase 3's goal — memory retrieval is functional, further gains are diminishing.

**Saved as**: `nano_phase3.flx` (best model restored from epoch 0, val_loss=3.31)

## Files Modified

| File | What changed |
|---|---|
| `flx/training/phase3_memory.py` | `train_phase3()`: Added `import random, warnings`. Froze trunk, cortices, decoder, thermal, router, bridges. Removed trunk/cortex/decoder from optimizer. Added per-epoch conversation shuffling. Wrapped LambdaLR in `warnings.catch_warnings()`. Added `requires_grad = True` restore before returning. |

## Key Insights

**Same lesson as Phase 2**: Don't fine-tune frozen-phase weights. Phase 2 froze trunk+cortices. Phase 3 accidentally unfroze them. Even at `lr * 0.1`, cumulative drift over thousands of steps degrades the carefully trained base. **Rule: once a component's phase is complete, freeze it in all subsequent phases.**

**Shuffle your data**: Conversations were in dataset order. Any clustering (by topic, difficulty, length) in the source dataset creates false trends in loss. Shuffle at the start of each epoch.

**Memory controller needs τ > 0.5 to retrieve**: The `MemoryController.retrieval_tau_min = 0.5` means retrieval is gated by thermal arousal. Phase 2 trained τ to ~0.45–0.54, so roughly half of conversations skip retrieval entirely. This limits how much the memory controller can learn. If Phase 3 doesn't converge, consider lowering `retrieval_tau_min` to 0.3.

## If This Still Doesn't Work

1. **Loss flat (not declining, not regressing)**: Memory controller not getting enough gradient. Lower `retrieval_tau_min` in `MemoryController.__init__()` from 0.5 to 0.3 so retrieval happens on every conversation.

2. **Loss still regressing**: Check that merger's low LR isn't still causing drift. Try freezing merger too (`lr * 0.1` → frozen).

3. **Episodes not helping later turns**: The compressor might be producing uninformative episodes. Check that compressor parameters have non-zero gradients. The `trunk_output.detach()` in compression means the compressor only learns from its own forward pass, not from downstream loss — this is by design but limits learning signal.

4. **Memory loops never trigger**: Loops need τ > 0.7 which rarely happens. This is fine — loops are Phase 3's advanced feature. The basic store+retrieve at τ > 0.5 is the primary learning objective.