# Phase 0 Debugging — Cortex Specialization

## Goal

Phase 0 trains the thalamic router and cortices so that **different input domains route to different cortices**. Math → math cortex, code → code cortex, etc. This is the foundation — if cortices don't specialize here, every later phase builds on a broken base.

## What "working" looks like

- `pred_loss`: Starts ~10.5 (random logits over 32K vocab), settles to **3.5–5.5** over epoch 0. Should NOT drop below ~2.0 with 1M samples (that signals memorization).
- `div_loss`: Should stay **above 0.1** throughout training. If it collapses to ~0.000x, the router is sending everything to one cortex. Values between 0.2–0.6 during training are healthy — it means the loss is pushing back against collapse.
- `bal_loss`: Should also stay above 0 — measures load balance across cortices.
- `val_loss`: Should track within ~0.5 of train loss. Gap > 1.0 = overfitting.

## Problem History

### Attempt 1: Covariance-based diversity_loss (FAILED)

**File**: `flx/router.py` — `diversity_loss()`

Original implementation used covariance between cortex activations:
```python
# BROKEN — produced zero gradients
mean = domain_scores.mean(dim=0)
centered = domain_scores - mean
cov = (centered.T @ centered) / batch_size
div = cov.triu(diagonal=1).pow(2).mean()
```

**Why it failed**: Router outputs are sigmoid (~0.5 initially). Mean-centered uniform values ≈ 0. Covariance of near-zero vectors = 0. No gradient ever flowed. `div` was 0.0000 for all 10 epochs.

### Attempt 2: Per-sample entropy diversity_loss (FAILED)

Replaced with entropy of each sample's routing distribution:
```python
# WRONG OBJECTIVE — rewarded collapse
probs = domain_scores / (domain_scores.sum(dim=-1, keepdim=True) + 1e-8)
entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)
return (entropy / max_entropy).mean()
```

**Why it failed**: This rewarded "spiky" per-sample routing (low entropy = low loss), but didn't care WHETHER ALL SAMPLES WERE SPIKY TOWARD THE SAME CORTEX. The router learned to slam cortex 0 to ~1.0 and the rest to ~0.0 for EVERY sample. `div` dropped from 0.998 → 0.0002 in 100 steps.

Additionally, the Phase 0 training step had a 0.2 activation threshold:
```python
if (score > 0.2).any():
    domain_scores[name] = score
```
Once a cortex's scores dropped below 0.2 (because the router stopped using it), that cortex received zero gradients and could never recover — a death spiral.

### Attempt 3: Batch-level utilization entropy (CURRENT)

**File**: `flx/router.py` — `diversity_loss()`

Current approach measures whether the batch AS A WHOLE uses all cortices:
```python
assignment = torch.softmax(domain_scores * 5.0, dim=-1)  # sharpen
utilization = assignment.mean(dim=0)                       # per-cortex usage across batch
entropy = -(utilization * (utilization + 1e-8).log()).sum()
return 1.0 - entropy / max_entropy                         # 0 = uniform, 1 = collapsed
```

**File**: `flx/training/phase0_cortex.py` — `phase0_training_step()`

Removed activation threshold — ALL cortices forward on every step during Phase 0.

**Verified values**:
| Scenario | div_loss |
|---|---|
| All samples → cortex 0 (collapsed) | 0.90 |
| Even spread across 5 cortices | 0.00 |
| All scores uniform 0.5 | 0.00 |
| One cortex biased to 0.95 | 0.88 |
| Two-cortex split (50/50) | 0.50 |

## Current Status

**NOT YET VALIDATED IN TRAINING.** The math checks out and tests pass (124/124), but the actual training run hasn't happened with this version yet.

## If This Still Doesn't Work

### Diagnosis checklist

1. **div stays near 0**: The softmax temperature (5.0) in `diversity_loss` may need tuning. If sigmoid outputs are all very close (e.g., all ~0.5001), softmax * 5.0 won't differentiate them. Try increasing to 10.0 or 20.0.

2. **div stays near 1**: Gradient isn't reaching the router. Check that `model.thalamic_router.parameters()` are in the optimizer param groups and `requires_grad=True`.

3. **pred_loss drops below 1.0 by end of epoch 0**: Memorization. Check that the data cache name is `phase01_pretrain_v2.pkl` (1M samples, not the old 100K).

4. **All cortex scores identical across samples**: The router's initial MLP weights may be too small to produce differentiated outputs. Consider initializing with larger weights or adding a learnable temperature.

### How to revert to Attempt 2 (per-sample entropy)

Replace the `diversity_loss` function in `flx/router.py`:
```python
def diversity_loss(domain_scores: Tensor) -> Tensor:
    K = domain_scores.shape[1]
    probs = domain_scores / (domain_scores.sum(dim=-1, keepdim=True) + 1e-8)
    entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)
    max_entropy = torch.tensor(float(K), device=domain_scores.device).log()
    return (entropy / (max_entropy + 1e-8)).mean()
```

And restore the threshold in `phase0_training_step()`:
```python
# Step 4 — restore threshold
domain_scores = {}
for i, name in enumerate(model.cortex_names):
    score = domain_scores_gated[:, i]
    if (score > 0.2).any():
        domain_scores[name] = score

# Step 5 — restore conditional forward
cortex_outputs = {}
for name, cortex in model.cortices.items():
    if name in domain_scores:
        cortex_outputs[name] = cortex(trunk_output, tau)
```

### Alternative approaches to try if batch-utilization entropy fails

1. **Cosine similarity penalty**: Penalize cortex weight vectors (or cortex outputs) that point in the same direction. Directly forces cortices to learn different representations rather than relying on the router.

2. **Domain labels**: Assign soft domain labels to data (language/math/code/science/reasoning based on source dataset) and add an auxiliary classification loss on the router outputs. This gives the router a direct supervised signal.

3. **Gumbel-Softmax hard routing**: Replace sigmoid with Gumbel-Softmax to produce discrete one-hot assignments during training. Forces the router to make hard choices. Combine with straight-through estimator for gradients.

4. **Temperature schedule on sigmoid**: Start Phase 0 with a high temperature (sigmoid outputs near 0.5, forces exploration), then anneal down (sharper routing). Similar to how RL does exploration → exploitation.

5. **Separate router pre-training**: Train the router alone on a domain classification task (labeled data from each source dataset mapped to cortex names) before Phase 0 begins. Then Phase 0 fine-tunes jointly.

## Files Modified

| File | What changed | Git diff reference |
|---|---|---|
| `flx/router.py` | `diversity_loss()` rewritten 3 times (cov → per-sample entropy → batch utilization) | Lines 125–148 |
| `flx/training/phase0_cortex.py` | Removed 0.2 activation threshold, route ALL cortices | Lines 67–77 |
| `flx/training/phase0_cortex.py` | Added `val_dataloader` param, early stopping on val loss | Function signature + epoch-end block |
| `flx/training/utils.py` | Added `evaluate_val_loss()`, `make_train_val_split()` | New functions |
| `tests/test_routing.py` | Updated diversity_loss test expectations for new semantics | `TestDiversityLoss` class |

## Other Phase 0 changes (not related to div_loss)

- **weight_decay=0.01** added to AdamW (prevents memorization)
- **Data scaled 10x**: 100K → 1M samples (512M tokens vs 51M)
- **Train/val split**: 90/10 split, early stopping on val_loss with patience=3
- **lambda_div default**: 0.1 → 1.0 (stronger diversity pressure)

## Key Insight

The fundamental tension in Phase 0 is: `pred_loss` wants ONE best cortex for everything (collapse), while `div_loss` wants ALL cortices used equally (spread). The balance between these two is what produces specialization — each cortex becomes "best" at its domain. If `div_loss` is too weak or broken, pred_loss wins and you get collapse. If `div_loss` is too strong, you get uniform mediocrity (no cortex is good at anything).

`lambda_div=1.0` means diversity has equal weight to prediction. This is aggressive. If training is unstable (loss oscillating wildly), try `lambda_div=0.5`.
