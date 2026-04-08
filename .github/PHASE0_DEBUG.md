# Phase 0 Debugging — Cortex Specialization

## Goal

Phase 0 trains the thalamic router and cortices so that **different input domains route to different cortices**. Math → math cortex, code → code cortex, etc. This is the foundation — if cortices don't specialize here, every later phase builds on a broken base.

## What "working" looks like

- `pred_loss`: Starts ~10.5 (random logits over 32K vocab), settles to **3.5–5.5** over epoch 0. Should NOT drop below ~2.0 with 1M samples (that signals memorization).
- `div_loss`: Starts ~0.5 (spikiness of uniform 0.5 scores). May spike briefly to ~1.0 in first 100 steps as router tests collapse, then should recover and settle to **0.08–0.25**. Sustained 0.0000 = dead gradient (Attempts 1–3). Sustained 1.0 = stale weights from a previous failed run (see Gotcha below).
- `bal_loss`: Starts ~2.5 (scores near 0.5 across all cortices). Should stay **above 0.5** through training. Collapse to 0.0 = scores going to zero (bypass collapse).
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

### Attempt 3: Batch-level utilization entropy (FAILED)

**File**: `flx/router.py` — `diversity_loss()`

Measures whether the batch AS A WHOLE uses all cortices:
```python
assignment = torch.softmax(domain_scores * 5.0, dim=-1)  # sharpen
utilization = assignment.mean(dim=0)                       # per-cortex usage across batch
entropy = -(utilization * (utilization + 1e-8).log()).sum()
return 1.0 - entropy / max_entropy                         # 0 = uniform, 1 = collapsed
```

**File**: `flx/training/phase0_cortex.py` — `phase0_training_step()`

Removed activation threshold — ALL cortices forward on every step during Phase 0.

**Why it failed**: The math was correct in isolation — unit tests passed (124/124). But it had a **zero-gradient plateau at initialization**. Router sigmoid outputs all start near 0.5. `softmax([0.5, 0.5, 0.5, 0.5, 0.5] * 5.0)` = perfectly uniform → max entropy → `div_loss = 0.0` → zero gradient. With no diversity signal, `pred_loss` discovered that cortex outputs at init were noisy/unhelpful and pushed ALL routing scores toward zero. The model learned to bypass cortices entirely via `CortexMerger`'s residual gate (`trunk_output` passthrough).

**Training evidence** (1000 steps, div never left zero):
```
step=0   | pred=4.4676 div=0.0000 bal=0.0003
step=100 | pred=4.8659 div=0.0000 bal=0.0000
step=500 | pred=5.1150 div=0.0000 bal=0.0000
step=1000| pred=4.7034 div=0.0000 bal=0.0000
```

`bal_loss` collapsing from 0.0003 → 0.0000 confirmed scores were going to zero (bypass collapse), not to one cortex (routing collapse). Two different failure modes.

**Verified values** (correct but useless — loss was already 0 at init):
| Scenario | div_loss |
|---|---|
| All samples → cortex 0 (collapsed) | 0.90 |
| Even spread across 5 cortices | 0.00 |
| All scores uniform 0.5 | **0.00** ← zero gradient here killed it |
| One cortex biased to 0.95 | 0.88 |
| Two-cortex split (50/50) | 0.50 |

### Attempt 4: Spikiness + spread (VALIDATED — WORKING)

**File**: `flx/router.py` — `diversity_loss()`

Two complementary terms that fix both failure modes:
```python
# Spikiness: push each sample's top cortex score toward 1.0
max_scores = domain_scores.max(dim=-1).values  # [batch]
spikiness = (1.0 - max_scores).mean()

# Spread: batch utilization entropy (Attempt 3 logic)
assignment = torch.softmax(domain_scores * 5.0, dim=-1)
utilization = assignment.mean(dim=0)
entropy = -(utilization * (utilization + 1e-8).log()).sum()
max_entropy = torch.tensor(float(K), device=domain_scores.device).log()
spread = 1.0 - entropy / (max_entropy + 1e-8)

return spikiness + spread
```

**Why this fixes the zero-gradient problem**: At init (all scores ~0.5), `spikiness = 1 - 0.5 = 0.5`. The `max()` gradient flows to the argmax element per sample — different samples have different argmax cortices due to random init weights, so the gradient naturally breaks symmetry across cortices. This is the bootstrap signal that Attempt 3 lacked.

**Why this fixes bypass collapse**: Spikiness penalizes low scores directly. If all scores go to 0.01, `spikiness = 0.99` — strong gradient pushing scores back up. The model can't escape cortices by zeroing all scores.

**Why this doesn't cause Attempt 2's all-to-same collapse**: Spikiness alone would allow that (every sample spiky toward the SAME cortex = zero spikiness loss). But the spread term catches it — if all samples prefer cortex 0, utilization is [1, 0, 0, 0, 0] → entropy = 0 → spread = 1.0.

**Verified values**:
| Scenario | spikiness | spread | div_loss (sum) |
|---|---|---|---|
| All samples → cortex 0 (collapsed) | 0.00 | 0.90 | 0.90 |
| Even spread across 5 cortices | 0.00 | 0.00 | 0.00 |
| All scores uniform 0.5 | **0.50** | 0.00 | **0.50** |
| All scores near zero (bypass) | **0.99** | 0.00 | **0.99** |
| One cortex biased to 0.95 | 0.05 | 0.88 | 0.93 |

Tests pass (126/126) including two new tests: `test_zero_scores_high_loss` and `test_gradient_at_uniform`.

## Current Status

**VALIDATED IN TRAINING.** All losses behaving as expected. Tests pass (126/126).

### Training evidence (epoch 0, in progress)

```
step=0   | pred=10.5661 div=0.4592 bal=2.4823 total=12.2664  ← fresh init, all 3 losses alive
step=100 | pred=7.1187  div=0.9977 bal=0.0061 total=8.1195   ← router tests collapse, gets punished
step=200 | pred=6.6751  div=0.9308 bal=0.1247 total=7.6683   ← recovering
step=300 | pred=6.1857  div=0.1975 bal=1.0612 total=6.9136   ← diversity restored, balance strong
step=400 | pred=5.8525  div=0.0811 bal=1.0241 total=6.4453   ← pred declining, div healthy
step=500 | pred=6.2051  div=0.1888 bal=1.2320 total=7.0093
step=600 | pred=5.9413  div=0.1832 bal=1.5166 total=6.8821
step=700 | pred=5.9424  div=0.1429 bal=1.3772 total=6.7732
step=800 | pred=5.8177  div=0.2050 bal=1.3788 total=6.7112   ← stable regime
```

**Key dynamics**: At step 100, the router attempted collapse (`div=0.998`, `bal=0.006`). The spikiness+spread penalty pushed back, and by step 300 diversity was restored. This push-pull between `pred_loss` (wants collapse) and `div_loss` (wants spread) IS the specialization process working.

### Gotcha: Must re-initialize model between attempts

Attempt 4 initially showed `div=1.0, bal=0.0` at step 0 because the model still carried collapsed weights from Attempt 3. The sigmoid logits were saturated at −10 (producing scores ≈ 0.00005), so the sigmoid gradient was near-zero and the spikiness term couldn't recover. **Always re-run the model creation cell** (cell 5 in `colab_runner.ipynb`) to get fresh `FLXNano()` + `ThalamicRouter()` before starting a new attempt. Fresh init markers: `pred ≈ 10.4`, `div ≈ 0.5`, `bal > 2.0`.

### Expected trajectory (rest of training)

- `pred_loss`: Should continue declining to **3.5–5.5** by end of epoch 0
- `div_loss`: Should hover **0.08–0.25** — low enough that cortices are decisive, high enough that diversity pressure is active
- `bal_loss`: Should stay **0.5–2.0** — cortices are all receiving traffic
- `val_loss`: Should track within ~0.5 of train pred_loss

## If This Still Doesn't Work

### Diagnosis checklist

1. **div_loss starts at ~0.5 but pred_loss goes UP**: Spikiness is forcing confident routing before cortices are useful. Reduce `lambda_div` from 1.0 to 0.5.

2. **div_loss drops but spread component stays near 0**: All samples are confidently routing to the SAME cortex. The spikiness term succeeded (scores are high) but spread failed (no diversity). Increase `lambda_bal_start` from 0.5 to 1.0.

3. **div stays near 1**: Gradient isn't reaching the router. Check that `model.thalamic_router.parameters()` are in the optimizer param groups and `requires_grad=True`.

4. **pred_loss drops below 1.0 by end of epoch 0**: Memorization. Check that the data cache name is `phase01_pretrain_v2.pkl` (1M samples, not the old 100K).

5. **bal_loss collapses to 0 again**: Scores going to zero again — the spikiness term gradient may be too weak relative to `pred_loss`. Increase `lambda_div` to 2.0.

### How to revert to Attempt 3 (batch-utilization entropy only)

Replace the `diversity_loss` function in `flx/router.py`:
```python
def diversity_loss(domain_scores: Tensor) -> Tensor:
    K = domain_scores.shape[1]
    assignment = torch.softmax(domain_scores * 5.0, dim=-1)
    utilization = assignment.mean(dim=0)
    entropy = -(utilization * (utilization + 1e-8).log()).sum()
    max_entropy = torch.tensor(float(K), device=domain_scores.device).log()
    return 1.0 - entropy / (max_entropy + 1e-8)
```

### Alternative approaches to try if spikiness + spread fails

1. **Cosine similarity penalty**: Penalize cortex weight vectors (or cortex outputs) that point in the same direction. Directly forces cortices to learn different representations rather than relying on the router.

2. **Domain labels**: Assign soft domain labels to data (language/math/code/science/reasoning based on source dataset) and add an auxiliary classification loss on the router outputs. This gives the router a direct supervised signal.

3. **Gumbel-Softmax hard routing**: Replace sigmoid with Gumbel-Softmax to produce discrete one-hot assignments during training. Forces the router to make hard choices. Combine with straight-through estimator for gradients.

4. **Temperature schedule on sigmoid**: Start Phase 0 with a high temperature (sigmoid outputs near 0.5, forces exploration), then anneal down (sharper routing). Similar to how RL does exploration → exploitation.

5. **Separate router pre-training**: Train the router alone on a domain classification task (labeled data from each source dataset mapped to cortex names) before Phase 0 begins. Then Phase 0 fine-tunes jointly.

6. **Disable the residual gate during Phase 0**: The `CortexMerger` residual gate lets the model bypass cortices entirely via `trunk_output`. During Phase 0, force the model to rely on cortex outputs by removing or weakening the residual path.

## Files Modified

| File | What changed | Git diff reference |
|---|---|---|
| `flx/router.py` | `diversity_loss()` rewritten 4 times (cov → per-sample entropy → batch utilization → spikiness+spread) | Lines 125–165 |
| `flx/training/phase0_cortex.py` | Removed 0.2 activation threshold, route ALL cortices | Lines 67–77 |
| `flx/training/phase0_cortex.py` | Added `val_dataloader` param, early stopping on val loss | Function signature + epoch-end block |
| `flx/training/utils.py` | Added `evaluate_val_loss()`, `make_train_val_split()` | New functions |
| `tests/test_routing.py` | Updated diversity_loss tests: added `test_zero_scores_high_loss`, `test_gradient_at_uniform`, updated `test_uniform_scores_penalized` | `TestDiversityLoss` class |

## Other Phase 0 changes (not related to div_loss)

- **weight_decay=0.01** added to AdamW (prevents memorization)
- **Data scaled 10x**: 100K → 1M samples (512M tokens vs 51M)
- **Train/val split**: 90/10 split, early stopping on val_loss with patience=3
- **lambda_div default**: 0.1 → 1.0 (stronger diversity pressure)

## Key Insights

**Tension**: `pred_loss` wants ONE best cortex for everything (collapse), while `div_loss` wants ALL cortices used equally (spread). The balance between these two is what produces specialization — each cortex becomes "best" at its domain.

**Two kinds of collapse**: Attempts 1–3 focused on "routing collapse" (all samples → one cortex). But Attempt 3's training revealed a second failure mode: "bypass collapse" (all scores → zero, model routes everything through the merger's residual gate and ignores cortices entirely). A working `div_loss` must penalize BOTH — the spread term handles routing collapse, the spikiness term handles bypass collapse.

**Init matters**: Three of four attempts failed due to zero gradient at initialization. The sigmoid → softmax pipeline produces perfectly uniform outputs at init, which are either a fixed point (Attempts 1, 3) or an unstable attractor toward the wrong minimum (Attempt 2). Any diversity loss for sigmoid-routed MoE must produce non-zero gradient when all outputs are ~0.5.

**Stale weights kill new loss functions**: Even a correct loss function fails on an already-collapsed model. Sigmoid saturation at −10 produces gradients ≈ 0.00005 — no loss term can overcome that. Always re-init when changing the diversity loss.

**Early collapse recovery is normal**: The router will test collapse in the first ~100 steps (div spike toward 1.0). If the loss function is working, the penalty pushes it back. This push-pull IS the specialization process.

`lambda_div=1.0` means diversity has equal weight to prediction. This is aggressive. If training is unstable (loss oscillating wildly), try `lambda_div=0.5`.
