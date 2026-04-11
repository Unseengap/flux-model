# Phase 4 Debugging — Online Delta Generation (Meta-Learning)

## Goal

Phase 4 trains a meta-delta generator that observes prediction errors and produces new low-rank delta matrices (A, B) targeting the correct cortex + stratum. Self-improvement with full rollback: generated deltas that hurt performance are rejected (popped from stack). All model weights frozen — only `MetaDeltaGenerator` trains.

## What "working" looks like

- `acceptance_rate`: Should stabilize **0.35–0.55**. Below 0.2 = meta-gen producing junk. Above 0.8 = acceptance threshold too permissive.
- `improvement`: Accepted deltas should improve loss by **0.001–0.01** (relative to loss_before). Near-zero (±0.0001) means deltas are noise.
- `target_cortex`: Should spread across **all 5 cortices** (language, math, code, science, reasoning). If >80% targets one cortex, the cortex selector has collapsed.
- `target_stratum`: Should hit all three levels (intermediate/expert/frontier). Frontier-heavy is fine (hardest = most room to improve).
- `cortex_entropy`: Should stay **>1.0** (max is ln(5) ≈ 1.61). Below 0.5 = cortex collapse.

## Problem History

### Attempt 1: RuntimeError — Delta stack at capacity (CRASHED)

**File**: `flx/training/phase4_meta.py` — `phase4_training_step()`

**Error**:
```
RuntimeError: Delta stack at capacity (8). Consolidate or increase capacity.
```

**Root cause**: `DeltaStack.push()` raises `RuntimeError` when `len(deltas) >= capacity`. Accepted deltas accumulate in the stack forever — only rejected deltas are popped. After 8 successful deltas across any single cortex/stratum combination, the stack overflows.

**Fix**: Evict oldest delta when stack is at capacity before pushing:
```python
target_stratum = model.cortices[cortex_name].strata[stratum_name]
if len(target_stratum.delta_stack.deltas) >= target_stratum.delta_stack.capacity:
    target_stratum.delta_stack.deltas = nn.ModuleList(
        list(target_stratum.delta_stack.deltas)[1:]
    )
target_stratum.delta_stack.push(candidate)
```

FIFO eviction preserves the most recent (presumably best) deltas.

### Attempt 2: RuntimeError — tensor does not require grad (CRASHED)

**Error**:
```
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

**Root cause**: `model.attach_meta_generator(meta_gen)` stores meta_gen as `self.meta_generator`, making it a submodule of model. The blanket freeze `for param in model.parameters(): param.requires_grad = False` froze meta_gen too.

**Fix**: Re-enable meta_gen gradients after the blanket freeze:
```python
for param in model.parameters():
    param.requires_grad = False
for param in meta_gen.parameters():
    param.requires_grad = True
```

### Attempt 3: Cortex collapse + noise-level rewards (RUNNING — degraded)

**Symptoms** at ~1000 steps:
- **Cortex collapse**: ~95% of deltas target `math/*`. `reasoning` appeared twice, `code` once, `language` and `science` zero.
- **Noise-level improvements**: Accepted deltas improve by ±0.0001 on loss ≈ 3.5. The REINFORCE reward signal is indistinguishable from zero.
- **Declining acceptance rate**: Dropped 0.43 → 0.34 over 1000 steps.

**Root cause 1 — Weak reward**: The raw improvement (e.g. 0.0001) is multiplied directly into the REINFORCE loss:
```python
reward = improvement.detach()  # ±0.0001
meta_loss = -reward * delta_contribution  # ≈ 0
```
With reward ≈ 0, the meta-gen receives no learning signal. It can't distinguish helpful deltas from harmful ones.

**Root cause 2 — No exploration pressure**: The cortex selector uses `argmax(cortex_logits)` with no entropy regularization. Once one cortex accumulates slightly more gradient signal (math, because the challenge data skews that way), the softmax sharpens and locks in.

**Fix** (two changes in `phase4_training_step()`):

1. **Scale reward relative to loss magnitude** — a 0.0001 improvement on loss=3.5 is a 0.003% gain; normalizing makes it a usable signal:
```python
scaled_reward = raw_reward / (loss_before.detach().clamp(min=0.1))
meta_loss = -scaled_reward * delta_contribution + reg_weight * delta_contribution
```

2. **Cortex entropy bonus** — penalize the meta-gen for collapsing to one cortex:
```python
cortex_probs = F.softmax(metadata_grad["cortex_logits"][0], dim=-1)
cortex_entropy = -(cortex_probs * (cortex_probs + 1e-10).log()).sum()
meta_loss = meta_loss - 0.1 * cortex_entropy
```
Max entropy for 5 cortices is ln(5) ≈ 1.61. The bonus encourages uniform exploration.

### Other fixes applied

- **Scheduler warning**: `LambdaLR` calls `step()` during `__init__`, emitting "scheduler.step() before optimizer.step()" warning. Wrapped both `LambdaLR` creations in `warnings.catch_warnings()`. Gated `scheduler.step()` behind `optimizer_stepped` flag so it never fires before the first `optimizer.step()`.

- **TransformerEncoder warnings**: `enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True` — harmless PyTorch diagnostic when using pre-norm transformers. Not actionable.

## Tests

157/157 passing after all fixes.
