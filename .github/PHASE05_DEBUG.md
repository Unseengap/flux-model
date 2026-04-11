# Phase 5 Debugging — Abstract Rule Induction (Few-Shot Learning)

## Goal

Phase 5 trains a HypothesisHead to induce abstract transformation rules from a handful of input→output demonstration pairs, then apply the induced rule to a novel test input. Uses a refinement loop with a TaskScratchpad to iteratively improve the hypothesis. All prior-phase components are present; trunk, router, and merger are frozen.

## What "working" looks like

- `pred_loss`: Should decrease steadily from ~15 → below 1.0. This is the primary objective — predicting test outputs.
- `consistency`: Should rise **gradually** — 0.3 → 0.5 → 0.7+ over hundreds of steps as predictions improve. Tracks exp(-pred_loss/2).
- `cons_target`: Supervised target for consistency, = exp(-pred_loss/2). Shows what consistency *should* be given current prediction quality.
- `loops`: Should average **1–2** during training. 0 = refinement loop is dead (consistency gate too easy). 3 = model can't converge on a good hypothesis.

## Problem History

### Attempt 1: Consistency collapse + dead refinement loop

**Symptoms** at step 120 (epoch 0):
```
step=0   | pred=15.20 cons=0.566 loops=3   ← only step with loops
step=10  | pred=3.92  cons=0.954 loops=0   ← consistency saturated
step=50  | pred=1.44  cons=0.997 loops=0
step=120 | pred=1.69  cons=0.999 loops=0   ← completely dead
```

- `consistency` jumped from 0.566 → 0.954 in 10 steps, then saturated at 0.999
- `loops=0` from step 10 onward — the refinement loop, scratchpad, and hypothesis trajectory are completely bypassed
- `pred_loss` plateaued around 1.3–1.7 — the model stopped improving

**Root cause 1 — Free consistency reward**: The consistency loss `lambda_cons * (1 - cons).mean()` gives a direct, uncontested gradient pushing consistency → 1.0. The consistency head learned to always output ~1.0 regardless of actual hypothesis quality. Nothing penalized overconfidence when `pred_loss` was still high.

**Root cause 2 — No minimum loops**: The loop breaks as soon as `cons > consistency_threshold (0.85)`. With consistency saturated at 0.999 from step 10, the loop breaks on the **first iteration every time**. The entire refinement mechanism — scratchpad trajectory, hypothesis re-merging, iterative improvement — never gets gradient signal after the first few steps.

**Fix** (two changes in `phase5_training_step()`):

1. **`min_loops` parameter** — force at least 1 refinement pass before the consistency check can break the loop. This guarantees the scratchpad/trajectory path always gets gradient signal:
```python
# Before: loop could break immediately
if cons_score.mean().item() >= consistency_threshold:
    break

# After: must complete min_loops first
if loop_idx >= min_loops:
    if cons_score.mean().item() >= consistency_threshold:
        break
    if not should_loop:
        break
if loop_idx >= max_loops:
    break
```

2. **Calibration loss** — penalize high consistency when prediction is poor. Gradient flows only through the consistency head (pred_loss is detached), teaching it to be honest about hypothesis quality:
```python
cal_loss = (final_consistency * pred_loss.detach()).mean()
total_loss = pred_loss + lambda_cons * cons_loss + lambda_loop * eff_loss + lambda_cal * cal_loss
```
When `pred_loss` is high (hypothesis is bad), high `consistency` is penalized. When `pred_loss` is low (hypothesis is good), high `consistency` is appropriate and barely penalized.

3. **Lowered `lambda_cons`** from 0.3 → 0.1 in the notebook cell so prediction loss dominates the training signal rather than the consistency reward.

**Config** (notebook training cell):
```python
min_loops=1,           # force ≥1 refinement pass
lambda_cons=0.1,       # lowered: pred_loss should dominate
lambda_cal=0.2,        # penalise overconfident consistency
```

### Attempt 2: Consistency crushed to zero (calibration overcorrection)

**Symptoms** across 3+ epochs (1060 steps):
```
step=0   | pred=14.61 cons=0.473 loops=3
step=10  | pred=2.93  cons=0.005 loops=3   ← cons collapsed immediately
step=100 | pred=1.57  cons=0.001 loops=3
step=500 | pred=1.22  cons=0.001 loops=3
step=980 | pred=1.19  cons=0.000 loops=3   ← flatlined at 0.000
```

- `pred_loss` decreased nicely: 14.6 → ~1.0 (model learned predictions)
- `val_loss` improved: 1.32 → 1.13 (no overfitting)
- `consistency` collapsed to 0.000 from step 10 onward — the **opposite** of Attempt 1
- `loops=3` every step — loop always maxes out because cons never reaches 0.85

**Root cause — Calibration overpowered consistency**: The gradient forces on the consistency head:
- **Push up** (cons_loss): `lambda_cons × 1 = 0.1`
- **Push down** (cal_loss): `lambda_cal × pred_loss = 0.2 × 1.3 = 0.26`

The downward force (0.26) always exceeded the upward force (0.1), so the head learned to always output 0. The self-assessment mechanism was completely dead — but predictions still improved because `pred_loss` dominates `total_loss`.

**Fix — Rebalance the forces**:

The equilibrium condition is `lambda_cons = lambda_cal × pred_loss`. With the new values:
- At `pred_loss=3.0` (bad): up=0.15, down=0.15 → balanced at ~0.5 (appropriate)
- At `pred_loss=1.3` (mid): up=0.15, down=0.065 → net up, cons rises moderately
- At `pred_loss=0.5` (good): up=0.15, down=0.025 → cons rises strongly (appropriate)

This means consistency naturally tracks prediction quality — high when predictions are good, moderate when mediocre, suppressed when bad.

**Config** (notebook training cell, Attempt 3):
```python
min_loops=1,           # force ≥1 refinement pass (kept from Attempt 1 fix)
lambda_cons=0.15,      # raised: stronger upward pull
lambda_cal=0.05,       # lowered: gentle downward correction
```

### Other notes

- **TransformerEncoder warnings**: `enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True` — same harmless PyTorch diagnostic as Phase 4. Affects `model.py:174`, `model.py:77`, `meta_gen.py:57`, `hypothesis.py:56`. Not actionable.

- **OOM on L4 (22 GB)**: Phase 5 requires more memory than Phases 0–4 due to the refinement loop (up to 3 passes through the full pipeline per step). A100 (40/80 GB) is recommended. A memory cleanup cell was added before the training cell to free prior-phase objects, but it wasn't sufficient on the L4.

## Tests

157/157 passing after all fixes.