# Phase 2 Debugging — Thermal Routing + Bridges

## Goal

Phase 2 trains the thermal estimator (τ), cross-cortical bridges, and strata gates. The model learns **when to think harder** (high τ → more strata, bridges) vs. **coast** (low τ → intermediate stratum only). Cortex bases and shared trunk are frozen from Phases 0-1.

## What "working" looks like

- `pred_loss`: Starts ~3.5 (carried from Phase 1). Should slowly improve to **3.0–3.5** as τ learns to activate expert/frontier strata on hard inputs.
- `compute_cost`: Should be **0.3–0.6** — the model uses real compute, not zero.
- `τ (tau)`: Should vary per batch: easy text → ~0.3–0.4, hard text → ~0.6–0.8. Batch mean should hover **0.35–0.55**. NOT collapse to a single value.
- `strata`: Should be **3–8** on easy, **8–15** on hard (5 cortices × 3 strata = 15 max).
- `bridges`: Should activate (>0) once τ regularly exceeds 0.3.
- `val_loss`: Should track within ~0.3 of train pred_loss.

## Problem History

### Attempt 1: `compute_cost = tau * num_strata_active` (FAILED — τ collapse)

**File**: `flx/training/phase2_thermal.py` — `phase2_training_step()`

Original implementation:
```python
compute_cost = tau_tensor.mean() * (num_strata_active + 0.5 * num_bridges_active)
total_loss = pred_loss + lambda_compute * compute_cost
```

**Why it failed**: Same death spiral as Phase 0's bypass collapse, but via τ instead of routing scores.

1. `compute_cost` gradient pushes τ **down** (lower τ = lower cost)
2. τ drops from 0.56 → 0.44 → 0.25 → **0.21** in ~300 steps
3. Once τ < 0.25 (intermediate stratum threshold), `num_strata_active = 0`
4. `compute_cost = tau * 0 = 0` → **zero gradient from compute term**
5. But pred_loss ALSO loses τ gradient: strata use hard gating (`if tau < tau_min: return zeros`), so cortex outputs don't depend on τ when below all thresholds
6. τ trapped at ~0.21 with **zero gradient from both terms**

**Training evidence** (τ collapses by step 300, strata/bridges stay at 0 forever):
```
step=0    | pred=3.5999 compute=4.4844 τ=0.561 strata=8 bridges=0
step=100  | pred=4.0787 compute=1.3164 τ=0.439 strata=3 bridges=0
step=200  | pred=3.4600 compute=1.0166 τ=0.254 strata=4 bridges=0
step=300  | pred=3.5260 compute=0.0000 τ=0.214 strata=0 bridges=0  ← dead zone
step=500  | pred=3.6196 compute=0.0000 τ=0.206 strata=0 bridges=0
step=1000 | pred=3.4387 compute=0.0000 τ=0.211 strata=0 bridges=0
step=2600 | pred=4.2324 compute=0.0000 τ=0.212 strata=0 bridges=0
```

**Two independent zero-gradient traps**:
| Term | Gradient on τ when τ < 0.25 |
|------|------|
| `compute_cost = tau * num_strata` | `num_strata = 0` → gradient = 0 |
| `pred_loss` via cortex outputs | Hard gating → cortex output = zeros → no τ dependency |

Also had scheduler warning: `lr_scheduler.step()` called before `optimizer.step()` on init (PyTorch LambdaLR calls step() internally during `__init__`).

### Attempt 2: Soft τ floor + simplified compute cost (FAILED — τ_raw → 0)

**File**: `flx/training/phase2_thermal.py` — `phase2_training_step()`

Two changes that break both zero-gradient traps:

```python
# 1. Soft floor: τ_floored ≥ 0.3 always, with smooth gradient
tau_floored = tau_floor + F.softplus(tau_tensor - tau_floor, beta=5.0)
tau = tau_floored.mean().item()  # used for stratum gating

# 2. Compute cost = tau_floored directly (no strata multiplier)
compute_cost = tau_floored.mean()
```

**What it fixed**: Strata no longer go to 0 — intermediate stratum always fires (strata=3-5). Soft floor works correctly.

**Why it still failed**: `compute_cost = tau_floored.mean()` is **unidirectional** — gradient always pushes τ down. And `pred_loss` has **zero gradient on τ** because `.item()` breaks the computation graph:

```python
tau = tau_floored.mean().item()  # ← .item() detaches from graph
# τ enters the forward pass as a plain float, not a tensor
# pred_loss has NO gradient path back to the thermal estimator
```

So the thermal estimator receives exactly one gradient signal: "make τ smaller". It obediently drives τ_raw → 0. The soft floor catches it at τ_floored ≈ 0.34, but the thermal estimator has learned **nothing** — it just minimizes τ.

**Training evidence** (τ_raw driven to 0, τ stuck at floor):
```
step=0    | pred=3.7501 compute=0.4698 τ=0.470 τ_raw=0.358
step=500  | pred=3.4160 compute=0.3474 τ=0.347 τ_raw=0.036
step=1000 | pred=3.3687 compute=0.3415 τ=0.341 τ_raw=0.007
step=2000 | pred=3.2588 compute=0.3405 τ=0.340 τ_raw=0.001
step=3200 | pred=3.3407 compute=0.3403 τ=0.340 τ_raw=0.000  ← thermal estimator dead
```

**Key: bridges=0 throughout** — bridges need the model to explore higher τ, which never happens because the only gradient signal says "go lower".

### Attempt 3: Difficulty-responsive τ target (CURRENT)

**File**: `flx/training/phase2_thermal.py` — `phase2_training_step()`

Core insight: since `pred_loss` can't provide gradient on τ (graph broken by `.item()`), the compute_cost term is the ONLY gradient source. It must be **bidirectional** — push τ up for hard batches, down for easy ones.

```python
# Use pred_loss relative to running EMA as difficulty proxy
if pred_loss_ema > 0:
    difficulty = torch.sigmoid(5.0 * (pred_loss.detach() - pred_loss_ema))
else:
    difficulty = torch.tensor(0.5, device=pred_loss.device)

tau_target = 0.3 + 0.4 * difficulty  # range [0.3, 0.7]
compute_cost = (tau_floored.mean() - tau_target) ** 2
```

**How it works**:

1. **difficulty signal**: `sigmoid(5 * (pred_loss - EMA))`. When a batch has higher-than-average loss (hard), difficulty → 1. When lower-than-average (easy), difficulty → 0. The sigmoid sensitivity of 5 means ±0.5 around baseline gives difficulty ≈ [0.08, 0.92].

2. **tau_target**: Maps difficulty to [0.3, 0.7]. Easy → 0.3 (only intermediate strata), hard → 0.7 (all strata + frontier).

3. **Squared loss**: `(tau - target)^2` provides **bidirectional gradient**:
   - `tau < target` (hard batch, need more compute): gradient pushes τ UP
   - `tau > target` (easy batch, wasting compute): gradient pushes τ DOWN

4. **EMA baseline** adapts as training progresses — the "average difficulty" shifts naturally.

**Gradient analysis at key points**:
| Batch type | pred_loss | difficulty | tau_target | If τ=0.35 | gradient direction |
|---|---|---|---|---|---|
| Easy | 2.8 (< EMA 3.5) | 0.03 | 0.31 | τ > target | push DOWN ↓ |
| Average | 3.5 (= EMA) | 0.50 | 0.50 | τ < target | push UP ↑ |
| Hard | 4.2 (> EMA) | 0.97 | 0.69 | τ < target | push UP ↑ |

The thermal estimator now gets a meaningful learning signal: "output high τ when the input will be hard to predict". This is exactly what Phase 2 is supposed to teach.

**Expected trajectory**:
- Steps 0–500: τ_raw rises from ~0 toward 0.3–0.5 as compute_cost pushes up toward targets
- Steps 500–2000: τ starts varying per-batch (easy ~0.3, hard ~0.6), strata count varies
- Steps 2000+: Expert strata (τ>0.5) fire on hard batches, bridges may activate
- τ_tgt should fluctuate between 0.3–0.7 showing the difficulty signal is active

**Log format updated**: Now shows `τ_tgt` alongside `τ` and `τ_raw`.

## Files Modified

| File | What changed |
|---|---|
| `flx/training/phase2_thermal.py` | `phase2_training_step()`: Added `tau_floor` param + softplus floor (Attempt 2), then `pred_loss_ema` param + difficulty-responsive `tau_target` + squared compute_cost (Attempt 3). Returns `tau_raw` and `tau_target` for debugging. |
| `flx/training/phase2_thermal.py` | `train_phase2()`: Passes `pred_loss_ema` to training step. Suppressed LambdaLR scheduler init warning. Log line shows `τ_raw` and `τ_tgt`. |

## Key Insights

**Three gradient traps in one module**: Phase 2 exposed three separate zero/unidirectional gradient problems:
1. `tau * num_strata` → zero gradient when strata=0 (Attempt 1)
2. `tau_floored.mean()` → unidirectional, always pushes τ down (Attempt 2)
3. `.item()` breaks the computation graph → pred_loss has zero gradient on τ (discovered in Attempt 2)

**When the graph is broken, make the surrogate loss bidirectional**: If you can't get gradient from the primary loss (pred_loss) through a control variable (τ), the auxiliary loss (compute_cost) is the ONLY gradient source. A unidirectional auxiliary loss will always drive the control variable to one extreme. Use a **target-based** loss that pushes both up and down.

**Difficulty-responsive targets are a form of reward shaping**: The `pred_loss vs EMA` signal is essentially telling the thermal estimator "you should have allocated more compute to this batch" (when loss > EMA) or "you wasted compute" (when loss < EMA). This is similar to REINFORCE with a baseline, but using L2 to a target instead of policy gradients.

**Softplus > clamp for floors**: `clamp(x, min=v)` has zero gradient below `v`. `v + softplus(x - v)` has non-zero gradient everywhere. Use softplus when you need a floor but still want the model to learn from below-floor signals.

## If This Still Doesn't Work

### Diagnosis checklist

1. **τ_raw stays near 0, τ stuck at floor (~0.34)**: The difficulty signal may be too weak. Increase `lambda_compute` from 0.01 to 0.05 or 0.1. The squared loss `(tau-target)^2` is small, so larger λ is needed to drive meaningful updates.

2. **τ tracks τ_tgt but τ_tgt doesn't vary much**: All batches have similar pred_loss (no easy/hard distinction). The training data may lack difficulty diversity. Check that the dataset includes both simple and complex text.

3. **τ goes to 0.7 for everything (no efficiency)**: Either lambda_compute is too high driving τ to always match the max target, or the EMA baseline is stale. The EMA decay (0.99) may be too slow — try 0.95 for faster adaptation.

4. **Bridges still = 0**: Check that `model.bridges` is not None. Bridge keys use `source_target` format — verify `build_bridges()` was called and bridges attached. Bridges need τ ≥ 0.3 AND both source and target cortices active.

5. **pred_loss doesn't improve despite τ varying**: Expert/frontier strata may have poor weights from Phase 0 (if Phase 0 used a fixed low τ). Check what τ was used during Phase 0 training.

6. **Gradient norm is zero for thermal estimator**: Check `model.thermal_estimator.parameters()` are in the trainable params list and `requires_grad=True`.

### Alternative approaches (if difficulty-responsive target fails)

1. **Straight-Through Estimator (STE)**: Pass τ as a tensor (not `.item()`) through the forward pass. Use STE for hard strata gating: forward uses hard gate, backward pretends it was sigmoid. This makes pred_loss provide gradient on τ directly. Requires model changes.

2. **REINFORCE / policy gradient**: Treat τ as a policy action. Use `reward = -pred_loss.detach()` with a baseline. `tau_loss = -reward * log_pi(tau)`. Works without differentiable gating but has high variance.

3. **Gumbel-Softmax strata gating**: Replace hard τ thresholds with Gumbel-Softmax to make gating fully differentiable. Most principled but largest code change.

4. **Curriculum on λ_compute**: Start with λ_compute=0 for first 2000 steps (let pred_loss gradient explore, if using STE), then anneal up to add efficiency pressure.
