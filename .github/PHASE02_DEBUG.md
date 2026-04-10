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

### Attempt 2: Soft τ floor + simplified compute cost (CURRENT)

**File**: `flx/training/phase2_thermal.py` — `phase2_training_step()`

Two changes that break both zero-gradient traps:

```python
# 1. Soft floor: τ_floored ≥ 0.3 always, with smooth gradient
tau_floored = tau_floor + F.softplus(tau_tensor - tau_floor, beta=5.0)
tau = tau_floored.mean().item()  # used for stratum gating

# 2. Compute cost = tau_floored directly (no strata multiplier)
compute_cost = tau_floored.mean()
```

**Why this fixes the zero-gradient problem**:

1. **Soft floor via softplus**: `softplus(x, beta=5)` ≈ `max(x, 0)` but smooth. When `tau_tensor < 0.3`, `tau_floored ≈ 0.3` with small gradient still flowing back to the thermal estimator. When `tau_tensor > 0.3`, `tau_floored ≈ tau_tensor` (gradient ≈ 1.0). Intermediate stratum (τ_min=0.25) **always fires** because tau_floored ≥ 0.3.

2. **Simplified compute cost**: `compute_cost = tau_floored.mean()` always provides gradient. No multiplication by `num_strata_active` that goes to zero. τ IS the compute proxy — higher τ causes more strata to fire, so penalizing τ directly achieves the same efficiency pressure.

**Why softplus instead of hard clamp**: `torch.clamp(tau, min=0.3)` has zero gradient when `tau < 0.3`, which would trap the thermal estimator's parameters. Softplus provides a smooth gradient everywhere.

Also fixed scheduler warning by wrapping `LambdaLR()` creation in `warnings.catch_warnings()`.

**Expected behavior**:
| Scenario | tau_raw | tau_floored | compute_cost | strata |
|---|---|---|---|---|
| Easy text, model learns low τ | 0.15 | ~0.30 | ~0.30 | 5 (intermediate only) |
| Medium text | 0.45 | ~0.45 | ~0.45 | 10 (intermediate + expert) |
| Hard text | 0.75 | ~0.75 | ~0.75 | 15 (all strata) |
| τ collapse attempt | 0.01 | ~0.30 | ~0.30 | 5 (floor prevents collapse) |

## Files Modified

| File | What changed |
|---|---|
| `flx/training/phase2_thermal.py` | `phase2_training_step()`: Added `tau_floor` param, soft floor via `F.softplus`, simplified `compute_cost`, added `tau_raw` to output dict |
| `flx/training/phase2_thermal.py` | `train_phase2()`: Suppressed LambdaLR scheduler init warning |
| `flx/training/phase2_thermal.py` | Log line: Added `τ_raw` field for debugging |

## Key Insights

**Same pattern as Phase 0**: Hard gating creates zero-gradient dead zones. In Phase 0 it was routing scores → 0 (bypass collapse). In Phase 2 it's τ → 0 (thermal collapse). Any time a differentiable signal gates discrete components via hard thresholds, the training loss must provide gradient EVEN WHEN the signal is below all thresholds.

**compute_cost must always have gradient**: The original `tau * num_strata` looked correct but created a trap. When designing efficiency penalties for gated systems, the penalty should depend on the continuous control signal (τ), not on the discrete outcome (strata count) which can zero out.

**Softplus > clamp for floors**: `clamp(x, min=v)` has zero gradient below `v`. `v + softplus(x - v)` has non-zero gradient everywhere. Use softplus when you need a floor but still want the model to learn from below-floor signals.

## If This Still Doesn't Work

### Diagnosis checklist

1. **τ still stuck at 0.3 (floor), strata=5 only**: Compute pressure is too strong relative to pred_loss improvement from higher strata. Reduce `lambda_compute` from 0.01 to 0.001.

2. **τ goes to 1.0 for everything (no efficiency)**: Compute pressure too weak. Increase `lambda_compute` to 0.05 or 0.1.

3. **τ varies but pred_loss doesn't improve**: Bridges may not be connecting. Check bridge key parsing (underscore-separated cortex names). Verify bridges are in the trainable params.

4. **pred_loss increases significantly**: Frozen cortex weights may be incompatible with thermal gating. Try unfreezing cortex layers at lower LR.

5. **Gradient norm is zero for thermal estimator**: Check `model.thermal_estimator.parameters()` are in the trainable params list and `requires_grad=True`.

### Alternative approaches

1. **Gumbel-Softmax strata gating**: Replace hard τ thresholds with Gumbel-Softmax to make gating fully differentiable.

2. **τ entropy regularization**: Add `- H(τ distribution across batch)` to encourage variance in τ values (easy vs hard inputs should get different τ).

3. **Curriculum on λ_compute**: Start with λ_compute=0 (learn quality first), then anneal up to 0.01 (add efficiency pressure after τ knows what helps quality).
