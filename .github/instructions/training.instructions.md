---
description: "Use when writing or editing FLX training phases. Covers phased curriculum, loss conventions, and training step patterns."
applyTo: "flx/training/**"
---
# Training Phase Conventions

## Phase Order

| Phase | Module | Trains | τ |
|-------|--------|--------|---|
| 0 | `phase0_cortex.py` | Router + cortex differentiation | Fixed 0.5 |
| 1 | `phase1_delta.py` | Cortex bases + delta pools | Fixed 0.5 |
| 2 | `phase2_thermal.py` | τ estimator, bridges, strata gates | Learned |
| 3 | `phase3_memory.py` | Episodic compressor, memory controller | Learned |
| 4 | `phase4_meta.py` | Meta-delta generator | Learned |

Each phase assumes prior phases are complete. Components from later phases are not attached yet.

## Conventions

- Loss functions are **standalone functions**, not class methods — e.g. `diversity_loss()`, `load_balance_loss()`
- Training steps return `dict[str, Tensor]` with named loss components: `pred_loss`, `total_loss`, etc.
- Assert preconditions at step entry: `assert model.thalamic_router is not None`
- Import from parent package: `from ..model import FLXNano`
- Use `EarlyStopState` from `utils.py` for patience-based stopping
- Shared helpers live in `flx/training/utils.py`; phase-specific logic stays in its own module
