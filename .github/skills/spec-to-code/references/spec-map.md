# Spec File → Module Mapping

| Spec File | Covers | Implementation |
|-----------|--------|----------------|
| `spec/02-architecture.md` | Full pipeline, component overview | `flx/model.py` |
| `spec/03-cortical-system.md` | Cortices, strata, routing, merging | `flx/model.py`, `flx/router.py` |
| `spec/04-training-curriculum.md` | Phased training, loss functions | `flx/training/phase0–4` |
| `spec/05-thermal-system.md` | τ estimator, thermal gating | `flx/thermal.py` |
| `spec/06-memory-subsystem.md` | Episodic buffer, compressor, controller | `flx/memory.py` |
| `spec/07-self-improvement.md` | Meta-delta generation | `flx/meta_gen.py` |
| `spec/09-flx-nano.md` | Nano config, parameter budgets | `flx/model.py` |
| `spec/11-gpu-efficiency.md` | Triton kernels, fused ops | `flx/kernels.py`, `flx/autograd_bridge.py` |
| `spec/12-implementation-guide.md` | Serialization, .flx format | `flx/serialization.py` |

## Cross-Cutting Concerns

- **Deltas** (`flx/delta.py`): Referenced across `03`, `04`, `07`, `09`
- **Bridges** (`flx/bridges.py`): Referenced in `02`, `03`, `05`
- **Training utils** (`flx/training/utils.py`): Shared across all training phases
