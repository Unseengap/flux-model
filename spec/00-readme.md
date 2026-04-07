# FLX Model Architecture — Spec Index

> **FLX** — A cortical, delta-native, thermally-routed LLM with persistent memory.  
> Not a format that holds a model. A model that defines the format.

---

## Spec Files

| # | File | Contents |
|---|------|----------|
| 00 | `00-readme.md` | This index |
| 01 | `01-thesis.md` | Core thesis: why model-first beats spec-first. The strategic pivot. |
| 02 | `02-architecture.md` | Full compute graph. Component list. Bridge mechanics. Delta composition. |
| 03 | `03-cortical-system.md` | Domain cortices, hierarchical strata, thalamic router, cross-cortical bridges, growth mechanics, knowledge magnetism. |
| 04 | `04-training-curriculum.md` | All 5 training phases (Phase 0–4) with pseudocode. Curriculum summary table. |
| 05 | `05-thermal-system.md` | The τ signal. Computation, what it gates, behavior by regime, thermal history. |
| 06 | `06-memory-subsystem.md` | Three-tier memory: working memory, episodic buffer, thermal history. Compressor, controller, lifecycle. Why it's not RAG. |
| 07 | `07-self-improvement.md` | Online delta accumulation. Meta-delta generator. Delta lifecycle. Safety properties. |
| 08 | `08-capabilities.md` | 12 native capabilities that fall out of the architecture. |
| 09 | `09-flx-nano.md` | Proof-of-concept spec: 145M params, 5 cortices, component sizing, 5 validation experiments, compute estimates. |
| 10 | `10-reality-checks.md` | Three critical friction points: parameter starvation, routing collapse, token-vs-sequence routing. Fixes for each. |
| 11 | `11-gpu-efficiency.md` | Triton kernels, Colab setup, hardware selection, anti-notebook pattern, PyTorch profiler, autograd integration. |
| 12 | `12-implementation-guide.md` | Repo structure, file layout, .flx serialization format, development workflow. |

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Training phases | 5 (Phase 0–4) |
| Domain cortices (Nano) | 5 |
| Strata per cortex | 4 (basic → intermediate → expert → frontier) |
| Delta rank | 32 |
| Model dimension | 512 |
| Total base params (Nano) | ~145M |
| Delta slots (Nano) | 60 across 5 cortices |

## Two Architectural Bets

1. **Does cortex specialization produce clean domain separation via the thalamic router?**
2. **Does delta-receptive pretraining within cortices produce a better compositional substrate than standard pretraining?**

FLX-Nano tests both at minimal cost. If they validate, scale to FLX-7B.
