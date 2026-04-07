# 12 — Implementation Guide

## Repository Structure

```
flx/
├── flx/
│   ├── __init__.py
│   ├── model.py              # FLXNano, DomainCortex, Stratum, CortexMerger
│   ├── router.py             # ThalamicRouter (chunk-level routing)
│   ├── delta.py              # FLXDelta, delta composition, delta stack
│   ├── thermal.py            # ThermalEstimator
│   ├── memory.py             # EpisodicCompressor, MemoryController
│   ├── bridges.py            # CrossCorticalBridge
│   ├── meta_gen.py           # MetaDeltaGenerator
│   ├── kernels.py            # Triton kernels (delta_compose, etc.)
│   ├── autograd_bridge.py    # torch.autograd.Function wrappers for Triton
│   ├── serialization.py      # .flx save/load (manifest, cortex maps, state hub)
│   └── training/
│       ├── __init__.py
│       ├── phase0_cortex.py  # Cortex specialization + diversity + balance loss
│       ├── phase1_delta.py   # Delta-receptive pretraining within cortices
│       ├── phase2_thermal.py # Thermal routing + bridge training
│       ├── phase3_memory.py  # Memory system on conversation chains
│       └── phase4_meta.py    # Meta-delta generator training
├── tests/
│   ├── test_routing.py       # Thalamic router + chunk-level routing
│   ├── test_thermal.py       # τ computation + gating behavior
│   ├── test_delta.py         # Delta composition + stack operations
│   ├── test_memory.py        # Episodic compression + retrieval
│   ├── test_cortex.py        # Cortex specialization + strata
│   └── test_serialization.py # .flx round-trip save/load
├── notebooks/
│   └── colab_runner.ipynb    # Thin execution wrapper (see 11-gpu-efficiency.md)
├── pyproject.toml
└── README.md
```

---

## .flx File Format — Serialization Structure

```
mymodel.flx/
├── manifest.yaml
│   # version, creation_date, base_model_hash
│   # cortex_registry: [language, math, code, science, reasoning]
│   # shared_trunk: {layers: 6, d_model: 512, params: ~100M}
│   # delta_count: 60, rank: 32
│
├── shared_trunk/
│   └── weights.bin               # Shared basic stratum (thick trunk)
│
├── thalamic_router/
│   └── weights.bin               # Domain classifier parameters
│
├── cortices/
│   ├── language/
│   │   ├── meta.yaml             # domain, stratum count, growth history
│   │   ├── intermediate/
│   │   │   ├── weights.bin       # Stratum base weights
│   │   │   └── deltas/
│   │   │       ├── d001.bin      # Delta A/B matrices
│   │   │       ├── d001.yaml     # Provenance, confidence, threshold
│   │   │       └── ...
│   │   ├── expert/
│   │   │   └── ... same structure ...
│   │   └── frontier/
│   │       └── ... same structure ...
│   ├── math/
│   │   └── ... same structure ...
│   ├── code/
│   ├── science/
│   └── reasoning/
│
├── bridges/
│   ├── lang_math.yaml            # Bandwidth, compatibility, proj weights
│   ├── lang_code.yaml
│   ├── math_code.yaml
│   ├── math_reasoning.yaml
│   ├── code_reasoning.yaml
│   ├── code_science.yaml
│   ├── science_reasoning.yaml
│   ├── lang_reasoning.yaml
│   ├── lang_science.yaml
│   └── math_science.yaml
│
├── state_hub/
│   ├── working_memory.bin        # Serialized KV cache
│   ├── episode_buffer.bin        # Compressed episodic vectors
│   ├── thermal.json              # τ history trajectory
│   └── cortex_activation_history.json
│
└── meta_generator/
    └── weights.bin               # Meta-delta generator parameters
```

### Key Serialization Notes

- **Shared trunk** is serialized once, not per-cortex. After applying the "thick trunk, thin branches" reality check (see [10-reality-checks.md](10-reality-checks.md)), the basic stratum is the shared trunk.
- **Delta files** are small (~64KB each at rank=32, d_model=512). The entire delta stack for Nano is <4MB.
- **State hub** enables exact resume. Load the `.flx`, set `working_memory` and `episode_buffer`, and inference continues from the exact prior state.
- **Manifest** includes hashes for integrity checks. Two models can only exchange cortices if their shared trunk hashes match.

---

## Development Workflow

```
1. Local dev (laptop)
   - Write model code in flx/ package
   - Run unit tests: pytest tests/
   - Git commit + push to GitHub

2. Colab (GPU execution)
   - Clone repo: !git clone ... && pip install -e flx/
   - Mount Google Drive for .flx state persistence
   - Run training phases sequentially
   - Save checkpoints to Drive after each phase

3. Multi-GPU training (when scaling beyond Nano)
   - Pull repo on GPU VM (Lambda Labs, RunPod, etc.)
   - Run distributed training with torchrun
   - Export .flx checkpoint to shared storage

4. Evaluation
   - Load .flx checkpoint
   - Run validation experiments (see 09-flx-nano.md)
   - Profile with PyTorch profiler (see 11-gpu-efficiency.md)
   - Log results, iterate
```

---

## Build Order — What to Implement First

| Priority | Component | Why First | Dependency |
|----------|-----------|-----------|------------|
| 1 | `delta.py` — FLXDelta + composition | Foundation primitive, everything uses deltas | None |
| 2 | `model.py` — Stratum + DomainCortex | Core compute unit, needed for all training | delta.py |
| 3 | `router.py` — ThalamicRouter | Needed for Phase 0 | model.py |
| 4 | `training/phase0_cortex.py` | First training phase, validates cortex separation | router.py, model.py |
| 5 | `serialization.py` — .flx save/load | Need checkpointing before long training runs | model.py, delta.py |
| 6 | `training/phase1_delta.py` | Core bet — delta-receptive pretraining | phase0, serialization |
| 7 | `thermal.py` + `bridges.py` | Needed for Phase 2 | model.py |
| 8 | `training/phase2_thermal.py` | Adaptive compute training | thermal.py, bridges.py |
| 9 | `memory.py` | Needed for Phase 3 | thermal.py |
| 10 | `training/phase3_memory.py` | Memory training on conversation chains | memory.py |
| 11 | `meta_gen.py` + `training/phase4_meta.py` | Self-improvement, last phase | All above |
| 12 | `kernels.py` + `autograd_bridge.py` | Optimization — only after profiling shows need | Pure PyTorch working first |
