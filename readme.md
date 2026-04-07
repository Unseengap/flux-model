# FLX

**A cortical, delta-native, thermally-routed LLM with persistent memory.**

Not a format that holds a model. A model that defines the format.

## Architecture

```
input → SharedTrunk → ThalamicRouter → DomainCortices → CrossCorticalBridges
      → CortexMerger → MemoryController → Decoder → logits
```

- **SharedTrunk**: Canonizer + embedder + 6 shared transformer layers (basic stratum)
- **ThalamicRouter**: Chunk-level multi-hot domain routing with load balancing
- **DomainCortices**: 5 specialized brain regions (Language, Math, Code, Science, Reasoning), each with intermediate/expert/frontier strata and per-stratum delta stacks
- **CrossCorticalBridges**: τ-gated communication channels between cortex pairs
- **CortexMerger**: Weighted combination of cortex outputs + residual gate
- **MemoryController**: Episodic retrieval + refinement loop gating
- **ThermalEstimator**: τ ∈ (0,1) — learned arousal signal that gates depth, bridges, memory, and loops
- **MetaDeltaGenerator**: Produces new deltas from prediction errors (self-improvement)

## Key Numbers (FLX-Nano)

| Metric | Value |
|--------|-------|
| Total base params | ~145M |
| Domain cortices | 5 |
| Strata per cortex | 3 (intermediate, expert, frontier) + shared basic trunk |
| Delta rank | 32 |
| Model dimension | 512 |
| Delta slots | 60 across 5 cortices |

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from flx.model import FLXNano
from flx.router import ThalamicRouter
from flx.thermal import ThermalEstimator
from flx.bridges import build_bridges
from flx.memory import MemoryController, EpisodicCompressor, EpisodicBuffer
from flx.serialization import save_flx, load_flx

# Create model
model = FLXNano()

# Attach components (phased — add as training progresses)
model.attach_router(ThalamicRouter(d_model=512))
model.attach_thermal(ThermalEstimator(d_model=512))
model.attach_bridges(build_bridges(model.cortex_names, d_model=512))
model.attach_memory(MemoryController(d_model=512))

# Forward pass — τ is computed automatically
import torch
input_ids = torch.randint(0, 32000, (1, 128))
logits = model(input_ids)

# Save to .flx format
save_flx(model, "my_model.flx")

# Load from .flx format (exact resume)
loaded_model, episodic_buffer, activation_history = load_flx("my_model.flx")
```

## Training Curriculum

| Phase | What Trains | Objective |
|-------|-------------|-----------|
| **0. Cortex specialization** | Thalamic router, cortex differentiation | Next-token + diversity + load balance |
| **1. Delta-receptive pretraining** | Cortex bases + delta pools | Next-token with random delta composition |
| **2. Thermal routing** | τ estimator, bridges, strata gates | Minimize loss + minimize compute |
| **3. Memory system** | Episodic compressor, memory controller | Cross-turn prediction on conversation chains |
| **4. Meta-delta generation** | Meta-generator | Produce deltas from error signals |

```python
from flx.training.phase0_cortex import train_phase0
from flx.training.phase1_delta import train_phase1
from flx.training.phase2_thermal import train_phase2
from flx.training.phase3_memory import train_phase3
from flx.training.phase4_meta import train_phase4
```

## Project Structure

```
flx/
├── __init__.py
├── model.py              # FLXNano, SharedTrunk, DomainCortex, Stratum, CortexMerger, Decoder
├── router.py             # ThalamicRouter, diversity_loss, load_balance_loss
├── delta.py              # FLXDelta, DeltaStack, compose_weights
├── thermal.py            # ThermalEstimator
├── memory.py             # EpisodicCompressor, EpisodicBuffer, MemoryController
├── bridges.py            # CrossCorticalBridge, build_bridges
├── meta_gen.py           # MetaDeltaGenerator
├── kernels.py            # Triton kernels (optional, with PyTorch fallback)
├── autograd_bridge.py    # torch.autograd.Function wrappers for Triton
├── serialization.py      # .flx save/load
└── training/
    ├── phase0_cortex.py  # Cortex specialization + diversity + balance loss
    ├── phase1_delta.py   # Delta-receptive pretraining within cortices
    ├── phase2_thermal.py # Thermal routing + bridge training
    ├── phase3_memory.py  # Memory system on conversation chains
    └── phase4_meta.py    # Meta-delta generator training
```

## Tests

```bash
pytest tests/ -v
```

## .flx File Format

```
mymodel.flx/
├── manifest.yaml
├── shared_trunk/weights.bin
├── thalamic_router/weights.bin
├── cortices/{domain}/{stratum}/weights.bin + deltas/
├── bridges/{src}_{tgt}.yaml + weights
├── state_hub/ (working_memory, episodes, thermal history)
├── thermal_estimator/weights.bin
├── memory_controller/weights.bin
└── meta_generator/weights.bin
```

## Design Decisions

- **Thick trunk, thin branches** (spec 10): Shared basic stratum (~100M) + specialized strata per cortex (~10M each). Avoids parameter starvation.
- **Load-balancer loss** (spec 10): Prevents Phase 0 routing collapse. Anneals from strong to permissive.
- **Chunk-level routing** (spec 10): Routes at sentence/chunk level, max-aggregated. Handles mixed-domain sequences.
- **Delta-receptive training** (spec 04): Base weights trained to compose cleanly with variable delta stacks.
- **Triton optional** (spec 11): Pure PyTorch is the reference. Triton kernels for delta composition only after profiling shows need.

See `spec/` for the full architecture specification.
