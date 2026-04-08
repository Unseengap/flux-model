# Dimension-Agnostic Cortices — Mixed-Size Expert Branches

## Concept

Allow each cortex to run at its own internal dimension, independent of the trunk's `d_model`. A pair of linear adapter projections at each cortex boundary handles the translation. The trunk, router, merger, and decoder all stay at `d_model=512` — each cortex is free to be whatever size it needs.

## Architecture

```
Trunk output (d_model=512)
  ↓
  proj_in: Linear(d_model → internal_dim)    ← adapter in
  ↓
  Stratum layers (running at internal_dim)    ← can be any size
  ↓
  proj_out: Linear(internal_dim → d_model)   ← adapter out
  ↓
Merger input (d_model=512)
```

When `internal_dim` is not set (or equals `d_model`), no adapters are created — behaves exactly like today.

## Why this matters

- **Transplant support**: Load layers from a 1024-dim CodeLlama into a 512-dim FLX without reshaping the entire model
- **Asymmetric capacity**: Give the math cortex more parameters (larger dim) than the language cortex if math needs more capacity
- **Scaling path**: Start small (all 512), then selectively upscale cortices that need more power
- **Model surgery**: Swap cortices between different FLX models even if they were trained at different scales

## Adapter cost

Tiny. Two matrices per cortex:

| d_model | internal_dim | Adapter params | % of cortex |
|---|---|---|---|
| 512 | 512 | 0 (no adapter) | 0% |
| 512 | 768 | 786K | ~8% |
| 512 | 1024 | 1.05M | ~10% |
| 512 | 2048 | 2.10M | ~20% |
| 512 | 4096 | 4.19M | ~40% |

For reference, each cortex at d_model=512 is ~10M params. Even a 4096-dim adapter adds less than half a cortex.

## Implementation sketch

Changes to `DomainCortex.__init__`:
```python
def __init__(self, domain_id, d_model=512, internal_dim=None, ...):
    super().__init__()
    self.internal_dim = internal_dim or d_model
    
    # Adapter projections (only if dimensions differ)
    if self.internal_dim != d_model:
        self.proj_in = nn.Linear(d_model, self.internal_dim)
        self.proj_out = nn.Linear(self.internal_dim, d_model)
    else:
        self.proj_in = nn.Identity()
        self.proj_out = nn.Identity()
    
    # Strata now use internal_dim instead of d_model
    self.strata = nn.ModuleDict({
        "intermediate": Stratum(self.internal_dim, ...),
        "expert": Stratum(self.internal_dim, ...),
        "frontier": Stratum(self.internal_dim, ...),
    })
```

Changes to `DomainCortex.forward`:
```python
def forward(self, x, tau):
    x = self.proj_in(x)      # [batch, seq, d_model] → [batch, seq, internal_dim]
    # ... existing stratum logic unchanged ...
    out = self.proj_out(out)  # [batch, seq, internal_dim] → [batch, seq, d_model]
    return out
```

Changes to `FLXNano.__init__` — accept per-cortex dimensions:
```python
cortex_dims = cortex_dims or {}  # e.g., {"math": 1024, "code": 768}
self.cortices = nn.ModuleDict({
    name: DomainCortex(
        domain_id=name,
        d_model=d_model,
        internal_dim=cortex_dims.get(name),
        ...
    )
    for name in self.cortex_names
})
```

## Example configurations

**Uniform (current behavior)**:
```python
model = FLXNano(d_model=512)  # all cortices at 512
```

**Asymmetric specialists**:
```python
model = FLXNano(d_model=512, cortex_dims={
    "math": 1024,       # math gets more capacity
    "code": 1024,       # code gets more capacity  
    "language": 512,    # language stays default
    "science": 768,     # science medium
    "reasoning": 1024,  # reasoning gets more
})
```

**Transplant from larger model**:
```python
model = FLXNano(d_model=512, cortex_dims={"code": 2048})
# Load CodeLlama layers into code cortex at native 2048 dim
model.cortices["code"].load_donor_weights(codelamma_layers)
```

## What this enables long-term

- **Mix-and-match cortices** between FLX models trained at different scales
- **Progressive scaling**: train at 512, identify which cortices are bottlenecked, upscale only those
- **Community ecosystem**: share individual cortex checkpoints (with their adapter weights) that others can plug into their FLX models
- **Hardware-aware routing**: on memory-constrained devices, use smaller cortices; on GPU, use larger ones — same model file, different cortex configs at load time
