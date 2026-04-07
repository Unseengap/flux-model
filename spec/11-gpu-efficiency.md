# 11 — GPU Efficiency

## Hardware Selection

### Target: A100 or L4 — Not T4

- **A100 (40GB/80GB):** Primary training target. 8× A100 node for Phase 0-4 training. bf16 native, high memory bandwidth, TF32 tensor cores.
- **L4 (24GB):** Inference and light fine-tuning. Good price-performance on cloud. Ada Lovelace architecture, good for Triton kernels.
- **NOT T4:** The T4 is Turing-era (2018), no bf16, limited memory (16GB). FLX-Nano's 150M params fit but the cortical routing adds overhead that makes T4 impractical for training. T4 is fine for smoking-testing inference only.

### Why This Matters

The thermal routing system (τ-gated strata + bridges) and cortical graph add control flow that's cheaper on modern hardware with better branch prediction and larger L2 caches. A100's 80GB HBM2e means all 5 cortices + bridges + memory controller fit in one GPU's memory without offloading.

---

## Google Colab Workflow — The Anti-Notebook Pattern

### Problem

Jupyter notebooks are great for exploration but terrible for reproducible model development. Code cells get run out of order, state leaks between experiments, and there's no version control.

### Solution: GitHub Repo + Colab as Execution Wrapper

```
GitHub repo (version-controlled source of truth):
  flx/
    model.py          # DomainCortex, Stratum, ThalamicRouter, etc.
    kernels.py         # Triton kernels
    training.py        # Phase 0-4 training loops
    memory.py          # EpisodicCompressor, MemoryController
    thermal.py         # ThermalEstimator
    meta_gen.py        # MetaDeltaGenerator
    serialization.py   # .flx save/load
  tests/
    test_routing.py
    test_thermal.py
    test_delta_composition.py

Google Colab notebook (execution wrapper only):
  cell 1: !git clone https://github.com/yourname/flx && pip install -e flx/
  cell 2: from flx.model import FLXNano; model = FLXNano()
  cell 3: from flx.training import phase0_train; phase0_train(model, data)
  cell 4: model.save_flx("/content/drive/MyDrive/flx_checkpoints/nano_v1.flx")
```

### Google Drive as Persistent State Hub

Colab VMs are ephemeral — runtime disconnects lose everything. Mount Google Drive and use it as the persistent `.flx` state hub:

```python
from google.colab import drive
drive.mount('/content/drive')

FLX_HUB = "/content/drive/MyDrive/flx_state/"

# Save after every phase
model.save_flx(f"{FLX_HUB}/nano_phase0.flx")

# Resume from any checkpoint
model = FLXNano.load_flx(f"{FLX_HUB}/nano_phase0.flx")
```

This means: code lives in GitHub (versioned, reviewable), compute happens in Colab (free/cheap GPUs), state persists in Drive (survives runtime restarts). The `.flx` format is the bridge between them.

---

## Triton Kernels

### Why Triton

The FLX compute graph has non-standard operations that PyTorch's built-in kernels don't optimize:

1. **Delta composition:** `W_effective = W_base + Σ(confidence * scale * B @ A)` across variable-length delta stacks per stratum
2. **Thermal gating:** Conditional activation of strata, bridges, memory based on τ thresholds
3. **Multi-cortex routing:** Parallel forward pass through active cortices weighted by domain scores

These are control-flow-heavy, data-dependent operations. Standard PyTorch compiles them into many small kernel launches. Triton lets you fuse them into single GPU kernels.

### Installation

```bash
pip install -U triton torch-tb-profiler
```

### Triton JIT Compilation Behavior

Triton kernels are JIT-compiled on first call:

```python
import triton
import triton.language as tl

@triton.jit
def delta_compose_kernel(
    W_base_ptr, A_ptr, B_ptr, conf_ptr, scale_ptr,
    W_out_ptr, n_deltas, d_model, rank,
    BLOCK_SIZE: tl.constexpr
):
    """Fused delta composition: W_out = W_base + Σ(conf * scale * B @ A)"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Load base weights
    w = tl.load(W_base_ptr + offsets)
    
    # Accumulate deltas
    for i in range(n_deltas):
        conf = tl.load(conf_ptr + i)
        scale = tl.load(scale_ptr + i)
        # Compute B @ A contribution for this block
        delta_contrib = compute_ba_block(A_ptr, B_ptr, i, offsets, d_model, rank)
        w += conf * scale * delta_contrib
    
    tl.store(W_out_ptr + offsets, w)

# First call: JIT compilation (1-3 seconds)
# Subsequent calls: near-zero overhead
```

**Expect 1-3 seconds of compilation time on first invocation.** This is normal. The compiled kernel is cached for the session. For training loops, this one-time cost is negligible.

---

## PyTorch Profiler for FLOP Measurement

The thermal system's value proposition is "40-60% fewer FLOPs on easy inputs." You need to measure this precisely.

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_flops=True
) as prof:
    with record_function("flx_forward"):
        output = model(input_batch)

# Print FLOP counts per operation
print(prof.key_averages().table(sort_by="flops", row_limit=20))

# Export for TensorBoard visualization
prof.export_chrome_trace("flx_trace.json")
```

### What to Measure

| Metric | How | Why |
|--------|-----|-----|
| FLOPs per token (easy input) | Profile on trivial inputs (τ < 0.3) | Baseline fast-path cost |
| FLOPs per token (hard input) | Profile on novel/contradictory inputs (τ > 0.7) | Full-depth cost |
| FLOP ratio (easy/hard) | Divide | Should be 0.4-0.6× (the efficiency claim) |
| Strata activation count | Log which strata fire per input | Verify thermal gating works |
| Bridge bandwidth per input | Log bridge gate values | Verify cross-cortical routing |
| Memory retrieval frequency | Count τ > 0.5 episodes | Verify memory controller selectivity |

---

## The Autograd Bridge: Triton ↔ PyTorch Backward Pass

### The Problem

Triton kernels are forward-only by default. PyTorch's autograd needs to know how to compute gradients for backpropagation. If you write a Triton kernel for delta composition, PyTorch doesn't know how to differentiate through it.

### The Fix: `torch.autograd.Function` Wrapper

```python
class DeltaCompose(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W_base, A_list, B_list, conf_list, scale_list):
        # Call Triton kernel for fast fused forward
        W_out = delta_compose_triton(W_base, A_list, B_list, conf_list, scale_list)
        
        # Save tensors needed for backward
        ctx.save_for_backward(W_base, *A_list, *B_list, *conf_list, *scale_list)
        return W_out
    
    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        W_base = saved[0]
        # ... unpack A_list, B_list, conf_list, scale_list
        
        # Gradient for W_base: just grad_output (identity contribution)
        grad_W_base = grad_output
        
        # Gradient for each delta's A, B:
        #   ∂L/∂A_i = conf_i * scale_i * B_i.T @ grad_output
        #   ∂L/∂B_i = conf_i * scale_i * grad_output @ A_i.T
        grad_A_list = []
        grad_B_list = []
        for i in range(n_deltas):
            grad_A_list.append(conf[i] * scale[i] * B[i].T @ grad_output)
            grad_B_list.append(conf[i] * scale[i] * grad_output @ A[i].T)
        
        # Gradient for confidence: ∂L/∂conf_i = scale_i * (B_i @ A_i) · grad_output
        grad_conf_list = [
            scale[i] * (grad_output * (B[i] @ A[i])).sum()
            for i in range(n_deltas)
        ]
        
        return grad_W_base, grad_A_list, grad_B_list, grad_conf_list, None

# Usage in model forward:
W_effective = DeltaCompose.apply(W_base, A_list, B_list, conf_list, scale_list)
```

**This is the critical bridge.** Without it, Triton kernels are inference-only. With it, the full training loop (Phases 0-4) can use fused Triton kernels for forward passes while PyTorch handles gradient computation through the autograd wrapper.

### When to Write Triton vs. Stay in PyTorch

| Operation | Use Triton? | Reason |
|-----------|------------|--------|
| Delta composition (variable-length stack) | Yes | Fuses N matmuls + accumulation into one kernel |
| Thermal gating (threshold comparisons) | Maybe | Only if profiler shows it's a bottleneck |
| Standard attention | No | PyTorch/Flash Attention already optimal |
| Cross-cortical bridges | No | Simple linear + sigmoid, PyTorch is fine |
| Episodic compression | No | 2-layer transformer encoder, use PyTorch |
| Thalamic routing | No | Small linear + sigmoid, negligible cost |

**Rule:** Write Triton kernels only when the profiler shows a bottleneck from many small kernel launches or non-standard memory access patterns. Start with pure PyTorch, profile, then optimize the hot spots.
