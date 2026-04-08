---
name: spec-to-code
description: "Translate FLX architecture spec into implementation code. Use when implementing new components, verifying spec compliance, or adding features described in spec/*.md files."
argument-hint: "Component or spec section to implement (e.g. 'memory subsystem', 'thermal gating')"
---

# Spec-to-Code Translation

## When to Use

- Implementing a new component described in `spec/`
- Verifying existing code matches the spec
- Adding features or behaviors specified in the architecture docs

## Procedure

1. **Identify the relevant spec file** using the mapping in [./references/spec-map.md](./references/spec-map.md)
2. **Read the spec section** — extract requirements, tensor shapes, formulas, and constraints
3. **Check existing implementation** — read the target module to understand current state
4. **Implement or update** following these rules:
   - Match tensor shapes exactly as specified (e.g. `[batch, seq, d_model]`)
   - Preserve the pipeline contract: each component's output feeds the next
   - Use `nn.ModuleDict`/`nn.ModuleList` for sub-module storage
   - Learnable parameters via `nn.Parameter(torch.tensor(...))`
   - Low-rank deltas: B zero-init, A kaiming-init
   - τ (tau) is always a scalar float, never batched
   - Router returns `dict[str, Tensor]` keyed by cortex name
5. **Write tests** — class-based, small dimensions, shape + value assertions
6. **Run `make test`** to verify nothing breaks

## Pipeline Contract

```
input_ids: [batch, seq]
    → SharedTrunk → [batch, seq, d_model]
    → ThalamicRouter → dict[cortex_name, Tensor[batch]]
    → DomainCortices → dict[cortex_name, Tensor[batch, seq, d_model]]
    → CrossCorticalBridges → dict[cortex_name, Tensor[batch, seq, d_model]]
    → CortexMerger → [batch, seq, d_model]
    → MemoryController → [batch, seq, d_model]
    → Decoder → [batch, seq, vocab_size]
```

## Key Spec Constants (FLX-Nano)

| Parameter | Value | Spec Source |
|-----------|-------|-------------|
| d_model | 512 | `09-flx-nano.md` |
| delta_rank | 32 | `09-flx-nano.md` |
| cortices | 5 (lang, math, code, science, reasoning) | `03-cortical-system.md` |
| strata per cortex | 3 (intermediate, expert, frontier) | `03-cortical-system.md` |
| delta slots | 60 total | `09-flx-nano.md` |
