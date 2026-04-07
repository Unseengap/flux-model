# 10 — Reality Checks

## Three Friction Points That Will Kill You If Ignored

---

## 1. Parameter Starvation in Nano

### The Problem

FLX-Nano at ~145M params with 5 cortices × 4 strata = 20 strata. Each stratum gets ~5M parameters (2 transformer encoder layers at d_model=512). That's paper-thin. A 2-layer transformer at d_model=512 can barely learn to copy sequences, let alone represent domain knowledge at calibrated difficulty levels. The architecture is right but the numbers don't work — you're asking each stratum to be a specialist with the parameter budget of a classifier.

### The Fix: Thick Trunk, Thin Branches

Don't distribute all parameters evenly across cortices. Instead:

```
Total: ~150M

Shared trunk (Universal Basic Stratum):  ~100M
  - Embedder: ~15M
  - Shared basic processing (4-6 transformer layers): ~70M
  - Canonizer + Decoder: ~15M

Branching cortices:                       ~50M
  - 5 cortices × ~10M each
  - Each cortex: intermediate/expert/frontier strata ONLY
  - Basic stratum is the shared trunk — all cortices share it
```

**Why this works:** The basic stratum across all cortices is doing the same thing anyway — foundational language modeling. Share it explicitly. Each cortex only needs to specialize at the intermediate, expert, and frontier levels. This gives each specialized stratum ~3× more parameters (~15M per cortex for 3 strata ≈ 5M/stratum → becomes ~100M shared + ~10M per cortex ≈ 3.3M per specialized stratum, but backing the shared trunk means each stratum's effective capacity is the shared trunk + its own parameters).

**The key insight:** This is actually how the brain works. Early visual cortex is shared. Specialization happens in higher regions. The "thick trunk, thin branches" pattern is neuroscience, not a hack.

### Impact on Architecture

- `DomainCortex.strata[0]` (basic) becomes a reference to the shared trunk, not a per-cortex module
- Phase 0 diversity pressure only applies to strata 1-3, not stratum 0
- Serialization: shared trunk serialized once, cortices reference it

---

## 2. Phase 0 Routing Collapse

### The Problem

Start with K identical cortices. Run diversity loss. The router will collapse — sending everything to one cortex that gets slightly ahead, starving the others. Diversity loss alone isn't enough because:

- The loss is symmetric but gradient noise breaks symmetry early
- One cortex getting more data → better loss → more routing → winner-take-all
- This is the exact failure mode documented in early MoE literature

### The Fix: Load-Balancer Loss (Switch Transformer Style)

Add an explicit routing load-balancer loss from the Switch Transformer paper (Fedus et al., 2021):

```python
def load_balance_loss(domain_scores, num_cortices):
    # domain_scores: [batch_size, num_cortices] after sigmoid
    
    # Fraction of tokens routed to each cortex
    fraction_routed = domain_scores.mean(dim=0)  # [num_cortices]
    
    # Fraction of routing probability assigned to each cortex
    fraction_prob = domain_scores.sum(dim=0) / domain_scores.sum()
    
    # Penalize imbalance: want fraction_routed ≈ 1/K for each cortex
    balance_loss = num_cortices * (fraction_routed * fraction_prob).sum()
    
    return balance_loss
```

Combined Phase 0 objective:

```python
loss = pred_loss + λ_div * diversity_loss + λ_bal * balance_loss
```

- `λ_bal` starts high (force equal routing) and anneals down (let natural specialization emerge)
- During early Phase 0, each cortex is guaranteed roughly equal data flow
- Diversity loss then pushes them apart on *what* they specialize on, not *whether* they get data
- After annealing, a cortex that genuinely handles more diverse inputs (e.g., Language) can get proportionally more routing

### Additional Safeguard: Dropout Routing

During Phase 0, randomly drop the top-scoring cortex with probability 0.1. Forces the second-choice cortex to be a viable fallback. Prevents hard collapse even if balance loss is imperfect.

---

## 3. Token-Level vs. Sequence-Level Routing

### The Problem

The thalamic router computes domain scores from `embedded_input.mean(dim=1)` — a sequence-level mean. But within a sequence, different tokens may belong to different domains. "Solve this integral using Python" has math tokens and code tokens. If routing is purely sequence-level, the router picks one dominant domain and loses the other.

Conversely, if routing is token-level, tokens get shattered across cortices — "Solve" goes to Language, "integral" goes to Math, "using Python" goes to Code. This prevents coherent processing within any single cortex.

### The Fix: Sequence/Sentence-Boundary Routing

Evaluate domain affinity at the sequence or sentence level, not token level:

```python
class ThalamicRouter(nn.Module):
    def forward(self, embedded_input):
        # 1. Segment input into semantic chunks
        #    (sentence boundaries, or fixed-size windows)
        chunks = segment(embedded_input)  # list of [chunk_len, d_model]
        
        # 2. Route each chunk independently
        chunk_scores = [
            sigmoid(self.domain_classifier(chunk.mean(dim=0)))
            for chunk in chunks
        ]
        
        # 3. Aggregate: sequence-level scores = weighted union of chunk scores
        #    Multi-hot: a sequence can route to multiple cortices
        sequence_scores = stack(chunk_scores).max(dim=0).values
        
        # "Solve this integral using Python" →
        #   chunk 1 "Solve this integral": {math: 0.9, language: 0.4}
        #   chunk 2 "using Python": {code: 0.85, language: 0.3}
        #   sequence: {math: 0.9, code: 0.85, language: 0.4}
        #   → Math and Code cortices both activate, Language at low weight
        
        return sequence_scores
```

**Why `max` aggregation:** The sequence needs every cortex that any chunk needs. If one sentence needs Math and another needs Code, both cortices should activate. Taking the max across chunks gives you the union of domain needs.

**Chunking strategy:** For FLX-Nano, use fixed-size windows (e.g., 64 tokens). For production, use sentence boundary detection. The router learns to handle both during Phase 0 training.

---

## Summary

| Friction Point | Symptom | Fix | Prior Art |
|---------------|---------|-----|-----------|
| Parameter starvation | Strata can't learn — too thin | Thick trunk + thin branches, shared basic stratum | ResNet shared stems, brain's shared early processing |
| Routing collapse | One cortex dominates Phase 0 | Load-balancer loss + annealing + dropout routing | Switch Transformer (Fedus 2021), GShard |
| Token vs. sequence routing | Incoherent domain splits | Chunk-level routing with max aggregation | Segment-level attention (various) |
