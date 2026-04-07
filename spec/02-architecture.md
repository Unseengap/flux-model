# 02 — Architecture

## Core Principle

Cortical specialization, bridge-connected, thermally gated.

The FLX model is not a monolithic transformer. It's a domain-cortical graph — input flows through a canonizer and embedder into a thalamic router that classifies domain and difficulty, then dispatches to specialized domain cortices (Language, Math, Code, Science, etc.). Each cortex has four hierarchical strata (basic → intermediate → expert → frontier) with their own delta stacks. Cross-cortical bridges enable multi-domain reasoning. A cortex merger combines outputs. The memory controller and decoder follow. The thermal signal τ gates everything: which strata fire, which bridges open, whether memory loops trigger. The entire graph is differentiable and trained end-to-end.

---

## Compute Graph

```
                              ┌─────────────────────────────────────┐
                              │         Thermal Controller (τ)       │
                              │  surprise_estimator → τ → all gates  │
                              └────────┬────────────────┬────────────┘
                                       │ τ broadcasts    │
                                       ▼                ▼
input → [Canonizer] → [Embedder] → [Thalamic Router]
                                       │
                    domain scores: multi-hot activation
                   ┌────────────┼────────────┬───────────┐
                   ▼            ▼            ▼           ▼
             [Language]   [Math]     [Code]    [Science]  ...
              ├─ basic     ├─ basic    ├─ basic    ├─ basic
              ├─ interm.   ├─ interm.  ├─ interm.  ├─ interm.
              ├─ expert    ├─ expert   ├─ expert   ├─ expert
              └─ frontier  └─ frontier └─ frontier └─ frontier
                   │            │            │           │
                   └─── cross-cortical bridges ───┘           │
                              │                            │
                        ┌─────┴────────────────────────────┘
                        ▼
                  [Cortex Merger] ── weighted by domain scores
                        │
         bridge(bw=τ)   │  bridge(bw=0.8)
             ┌──────────┼──────────┐
             ▼          │          ▼
      [Memory Controller] │    [Decoder] → output
             │          │
             └──cycle───┘
        (loop back to cortex merger
         when τ > 0.7, max 3 loops)
```

---

## Component Responsibilities

### Canonizer
**Input normalization + tokenization alignment.**
Normalizes input representations into the model's canonical embedding space. Handles token-level preprocessing the base was trained to expect. Small, fast, rarely needs deltas — it's the stable entry point.

### Embedder
**Token → dense representation.**
Maps tokens into the compositional embedding space. Trained as part of the delta-receptive base — the embedding space is optimized for clean delta composition, meaning "directions" in the space remain interpretable across delta stacks. This is the shared coordinate system that makes deltas portable across cortices.

### Thalamic Router
**Domain classification + multi-hot dispatch.**
Like the brain's thalamus — a small learned network that detects which knowledge domains the input requires and routes accordingly. Outputs a multi-hot domain score vector: "solve this word problem" → math=0.9, language=0.7. Multiple cortices can activate simultaneously. The router also estimates input difficulty, which determines how deep within each cortex the signal penetrates (which strata activate). Trained in Phase 0 with diversity pressure to force clean domain separation.

### Domain Cortices
**Specialized brain regions with hierarchical depth.**
The heart of the architecture. Each cortex is a separate sub-network dedicated to a knowledge domain (Language, Math, Code, Science, Reasoning, etc.). Each cortex has four hierarchical strata: basic (always active, high confidence), intermediate, expert, and frontier (only active at high τ, low confidence, where new learning lives). Each stratum has its own delta stack. Cortices don't share weight space — a math delta and a medical delta live in completely separate parameter regions. This is fundamentally stronger separation than deltas on a shared substrate. See `03-cortical-system.md` for full details.

### Cross-Cortical Bridges
**Multi-domain reasoning pathways.**
Learned bridge connections between cortices that enable multi-domain reasoning. "Solve this word problem" requires the Math cortex to receive parsed structure from the Language cortex. "Explain this code's algorithmic complexity" requires Code → Math communication. Bridges are bandwidth-gated and τ-responsive — cross-cortical communication increases when the model detects multi-domain input. Each bridge has a learned compatibility score; mismatched cortex pairs have low-bandwidth bridges that the model learns to ignore.

### Cortex Merger
**Weighted combination of cortical outputs.**
Takes outputs from all active cortices and combines them, weighted by the thalamic router's domain scores. A pure-math input merges 90% from Math cortex, 10% from Language. A cross-domain input gets a balanced merge. The merger also applies a learned residual gate to preserve information flow from the embedder, ensuring the model doesn't lose context that no cortex explicitly captured.

### Memory Controller
**Episodic retrieval + loop gating.**
The only component that can trigger a cycle. Reads from the episodic buffer (compressed previous sessions) and working memory (current KV cache). When τ is high, the memory controller retrieves relevant episode vectors and feeds them back through the cortex merger for a second (or third) pass. The number of loops is learned — τ gates both the decision to loop and the retrieved context density.

### Decoder
**Representation → logits.**
Final projection to vocabulary logits. Relatively simple — the expressive work happens upstream in the cortices. Decoder deltas are rare and typically capture output-format preferences (e.g., code formatting style, citation patterns).

---

## Bridge Mechanics

```python
class Bridge(nn.Module):
    def __init__(self, dim_in, dim_out, tau_min, tau_max):
        self.proj = nn.Linear(dim_in, dim_out)     # learned, or identity if dims match
        self.bandwidth = nn.Parameter(tensor(0.7))  # learned scalar 0-1
        self.tau_min = tau_min
        self.tau_max = tau_max

    def forward(self, x, tau):
        # Thermal gating: bridge inactive outside its tau range
        gate = sigmoid((tau - self.tau_min) * 10) * sigmoid((self.tau_max - tau) * 10)
        return gate * self.bandwidth * self.proj(x)

# bandwidth is learned during training
# tau range determines when this connection is active
# the model learns which paths to use at which arousal levels
```

---

## Delta Composition at Each Component

```python
def compose_weights(component, base_weights, active_deltas, tau):
    W = base_weights.clone()
    for delta in active_deltas:
        if delta.thermal_threshold <= tau:
            # delta.A: [rank × d_in], delta.B: [d_out × rank]
            W += delta.confidence * delta.scale * (delta.B @ delta.A)
    return W

# The base never sees "deltas" during forward pass
# It just gets a composed weight tensor
# But crucially: the base was TRAINED for this composition
```
