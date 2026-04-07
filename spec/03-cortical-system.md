# 03 — Cortical System

## Core Concept: Brain Regions, Not Shared Weight Space

Standard transformers (and the original FLX design) put all domain knowledge in a shared weight space — a math delta and a medical delta both modulate the same matrices. They're additive so they don't catastrophically interfere, but they're not truly independent. Domain cortices fix this at the structural level. Each knowledge domain gets its own sub-network with its own parameters. They don't share weight space at all — they only communicate through cross-cortical bridges.

---

## Why This Is Not Mixture of Experts

| Existing Approach | What's Different in FLX Cortices |
|-------------------|----------------------------------|
| MoE (Mixture of Experts) | Experts are interchangeable and anonymous. Cortices are named, persistent, hierarchical, and growable. You can inspect what the Math cortex knows vs. the Code cortex. |
| Progressive Neural Networks | Add entire columns per task. Cortices add deltas or strata within a domain — finer grained, more parameter-efficient. |
| Modular Networks | Static modules. Cortices grow via delta accumulation and stratum addition — they physically expand. |
| Multi-task heads | Task-specific output layers on a shared trunk. Cortices are separate processing regions, not just separate outputs. |

---

## Domain Cortex — The Building Block

```python
class DomainCortex(nn.Module):
    """One brain region — e.g., Math, Language, Code, Science."""

    def __init__(self, domain_id, d_model):
        self.domain_id = domain_id  # "math", "language", "code", etc.
        self.strata = nn.ModuleList([
            Stratum(d_model, depth="basic",        tau_min=0.0),   # always active
            Stratum(d_model, depth="intermediate", tau_min=0.25),
            Stratum(d_model, depth="expert",       tau_min=0.5),
            Stratum(d_model, depth="frontier",     tau_min=0.7),   # high τ only
        ])
        self.difficulty_gate = nn.Linear(d_model, len(self.strata))

    def forward(self, x, tau):
        # Difficulty gate determines stratum weighting
        stratum_weights = softmax(self.difficulty_gate(x.mean(dim=1)))

        out = 0
        for i, stratum in enumerate(self.strata):
            if tau >= stratum.tau_min and stratum_weights[i] > 0.1:
                out += stratum_weights[i] * stratum(x, tau)
        return out
```

---

## Hierarchical Strata — Basic to Frontier

**The key insight: the model knows what it knows, and at what depth.**

Current LLMs have flat knowledge — the model either knows something or it doesn't. Strata give the model calibrated depth per domain. The basic stratum has high confidence and always fires. The frontier stratum has low confidence, only fires at high τ, and is where new uncertain knowledge lives.

This means the model natively knows: "I can answer your basic chemistry question confidently (basic stratum, conf=0.95) but this organic synthesis question is at my frontier (frontier stratum, conf=0.3)." Honest uncertainty falls out of the architecture.

```python
class Stratum(nn.Module):
    """One difficulty layer within a cortex."""

    def __init__(self, d_model, depth, tau_min):
        self.depth = depth           # "basic", "intermediate", "expert", "frontier"
        self.tau_min = tau_min        # minimum τ to activate this stratum
        self.layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=8), num_layers=2
        )
        self.confidence = nn.Parameter(
            tensor(1.0 if depth == "basic" else 0.5)
        )
        self.delta_slots = []        # deltas attach here, at the right stratum

    def forward(self, x, tau):
        W = self.compose_stratum_weights(tau)
        return self.confidence * self.layers(x)

    def compose_stratum_weights(self, tau):
        for delta in self.delta_slots:
            if delta.thermal_threshold <= tau:
                self.apply_delta(delta)
```

### Stratum Behavior by Thermal Level

| Stratum | τ Threshold | Initial Confidence | Knowledge Type | Example (Math Cortex) |
|---------|-------------|-------------------|----------------|----------------------|
| Basic | 0.0 (always on) | 1.0 | Foundational, high-certainty | Arithmetic, basic algebra, number properties |
| Intermediate | 0.25 | 0.7 | Standard domain knowledge | Calculus, linear algebra, statistics |
| Expert | 0.5 | 0.5 | Specialized, lower certainty | Abstract algebra, topology, differential geometry |
| Frontier | 0.7 | 0.3 | Cutting-edge, uncertain, growing | Novel proof strategies, recent conjectures, experimental methods |

---

## Thalamic Router — The Brain's Switchboard

```python
class ThalamicRouter(nn.Module):
    """Routes input to relevant domain cortices. Like the brain's thalamus."""

    def __init__(self, d_model, cortex_names):
        self.domain_classifier = nn.Linear(d_model, len(cortex_names))
        self.cortex_names = cortex_names  # ["language", "math", "code", ...]

    def forward(self, embedded_input):
        # Multi-hot: which domains does this input need?
        domain_scores = sigmoid(self.domain_classifier(embedded_input.mean(dim=1)))
        # → "solve this word problem" → {math: 0.9, language: 0.7, code: 0.05}

        # Only cortices above activation threshold participate
        active = {name: score for name, score
                  in zip(self.cortex_names, domain_scores)
                  if score > 0.2}
        return active

# The router learns clean domain boundaries during Phase 0
# Diversity pressure prevents all cortices from activating on everything
# Multi-hot activation enables cross-domain reasoning
```

---

## Cross-Cortical Bridges — Multi-Domain Reasoning

"Parse this legal contract and compute the financial implications" needs Language → Legal → Math. "Debug this bioinformatics pipeline" needs Code → Science → Reasoning.

Cross-cortical bridges are learned linear projections with bandwidth gates between cortex pairs. At low τ, bridges are quiet and cortices work independently. At high τ, bridges open and cortices collaborate. The model learns which cortex pairs need strong bridges (Math↔Code, Language↔Reasoning) and which are rarely co-activated.

```python
class CrossCorticalBridge(nn.Module):
    """Learned communication channel between two cortices."""

    def __init__(self, d_model, source_cortex, target_cortex):
        self.source = source_cortex
        self.target = target_cortex
        self.proj = nn.Linear(d_model, d_model)
        self.bandwidth = nn.Parameter(tensor(0.5))
        self.compatibility = nn.Parameter(tensor(0.5))  # learned affinity

    def forward(self, source_output, tau):
        # Bridge is more active at higher τ (cross-domain = harder)
        gate = sigmoid((tau - 0.3) * 10) * self.bandwidth
        return gate * self.compatibility * self.proj(source_output)

# Example bridge activations for "solve this word problem":
#   Language → Math bridge: bw=0.8 (need to parse English into math)
#   Language → Code bridge: bw=0.1 (not relevant)
#   Math → Language bridge: bw=0.3 (need to express answer in English)
```

---

## Cortex Merger

```python
class CortexMerger(nn.Module):
    """Weighted combination of all active cortex outputs."""

    def __init__(self, d_model):
        self.residual_gate = nn.Linear(d_model * 2, d_model)

    def forward(self, cortex_outputs, domain_scores, embedder_output):
        merged = sum(
            domain_scores[name] * output
            for name, output in cortex_outputs.items()
        )

        # Residual gate preserves embedder information
        merged = self.residual_gate(cat([merged, embedder_output], dim=-1))
        return merged
```

---

## Growth — How Cortices Expand Over Time

### Growth via Deltas, Not Raw Neurons

Growing raw neurons mid-deployment is messy — initialization problems, shape changes that break compiled graphs, unbounded memory growth. Instead, cortices grow through structured delta accumulation:

- **New knowledge** = new delta targeted at the right cortex + stratum
- **Saturation** = when a stratum has many deltas with diminishing returns → consolidate (merge high-confidence deltas into stratum base weights)
- **Full cortex saturation** = all strata saturated → add a new stratum (coarser, safer than individual neurons)

### New Cortex Creation

When the thalamic router consistently routes inputs to no existing cortex with high confidence (domain_score < 0.4 for all), this signals an unrecognized domain. A new cortex can be spawned:

1. Initialize from the closest existing cortex (warm start)
2. Register with the router
3. Begin accumulating domain-specific deltas

This is a rare, heavyweight operation — not per-batch but per-epoch or triggered by a human operator reviewing router confusion signals.

---

## Knowledge Magnetism — New Learning Finds Its Cortex

```python
def route_new_learning(error_buffer, cortices, thalamic_router):
    # 1. Classify error domain via the thalamic router
    domain_affinity = thalamic_router(error_buffer.embeddings)

    # 2. Pick target cortex
    target_cortex = cortices[domain_affinity.argmax()]

    # 3. Pick target stratum based on error difficulty
    difficulty = estimate_difficulty(error_buffer)
    if difficulty < 0.3:   target_stratum = target_cortex.strata[0]  # basic
    elif difficulty < 0.5: target_stratum = target_cortex.strata[1]  # intermediate
    elif difficulty < 0.7: target_stratum = target_cortex.strata[2]  # expert
    else:                  target_stratum = target_cortex.strata[3]  # frontier

    # 4. Generate delta scoped to that cortex + stratum
    delta = meta_generator(error_buffer, scope=(target_cortex, target_stratum))
    target_stratum.delta_slots.append(delta)

    # 5. If stratum is saturated → consolidate
    if target_stratum.is_saturated():
        target_stratum.consolidate()
```

---

## .flx Serialization — Cortical Map

```
mymodel.flx/
├── manifest.yaml                     # cortex registry
├── thalamic_router/
│   └── weights.bin                   # domain classifier parameters
├── cortices/
│   ├── language/
│   │   ├── meta.yaml                 # domain, stratum count, growth history
│   │   ├── basic/
│   │   │   ├── weights.bin           # stratum base weights
│   │   │   └── deltas/              # stratum-scoped delta stack
│   │   ├── intermediate/
│   │   ├── expert/
│   │   └── frontier/
│   ├── math/
│   │   └── ... same structure ...
│   ├── code/
│   └── science/
├── bridges/
│   ├── cross_lang_math.yaml
│   ├── cross_math_code.yaml
│   └── cross_code_science.yaml
├── state_hub/
│   ├── working_memory.bin
│   ├── episode_buffer.bin
│   ├── thermal.json
│   └── cortex_activation_history.json
└── meta_generator/
    └── weights.bin
```
