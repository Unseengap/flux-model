# 05 — Thermal System

## What τ Is

τ ∈ (0, 1) is a **differentiable arousal signal**, not a config knob. It's a scalar computed by a small learned network (the thermal estimator) from the input representation + current episodic context. It broadcasts to every gate in the model: strata activation thresholds, bridge bandwidth, memory retrieval trigger, loop count. It's trained end-to-end in Phase 2 with the dual objective of prediction accuracy and compute efficiency.

---

## τ Computation

```python
class ThermalEstimator(nn.Module):
    """Small network that computes τ from input + context."""

    def __init__(self, d_model):
        self.surprise_head = nn.Linear(d_model, 1)
        self.context_head = nn.Linear(d_model, 1)
        self.history_weight = nn.Parameter(tensor(0.3))

    def forward(self, hidden_state, prev_tau_history):
        # Surprise: how unexpected is this input?
        surprise = sigmoid(self.surprise_head(hidden_state.mean(dim=1)))

        # Context: does episodic memory suggest this is novel?
        context_novelty = sigmoid(self.context_head(hidden_state[:, 0]))

        # Blend with history for smooth transitions
        raw_tau = 0.6 * surprise + 0.4 * context_novelty
        tau = (1 - self.history_weight) * raw_tau + \
              self.history_weight * prev_tau_history.mean()

        return tau  # scalar in (0, 1)
```

---

## What τ Controls — Four Gating Domains

### 1. Strata Activation

Each stratum within each cortex has a `tau_min` threshold. Low τ → only basic strata fire (foundational knowledge). High τ → expert and frontier strata activate, bringing specialized and uncertain knowledge into play. This replaces flat delta thresholds with hierarchical depth gating per domain.

### 2. Cross-Cortical Bridges

Cross-cortical bridges have `tau_min=0.3`. At low τ, cortices work independently. At high τ, bridges open and cortices collaborate — the Language cortex sends parsed structure to Math, Code sends logic patterns to Reasoning. Multi-domain reasoning scales with thermal arousal.

### 3. Memory Retrieval

The memory controller's bridge has `tau_min=0.5`. Below that, no episodic retrieval — the model handles the input from cortical outputs alone. Above 0.5, the model says "I need to remember something" and queries the episode buffer.

### 4. Refinement Loops

The memory → cortex merger cycle is gated by `τ > 0.7`. The model loops back through the cortex merger with retrieved memory context. Max 3 loops, learned. Each loop is "let me reconsider with this additional context from my memory and deeper cortical processing."

---

## Thermal Behavior by Regime

| τ Range | Behavior | Example Inputs |
|---------|----------|----------------|
| 0.0 – 0.3 | **Fast path.** Only basic strata in primary cortex. No cross-cortical bridges, no memory, no loops. Minimum compute. | "Hello", "What's 2+2?", continuing a familiar topic |
| 0.3 – 0.5 | **Standard path.** Basic + intermediate strata, cross-cortical bridges activate for multi-domain input. Moderate compute. | General questions, code completion, straightforward reasoning |
| 0.5 – 0.7 | **Deep path.** Expert strata active, full cross-cortical bridges, episodic retrieval enabled. Significant compute. | Complex coding tasks, multi-step math, unfamiliar domain |
| 0.7 – 1.0 | **Full depth.** All strata including frontier + memory loops + all cortices wide open. Maximum compute per token. | Novel contradictions, cross-domain synthesis, "this conflicts with what I remember" |

---

## Thermal History as a Time Series

τ has memory. The thermal estimator takes `prev_tau_history` as input, creating smooth transitions rather than jumps. A conversation that starts easy (τ≈0.2) and gradually gets harder sees τ ramp up smoothly. A surprising input mid-conversation causes a τ spike that decays.

This history is persisted in the state hub — resume a session and the thermal trajectory continues. The model "warms up" on familiar territory and "deepens" on novel ground, and this pattern carries across sessions.
