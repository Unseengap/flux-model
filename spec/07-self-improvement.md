# 07 — Self-Improvement

## Online Delta Accumulation

Not "the model decides to learn." An engineered, tight feedback loop where prediction errors during deployment are accumulated, a meta-generator produces candidate deltas, and validated improvements are added to the stack. Every new piece of knowledge is modular, tagged with provenance, confidence-scored, and removable. Self-improvement with full rollback capability.

---

## The Online Learning Cycle

```
for each inference batch:
│
│  1. Forward pass, produce output
│  2. Observe error signal:
│     - prediction entropy (model is uncertain)
│     - user correction (explicit feedback)
│     - downstream task failure (tool call failed, code didn't compile)
│  3. Store (input, error_signal) in rolling buffer
│
when buffer.size > threshold:
│
│  4. Meta-generator produces candidate delta
│     candidate_A, candidate_B = meta_gen(error_buffer, current_stack)
│
│  5. Validate on held-out slice of buffer
│     loss_before = eval(model, holdout)
│     model.push(candidate)
│     loss_after = eval(model, holdout)
│
│  6. Accept or reject
│     if loss_after < loss_before:
│         keep delta, confidence = 0.1 (probationary)
│     else:
│         model.pop()  ← clean rollback
│
│  7. Confidence grows with continued success
│     Over time: probationary → active → consolidated into promoted base
```

---

## The Meta-Delta Generator

Instead of spawning a full fine-tuning job on error data, the meta-generator is a small network that takes (error summary, current delta stack state) as input and outputs the A and B matrices for a new delta in a single forward pass. Trained in Phase 4 with meta-learning: "given these errors, what delta would have helped?" This collapses learning from hours to milliseconds — bounded only by validation time.

```python
class MetaDeltaGenerator(nn.Module):
    """Generates delta A/B matrices from accumulated error signals."""

    def __init__(self, d_model, delta_rank=32):
        self.error_encoder = nn.TransformerEncoder(...)   # 2-layer, small
        self.stack_encoder = nn.Linear(...)               # encodes current delta state
        self.A_head = nn.Linear(d_model, delta_rank * d_model)
        self.B_head = nn.Linear(d_model, d_model * delta_rank)
        self.meta_head = nn.Linear(d_model, 4)            # cortex, stratum, scope, threshold

    def forward(self, error_buffer, current_stack):
        err_repr = self.error_encoder(error_buffer)
        stk_repr = self.stack_encoder(current_stack.summary())
        combined = err_repr + stk_repr

        A = self.A_head(combined).reshape(delta_rank, d_model)
        B = self.B_head(combined).reshape(d_model, delta_rank)
        scope, threshold, component = self.meta_head(combined).unbind(-1)

        return FLXDelta(A=A, B=B,
            confidence=0.1,
            scope=scope,
            thermal_threshold=sigmoid(threshold),
            target_cortex=component[0],
            target_stratum=component[1]
        )
```

---

## Delta Lifecycle

```
Error signal → Buffer accumulates → Meta-gen produces delta → Validation pass

Accept (conf=0.1) → Probationary period → Confidence grows → Consolidate into promoted base

Reject / regress → Pop from stack → Clean rollback, no damage
```

---

## Safety Properties

Every self-generated delta is:

1. **Validated before acceptance** — must improve on held-out data
2. **Probationary** — starts with low confidence, only activates when τ is high
3. **Reversible** — pop it from the stack and the model reverts exactly
4. **Auditable** — tagged with provenance (which errors produced it, when, what improved)

There is no uncontrolled weight modification. The base is immutable. The delta stack is append-only with rollback. The worst case is a bad delta that gets caught by validation or pruned during confidence review. Compare to standard fine-tuning where bad training corrupts all weights with no undo.
