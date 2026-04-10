# 13 — Phase 5: Abstract Rule Induction

## Why a Fifth Phase

Phases 0–4 produce a model that accumulates domain expertise, routes by difficulty, remembers across sessions, and learns from deployment errors. What it cannot do is induce an abstract transformation rule from a handful of input-output examples and apply that rule to a novel test case. That capability — "real intelligence" in the ARC-AGI sense — requires a different training signal: not "predict the next token" or "fix this error," but "what rule explains these demonstrations?"

Phase 5 depends on all prior phases:

| Dependency | Why Required |
|------------|-------------|
| Phase 0 — Cortex specialization | Rule induction benefits from cortices being differentiated so pattern primitives (spatial, numerical, logical) can be routed cleanly. |
| Phase 1 — Delta composition | Rule-encoding deltas must compose on substrates already trained for composition. |
| Phase 2 — Thermal routing + bridges | Novel tasks spike τ automatically, activating deep strata and cross-cortical bridges needed for multi-primitive reasoning. |
| Phase 3 — Episodic memory | Demonstration pairs are encoded as episodes; the memory controller retrieves them during hypothesis refinement. |
| Phase 4 — Meta-delta generation | The meta-generator architecture is extended with a new head; its error-encoding backbone is reused. |

---

## Goal

Train the model to solve few-shot transformation tasks: given N demonstration pairs `(input_i, output_i)` and one test input, predict the test output by inducing the underlying rule. The model does not memorize answers — it learns to learn.

---

## New Components

### HypothesisHead

A small network that sits on top of the memory controller's fused output and produces a dense hypothesis vector — an internal representation of "what transformation rule explains these demonstrations."

```python
class HypothesisHead(nn.Module):
    """Encodes a candidate transformation rule as a dense vector.

    Sits between MemoryController output and the refinement loop.
    Each loop iteration refines the hypothesis based on demonstration
    feedback before the final prediction.

    Args:
        d_model: Model dimension.
        hypothesis_dim: Dimension of hypothesis vectors (default = d_model).
    """

    def __init__(self, d_model=512, hypothesis_dim=512, nhead=4, num_layers=1):
        self.hypothesis_encoder = TransformerEncoder(d_model, nhead, num_layers)
        self.hypothesis_proj = nn.Linear(d_model, hypothesis_dim)
        self.consistency_head = nn.Linear(hypothesis_dim, 1)

    def forward(self, fused_repr, demo_embeddings=None):
        # fused_repr: [batch, seq, d_model] — from memory controller
        # demo_embeddings: [batch, N, d_model] — encoded demonstrations
        # Returns: (hypothesis: [batch, hypothesis_dim],
        #           consistency: [batch] — how well hypothesis explains demos)
```

**Key property:** The hypothesis vector is fed back into the refinement loop. Each pass through cortex merger → memory controller → hypothesis head refines the rule representation. The `consistency` score is a self-assessment: "does my current hypothesis explain all the demonstrations?"

### TaskScratchpad

A task-scoped working memory buffer that lives only for the duration of one problem. Unlike EpisodicBuffer (cross-session, long-lived), the scratchpad tracks:
- Previous hypothesis vectors (what rules were tried)
- Per-demo residuals (which examples the current hypothesis fails on)
- Dead-end markers (hypotheses that scored poorly — don't retry)

```python
class TaskScratchpad:
    """Task-scoped working memory for hypothesis tracking.

    Created at the start of each few-shot task, discarded after.
    Not an nn.Module — pure state container.
    """

    def __init__(self, d_model=512, max_hypotheses=8):
        self.hypotheses: list[Tensor]   # previous hypothesis vectors
        self.residuals: list[Tensor]    # per-demo error signals
        self.scores: list[float]        # consistency scores per hypothesis

    def add_hypothesis(self, h, score): ...
    def get_best(self) -> Tensor | None: ...
    def get_trajectory(self) -> Tensor: ...  # [num_tried, d_model]
    def clear(self): ...
```

**Not an nn.Module.** It's a plain data container — no learnable parameters. The scratchpad is created per-task and discarded after prediction.

---

## Extended Pipeline (Phase 5 Path)

When operating on a few-shot task at high τ:

```
demo_pairs → encode each via SharedTrunk
          → route through cortices
          → store demo embeddings in TaskScratchpad

test_input → SharedTrunk → Router → Cortices → Bridges → CortexMerger → merged

┌─── Hypothesis refinement loop (up to max_loops) ──────────────────┐
│  merged + demo_embeddings → MemoryController (retrieve episodes)  │
│  fused → HypothesisHead → (hypothesis, consistency)               │
│  if consistency < threshold and loops_remaining:                   │
│      store hypothesis + residuals in scratchpad                    │
│      re-merge with scratchpad trajectory → loop back               │
│  else:                                                             │
│      break                                                         │
└────────────────────────────────────────────────────────────────────┘

hypothesis → condition the Decoder → predict test_output
```

The existing refinement loop (max 3 passes at τ > 0.7) is reused. The difference: in Phase 5, each loop pass has access to the scratchpad's hypothesis trajectory, so successive iterations refine rather than repeat.

---

## Training Objective

### Data Format

Each training example is a few-shot task:
```python
task = {
    "demos": [(input_1, output_1), ..., (input_N, output_N)],  # N = 2-5
    "test_input": Tensor,
    "test_output": Tensor,
}
```

### Loss Function

Three terms:

```python
def phase5_loss(pred_logits, test_target, hypothesis, demo_embeddings, consistency):
    # 1. Prediction loss — did the model get the right answer?
    pred_loss = cross_entropy(pred_logits, test_target)

    # 2. Consistency loss — does the hypothesis explain all demos?
    #    Supervisory signal: hypothesis applied to each demo input
    #    should reconstruct each demo output.
    consistency_loss = (1.0 - consistency).mean()

    # 3. Efficiency pressure — fewer loops is better (same λ as Phase 2)
    loop_cost = λ_loop * num_loops_used

    return pred_loss + λ_cons * consistency_loss + loop_cost
```

**The gradient path:** `pred_loss` flows back through decoder → hypothesis head → memory controller → cortex merger → cortices → trunk. The hypothesis head learns what rule representations lead to correct predictions. The refinement loop learns when to stop (consistency is high enough).

### Meta-Generator Extension

The Phase 4 meta-generator gets a new training signal alongside its existing error-correction objective:

```python
# Existing Phase 4 objective (preserved):
meta_loss_error = -improvement * delta_norm  # REINFORCE on error correction

# New Phase 5 objective (added):
# Given demos that the model got wrong, what delta would encode the
# missing rule? Uses the hypothesis vector as conditioning.
meta_loss_rule = -rule_improvement * delta_norm

# Combined:
meta_loss = meta_loss_error + λ_rule * meta_loss_rule
```

This is NOT replacing REINFORCE with MAML. Phase 4's online correction remains REINFORCE (appropriate for its use case). Phase 5 adds a rule-induction signal that trains the meta-generator to produce rule-encoding deltas conditioned on hypothesis vectors.

---

## Training Procedure

```python
def phase5_training_step(model, hypothesis_head, meta_gen, task):
    # 1. Encode demonstrations
    demo_embeddings = []
    for demo_input, demo_output in task["demos"]:
        trunk_out = model.shared_trunk(demo_input)
        demo_embeddings.append(trunk_out.mean(dim=1))

    # 2. Forward test input through full pipeline at high τ
    tau = 0.8  # Force high τ for deep reasoning path
    logits = model(task["test_input"], tau=tau)  # triggers deep path

    # 3. Hypothesis refinement (uses existing loop infrastructure)
    #    HypothesisHead sits inside the memory controller loop
    #    Each iteration: fused → hypothesis → check consistency → maybe loop

    # 4. Final prediction
    pred_loss = cross_entropy(logits, task["test_output"])

    # 5. Consistency loss from hypothesis head
    consistency_loss = (1.0 - consistency_score).mean()

    # 6. Combined loss
    total_loss = pred_loss + λ_cons * consistency_loss

    return {"pred_loss": pred_loss, "consistency_loss": consistency_loss,
            "total_loss": total_loss, "consistency": consistency_score,
            "num_loops": loops_used}
```

### What Trains

| Component | Frozen/Trained | Why |
|-----------|---------------|-----|
| SharedTrunk | Frozen | Stable substrate from Phase 0-1 |
| Thalamic Router | Frozen | Routing established in Phase 0 |
| Domain Cortices | Fine-tuned (low LR) | Strata learn primitive patterns for rule induction |
| Cross-Cortical Bridges | Fine-tuned (low LR) | Bridges learn structural analogy mapping |
| CortexMerger | Frozen | Merging established in Phase 2 |
| MemoryController | Fine-tuned (low LR) | Learns to retrieve relevant demonstrations |
| **HypothesisHead** | **Trained (full LR)** | **New component — learns rule representations** |
| **TaskScratchpad** | N/A | No parameters — pure state |
| Meta-Generator | Fine-tuned (low LR) | Learns rule-encoding delta generation |
| Decoder | Fine-tuned (low LR) | Learns hypothesis-conditioned output |

### τ During Phase 5

Fixed at 0.8. Phase 5 always uses the deep path — few-shot tasks are inherently novel and require all strata, bridges, memory, and loops. The thermal estimator is not trained further; τ = 0.8 is hardcoded during Phase 5 training to ensure the full hypothesis refinement path activates.

---

## Integration with Existing Architecture

### No Pipeline Changes Required

The hypothesis head plugs in at the memory controller's output, before the decoder. The scratchpad is a plain data container passed as a parameter. No changes to the compute graph order.

### Attachment Convention

Following the phased attachment pattern (`attach_router`, `attach_thermal`, etc.):

```python
class FLXNano:
    self.hypothesis_head: nn.Module | None = None

    def attach_hypothesis_head(self, head: nn.Module) -> None:
        self.hypothesis_head = head
```

### Serialization

HypothesisHead weights serialize into:
```
cortices/
├── hypothesis_head/
│   └── weights.bin
├── task_scratchpad/   # empty at rest — created per-task at runtime
```

---

## Training Data Requirements

Phase 5 requires few-shot transformation tasks, not next-token prediction data. Sources:

1. **Programmatic generation** — Grid transformations, sequence analogies, pattern completion tasks with known rules. Arbitrarily many. This is the primary source.
2. **ARC-AGI style tasks** — Abstract reasoning corpus (public subset) for evaluation and fine-tuning.
3. **Synthetic analogies from existing cortex domains** — "What is to X as Y is to Z?" constructed from domain knowledge. Math: `2→4, 3→6, 5→?`. Code: `if→else, while→break, for→?`.

### Curriculum Within Phase 5

| Stage | N demos | Rule complexity | Loops expected |
|-------|---------|----------------|----------------|
| 5a | 4-5 | Single primitive (rotate, count, negate) | 1 |
| 5b | 3-4 | Two primitives composed (rotate + scale) | 1-2 |
| 5c | 2-3 | Multi-step with branching (if-then rules) | 2-3 |

Start with many demonstrations and simple rules, reduce demonstrations and increase complexity.

---

## Success Criteria

Phase 5 is complete when:

1. **Few-shot accuracy** — Model solves >60% of held-out single-primitive tasks given 3 demonstrations
2. **Consistency** — Hypothesis `consistency` score correlates with actual correctness (>0.7 Spearman ρ)
3. **Loop efficiency** — Average loops used is <2.5 (model doesn't always max out)
4. **No regression** — Phases 0-4 metrics remain within 2% of pre-Phase-5 baselines
5. **Generalization** — Accuracy on unseen rule types (not in training set) >30%
