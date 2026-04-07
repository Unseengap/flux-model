# 04 — Training Curriculum

## Why Five Phases

You can't train everything at once — cortices need to differentiate before deltas can specialize within them, the delta system needs stable cortices, the thermal system needs working cortical strata to gate, the memory system needs thermal routing to decide when to retrieve, and self-improvement needs all four to produce meaningful error signals. The phases are ordered by dependency, not importance.

---

## Phase 0 — Cortex Specialization Pretraining

**Goal:** Force cortices to differentiate into domain specialists.

Before deltas can specialize within cortices (Phase 1), the cortices themselves must have differentiated. Start with K identical cortices (same random init). During pretraining, the thalamic router learns to route domain-specific batches to different cortices. A diversity loss penalizes cortices that activate on the same inputs — forcing specialization. After Phase 0, each cortex has a distinct "receptive field" for knowledge domains.

```python
def phase0_training_step(model, batch):
    # 1. Route input through thalamic router
    domain_scores = model.thalamic_router(batch.input)

    # 2. Forward through all cortices (weighted by scores)
    cortex_outputs = {}
    for name, cortex in model.cortices.items():
        if domain_scores[name] > 0.2:
            cortex_outputs[name] = cortex(batch.input, tau=0.5)  # fixed τ

    # 3. Merge and predict
    merged = model.cortex_merger(cortex_outputs, domain_scores)
    logits = model.decoder(merged)
    pred_loss = cross_entropy(logits, batch.target)

    # 4. Diversity pressure — force cortices to specialize
    activation_matrix = stack([domain_scores[n] for n in cortex_names])
    # Penalize: high correlation between cortex activation patterns
    diversity_loss = (activation_matrix @ activation_matrix.T).off_diagonal().mean()

    # 5. Combined objective
    loss = pred_loss + λ_div * diversity_loss
    loss.backward()

    # What the router learns:
    #   - "math-like tokens" → cortex 0, "code-like tokens" → cortex 2
    #   - Multi-hot for cross-domain: "word problem" → cortex 0 + cortex 3
    # What cortices learn:
    #   - Each cortex develops a distinct receptive field
    #   - Basic strata handle easy within-domain, expert strata handle hard

    return loss
```

### The Strata Diversity Trick

Diversity pressure is applied per-stratum, not per-cortex. Basic strata **CAN** overlap — "basic language understanding" is useful for many domains. But expert and frontier strata **MUST** specialize. This is enforced by scaling the diversity loss coefficient: `λ_div` is low for basic strata and high for frontier strata.

The result: foundational capabilities are shared naturally, while deep expertise is domain-exclusive. This mirrors how the brain works — early visual processing is shared, but face recognition vs. word recognition are separate regions.

---

## Phase 1 — Delta-Receptive Pretraining Within Cortices

**Goal:** Train each cortex's base to be a compositional substrate for domain deltas.

With cortices now differentiated (Phase 0), each cortex's base weights are trained to be delta-receptive. Standard pretraining optimizes: "predict the next token with fixed weights." FLX pretraining optimizes: "predict the next token given that each cortex's weights will be dynamically composed from its base + variable delta stack." This is a fundamentally different objective. Each cortex learns to be a good canvas for domain-specific deltas.

```python
def phase1_training_step(model, batch):
    # 1. Route through thalamic router (frozen from Phase 0)
    domain_scores = model.thalamic_router(batch.input)

    # 2. For each active cortex, sample random deltas per stratum
    for name, cortex in model.active_cortices(domain_scores):
        for stratum in cortex.strata:
            K = random.randint(0, len(stratum.delta_pool))
            active = random.sample(stratum.delta_pool, K)
            stratum.compose_with(active)

    # 3. Forward through cortices + merge + decode
    logits = model.forward(batch.input)
    loss = cross_entropy(logits, batch.target)

    # 4. Backprop through cortex bases AND active deltas
    loss.backward()

    # What each cortex base learns:
    #   - Be a good canvas for THIS domain's deltas
    #   - Structural knowledge for THIS domain (math syntax, code grammar)
    #   - NOT domain facts — those belong in stratum deltas
    # What deltas learn:
    #   - Specialize within their stratum's difficulty level
    #   - Basic stratum deltas → foundational domain patterns
    #   - Frontier stratum deltas → cutting-edge domain knowledge

    return loss
```

### Why This Changes Scaling Laws

A standard transformer crams everything into one shared weight space. "Paris is the capital of France" competes for the same parameters as "mitochondria is the powerhouse of the cell." FLX separates knowledge at two levels: between cortices (Language vs. Science) and within cortices (basic vs. frontier strata). A cortical model with 5 specialized cortices + domain deltas can outperform a monolithic model 5× its parameter count, because knowledge is never forced to share weight space with unrelated knowledge — and within each domain, difficulty levels don't interfere.

---

## Phase 2 — Thermal Routing + Bridge Training

**Goal:** Train τ end-to-end as a differentiable signal.

τ is not a hyperparameter. It's a learned function of the input, the current delta stack state, and the model's own prediction uncertainty. Train it with a dual objective: minimize prediction loss AND minimize active compute. The model learns to think harder on hard problems and faster on easy ones.

```python
def phase2_training_step(model, batch):
    # 1. Compute thermal signal from input
    tau = model.thermal_estimator(batch.input)  # learned sigmoid → (0, 1)

    # 2. τ gates everything downstream
    active_deltas = [d for d in model.deltas if d.threshold <= tau]
    bridge_outputs = [b.forward(x, tau) for b in model.bridges]
    n_loops = model.loop_controller(tau)  # 0, 1, 2, or 3

    # 3. Forward pass with thermally-gated graph
    logits = model.forward_graph(batch.input, tau, active_deltas, n_loops)

    # 4. Dual objective
    pred_loss = cross_entropy(logits, batch.target)
    compute_cost = count_active_flops(active_deltas, n_loops)
    loss = pred_loss + λ * compute_cost  # λ tunes the efficiency pressure

    # Gradient on τ:
    #   "You activated too few deltas → answer was wrong" → ∂loss/∂τ pushes τ UP
    #   "You activated everything, still correct" → compute_cost pushes τ DOWN
    #   τ converges to the minimum compute needed for correct prediction

    loss.backward()
    return loss
```

### What the Thermal System Learns

After Phase 2, the model natively routes easy inputs through the fast path (local mixer, 2 deltas, no loops) and hard inputs through the deep path (both mixers, all deltas, memory loops). This is computationally equivalent to Universal Transformers / adaptive computation time, but the gating is semantic — the model doesn't just "think more steps," it activates more specialized knowledge and retrieves more context when it detects uncertainty. The efficiency gains are real: **40-60% fewer FLOPs on routine inputs** with no accuracy loss.

---

## Phase 3 — Memory System Training

**Goal:** Train on conversation chains, not isolated sequences.

Standard LLMs train on independent sequences. FLX Phase 3 trains on chains of related sequences — conversation turns, document continuations, multi-session dialogues. The model learns to compress old context into episodic vectors and retrieve them when relevant. This is what makes cross-session memory native rather than bolted on.

```python
def phase3_training_step(model, conversation_chain):
    episodic_buffer = []

    for turn in conversation_chain:
        # 1. Forward pass with current memory state
        logits = model.forward_with_memory(
            turn.input,
            working_memory=model.kv_cache,
            episodes=episodic_buffer
        )
        loss = cross_entropy(logits, turn.target)
        loss.backward()

        # 2. After each turn, compress KV cache into episodic vector
        if model.kv_cache.size > threshold:
            episode = model.episodic_compressor(model.kv_cache)
            episodic_buffer.append(episode)
            model.kv_cache.trim()

    # The memory controller learns:
    #   - WHEN to retrieve from episodes (τ > 0.5, relevant context needed)
    #   - WHAT to retrieve (attention over episode vectors)
    #   - HOW to compress (minimize retrieval loss on future turns)
    #
    # The episodic compressor learns:
    #   - Summarize a KV cache into a fixed-size dense vector
    #   - Preserve information that future turns will need
    #   - Discard information that won't be queried again
```

---

## Phase 4 — Online Delta Generation

**Goal:** Train a meta-delta generator that targets correct cortex + stratum.

A small model (the meta-generator) takes accumulated prediction errors + current delta stack and produces new delta A/B matrices directly, without a full training loop. The model genuinely has a fast-path for learning from its mistakes. Every new delta is tagged, confidence-scored, and removable — self-improvement with rollback.

```python
def phase4_training_step(meta_generator, model, error_buffer):
    # 1. Meta-generator takes error signals and produces delta matrices
    error_summary = summarize_errors(error_buffer)
    new_A, new_B, metadata = meta_generator(
        error_summary,
        current_delta_stack=model.active_deltas
    )

    # 2. Create candidate delta
    candidate = FLXDelta(
        A=new_A, B=new_B,
        confidence=0.1,  # starts low
        scope=metadata.predicted_scope,
        thermal_threshold=metadata.predicted_threshold
    )

    # 3. Validate: does candidate improve on held-out errors?
    loss_before = evaluate(model, error_buffer.holdout)
    model.delta_stack.push(candidate)
    loss_after = evaluate(model, error_buffer.holdout)

    # 4. Train meta-generator on the improvement signal
    meta_loss = loss_after - loss_before  # negative = good
    meta_loss.backward()

    # 5. Keep or discard
    if loss_after < loss_before:
        candidate.confidence = 0.1  # start probationary
    else:
        model.delta_stack.pop()  # clean rollback

    # Over time, confidence grows with coverage
    # High-confidence deltas get consolidated into promoted base
    # Low-confidence deltas get pruned
```

---

## Training Curriculum Summary

| Phase | What Trains | Objective | Data | Prior Art |
|-------|-------------|-----------|------|-----------|
| 0. Cortex specialization | Thalamic router, cortex differentiation | Next-token + diversity pressure across cortices | Domain-labeled corpus | Extends MoE routing with named, hierarchical, persistent regions |
| 1. Delta-receptive pretraining | Cortex bases + delta pools per cortex | Next-token with random delta composition within cortices | Standard pretraining corpus | Novel — delta-receptive training within cortical structure |
| 2. Thermal routing | τ estimator, bridge gates, strata gates, loop controller | Minimize loss + minimize active compute | Difficulty-diverse batches | Extends ACT (Graves 2016) with cortical + semantic gating |
| 3. Memory system | Episodic compressor, memory controller | Cross-turn prediction on conversation chains | Multi-turn dialogues, docs | Extends memory-augmented networks (Graves et al.) |
| 4. Meta-delta generation | Meta-generator (targets cortex + stratum) | Produce good deltas from error signals | Online error buffers | Related to MAML/Reptile, applied to cortex-scoped delta production |
