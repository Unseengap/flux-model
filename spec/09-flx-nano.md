# 09 — FLX-Nano

## Purpose

FLX-Nano is a small, trainable model designed to answer the questions that matter: does cortex specialization produce cleaner domain separation than a shared weight space? Do hierarchical strata enable calibrated depth? Does delta-receptive pretraining within cortices outperform standard training? If yes, everything else follows. If no, we learn why before burning compute at scale.

---

## Specs

| Dimension | Value |
|-----------|-------|
| Total base parameters | ~150M |
| Domain cortices | 5 (Language, Math, Code, Science, Reasoning) |
| Strata per cortex | 4 (basic, intermediate, expert, frontier) |
| Delta rank | 32 |
| Model dimension (d_model) | 512 |
| Attention heads | 8 |

---

## Component Sizing

| Component | Layers | Parameters (approx) | Notes |
|-----------|--------|---------------------|-------|
| Canonizer | 1 | ~2M | Stable entry point |
| Embedder | 2 | ~15M | Shared coordinate system |
| Thalamic Router | 1 | ~2M | Domain classifier + difficulty estimator |
| Language Cortex (4 strata) | 8 | ~20M | 2 layers per stratum, 3 delta slots each |
| Math Cortex (4 strata) | 8 | ~20M | 2 layers per stratum, 3 delta slots each |
| Code Cortex (4 strata) | 8 | ~20M | 2 layers per stratum, 3 delta slots each |
| Science Cortex (4 strata) | 8 | ~20M | 2 layers per stratum, 3 delta slots each |
| Reasoning Cortex (4 strata) | 8 | ~20M | 2 layers per stratum, 3 delta slots each |
| Cross-cortical bridges | — | ~5M | 10 bridge pairs (all combinations) |
| Cortex Merger | 1 | ~3M | Weighted combination + residual gate |
| Memory Controller | 2 | ~10M | Episodic retrieval + loop gating |
| Decoder | 1 | ~8M | Final projection to logits |
| **Total base** | **~48** | **~145M** | **60 delta slots across 5 cortices** |
| Thermal estimator | — | ~0.5M | Surprise + context novelty + history |
| Episodic compressor | 2 | ~3M | KV cache → episode vectors |
| Meta-delta generator | 2 | ~5M | Targets cortex + stratum |

---

## Validation Experiments

### Experiment 0 — Cortex Specialization

**Hypothesis:** After Phase 0 training, each cortex develops a distinct domain receptive field with minimal overlap in expert/frontier strata.

**Protocol:** Train FLX-Nano with 5 cortices and diversity pressure. After Phase 0, measure activation patterns on domain-labeled test data. Compute specialization score: for each cortex, what fraction of its top-scoring inputs belong to one domain?

**Key metric:** Specialization score > 0.7 for expert strata. Basic strata may overlap (that's expected and healthy).

### Experiment 1 — Delta Receptivity (The Core Bet)

**Hypothesis:** A cortical model with delta-receptive cortices outperforms a standard transformer of equivalent total parameters on domain-specific tasks.

**Protocol:** Train FLX-Nano (145M base across 5 cortices + 60 delta slots) and a standard transformer (150M, same total param budget) on the same corpus. Evaluate on 5 domains (code, math, medical, legal, general). If FLX-Nano with domain deltas active beats the standard model on domain tasks while matching on general, the core bet validates.

**Key metric:** Accuracy per parameter on domain transfer.

### Experiment 2 — Thermal Efficiency

**Hypothesis:** Thermally-routed FLX-Nano uses 40-60% fewer FLOPs on easy inputs with no accuracy loss.

**Protocol:** After Phase 2 training, measure FLOPs per token on a difficulty-stratified test set (trivial, moderate, hard). Compare accuracy-vs-compute curve against standard model.

**Key metric:** FLOPs saved at iso-accuracy.

### Experiment 3 — Cross-Session Memory

**Hypothesis:** FLX-Nano can retrieve relevant information from compressed episodes of prior sessions.

**Protocol:** Run 10-turn conversation chains. At turn K, ask about information from turn 1. Compare retrieval accuracy: FLX memory controller vs. RAG baseline vs. full-context baseline.

**Key metric:** Recall@1 on cross-turn factual questions.

### Experiment 4 — Online Delta Quality

**Hypothesis:** Meta-generated deltas improve model performance on error-producing input classes.

**Protocol:** Deploy FLX-Nano, accumulate errors on a specific domain (e.g., math), let meta-generator produce deltas, measure accuracy improvement on held-out math problems. Compare against standard fine-tuning on the same error data.

**Key metric:** Accuracy improvement per delta, with compute cost comparison.

---

## Training Compute Estimate

145M base model is slightly above GPT-2 scale.

| Phase | Duration (8× A100) | Notes |
|-------|---------------------|-------|
| Phase 0 — Cortex specialization | ~3-5 days | Needs domain-labeled data |
| Phase 1 — Delta-receptive pretraining | ~1-2 weeks | Standard corpus, delta composition within cortices |
| Phase 2 — Thermal training | ~2-3 days | Frozen cortex bases, train τ + strata gates |
| Phase 3 — Memory training | ~3-5 days | Conversation chains |
| Phase 4 — Meta-generator | ~1-2 days | Smallest phase |
| **Total** | **~5 weeks** | Single 8-GPU node |

This is intentionally small — FLX-Nano is a validation vehicle, not a production model.

---

## What Success Looks Like

The research risks are twofold:
1. Does cortex specialization produce clean domain separation via the thalamic router?
2. Does delta-receptive pretraining within cortices produce a better compositional substrate?

If Experiment 0 shows clean cortical differentiation and Experiment 1 shows clear wins on domain transfer per parameter, the path is:

1. Publish the architecture paper with Nano results
2. Train FLX-7B with the validated 5-phase curriculum and more cortices
3. Open-source the base + cortical format
4. The .flx format spec writes itself from whatever the cortical model needs to serialize — now including cortex maps, strata weights, and cross-cortical bridge state

The model creates the gravity. The format follows. The ecosystem builds on top.
