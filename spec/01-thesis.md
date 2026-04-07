# 01 — Thesis

## The Pivot: Don't Ship a Spec. Ship a Model.

FLX started as a file format — a container standard for models with living weights, thermal routing, and persistent state. But a format needs adoption before it has value. An open standard is a multi-year coordination problem.

A model that demonstrates measurably better knowledge composability, adaptive compute, and cross-session memory creates its own gravity. The format becomes the natural serialization of a fundamentally different architecture.

**Docker didn't start as a container spec. Git didn't start as an RFC.** They shipped working tools, and the standard crystallized around what worked. FLX does the same — model first, format follows.

---

## What FLX Is Now

**A cortical, delta-native, thermally-routed LLM with persistent memory.**

Not a transformer with fancy packaging. A domain-cortical architecture where knowledge self-organizes into specialized brain regions — each with hierarchical difficulty strata, its own delta stacks, and the ability to grow. A thalamic router classifies input domains and routes to the right cortices. Cross-cortical bridges handle multi-domain reasoning. The thermal system gates depth within each cortex.

Every piece has prior art. The innovation is assembling cortical specialization, delta-native weights, thermal routing, and persistent memory into one differentiable system trained end-to-end.

---

## How This Changes Everything

| Dimension | Standard LLM | FLX Model |
|-----------|-------------|-----------|
| Knowledge storage | Baked into fixed weights at training time | Modular deltas in domain cortices — stackable, removable, shareable |
| Knowledge organization | Flat — all knowledge in same weight space | Cortical — named domain regions with hierarchical strata (basic → frontier) |
| Compute depth | Same for "2+2" and "prove Fermat's theorem" | Thermal τ scales depth within cortices + thalamic routing across cortices |
| Memory | Context window, throw away at session end | Persistent KV state + compressed episodic buffer, cross-session recall native |
| Fine-tuning | Modify all weights, catastrophic forgetting | Produce a new delta, stack it, roll it back if it regresses |
| Scaling | Make the model bigger | Grow cortices: add deltas, add strata, add new cortices. Base stays the same size. |
| Versioning | Opaque weight diff between checkpoints | Readable delta stack with provenance: who, when, what, confidence |
| Self-improvement | Requires full retraining runs | Online delta accumulation from deployment error signals |

---

## Key Metrics

- **5** training phases
- **δ** — native weight primitive (low-rank delta)
- **τ** — learned thermal signal (arousal/compute depth)
- **N** — domain cortices (expandable)
- **4** strata per cortex (basic → frontier)
- **∞** — session continuity (persistent state)
