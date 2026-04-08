# Surgical Implants — Transplanting Specialist LLMs into Cortices

## Concept

Instead of training each cortex from scratch, graft pre-trained transformer layers from existing open-source specialist LLMs directly into FLX's cortex slots. The router, merger, and trunk stay as-is — only the cortex internals change.

The result: a 145M-parameter model with specialist knowledge distilled from billion-parameter experts.

## Why FLX is suited for this

- Cortices are **cleanly isolated modules** — each is just a stack of transformer layers with a standard input/output interface (`[batch, seq, d_model]` in, same shape out)
- The **router doesn't care** where cortex weights came from — it learns to route based on output quality
- The **merger is weight-agnostic** — it combines whatever the cortices produce
- Phase 0 already trains the router to distribute traffic — after transplant, a short fine-tuning round re-learns routing for the new cortex capabilities

## Candidate donors

| Cortex slot | Donor model | Why |
|---|---|---|
| code | CodeLlama, DeepSeek-Coder, StarCoder2 | Purpose-built for code generation |
| math | Llemma, DeepSeek-Math, InternLM-Math | Trained on math corpora + reasoning chains |
| science | Galactica, SciGLM | Trained on papers, citations, scientific text |
| language | Phi-2/3, TinyLlama, Qwen2 | Strong general language at small scale |
| reasoning | Mistral, Orca-2 | Instruction-tuned with chain-of-thought |

## Transplant process

1. **Extract layers** from donor model (e.g., layers 4–8 of a 32-layer model — the "middle knowledge" layers, not the embedding or head)
2. **Dimension adapter** — if donor `d_model` differs from FLX's 512, add projection layers at input/output (see `dimension-agnostic.md`)
3. **Load into cortex** — replace the cortex's `Stratum` transformer layers with donor layers
4. **Freeze cortex, fine-tune interface** — short training run (~1K steps) where only the trunk output layer, adapter projections, router, and merger train. This teaches the system to "talk to" the new cortex
5. **Unfreeze and fine-tune end-to-end** — optional pass at low learning rate to let everything adapt together

## What's needed to build this

- [ ] `DomainCortex.load_donor_weights(state_dict, layer_range, adapter_dim)` method
- [ ] Layer mapping utility — maps donor layer names → FLX stratum layer names
- [ ] Dimension-agnostic cortices (see `dimension-agnostic.md`)
- [ ] A "transplant fine-tuning" phase (Phase 0.5?) — freeze cortex weights, train only adapters + router + merger
- [ ] Validation: feed domain-labeled data through `forward_raw()` to confirm each transplanted cortex handles its domain

## Risks

- **Representation mismatch**: Donor layers expect input distributions from their own earlier layers, not from FLX's trunk. The fine-tuning step addresses this but may not fully bridge the gap for very different architectures.
- **Stratum structure**: FLX cortices have 3 strata (intermediate/expert/frontier) with thermal gating. Donor layers need to be split across these, or loaded into a single stratum with the others left empty/minimal.
- **Tokenizer mismatch**: If the donor used a different tokenizer, the embedding expectations won't align. The trunk's embedder is the interface — as long as trunk output is reasonable, this shouldn't matter at the cortex level.

## Open questions

- How many donor layers per cortex is optimal? (2 per stratum = 6 total, or load all into one stratum?)
- Is it better to distill (train a small model to mimic the large one) or directly transplant (copy raw weights with adapters)?
- Could this work incrementally — start with from-scratch Phase 0, then progressively replace cortices with transplants as better donors become available?
