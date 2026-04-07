# 08 — Capabilities

## Native Capabilities — What Falls Out of the Architecture for Free

### 1. Mid-Conversation Save/Resume with Zero Loss

State hub persists KV cache + episodic buffer + thermal history. Close your laptop, open it tomorrow, and the model is exactly where it was. Not re-prompted with a summary — actually mid-thought. No other architecture does this without external scaffolding.

### 2. Cross-Session Memory Without RAG

Episodic buffer compresses old context into dense vectors. Memory controller retrieves them when relevant. "Hey, last week you mentioned X" just works — the model remembers as a native inference capability, not a pipeline add-on.

### 3. Hot-Swappable Expertise at Runtime

Activate or deactivate entire cortices, individual strata, or specific deltas within a stratum. Need code expertise? Make sure the Code cortex is loaded with its expert strata. Don't need medical knowledge on this device? Unload the Medical cortex entirely. Runtime knowledge composition at cortex, stratum, and delta granularity.

### 4. Automatic Depth Scaling by Input Difficulty

τ rises on prediction surprise, falls on familiar patterns. Higher τ activates deeper strata within cortices, opens cross-cortical bridges, enables memory retrieval, and triggers refinement loops. The model automatically thinks harder on hard problems and faster on easy ones — not just by using more compute, but by activating deeper knowledge strata and bringing in more domains.

### 5. Calibrated Uncertainty per Domain and Depth

Because knowledge is organized into cortices with hierarchical strata, the model natively knows what it knows and at what level. "I can answer your basic chemistry question confidently (basic stratum, conf=0.95) but this organic synthesis question is at my frontier (frontier stratum, conf=0.3)." Honest uncertainty falls out of the architecture — no calibration post-hoc needed.

### 6. Meaningful Model Versioning and Diffing

Knowledge is organized by cortex, stratum, and delta with provenance. Diff two snapshots: "d003 added to Math cortex / expert stratum, trained on competition_math_v3, confidence 0.91." Model debugging is tractable — regression means identifying which cortex and which delta caused it.

### 7. Selective, Clean Forgetting

Remove a delta from a specific cortex stratum. Or unload an entire cortex. The knowledge is gone, cleanly, without affecting other domains. A bad medical delta doesn't touch the math cortex at all — they're in separate weight spaces. Current models have no mechanism for domain-scoped forgetting.

### 8. Cortex Transplantation

Two FLX models sharing the same base can exchange entire cortices. Take the Math cortex from a math-specialized model and transplant it into a general-purpose model, replacing its weaker Math cortex. This is stronger than delta sharing — it's transplanting an entire specialized brain region, strata and all, between models.

### 9. Collaborative Reasoning

Hand someone your .flx file mid-conversation. They resume with the same model state, memory, thermal history, and cortical activation state. Two people can work on the same reasoning chain asynchronously. Natively impossible with current serving infrastructure.

### 10. Graceful Degradation on Constrained Hardware

On a weaker device, unload non-essential cortices or raise the stratum floor so only basic strata fire. The model loses deep domain expertise but keeps foundational capabilities. Degrades structurally (less specialized knowledge, fewer domains) rather than randomly (quantization artifacts). You choose which cortices and which depth to shed.

### 11. Efficient Fine-Tuning That Stacks Without Forgetting

New learning = new delta in the right cortex + stratum. Deltas compose by design within their stratum. Fine-tune on medical, then legal — they land in different cortices entirely, zero interference. Not even additive — completely separate weight spaces.

### 12. Reproducible Inference

Same .flx file + same input = same output. Thermal history, cortical state, delta stacks per stratum, episodic buffer — all deterministic and serialized. Current models are reproducible only with externally controlled runtime state. Here it's in the file.
