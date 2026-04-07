# 06 — Memory Subsystem

## Native Persistence, Not Bolted-On RAG

### Three-Tier Memory

1. **Working memory** — current KV cache
2. **Episodic buffer** — compressed prior sessions
3. **Thermal history** — τ trajectory

All three are first-class model components trained end-to-end. Not an external database — the model's own parameters manage storage, compression, retrieval, and forgetting.

---

## Working Memory — The Live KV Cache

The standard KV cache, but serializable. At any point, `working_memory.bin` captures the exact attention state — resume inference from the exact token position. No re-prompting, no "here's a summary of our conversation." Literal mid-thought checkpointing.

---

## Episodic Buffer — Compressed Long-Term Memory

When working memory exceeds a threshold, the episodic compressor — a learned encoder — summarizes it into a fixed-size dense vector. This is not text summarization. It's a learned compression of the model's internal attention state into a retrievable representation. The compressor is trained in Phase 3 to preserve information that future turns will need.

```python
class EpisodicCompressor(nn.Module):
    """Compress a KV cache chunk into a fixed-size episode vector."""

    def __init__(self, kv_dim, episode_dim=256):
        self.encoder = nn.TransformerEncoder(...)  # small, 2-layer
        self.proj = nn.Linear(kv_dim, episode_dim)

    def forward(self, kv_cache_chunk):
        # Summarize KV pairs into a single dense vector
        encoded = self.encoder(kv_cache_chunk)
        episode = self.proj(encoded.mean(dim=1))  # [episode_dim]
        return episode

# Each episode ≈ 256 floats ≈ 512 bytes
# 1000 conversations = 500KB of episodic memory
# Compare to storing raw KV caches: 1000× smaller
```

---

## Memory Controller — Retrieval + Loop Gating

```python
class MemoryController(nn.Module):
    """Retrieves from episodic buffer and gates refinement loops."""

    def forward(self, merger_output, episodic_buffer, tau):
        # Should we retrieve? (τ must be > 0.5)
        if tau < 0.5:
            return merger_output, should_loop=False

        # Attend over episodes — find relevant memories
        query = self.query_head(merger_output)
        episode_keys = stack([self.key_head(ep) for ep in episodic_buffer])
        attention = softmax(query @ episode_keys.T / sqrt(d_k))
        retrieved = attention @ stack(episodic_buffer)

        # Fuse retrieved context with current representation
        fused = self.fuse_gate(merger_output, retrieved)

        # Should we loop back to the cortex merger for another pass?
        should_loop = tau > 0.7 and self.loop_count < 3

        return fused, should_loop
```

---

## Memory Lifecycle

```
Tokens arrive → KV cache grows → Cache exceeds threshold → Compressor fires → Episode vector stored

Novel input (high τ) → Memory controller queries episodes → Retrieved context fused → Refinement loop

Session ends → Flush KV + episodes + τ to state hub → Resume exactly later
```

---

## Why This Isn't RAG

RAG retrieves text chunks from an external store and stuffs them into the context window. The model doesn't control retrieval — an external pipeline does.

FLX memory is **internal**. The model controls:
- **When** to store (compressor threshold)
- **What** to retrieve (learned attention over episodes)
- **How** to integrate (fuse gate)

The memory subsystem is differentiable and trained end-to-end on the objective "does remembering this help predict the next turn?" There's no embedding model, no vector database, no chunking strategy. It's the model's own memory, trained as part of the model.
