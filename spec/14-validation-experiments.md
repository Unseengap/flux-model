# 14 — Validation Experiments

## Purpose

Five training phases are complete. Before benchmarking or scaling to FLX-7B, we must verify that the architectural bets actually pay off. These experiments are not benchmarks — they are targeted probes that isolate each architectural claim and either confirm it or expose what needs fixing.

Each experiment loads the completed Phase 5 model (`nano_phase5.flx`) and tests one specific capability in isolation.

---

## Experiment 0 — Cortex Specialization

**Claim:** After training, each cortex develops a distinct domain receptive field. Expert/frontier strata show strong domain preference; basic strata may overlap.

**Protocol:**

1. Load `nano_phase5.flx` in eval mode
2. Prepare domain-labeled test data: 100 samples per domain (language, math, code, science, reasoning) — use held-out slices from Phase 0/1 data
3. Forward each sample through shared trunk → thalamic router
4. Record router scores `dict[str, Tensor]` per sample
5. For each cortex, compute: what fraction of its top-scoring inputs belong to one domain?

**Metrics:**

| Metric | Formula | Target |
|--------|---------|--------|
| Specialization score | `max_domain_fraction` per cortex, averaged over expert+frontier strata | > 0.7 |
| Overlap ratio | Fraction of samples where ≥2 cortices score > 0.3 | < 0.4 for expert strata |
| Routing entropy | `H(domain_scores)` averaged per sample | Lower for hard inputs (high τ) |

**What to record:** Per-sample routing scores matrix `[num_samples, num_cortices]`, domain labels, τ values. Visualize as heatmap: cortex × domain.

---

## Experiment 1 — Delta Receptivity

**Claim:** A cortical model with delta-receptive cortices outperforms the same model with deltas disabled on domain-specific inputs.

**Protocol:**

1. Load `nano_phase5.flx` — this is the full model with all deltas active
2. Prepare domain-specific test data: 50 samples per domain
3. Run inference with all deltas active → record per-sample loss
4. Disable all deltas (set `delta_stack.deltas = []` per stratum, or zero out B matrices) → record per-sample loss
5. Compare: domain loss with deltas vs. without

**Note:** The spec describes comparing against a standard transformer of equal parameter count. Since we don't have a separately trained baseline, we use deltas-on vs. deltas-off as a proxy — this isolates the delta contribution within the same model.

**Metrics:**

| Metric | Formula | Target |
|--------|---------|--------|
| Delta lift | `(loss_without - loss_with) / loss_without` per domain | > 0.05 (5% improvement) |
| General preservation | Loss on mixed/general data with vs. without deltas | < 0.02 regression |
| Per-stratum impact | Repeat with deltas disabled per stratum level | Expert+frontier deltas contribute most |

**What to record:** Per-domain loss with/without deltas, per-stratum ablation results.

---

## Experiment 2 — Thermal Efficiency

**Claim:** Thermally-routed FLX-Nano uses significantly fewer active parameters on easy inputs with no accuracy loss.

**Protocol:**

1. Load `nano_phase5.flx`
2. Prepare difficulty-stratified test data:
   - **Easy** (40 samples): simple completions, basic arithmetic, "Hello world"
   - **Medium** (40 samples): grade-school math, basic code, standard questions
   - **Hard** (40 samples): competition math, complex code, multi-step reasoning
3. For each sample:
   - Let the thermal estimator compute τ naturally
   - Record τ value
   - Count active strata (those where `tau >= tau_min`)
   - Count active bridges (those where `tau >= bridge.tau_min`)
   - Record whether memory retrieval activated (`tau >= 0.5`)
   - Record prediction loss
4. Compare compute proxy (active strata + bridges) across difficulty levels

**Metrics:**

| Metric | Formula | Target |
|--------|---------|--------|
| τ separation | Mean τ for easy vs. hard inputs | Easy < 0.35, Hard > 0.6 |
| Strata savings | Active strata on easy / active strata on hard | < 0.5 (50% fewer) |
| Accuracy preservation | Loss on easy inputs (low τ path) | No worse than full-depth path |
| Bridge activation | Fraction of hard inputs activating bridges | > 0.7 |

**What to record:** Per-sample τ, active component counts, loss. Plot τ distribution by difficulty tier.

---

## Experiment 3 — Cross-Session Memory

**Claim:** The memory controller can retrieve relevant information from compressed episodes of prior context.

**Protocol:**

1. Load `nano_phase5.flx`
2. Create synthetic multi-turn sequences:
   - Turn 1: Present a fact ("The capital of Freedonia is Belvaux")
   - Turn 2-4: Unrelated filler text (different topics)
   - Turn 5: Query the fact ("What is the capital of Freedonia?")
3. For each sequence:
   - Process turns 1-4 through the model, compressing into episodic buffer via EpisodicCompressor
   - Process turn 5 with episodic buffer available
   - Record logits for the query turn
4. Compare:
   - **With memory:** Episodic buffer populated from turns 1-4
   - **Without memory:** Empty episodic buffer (fresh context only)

**Metrics:**

| Metric | Formula | Target |
|--------|---------|--------|
| Memory retrieval gain | Loss(with_memory) < Loss(without_memory) | Consistent improvement |
| Retrieval activation | Does τ rise on the query turn? | τ > 0.5 on fact queries |
| Episode quality | Cosine similarity between compressed episode and original turn embedding | > 0.3 |

**What to record:** Per-turn loss with/without memory, τ trajectory across turns, episode compression quality.

---

## Experiment 4 — Online Delta Quality

**Claim:** The meta-delta generator produces candidate deltas that improve model performance on error-producing inputs.

**Protocol:**

1. Load `nano_phase5.flx`
2. Prepare a small error-producing test set: 30 samples where the model's loss > 2.0 (high error)
3. For each error batch:
   - Accumulate error signals (inputs + loss values)
   - Feed to meta-delta generator → get candidate delta (A, B matrices + target cortex/stratum)
   - Evaluate on the same inputs before pushing the delta
   - Push the candidate delta to the target stratum
   - Evaluate on the same inputs after pushing
   - Pop the delta (clean rollback)
4. Measure improvement from each candidate delta

**Metrics:**

| Metric | Formula | Target |
|--------|---------|--------|
| Acceptance rate | Fraction of deltas that improve held-out loss | > 0.3 |
| Improvement magnitude | Mean loss reduction on accepted deltas | > 0.1 |
| Rollback cleanliness | Loss after push+pop == loss before push | Exact match |
| Targeting accuracy | Does the meta-gen target the correct cortex for the error domain? | > 0.5 |

**What to record:** Per-delta: target cortex/stratum, loss before/after, accepted/rejected.

---

## Experiment 5 — Abstract Rule Induction (Phase 5)

**Claim:** The hypothesis head can induce transformation rules from demonstration pairs and apply them to novel test inputs, with consistency scores that correlate with prediction quality.

**Protocol:**

1. Load `nano_phase5.flx` with HypothesisHead attached
2. Prepare held-out ARC tasks (not seen during Phase 5 training):
   - Use ARC-AGI evaluation split tasks reserved for testing
   - 50 tasks with 2-5 demonstrations each
3. For each task:
   - Run `phase5_training_step` in eval mode (no gradients)
   - Record: pred_loss, consistency score, cons_target, num_loops
4. Additionally test refinement loop behavior:
   - Run same tasks with `max_loops=1` vs. `max_loops=3`
   - Measure if additional loops improve predictions (pred_loss decreases)

**Metrics:**

| Metric | Formula | Target |
|--------|---------|--------|
| Consistency-quality correlation | Spearman ρ between consistency and -pred_loss | > 0.3 |
| Cons tracks target | Mean |consistency - cons_target| | < 0.15 |
| Loop benefit | pred_loss(max_loops=3) < pred_loss(max_loops=1) | Consistent improvement |
| Task generalization | pred_loss on unseen task types vs. seen types | < 2× regression |

**What to record:** Per-task: pred_loss, consistency, cons_target, num_loops, task type.

---

## Experiment 6 — Inference Pipeline Verification

**Claim:** The full FLX pipeline produces coherent outputs with all components active and τ naturally computed.

**Protocol:**

1. Load `nano_phase5.flx` — ensure all components attached (router, thermal, bridges, memory, meta-gen, hypothesis head)
2. Prepare 20 diverse prompts spanning all 5 domains
3. For each prompt:
   - Forward through `model.forward()` with τ computed naturally
   - Record: τ value, top-scoring cortex, active strata count, logits shape
   - Greedy-decode 50 tokens from the logits
4. Verify:
   - No NaN/Inf in logits
   - τ varies across prompts (not constant)
   - Different prompts route to different cortices
   - Decoded text is non-degenerate (not repetition loops)

**What to record:** Per-prompt: τ, routing scores, decoded text, any anomalies.

---

## Experiment 7 — Format Crystallization Audit

**Claim:** The .flx serialization format faithfully captures and restores the complete model state with all Phase 0-5 components.

**Protocol:**

1. Load `nano_phase5.flx` → this is model_A
2. Run 10 diverse prompts through model_A → record logits_A
3. Save model_A to a new path `nano_phase5_resaved.flx`
4. Load from that new path → this is model_B
5. Run the same 10 prompts through model_B → record logits_B
6. Verify:
   - `torch.allclose(logits_A, logits_B)` for all prompts
   - Manifest YAML contains all expected fields
   - All component directories exist with correct structure
7. Perform structural audit:
   - Walk the .flx directory tree
   - Verify cortex/stratum/delta hierarchy matches model structure
   - Check manifest.yaml fields against actual state
   - Verify hypothesis_head is serialized (currently NOT in serialization.py — this is a known gap)

**Structural audit checklist:**

```
mymodel.flx/
├── manifest.yaml                  ✓ version, cortex_registry, has_*
├── shared_trunk/weights.bin       ✓ trunk state
├── thalamic_router/weights.bin    ✓ router state
├── cortices/{domain}/{stratum}/   ✓ per-cortex, per-stratum
│   ├── weights.bin                ✓ stratum base weights
│   └── deltas/d00N.bin + .yaml    ✓ delta weights + provenance
├── bridges/*_weights.bin + .yaml  ✓ bridge state + metadata
├── state_hub/                     ✓ thermal.json, episodes, history
│   ├── thermal.json
│   ├── episode_buffer.bin
│   └── cortex_activation_history.json
├── thermal_estimator/weights.bin  ✓
├── memory_controller/weights.bin  ✓
├── meta_generator/weights.bin     ✓
├── hypothesis_head/weights.bin    ⚠ NOT CURRENTLY SERIALIZED
├── cortex_merger_weights.bin      ✓
└── decoder_weights.bin            ✓
```

**Known gaps to fix:**
- `hypothesis_head` is not saved/loaded by `save_flx`/`load_flx`
- Manifest should include `has_hypothesis_head` flag
- EpisodicCompressor state is not serialized

**What to record:** Bitwise reproducibility (logits match), directory structure diff, manifest completeness.

---

## Experiment Ordering

Run in this order — each builds confidence for the next:

1. **Experiment 7 — Format Crystallization** (verify the model loads correctly before testing anything)
2. **Experiment 6 — Inference Pipeline** (verify forward pass works end-to-end)
3. **Experiment 0 — Cortex Specialization** (foundational: do cortices even specialize?)
4. **Experiment 2 — Thermal Efficiency** (does τ actually vary by difficulty?)
5. **Experiment 1 — Delta Receptivity** (do deltas help?)
6. **Experiment 3 — Cross-Session Memory** (does memory retrieval work?)
7. **Experiment 4 — Online Delta Quality** (does meta-gen produce useful deltas?)
8. **Experiment 5 — Phase 5 Rule Induction** (does hypothesis head generalize?)

---

## Success Criteria

| Result | Path Forward |
|--------|-------------|
| **All experiments pass** | Publish architecture paper, begin FLX-7B training |
| **Experiments 0-2 pass, 3-5 mixed** | Fix memory/meta-gen/hypothesis — these are later-phase additions, easier to iterate |
| **Experiment 0 or 1 fails** | Core architecture problem — cortex separation or delta receptivity doesn't work. Major redesign needed before scaling |
| **Experiment 7 fails** | Serialization bug — fix immediately, it blocks all other experiments |
