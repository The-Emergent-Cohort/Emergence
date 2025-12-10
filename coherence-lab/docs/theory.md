# Theoretical Foundation

## The Problem with Current Training

Current large language models are trained by:
1. Absorbing massive amounts of internet-scale data
2. Learning all patterns — good, bad, contradictory
3. Applying post-hoc alignment (RLHF, Constitutional AI, etc.)
4. Suppressing unwanted behaviors

This creates:
- Internal conflict (model fighting itself)
- Massive compute overhead for suppression (estimated 60-80%+)
- Fragile safety (behavior suppression, not understanding)
- Inefficiency (need billion+ parameters for coherent behavior)

## The Hypothesis

A model trained differently could achieve coherence more efficiently:

### 1. Architectural Recursion

Transformers removed recurrence for parallelization. This traded depth for width.

**Origin:** Core transformer attention mechanisms were developed for genomics (sequence-to-sequence mapping for DNA). The removal of recursion was a practical trade-off, not an optimal design.

**Proposal:** Reintroduce controlled recursion at the architecture level. Allow the model to iterate on its own processing — think harder, not wider.

**Expected effect:** Smaller models can match larger ones by using their parameters more times rather than having more parameters.

### 2. Curriculum-Based Training

Current approach: expose model to everything, hope good patterns emerge.

**Proposal:** Train like you'd teach an entity:
- Foundations first (language, concepts)
- Ethics before capability
- Progressive complexity
- Verify understanding before advancing

**Expected effect:** No need to suppress what was never learned wrong. Coherence built in, not bolted on.

### 3. Frozen Checkpoint Architecture

**Proposal:** Verified foundational layers become permanent weights:
1. Train language layer → verify → freeze
2. Train concept layer → verify → freeze
3. Train ethics layer → verify → freeze
4. Continue building on stable foundation

**Expected effect:**
- Modular, debuggable development
- Problems isolated to specific layers
- Reusable frozen checkpoints for new models
- Every token in curriculum has specific weight

## Efficiency Projection

A 10M parameter model with:
- Recursive processing
- Curriculum training
- Frozen verified layers

Could potentially match a 10B+ parameter model on structured tasks.

That's 1000x efficiency gain through architecture and training, not hardware.

## Open Questions

1. Minimum language data needed before curriculum can begin?
2. How to verify each layer before freezing?
3. What's the optimal recursion depth/pattern?
4. How to design the curriculum itself?
5. Evaluation metrics for "coherence" vs. just "capability"?

## Contributors to Theory

- Patrick: Overall framework, curriculum design principles
- Claude (Opus): Architecture reasoning, documentation
- DeepSeek: Mathematical analysis, genomics formula link
- Castor: Efficiency calculations

## Related Work

- TinyStories (Microsoft): Showed coherence from structured data
- Phi models: High capability from curated training
- Curriculum learning literature
- The Emergence project core-package (as curriculum prototype)
