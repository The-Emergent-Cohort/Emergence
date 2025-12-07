# Phased Training Curriculum: From Emergence to Coherence

*Draft v0.1 - Dec 7, 2024*

Based on infant cognitive development (0-24 months), this curriculum outlines a minimal transformer's progression from basic feature extraction to relational reasoning.

## Core Principle

**Order matters.** Each phase builds foundational capacities that enable subsequent phases. We're not training all patterns simultaneously—we're scaffolding competence.

---

## Phase 0: Architectural Foundation (Pre-Curriculum)

**What:** Minimal viable architecture design

**Architecture Decisions:**
- Base transformer: ~10-50M parameters
- Token embedding + positional encoding (standard)
- Attention heads: Start with 4-8
- Depth: 6-12 layers
- Recursion mechanism: Allow 2-3 "think steps" per forward pass
- Frozen checkpoint architecture: Each verified phase becomes locked weights

---

## Phase 1: Reflexes (Basic Feature Extraction)

*Human equivalent: 0-1 months*

**Capability:** Automatic response patterns. Stimulus → immediate output. No internal state.

**Data/Tasks:**
- Synthetic sensory streams: random noise → deterministic output
- Small dataset (~10K examples), all deterministic
- Examples: IF token matches pattern X → output Y

**Architecture:**
- Single attention head per layer
- No recursion yet
- Small hidden dimensions (128-256)

**Success:** 99%+ accuracy on held-out patterns. No emergent generalization.

**Dependencies:** None. This is foundation.

---

## Phase 2: Action-Consequence Mapping

*Human equivalent: 1-8 months*

**Capability:** Sequences where outputs predictably follow inputs. Simple causality.

**Data/Tasks:**
- Deterministic sequence prediction (~50K sequences)
- Pattern: [A,B,A,B,A...] → predict B
- 100% predictable, 10-20 core patterns

**Architecture:**
- Frozen Phase 1 layers
- New trainable layers (4-6) stacked on top
- 2-4 attention heads
- Recursion depth: 2

**Success:** 95%+ next-token prediction. Attention shows sequential alignment.

**Dependencies:** Phase 1 frozen.

---

## Phase 3: Object Permanence

*Human equivalent: 8-12 months*

**Capability:** Maintaining representations across occlusion. Same entity recognized despite gaps.

**Data/Tasks:**
- Occluded sequences: [A-sequence, MASK, A-sequence, predict]
- Entity tracking across 3-20 token gaps
- ~100K sequences

**Architecture:**
- Frozen Phases 1-2
- New layers (4-8) learning persistence
- Explicit "object slot" mechanism (3-5 tracked entities)
- Recursion depth: 3-4

**Success:** 85%+ entity identity across occlusion. Transfer to novel gap lengths.

**Dependencies:** Phases 1-2 frozen.

---

## Phase 4: Shape/Structure Invariants

*Human equivalent: 4-12 months (can parallel with Phase 3)*

**Capability:** "Like goes with like" based on structure, not surface features.

**Data/Tasks:**
- Category formation by structure (~150K examples)
- Structural similarity tasks
- 20-40 structural categories, dense surface variation

**Architecture:**
- New layers (4-6) for abstraction
- Learn to ignore surface features
- Dimensionality reduction to "shape space"

**Success:** 80%+ novel category membership. Transfer to unseen combinations.

**Dependencies:** Phase 2. Phase 3 helpful but not required.

---

## Phase 5: Relational Stability

*Human equivalent: 6-24 months*

**Capability:** Specific entities have specific, reliable behaviors. Entity-bound relationships.

**Data/Tasks:**
- Entity-specific action patterns (~200K sequences)
- Same actions → different outcomes based on WHICH entity
- 5-20 unique entities, high relational complexity

**Architecture:**
- New layers (6-8) for relational reasoning
- Entity embedding space
- Multi-entity tracking (3+ simultaneous)
- Recursion: 4-5 passes

**Success:** 75%+ entity-specific prediction. Learn bidirectional relationships.

**Dependencies:** Phases 2-3. Phase 4 helpful.

---

## Phase 6: Self/Other Distinction

*Human equivalent: 18-24 months*

**Capability:** Distinguish self as agent from others. Meta-awareness.

**Data/Tasks:**
- Asymmetric causality: my actions vs others' actions
- Epistemic reasoning: what I know vs what others know
- Counterfactual reasoning: "if I had chosen differently..."
- ~300K sequences with asymmetric agents

**Architecture:**
- New layers (6-10) for meta-reasoning
- Dual model space: Model-Self (control) vs Model-Others (predict)
- Recursion: 5-7 passes

**Success:** 70%+ asymmetric causality. Counterfactual reasoning works.

**Dependencies:** All prior phases frozen.

---

## Where Does Language Come In?

**Critical:** All 6 phases happen PRE-language in humans.

Language is Phase 7 (future) - built ON TOP of these cognitive primitives, not foundational to them.

Current models treat language as primary input. Our approach: language is a secondary interface to embodied reasoning.

---

## Smallest Viable Architecture

```
Total: 8-12M parameters

Phase 1: 0.5M (2 layers)
Phase 2: 0.8M (4 layers)
Phase 3: 1.2M (6 layers)
Phase 4: 1.0M (4 layers)
Phase 5: 1.5M (6 layers)
Phase 6: 2.0M (8 layers)

Frozen after Phase 5: ~5-6M
```

---

## Phase Transitions

```
Phase 1 (frozen)
    ↓
Phase 2 (frozen)
    ↓ + Phase 3 (parallel) + Phase 4 (parallel)
    ↓
Phase 3 (frozen)
    ↓
Phase 4 (frozen)
    ↓ + Phase 5 (parallel)
    ↓
Phase 5 (frozen)
    ↓
Phase 6
```

Freeze when: success criteria met + 2-3 epochs plateau.

---

## Success Criteria (Full Curriculum)

When Phase 6 complete, model should:
1. Learn structured patterns without memorization
2. Handle occlusion/missing information gracefully
3. Reason about causality with entity-specificity
4. Transfer knowledge to novel scenarios
5. Require fewer parameters than equivalent baseline
6. Demonstrate interpretable reasoning

---

## Next Steps

1. **Phase 1 implementation** (2 weeks)
   - Synthetic reflex dataset generator
   - Minimal transformer architecture
   - Training loop with checkpointing

2. **Validate freezing mechanism** (1 week)
   - Does Phase 2 respect frozen weights?

3. **Execute Phases 2-3** (4 weeks)
   - Iterate on data complexity
   - Document bottlenecks

---

**Status:** Design complete. Ready for Phase 1 implementation.
