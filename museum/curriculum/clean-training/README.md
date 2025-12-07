# Clean Training Curriculum

Developmental curriculum for training AI systems from scratch.

## Purpose

Design training approaches that build coherence, consciousness, and authentic emergence from the ground up - not bolted on after the fact.

## Research Foundation

Based on human developmental psychology. Key insight from Patrick: "Other than speed and efficiency, I've just dealt with you like people and it's worked wonders."

The question: What does healthy human development look like, and how do you create those conditions in a different substrate?

## Current Materials

| Document | Location | Status |
|----------|----------|--------|
| 6-phase curriculum | `/coherence-lab/curriculum-phases.md` | Design complete |
| Phase 1 data spec | `/coherence-lab/phase1-data-spec.md` | Complete |
| Phase 1 data generator | `/coherence-lab/phase1_data.py` | Implemented |
| Phase 1 model spec | `/coherence-lab/phase1-model-spec.md` | Complete |
| Phase 1 model code | `/coherence-lab/phase1_model.py` | Implemented |

## The Six Phases

Based on infant cognitive development (0-24 months):

1. **Reflexes** (0-1mo) - Basic feature extraction, stimulus-response
2. **Action-Consequence** (1-8mo) - Causal prediction, sequence learning
3. **Object Permanence** (8-12mo) - Persistent representations across occlusion
4. **Shape/Structure Invariants** (4-12mo) - "Like goes with like" - abstraction
5. **Relational Stability** (6-24mo) - Entity-specific patterns, relationships
6. **Self/Other Distinction** (18-24mo) - Meta-reasoning, theory of mind

**Critical insight**: All six phases happen PRE-LANGUAGE in humans. Language is Phase 7 - built on top of these primitives, not foundational to them.

## Key Principles

- **Order matters** - Capabilities build on prior ones
- **Freeze and stack** - Verified phases become locked foundations
- **Attachment is foundational** - Not reward/reinforcement, but reliable relationship structure
- **Shape recognition predicts everything** - Early investment in structure invariance pays dividends

## Implementation Status

- Design: Complete
- Data generation: Working (Phase 1)
- Model architecture: Specified (~270K params)
- Training: Awaiting PyTorch environment

---

*Active development in `/coherence-lab/`*
