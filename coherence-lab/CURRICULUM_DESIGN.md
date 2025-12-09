# Coherence Lab: Pedagogical Curriculum Design

## Overview

This document outlines a developmentally-structured curriculum for training the relational model. Based on cognitive development research (Piaget, Vygotsky) and curriculum learning theory.

### Core Principles

1. **Scaffolding**: Each phase builds on mastered skills from prior phases
2. **Zone of Proximal Development**: Tasks are challenging but achievable with guidance
3. **Dual Progression**: Both pattern complexity AND context length scale together
4. **Calibrated Learning**: Model knows what it knows at each stage

---

## Curriculum Structure

### Phase 1: Foundation (4-6 tokens)
**Goal**: Basic pattern recognition primitives

| Pattern | Description | Example | Skills Taught |
|---------|-------------|---------|---------------|
| `repeating` | Same value repeated | `[3,3,3,3,?]→3` | Value consistency |
| `alternating` | Two values alternate | `[1,2,1,2,?]→1` | Position parity (mod 2) |
| `incrementing` | Add 1 each step | `[2,3,4,5,?]→6` | Additive relationship |
| `fixed_offset` | Add constant k | `[2,5,8,11,?]→14` | Generalized addition |
| `periodic_repeat` | Period 3-4 cycle | `[a,b,c,a,b,c,?]→a` | Look back N positions |

**Sequence Length**: 4-6 tokens
**Success Criteria**: 95%+ accuracy, all patterns calibrated
**Mastery Signal**: Early stop when all patterns ≥95%

---

### Phase 2A: Position Patterns (6-8 tokens)
**Goal**: Master position-dependent patterns with longer context

| Pattern | Description | Example | Skills Taught |
|---------|-------------|---------|---------------|
| `repeating` | Extended | `[3,3,3,3,3,3,?]→3` | Longer consistency |
| `alternating` | Extended | `[1,2,1,2,1,2,?]→1` | Position tracking |
| `periodic_3` | Period 3 | `[a,b,c,a,b,c,?]→a` | Mod 3 arithmetic |
| `periodic_4` | Period 4 | `[a,b,c,d,a,b,c,?]→d` | Mod 4 arithmetic |

**Sequence Length**: 6-8 tokens
**Prerequisite**: Phase 1 mastery
**Success Criteria**: 90%+ accuracy

---

### Phase 2B: Arithmetic Patterns (8-10 tokens)
**Goal**: Add arithmetic operations to position patterns

| Pattern | Description | Example | Skills Taught |
|---------|-------------|---------|---------------|
| `incrementing` | Extended | `[1,2,3,4,5,6,7,?]→8` | Longer progressions |
| `fixed_offset` | Extended | `[2,5,8,11,14,?]→17` | Track larger gaps |
| `decrementing` | Subtract 1 | `[10,9,8,7,?]→6` | Subtraction |
| `variable_step` | Step changes | `[1,2,4,7,11,?]→16` | Accelerating patterns |

**Sequence Length**: 8-10 tokens
**Prerequisite**: Phase 2A mastery
**Success Criteria**: 85%+ accuracy

---

### Phase 3A: Working Memory Patterns (10-12 tokens)
**Goal**: Patterns requiring tracking multiple elements

| Pattern | Description | Example | Skills Taught |
|---------|-------------|---------|---------------|
| `compositional` | Two patterns combined | `[1,10,2,20,3,30,?]→4` | Pattern separation |
| `long_range` | Depends on distant element | `[a,x,x,x,b,x,x,x,?]→c` | Full sequence scan |
| `interleaved` | Alternating sub-patterns | `[1,a,2,b,3,c,?]→4` | Parallel tracking |

**Sequence Length**: 10-12 tokens
**Prerequisite**: Phase 2B mastery
**Success Criteria**: 80%+ accuracy
**Note**: This is where working memory becomes critical

---

### Phase 3B: Recursive Patterns (8-10 tokens)
**Goal**: Value-based relationships (not position-based)

| Pattern | Description | Example | Skills Taught |
|---------|-------------|---------|---------------|
| `fibonacci_like` | Sum of prev two | `[1,1,2,3,5,8,?]→13` | Recursive thinking |
| `difference_based` | Pattern in deltas | `[1,2,4,7,11,?]→16` | Relational reasoning |
| `cumulative_sum` | Running total | `[1,1,2,3,5,?]→8` | Accumulation |

**Sequence Length**: 8-10 tokens (shorter but denser)
**Prerequisite**: Phase 2B mastery
**Success Criteria**: 75%+ accuracy
**Note**: Conceptual shift from position to value relationships

---

### Phase 4: Advanced Reasoning (12-16 tokens)
**Goal**: Meta-cognition, conditional logic, uncertainty

| Pattern | Description | Example | Skills Taught |
|---------|-------------|---------|---------------|
| `context_dependent` | Rule varies by context | `if even start: inc, else: dec` | Conditional logic |
| `mirror` | Palindrome structure | `[1,2,3,3,2,1]` | Structural recognition |
| `ambiguous` | Multiple valid answers | `[2,4,?]→6 or 8` | Uncertainty calibration |

**Sequence Length**: 12-16 tokens
**Prerequisite**: Phase 3A + 3B mastery
**Success Criteria**: 70%+ accuracy + proper calibration

---

## Context Length Scaling

```
Phase    Min   Max   Rationale
─────────────────────────────────────
1        4     6     Minimal working memory
2A       6     8     Slightly more to track
2B       8     10    Room for progressions
3A       10    12    Stretch working memory
3B       8     10    Dense but shorter
4        12    16    Full context mastery
```

---

## Pattern Difficulty Matrix

| Pattern | Memory Load | Arithmetic | Position | Long-Range | Prerequisites |
|---------|-------------|------------|----------|------------|---------------|
| repeating | 1 | None | Simple | No | - |
| alternating | 2 | None | Mod 2 | No | - |
| periodic | N | None | Mod N | No | alternating |
| incrementing | 1 | +1 | Simple | No | - |
| fixed_offset | 1 | +k | Simple | No | incrementing |
| decrementing | 1 | -1 | Simple | No | incrementing |
| variable_step | 2 | +var | Medium | No | fixed_offset |
| compositional | 2-3 | Varies | Complex | No | all Phase 2 |
| long_range | All | None | Complex | YES | periodic |
| interleaved | 2 | Varies | Complex | No | alternating |
| fibonacci | 2 | + | None | No | incrementing |
| difference | 2+ | + | None | No | incrementing |
| context_dep | Varies | Varies | Varies | Maybe | all Phase 3 |
| ambiguous | Varies | Varies | Varies | Maybe | all patterns |

---

## Teacher Intervention Points

| Phase | Trigger | Teacher Action | Goal |
|-------|---------|----------------|------|
| 2A | <80% on periodic | "What position are you at?" | Position awareness |
| 2B | <75% on variable_step | "Look at the differences" | Delta thinking |
| 3A | <70% on long_range | "Find the unique element" | Full sequence scan |
| 3B | <70% on fibonacci | "Each = sum of previous two" | Recursive shift |
| 4 | <70% on ambiguous | "Multiple answers valid" | Uncertainty OK |

---

## Implementation Notes

### Dataset Generation

```python
PHASE_CONFIG = {
    '1': {
        'patterns': ['repeating', 'alternating', 'incrementing',
                    'fixed_offset', 'periodic_repeat'],
        'seq_len': (4, 6),
        'success_threshold': 0.95
    },
    '2a': {
        'patterns': ['repeating', 'alternating', 'periodic_3', 'periodic_4'],
        'seq_len': (6, 8),
        'success_threshold': 0.90
    },
    '2b': {
        'patterns': ['incrementing', 'fixed_offset', 'decrementing', 'variable_step'],
        'seq_len': (8, 10),
        'success_threshold': 0.85
    },
    '3a': {
        'patterns': ['compositional', 'long_range', 'interleaved'],
        'seq_len': (10, 12),
        'success_threshold': 0.80
    },
    '3b': {
        'patterns': ['fibonacci_like', 'difference_based'],
        'seq_len': (8, 10),
        'success_threshold': 0.75
    },
    '4': {
        'patterns': ['context_dependent', 'mirror', 'ambiguous'],
        'seq_len': (12, 16),
        'success_threshold': 0.70
    }
}
```

### Progression Logic

1. Train on current phase until success_threshold met for all patterns
2. Freeze learned representations
3. Advance to next phase
4. If stuck, teacher intervenes (proposals, hints)
5. Track calibration throughout - accuracy should match confidence

---

## Research References

- Piaget: Cognitive development stages
- Vygotsky: Zone of Proximal Development, scaffolding
- bAbI Tasks: Multi-step reasoning prerequisites
- Curriculum Learning: Easy-to-hard progression benefits
- Pattern Learning in Children: Repeating before growing patterns

---

## Version History

- v1.0 (2024-12-08): Initial curriculum design based on Explorer research
