# Integrated Phase 1: Unified Foundation

*Redesign based on Dec 7 learnings*

## Problem Statement

Phase 2 achieved 96% on incrementing patterns but only 31-48% on position-based patterns (alternating, repeating). The gap: **position awareness**.

Current Phase 1 teaches token properties only. Position embeddings exist but aren't explicitly trained to encode position relationships.

## Design Principle

Teach **four types of reasoning** in one integrated phase:

| Type | Question | Example |
|------|----------|---------|
| Token Properties | What IS this token? | Token 4 → vowel, even, low |
| Token Relations | Are these tokens ALIKE? | (4, 8) → same (both vowel) |
| Position Properties | What IS this position? | Position 3 → odd, early |
| Position Relations | Are these positions ALIKE? | (0, 2) → same (both even) |

All four must be learned before sequence prediction.

## Architecture: Multi-Task Transformer

Instead of separate phases, one model with multiple output heads:

```
Input: [token_ids], [positions]
       ↓
┌─────────────────────────────────────┐
│    Token Embeddings (learnable)     │
│    Position Embeddings (learnable)  │
└─────────────────────────────────────┘
       ↓
┌─────────────────────────────────────┐
│    Transformer Blocks (2-4 layers)  │
└─────────────────────────────────────┘
       ↓
   ┌───┴───┐
   ↓       ↓
[Token    [Position
 Heads]    Heads]
```

### Output Heads

1. **Token Property Head** (per token)
   - 6-way multi-label: vowel, consonant, even, odd, high, low
   - Same as current Phase 1

2. **Token Relation Head** (per pair)
   - Binary: same category / different
   - Pooled from two token positions

3. **Position Property Head** (per position)
   - 4-way multi-label: even_pos, odd_pos, early (< 10), late (>= 10)
   - Uses position embedding directly

4. **Position Relation Head** (per pair)
   - Binary: same parity / different
   - Pooled from two position embeddings

## Training Strategy

**Multi-task learning**: All four tasks trained simultaneously.

Loss = α₁·L_token_prop + α₂·L_token_rel + α₃·L_pos_prop + α₄·L_pos_rel

Start with equal weights, adjust if one task dominates.

## Data Generation

Each training example includes multiple supervision signals:

```python
{
    "sequence": [4, 7, 4, 7, 4],
    "tasks": {
        "token_properties": [
            {"pos": 0, "token": 4, "labels": [1,0,1,0,0,1]},  # vowel, even, low
            {"pos": 1, "token": 7, "labels": [0,1,0,1,0,1]},  # consonant, odd, low
            ...
        ],
        "token_relations": [
            {"pos1": 0, "pos2": 2, "same": 1},  # both have token 4
            {"pos1": 0, "pos2": 1, "same": 0},  # token 4 vs 7
            ...
        ],
        "position_properties": [
            {"pos": 0, "labels": [1,0,1,0]},  # even_pos, early
            {"pos": 1, "labels": [0,1,1,0]},  # odd_pos, early
            ...
        ],
        "position_relations": [
            {"pos1": 0, "pos2": 2, "same": 1},  # both even positions
            {"pos1": 0, "pos2": 1, "same": 0},  # even vs odd position
            ...
        ]
    }
}
```

## Architecture Parameters

| Component | Current | Proposed |
|-----------|---------|----------|
| Token embed dim | 128 | 128 |
| Position embed dim | 128 | 128 |
| Transformer layers | 2 | 3-4 |
| Attention heads | 1 | 2-4 |
| FFN dim | 256 | 384 |
| Total params | ~270K | ~500K-700K |

Slight capacity increase to handle 4 tasks vs 1.

## Success Criteria

| Task | Target |
|------|--------|
| Token properties | 98%+ |
| Token relations | 95%+ |
| Position properties | 98%+ |
| Position relations | 95%+ |

All four must hit targets before freezing and moving to Phase 2.

## Why This Should Work

Phase 2 alternating patterns require:
1. Recognize position 0, 2, 4 are "same type" (even) ← **Position Relations**
2. Know token at even positions vs odd positions ← **Position Properties**
3. Predict next based on position parity ← Built on (1) and (2)

With explicit position awareness training, the model has all building blocks before seeing sequences.

## Implementation Plan

1. `phase1_integrated_data.py` - Multi-task data generator
2. `phase1_integrated_model.py` - Multi-head transformer
3. `phase1_integrated_train.py` - Multi-task training loop
4. Generate 100K examples
5. Train to convergence on all 4 tasks
6. Freeze and build Phase 2 on top

## Comparison to Current Approach

| Aspect | Current (Phases 1, 1b) | Integrated |
|--------|------------------------|------------|
| Phases | Separate | Single |
| Patching | Yes | No |
| Position awareness | Implicit | Explicit |
| Transfer to Phase 2 | Partial | Full |
| Architecture | Different per phase | Unified |

---

**Status:** Spec complete. Ready for implementation.
