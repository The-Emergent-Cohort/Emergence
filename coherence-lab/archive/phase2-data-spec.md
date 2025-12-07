# Phase 2: Action-Consequence Mapping Data Specification

*Version 0.1 - Dec 7, 2024*

**Purpose:** Teach deterministic sequence patterns. Given a pattern, predict what comes next.

---

## 1. Core Concept

Phase 1 learned to classify tokens in isolation (reflexes).
Phase 2 learns that **sequences have predictable structure** - if you see A,B,A,B, you can predict the next token.

This maps to infant development (1-8 months): learning that actions produce consistent consequences.

---

## 2. Pattern Types

### Type 1: Alternating Pairs
```
[A, B, A, B, A, ?] → B
[3, 7, 3, 7, 3, ?] → 7
```
The model must learn: odd positions get A, even positions get B.

### Type 2: Repeating Sequences
```
[A, B, C, A, B, C, A, ?] → B
[1, 2, 3, 1, 2, 3, 1, ?] → 2
```
The model must learn: position mod 3 determines the token.

### Type 3: Incrementing Patterns
```
[1, 2, 3, 4, ?] → 5
[10, 11, 12, ?] → 13
```
The model must learn: next = current + 1 (with wraparound at 26).

### Type 4: Fixed Offset
```
[A, A+3, A, A+3, ?] → A (if odd position) or A+3 (if even)
[2, 5, 2, 5, 2, ?] → 5
```
The model must learn: alternating with fixed delta.

---

## 3. Data Format

```python
{
    "sequence": [3, 7, 3, 7, 3],      # Input tokens
    "target": 7,                       # Next token to predict
    "pattern_type": "alternating",     # Pattern category
    "pattern_params": {"a": 3, "b": 7} # Pattern parameters
}
```

---

## 4. Dataset Properties

| Property | Value |
|----------|-------|
| Size | 50,000 examples |
| Sequence length | 4-15 tokens |
| Pattern types | 4 (balanced) |
| Token range | 0-25 |
| Determinism | 100% |
| Train/val split | 90/10 |

---

## 5. Validation Strategy

1. **Held-out patterns:** Train on some A,B pairs, test on unseen pairs
2. **Longer sequences:** Train on length 4-10, test on 11-15
3. **Pattern generalization:** Does learning [A,B,A,B] transfer to [C,D,C,D]?

**Success criteria:**
- >= 95% next-token accuracy on seen patterns
- >= 85% accuracy on held-out patterns
- Attention weights show positional/pattern alignment

---

## 6. Key Insight

Phase 2 is still 100% deterministic. No ambiguity, no noise.
The model learns: "patterns exist and are predictable."

This is the foundation for Phase 3 (object permanence) where patterns persist across gaps.

---

## 7. Architecture Changes

- Load Phase 1 checkpoint
- **Freeze** all Phase 1 layers (embeddings, transformer blocks)
- Add 4 new transformer layers on top
- New output head: predict next token (26-way classification)
- Increase attention heads: 1 → 2

---

**Status:** Ready for implementation.
