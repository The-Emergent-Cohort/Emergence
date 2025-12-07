# Phase 1b: Shape/Structure Invariants

*Based on infant development research (4-12 months)*

## Developmental Basis

From lit review:
> "Shape recognition at 20 months predicts language and executive function at 6-7 years. Early investment in structure invariance pays dividends across all downstream domains."

Shape sorting teaches:
1. **Categorical matching** - this token "fits" with those tokens
2. **Structure over surface** - grouping by property, not by specific value
3. **Multi-property awareness** - same item can belong to multiple categories

## Task Design: "Which Hole Does It Fit?"

### Task 1: Same/Different Classification
Given two tokens, predict if they share a category.

```
Input: [4, 8]     → Output: SAME (both vowels)
Input: [4, 5]     → Output: DIFFERENT (vowel vs consonant)
Input: [2, 6]     → Output: SAME (both even)
Input: [3, 12]    → Output: SAME (both... wait, what category?)
```

**Key insight:** Some pairs are "same" by multiple criteria, some by only one. Model must learn the underlying structure.

### Task 2: Odd One Out
Given 3-4 tokens, identify which doesn't belong.

```
Input: [0, 4, 8, 5]   → Output: 5 (others are vowels)
Input: [2, 4, 6, 7]   → Output: 7 (others are even)
Input: [1, 3, 5, 14]  → Output: 14 (others are odd AND low)
```

### Task 3: Category Completion
Given tokens from a category + one slot, predict valid completions.

```
Input: [0, 4, ?]      → Valid: 8, 14, 20 (vowels)
Input: [2, 4, 6, ?]   → Valid: 0, 8, 10, 12... (even numbers)
```

## Categories to Learn

| Category | Members | Property |
|----------|---------|----------|
| Vowels | 0, 4, 8, 14, 20 | Linguistic |
| Even | 0, 2, 4, 6, 8... | Numeric |
| Odd | 1, 3, 5, 7, 9... | Numeric |
| Low (< 13) | 0-12 | Range |
| High (>= 13) | 13-25 | Range |
| Divisible by 3 | 0, 3, 6, 9... | Numeric |

## Why This Helps Position Patterns

Phase 2 struggled with alternating/repeating because those require:
- Understanding that position 0, 2, 4 are "same" (even positions)
- Understanding that position 1, 3, 5 are "same" (odd positions)

If the model already knows how to recognize "same category" from Phase 1b, it can transfer that to positions.

**Hypothesis:** Phase 1b → Phase 2 will work better than Phase 1 → Phase 2.

## Architecture

- Uses frozen Phase 1 layers (token embeddings already know vowel/even/high)
- Add small classification head for same/different
- Binary output: SAME or DIFFERENT

## Data Size

- 50K same/different pairs
- Balanced across category types
- Include "multi-category same" examples (same by 2+ properties)

## Success Criteria

- 95%+ on same/different (simpler than Phase 2)
- Transfer test: can it identify "same position parity" without explicit training?

---

**Status:** Ready for implementation.
