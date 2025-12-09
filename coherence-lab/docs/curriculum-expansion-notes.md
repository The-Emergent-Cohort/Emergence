# Curriculum Expansion Notes

*Session: Dec 9, 2025 - Exploring datasets and curriculum gaps*

## Current Curriculum Coverage

Our 13 synthetic patterns focus heavily on **induction** (learning rules from examples):
- identity, reverse, shift_right, shift_left (positional transforms)
- periodic_repeat, interleave, alternating (structural combination)
- indexed_lookup (reference/indirection)
- arithmetic patterns

**Core strength:** Domain-agnostic relational abstraction. The model learns *relationships*, not content.

## Why Relational Patterns First?

1. **Compositionality** - reverse + shift can compose without relearning
2. **Clean ground truth** - unambiguous right/wrong for metacognition development
3. **The self-model is the real prize** - patterns are scaffolding for confidence calibration and help-seeking behavior
4. **Transfer potential** - structural transforms work across any domain

## Gaps Identified (via bAbI comparison)

bAbI has 20 reasoning tasks. We cover ~2 well (induction, positional). Missing:

| Skill | Priority | Notes |
|-------|----------|-------|
| **Negation** | High | "NOT this pattern" - fundamental for reasoning |
| **Deduction chains** | High | A→B, B→C ∴ A→C - compositional reasoning |
| **Counting/cardinality** | Medium | "N of X" - could add as pattern type |
| **Coreference** | Medium | Tracking references through transforms |
| **Fact chaining** | Lower | 1-2-3 hop reasoning |
| **Time reasoning** | Lower | Before/after relationships |
| **Path finding** | Lower | Graph traversal (hardest bAbI task) |

## Promising External Datasets

### Closest matches to our approach:
- **ARC** (450KB) - Grid-based pattern completion, exactly our task structure but visual
- **bAbI** (17MB) - 20 reasoning primitives, 1K train/test each
- **OEIS** (69MB) - 390K+ real mathematical sequences
- **Integer Sequences for Representation Learning** (36MB) - self-supervised number sequences

### For later stages:
- **CLEVR** (19GB) - Visual reasoning, object relationships
- **GSM8K** (3.4MB) - Multi-step math word problems
- **Word Analogy Test** (183KB) - A:B::C:? relational mapping

## Potential Synthetic Pattern Additions

Could implement without external data:

```
# Negation patterns
"not_identity" - output anything BUT the input
"exclude" - remove elements matching criteria

# Deduction patterns
"transitive" - if A→B and B→C shown, infer A→C
"syllogism" - categorical reasoning

# Counting patterns
"count_and_repeat" - repeat element N times where N derived from input
"cardinality" - output the count of something

# Conditional patterns
"if_then_else" - conditional transformation based on input property
```

## Design Principle

Goal is **general learning ability**, not math. The relational patterns are chosen because:
- They test abstract rule extraction
- They're composable
- They support metacognition development
- They transfer across domains

The specific patterns matter less than the *skill of learning patterns*.

## Next Steps (when ready)

1. Consider adding negation and deduction patterns to curriculum
2. Look at ARC tasks for inspiration on novel pattern types
3. Possibly adapt bAbI format for sequence completion (vs Q&A)
4. Think about multi-step / compositional patterns

## Kaggle Resources

API works with: `export KAGGLE_API_TOKEN=<token>`
- 30hr/week free GPU (P100 16GB or 2xT4)
- Can push notebooks for remote training
- Dataset search: `kaggle datasets list --search "term" --sort-by votes`
