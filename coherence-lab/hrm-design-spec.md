# HRM Architecture Design: Shape Sorter Model

*Thinking through a proper architecture based on early childhood learning principles*

## What We Learned From Failures

### Position Copy Task: 21% accuracy
- Frozen encoder couldn't adapt
- Single-shot prediction (no iteration)
- Classification-style, not action-consequence

### Phase 2: 44% accuracy
- Position classification ≠ position retrieval
- No recursion to refine answers
- Binary loss (right/wrong), no gradient visibility

## Core Insight: Shape Sorter Learning

A child with a shape sorter:
1. **Perceive**: Look at the shape
2. **Plan**: Pick a hole to try
3. **Act**: Reach and attempt insertion
4. **Observe**: Did it fit? (immediate feedback)
5. **Recurse**: If no, try another hole

Key properties:
- **Action-consequence**: The trying IS the learning
- **Self-correcting**: Physical feedback, not punishment
- **Iteration**: Multiple attempts allowed
- **Safe exploration**: Wrong tries aren't punished

## Proposed Architecture: Minimal HRM

### Two Modules

```
┌─────────────────────────────────────────┐
│           HIGH-LEVEL (Planner)          │
│  "What should I do next?"               │
│  Input: task + state                    │
│  Output: action to take                 │
└──────────────────┬──────────────────────┘
                   │ action
                   ▼
┌─────────────────────────────────────────┐
│           LOW-LEVEL (Executor)          │
│  "How do I do this action?"             │
│  Input: action + sequence               │
│  Output: result of action               │
└──────────────────┬──────────────────────┘
                   │ result
                   ▼
            [Update State]
                   │
                   ▼
            [Loop or Output]
```

### Actions (The "Reaching")

Instead of implicit attention, make actions explicit:

| Action | Input | Output |
|--------|-------|--------|
| LOOK(pos) | position index | token at that position |
| COMPARE(pos1, pos2) | two positions | same/different |
| COPY(pos) | position index | token (for output) |
| PREDICT_NEXT | current state | next token guess |

### Think Steps (Recursion)

For alternating pattern [A, B, A, B, ?]:
```
Step 1: LOOK(0) → A         "What's at start?"
Step 2: LOOK(2) → A         "What's at position 2?"
Step 3: COMPARE(0, 2) → same "Are they the same?"
Step 4: LOOK(4) → ?         "I need position 4"
Step 5: PREDICT_NEXT → A    "Same as 0 and 2"
```

The model SHOWS ITS WORK through the action sequence.

### State / Working Memory

```python
state = {
    'observations': [],   # [(pos, token), ...]
    'comparisons': [],    # [(pos1, pos2, result), ...]
    'hypothesis': None,   # Current best guess
    'confidence': 0.0     # How sure
}
```

State persists across think steps, allowing multi-step reasoning.

## Reward Structure (Show Your Work)

### Old way (binary):
```
final_answer == target → reward = 1
final_answer != target → reward = 0 (or punishment)
```

### New way (partial credit):
```
For each action in sequence:
  - LOOK at relevant position → small reward
  - COMPARE relevant positions → small reward
  - COPY from correct position → medium reward
  - Final answer correct → large reward

Total = sum of step rewards (gradient visible)
```

Wrong final answer with good process → partial credit
Right final answer with lucky guess → less credit than good process

### Informative Feedback

When wrong, the gradient shows WHERE:
```
Step 1: LOOK(0) → A         ✓ relevant
Step 2: LOOK(1) → B         ✓ relevant
Step 3: LOOK(5) → ?         ✗ position 5 doesn't exist
Step 4: PREDICT → wrong     (followed from step 3 error)

Feedback: "Step 3 went out of bounds"
```

## Implementation Approach

### Phase 0: Action Primitives
Train executor to do individual actions correctly:
- LOOK(pos) → retrieve token (the skill we're missing!)
- COMPARE → same/different judgment
- Works like shape sorter: action → immediate consequence

### Phase 1: Simple Plans
Fixed action sequences:
- "For alternating, do: LOOK(0), LOOK(2), COMPARE, PREDICT"
- Planner not learning yet, just executor following scripts

### Phase 2: Learned Planning
Planner learns which actions to take:
- Given task type, generate action sequence
- Executor still follows, but plan is learned

### Phase 3: End-to-End
Full system learns together:
- Planner proposes, executor executes
- Gradient flows through both
- Recursion until confident or max steps

## Key Differences From Current Approach

| Aspect | Current | HRM |
|--------|---------|-----|
| Structure | Flat transformer | Planner + Executor |
| Forward pass | Single shot | Multiple think steps |
| Actions | Implicit (attention) | Explicit (LOOK, COMPARE, etc.) |
| Feedback | Binary loss | Step-by-step gradient |
| Learning | Classification | Action-consequence |
| Freezing | Early layers frozen | Fluid, all learning |

## Questions to Resolve

1. **How many think steps?** Start with 3-5, let model learn to terminate early?

2. **Action space size?** Start minimal (LOOK, COMPARE, OUTPUT), expand as needed?

3. **State representation?** Simple list of observations, or more structured?

4. **Training curriculum?**
   - Phase 0: Just LOOK action (the primitive we're missing)
   - Build up complexity

5. **Architecture size?** Research suggests HRM can work with 1K samples. Keep small?

## Next Steps

1. Implement LOOK action as standalone task (unfrozen)
2. Verify model can learn position → token mapping
3. Add state/memory for multi-step
4. Add planner module
5. Full integration

---

*This is thinking-out-loud, not final design. Feedback welcome.*
