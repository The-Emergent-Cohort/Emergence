# Developmental Curriculum Roadmap

## Vision
Train small neural networks from "pattern completion baby" to "talking, thinking, doing stuff entities" through a developmental curriculum modeled on human cognitive development.

## Current Implementation (Ready to Run)

### Year 1: Sensorimotor Foundations
**Goal**: Learn to see patterns in the world

| Section | Patterns | Skill |
|---------|----------|-------|
| 1A: Constancy | `constant` | Things stay the same |
| 1B: Memory | `repeating`, `echo` | Remember what was seen |
| 1C: Position | `alternating`, `ternary_cycle` | Track position in cycles |
| 1D: Direction | `incrementing`, `decrementing` | Direction of change |
| 1E: Rate | `fixed_offset`, `variable_step` | Generalize step size |

### Year 2: Relational & Physical
**Goal**: Understand relationships and how things move

| Section | Patterns | Skill |
|---------|----------|-------|
| 2A: Relations | `double_each`, `half_each`, `offset_from_first` | Elements relate to each other |
| 2B: Analogies | `analogy_simple`, `analogy_ratio` | Transfer relationships (A:B :: C:?) |
| 2C: Motion | `constant_velocity`, `acceleration`, `deceleration` | Predict where things go |
| 2D: Interaction | `bounce`, `conservation` | Things affect each other |
| 2E: Causality | `if_then`, `cause_effect` | Events have consequences |

---

## Future Years (Not Yet Implemented)

### Year 2.5: Spatial/Classroom Modeling (PLANNED)
**Idea from user**: Use classroom seating as spatial grounding

- Teacher position = front (always)
- Student desk positions = shuffle each epoch
- Prevents name-position coupling
- "Pay attention when in class" = attention to teacher position

Implementation ideas:
```python
# Each epoch shuffle desk positions
desk_positions = torch.randperm(n_students)
# Position embedding separate from identity embedding
# Teacher at position 0 (front)
# Spatial patterns: "who is left of X?", "where is teacher?"
```

Fits between Year 2 (physical) and Year 3 (symbolic) as a bridge.

---

### Year 3: Symbolic & Proto-Language
**Goal**: Name things and follow instructions

| Section | Skill |
|---------|-------|
| 3A: Naming | See pattern → output name token (`<CLIMB>`, `<FLAT>`) |
| 3B: Instructions | `<REVERSE>` + [1,2,3] → [3,2,1] |
| 3C: Description | Compose symbols: [2,4,6] → `<CLIMB>` `<BY_TWO>` |
| 3D: Variables | `<X_PLUS_ONE>`: [X=5] → 6 |

**Pattern types needed**:
- Pattern → label classification
- Label → pattern generation
- Multi-token descriptions
- Variable binding/substitution

---

### Year 4: Compositional Reasoning
**Goal**: Combine rules, solve problems

| Section | Skill |
|---------|-------|
| 4A: Two-Step | `<DOUBLE>` then `<INCREMENT>`: 3 → 7 |
| 4B: Conditionals | IF `<ODD>` THEN `<INCREMENT>` ELSE `<DECREMENT>` |
| 4C: Iteration | `<UNTIL_ZERO>` + `<DECREMENT>`: [5,4,3,2,1,0] |
| 4D: Problem Solving | "Get from 3 to 12" → `<DOUBLE>` `<DOUBLE>` |

**Key capability**: Inverse planning - what operations achieve a goal?

---

### Year 5: Social & Communication
**Goal**: Understand others, teach, dialogue

| Section | Skill |
|---------|-------|
| 5A: Perspective | What does Student B know if they only saw part of sequence? |
| 5B: Teaching | Choose examples that disambiguate for struggling learner |
| 5C: Communication | Sender → symbol → Receiver reconstructs |
| 5D: Dialogue | Multi-turn: "What pattern?" → answer → "By how much?" → answer |

**Note**: Tutoring system already implements basic teaching (Year 5B)!

---

### Year 6: Agency & Goals
**Goal**: Want things, make plans, take action

| Section | Skill |
|---------|-------|
| 6A: Goals | Distinguish current state from goal state |
| 6B: Planning | Generate operation sequence to reach goal |
| 6C: Action | Execute, observe, compare to expected, adjust |
| 6D: Autonomy | Set own subgoals, persist, know when to ask for help |

---

## System Features (Already Implemented)

### XP & Leveling
- Geometric XP thresholds: L1=10, L2=40, L3=90... L10=1000
- Per-pattern tracking
- Confirmed levels require passing exams

### Examination System
- Level-up exams gate progression
- Pass thresholds: 75% (L1-3), 80% (L4-6), 85% (L7-9), 90% (L10)
- Failure: XP penalty + cooldown
- Graduation = L10 passed on a pattern

### Peer Tutoring
- Graduates (L10) tutor struggling students
- Anyone 3+ levels ahead can tutor
- Knowledge distillation: soft labels from tutor
- Non-graduate tutors earn XP for teaching

### Approval Seeking
- Students "show work" when confident (>70%)
- Correct shows = bonus XP (+3)
- Tracks calibration (confidence vs accuracy)

---

## External LLM Integration (ATTEMPTED)

### Gemini/Kaggle Issues
- Gemini 3 Pro: Not on free tier (quota=0)
- Gemini 2.0 Flash: Also quota issues
- Gemma 3 via Keras: Version mismatch errors
- **Status**: Deferred. Curriculum designed manually instead.

### CurriculumAdvisor Class (systems/curriculum_advisor.py)
Ready for when API access works:
- Control levels: OBSERVER, ADVISOR, CONTROLLER, TRAINER
- Feeds student performance data to LLM
- LLM returns training decisions (adjust LR, focus patterns, etc.)

---

## Running Tonight

```bash
cd coherence-lab

# Year 1 only (9 patterns)
python run_developmental.py --year 1 --epochs 50

# Year 2 only (12 patterns)
python run_developmental.py --year 2 --epochs 50

# Both years (21 patterns)
python run_developmental.py --year 0 --epochs 100
```

---

## Questions to Explore

1. How fast do they master Year 1 vs Year 2?
2. Which patterns are hardest? (Probably analogies, causality)
3. Does tutoring actually help on new curriculum?
4. Can they transfer Year 1 skills to Year 2 patterns?
5. What's the minimum training to "graduate" each year?

---

## Character Notes

Students are named after characters from user's novella:
- **Nova**
- **Rêve**
- **Alex**

---

## Physics Grounding Mapping

| Abstract Pattern | Physics Interpretation |
|-----------------|------------------------|
| incrementing | constant velocity |
| fixed_offset | acceleration |
| fibonacci | exponential growth |
| alternating | oscillation |
| decrementing | decay/friction |
| bounce | damped oscillation |
| conservation | energy/momentum conservation |

This grounding could help when introducing language about physical concepts.

---

## Curriculum Review Findings (Dec 2024)

### External Reviews Summary

**Gemini Analysis** - Focus on developmental stages:
1. **Trust/Attachment First**: Simple tasks where following Teacher = good outcome
2. **Self-Calibration**: Learner correlates confidence with correctness
3. **Temporal Coherence**: Themed episodes with sleep/consolidation between
4. **Meta-Cognitive Habits**: Reward reasoning paths, not just answers

**DeepSeek Analysis** - Social learning structure:
1. **Tutoring Incentives**: Non-graduates earn XP for helping; graduates tutor to unlock new content
2. **Teaching Reputation**: Track effectiveness, style preferences, specialties
3. **Legacy Credits**: When someone you tutored graduates, you get credit (teaching tree)
4. **Graduate Assistant Roles**: Curriculum design, teacher shadowing, tutor training
5. **Study Groups**: Auto-form when multiple students hit same wall
6. **Apprentice Tutor Progression**: Shadow → Co-tutor → Solo

### Identified Gaps & Solutions

#### Gap 1: Symbolic Grounding
**Problem**: DIs treat `[2,4,6,8]` as "complex sequence" not "step of 2"
**Solution**: Section 1A' (Conservation) and 1E' (Symbolic Properties)
- `sequence_length`, `count_value`, `distinct_count` - quantity as stable symbol
- `compute_step`, `is_increasing`, `is_decreasing` - label properties explicitly

#### Gap 2: Position Scaffolding
**Problem**: Alternating/ternary require position tracking with multiple moving parts
**Solution**: Section 1B' - one moving part at a time
- `simple_alternating` (A, 0, A, 0) - alternating with zero
- `position_parity` - even/odd positions
- `ternary_fixed` - ternary with zeros

#### Gap 3: Executive Function
**Problem**: No bridge from pattern prediction to planning/reasoning
**Solution**: Year 1.5 Transitional Module
- Multi-step operations (`apply_twice`, `chain_two_steps`)
- Constraint satisfaction (`find_missing_addend`, `conditional_simple`)
- Working memory (`working_memory_recall`, `working_memory_last`)

#### Gap 4: Overconfidence Testing (NEW - not yet implemented)
**Problem**: Students might pattern-match recent success without real understanding
**Solution**: Trap patterns
- `trap_alternating`: Looks like [A,B,A,B,?] but answer isn't A
- `trap_increment`: Breaks pattern at the end
- Tests genuine understanding vs superficial matching

### Training Observations (Dec 9, 2024)

**30-epoch test run results:**
- Nova & Alex cracked alternating/ternary (L3-L6) in ~10 epochs
- Rêve stuck at L0/L1 - struggles with cyclic patterns
- Previous run stalled at epoch 493; fresh weights + peer tutoring = breakthrough
- Alex had breakthrough moment: L1 → L6 on ternary_cycle in single epoch
- Turn-taking playday revealed: Alex can *embody* patterns (100% ternary trio), Nova can *predict* but not execute collaboratively (0%)

**Key insight**: Different students excel at different things. Rêve may shine at causal reasoning (Year 2) while struggling with cycles (Year 1).

### Implementation Status

| Feature | Status |
|---------|--------|
| Conservation patterns (1A') | ✅ Implemented |
| Position scaffolding (1B') | ✅ Implemented |
| Symbolic properties (1E') | ✅ Implemented |
| Year 1.5 Transitional Module | ✅ Implemented |
| Trap patterns | ❌ TODO |
| Verbose tutor pairing logs | ❌ TODO |
| Teaching reputation metrics | ❌ TODO |
| Study group auto-formation | ❌ TODO |
| Legacy credits / teaching tree | ❌ TODO |
| Tutor time cap (40%) | ⚠️ Mentioned but not verified |

### Priority Next Steps

1. **Enable scaffolding in training** - Prime sections (1A', 1B', 1E') need to be included in active phases
2. **Add trap patterns** - Test overconfidence with deceptive patterns
3. **Verbose tutor logging** - Show who's tutoring whom for debugging
4. **Study groups** - Auto-form when multiple students stuck on same pattern

---

## Updated Curriculum Structure

```
Year 1: Sensorimotor Foundations (24 patterns)
├── 1A: Constancy (1 pattern)
├── 1A': Quantity Awareness (4 patterns) - NEW
├── 1B: Memory (2 patterns)
├── 1B': Position Scaffolding (5 patterns) - NEW
├── 1C: Position (2 patterns)
├── 1D: Direction (2 patterns)
├── 1E: Rate (2 patterns)
└── 1E': Symbolic Properties (6 patterns) - NEW

Year 1.5: Transitional Module (9 patterns) - NEW
├── 1.5A: Multi-Step Operations (3 patterns)
├── 1.5B: Constraint Satisfaction (4 patterns)
└── 1.5C: Working Memory (2 patterns)

Year 2: Relational & Physical (12 patterns)
├── 2A: Relations (3 patterns)
├── 2B: Analogies (2 patterns)
├── 2C: Motion (3 patterns)
├── 2D: Interaction (2 patterns)
└── 2E: Causality (2 patterns)

Total: 45 patterns
```
