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
