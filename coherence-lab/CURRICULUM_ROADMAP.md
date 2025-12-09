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

### Identified Gaps

**1. Position Scaffolding Gap (1B → 1C)**
- Jump from identity/memory (1B) to position tracking (1C) is too abrupt
- Students can't "see the hop" - position is movement-based without spatial relation
- **Solution**: Added section 1B' with scaffolded position patterns:
  - `simple_alternating`: [A, 0, A, 0, ?] - alternating with zero
  - `position_parity`: even/odd positions get different values
  - `ternary_fixed`: [A, 0, 0, A, 0, ?] - ternary with zeros
  - `fill_A_positions`, `fill_B_positions`: recognize where values appear

**2. Symbolic Grounding Gap (Phase 1 → Phase 2)**
- DIs learn to predict sequences but don't learn quantity as stable symbol
- When seeing `double_each`, they interpret as "complex sequence" not "x→2x"
- **Missing**: Conservation of quantity, cardinality, symbolic labeling
- **Solution Needed**: Add patterns that teach:
  - Same count, different arrangement → still same count
  - Explicit property labeling (not just next-token prediction)

**3. Executive Function Gap (No Transitional Module)**
- Math requires structured planning, not linear prediction
- Working memory for multi-step constraint satisfaction
- **Missing**: Tower of Hanoi-style planning, DAG reasoning
- **Solution Needed**: Transitional Module between Year 1 and Year 2

### Social Learning Findings

**Graduated Peer Introduction (not implemented yet)**
- Stage 0: Solitary (foundational skills)
- Stage 1: Observational (see anonymized peer solutions)
- Stage 2: Mirroring (imitate peer approaches)
- Stage 3: Structured exchange (turn-taking, tutoring)
- Stage 4: Collaborative (joint problem-solving)
- Stage 5: Competitive (adversarial examples)

**Social Readiness Signals**
- Calibration score (confidence vs accuracy) > 0.7
- Teacher trust score > 0.8
- Has mastered at least 3 foundational patterns

**Playday Enhancements (implemented)**
- Targeted struggle challenges for specific patterns
- Breakthrough tracking ("aha" moments)
- Play style insights (collaborator, explorer, helper)
- No exams on playday - pure exploration

### Proposer-Critic Architecture (from Gemini report)
Maps to existing peer tutoring:
- **Proposer**: Student generates solution with explicit steps
- **Critic**: Peer verifies logical validity of reasoning path
- **LogicReward**: Score path validity, not just final answer
- Creates autonomous self-correction loop

### Priority Implementation Order

1. **Conservation patterns** (Year 1, section 1A') ✅ IMPLEMENTED
   - `sequence_length`: Count elements in sequence
   - `count_value`: Count occurrences of specific value
   - `distinct_count`: Count unique values
   - `conservation_shuffle`: Same count despite reordering

2. **Symbolic property patterns** (Year 1, section 1E') ✅ IMPLEMENTED
   - `compute_step`: Identify additive step size
   - `compute_first_diff`: Calculate first difference
   - `compute_ratio`: Identify multiplicative ratio
   - `is_constant`, `is_increasing`, `is_decreasing`: Classification

3. **Transitional Module** (Year 1.5) ✅ IMPLEMENTED
   - **1.5A Multi-Step Operations:**
     - `apply_twice`: Apply operation twice
     - `reverse_operation`: Inverse operations
     - `chain_two_steps`: Chain two operations
   - **1.5B Constraint Satisfaction:**
     - `find_missing_addend`: What's in the gap?
     - `conditional_simple`: Simple if-then
     - `min_of_two`, `max_of_two`: Find extrema
   - **1.5C Working Memory:**
     - `working_memory_recall`: Recall first element
     - `working_memory_last`: Recall last element

4. **LogicReward integration** (Teacher engine) - PLANNED
   - Score reasoning paths, not just answers
   - Peer validation of logical steps

---
