# Integration Architecture

*How curriculum, teacher, and generators work together*

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    CURRICULUM (JSON)                             │
│  Teaching content: explanations, variants, CPA stages           │
│  "What to teach and how to explain it"                          │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TEACHER (Claude API)                          │
│  Reads curriculum, selects variants, presents to student        │
│  "The guide who interprets and delivers"                        │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GENERATORS (Pattern Functions)                │
│  Create practice problems: [1, 2, 3, ?] → 4                     │
│  "The problem bank"                                             │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STUDENT (RelationalSystem)                    │
│  Processes numeric sequences, learns patterns                   │
│  "The developing mind"                                          │
└─────────────────────────────────────────────────────────────────┘
```

## The Three Layers

### Layer 1: Curriculum (Teaching Content)

**What it is:** JSON files following `curriculum_format_spec.md`

**What it contains:**
- Hierarchical structure (Subject → Course → Unit → Lesson → Activity)
- Multiple variants per activity (Claude, GPT, DeepSeek explanations)
- CPA stages (Concrete → Pictorial → Abstract)
- Instructional phases (I Do → We Do → You Do)
- Problem generator references

**Key insight:** The curriculum is for the TEACHER, not the student. The student model doesn't read JSON or text explanations.

### Layer 2: Teacher (Claude API)

**What it does:**
1. Reads current activity from curriculum
2. Selects a variant based on student state
3. Uses the explanation to guide teaching
4. Calls the problem generator for practice problems
5. Evaluates student responses
6. Logs what worked for future optimization

**Two modes:**
- **Direct Teaching:** Teacher generates explanations on the fly
- **Variant Coordination:** Teacher selects from pre-compiled variants

### Layer 3: Generators (Problem Bank)

**What they are:** Functions in `developmental_curriculum.py`

**Examples:**
- `gen_counting(vocab_size)` → `[0, 1, 2, 3, ?]` → 4
- `gen_add_one(vocab_size)` → `[3, 1, 4, 0, 5, 1, ?]` → 6
- `gen_alternating(vocab_size)` → `[A, B, A, B, ?]` → A

**Key insight:** Generators are NOT the curriculum. They're subordinate to it - they create practice problems at the level specified by the curriculum activity.

## Data Flow: Teaching Session

```
1. CURRICULUM LOADER
   curriculum.json → CurriculumLoader
   Returns: Current Activity with variants, CPA stage, generator ref

2. TEACHER DECISION
   Activity + Student State → Claude API
   Returns: Selected variant, scaffolding level, approach

3. TEACHER PRESENTS
   Teacher reads variant.explanation
   Teacher demonstrates/explains to student
   (This is logged but doesn't directly train the model)

4. PROBLEM GENERATION
   activity.problem_generator.name → gen_counting()
   activity.problem_generator.params → {max_length: 5}
   Returns: Numeric sequence [1, 2, 3, 4, ?]

5. STUDENT ATTEMPTS
   Sequence → RelationalSystem.forward()
   Returns: Prediction, confidence, emotional state

6. TEACHER EVALUATES
   Prediction vs target → correct/incorrect
   Student state → struggling/progressing

7. LOG & ADJUST
   Log variant effectiveness
   If struggling: teacher selects different variant
   If progressing: advance difficulty or curriculum position
```

## Curriculum Activity Example

```json
{
  "activity_id": "math.counting.unit01.lesson01.act01",
  "activity_type": "demonstration",
  "variants": [
    {
      "variant_id": "claude_en",
      "approach": "procedural",
      "explanation": "Let's count together! One... two... three..."
    },
    {
      "variant_id": "gpt4_en",
      "approach": "narrative",
      "explanation": "Once upon a time, blocks came to visit..."
    }
  ],
  "extensions": {
    "pedagogy": {
      "cpa_stage": "concrete",
      "instructional_phase": "i_do",
      "scaffolding_level": 1.0
    },
    "di_training": {
      "problem_generator": {
        "name": "counting",
        "params": {"max_length": 5}
      }
    }
  }
}
```

## Components Needed

| Component | Purpose | File |
|-----------|---------|------|
| CurriculumLoader | Parse JSON, track position, map to generators | `curriculum_loader.py` |
| TeacherInterface | Claude API calls, variant selection | `teacher_interface.py` |
| StudentStateBridge | Extract emotions, XP, struggle detection | `student_state_bridge.py` |
| IntegratedTrainingLoop | Coordinate all pieces | `integrated_train.py` |

## What Stays the Same

- `developmental_curriculum.py` generators (now as problem bank)
- `relational_model.py` student architecture
- `curriculum_sequencer.py` progression logic
- `train.py` training loop (with optional curriculum_loader)

## Key Files

| Existing | Role in Integration |
|----------|---------------------|
| `research/curriculum_format_spec.md` | Schema definition |
| `examples/mini_curriculum.json` | Working example |
| `research/curriculum_compiler_prompt.md` | LLM compilation guide |
| `developmental_curriculum.py` | Problem generators |
| `.claude/skills/teacher/` | Teacher skill context |

---

*The curriculum teaches through the teacher. The generators provide practice. The student learns from both.*
