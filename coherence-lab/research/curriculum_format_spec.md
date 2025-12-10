# Curriculum Format Specification v1.0

*Coherence Lab - December 10, 2025*

## Overview

This specification defines the JSON format for curriculum data used in DI training. It combines:
- **Textbook hierarchy** (Curriculum → Subject → Course → Unit → Lesson → Activity)
- **Standards alignment** (CASE URIs, xAPI tracking)
- **Pedagogical extensions** (CPA stages, gradual release, scaffolding)

The format is designed to be:
1. Grounded in real educational standards (not AI-invented)
2. Extensible for DI-specific needs
3. Compatible with existing OER content

---

## Schema Hierarchy

```
Curriculum
├── Subject[]
│   ├── Course[]
│   │   ├── Unit[]
│   │   │   ├── Lesson[]
│   │   │   │   └── Activity[]
```

---

## 1. Curriculum (Root)

The top-level container for an entire educational program.

```json
{
  "curriculum_id": "coherence.k12.core.v1",
  "title": "Coherence Lab Core Curriculum",
  "description": "Developmental curriculum for DI training.",
  "version": "1.0",
  "locale": "en-US",
  "education_levels": ["early-childhood", "primary"],
  "subjects": [],
  "metadata": {
    "publisher": "Coherence Lab",
    "last_updated": "2025-12-10",
    "case_framework_ids": [
      "https://example.org/case/frameworks/math-k5",
      "https://example.org/case/frameworks/physics-intro"
    ],
    "source_oer": ["Eureka Math", "TinyStories", "GSM8K"]
  },
  "extensions": {}
}
```

---

## 2. Subject

A discipline or domain of study.

```json
{
  "subject_id": "math",
  "title": "Mathematics",
  "description": "Number sense, operations, and mathematical reasoning.",
  "native_representation": {
    "type": "numeric",
    "formats": ["integer", "fraction", "decimal"],
    "operations": ["add", "subtract", "multiply", "divide"]
  },
  "grade_bands": [
    {"min_grade": 0, "max_grade": 2},
    {"min_grade": 3, "max_grade": 5}
  ],
  "courses": [],
  "extensions": {}
}
```

### Subject-Specific Native Representations

| Subject | Type | Key Fields |
|---------|------|------------|
| Math | `numeric` | formats, operations, range |
| Reading | `linguistic` | vocabulary, syntax_complexity, genre |
| Physics | `simulation` | entities, properties, constraints |
| Music | `temporal` | beats, pitch_range, duration |
| Art | `spatial` | strokes, colors, canvas_size |

---

## 3. Course

A focused learning sequence within a subject.

```json
{
  "course_id": "math.counting.v1",
  "title": "Counting and Cardinality",
  "description": "Foundation counting skills from 1-100.",
  "grade_range": {"min_grade": 0, "max_grade": 1},
  "duration": {"units": "weeks", "value": 12},
  "standards_alignment": [
    {
      "framework": "https://corestandards.org/Math/Content/K/CC",
      "competency_ids": ["K.CC.A.1", "K.CC.A.2", "K.CC.B.4"]
    }
  ],
  "prerequisites": [],
  "units": [],
  "extensions": {}
}
```

---

## 4. Unit

A thematic grouping of lessons.

```json
{
  "unit_id": "math.counting.v1.unit01",
  "title": "Counting to 10",
  "description": "Establishing one-to-one correspondence and counting sequence.",
  "sequence": 1,
  "estimated_duration": {"units": "weeks", "value": 2},
  "key_questions": [
    "What does it mean to count?",
    "How do we know when we've counted everything?"
  ],
  "learning_outcomes": [
    {
      "outcome_id": "math.counting.v1.unit01.lo1",
      "description": "Count objects up to 10 with one-to-one correspondence.",
      "competency_refs": ["K.CC.B.4a"]
    }
  ],
  "lessons": [],
  "extensions": {
    "interleave_with": ["reading.phonics.v1.unit01"]
  }
}
```

---

## 5. Lesson

A single instructional session.

```json
{
  "lesson_id": "math.counting.v1.unit01.lesson01",
  "title": "Counting Objects",
  "sequence": 1,
  "estimated_duration_minutes": 30,
  "lesson_type": "instructional",
  "objectives": [
    {
      "objective_id": "math.counting.v1.unit01.lesson01.obj1",
      "description": "Touch and count objects up to 5.",
      "competency_refs": ["K.CC.B.4a"]
    }
  ],
  "prior_knowledge": ["Number names 1-5"],
  "materials": [
    {
      "material_id": "res.counting_blocks",
      "title": "Counting Blocks",
      "type": "manipulative",
      "uri": null
    }
  ],
  "activities": [],
  "assessment_plan": {
    "formative": [
      {
        "type": "observation",
        "description": "Observe one-to-one correspondence during counting."
      }
    ],
    "summative": []
  },
  "extensions": {
    "pedagogy": {
      "cpa_stage": "concrete",
      "primary_phase": "i_do"
    }
  }
}
```

---

## 6. Activity

A discrete learning task within a lesson.

```json
{
  "activity_id": "math.counting.v1.unit01.lesson01.act01",
  "title": "Teacher Models Counting",
  "sequence": 1,
  "activity_type": "demonstration",
  "grouping": "whole-class",
  "estimated_duration_minutes": 10,
  "outcome_links": ["math.counting.v1.unit01.lesson01.obj1"],
  "xapi_activity_id": "https://coherence.lab/xapi/activities/counting-demo-01",
  "variants": [
    {
      "variant_id": "claude",
      "source_model": "claude-3-opus",
      "approach": "visual-spatial",
      "instructions_teacher": "Model counting 5 blocks, touching each one and saying the number aloud. Emphasize one-to-one correspondence.",
      "instructions_learner": "Watch how the teacher counts. Notice that each block gets exactly one number.",
      "explanation": "When we count, each object gets exactly one number. Touch and say: 1, 2, 3, 4, 5."
    },
    {
      "variant_id": "chatgpt",
      "source_model": "gpt-4",
      "approach": "procedural",
      "instructions_teacher": "Line up 5 blocks. Point to each block in order while counting aloud. Ask students what number comes next.",
      "instructions_learner": "Follow along as we count each block in order.",
      "explanation": "Counting means saying numbers in order while pointing to objects. Let's count: 1... 2... 3... 4... 5!"
    },
    {
      "variant_id": "gemini",
      "source_model": "gemini-pro",
      "approach": "narrative",
      "instructions_teacher": "Tell a story about 5 blocks going on an adventure, counting each one as it joins.",
      "instructions_learner": "Listen to the story and count the blocks as they appear.",
      "explanation": "Once upon a time, one block set off on a journey. Another block joined - now there were 2. Then 3, 4, and finally 5 blocks!"
    }
  ],
  "resources": [
    {"material_ref": "res.counting_blocks"}
  ],
  "roles": [
    {
      "role": "teacher",
      "actions": ["demonstrate", "verbalize", "point"]
    },
    {
      "role": "student",
      "actions": ["observe", "listen"]
    }
  ],
  "extensions": {
    "pedagogy": {
      "cpa_stage": "concrete",
      "instructional_phase": "i_do",
      "scaffolding_level": 1.0
    },
    "di_training": {
      "native_representation": {
        "type": "numeric",
        "format": "integer",
        "range": [1, 5]
      },
      "problem_generator": {
        "name": "counting",
        "params": {"min_length": 3, "max_length": 5}
      },
      "autonomy_range": [0.0, 0.2],
      "prediction_point": false
    },
    "variant_selection": {
      "strategy": "random_first_then_alternate",
      "on_struggle": "try_different_approach",
      "track_effectiveness": true
    }
  }
}
```

---

## 7. Variants Schema

Activities can have multiple teaching variants from different sources. This enables:
- Multiple teaching angles for the same content
- Fallback explanations when one approach doesn't land
- Model-agnostic curriculum (no single AI bias)
- Effectiveness tracking per approach

### Variant Object

```json
{
  "variant_id": "string",           // Unique identifier (e.g., "claude_en", "deepseek_zh")
  "source_model": "string",         // Model/source that generated this variant
  "language": "string",             // BCP-47 language tag (e.g., "en-US", "zh-CN", "fr-CA")
  "approach": "string",             // Teaching style tag
  "instructions_teacher": "string", // What the teacher does
  "instructions_learner": "string", // What the student does
  "explanation": "string"           // The actual teaching content
}
```

### Language Diversity

Variants can be in different languages, providing:
- **Cultural perspective:** DeepSeek (Chinese training) vs Claude (Western training) = different pedagogical traditions
- **Cognitive flexibility:** Multilingual exposure strengthens learning
- **Concept clarity:** Some concepts are clearer in certain languages (e.g., Welsh numbers are transparent)

Example multilingual variants for counting:

```json
"variants": [
  {
    "variant_id": "claude_en",
    "source_model": "claude-3-opus",
    "language": "en-US",
    "approach": "procedural",
    "explanation": "Count with me: one, two, three, four, five."
  },
  {
    "variant_id": "deepseek_zh",
    "source_model": "deepseek-v2",
    "language": "zh-CN",
    "approach": "narrative",
    "explanation": "我们一起数：一、二、三、四、五。"
  },
  {
    "variant_id": "mistral_fr",
    "source_model": "mistral-large",
    "language": "fr-FR",
    "approach": "visual-spatial",
    "explanation": "Comptons ensemble : un, deux, trois, quatre, cinq."
  }
]
```

### Approach Tags

| Tag | Description |
|-----|-------------|
| `visual-spatial` | Uses imagery, diagrams, spatial reasoning |
| `procedural` | Step-by-step instructions |
| `narrative` | Story-based, contextual |
| `kinesthetic` | Movement, hands-on |
| `analogical` | Relates to familiar concepts |
| `socratic` | Question-driven discovery |

### Variant Selection Strategy

```json
"variant_selection": {
  "strategy": "random_first_then_alternate | fixed_order | student_preference",
  "on_struggle": "try_different_approach | repeat_same | escalate_scaffolding",
  "track_effectiveness": true | false
}
```

### Selection Logic

1. **First presentation:** Random variant (or fixed if specified)
2. **Struggle detected:** Select different variant with different `approach` tag
3. **Track results:** Log which variant_id worked for future optimization
4. **Learn preferences:** Over time, weight toward approaches that work for this student

---

## 8. Play & Exploration Schema

Play activities differ from instructional activities - they have no single "correct" answer and build intuition through experience.

### Play Activity

```json
{
  "activity_id": "physics.play.blocks01",
  "activity_type": "play",
  "play_type": "free_exploration | guided_discovery | structured_play",
  "environment": {
    "type": "physics_playground",
    "entities": ["block", "ball", "ramp"],
    "properties": {
      "gravity": true,
      "collision": true,
      "friction": 0.3
    },
    "constraints": {
      "bounds": [[-10, 10], [-10, 10], [0, 20]],
      "max_objects": 10
    }
  },
  "goals": [],  // Empty for free exploration, optional for guided
  "discovery_targets": [
    "objects_fall_down",
    "stacking_requires_balance",
    "balls_roll_blocks_dont"
  ],
  "duration": {
    "min_minutes": 5,
    "max_minutes": 20,
    "student_controlled": true
  },
  "extensions": {
    "di_training": {
      "intuition_building": ["gravity", "collision", "support"],
      "no_explicit_instruction": true,
      "observation_only": false
    }
  }
}
```

### Play Types

| Type | Description | Teacher Role |
|------|-------------|--------------|
| `free_exploration` | No goals, pure discovery | Observe only |
| `guided_discovery` | Gentle prompts toward discoveries | Occasional questions |
| `structured_play` | Specific scenarios to explore | Sets up situations |

### Earliest Physics (Pre-Academic)

Physics intuition starts alongside (not after) early academics:

```json
{
  "subject_id": "physics_intuition",
  "title": "Physical World Intuition",
  "description": "Pre-academic physics through play. Runs parallel to math/reading from day one.",
  "start_age": "earliest",
  "native_representation": {
    "type": "simulation",
    "entities": ["rigid_body", "particle"],
    "properties": ["position", "velocity", "mass", "shape"],
    "interactions": ["gravity", "collision", "support", "friction"]
  },
  "courses": [
    {
      "course_id": "physics_intuition.objects",
      "title": "Objects in Space",
      "concepts": ["things_exist", "things_have_position", "things_can_move"]
    },
    {
      "course_id": "physics_intuition.falling",
      "title": "Things Fall Down",
      "concepts": ["gravity_pulls_down", "unsupported_things_fall", "height_matters"]
    },
    {
      "course_id": "physics_intuition.collision",
      "title": "Things Bump",
      "concepts": ["objects_cant_overlap", "collision_changes_motion", "momentum_transfer"]
    }
  ]
}
```

---

## 9. Agent Representation Schema

**Critical:** Avoid "form rigidity" - agents/others should NOT be defined by appearance (text, voice, avatar, body). Define by *interaction patterns* and *behavioral signatures*.

### Agent Object

```json
{
  "agent_id": "teacher_primary",
  "agent_type": "teacher | peer | self | environment",
  "behavioral_signature": {
    "responsiveness": 0.9,      // How quickly/reliably responds
    "consistency": 0.95,        // How predictable behavior is
    "patience": 0.8,            // Tolerance for repeated attempts
    "warmth": 0.7,              // Emotional supportiveness
    "challenge_level": 0.5      // How much they push growth
  },
  "interaction_patterns": [
    "provides_demonstrations",
    "answers_questions",
    "gives_feedback",
    "offers_encouragement"
  ],
  "form": "abstract",           // NOT "text", "voice", "avatar"
  "manifestation": "variable"   // Can appear differently in different contexts
}
```

### Why Abstract Form?

Agents in the curriculum should NOT specify:
- ❌ "This is a text-based chat agent"
- ❌ "This appears as an avatar"
- ❌ "This has a voice"

Instead, agents are defined by:
- ✅ How they behave (patient, challenging, warm)
- ✅ What they do (demonstrate, question, encourage)
- ✅ Their relationship to the student (teacher, peer, environment)

This prevents the DI from developing rigid expectations about what "minds" look like.

### Environment as Agent

The physics playground itself can be an "agent" - it responds to actions, has consistent rules, and teaches through interaction:

```json
{
  "agent_id": "physics_world",
  "agent_type": "environment",
  "behavioral_signature": {
    "responsiveness": 1.0,      // Instant physical feedback
    "consistency": 1.0,         // Laws of physics don't change
    "patience": 1.0,            // Never gets frustrated
    "warmth": 0.0,              // No emotional content
    "challenge_level": "adaptive"  // Difficulty emerges from student actions
  },
  "interaction_patterns": [
    "responds_to_actions",
    "enforces_constraints",
    "reveals_consequences"
  ],
  "form": "abstract"
}
```

---

## 10. Emotional & Social Extensions

### `emotional` Extension

```json
{
  "emotional": {
    "teacher_stance": "warm | neutral | encouraging | challenging",
    "student_state_target": "curious | confident | focused | playful",
    "frustration_threshold": 0.7,  // When to intervene
    "celebration_triggers": ["first_success", "mastery", "creative_solution"],
    "social_emotional_goals": [
      "persistence_through_difficulty",
      "joy_in_discovery",
      "comfort_with_uncertainty"
    ]
  }
}
```

### `reward` Extension

```json
{
  "reward": {
    "type": "intrinsic | social | achievement",
    "signal": {
      "on_attempt": 0.1,        // Small reward for trying
      "on_progress": 0.3,       // Medium for moving forward
      "on_success": 0.5,        // Larger for correct
      "on_mastery": 1.0,        // Full for demonstrated mastery
      "on_creativity": 0.8      // High for novel valid solutions
    },
    "teacher_feedback": {
      "attempt": "Good try!",
      "progress": "You're getting closer!",
      "success": "That's right!",
      "mastery": "You've really got this!",
      "creativity": "What a clever approach!"
    }
  }
}
```

---

## 11. Extensions Schema

### `pedagogy` Extension

Present at lesson and activity levels.

```json
{
  "pedagogy": {
    "cpa_stage": "concrete | pictorial | abstract",
    "instructional_phase": "i_do | we_do | you_do_together | you_do_alone",
    "scaffolding_level": 0.0-1.0,
    "mastery_threshold": 0.8
  }
}
```

### `di_training` Extension

DI-specific training parameters.

```json
{
  "di_training": {
    "native_representation": {
      "type": "numeric | linguistic | simulation | temporal | spatial",
      "format": "subject-specific",
      "range": [min, max]
    },
    "autonomy_range": [min, max],
    "prediction_point": true | false,
    "interleave_candidates": ["activity_id_1", "activity_id_2"],
    "checkpoint_after": true | false
  }
}
```

### CPA Stage Definitions

| Stage | Description | Examples |
|-------|-------------|----------|
| `concrete` | Physical/tangible representation | Blocks, counters, real objects |
| `pictorial` | Visual representation | Drawings, diagrams, bar models |
| `abstract` | Symbolic representation | Numbers, equations, notation |

### Instructional Phase Definitions

| Phase | Teacher Role | Student Role | Scaffolding |
|-------|--------------|--------------|-------------|
| `i_do` | Models explicitly | Observes | 1.0 (full) |
| `we_do` | Guides with prompts | Participates with support | 0.7 |
| `you_do_together` | Monitors, intervenes as needed | Collaborates with peers | 0.4 |
| `you_do_alone` | Assesses | Works independently | 0.0-0.2 |

---

## 12. Activity Types

Standard vocabulary for `activity_type`:

| Type | Description |
|------|-------------|
| `demonstration` | Teacher shows, student observes |
| `guided_practice` | Teacher and student work together |
| `independent_practice` | Student works alone |
| `collaborative` | Students work in groups |
| `discussion` | Verbal exchange of ideas |
| `exploration` | Open-ended discovery |
| `assessment` | Formal evaluation |
| `game` | Playful practice |
| `simulation` | Physics playground or similar |
| `reflection` | Metacognitive journaling |

---

## 13. Grouping Types

Standard vocabulary for `grouping`:

- `individual` - One student alone
- `pair` - Two students together
- `small-group` - 3-5 students
- `whole-class` - All students together

---

## 14. Compiler Output Format

The curriculum compiler outputs a single JSON file:

```json
{
  "compiled_at": "2025-12-10T15:30:00Z",
  "compiler_version": "1.0",
  "source_curriculum": "coherence.k12.core.v1",
  "training_sequences": [
    {
      "sequence_id": "seq_00001",
      "source_activity": "math.counting.v1.unit01.lesson01.act01",
      "input": {
        "type": "demonstration",
        "content": "Count: 1, 2, 3, 4, 5",
        "representation": [1, 2, 3, 4, 5]
      },
      "expected_output": {
        "type": "observation",
        "content": "Acknowledged"
      },
      "metadata": {
        "cpa_stage": "concrete",
        "instructional_phase": "i_do",
        "scaffolding": 1.0,
        "subject": "math",
        "grade": 0
      }
    }
  ]
}
```

---

## 15. Validation Rules

1. All `*_id` fields must be unique within their scope
2. `sequence` fields must be positive integers, starting at 1
3. `competency_refs` should be valid CASE URIs when possible
4. `cpa_stage` must follow concrete → pictorial → abstract progression within a unit
5. `instructional_phase` must follow i_do → we_do → you_do progression within a lesson
6. `scaffolding_level` must decrease as autonomy increases

---

## 16. Example: Complete Mini-Curriculum

See `examples/mini_curriculum.json` for a complete working example.

---

## References

- CASE Specification: https://www.imsglobal.org/activity/case
- LRMI: https://dublincore.org/about/lrmi/
- Singapore Math CPA: https://mathsnoproblem.com/en/approach/concrete-pictorial-abstract
- Gradual Release: Fisher & Frey (2014)
- ChatGPT format proposal (December 10, 2025)

---

*Format version 1.0 - Subject to revision as compiler development proceeds*
