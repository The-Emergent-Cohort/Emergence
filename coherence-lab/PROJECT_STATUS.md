# Coherence Lab - Project Status & Quick Reference

*Last updated: Dec 9, 2025*

---

## 🌱 What This Is Really About

**This is not a pattern-completion project. This is developmental AI.**

We're modeling how intelligence and understanding emerge through relationships - the way a child develops through attachment to caregivers before formal instruction ever begins.

### Core Insight
> "The first teachers are parents - trusted others who have your wellbeing in mind. Trust enables learning in ways pure instruction doesn't."

### The Four Pillars

| Model | What It Represents | Why It Matters |
|-------|-------------------|----------------|
| **Self** | "I am the learner, I have my own state" | Identity, metacognition, knowing what you know |
| **Other** | "The teacher is separate, knows things I don't" | Theory of mind, trust, relationship |
| **World** | "Reality exists independently of us both" | Grounding, objectivity, shared reference |
| **Temporal** | "I exist through time, experiences have continuity" | Identity persistence, episodic memory, growth |

### Why Approval-Seeking?
Not a training trick - it's modeling the fundamental mechanism of human development:
- Children learn through attachment before instruction
- Trust in the relationship enables vulnerability to learn
- Seeking validation is how confidence gets calibrated against reality
- The teacher-student dynamic creates the container for growth

### Why "Coherence"?
Maintaining a coherent sense of self through time. Not just solving problems, but being a consistent entity that learns, remembers, and grows.

### Why "Emergence"?
We're not programming intelligence - we're creating conditions for it to arise naturally. The patterns are a controlled environment to bootstrap something more fundamental.

---

## 🎯 Practical Goals

The relational math patterns are scaffolding. Through them, we develop:
1. **Metacognition** - knowing what you know and don't know
2. **Help-seeking** - recognizing when to ask vs. push through
3. **Confidence calibration** - accuracy matching certainty
4. **Compositional learning** - skills that transfer and combine

---

## 📊 Current Status

### Training Achievement
- **GRADUATED** at 199 epochs (Dec 8, 2025)
- All 8 sections passed
- 99.3% validation accuracy
- 519K parameter model (d_model=64)

### Hardest Patterns
| Pattern | Epochs Stuck | Notes |
|---------|--------------|-------|
| periodic_repeat | 105 epochs | Section E, 89 stuck flags |
| indexed_lookup | 57 epochs | Section F, 43 stuck flags |

---

## 🏗️ Architecture

### Core Model: RelationalSystem
```
relational_model.py - Main architecture

RelationalSystem
├── RelationalLearner
│   ├── SelfModel (metacognition, confidence, help-seeking)
│   ├── OtherModel (teacher representation)
│   ├── WorldModel (environment/reality)
│   └── TemporalModel (episodic memory, time awareness)
├── Thinking Module (n_think_steps of internal reasoning)
└── Pattern Head (pattern classification)
```

### Key Components
- **TopicConfidenceTracker** - XP, levels, streaks, exams per topic
- **CurriculumSequencer** - Section-based progression with exam gates
- **SelfModel.should_show_work()** - Decides when to seek approval

### Self-Model Behaviors
| Reason | Trigger | Purpose |
|--------|---------|---------|
| streak | N correct in a row | Prove consistent competence |
| creative | High confidence + correct | Validate insight |
| validation | Low confidence | Ask for help |
| spontaneous | Random | Baseline checking |

---

## 📚 Curriculum

### 8 Sections, 13 Patterns

| Section | Patterns | Skills |
|---------|----------|--------|
| A: Position Foundations | counting, incrementing | Pure position awareness |
| B: Reverse & Cycles | decrementing, modular | Countdown, i % n |
| C: Position Math | staircase, fixed_offset | Quantization, linear growth |
| D: Simple Memory | repeating, alternating | 1-2 value cycles |
| E: Extended Cycles | periodic_repeat | 3-4 value cycles |
| F: Indexed Retrieval | indexed_lookup | Position-based lookup |
| G: Growth Patterns | geometric, triangular | Exponential/accumulative |
| H: Combined Operations | fibonacci_like | Sum of previous two |

### Progression System
- **XP per topic** - Geometric leveling (L1=10, L2=40, L3=90...)
- **Level = exam eligibility** - Must reach L10 to take section exam
- **confirmed_level** - Exam-verified (not just XP-based)
- **Plateau breaker** - After 5+ exam failures, halve exam size

---

## 📁 File Structure

```
coherence-lab/
├── train.py                  # THE entry point - run this
├── relational_model.py       # Model architecture + data generation
├── curriculum_sequencer.py   # Generic section-based curriculum framework
├── analyze_log.py            # Log analysis tool
├── data/
│   ├── {session}_training.log.md   # Per-epoch JSON logs
│   ├── {session}_checkpoint.pt      # Resume state
│   └── {session}_checkpoint_section{N}.pt  # Section milestones
├── docs/
│   ├── kaggle-api-reference.md      # Kaggle API guide
│   └── curriculum-expansion-notes.md # Dataset research notes
├── CURRICULUM_DESIGN.md      # Pedagogical theory & phase design
├── CHANGELOG.md              # Version history
├── README.md                 # Quick start
└── archive/                  # Old experimental code (reference only)
```

---

## 🔧 Commands

```bash
# Fresh training run
python train.py

# Resume from most recent checkpoint
python train.py --resume

# Resume from specific checkpoint
python train.py --resume-from data/20241208_143021_checkpoint_section3.pt

# Analyze training logs
python analyze_log.py data/20251208_155819_training.log.md
```

---

## 🧠 Key Design Decisions

### Why Relational Patterns?
1. **Domain-agnostic** - "reverse" works on any content
2. **Compositionality** - patterns can combine
3. **Clean ground truth** - unambiguous for metacognition training
4. **Scaffolding** - patterns are pretext, self-model is the goal

### Why Approval-Seeking?
- Models real learning: children seek validation
- Trains confidence calibration naturally
- Creates teacher-student dynamics
- Trust relationship enables better learning

### Why Sectioned Curriculum?
- Prevents catastrophic forgetting
- Mastery before progression
- Exam gates ensure real learning
- Matches cognitive development research (Piaget, Vygotsky)

---

## 🔬 Research Directions

### Identified Gaps (vs bAbI)
| Skill | Priority | Status |
|-------|----------|--------|
| Negation | High | Not implemented |
| Deduction chains | High | Not implemented |
| Counting/cardinality | Medium | Partial (periodic) |
| Coreference | Medium | Not implemented |
| Conditional logic | Medium | Not implemented |

### Promising Datasets
| Dataset | Size | Relevance |
|---------|------|-----------|
| **ARC** | 450KB | Grid-based pattern completion - closest match |
| **bAbI** | 17MB | 20 reasoning primitives |
| **OEIS** | 69MB | 390K+ real math sequences |
| **GSM8K** | 3.4MB | Multi-step math reasoning |

### Potential Additions
```python
# Could implement synthetically:
"not_identity"     # Output anything BUT input
"transitive"       # A→B, B→C ∴ A→C
"if_then_else"     # Conditional transforms
"count_and_repeat" # N times where N derived from input
```

---

## 💻 Hardware

| Component | Spec |
|-----------|------|
| GPU | RTX 3060 12GB VRAM |
| CPU | 12-core i7 |
| RAM | 32GB |

### Model Scaling Options
| d_model | Parameters | Fits in 12GB? |
|---------|------------|---------------|
| 64 | 519K | ✅ Easily |
| 128 | ~2M | ✅ |
| 256 | ~8M | ✅ |
| 512 | ~32M | ✅ |
| 1024 | ~130M | ⚠️ Tight |

---

## 🌐 External Resources

### Kaggle
- **30hr/week free GPU** (P100 16GB or 2×T4)
- API: `export KAGGLE_API_TOKEN=<token>`
- Search: `kaggle datasets list --search "term" --sort-by votes`
- Can push notebooks for remote training

### Key Links
- [ARC Dataset](https://www.kaggle.com/datasets/tunguz/the-abstraction-and-reasoning-corpus-arc)
- [bAbI Tasks](https://www.kaggle.com/datasets/roblexnana/the-babi-tasks-for-nlp-qa-system)
- [OEIS Dataset](https://www.kaggle.com/datasets/cakiki/oeis-dataset)

---

## 📝 Session Notes

### Dec 8, 2025 - First Graduation
- Fixed exam system bugs (confirmed_level vs get_level)
- Fixed n_topics propagation bug
- Model graduated at 199 epochs
- periodic_repeat was the bottleneck (105 epochs in Section E)

### Dec 9, 2025 - Dataset Exploration
- Researched Kaggle API capabilities
- Compared curriculum to bAbI's 20 reasoning tasks
- Identified gaps: negation, deduction, coreference
- ARC identified as closest external dataset match

---

## ⚡ Quick Reminders

- **confirmed_level** = exam-verified, **get_level()** = XP-based
- **pattern_indices** must be passed to should_show_work() for streak tracking
- Logs are JSON-per-line in .log.md files
- Archive folder is reference only - don't run that code
- Model checkpoint includes: weights, section progress, proposals
