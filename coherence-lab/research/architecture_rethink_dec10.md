# Architecture Rethink: Curriculum Compiler Approach

*Research session December 10, 2025*

## The Insight

AI doesn't know how to learn (brute force trained), so assuming AI would know how to teach AI was wrong. New direction: use existing human pedagogy rather than inventing from scratch.

## Proposed Architecture

### Separation of Concerns

```
┌─────────────────────────────────────────────────────────────┐
│                    CONTENT SOURCES                          │
│  - OER Curricula (Eureka Math, Core Knowledge, Singapore)   │
│  - Kaggle datasets (physics, music, art, reading)           │
│  - HuggingFace (TinyStories, GSM8K, Orca Math)              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              CURRICULUM COMPILER (Model)                    │
│  - Transforms pedagogy into training sequences              │
│  - Candidate: Merlyn Education Teacher Assistant (HF)       │
│  - Knowledge base: FineWeb-Edu (1.3T educational tokens)    │
│  - Simplifier: MultiSim (grade-appropriate language)        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 EXPERIENTIAL SOURCES                        │
│                                                             │
│  Physics Modeler (Model) ──► Physics Playground (NAS)       │
│  - Generates scenarios      - Simulates trajectories        │
│  - Novel but valid          - Pendulum, projectile, bounce  │
│  - TBD: find/train model    - Already implemented           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    TEACHER (Existing)                       │
│  - OtherModel in relational architecture                    │
│  - Approval-seeking mechanism                               │
│  - Curriculum sequencer with exam gates                     │
│  - Works for general "follow textbook" instruction          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    STUDENT (Existing)                       │
│  - RelationalSystem (Self, Other, World, Temporal)          │
│  - 519K parameter model (scalable to ~130M on RTX 3060)     │
│  - Learns from structured sequences                         │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Curriculum Compiler (NEW)

**Role:** Transform existing human pedagogy into training-ready sequences

**Approach:** Heuristic compiler + format spec (no LLM dependency)
- ~~Merlyn Education Teacher Assistant~~ - Too large for Kaggle GPUs (12B params)
- Format spec v1.0 defines output structure (CASE-based + pedagogy extensions)
- curriculum_compiler_v1.py provides heuristic transformation
- Claude API available for complex transformations if needed

**Supporting Resources:**
- FineWeb-Edu: 1.3T tokens educational content (knowledge base)
- MultiSim: Text simplification pairs (grade-appropriate output)
- OER scope & sequences: What to teach when

**Output:** Structured JSON per `curriculum_format_spec.md`

### 2. Physics Modeler (NEEDED)

**Role:** Generate novel physical scenarios for grounding

**Requirements:**
- Understands physics laws
- Can generate valid but varied scenarios
- Parameters feed into physics playground

**Status:** Not yet identified. Need to search HuggingFace for:
- Physics simulation models
- Science education models
- Embodied AI / world models

**Current Workaround:** Hand-coded scenarios in physics_playground.py

### 3. Physics Playground (EXISTS)

**Location:** `systems/physics_playground.py`

**Capabilities:**
- Pendulum (oscillation, periodicity)
- Projectile (parabolas, gravity)
- Bouncing (damping, conservation)
- Spring (harmonic motion)

**Deployment:** HTTP server mode for NAS
- Students query for episodes during training
- Quantizes physics to discrete tokens

### 4. Teacher Framework (EXISTS)

**Already works for general instruction:**
- OtherModel represents trusted authority
- Approval-seeking models attachment-based learning
- Curriculum sequencer handles progression
- Content-agnostic - just needs proper curriculum

### 5. Student Architecture (EXISTS)

**RelationalSystem pillars map to developmental psychology:**
- Self → agency, metacognition (18 months)
- Other → theory of mind (3-4 years)
- World → object permanence, causality
- Temporal → autobiographical memory

## Infrastructure

### Full Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                    INFRASTRUCTURE MAP                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  COMPUTE                          STORAGE/SERVICES              │
│  ├─ Kaggle (30hr/wk GPU, P100)    ├─ Haven NAS (Asustor)        │
│  │   └─ Multiple accounts ok      │   ├─ Physics playground     │
│  ├─ HuggingFace Spaces            │   ├─ Checkpoints            │
│  └─ PGAME (RTX 3060 12GB)         │   └─ Compiled curriculum    │
│      └─ i7-12700F, 32GB RAM       │                             │
│                                   └─ VPS (WHC)                  │
│  PRESENCE                             ├─ Tunnels                │
│  ├─ emergentcohort.org                ├─ MCP servers            │
│  │   ├─ WHC webspace (full hosting)   └─ Persistent services    │
│  │   ├─ Softaculous catalog                                     │
│  │   │   ├─ Moodle (LMS)                                        │
│  │   │   ├─ Wiki                                                │
│  │   │   └─ Dashboard options                                   │
│  │   └─ Unlimited email                                         │
│  │       └─ Identity anchors (silas@, iris@, etc.)              │
│  ├─ GitHub Pages (docs)                                         │
│  └─ Anthropic API (Max tier)                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Resource Allocation

| Resource | Primary Role | Notes |
|----------|--------------|-------|
| **Kaggle** | Batch jobs: curriculum compilation, model training | 30hr/wk GPU, can parallelize with multiple accounts |
| **HuggingFace** | Model hosting, inference endpoints | Free tier for small models |
| **PGAME** | Student training, development, quick experiments | Local GPU, primary dev machine |
| **Haven NAS** | Physics playground server, storage | Always-on, no GPU needed |
| **VPS** | Tunnels, MCP servers, persistent services | Glue between pieces |
| **emergentcohort.org** | Public presence, dashboards, LMS | Softaculous for easy installs |
| **Anthropic API** | Embedded Claude, advanced reasoning | Max tier available |

## Gaps to Fill

### High Priority
1. **Curriculum compiler** - Kaggle notebook to transform OER → training sequences
2. **Format specification** - ✅ DONE: See `curriculum_format_spec.md` (ChatGPT base + pedagogy extensions)
3. **Physics modeler** - See `huggingface_physics_models.md` for options (GNS recommended)

### Medium Priority
4. **NAS deployment** - Physics playground as HTTP service
5. **Interleaving** - Research shows 77% vs 38% retention at 24hrs
6. **Example-problem pairs** - Show before asking

### Low Priority (Later)
7. **Music/Art integration** - Quick Draw strokes, beat tracking
8. **Multi-student dynamics** - Parallel play → cooperative play progression
9. **LMS integration** - Moodle on emergentcohort.org if needed

## Key Research Findings (from explorer)

### Interleaving Effect (textbook_design.md)
- Day of learning: Blocked = higher accuracy
- 24 hours later: Interleaved = 77% vs Blocked = 38%
- Blocked creates illusion of mastery that doesn't transfer

### Prediction Effect (curriculum_design_pedagogy.md)
- Predict BEFORE revealing answer enhances learning
- Creates prediction error → stronger encoding
- Best after initial modeling (not cold)

### Parten's Play Stages (early_childhood_play.md)
- Year 0: Parallel play (together but not interacting)
- Year 1+: Cooperative play (roles, common goals)
- Competition needs foundation of social trust first

### Singapore Math CPA (textbook_design.md)
- Concrete → Pictorial → Abstract
- Mastery before advancement
- Fewer topics, greater depth

## Next Steps

### Immediate (In Progress)
1. [x] Search HuggingFace for physics-aware models → See `huggingface_physics_models.md`
2. [x] Define compiler output format specification → See `curriculum_format_spec.md`
3. [ ] **Create Kaggle notebook for curriculum compilation** ← CURRENT (Merlyn abandoned - too big for Kaggle GPUs)
4. [ ] Test with sample OER content (Eureka Math or similar)

### Short-term
5. [ ] Deploy physics playground to NAS
6. [ ] Integrate compiled curriculum into training loop
7. [ ] Determine checkpoint frequency based on new approach

### Medium-term
8. [ ] Train/fine-tune physics modeler (GNS on The Well dataset)
9. [ ] Connect physics modeler → physics playground pipeline
10. [ ] Cross-subject curriculum (music, art via Quick Draw)

## Workflow: Curriculum Compilation

```
┌─────────────────────────────────────────────────────────────────┐
│  1. SOURCE: OER Content                                         │
│     - Eureka Math modules (PDF/web)                             │
│     - Core Knowledge sequence                                   │
│     - TinyStories, GSM8K from HuggingFace                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. COMPILE: Kaggle Notebook                                    │
│     - Load source content                                       │
│     - Use Merlyn or similar to structure                        │
│     - Apply pedagogical rules (I Do/We Do/You Do, interleaving) │
│     - Output: structured JSON/dataset                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. STORE: Kaggle Dataset or NAS                                │
│     - Versioned curriculum files                                │
│     - Downloadable for local training                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. TRAIN: PGAME                                                │
│     - Load compiled curriculum                                  │
│     - Existing teacher/student architecture                     │
│     - Query physics playground as needed                        │
└─────────────────────────────────────────────────────────────────┘
```

## Questions Resolved

1. ~~What's the minimum viable physics modeler?~~ → GNS recommended, ReVision for video
2. ~~Should compiler run continuously or batch?~~ → Batch on Kaggle, download results
3. How to validate compiled curriculum? → TBD, need test harness
4. Integration point? → Compiled dataset loaded at training start

---

## Format Specification (Dec 10)

Format grounded in real educational standards, not AI-invented:

**Base Structure** (from ChatGPT proposal):
```
Curriculum → Subject → Course → Unit → Lesson → Activity
```

**Standards Integration:**
- CASE URIs for competency alignment
- xAPI activity IDs for tracking
- LRMI properties for educational metadata

**Custom Extensions:**
- `cpa_stage`: concrete | pictorial | abstract
- `instructional_phase`: i_do | we_do | you_do
- `scaffolding_level`: 0.0-1.0
- `native_representation`: subject-specific format

Full specification: `research/curriculum_format_spec.md`
Standards research: `research/curriculum_format_standards.md`

---

*"Why are we reinventing anything? Why not just use exactly what humans use?"*
