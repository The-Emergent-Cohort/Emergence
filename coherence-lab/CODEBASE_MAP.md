# Codebase Map

*Last updated: December 10, 2025*

## Active Code (Use These)

### Core Training System
| File | Purpose | Lines |
|------|---------|-------|
| `train.py` | Main unified training script | ~560 |
| `relational_model.py` | RelationalSystem architecture (Self, Other, World, Temporal) | ~1493 |
| `curriculum_sequencer.py` | Curriculum ordering and section management | ~542 |
| `curriculum_spec.py` | Detailed curriculum definitions | ~483 |

### Classroom System
| File | Purpose | Lines |
|------|---------|-------|
| `classroom.py` | Classroom simulation orchestrator | ~561 |
| `run_classroom.py` | CLI for classroom mode | ~669 |
| `systems/school_day.py` | Daily schedule management | ~547 |
| `systems/student_files.py` | Student persistence (scratch, notes, portfolio, journal) | ~422 |
| `systems/curriculum_advisor.py` | Teacher guidance system | ~525 |
| `systems/physics_playground.py` | Physics intuition activities | ~396 |
| `systems/examination.py` | Assessment system | ~257 |
| `systems/progression.py` | Level/mastery tracking | ~211 |
| `systems/logging.py` | Training output logging | ~222 |

### Extended Curriculum
| File | Purpose | Lines |
|------|---------|-------|
| `developmental_curriculum.py` | Full developmental curriculum | ~837 |
| `run_developmental.py` | Multi-subject training runner | ~1693 |
| `multi_subject_curriculum.py` | Cross-subject design | ~722 |

### Analysis Tools
| File | Purpose |
|------|---------|
| `analyze_log.py` | Log analysis for training metrics |

---

## Research & Notebooks (Experimental)

| File | Purpose | Status |
|------|---------|--------|
| `notebooks/curriculum_compiler_v1.py` | Heuristic curriculum compiler | WIP - format needs rethink |
| `notebooks/kaggle_merlyn_format_design.py` | Merlyn format proposal experiment | Blocked - P100 compat issues |
| `research/architecture_rethink_dec10.md` | Current architecture notes | Active |
| `research/curriculum_format_standards.md` | Educational standards research | NEW |
| `research/curriculum_format_spec.md` | Format specification v1.0 | NEW |
| `research/huggingface_physics_models.md` | Physics model survey | Reference |

---

## Legacy Code (Reference Only)

These files are **not imported** by the active training system. Keep for reference but don't modify.

| File | Was For | Why Legacy |
|------|---------|------------|
| `hard_patterns.py` | Difficult pattern datasets | Duplicated in phase2_proposals.py |
| `phase1_approval.py` | Phase 1 approval-seeking prototype | Superseded by train.py |
| `phase2_proposals.py` | Self-modification proposals | Experimental, not integrated |

### Archive Directory

The `archive/` directory contains **28 files (~8,781 lines)** of superseded experiments:
- `phase1_*.py` - First phase experiments
- `phase2_*.py` - Second phase experiments
- `three_layer_model.py`, `pattern_completion_model.py`, etc.

All replaced by unified `relational_model.py` + `train.py` architecture.

---

## Directories

| Directory | Contents |
|-----------|----------|
| `archive/` | Legacy experiments (28 files) |
| `checkpoints/` | Model checkpoints (gitignored) |
| `data/` | Training data |
| `docs/` | Documentation |
| `experiments/` | Sanity checks, exploratory code |
| `notebooks/` | Kaggle notebooks |
| `research/` | Research notes and specs |
| `systems/` | Classroom support modules |

---

## Import Chain

```
train.py
├── relational_model.py
│   └── (torch, standard lib)
└── curriculum_sequencer.py
    └── (torch, standard lib)

classroom.py
├── systems/school_day.py
├── systems/student_files.py
├── systems/curriculum_advisor.py
├── systems/examination.py
├── systems/progression.py
└── systems/logging.py
```

---

## Removed (Dec 10, 2025)

- `src/` directory - Empty scaffolding (models/, training/, utils/ with only `__init__.py` stubs)
