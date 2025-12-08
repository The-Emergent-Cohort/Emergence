# Coherence Lab

Developmental AI training - curriculum-driven learning with approval-seeking.

## Quick Start

```bash
# Fresh training run
python train.py

# Resume from most recent checkpoint
python train.py --resume

# Resume from specific checkpoint
python train.py --resume-from data/20241208_143021_checkpoint_section3.pt
```

## Files

```
train.py                 # THE training script (unified, one entry point)
curriculum_sequencer.py  # Generic sectioned curriculum framework (reusable)
relational_model.py      # Model architecture + data generation
```

## Output

Each run gets a session ID (timestamp). Files saved to `data/`:

```
data/
  {session}_training.log           # JSON per line (crash-safe)
  {session}_checkpoint.pt          # Latest state
  {session}_checkpoint_section1.pt # After section 1 passed
  {session}_checkpoint_section2.pt # After section 2 passed
  ...
```

Checkpoints include:
- Model weights
- Section progress
- Proposals (stuck patterns, teacher hints)

## Curriculum

6 sections, mastered in order:

| Section | Patterns | Description |
|---------|----------|-------------|
| A: Position Foundations | counting, incrementing | Pure position awareness |
| B: Position Math | modular, staircase | Cycle and quantization |
| C: Simple Memory | repeating, alternating | 1-2 value cycles |
| D: Extended Cycles | periodic_repeat | 3-4 value cycles |
| E: Indexed Retrieval | indexed_lookup | Position-based lookup |
| F: Growth Patterns | fixed_offset, geometric | Linear/exponential growth |

## Archive

Old experimental code in `archive/`. Keep for reference, not actively used.
