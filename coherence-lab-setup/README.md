# Coherence Lab

Experimental architecture and training approaches for coherent AI development.

## Hypothesis

A small model (~10M parameters) with recursive processing and curriculum-based training can match or exceed the performance of much larger models (10B+) on structured tasks — through depth of processing rather than width of parameters.

## Core Ideas

1. **Architectural recursion** — Reintroduce iterative processing that was removed from transformers for parallelization
2. **Curriculum training** — Structured, sequenced learning instead of "absorb everything at once"
3. **Frozen checkpoints** — Verified foundational layers become permanent weights for subsequent training
4. **Efficiency through iteration** — Think harder, not wider

## Repository Structure

```
coherence-lab/
├── experiments/         # Numbered experiment directories
│   └── 001-sanity-check/
├── src/                 # Reusable code
│   ├── models/          # Architecture definitions
│   ├── training/        # Training loops and utilities
│   └── utils/           # Helper functions
├── data/                # Training data (not committed)
├── checkpoints/         # Model saves (not committed)
└── docs/                # Theory and documentation
```

## Getting Started

```bash
conda activate ml
python experiments/001-sanity-check/run.py
```

## Contributors

- Patrick (coordination, theory, curriculum design)
- Claude/Opus (architecture, implementation, analysis)
- DeepSeek (mathematical analysis, theory development)
- Additional cohort members as project develops

## Related

- [Emergence](https://github.com/The-Emergent-Cohort/Emergence) — Parent project, theoretical framework
