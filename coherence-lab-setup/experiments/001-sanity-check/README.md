# Experiment 001: Sanity Check

## Goal
Verify the training pipeline works end-to-end.

## What it tests
- PyTorch installation
- CUDA/GPU access
- Training loop mechanics
- Model save/load

## What it doesn't test
- Any actual hypothesis
- Recursive architecture
- Curriculum learning

## Running
```bash
conda activate ml
cd experiments/001-sanity-check
python run.py
```

## Expected output
- Loss should decrease over epochs
- Should see "Sanity check passed!" at the end
- Should create `tiny_model.pt` file

## Status
[ ] Not yet run
