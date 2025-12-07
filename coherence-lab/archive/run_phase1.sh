#!/bin/bash
# Phase 1 Training Runner
# Run from coherence-lab directory

set -e

echo "=== Phase 1: Reflex Classification Training ==="
echo ""

# Check for PyTorch
python3 -c "import torch; print(f'PyTorch {torch.__version__}')" || {
    echo "PyTorch not installed. Installing..."
    pip install torch
}

echo ""
echo "Starting training..."
echo ""

# Run training with default params
# - 50 epochs max
# - Early stop at 98% accuracy
# - Patience of 5 epochs
python3 phase1_train.py --data-dir data --epochs 50 --target-acc 0.98 --patience 5

echo ""
echo "=== Training Complete ==="
echo "Checkpoint saved to: data/phase1_checkpoint.pt"
echo "History saved to: data/phase1_history.json"
