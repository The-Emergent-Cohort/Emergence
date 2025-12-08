#!/usr/bin/env python
"""
Runner script for Integrated Phase 1 training.
Coherence Lab - Emergence Project

Usage:
    python run_integrated_phase1.py
"""

import subprocess
import sys
from pathlib import Path

# Change to coherence-lab directory
script_dir = Path(__file__).parent
data_dir = script_dir / "data"

print("=" * 65)
print("Integrated Phase 1: Multi-Task Foundation Training")
print("=" * 65)
print()
print("This trains a single model on 4 tasks simultaneously:")
print("  1. Token properties (vowel, even, high, etc.)")
print("  2. Token relations (do two tokens share a category?)")
print("  3. Position properties (even position, early/late, etc.)")
print("  4. Position relations (do two positions have same parity?)")
print()
print("The goal is 95%+ accuracy on ALL tasks before proceeding to Phase 2.")
print()

# Run the training
train_script = script_dir / "phase1_integrated_train.py"
cmd = [sys.executable, str(train_script), "--data-dir", str(data_dir)]

subprocess.run(cmd)
