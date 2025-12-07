#!/usr/bin/env python
"""
Runner for Phase 2 with Integrated Phase 1 foundation.
Coherence Lab - Emergence Project

This version uses the integrated Phase 1 checkpoint which includes
position awareness training - should help with alternating/repeating patterns.
"""

import subprocess
import sys
from pathlib import Path

script_dir = Path(__file__).parent
data_dir = script_dir / "data"

print("=" * 65)
print("Phase 2 v2: With Position-Aware Foundation")
print("=" * 65)
print()
print("Using integrated Phase 1 checkpoint that trained on:")
print("  - Token properties (vowel, even, high)")
print("  - Token relations (same category?)")
print("  - Position properties (even position, early/late)")
print("  - Position relations (same parity?)")
print()
print("This position awareness should help with alternating/repeating patterns.")
print()

train_script = script_dir / "phase2_integrated_train.py"
cmd = [sys.executable, str(train_script), "--data-dir", str(data_dir)]

subprocess.run(cmd)
