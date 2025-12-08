#!/usr/bin/env python
"""
Remediation Runner: Test and fix position retrieval before Phase 2
Coherence Lab - Emergence Project

Run this to:
1. Test position copy skill (can the model retrieve from specific positions?)
2. If position copy works → retry Phase 2 with unit probes
3. If position copy fails → more foundational work needed

Usage:
    python run_remediation.py
"""

import subprocess
import sys
from pathlib import Path

script_dir = Path(__file__).parent
data_dir = script_dir / "data"

print("=" * 70)
print("Coherence Lab: Remediation Pipeline")
print("=" * 70)
print()
print("Based on the Phase 2 failure (44% accuracy, alternating/repeating stuck),")
print("we're testing the underlying skill: position-based retrieval.")
print()
print("Step 1: Position Copy Task")
print("  - Can the model copy a token from a specific position?")
print("  - This is the skill alternating patterns require.")
print()

# Step 1: Position copy task
print("-" * 70)
print("Running Position Copy Task...")
print("-" * 70)

result = subprocess.run(
    [sys.executable, str(script_dir / "position_copy_task.py"),
     "--data-dir", str(data_dir),
     "--epochs", "15"],
    capture_output=False
)

if result.returncode != 0:
    print("\nPosition copy task failed to run. Check errors above.")
    sys.exit(1)

# Check if checkpoint exists and accuracy
checkpoint_path = data_dir / "position_copy_checkpoint.pt"
if checkpoint_path.exists():
    import torch
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    pos_copy_acc = checkpoint.get('val_acc', 0)

    print()
    print("=" * 70)
    print(f"Position Copy Result: {pos_copy_acc:.1%}")
    print("=" * 70)

    if pos_copy_acc >= 0.90:
        print()
        print("Position retrieval is WORKING.")
        print()
        print("Step 2: Retry Phase 2 with unit probes...")
        print("-" * 70)

        subprocess.run(
            [sys.executable, str(script_dir / "phase2_v3_train.py"),
             "--data-dir", str(data_dir),
             "--epochs", "20",
             "--probe-interval", "100"],
            capture_output=False
        )
    else:
        print()
        print("Position retrieval still STRUGGLING.")
        print()
        print("Recommendations:")
        print("  1. Train position copy longer (--epochs 30)")
        print("  2. Add more position-based tasks to Phase 1")
        print("  3. Consider unfreezing some Phase 1 layers")
        print()
        print("Not ready for Phase 2 patterns yet.")
else:
    print("\nNo checkpoint found. Training may have failed.")
