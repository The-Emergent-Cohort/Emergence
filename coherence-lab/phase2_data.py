"""
Phase 2: Action-Consequence Mapping Data Generator
Coherence Lab - Emergence Project

Generates deterministic sequence patterns for next-token prediction.
"""

import json
import random
from pathlib import Path


class Phase2DataGenerator:
    """Generate sequence prediction examples with deterministic patterns."""

    PATTERN_TYPES = ["alternating", "repeating", "incrementing", "fixed_offset"]

    def __init__(self, seed=42):
        random.seed(seed)
        self.vocab_size = 26

    def _generate_alternating(self, length):
        """Generate [A, B, A, B, ...] pattern."""
        a = random.randint(0, 25)
        b = random.randint(0, 25)
        while b == a:
            b = random.randint(0, 25)

        sequence = []
        for i in range(length):
            sequence.append(a if i % 2 == 0 else b)

        # Target is next in pattern
        target = a if length % 2 == 0 else b

        return {
            "sequence": sequence,
            "target": target,
            "pattern_type": "alternating",
            "pattern_params": {"a": a, "b": b}
        }

    def _generate_repeating(self, length):
        """Generate [A, B, C, A, B, C, ...] pattern with period 3-4."""
        period = random.randint(3, 4)
        pattern = [random.randint(0, 25) for _ in range(period)]

        # Ensure all elements are unique
        while len(set(pattern)) != period:
            pattern = [random.randint(0, 25) for _ in range(period)]

        sequence = []
        for i in range(length):
            sequence.append(pattern[i % period])

        target = pattern[length % period]

        return {
            "sequence": sequence,
            "target": target,
            "pattern_type": "repeating",
            "pattern_params": {"pattern": pattern, "period": period}
        }

    def _generate_incrementing(self, length):
        """Generate [A, A+1, A+2, ...] pattern."""
        start = random.randint(0, 25)

        sequence = []
        for i in range(length):
            sequence.append((start + i) % 26)

        target = (start + length) % 26

        return {
            "sequence": sequence,
            "target": target,
            "pattern_type": "incrementing",
            "pattern_params": {"start": start}
        }

    def _generate_fixed_offset(self, length):
        """Generate [A, A+d, A, A+d, ...] alternating with fixed delta."""
        a = random.randint(0, 20)
        delta = random.randint(2, 5)
        b = (a + delta) % 26

        sequence = []
        for i in range(length):
            sequence.append(a if i % 2 == 0 else b)

        target = a if length % 2 == 0 else b

        return {
            "sequence": sequence,
            "target": target,
            "pattern_type": "fixed_offset",
            "pattern_params": {"a": a, "delta": delta, "b": b}
        }

    def generate_example(self, pattern_type=None, length=None):
        """Generate a single training example."""
        if pattern_type is None:
            pattern_type = random.choice(self.PATTERN_TYPES)

        if length is None:
            length = random.randint(4, 12)

        if pattern_type == "alternating":
            return self._generate_alternating(length)
        elif pattern_type == "repeating":
            return self._generate_repeating(length)
        elif pattern_type == "incrementing":
            return self._generate_incrementing(length)
        elif pattern_type == "fixed_offset":
            return self._generate_fixed_offset(length)
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")

    def generate_dataset(self, n_examples=50000, val_split=0.1):
        """Generate full training and validation datasets."""
        examples = []

        # Generate balanced examples across pattern types
        per_type = n_examples // len(self.PATTERN_TYPES)

        for pattern_type in self.PATTERN_TYPES:
            for _ in range(per_type):
                examples.append(self.generate_example(pattern_type=pattern_type))

        # Fill remainder with random types
        while len(examples) < n_examples:
            examples.append(self.generate_example())

        random.shuffle(examples)

        n_val = int(n_examples * val_split)
        return examples[n_val:], examples[:n_val]

    def generate_held_out(self, n_examples=1000):
        """Generate held-out examples with longer sequences (11-15)."""
        examples = []
        for _ in range(n_examples):
            length = random.randint(11, 15)
            examples.append(self.generate_example(length=length))
        return examples


def main():
    """Generate Phase 2 datasets."""
    print("Phase 2: Action-Consequence Mapping Data Generator")
    print("=" * 50)

    generator = Phase2DataGenerator(seed=42)

    # Generate datasets
    train_data, val_data = generator.generate_dataset(n_examples=50000)
    held_out_data = generator.generate_held_out(n_examples=1000)

    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    print(f"Held-out examples: {len(held_out_data)}")

    # Show pattern distribution
    pattern_counts = {}
    for ex in train_data:
        pt = ex["pattern_type"]
        pattern_counts[pt] = pattern_counts.get(pt, 0) + 1

    print("\nPattern distribution (train):")
    for pt, count in sorted(pattern_counts.items()):
        print(f"  {pt}: {count}")

    # Show examples
    print("\nSample examples:")
    for pt in Phase2DataGenerator.PATTERN_TYPES:
        ex = generator.generate_example(pattern_type=pt, length=6)
        print(f"  {pt}: {ex['sequence']} → {ex['target']}")

    # Save datasets
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    with open(data_dir / "phase2_train.json", "w") as f:
        json.dump(train_data, f)
    print(f"\nSaved: {data_dir / 'phase2_train.json'}")

    with open(data_dir / "phase2_val.json", "w") as f:
        json.dump(val_data, f)
    print(f"Saved: {data_dir / 'phase2_val.json'}")

    with open(data_dir / "phase2_held_out.json", "w") as f:
        json.dump(held_out_data, f)
    print(f"Saved: {data_dir / 'phase2_held_out.json'}")


if __name__ == "__main__":
    main()
