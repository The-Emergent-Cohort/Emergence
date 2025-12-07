"""
Phase 1: Reflexes Data Generator
Coherence Lab - Emergence Project

Generates deterministic reflex training data for Phase 1.
No randomness in token→label mapping. Randomness only in sequence construction.
"""

import random
import json
from pathlib import Path


class Phase1ReflexDataGenerator:
    """Generate trivially learnable deterministic patterns."""

    VOWELS = {0, 4, 8, 14, 20}  # A, E, I, O, U
    REFLEX_LABELS = [
        "VOWEL_MATCH",
        "CONSONANT_MATCH",
        "EVEN_VALUE",
        "ODD_VALUE",
        "HIGH_HALF",
        "LOW_HALF"
    ]

    def __init__(self, seed=42):
        random.seed(seed)

    def _get_reflex_labels(self, token):
        """
        Pure function: token → set of applicable reflexes.
        Deterministic. Always returns same labels for same token.
        """
        labels = []

        # Reflex 1 & 2: Vowel/Consonant
        if token in self.VOWELS:
            labels.append("VOWEL_MATCH")
        else:
            labels.append("CONSONANT_MATCH")

        # Reflex 3 & 4: Parity
        if token % 2 == 0:
            labels.append("EVEN_VALUE")
        else:
            labels.append("ODD_VALUE")

        # Reflex 5 & 6: Value range
        if token >= 13:
            labels.append("HIGH_HALF")
        else:
            labels.append("LOW_HALF")

        return labels

    def _labels_to_multihot(self, labels):
        """Convert label list to multi-hot vector."""
        multihot = [0] * len(self.REFLEX_LABELS)
        for label in labels:
            idx = self.REFLEX_LABELS.index(label)
            multihot[idx] = 1
        return multihot

    def _generate_random_sequence(self, min_len=5, max_len=20):
        """Generate random sequence of tokens (0-25)."""
        length = random.randint(min_len, max_len)
        return [random.randint(0, 25) for _ in range(length)]

    def generate_example(self):
        """Generate one training example."""
        sequence = self._generate_random_sequence()
        target_position = random.randint(0, len(sequence) - 1)
        target_token = sequence[target_position]
        labels = self._get_reflex_labels(target_token)

        return {
            "sequence": sequence,
            "target_position": target_position,
            "target_token": target_token,
            "reflex_labels": labels,
            "label_indices": [self.REFLEX_LABELS.index(l) for l in labels],
            "multihot": self._labels_to_multihot(labels)
        }

    def generate_dataset(self, n_examples=10000, val_split=0.1):
        """Generate full dataset with train/val split."""
        all_examples = [self.generate_example() for _ in range(n_examples)]

        n_val = int(n_examples * val_split)
        val_data = all_examples[:n_val]
        train_data = all_examples[n_val:]

        return train_data, val_data

    def generate_held_out_validation(self, n_per_token=50, held_out_tokens=None):
        """
        Generate validation set with held-out tokens.
        Tests generalization to unseen token values.
        """
        if held_out_tokens is None:
            held_out_tokens = [21, 22, 23, 24, 25]

        val_set = []
        for token in held_out_tokens:
            for _ in range(n_per_token):
                sequence = self._generate_random_sequence()
                target_position = random.randint(0, len(sequence) - 1)
                sequence[target_position] = token
                labels = self._get_reflex_labels(token)

                val_set.append({
                    "sequence": sequence,
                    "target_position": target_position,
                    "target_token": token,
                    "reflex_labels": labels,
                    "label_indices": [self.REFLEX_LABELS.index(l) for l in labels],
                    "multihot": self._labels_to_multihot(labels),
                    "held_out": True
                })

        return val_set

    def generate_reflex_balanced_validation(self, n_per_reflex=100):
        """Generate validation set balanced by reflex type."""
        val_set = []

        for reflex_label in self.REFLEX_LABELS:
            for _ in range(n_per_reflex):
                # Find token that triggers this reflex
                while True:
                    token = random.randint(0, 25)
                    if reflex_label in self._get_reflex_labels(token):
                        break

                sequence = self._generate_random_sequence()
                target_position = random.randint(0, len(sequence) - 1)
                sequence[target_position] = token
                labels = self._get_reflex_labels(token)

                val_set.append({
                    "sequence": sequence,
                    "target_position": target_position,
                    "target_token": token,
                    "reflex_labels": labels,
                    "label_indices": [self.REFLEX_LABELS.index(l) for l in labels],
                    "multihot": self._labels_to_multihot(labels),
                    "tested_reflex": reflex_label
                })

        return val_set

    def save_dataset(self, train_data, val_data, output_dir="data"):
        """Save dataset to JSON files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with open(output_path / "phase1_train.json", "w") as f:
            json.dump(train_data, f)

        with open(output_path / "phase1_val.json", "w") as f:
            json.dump(val_data, f)

        print(f"Saved {len(train_data)} training examples")
        print(f"Saved {len(val_data)} validation examples")

    def print_statistics(self, data, name="Dataset"):
        """Print dataset statistics."""
        print(f"\n{name} Statistics:")
        print(f"  Total examples: {len(data)}")

        # Count reflexes
        reflex_counts = {label: 0 for label in self.REFLEX_LABELS}
        for example in data:
            for label in example["reflex_labels"]:
                reflex_counts[label] += 1

        print("  Reflex distribution:")
        for label, count in reflex_counts.items():
            pct = 100 * count / len(data)
            print(f"    {label}: {count} ({pct:.1f}%)")

        # Sequence length distribution
        lengths = [len(ex["sequence"]) for ex in data]
        avg_len = sum(lengths) / len(lengths)
        print(f"  Sequence length: min={min(lengths)}, max={max(lengths)}, avg={avg_len:.1f}")


def main():
    """Generate and save Phase 1 dataset."""
    print("Phase 1: Reflexes Data Generator")
    print("=" * 40)

    gen = Phase1ReflexDataGenerator(seed=42)

    # Generate main dataset
    train_data, val_data = gen.generate_dataset(n_examples=10000, val_split=0.1)

    # Generate special validation sets
    held_out_val = gen.generate_held_out_validation(n_per_token=50)
    balanced_val = gen.generate_reflex_balanced_validation(n_per_reflex=100)

    # Print statistics
    gen.print_statistics(train_data, "Training")
    gen.print_statistics(val_data, "Validation")
    gen.print_statistics(held_out_val, "Held-out Tokens")
    gen.print_statistics(balanced_val, "Reflex-balanced")

    # Save to files
    output_dir = Path(__file__).parent / "data"
    gen.save_dataset(train_data, val_data, output_dir)

    with open(output_dir / "phase1_held_out.json", "w") as f:
        json.dump(held_out_val, f)

    with open(output_dir / "phase1_balanced.json", "w") as f:
        json.dump(balanced_val, f)

    print(f"\nAll data saved to {output_dir}/")

    # Print sample example
    print("\nSample example:")
    ex = train_data[0]
    print(f"  Sequence: {ex['sequence']}")
    print(f"  Target position: {ex['target_position']}")
    print(f"  Target token: {ex['target_token']}")
    print(f"  Reflexes: {ex['reflex_labels']}")
    print(f"  Multihot: {ex['multihot']}")


if __name__ == "__main__":
    main()
