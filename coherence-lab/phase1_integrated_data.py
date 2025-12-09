"""
Integrated Phase 1: Multi-Task Foundation Data Generator
Coherence Lab - Emergence Project

Generates data for simultaneous training on:
1. Token properties (reflexes)
2. Token relations (same category)
3. Position properties (even/odd position)
4. Position relations (same parity)
"""

import json
import random
from pathlib import Path
from itertools import combinations


class IntegratedPhase1DataGenerator:
    """Generate multi-task training examples."""

    # Token categories
    VOWELS = {0, 4, 8, 14, 20}
    EVENS = {i for i in range(26) if i % 2 == 0}
    LOW = {i for i in range(13)}

    def __init__(self, seed=42, max_seq_len=12):
        random.seed(seed)
        self.max_seq_len = max_seq_len

    def _get_token_labels(self, token):
        """6-way multi-label for token properties."""
        return [
            1 if token in self.VOWELS else 0,      # vowel
            1 if token not in self.VOWELS else 0,  # consonant
            1 if token % 2 == 0 else 0,            # even
            1 if token % 2 == 1 else 0,            # odd
            1 if token >= 13 else 0,               # high
            1 if token < 13 else 0,                # low
        ]

    def _get_position_labels(self, pos):
        """4-way multi-label for position properties."""
        return [
            1 if pos % 2 == 0 else 0,  # even_position
            1 if pos % 2 == 1 else 0,  # odd_position
            1 if pos < 6 else 0,       # early (first half)
            1 if pos >= 6 else 0,      # late (second half)
        ]

    def _tokens_share_category(self, t1, t2):
        """Check if tokens share any category."""
        # Check vowel/consonant
        if (t1 in self.VOWELS) == (t2 in self.VOWELS):
            return True
        # Check even/odd
        if (t1 % 2) == (t2 % 2):
            return True
        # Check high/low
        if (t1 >= 13) == (t2 >= 13):
            return True
        return False

    def _positions_same_parity(self, p1, p2):
        """Check if positions have same parity."""
        return (p1 % 2) == (p2 % 2)

    def generate_example(self):
        """Generate one multi-task training example."""
        # Random sequence length
        seq_len = random.randint(6, self.max_seq_len)

        # Generate random sequence
        sequence = [random.randint(0, 25) for _ in range(seq_len)]

        # Task 1: Token properties for each position
        token_properties = []
        for pos, token in enumerate(sequence):
            token_properties.append({
                'pos': pos,
                'token': token,
                'labels': self._get_token_labels(token)
            })

        # Task 2: Token relations (sample pairs)
        token_relations = []
        positions = list(range(seq_len))
        # Sample up to 4 pairs
        if len(positions) >= 2:
            pairs = list(combinations(positions, 2))
            sampled_pairs = random.sample(pairs, min(4, len(pairs)))
            for p1, p2 in sampled_pairs:
                same = 1 if self._tokens_share_category(sequence[p1], sequence[p2]) else 0
                token_relations.append({
                    'pos1': p1,
                    'pos2': p2,
                    'tokens': [sequence[p1], sequence[p2]],
                    'same': same
                })

        # Task 3: Position properties
        position_properties = []
        for pos in range(seq_len):
            position_properties.append({
                'pos': pos,
                'labels': self._get_position_labels(pos)
            })

        # Task 4: Position relations (sample pairs)
        position_relations = []
        if len(positions) >= 2:
            pairs = list(combinations(positions, 2))
            sampled_pairs = random.sample(pairs, min(4, len(pairs)))
            for p1, p2 in sampled_pairs:
                same = 1 if self._positions_same_parity(p1, p2) else 0
                position_relations.append({
                    'pos1': p1,
                    'pos2': p2,
                    'same': same
                })

        return {
            'sequence': sequence,
            'seq_len': seq_len,
            'tasks': {
                'token_properties': token_properties,
                'token_relations': token_relations,
                'position_properties': position_properties,
                'position_relations': position_relations
            }
        }

    def generate_dataset(self, n_examples=100000, val_split=0.1):
        """Generate full dataset."""
        examples = [self.generate_example() for _ in range(n_examples)]
        random.shuffle(examples)
        n_val = int(n_examples * val_split)
        return examples[n_val:], examples[:n_val]

    def get_stats(self, examples):
        """Get statistics about the dataset."""
        stats = {
            'n_examples': len(examples),
            'avg_seq_len': 0,
            'token_relation_balance': {'same': 0, 'diff': 0},
            'position_relation_balance': {'same': 0, 'diff': 0}
        }

        total_len = 0
        for ex in examples:
            total_len += ex['seq_len']
            for tr in ex['tasks']['token_relations']:
                if tr['same']:
                    stats['token_relation_balance']['same'] += 1
                else:
                    stats['token_relation_balance']['diff'] += 1
            for pr in ex['tasks']['position_relations']:
                if pr['same']:
                    stats['position_relation_balance']['same'] += 1
                else:
                    stats['position_relation_balance']['diff'] += 1

        stats['avg_seq_len'] = total_len / len(examples)
        return stats


def main():
    print("Integrated Phase 1: Multi-Task Data Generator")
    print("=" * 55)

    generator = IntegratedPhase1DataGenerator(seed=42)

    # Generate datasets
    train_data, val_data = generator.generate_dataset(n_examples=100000)

    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")

    # Stats
    train_stats = generator.get_stats(train_data)
    print(f"\nAverage sequence length: {train_stats['avg_seq_len']:.1f}")
    print(f"Token relation balance: {train_stats['token_relation_balance']}")
    print(f"Position relation balance: {train_stats['position_relation_balance']}")

    # Show example
    print("\nSample example:")
    ex = train_data[0]
    print(f"  Sequence: {ex['sequence']}")
    print(f"  Token properties (first 2):")
    for tp in ex['tasks']['token_properties'][:2]:
        print(f"    pos {tp['pos']}, token {tp['token']}: {tp['labels']}")
    print(f"  Token relations:")
    for tr in ex['tasks']['token_relations'][:2]:
        print(f"    pos {tr['pos1']} vs {tr['pos2']}: {'SAME' if tr['same'] else 'DIFF'}")
    print(f"  Position relations:")
    for pr in ex['tasks']['position_relations'][:2]:
        print(f"    pos {pr['pos1']} vs {pr['pos2']}: {'SAME' if pr['same'] else 'DIFF'}")

    # Save
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    with open(data_dir / "phase1_integrated_train.json", "w") as f:
        json.dump(train_data, f)
    print(f"\nSaved: {data_dir / 'phase1_integrated_train.json'}")

    with open(data_dir / "phase1_integrated_val.json", "w") as f:
        json.dump(val_data, f)
    print(f"Saved: {data_dir / 'phase1_integrated_val.json'}")


if __name__ == "__main__":
    main()
