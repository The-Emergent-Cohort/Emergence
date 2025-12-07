"""
Phase 1b: Shape/Structure Invariants Data Generator
Coherence Lab - Emergence Project

Teaches categorical matching: "like goes with like"
Based on infant shape-sorting development (4-12 months).
"""

import json
import random
from pathlib import Path


class Phase1bDataGenerator:
    """Generate same/different classification examples."""

    # Category definitions
    VOWELS = {0, 4, 8, 14, 20}
    EVENS = {i for i in range(26) if i % 2 == 0}
    ODDS = {i for i in range(26) if i % 2 == 1}
    LOW = {i for i in range(13)}
    HIGH = {i for i in range(13, 26)}
    DIV3 = {i for i in range(26) if i % 3 == 0}

    CATEGORIES = {
        'vowel': VOWELS,
        'consonant': set(range(26)) - VOWELS,
        'even': EVENS,
        'odd': ODDS,
        'low': LOW,
        'high': HIGH,
        'div3': DIV3,
    }

    def __init__(self, seed=42):
        random.seed(seed)

    def _get_categories(self, token):
        """Return all categories a token belongs to."""
        cats = []
        for name, members in self.CATEGORIES.items():
            if token in members:
                cats.append(name)
        return cats

    def _share_category(self, t1, t2):
        """Check if two tokens share at least one category."""
        cats1 = set(self._get_categories(t1))
        cats2 = set(self._get_categories(t2))
        shared = cats1 & cats2
        return len(shared) > 0, list(shared)

    def generate_same_pair(self):
        """Generate a pair of tokens that share a category."""
        # Pick a category
        cat_name = random.choice(list(self.CATEGORIES.keys()))
        members = list(self.CATEGORIES[cat_name])

        # Pick two different members
        t1 = random.choice(members)
        t2 = random.choice(members)
        while t2 == t1 and len(members) > 1:
            t2 = random.choice(members)

        _, shared = self._share_category(t1, t2)

        return {
            'tokens': [t1, t2],
            'label': 1,  # SAME
            'label_name': 'SAME',
            'shared_categories': shared,
            'primary_category': cat_name
        }

    def generate_different_pair(self):
        """Generate a pair of tokens that share NO category."""
        # This is tricky - most pairs share at least even/odd
        # We need to find genuinely different pairs

        attempts = 0
        while attempts < 100:
            t1 = random.randint(0, 25)
            t2 = random.randint(0, 25)

            if t1 == t2:
                continue

            shares, shared = self._share_category(t1, t2)
            if not shares:
                return {
                    'tokens': [t1, t2],
                    'label': 0,  # DIFFERENT
                    'label_name': 'DIFFERENT',
                    'shared_categories': [],
                    'primary_category': None
                }
            attempts += 1

        # If we can't find a truly different pair, use "weakly different"
        # (shares only one category, the least informative one)
        t1 = random.randint(0, 25)
        t2 = random.randint(0, 25)
        while t1 == t2:
            t2 = random.randint(0, 25)

        _, shared = self._share_category(t1, t2)
        return {
            'tokens': [t1, t2],
            'label': 0,  # DIFFERENT (weakly)
            'label_name': 'DIFFERENT',
            'shared_categories': shared,  # May have some overlap
            'primary_category': None,
            'note': 'weak_different'
        }

    def generate_example(self):
        """Generate a random same or different example."""
        if random.random() < 0.5:
            return self.generate_same_pair()
        else:
            return self.generate_different_pair()

    def generate_dataset(self, n_examples=50000, val_split=0.1):
        """Generate balanced dataset."""
        examples = []

        # Generate balanced same/different
        n_same = n_examples // 2
        n_diff = n_examples - n_same

        for _ in range(n_same):
            examples.append(self.generate_same_pair())

        for _ in range(n_diff):
            examples.append(self.generate_different_pair())

        random.shuffle(examples)

        n_val = int(n_examples * val_split)
        return examples[n_val:], examples[:n_val]

    def generate_transfer_test(self, n_examples=1000):
        """
        Generate position-based same/different for transfer testing.
        Can the model recognize 'same position parity' without training?
        """
        examples = []

        for _ in range(n_examples):
            # Generate positions instead of tokens
            pos1 = random.randint(0, 19)
            pos2 = random.randint(0, 19)
            while pos2 == pos1:
                pos2 = random.randint(0, 19)

            # Same if both even or both odd positions
            same_parity = (pos1 % 2) == (pos2 % 2)

            examples.append({
                'positions': [pos1, pos2],
                'label': 1 if same_parity else 0,
                'label_name': 'SAME' if same_parity else 'DIFFERENT',
                'reason': 'position_parity'
            })

        return examples


def main():
    """Generate Phase 1b datasets."""
    print("Phase 1b: Shape/Structure Invariants Data Generator")
    print("=" * 55)

    generator = Phase1bDataGenerator(seed=42)

    # Generate datasets
    train_data, val_data = generator.generate_dataset(n_examples=50000)
    transfer_test = generator.generate_transfer_test(n_examples=1000)

    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    print(f"Transfer test examples: {len(transfer_test)}")

    # Show label distribution
    train_same = sum(1 for ex in train_data if ex['label'] == 1)
    train_diff = len(train_data) - train_same
    print(f"\nLabel distribution (train):")
    print(f"  SAME: {train_same} ({100*train_same/len(train_data):.1f}%)")
    print(f"  DIFFERENT: {train_diff} ({100*train_diff/len(train_data):.1f}%)")

    # Show category distribution for SAME examples
    cat_counts = {}
    for ex in train_data:
        if ex['label'] == 1:
            cat = ex.get('primary_category', 'unknown')
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

    print(f"\nCategory distribution (SAME examples):")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count}")

    # Show examples
    print("\nSample examples:")
    for i in range(5):
        ex = train_data[i]
        print(f"  {ex['tokens']} → {ex['label_name']} (shared: {ex['shared_categories']})")

    # Save datasets
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    with open(data_dir / "phase1b_train.json", "w") as f:
        json.dump(train_data, f)
    print(f"\nSaved: {data_dir / 'phase1b_train.json'}")

    with open(data_dir / "phase1b_val.json", "w") as f:
        json.dump(val_data, f)
    print(f"Saved: {data_dir / 'phase1b_val.json'}")

    with open(data_dir / "phase1b_transfer.json", "w") as f:
        json.dump(transfer_test, f)
    print(f"Saved: {data_dir / 'phase1b_transfer.json'}")


if __name__ == "__main__":
    main()
