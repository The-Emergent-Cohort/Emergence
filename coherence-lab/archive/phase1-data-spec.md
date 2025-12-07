# Phase 1: Reflexes Data Generator Specification

*Version 0.1 - Dec 7, 2024*

**Purpose:** Establish baseline frozen-layer architecture through trivially learnable deterministic patterns

---

## 1. Input Format

**Sequence Structure:**
- Variable-length sequences of 5-20 tokens
- Token vocabulary: integers 0-25 (representing A-Z)
- No special tokens in Phase 1

**Example input:**
```
[3, 0, 15, 19, 4]  # D, A, P, T, E
```

**Training example structure:**
```python
{
    "sequence": [3, 0, 15, 19, 4],
    "target_position": 1,
    "reflex_label": "VOWEL_MATCH"
}
```

---

## 2. Output Format

**Reflex Label Vocabulary (6 categories):**

| Label | Meaning | Trigger |
|-------|---------|---------|
| `VOWEL_MATCH` | Token is a vowel | token in {0,4,8,14,20} |
| `CONSONANT_MATCH` | Token is a consonant | token not in vowels |
| `EVEN_VALUE` | Token is even | token % 2 == 0 |
| `ODD_VALUE` | Token is odd | token % 2 == 1 |
| `HIGH_HALF` | Token >= 13 | token >= 13 |
| `LOW_HALF` | Token < 13 | token < 13 |

Labels are non-exclusive (token 4 is VOWEL_MATCH and EVEN_VALUE and LOW_HALF).
Use multi-label classification.

---

## 3. Reflex Patterns

Each reflex is a **pure function** of the target token's value. No context dependency.

```
Reflex 1: VOWEL_MATCH    → token in {0, 4, 8, 14, 20}
Reflex 2: CONSONANT_MATCH → token not in vowels
Reflex 3: EVEN_VALUE     → token % 2 == 0
Reflex 4: ODD_VALUE      → token % 2 == 1
Reflex 5: HIGH_HALF      → token >= 13
Reflex 6: LOW_HALF       → token < 13
```

**Why these patterns?**
- Token-level (no context) = reflexes, not reasoning
- Orthogonal dimensions = multiple valid outputs per token
- Numerically simple = trivial to learn

---

## 4. Python Generator

```python
import random
import numpy as np

class Phase1ReflexDataGenerator:
    VOWELS = {0, 4, 8, 14, 20}
    REFLEX_LABELS = [
        "VOWEL_MATCH", "CONSONANT_MATCH",
        "EVEN_VALUE", "ODD_VALUE",
        "HIGH_HALF", "LOW_HALF"
    ]

    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)

    def _get_reflex_labels(self, token):
        """Pure function: token → set of applicable reflexes."""
        labels = []

        if token in self.VOWELS:
            labels.append("VOWEL_MATCH")
        else:
            labels.append("CONSONANT_MATCH")

        if token % 2 == 0:
            labels.append("EVEN_VALUE")
        else:
            labels.append("ODD_VALUE")

        if token >= 13:
            labels.append("HIGH_HALF")
        else:
            labels.append("LOW_HALF")

        return labels

    def _generate_random_sequence(self, length=None):
        if length is None:
            length = random.randint(5, 20)
        return [random.randint(0, 25) for _ in range(length)]

    def generate_example(self):
        sequence = self._generate_random_sequence()
        target_position = random.randint(0, len(sequence) - 1)
        target_token = sequence[target_position]
        labels = self._get_reflex_labels(target_token)

        return {
            "sequence": sequence,
            "target_position": target_position,
            "target_token": target_token,
            "reflex_labels": labels,
            "label_indices": [
                self.REFLEX_LABELS.index(label) for label in labels
            ]
        }

    def generate_dataset(self, n_examples=10000, test_split=0.1):
        all_examples = [self.generate_example() for _ in range(n_examples)]
        n_val = int(n_examples * test_split)
        return all_examples[n_val:], all_examples[:n_val]
```

---

## 5. Validation Strategy

**Goal:** Ensure model learned patterns, not memorized tokens.

1. **Held-out token range:**
   - Train on tokens 0-20
   - Validate on tokens 21-25 (unseen)

2. **Reflex-balanced validation:**
   - 100-200 examples per reflex label
   - Each reflex tested across multiple tokens

3. **Success criteria:**
   - >= 98% accuracy per reflex
   - Generalization within 0.5% of seen tokens
   - Similar accuracy across all sequences

---

## 6. Dataset Properties

| Property | Value |
|----------|-------|
| Size | 10,000 examples |
| Determinism | 100% |
| Sequence length | 5-20 tokens |
| Token distribution | Uniform 0-25 |
| Train/val split | 90/10 |
| Random seed | 42 |

---

## 7. Expected Learning Curve

```
Epoch 1:  ~16% (random baseline)
Epoch 5:  ~85% (learning patterns)
Epoch 10: ~98%+ (plateau)
Val:      ~97%+ (generalization holds)
```

---

## 8. Implementation Notes

- Multi-label: Use BCEWithLogitsLoss
- Batch by sequence length for efficient padding
- Track accuracy per reflex separately
- Early stop when val plateaus 2+ epochs

---

**Status:** Ready for implementation.
