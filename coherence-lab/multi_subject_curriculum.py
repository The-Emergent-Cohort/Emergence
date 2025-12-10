"""
Multi-Subject Curriculum - Beyond Math

A school day has many subjects. Each can be encoded as learnable patterns
for small transformers. This creates interleaved learning across domains.

Subjects:
    - Reading: Letter sequences, word patterns, phonics
    - Music: Rhythm patterns, melody sequences, harmony
    - Science: Cause-effect, classification, cycles
    - Art: Color mixing, shape sequences, symmetry
    - Social: Turn-taking, sharing, patterns of interaction

Key insight: ALL subjects reduce to pattern completion at some level.
The question is encoding them appropriately for small models.
"""

import random
from typing import Dict, List, Callable
from dataclasses import dataclass


# =============================================================================
# VOCABULARY MAPPINGS
# =============================================================================

# Letters (0-25 = a-z)
LETTERS = list('abcdefghijklmnopqrstuvwxyz')

# Musical notes (simplified - 8 notes in an octave)
NOTES = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C2']  # 0-7

# Colors (primary + secondary + black/white)
COLORS = ['red', 'blue', 'yellow', 'green', 'orange', 'purple', 'black', 'white']  # 0-7

# Rhythm values (note durations)
RHYTHMS = ['whole', 'half', 'quarter', 'eighth', 'rest']  # 0-4

# Weather/Nature states
WEATHER = ['sunny', 'cloudy', 'rainy', 'snowy', 'windy', 'foggy']  # 0-5

# Emotions (for social learning)
EMOTIONS = ['happy', 'sad', 'angry', 'scared', 'surprised', 'calm']  # 0-5


# =============================================================================
# READING CURRICULUM
# =============================================================================

def gen_alphabet_sequence(vocab_size: int = 26) -> Dict:
    """[a, b, c, ?] → d - The alphabet is just incrementing!"""
    length = random.randint(3, 6)
    start = random.randint(0, vocab_size - length - 2)
    seq = [start + i for i in range(length)]
    target = start + length
    return {
        'sequence': seq,
        'target': target,
        'subject': 'reading',
        'skill': 'alphabet_order',
        'human_readable': f"[{', '.join(LETTERS[i] for i in seq)}, ?] → {LETTERS[target]}"
    }


def gen_vowel_consonant(vocab_size: int = 26) -> Dict:
    """[a, b, e, f, i, ?] → j - Vowels and consonants alternate."""
    vowels = [0, 4, 8, 14, 20]  # a, e, i, o, u
    consonants = [i for i in range(26) if i not in vowels]

    length = random.randint(4, 8)
    seq = []
    for i in range(length):
        if i % 2 == 0:
            seq.append(random.choice(vowels))
        else:
            seq.append(random.choice(consonants))

    # Target follows the pattern
    if length % 2 == 0:
        target = random.choice(vowels)
    else:
        target = random.choice(consonants)

    return {
        'sequence': seq,
        'target': target,
        'subject': 'reading',
        'skill': 'vowel_consonant',
        'human_readable': f"[{', '.join(LETTERS[i] for i in seq)}, ?] → {LETTERS[target]}"
    }


def gen_rhyme_pattern(vocab_size: int = 26) -> Dict:
    """[c, a, t, 0, b, a, t, 0, h, a, ?] → t - Word families rhyme!

    Uses 0 as word separator. Words in same family share ending.
    """
    # Common word family endings
    families = [
        ([0, 19], 'at'),   # -at: cat, bat, hat, mat
        ([0, 19], 'an'),   # -an: can, fan, man, pan
        ([8, 19], 'it'),   # -it: bit, fit, hit, sit
        ([14, 19], 'ot'),  # -ot: cot, dot, got, hot
    ]

    ending_indices, ending_name = random.choice(families)

    # Generate 2-3 words with same ending
    n_words = random.randint(2, 3)
    consonants = [2, 3, 5, 7, 10, 12, 13, 15, 17, 18, 22]  # common starting consonants

    seq = []
    for i in range(n_words):
        start = random.choice(consonants)
        seq.append(start)
        seq.extend(ending_indices)
        if i < n_words - 1:
            seq.append(0)  # word separator (using 0/a as separator isn't ideal but works)

    # Query: new word with same family, missing last letter
    query_start = random.choice([c for c in consonants if c != seq[-len(ending_indices)-1]])
    seq.append(0)  # separator
    seq.append(query_start)
    seq.append(ending_indices[0])  # first letter of ending

    target = ending_indices[1]  # last letter of ending

    return {
        'sequence': seq,
        'target': target,
        'subject': 'reading',
        'skill': 'rhyme_families',
        'human_readable': f"Word family: -{ending_name}"
    }


def gen_double_letter(vocab_size: int = 26) -> Dict:
    """[a, a, b, b, c, ?] → c - Doubling pattern."""
    length = random.randint(2, 4)
    letters = random.sample(range(vocab_size), length)

    seq = []
    for letter in letters:
        seq.append(letter)
        seq.append(letter)

    # Remove last one - that's what we predict
    target = seq[-1]
    seq = seq[:-1]

    return {
        'sequence': seq,
        'target': target,
        'subject': 'reading',
        'skill': 'double_letters',
        'human_readable': f"[{', '.join(LETTERS[i] for i in seq)}, ?]"
    }


# =============================================================================
# MUSIC CURRICULUM
# =============================================================================

def gen_scale_up(vocab_size: int = 8) -> Dict:
    """[C, D, E, F, ?] → G - Musical scale ascending."""
    length = random.randint(3, 6)
    start = random.randint(0, vocab_size - length - 2)
    seq = [start + i for i in range(length)]
    target = start + length

    if target >= vocab_size:
        return gen_scale_up(vocab_size)

    return {
        'sequence': seq,
        'target': target,
        'subject': 'music',
        'skill': 'scale_ascending',
        'human_readable': f"[{', '.join(NOTES[i] for i in seq)}, ?] → {NOTES[target]}"
    }


def gen_scale_down(vocab_size: int = 8) -> Dict:
    """[G, F, E, D, ?] → C - Musical scale descending."""
    length = random.randint(3, 6)
    start = random.randint(length, vocab_size - 1)
    seq = [start - i for i in range(length)]
    target = start - length

    if target < 0:
        return gen_scale_down(vocab_size)

    return {
        'sequence': seq,
        'target': target,
        'subject': 'music',
        'skill': 'scale_descending',
        'human_readable': f"[{', '.join(NOTES[i] for i in seq)}, ?] → {NOTES[target]}"
    }


def gen_rhythm_pattern(vocab_size: int = 5) -> Dict:
    """[quarter, quarter, half, quarter, quarter, ?] → half - Rhythm cycles."""
    # Common rhythm patterns (in 4/4 time)
    patterns = [
        [2, 2, 1],           # quarter, quarter, half
        [2, 2, 2, 2],        # four quarters
        [1, 1],              # two halves
        [2, 3, 3, 2],        # quarter, eighth, eighth, quarter
        [3, 3, 3, 3, 2, 2],  # four eighths, two quarters
    ]

    pattern = random.choice(patterns)
    n_repeats = random.randint(2, 3)

    seq = pattern * n_repeats
    seq = seq[:-1]  # Remove last - that's target
    target = pattern[-1]

    return {
        'sequence': seq,
        'target': target,
        'subject': 'music',
        'skill': 'rhythm_patterns',
        'human_readable': f"[{', '.join(RHYTHMS[i] for i in seq)}, ?] → {RHYTHMS[target]}"
    }


def gen_melody_echo(vocab_size: int = 8) -> Dict:
    """[C, E, G, 0, C, E, ?] → G - Melody repeats (echo)."""
    # Generate a short melodic phrase
    phrase_len = random.randint(2, 4)
    phrase = [random.randint(0, vocab_size - 1) for _ in range(phrase_len)]

    # Echo it
    seq = phrase + [0] + phrase[:-1]  # 0 as separator
    target = phrase[-1]

    return {
        'sequence': seq,
        'target': target,
        'subject': 'music',
        'skill': 'melody_echo',
        'human_readable': f"Phrase: [{', '.join(NOTES[i] if i < len(NOTES) else '|' for i in phrase)}]"
    }


def gen_interval_pattern(vocab_size: int = 8) -> Dict:
    """[C, E, D, F, E, ?] → G - Consistent interval (skip)."""
    interval = random.randint(1, 2)  # step of 1 or 2
    length = random.randint(3, 4)
    max_start = vocab_size - 1 - interval * length

    if max_start < 0:
        # Fallback for small vocab
        interval = 1
        length = 3
        max_start = vocab_size - 1 - length

    start = random.randint(0, max(0, max_start))
    seq = [start + i * interval for i in range(length)]
    target = start + length * interval

    if target >= vocab_size:
        target = vocab_size - 1

    return {
        'sequence': seq,
        'target': target,
        'subject': 'music',
        'skill': 'intervals',
        'human_readable': f"Interval of {interval}: [{', '.join(NOTES[i] if i < len(NOTES) else str(i) for i in seq)}, ?]"
    }


# =============================================================================
# SCIENCE CURRICULUM
# =============================================================================

def gen_cause_effect(vocab_size: int = 10) -> Dict:
    """[rain, wet, sun, dry, rain, ?] → wet - Cause leads to effect."""
    # Cause-effect pairs (using indices)
    pairs = [
        (0, 1),  # rain → wet
        (2, 3),  # sun → dry
        (4, 5),  # cold → freeze
        (6, 7),  # heat → melt
        (8, 9),  # plant → grow
    ]

    # Show 2-3 examples, then query
    n_examples = random.randint(2, 3)
    selected_pairs = random.sample(pairs, min(n_examples + 1, len(pairs)))

    seq = []
    for cause, effect in selected_pairs[:-1]:
        seq.extend([cause, effect])

    # Query pair
    query_cause, query_effect = selected_pairs[-1]
    seq.append(query_cause)
    target = query_effect

    return {
        'sequence': seq,
        'target': target,
        'subject': 'science',
        'skill': 'cause_effect',
        'human_readable': f"Cause-effect pattern"
    }


def gen_life_cycle(vocab_size: int = 6) -> Dict:
    """[egg, caterpillar, chrysalis, butterfly, egg, ?] → caterpillar - Cycles!"""
    # Life cycles (using indices 0-5)
    cycles = [
        [0, 1, 2, 3],  # butterfly: egg, caterpillar, chrysalis, butterfly
        [0, 1, 2],     # frog: egg, tadpole, frog
        [0, 1, 2, 3],  # plant: seed, sprout, plant, flower
    ]

    cycle = random.choice(cycles)
    n_repeats = random.randint(1, 2)

    seq = []
    for _ in range(n_repeats):
        seq.extend(cycle)

    # Add partial cycle for query
    partial_len = random.randint(1, len(cycle) - 1)
    seq.extend(cycle[:partial_len])
    target = cycle[partial_len % len(cycle)]

    return {
        'sequence': seq,
        'target': target,
        'subject': 'science',
        'skill': 'life_cycles',
        'human_readable': f"Life cycle with {len(cycle)} stages"
    }


def gen_classification(vocab_size: int = 12) -> Dict:
    """[dog, cat, bird, 0, apple, banana, orange, 0, dog, cat, ?] → bird

    Things in the same category follow a pattern.
    """
    # Categories
    categories = [
        [0, 1, 2],     # animals: dog, cat, bird
        [3, 4, 5],     # fruits: apple, banana, orange
        [6, 7, 8],     # vehicles: car, bus, train
        [9, 10, 11],   # colors: red, blue, green
    ]

    # Show 1-2 complete categories, then partial query
    n_categories = random.randint(1, 2)
    selected = random.sample(categories, n_categories + 1)

    seq = []
    for cat in selected[:-1]:
        seq.extend(cat)
        seq.append(0)  # category separator

    # Partial query category
    query_cat = selected[-1]
    partial_len = random.randint(1, len(query_cat) - 1)
    seq.extend(query_cat[:partial_len])
    target = query_cat[partial_len]

    return {
        'sequence': seq,
        'target': target,
        'subject': 'science',
        'skill': 'classification',
        'human_readable': f"Category completion"
    }


def gen_growth_sequence(vocab_size: int = 10) -> Dict:
    """[1, 2, 4, ?] → 8 - Exponential growth (like cells dividing)."""
    # This is actually doubling - biological growth!
    start = 1
    length = random.randint(3, 5)

    seq = [start]
    for _ in range(length - 1):
        seq.append(seq[-1] * 2)

    target = seq[-1] * 2

    if target >= vocab_size or any(x >= vocab_size for x in seq):
        # Fall back to smaller numbers
        seq = [1, 2, 4]
        target = 8
        if target >= vocab_size:
            seq = [1, 2]
            target = 4

    return {
        'sequence': seq,
        'target': target,
        'subject': 'science',
        'skill': 'growth_patterns',
        'human_readable': f"Doubling: [{', '.join(str(x) for x in seq)}, ?] → {target}"
    }


# =============================================================================
# ART CURRICULUM
# =============================================================================

def gen_color_mixing(vocab_size: int = 8) -> Dict:
    """[red, yellow, orange, blue, yellow, ?] → green - Color theory!

    Primary + Primary = Secondary:
    - red(0) + yellow(2) = orange(4)
    - blue(1) + yellow(2) = green(3)
    - red(0) + blue(1) = purple(5)
    """
    # Mixing rules: (primary1, primary2) → secondary
    rules = [
        (0, 2, 4),  # red + yellow = orange
        (1, 2, 3),  # blue + yellow = green
        (0, 1, 5),  # red + blue = purple
    ]

    # Show 1-2 examples, then query
    n_examples = random.randint(1, 2)
    selected = random.sample(rules, min(n_examples + 1, len(rules)))

    seq = []
    for p1, p2, result in selected[:-1]:
        seq.extend([p1, p2, result])

    # Query
    p1, p2, result = selected[-1]
    seq.extend([p1, p2])
    target = result

    return {
        'sequence': seq,
        'target': target,
        'subject': 'art',
        'skill': 'color_mixing',
        'human_readable': f"Color mixing"
    }


def gen_symmetry(vocab_size: int = 8) -> Dict:
    """[1, 2, 3, 2, ?] → 1 - Mirror symmetry!"""
    half_len = random.randint(2, 4)
    half = [random.randint(0, vocab_size - 1) for _ in range(half_len)]

    # Create symmetric sequence, missing last element
    full = half + half[-2::-1]  # mirror without repeating middle
    target = full[-1]
    seq = full[:-1]

    return {
        'sequence': seq,
        'target': target,
        'subject': 'art',
        'skill': 'symmetry',
        'human_readable': f"Symmetry pattern"
    }


def gen_pattern_repeat(vocab_size: int = 8) -> Dict:
    """[red, blue, red, blue, red, ?] → blue - Visual pattern repetition."""
    pattern_len = random.randint(2, 4)
    pattern = [random.randint(0, vocab_size - 1) for _ in range(pattern_len)]

    n_repeats = random.randint(2, 3)
    seq = pattern * n_repeats
    seq = seq[:-1]
    target = pattern[(len(seq)) % len(pattern)]

    return {
        'sequence': seq,
        'target': target,
        'subject': 'art',
        'skill': 'pattern_repeat',
        'human_readable': f"Visual pattern: {[COLORS[i] if i < len(COLORS) else i for i in pattern]}"
    }


def gen_gradient(vocab_size: int = 8) -> Dict:
    """[0, 1, 2, 3, ?] → 4 - Color gradient (light to dark)."""
    # This is essentially incrementing but in color space
    length = random.randint(3, 5)
    start = random.randint(0, vocab_size - length - 2)
    seq = [start + i for i in range(length)]
    target = start + length

    if target >= vocab_size:
        return gen_gradient(vocab_size)

    return {
        'sequence': seq,
        'target': target,
        'subject': 'art',
        'skill': 'gradient',
        'human_readable': f"Gradient: light → dark"
    }


# =============================================================================
# SOCIAL CURRICULUM
# =============================================================================

def gen_turn_taking(vocab_size: int = 4) -> Dict:
    """[A, B, C, A, B, ?] → C - Taking turns!"""
    n_participants = random.randint(2, 4)
    participants = list(range(n_participants))

    n_rounds = random.randint(2, 3)
    seq = participants * n_rounds
    seq = seq[:-1]
    target = participants[(len(seq)) % n_participants]

    return {
        'sequence': seq,
        'target': target,
        'subject': 'social',
        'skill': 'turn_taking',
        'human_readable': f"{n_participants} friends taking turns"
    }


def gen_sharing(vocab_size: int = 10) -> Dict:
    """[6, 3, 3, 0, 8, 4, ?] → 4 - Fair sharing (division)!

    Total, share1, share2 pattern where shares are equal.
    """
    total = random.choice([4, 6, 8, 10])  # Even numbers for fair sharing
    share = total // 2

    # Show 1-2 examples
    n_examples = random.randint(1, 2)
    seq = []
    for _ in range(n_examples):
        t = random.choice([4, 6, 8, 10])
        s = t // 2
        seq.extend([t, s, s, 0])

    # Query
    seq.extend([total, share])
    target = share

    return {
        'sequence': seq,
        'target': target,
        'subject': 'social',
        'skill': 'fair_sharing',
        'human_readable': f"Sharing {total} cookies equally"
    }


def gen_emotion_response(vocab_size: int = 6) -> Dict:
    """[gift, happy, loss, sad, hug, ?] → happy - Emotional cause-effect."""
    # Situation → Emotion pairs
    pairs = [
        (0, 1),  # gift → happy
        (2, 3),  # loss → sad
        (4, 1),  # hug → happy
        (5, 3),  # alone → sad
    ]

    n_examples = random.randint(2, 3)
    selected = random.sample(pairs, min(n_examples + 1, len(pairs)))

    seq = []
    for situation, emotion in selected[:-1]:
        seq.extend([situation, emotion])

    query_situation, query_emotion = selected[-1]
    seq.append(query_situation)
    target = query_emotion

    return {
        'sequence': seq,
        'target': target,
        'subject': 'social',
        'skill': 'emotional_intelligence',
        'human_readable': f"Emotion understanding"
    }


# =============================================================================
# CURRICULUM REGISTRY
# =============================================================================

@dataclass
class SubjectPattern:
    name: str
    generator: Callable
    subject: str
    vocab_size: int
    difficulty: int  # 1-10
    description: str


READING_PATTERNS = [
    SubjectPattern('alphabet_sequence', gen_alphabet_sequence, 'reading', 26, 1, "ABC order"),
    SubjectPattern('vowel_consonant', gen_vowel_consonant, 'reading', 26, 3, "Vowels vs consonants"),
    SubjectPattern('rhyme_pattern', gen_rhyme_pattern, 'reading', 26, 4, "Word families"),
    SubjectPattern('double_letter', gen_double_letter, 'reading', 26, 2, "Letter doubling"),
]

MUSIC_PATTERNS = [
    SubjectPattern('scale_up', gen_scale_up, 'music', 8, 1, "Ascending scale"),
    SubjectPattern('scale_down', gen_scale_down, 'music', 8, 1, "Descending scale"),
    SubjectPattern('rhythm_pattern', gen_rhythm_pattern, 'music', 5, 3, "Rhythm cycles"),
    SubjectPattern('melody_echo', gen_melody_echo, 'music', 8, 3, "Melody repetition"),
    SubjectPattern('interval_pattern', gen_interval_pattern, 'music', 8, 4, "Musical intervals"),
]

SCIENCE_PATTERNS = [
    SubjectPattern('cause_effect', gen_cause_effect, 'science', 10, 3, "Cause and effect"),
    SubjectPattern('life_cycle', gen_life_cycle, 'science', 6, 4, "Life cycles"),
    SubjectPattern('classification', gen_classification, 'science', 12, 4, "Categorization"),
    SubjectPattern('growth_sequence', gen_growth_sequence, 'science', 16, 5, "Growth/doubling"),
]

ART_PATTERNS = [
    SubjectPattern('color_mixing', gen_color_mixing, 'art', 8, 3, "Color theory"),
    SubjectPattern('symmetry', gen_symmetry, 'art', 8, 4, "Mirror symmetry"),
    SubjectPattern('pattern_repeat', gen_pattern_repeat, 'art', 8, 2, "Visual patterns"),
    SubjectPattern('gradient', gen_gradient, 'art', 8, 1, "Color gradients"),
]

SOCIAL_PATTERNS = [
    SubjectPattern('turn_taking', gen_turn_taking, 'social', 4, 1, "Taking turns"),
    SubjectPattern('sharing', gen_sharing, 'social', 10, 3, "Fair sharing"),
    SubjectPattern('emotion_response', gen_emotion_response, 'social', 6, 4, "Emotional understanding"),
]

ALL_SUBJECT_PATTERNS = (
    READING_PATTERNS +
    MUSIC_PATTERNS +
    SCIENCE_PATTERNS +
    ART_PATTERNS +
    SOCIAL_PATTERNS
)

SUBJECTS = {
    'reading': READING_PATTERNS,
    'music': MUSIC_PATTERNS,
    'science': SCIENCE_PATTERNS,
    'art': ART_PATTERNS,
    'social': SOCIAL_PATTERNS,
}


def get_patterns_by_subject(subject: str) -> List[SubjectPattern]:
    return SUBJECTS.get(subject, [])


def generate_example(pattern: SubjectPattern) -> Dict:
    """Generate a single example from a pattern."""
    return pattern.generator(pattern.vocab_size)


def generate_school_day_batch(n_per_subject: int = 10) -> List[Dict]:
    """Generate a batch of examples from all subjects - a school day!"""
    batch = []
    for subject, patterns in SUBJECTS.items():
        for _ in range(n_per_subject):
            pattern = random.choice(patterns)
            example = generate_example(pattern)
            example['pattern_name'] = pattern.name
            batch.append(example)

    random.shuffle(batch)  # Interleave subjects
    return batch


# =============================================================================
# CURRICULUM SUMMARY
# =============================================================================

def print_curriculum():
    """Print the multi-subject curriculum."""
    print("=" * 70)
    print("MULTI-SUBJECT CURRICULUM")
    print("=" * 70)

    for subject, patterns in SUBJECTS.items():
        print(f"\n{subject.upper()}")
        print("-" * 40)
        for p in patterns:
            print(f"  {p.name}: {p.description} (difficulty {p.difficulty}, vocab {p.vocab_size})")

    print(f"\n{'=' * 70}")
    print(f"Total: {len(ALL_SUBJECT_PATTERNS)} patterns across {len(SUBJECTS)} subjects")


# =============================================================================
# DEMO
# =============================================================================

if __name__ == '__main__':
    print_curriculum()

    print("\n\n=== SAMPLE SCHOOL DAY ===\n")

    # Generate samples from each subject
    for subject, patterns in SUBJECTS.items():
        print(f"\n--- {subject.upper()} ---")
        for pattern in patterns[:2]:  # Show 2 examples per subject
            example = generate_example(pattern)
            print(f"\n  {pattern.name}:")
            print(f"    Sequence: {example['sequence']}")
            print(f"    Target: {example['target']}")
            if 'human_readable' in example:
                print(f"    Human: {example['human_readable']}")

    print("\n\n=== INTERLEAVED BATCH (School Day) ===\n")
    batch = generate_school_day_batch(n_per_subject=2)
    for i, example in enumerate(batch[:10]):
        print(f"{i+1}. [{example['subject']}] {example['pattern_name']}: {example['sequence']} → {example['target']}")
