"""
Developmental Curriculum - Phase 1 & 2
From Numbers to Relations

Phase 1 (Year 1): Foundational Numeracy - Following Singapore Math CPA approach
    1A: Number Operations - counting, +1, -1 (Concrete operations FIRST)
    1B: Constancy - things stay the same
    1C: Repetition & Memory - remember what was seen
    1D: Alternation & Position - cycles and turns
    1E: Linear Change - counting sequences (+1, -1)
    1F: Rate of Change - skip counting (fixed step)

Phase 2 (Year 2): Relational & Physical Understanding
    2A: Simple Relations - doubling, halving, offsets
    2B: Analogies - transfer relationships
    2C: Physical Motion - velocity, acceleration
    2D: Physical Interaction - bounce, conservation
    2E: Causality - if-then, cause-effect
"""

import random
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field


# =============================================================================
# PHASE 1A: NUMBER OPERATIONS (Foundational)
# Following Singapore Math CPA - Concrete operations BEFORE abstract patterns
# =============================================================================

def gen_counting(vocab_size: int) -> Dict:
    """[0, 1, 2, 3, ?] → 4 - The number line itself."""
    length = random.randint(3, 6)
    start = 0  # Always start from 0 for counting
    seq = [start + i for i in range(length)]
    target = start + length
    if target >= vocab_size:
        return gen_counting(vocab_size)
    return {'sequence': seq, 'target': target}


def gen_add_one(vocab_size: int) -> Dict:
    """[3, 4, 7, ?] → 8 - See +1 pattern, then apply it."""
    # Always show the operation first (worked example), then ask to apply
    # This makes it unambiguous: "I saw +1, so I do +1"
    n = random.randint(0, vocab_size - 2)
    # Show 2-3 examples of +1, then the query
    n_examples = random.randint(2, 3)
    seq = []
    for _ in range(n_examples):
        prev = random.randint(0, vocab_size - 3)
        seq.extend([prev, prev + 1])
    seq.append(n)
    target = n + 1
    return {'sequence': seq, 'target': target}


def gen_subtract_one(vocab_size: int) -> Dict:
    """[9, 8, 5, ?] → 4 - See -1 pattern, then apply it."""
    # Always show the operation first (worked example), then ask to apply
    n = random.randint(1, vocab_size - 1)
    # Show 2-3 examples of -1, then the query
    n_examples = random.randint(2, 3)
    seq = []
    for _ in range(n_examples):
        prev = random.randint(2, vocab_size - 1)
        seq.extend([prev, prev - 1])
    seq.append(n)
    target = n - 1
    return {'sequence': seq, 'target': target}


# =============================================================================
# PHASE 1B: CONSTANCY & STABILITY
# =============================================================================

def gen_constant(vocab_size: int) -> Dict:
    """[5, 5, 5, ?] → 5 - Things can stay the same."""
    a = random.randint(0, vocab_size - 1)
    length = random.randint(3, 6)
    return {'sequence': [a] * length, 'target': a}


# =============================================================================
# PHASE 1C: REPETITION & MEMORY
# =============================================================================

def gen_repeating(vocab_size: int) -> Dict:
    """[3, 3, 3, 3, ?] → 3 - Remember what was seen."""
    return gen_constant(vocab_size)  # Same as constant for now


def gen_echo(vocab_size: int) -> Dict:
    """[7, 0, 7, 0, ?] → 7 - Pattern with gaps."""
    a = random.randint(1, vocab_size - 1)  # Non-zero
    length = random.randint(2, 4)
    seq = []
    for _ in range(length):
        seq.extend([a, 0])
    # Remove last 0, that's the gap before answer
    seq = seq[:-1]
    target = a if len(seq) % 2 == 0 else 0
    return {'sequence': seq, 'target': target}


# =============================================================================
# PHASE 1D: ALTERNATION & POSITION
# =============================================================================

def gen_alternating(vocab_size: int) -> Dict:
    """[A, B, A, B, ?] → A - Two-element cycle."""
    a, b = random.sample(range(vocab_size), 2)
    length = random.randint(4, 8)
    seq = [a if i % 2 == 0 else b for i in range(length)]
    target = a if length % 2 == 0 else b
    return {'sequence': seq, 'target': target}


def gen_ternary_cycle(vocab_size: int) -> Dict:
    """[A, B, C, A, B, ?] → C - Three-element cycle."""
    a, b, c = random.sample(range(vocab_size), 3)
    cycle = [a, b, c]
    length = random.randint(5, 9)
    seq = [cycle[i % 3] for i in range(length)]
    target = cycle[length % 3]
    return {'sequence': seq, 'target': target}


# =============================================================================
# PHASE 1E: LINEAR CHANGE (Counting Sequences)
# =============================================================================

def gen_incrementing(vocab_size: int) -> Dict:
    """[1, 2, 3, ?] → 4 - Count up by 1."""
    length = random.randint(3, 6)
    start = random.randint(0, max(0, vocab_size - length - 2))
    seq = [start + i for i in range(length)]
    target = start + length
    if target >= vocab_size:
        return gen_incrementing(vocab_size)
    return {'sequence': seq, 'target': target}


def gen_decrementing(vocab_size: int) -> Dict:
    """[9, 8, 7, ?] → 6 - Count down by 1."""
    length = random.randint(3, 6)
    start = random.randint(length + 1, vocab_size - 1)
    seq = [start - i for i in range(length)]
    target = start - length
    if target < 0:
        return gen_decrementing(vocab_size)
    return {'sequence': seq, 'target': target}


# =============================================================================
# PHASE 1F: RATE OF CHANGE (Skip Counting)
# =============================================================================

def gen_fixed_offset(vocab_size: int) -> Dict:
    """[2, 5, 8, ?] → 11 - Count by fixed step."""
    length = random.randint(3, 5)
    step = random.randint(2, 4)
    start = random.randint(0, max(0, vocab_size - step * (length + 1)))
    seq = [start + i * step for i in range(length)]
    target = start + length * step
    if target >= vocab_size:
        return gen_fixed_offset(vocab_size)
    return {'sequence': seq, 'target': target}


def gen_variable_step(vocab_size: int) -> Dict:
    """[1, 2, 4, 7, ?] → 11 - Step increases by 1 each time."""
    length = random.randint(3, 5)
    start = random.randint(0, 5)
    seq = [start]
    step = 1
    for _ in range(length - 1):
        seq.append(seq[-1] + step)
        step += 1
    target = seq[-1] + step
    if target >= vocab_size or any(x >= vocab_size for x in seq):
        return gen_variable_step(vocab_size)
    return {'sequence': seq, 'target': target}


# =============================================================================
# YEAR 2: RELATIONAL & PHYSICAL
# =============================================================================

def gen_double_each(vocab_size: int) -> Dict:
    """[2, 4, 8, ?] → 16 - Each element is double previous."""
    length = random.randint(3, 4)
    start = random.randint(1, 3)
    seq = [start]
    for _ in range(length - 1):
        seq.append(seq[-1] * 2)
    target = seq[-1] * 2
    if target >= vocab_size or any(x >= vocab_size for x in seq):
        return gen_double_each(vocab_size)
    return {'sequence': seq, 'target': target}


def gen_half_each(vocab_size: int) -> Dict:
    """[16, 8, 4, ?] → 2 - Each element is half previous."""
    length = random.randint(3, 4)
    # Start with a power of 2
    start = random.choice([8, 16, 24])
    if start >= vocab_size:
        start = 16
    seq = [start]
    for _ in range(length - 1):
        seq.append(seq[-1] // 2)
    target = seq[-1] // 2
    if target < 0:
        return gen_half_each(vocab_size)
    return {'sequence': seq, 'target': target}


def gen_offset_from_first(vocab_size: int) -> Dict:
    """[5, 7, 9, 11, ?] → 13 - All relative to first element."""
    base = random.randint(2, 10)
    step = random.randint(1, 3)
    length = random.randint(3, 5)
    seq = [base + i * step for i in range(length)]
    target = base + length * step
    if target >= vocab_size:
        return gen_offset_from_first(vocab_size)
    return {'sequence': seq, 'target': target}


def gen_analogy_simple(vocab_size: int) -> Dict:
    """[A, A+2, B, ?] → B+2 - Transfer relationship."""
    offset = random.randint(1, 4)
    a = random.randint(0, vocab_size // 2 - offset)
    b = random.randint(0, vocab_size // 2 - offset)
    while b == a:
        b = random.randint(0, vocab_size // 2 - offset)
    seq = [a, a + offset, b]
    target = b + offset
    if target >= vocab_size:
        return gen_analogy_simple(vocab_size)
    return {'sequence': seq, 'target': target}


def gen_analogy_ratio(vocab_size: int) -> Dict:
    """[2, 6, 3, ?] → 9 - Multiplicative relationship."""
    multiplier = random.randint(2, 3)
    a = random.randint(1, 5)
    b = random.randint(1, 5)
    while b == a:
        b = random.randint(1, 5)
    seq = [a, a * multiplier, b]
    target = b * multiplier
    if target >= vocab_size:
        return gen_analogy_ratio(vocab_size)
    return {'sequence': seq, 'target': target}


def gen_constant_velocity(vocab_size: int) -> Dict:
    """Position with constant velocity: [0, 2, 4, ?] → 6."""
    velocity = random.randint(1, 3)
    start = random.randint(0, 5)
    length = random.randint(3, 5)
    seq = [start + i * velocity for i in range(length)]
    target = start + length * velocity
    if target >= vocab_size:
        return gen_constant_velocity(vocab_size)
    return {'sequence': seq, 'target': target}


def gen_acceleration(vocab_size: int) -> Dict:
    """Position with acceleration: [0, 1, 4, 9, ?] → 16 (quadratic)."""
    length = random.randint(3, 5)
    start = random.randint(0, 3)
    # Positions: start, start+1, start+4, start+9, ... (perfect squares)
    seq = [start + i * i for i in range(length)]
    target = start + length * length
    if target >= vocab_size or any(x >= vocab_size for x in seq):
        return gen_acceleration(vocab_size)
    return {'sequence': seq, 'target': target}


def gen_deceleration(vocab_size: int) -> Dict:
    """Slowing down: decreasing steps."""
    start = random.randint(15, vocab_size - 1)
    steps = [4, 3, 2, 1]  # Decreasing steps
    seq = [start]
    for i, step in enumerate(steps[:random.randint(2, 3)]):
        seq.append(seq[-1] - step)
        if seq[-1] < 0:
            return gen_deceleration(vocab_size)
    # Next step continues the pattern
    last_step = start - seq[-1] if len(seq) > 1 else 3
    next_step = max(0, last_step - 1)
    target = seq[-1] - next_step
    if target < 0:
        target = seq[-1]  # Stopped
    return {'sequence': seq, 'target': target}


def gen_bounce(vocab_size: int) -> Dict:
    """Bouncing ball: [10, 7, 5, 4, ?] → 3 (damped oscillation)."""
    peak = random.randint(10, 15)
    decay = 0.7  # Each bounce is 70% of previous
    length = random.randint(3, 5)
    seq = [peak]
    current = peak
    for _ in range(length - 1):
        current = int(current * decay)
        seq.append(current)
    target = int(seq[-1] * decay)
    return {'sequence': seq, 'target': target}


def gen_conservation(vocab_size: int) -> Dict:
    """Total stays same: [5, 3, 7, 1, ?] where pairs sum to 8."""
    total = random.randint(6, 12)
    length = random.randint(3, 5)
    seq = []
    for _ in range(length):
        val = random.randint(0, min(total, vocab_size - 1))
        seq.append(val)
    # Target: what makes the sum pattern continue?
    # For now: next element continues alternating with complement
    if len(seq) >= 2:
        target = total - seq[-1]
    else:
        target = total - seq[0]
    target = max(0, min(target, vocab_size - 1))
    return {'sequence': seq, 'target': target}


def gen_if_then(vocab_size: int) -> Dict:
    """[trigger, result, other, other, trigger, ?] → result."""
    trigger = random.randint(1, vocab_size // 3)
    result = random.randint(vocab_size // 3, 2 * vocab_size // 3)
    other1 = random.randint(0, vocab_size - 1)
    other2 = random.randint(0, vocab_size - 1)
    while other1 == trigger or other1 == result:
        other1 = random.randint(0, vocab_size - 1)
    while other2 == trigger or other2 == result:
        other2 = random.randint(0, vocab_size - 1)
    seq = [trigger, result, other1, other2, trigger]
    target = result
    return {'sequence': seq, 'target': target}


def gen_cause_effect(vocab_size: int) -> Dict:
    """Cause precedes effect with delay: [C, _, E, C, ?] → _ then E."""
    cause = random.randint(1, vocab_size // 3)
    effect = random.randint(vocab_size // 2, vocab_size - 1)
    filler = random.randint(vocab_size // 3, vocab_size // 2)
    while filler == cause or filler == effect:
        filler = random.randint(vocab_size // 3, vocab_size // 2)
    seq = [cause, filler, effect, cause]
    target = filler  # The delay element
    return {'sequence': seq, 'target': target}


# =============================================================================
# CURRICULUM STRUCTURE
# =============================================================================

@dataclass
class PatternType:
    name: str
    generator: Callable
    difficulty: int  # 1-10
    year: int
    section: str
    description: str


@dataclass
class PlaydaySpec:
    """
    Specification for section-specific playday activities and awards.

    Playdays are curriculum-aware celebrations that combine:
    - Fun activities appropriate for current skill level
    - Star awards recognizing different strengths
    - PARTY TIME when the whole class shines
    """
    section: str
    activities: List[str]  # Activity types for this section
    focus_skills: List[str]  # Skills being celebrated
    star_categories: Dict[str, str] = field(default_factory=dict)  # category -> description

    def __post_init__(self):
        # Default star categories if not specified
        if not self.star_categories:
            self.star_categories = {
                'accuracy': '⭐ Getting answers right',
                'patience': '⭐ Taking time to think',
                'curiosity': '⭐ Trying new things',
                'creativity': '⭐ Finding new patterns'
            }


# Playday specs for each section
PLAYDAY_SPECS = {
    # =================================================================
    # Phase 1: Foundational Numeracy
    # =================================================================
    '1A': PlaydaySpec(
        section='1A',
        activities=['count_together', 'whats_next', 'whats_before'],
        focus_skills=['counting', 'add_one', 'subtract_one'],
        star_categories={
            'accuracy': '⭐ Counting correctly',
            'speed': '⭐ Quick with numbers',
            'curiosity': '⭐ Finding number neighbors',
            'confidence': '⭐ Trying without hints'
        }
    ),
    '1B': PlaydaySpec(
        section='1B',
        activities=['spot_the_same', 'steady_eddie'],
        focus_skills=['constancy', 'attention'],
        star_categories={
            'accuracy': '⭐ Seeing what stays the same',
            'patience': '⭐ Watching carefully',
            'curiosity': '⭐ Looking for patterns',
            'focus': '⭐ Staying on track'
        }
    ),
    '1C': PlaydaySpec(
        section='1C',
        activities=['memory_game', 'echo_back'],
        focus_skills=['repetition', 'memory'],
        star_categories={
            'accuracy': '⭐ Remembering well',
            'patience': '⭐ Taking turns nicely',
            'curiosity': '⭐ Finding echoes',
            'memory': '⭐ Long memory chain'
        }
    ),
    '1D': PlaydaySpec(
        section='1D',
        activities=['turn_taking', 'rhythm_game'],
        focus_skills=['alternation', 'position'],
        star_categories={
            'accuracy': '⭐ Knowing your turn',
            'rhythm': '⭐ Feeling the beat',
            'teamwork': '⭐ Working together',
            'creativity': '⭐ Making new rhythms'
        }
    ),
    '1E': PlaydaySpec(
        section='1E',
        activities=['count_up', 'count_down', 'number_line'],
        focus_skills=['incrementing', 'decrementing'],
        star_categories={
            'accuracy': '⭐ Counting correctly',
            'speed': '⭐ Quick counting',
            'backwards': '⭐ Counting backwards',
            'creativity': '⭐ Number patterns'
        }
    ),
    '1F': PlaydaySpec(
        section='1F',
        activities=['skip_count', 'stair_climb', 'rocket_launch'],
        focus_skills=['fixed_offset', 'variable_step'],
        star_categories={
            'accuracy': '⭐ Perfect steps',
            'speed': '⭐ Fast climber',
            'pattern': '⭐ Finding the step',
            'creativity': '⭐ Making new steps'
        }
    ),
    '2A': PlaydaySpec(
        section='2A',
        activities=['double_trouble', 'half_time', 'relative_race'],
        focus_skills=['doubling', 'halving', 'relations'],
        star_categories={
            'accuracy': '⭐ Perfect relations',
            'speed': '⭐ Quick thinker',
            'insight': '⭐ Seeing connections',
            'creativity': '⭐ New relations'
        }
    ),
    '2B': PlaydaySpec(
        section='2B',
        activities=['analogy_hunt', 'pattern_transfer'],
        focus_skills=['analogy', 'transfer'],
        star_categories={
            'accuracy': '⭐ Perfect analogies',
            'transfer': '⭐ Quick transfer',
            'insight': '⭐ Deep connections',
            'creativity': '⭐ Novel analogies'
        }
    ),
    '2C': PlaydaySpec(
        section='2C',
        activities=['motion_predict', 'speed_race', 'accel_game'],
        focus_skills=['velocity', 'acceleration'],
        star_categories={
            'accuracy': '⭐ Perfect prediction',
            'physics': '⭐ Physics intuition',
            'speed': '⭐ Quick tracker',
            'creativity': '⭐ New motions'
        }
    ),
    '2D': PlaydaySpec(
        section='2D',
        activities=['bounce_predict', 'conservation_check'],
        focus_skills=['bounce', 'conservation'],
        star_categories={
            'accuracy': '⭐ Perfect physics',
            'intuition': '⭐ Physical intuition',
            'conservation': '⭐ Nothing lost',
            'creativity': '⭐ New interactions'
        }
    ),
    '2E': PlaydaySpec(
        section='2E',
        activities=['cause_hunt', 'if_then_game', 'chain_reaction'],
        focus_skills=['causality', 'conditionals'],
        star_categories={
            'accuracy': '⭐ Perfect causes',
            'logic': '⭐ Clear thinking',
            'prediction': '⭐ What comes next',
            'creativity': '⭐ New causes'
        }
    ),
}


def get_playday_spec(section: str) -> PlaydaySpec:
    """Get playday spec for a section."""
    return PLAYDAY_SPECS.get(section, PLAYDAY_SPECS['1A'])


YEAR_1_PATTERNS = [
    # =================================================================
    # Phase 1: Foundational Numeracy (Singapore Math CPA approach)
    # Concrete operations BEFORE abstract pattern recognition
    # =================================================================

    # 1A: Number Operations - The truly foundational skills
    PatternType('counting', gen_counting, 1, 1, '1A', 'The number line'),
    PatternType('add_one', gen_add_one, 1, 1, '1A', 'What comes next (+1)'),
    PatternType('subtract_one', gen_subtract_one, 1, 1, '1A', 'What comes before (-1)'),

    # 1B: Constancy - First pattern recognition
    PatternType('constant', gen_constant, 1, 1, '1B', 'Things stay the same'),

    # 1C: Repetition & Memory
    PatternType('repeating', gen_repeating, 1, 1, '1C', 'Remember what was seen'),
    PatternType('echo', gen_echo, 2, 1, '1C', 'Pattern with gaps'),

    # 1D: Alternation & Position
    PatternType('alternating', gen_alternating, 3, 1, '1D', 'Two-element cycle'),
    PatternType('ternary_cycle', gen_ternary_cycle, 4, 1, '1D', 'Three-element cycle'),

    # 1E: Linear Change (Counting Sequences)
    PatternType('incrementing', gen_incrementing, 2, 1, '1E', 'Count up by 1'),
    PatternType('decrementing', gen_decrementing, 2, 1, '1E', 'Count down by 1'),

    # 1F: Rate of Change (Skip Counting)
    PatternType('fixed_offset', gen_fixed_offset, 3, 1, '1F', 'Count by fixed step'),
    PatternType('variable_step', gen_variable_step, 4, 1, '1F', 'Increasing step size'),
]

YEAR_2_PATTERNS = [
    # 2A: Simple Relations
    PatternType('double_each', gen_double_each, 4, 2, '2A', 'Doubling sequence'),
    PatternType('half_each', gen_half_each, 4, 2, '2A', 'Halving sequence'),
    PatternType('offset_from_first', gen_offset_from_first, 3, 2, '2A', 'Relative to first'),

    # 2B: Analogies
    PatternType('analogy_simple', gen_analogy_simple, 5, 2, '2B', 'Additive analogy'),
    PatternType('analogy_ratio', gen_analogy_ratio, 6, 2, '2B', 'Multiplicative analogy'),

    # 2C: Physical Motion
    PatternType('constant_velocity', gen_constant_velocity, 3, 2, '2C', 'Linear motion'),
    PatternType('acceleration', gen_acceleration, 5, 2, '2C', 'Quadratic motion'),
    PatternType('deceleration', gen_deceleration, 5, 2, '2C', 'Slowing down'),

    # 2D: Physical Interaction
    PatternType('bounce', gen_bounce, 5, 2, '2D', 'Damped oscillation'),
    PatternType('conservation', gen_conservation, 6, 2, '2D', 'Quantity conservation'),

    # 2E: Causality
    PatternType('if_then', gen_if_then, 5, 2, '2E', 'Trigger-response'),
    PatternType('cause_effect', gen_cause_effect, 6, 2, '2E', 'Delayed causation'),
]

ALL_PATTERNS = YEAR_1_PATTERNS + YEAR_2_PATTERNS


def get_patterns_by_year(year: int) -> List[PatternType]:
    return [p for p in ALL_PATTERNS if p.year == year]


def get_patterns_by_section(section: str) -> List[PatternType]:
    return [p for p in ALL_PATTERNS if p.section == section]


def get_pattern_names(patterns: List[PatternType] = None) -> List[str]:
    patterns = patterns or ALL_PATTERNS
    return [p.name for p in patterns]


# =============================================================================
# DEVELOPMENTAL DATASET
# =============================================================================

class DevelopmentalDataset(Dataset):
    """
    Dataset that generates patterns from the developmental curriculum.

    Can be configured to use:
    - Specific years: year=1, year=2, or year=[1,2]
    - Specific sections: sections=['1A', '1B']
    - Specific patterns: patterns=['constant', 'incrementing']
    """

    def __init__(
        self,
        n_examples: int = 50000,
        vocab_size: int = 26,
        seed: int = None,
        year: int = None,
        sections: List[str] = None,
        patterns: List[str] = None
    ):
        if seed is not None:
            random.seed(seed)

        self.vocab_size = vocab_size

        # Determine which patterns to use
        if patterns:
            self.pattern_types = [p for p in ALL_PATTERNS if p.name in patterns]
        elif sections:
            self.pattern_types = [p for p in ALL_PATTERNS if p.section in sections]
        elif year:
            if isinstance(year, int):
                year = [year]
            self.pattern_types = [p for p in ALL_PATTERNS if p.year in year]
        else:
            self.pattern_types = ALL_PATTERNS

        if not self.pattern_types:
            raise ValueError("No patterns selected!")

        self.pattern_names = [p.name for p in self.pattern_types]

        # Generate examples
        self.examples = []
        for _ in range(n_examples):
            pattern = random.choice(self.pattern_types)
            example = pattern.generator(vocab_size)
            example['pattern_type'] = pattern.name
            example['year'] = pattern.year
            example['section'] = pattern.section
            self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        # Pad sequence to fixed length
        max_len = 12
        seq = ex['sequence']
        padded = seq + [0] * (max_len - len(seq))
        return {
            'sequence': torch.tensor(padded[:max_len], dtype=torch.long),
            'target': torch.tensor(ex['target'], dtype=torch.long),
            'seq_len': len(seq),
            'pattern_type': ex['pattern_type'],
            'year': ex['year'],
            'section': ex['section']
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    return {
        'tokens': torch.stack([b['sequence'] for b in batch]),
        'target': torch.stack([b['target'] for b in batch]),
        'seq_len': [b['seq_len'] for b in batch],
        'pattern_type': [b['pattern_type'] for b in batch],
        'year': [b['year'] for b in batch],
        'section': [b['section'] for b in batch]
    }


# =============================================================================
# CURRICULUM PRINTER
# =============================================================================

def print_curriculum():
    """Print the full curriculum structure."""
    print("=" * 70)
    print("DEVELOPMENTAL CURRICULUM - Years 1 & 2")
    print("=" * 70)

    for year in [1, 2]:
        year_patterns = get_patterns_by_year(year)
        print(f"\n{'='*50}")
        print(f"YEAR {year}: {'SENSORIMOTOR FOUNDATIONS' if year == 1 else 'RELATIONAL & PHYSICAL'}")
        print(f"{'='*50}")

        sections = sorted(set(p.section for p in year_patterns))
        for section in sections:
            section_patterns = [p for p in year_patterns if p.section == section]
            print(f"\n  Section {section}:")
            for p in section_patterns:
                print(f"    - {p.name} (difficulty {p.difficulty}): {p.description}")

    print(f"\n{'='*70}")
    print(f"Total: {len(ALL_PATTERNS)} pattern types")
    print(f"  Year 1: {len(YEAR_1_PATTERNS)} patterns")
    print(f"  Year 2: {len(YEAR_2_PATTERNS)} patterns")


if __name__ == '__main__':
    print_curriculum()

    # Test dataset
    print("\n\nTesting dataset generation...")
    ds = DevelopmentalDataset(n_examples=100, year=1, seed=42)
    print(f"Generated {len(ds)} Year 1 examples")

    # Show some examples
    for i in range(5):
        ex = ds[i]
        seq = ex['sequence'][:ex['seq_len']].tolist()
        print(f"  {ex['pattern_type']}: {seq} → {ex['target'].item()}")
