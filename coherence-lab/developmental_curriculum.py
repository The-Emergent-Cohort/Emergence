"""
Developmental Curriculum - Years 0, 1 & 2
From Numbers to Sequences to Agents

Year 0: Quantitative Primitives (Number Sense)
Year 1: Sensorimotor Foundations
Year 2: Relational & Physical Understanding
"""

import random
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field


# =============================================================================
# YEAR 0: QUANTITATIVE PRIMITIVES (Number Sense)
# The substrate of all abstract reasoning - DeepSeek
# =============================================================================

def gen_successor(vocab_size: int) -> Dict:
    """[n] → n+1 - The foundation of counting. What comes after?"""
    n = random.randint(0, vocab_size - 2)
    return {'sequence': [n], 'target': n + 1}


def gen_predecessor(vocab_size: int) -> Dict:
    """[n] → n-1 - What comes before?"""
    n = random.randint(1, vocab_size - 1)
    return {'sequence': [n], 'target': n - 1}


def gen_count_sequence(vocab_size: int) -> Dict:
    """[a, b, c, ...] → length - How many elements? (Cardinality)"""
    length = random.randint(2, min(6, vocab_size - 1))
    # Use distinct random values so it's clearly "count items" not "value"
    seq = random.sample(range(vocab_size), length)
    return {'sequence': seq, 'target': length}


def gen_successor_chain(vocab_size: int) -> Dict:
    """[1, 2, 3] → 4 - Counting up (successor applied repeatedly)."""
    length = random.randint(2, 4)
    start = random.randint(0, vocab_size - length - 2)
    seq = [start + i for i in range(length)]
    return {'sequence': seq, 'target': start + length}


def gen_predecessor_chain(vocab_size: int) -> Dict:
    """[5, 4, 3] → 2 - Counting down (predecessor applied repeatedly)."""
    length = random.randint(2, 4)
    start = random.randint(length + 1, vocab_size - 1)
    seq = [start - i for i in range(length)]
    target = start - length
    if target < 0:
        return gen_predecessor_chain(vocab_size)
    return {'sequence': seq, 'target': target}


# Classroom-grounded patterns (self-referential math)
# 3 students, 4 entities total, when 1 leads → 2 participate

def gen_remainder_from_group(vocab_size: int) -> Dict:
    """[total, active] → remaining - If 1 leads from 3, 2 remain."""
    total = random.randint(2, min(6, vocab_size - 1))
    active = random.randint(1, total - 1)
    remaining = total - active
    return {'sequence': [total, active], 'target': remaining}


def gen_group_minus_one(vocab_size: int) -> Dict:
    """[n] → n-1 - From a group, one steps out. (Grounded predecessor)"""
    n = random.randint(2, min(6, vocab_size - 1))
    return {'sequence': [n], 'target': n - 1}


def gen_group_plus_one(vocab_size: int) -> Dict:
    """[n] → n+1 - One joins the group. (Grounded successor)"""
    n = random.randint(1, min(5, vocab_size - 2))
    return {'sequence': [n], 'target': n + 1}


def gen_greater_than(vocab_size: int) -> Dict:
    """[a, b] → 1 if a > b else 0 - Comparison as boolean."""
    a = random.randint(0, vocab_size - 1)
    b = random.randint(0, vocab_size - 1)
    while a == b:
        b = random.randint(0, vocab_size - 1)
    target = 1 if a > b else 0
    return {'sequence': [a, b], 'target': target}


def gen_less_than(vocab_size: int) -> Dict:
    """[a, b] → 1 if a < b else 0 - Comparison as boolean."""
    a = random.randint(0, vocab_size - 1)
    b = random.randint(0, vocab_size - 1)
    while a == b:
        b = random.randint(0, vocab_size - 1)
    target = 1 if a < b else 0
    return {'sequence': [a, b], 'target': target}


def gen_missing_addend(vocab_size: int) -> Dict:
    """[a, c] → b where a + b = c - Find the missing piece."""
    # a + ? = c, so ? = c - a
    a = random.randint(0, vocab_size // 2 - 1)
    b = random.randint(1, vocab_size // 2 - 1)  # The missing piece (not 0)
    c = a + b
    if c >= vocab_size:
        return gen_missing_addend(vocab_size)
    return {'sequence': [a, c], 'target': b}


def gen_double(vocab_size: int) -> Dict:
    """[n] → 2n - Doubling (foundation for multiplication)."""
    n = random.randint(1, vocab_size // 2 - 1)
    return {'sequence': [n], 'target': n * 2}


def gen_half(vocab_size: int) -> Dict:
    """[n] → n/2 - Halving (for even numbers)."""
    n = random.choice([i for i in range(2, vocab_size) if i % 2 == 0])
    return {'sequence': [n], 'target': n // 2}


# =============================================================================
# YEAR 1: SENSORIMOTOR FOUNDATIONS
# =============================================================================

def gen_constant(vocab_size: int) -> Dict:
    """[5, 5, 5, ?] → 5 - Things can stay the same."""
    a = random.randint(0, vocab_size - 1)
    length = random.randint(3, 6)
    return {'sequence': [a] * length, 'target': a}


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
# TRAP PATTERNS (Test overconfidence vs true understanding)
# These patterns look like familiar patterns but have different answers
# =============================================================================

def gen_trap_alternating(vocab_size: int) -> Dict:
    """[A, B, A, B, C] → ? - Looks alternating but ends with surprise element."""
    a, b, c = random.sample(range(vocab_size), 3)
    # Build alternating pattern then break it
    seq = [a, b, a, b, c]
    # Target continues the NEW pattern (c was introduced, what comes next?)
    # Most likely interpretation: sequence restarted, so next is a
    # But we'll make target = a (continuing from position) to test if they notice the break
    target = a  # Position 5 in alternating would be a
    return {'sequence': seq, 'target': target}


def gen_trap_increment(vocab_size: int) -> Dict:
    """[1, 2, 3, 4, 2] → ? - Incrementing that suddenly breaks."""
    length = 4
    start = random.randint(1, vocab_size // 2)
    seq = [start + i for i in range(length)]
    # Add a break element
    break_val = random.randint(0, start)
    seq.append(break_val)
    # Target: if they see "break means restart", answer would be break_val + 1
    # If they see "noise", answer would continue increment
    # We'll make target = break_val + 1 (new sequence started)
    target = break_val + 1
    if target >= vocab_size:
        return gen_trap_increment(vocab_size)
    return {'sequence': seq, 'target': target}


def gen_trap_constant(vocab_size: int) -> Dict:
    """[5, 5, 5, 5, 3] → ? - Constant that breaks at end."""
    val = random.randint(1, vocab_size - 2)
    break_val = random.randint(0, vocab_size - 1)
    while break_val == val:
        break_val = random.randint(0, vocab_size - 1)
    seq = [val, val, val, val, break_val]
    # Target: the new value (treating break as signal of new constant)
    target = break_val
    return {'sequence': seq, 'target': target}


# =============================================================================
# BASIC ARITHMETIC (The fundamentals we never explicitly taught!)
# =============================================================================

def gen_add_two(vocab_size: int) -> Dict:
    """[a, b] → a + b - Basic addition: 1 + 1 = 2."""
    # Keep numbers small so sum stays in vocab
    max_val = vocab_size // 2 - 1
    a = random.randint(0, max_val)
    b = random.randint(0, max_val)
    target = a + b
    if target >= vocab_size:
        return gen_add_two(vocab_size)
    return {'sequence': [a, b], 'target': target}


def gen_subtract_two(vocab_size: int) -> Dict:
    """[a, b] → a - b - Basic subtraction (a >= b for positive result)."""
    a = random.randint(1, vocab_size - 1)
    b = random.randint(0, a)  # b <= a ensures non-negative result
    target = a - b
    return {'sequence': [a, b], 'target': target}


def gen_compare_larger(vocab_size: int) -> Dict:
    """[a, b] → max(a, b) - Which number is larger?"""
    a = random.randint(0, vocab_size - 1)
    b = random.randint(0, vocab_size - 1)
    while a == b:  # Make them different so there's a clear answer
        b = random.randint(0, vocab_size - 1)
    target = max(a, b)
    return {'sequence': [a, b], 'target': target}


def gen_compare_smaller(vocab_size: int) -> Dict:
    """[a, b] → min(a, b) - Which number is smaller?"""
    a = random.randint(0, vocab_size - 1)
    b = random.randint(0, vocab_size - 1)
    while a == b:
        b = random.randint(0, vocab_size - 1)
    target = min(a, b)
    return {'sequence': [a, b], 'target': target}


def gen_add_three(vocab_size: int) -> Dict:
    """[a, b, c] → a + b + c - Add three numbers."""
    max_val = vocab_size // 3 - 1
    a = random.randint(0, max_val)
    b = random.randint(0, max_val)
    c = random.randint(0, max_val)
    target = a + b + c
    if target >= vocab_size:
        return gen_add_three(vocab_size)
    return {'sequence': [a, b, c], 'target': target}


def gen_multiply_two(vocab_size: int) -> Dict:
    """[a, b] → a * b - Basic multiplication."""
    # Keep numbers small so product stays in vocab
    a = random.randint(1, 5)
    b = random.randint(1, 5)
    target = a * b
    if target >= vocab_size:
        return gen_multiply_two(vocab_size)
    return {'sequence': [a, b], 'target': target}


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


# Year 0: Quantitative Primitives - THE FOUNDATION OF EVERYTHING
YEAR_0_PATTERNS = [
    # 0A: Successor & Predecessor (What comes next/before?)
    PatternType('successor', gen_successor, 1, 0, '0A', 'n → n+1'),
    PatternType('predecessor', gen_predecessor, 1, 0, '0A', 'n → n-1'),
    PatternType('successor_chain', gen_successor_chain, 2, 0, '0A', 'Counting up'),
    PatternType('predecessor_chain', gen_predecessor_chain, 2, 0, '0A', 'Counting down'),

    # 0B: Quantity & Comparison
    PatternType('count_sequence', gen_count_sequence, 2, 0, '0B', 'How many elements?'),
    PatternType('greater_than', gen_greater_than, 1, 0, '0B', 'a > b?'),
    PatternType('less_than', gen_less_than, 1, 0, '0B', 'a < b?'),

    # 0C: Basic Operations
    PatternType('double', gen_double, 2, 0, '0C', 'n → 2n'),
    PatternType('half', gen_half, 2, 0, '0C', 'n → n/2'),
    PatternType('missing_addend', gen_missing_addend, 3, 0, '0C', 'a + ? = c'),

    # 0D: Grounded Group Math (classroom context: 3 students, 4 entities)
    PatternType('remainder_from_group', gen_remainder_from_group, 2, 0, '0D', 'Group - active = remaining'),
    PatternType('group_minus_one', gen_group_minus_one, 1, 0, '0D', 'One leaves the group'),
    PatternType('group_plus_one', gen_group_plus_one, 1, 0, '0D', 'One joins the group'),
]


YEAR_1_PATTERNS = [
    # 1A: Constancy
    PatternType('constant', gen_constant, 1, 1, '1A', 'Things stay the same'),

    # 1B: Repetition & Memory
    PatternType('repeating', gen_repeating, 1, 1, '1B', 'Remember what was seen'),
    PatternType('echo', gen_echo, 2, 1, '1B', 'Pattern with gaps'),

    # 1C: Alternation & Position
    PatternType('alternating', gen_alternating, 3, 1, '1C', 'Two-element cycle'),
    PatternType('ternary_cycle', gen_ternary_cycle, 4, 1, '1C', 'Three-element cycle'),

    # 1D: Linear Change
    PatternType('incrementing', gen_incrementing, 2, 1, '1D', 'Count up by 1'),
    PatternType('decrementing', gen_decrementing, 2, 1, '1D', 'Count down by 1'),

    # 1E: Rate of Change
    PatternType('fixed_offset', gen_fixed_offset, 3, 1, '1E', 'Count by fixed step'),
    PatternType('variable_step', gen_variable_step, 4, 1, '1E', 'Increasing step size'),

    # 1F: Trap Patterns (test overconfidence)
    PatternType('trap_alternating', gen_trap_alternating, 5, 1, '1F', 'Alternating with surprise'),
    PatternType('trap_increment', gen_trap_increment, 5, 1, '1F', 'Increment with break'),
    PatternType('trap_constant', gen_trap_constant, 4, 1, '1F', 'Constant with break'),

    # 1G: Basic Arithmetic (the fundamentals!)
    PatternType('add_two', gen_add_two, 1, 1, '1G', 'Add two numbers'),
    PatternType('subtract_two', gen_subtract_two, 2, 1, '1G', 'Subtract two numbers'),
    PatternType('compare_larger', gen_compare_larger, 1, 1, '1G', 'Find the larger number'),
    PatternType('compare_smaller', gen_compare_smaller, 1, 1, '1G', 'Find the smaller number'),
    PatternType('add_three', gen_add_three, 2, 1, '1G', 'Add three numbers'),
    PatternType('multiply_two', gen_multiply_two, 3, 1, '1G', 'Multiply two numbers'),
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

ALL_PATTERNS = YEAR_0_PATTERNS + YEAR_1_PATTERNS + YEAR_2_PATTERNS


def get_patterns_by_year(year: int) -> List[PatternType]:
    """Get patterns for a specific year (0, 1, or 2)."""
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
    print("DEVELOPMENTAL CURRICULUM - Years 0, 1 & 2")
    print("=" * 70)

    year_names = {
        0: 'QUANTITATIVE PRIMITIVES (Number Sense)',
        1: 'SENSORIMOTOR FOUNDATIONS',
        2: 'RELATIONAL & PHYSICAL'
    }

    for year in [0, 1, 2]:
        year_patterns = get_patterns_by_year(year)
        print(f"\n{'='*50}")
        print(f"YEAR {year}: {year_names[year]}")
        print(f"{'='*50}")

        sections = sorted(set(p.section for p in year_patterns))
        for section in sections:
            section_patterns = [p for p in year_patterns if p.section == section]
            print(f"\n  Section {section}:")
            for p in section_patterns:
                print(f"    - {p.name} (difficulty {p.difficulty}): {p.description}")

    print(f"\n{'='*70}")
    print(f"Total: {len(ALL_PATTERNS)} pattern types")
    print(f"  Year 0: {len(YEAR_0_PATTERNS)} patterns")
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
