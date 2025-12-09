"""
Developmental Curriculum - Years 1 & 2
From Sequences to Agents

Year 1: Sensorimotor Foundations
Year 2: Relational & Physical Understanding
"""

import random
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field


# =============================================================================
# YEAR 1: SENSORIMOTOR FOUNDATIONS
# =============================================================================

def gen_constant(vocab_size: int) -> Dict:
    """[5, 5, 5, ?] → 5 - Things can stay the same."""
    a = random.randint(0, vocab_size - 1)
    length = random.randint(3, 6)
    return {'sequence': [a] * length, 'target': a}


# =============================================================================
# 1A': QUANTITY AWARENESS (Conservation & Cardinality)
# These patterns teach that quantity is a stable property of sequences
# Critical foundation for symbolic math - quantity exists independent of order
# =============================================================================

def gen_sequence_length(vocab_size: int) -> Dict:
    """[A, B, C, D] → 4 - Count elements (cardinality)."""
    length = random.randint(2, min(8, vocab_size - 1))
    # Generate distinct or repeated elements (doesn't matter - count is count)
    seq = [random.randint(0, vocab_size - 1) for _ in range(length)]
    return {'sequence': seq, 'target': length}


def gen_count_value(vocab_size: int) -> Dict:
    """[5, 3, 5, 5, 2] → 3 - How many times does first element appear?"""
    length = random.randint(4, 8)
    target_val = random.randint(1, vocab_size - 1)
    count = random.randint(2, min(length - 1, vocab_size - 1))

    # Build sequence with exactly 'count' occurrences of target_val
    seq = [target_val] * count
    # Fill rest with other values
    for _ in range(length - count):
        other = random.randint(0, vocab_size - 1)
        while other == target_val:
            other = random.randint(0, vocab_size - 1)
        seq.append(other)

    random.shuffle(seq)
    # Put target_val first so "count of first element" is the task
    if seq[0] != target_val:
        idx = seq.index(target_val)
        seq[0], seq[idx] = seq[idx], seq[0]

    return {'sequence': seq, 'target': count}


def gen_distinct_count(vocab_size: int) -> Dict:
    """[A, A, B, A, C] → 3 - How many distinct values?"""
    n_distinct = random.randint(2, min(5, vocab_size - 1))
    length = random.randint(n_distinct + 1, n_distinct + 4)

    # Pick n_distinct unique values
    values = random.sample(range(vocab_size), n_distinct)

    # Build sequence ensuring all values appear at least once
    seq = values.copy()
    for _ in range(length - n_distinct):
        seq.append(random.choice(values))

    random.shuffle(seq)
    return {'sequence': seq, 'target': n_distinct}


def gen_conservation_shuffle(vocab_size: int) -> Dict:
    """[C, A, B] → 3 - Length is invariant to order (conservation of quantity)."""
    # Same as sequence_length but explicitly framed as conservation
    # The key insight: no matter how elements are arranged, count stays same
    length = random.randint(3, min(7, vocab_size - 1))
    values = [random.randint(0, vocab_size - 1) for _ in range(length)]
    random.shuffle(values)  # Explicit shuffle to emphasize order doesn't matter
    return {'sequence': values, 'target': length}


# =============================================================================
# 1E': SYMBOLIC PROPERTY RECOGNITION (Bridge to math operations)
# These patterns teach recognizing and labeling sequence properties
# Critical for understanding math: [2,4,6,8] is "step 2" not just "complex sequence"
# =============================================================================

def gen_compute_step(vocab_size: int) -> Dict:
    """[2, 5, 8, 11] → 3 - Identify the additive step between elements."""
    length = random.randint(3, 5)
    # Ensure step doesn't make sequence exceed vocab
    max_step = max(1, (vocab_size - 1) // length)
    step = random.randint(1, min(5, max_step))
    max_start = max(0, vocab_size - step * length - 1)
    start = random.randint(0, max_start)
    seq = [start + i * step for i in range(length)]
    return {'sequence': seq, 'target': step}


def gen_compute_first_diff(vocab_size: int) -> Dict:
    """[3, 7, 11, 15] → 4 - What's seq[1] - seq[0]? (explicit difference)."""
    length = random.randint(3, 5)
    # Ensure step doesn't make sequence exceed vocab
    max_step = max(1, (vocab_size - 1) // length)
    step = random.randint(1, min(8, max_step))
    max_start = max(0, vocab_size - step * length - 1)
    start = random.randint(0, max_start)
    seq = [start + i * step for i in range(length)]
    # Target is the first difference (seq[1] - seq[0])
    return {'sequence': seq, 'target': step}


def gen_compute_ratio(vocab_size: int) -> Dict:
    """[2, 4, 8, 16] → 2 - Identify the multiplicative ratio."""
    ratio = random.randint(2, 3)  # Keep ratios small to stay in vocab
    length = random.randint(3, 4)
    start = random.randint(1, 3)
    seq = [start]
    for _ in range(length - 1):
        next_val = seq[-1] * ratio
        if next_val >= vocab_size:
            # Sequence would exceed vocab, regenerate
            return gen_compute_ratio(vocab_size)
        seq.append(next_val)
    return {'sequence': seq, 'target': ratio}


def gen_is_constant(vocab_size: int) -> Dict:
    """[5, 5, 5, 5] → 1 (yes) or [5, 6, 5, 5] → 0 (no) - Is sequence constant?"""
    length = random.randint(4, 6)
    is_constant = random.choice([True, False])

    if is_constant:
        val = random.randint(0, vocab_size - 1)
        seq = [val] * length
        target = 1
    else:
        val = random.randint(0, vocab_size - 1)
        seq = [val] * length
        # Change one element
        change_idx = random.randint(0, length - 1)
        seq[change_idx] = (val + random.randint(1, 5)) % vocab_size
        target = 0

    return {'sequence': seq, 'target': target}


def gen_is_increasing(vocab_size: int) -> Dict:
    """[1, 2, 3, 4] → 1 (yes) or [1, 3, 2, 4] → 0 (no) - Is sequence strictly increasing?"""
    length = random.randint(4, 6)
    is_increasing = random.choice([True, False])

    if is_increasing:
        # Generate strictly increasing sequence
        start = random.randint(0, vocab_size - length - 1)
        step = random.randint(1, 3)
        seq = [start + i * step for i in range(length)]
        if seq[-1] >= vocab_size:
            return gen_is_increasing(vocab_size)
        target = 1
    else:
        # Generate non-monotonic sequence
        start = random.randint(0, vocab_size - length - 1)
        seq = [start + i for i in range(length)]
        # Swap two elements to break monotonicity
        i, j = random.sample(range(length), 2)
        seq[i], seq[j] = seq[j], seq[i]
        target = 0

    return {'sequence': seq, 'target': target}


def gen_is_decreasing(vocab_size: int) -> Dict:
    """[9, 7, 5, 3] → 1 (yes) or [9, 5, 7, 3] → 0 (no) - Is sequence strictly decreasing?"""
    length = random.randint(4, 6)
    is_decreasing = random.choice([True, False])

    if is_decreasing:
        # Generate strictly decreasing sequence
        start = random.randint(length, vocab_size - 1)
        step = random.randint(1, 2)
        seq = [start - i * step for i in range(length)]
        if seq[-1] < 0:
            return gen_is_decreasing(vocab_size)
        target = 1
    else:
        # Generate non-monotonic sequence
        start = random.randint(length, vocab_size - 1)
        seq = [start - i for i in range(length)]
        # Swap two elements to break monotonicity
        i, j = random.sample(range(length), 2)
        seq[i], seq[j] = seq[j], seq[i]
        target = 0

    return {'sequence': seq, 'target': target}


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
# 1B': POSITION SCAFFOLDING (Bridge to cycles)
# These patterns teach position awareness before full alternating/ternary
# =============================================================================

def gen_simple_alternating(vocab_size: int) -> Dict:
    """[A, 0, A, 0, ?] → A or 0 - Alternating with zero (easiest position tracking)."""
    a = random.randint(1, vocab_size - 1)  # Non-zero value
    length = random.randint(4, 8)
    seq = [a if i % 2 == 0 else 0 for i in range(length)]
    target = a if length % 2 == 0 else 0
    return {'sequence': seq, 'target': target}


def gen_ternary_fixed(vocab_size: int) -> Dict:
    """[A, 0, 0, A, 0, ?] → A, 0, or 0 - Ternary with zeros (position mod 3)."""
    a = random.randint(1, vocab_size - 1)  # Non-zero value
    length = random.randint(5, 9)
    seq = [a if i % 3 == 0 else 0 for i in range(length)]
    target = a if length % 3 == 0 else 0
    return {'sequence': seq, 'target': target}


def gen_position_parity(vocab_size: int) -> Dict:
    """[1, 2, 1, 2, ?] → 1 or 2 - Even/odd positions get different values."""
    # Two fixed values based on position parity
    even_val = random.randint(1, vocab_size // 2)
    odd_val = random.randint(vocab_size // 2, vocab_size - 1)
    length = random.randint(4, 8)
    seq = [even_val if i % 2 == 0 else odd_val for i in range(length)]
    target = even_val if length % 2 == 0 else odd_val
    return {'sequence': seq, 'target': target}


def gen_fill_A_positions(vocab_size: int) -> Dict:
    """[A, _, _, A, _, _] pattern - Learn where A appears (every 3rd, starting at 0)."""
    a = random.randint(1, vocab_size - 1)
    filler = 0  # Use 0 as filler
    length = random.randint(5, 9)
    seq = [a if i % 3 == 0 else filler for i in range(length)]
    target = a if length % 3 == 0 else filler
    return {'sequence': seq, 'target': target}


def gen_fill_B_positions(vocab_size: int) -> Dict:
    """[_, B, _, _, B, _] pattern - Learn where B appears (every 3rd, starting at 1)."""
    b = random.randint(1, vocab_size - 1)
    filler = 0
    length = random.randint(5, 9)
    seq = [b if i % 3 == 1 else filler for i in range(length)]
    target = b if length % 3 == 1 else filler
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
# YEAR 1.5: TRANSITIONAL MODULE (Executive Function)
# These patterns bridge from sequence prediction to planning/reasoning
# Key skills: multi-step operations, constraint satisfaction, working memory
# =============================================================================

def gen_apply_twice(vocab_size: int) -> Dict:
    """[5, +2] → 9 - Apply operation twice: 5+2=7, 7+2=9."""
    step = random.randint(1, 3)
    start = random.randint(0, vocab_size - 2 * step - 1)
    # Sequence shows start and the step
    seq = [start, step]
    # Target is start + step + step
    target = start + 2 * step
    if target >= vocab_size:
        return gen_apply_twice(vocab_size)
    return {'sequence': seq, 'target': target}


def gen_reverse_operation(vocab_size: int) -> Dict:
    """[10, -3] → 7 - Understand inverse: what undoes +3?"""
    step = random.randint(1, 5)
    start = random.randint(step, vocab_size - 1)
    seq = [start, step]  # step represents amount to subtract
    target = start - step
    if target < 0:
        return gen_reverse_operation(vocab_size)
    return {'sequence': seq, 'target': target}


def gen_find_missing_addend(vocab_size: int) -> Dict:
    """[3, ?, 8] → 5 - What goes in the gap? (3 + ? = 8)"""
    a = random.randint(1, vocab_size // 2)
    c = random.randint(a + 1, min(vocab_size - 1, a + vocab_size // 2))
    # Sequence: [a, MASK, c] -> target is c - a
    seq = [a, c]  # The model learns: output c - a
    target = c - a
    return {'sequence': seq, 'target': target}


def gen_chain_two_steps(vocab_size: int) -> Dict:
    """[2, +3, +4] → 9 - Apply two operations: 2+3=5, 5+4=9."""
    start = random.randint(0, 5)
    step1 = random.randint(1, 3)
    step2 = random.randint(1, 3)
    seq = [start, step1, step2]
    target = start + step1 + step2
    if target >= vocab_size:
        return gen_chain_two_steps(vocab_size)
    return {'sequence': seq, 'target': target}


def gen_conditional_simple(vocab_size: int) -> Dict:
    """[val, threshold] → 1 if val > threshold else 0."""
    threshold = random.randint(vocab_size // 4, vocab_size * 3 // 4)
    val = random.randint(0, vocab_size - 1)
    seq = [val, threshold]
    target = 1 if val > threshold else 0
    return {'sequence': seq, 'target': target}


def gen_min_of_two(vocab_size: int) -> Dict:
    """[a, b] → min(a, b) - Find smaller value."""
    a = random.randint(0, vocab_size - 1)
    b = random.randint(0, vocab_size - 1)
    seq = [a, b]
    target = min(a, b)
    return {'sequence': seq, 'target': target}


def gen_max_of_two(vocab_size: int) -> Dict:
    """[a, b] → max(a, b) - Find larger value."""
    a = random.randint(0, vocab_size - 1)
    b = random.randint(0, vocab_size - 1)
    seq = [a, b]
    target = max(a, b)
    return {'sequence': seq, 'target': target}


def gen_working_memory_recall(vocab_size: int) -> Dict:
    """[A, B, C, D, 0] → A - Recall first element after distractors."""
    length = random.randint(3, 5)
    values = [random.randint(1, vocab_size - 1) for _ in range(length)]
    seq = values + [0]  # 0 signals "recall first"
    target = values[0]
    return {'sequence': seq, 'target': target}


def gen_working_memory_last(vocab_size: int) -> Dict:
    """[A, B, C, D, 1] → D - Recall last non-signal element."""
    length = random.randint(3, 5)
    values = [random.randint(1, vocab_size - 1) for _ in range(length)]
    seq = values + [1]  # 1 signals "recall last"
    target = values[-1]
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


YEAR_1_PATTERNS = [
    # 1A: Constancy
    PatternType('constant', gen_constant, 1, 1, '1A', 'Things stay the same'),

    # 1A': Quantity Awareness (Conservation & Cardinality)
    # Critical foundation: quantity is a stable property, independent of arrangement
    PatternType('sequence_length', gen_sequence_length, 2, 1, "1A'", 'Count elements'),
    PatternType('count_value', gen_count_value, 2, 1, "1A'", 'Count occurrences of value'),
    PatternType('distinct_count', gen_distinct_count, 2, 1, "1A'", 'Count distinct values'),
    PatternType('conservation_shuffle', gen_conservation_shuffle, 2, 1, "1A'", 'Count invariant to order'),

    # 1B: Repetition & Memory
    PatternType('repeating', gen_repeating, 1, 1, '1B', 'Remember what was seen'),
    PatternType('echo', gen_echo, 2, 1, '1B', 'Pattern with gaps'),

    # 1B': Position Scaffolding (bridge to cycles)
    # These teach position awareness with simpler cognitive load
    PatternType('simple_alternating', gen_simple_alternating, 2, 1, "1B'", 'Alternating with zero'),
    PatternType('position_parity', gen_position_parity, 2, 1, "1B'", 'Even/odd positions'),
    PatternType('ternary_fixed', gen_ternary_fixed, 2, 1, "1B'", 'Ternary with zeros'),
    PatternType('fill_A_positions', gen_fill_A_positions, 2, 1, "1B'", 'Recognize A positions'),
    PatternType('fill_B_positions', gen_fill_B_positions, 2, 1, "1B'", 'Recognize B positions'),

    # 1C: Alternation & Position (full complexity)
    PatternType('alternating', gen_alternating, 3, 1, '1C', 'Two-element cycle'),
    PatternType('ternary_cycle', gen_ternary_cycle, 4, 1, '1C', 'Three-element cycle'),

    # 1D: Linear Change
    PatternType('incrementing', gen_incrementing, 2, 1, '1D', 'Count up by 1'),
    PatternType('decrementing', gen_decrementing, 2, 1, '1D', 'Count down by 1'),

    # 1E: Rate of Change
    PatternType('fixed_offset', gen_fixed_offset, 3, 1, '1E', 'Count by fixed step'),
    PatternType('variable_step', gen_variable_step, 4, 1, '1E', 'Increasing step size'),

    # 1E': Symbolic Property Recognition (Bridge to math operations)
    PatternType('compute_step', gen_compute_step, 3, 1, "1E'", 'Identify additive step'),
    PatternType('compute_first_diff', gen_compute_first_diff, 3, 1, "1E'", 'Compute first difference'),
    PatternType('compute_ratio', gen_compute_ratio, 4, 1, "1E'", 'Identify multiplicative ratio'),
    PatternType('is_constant', gen_is_constant, 2, 1, "1E'", 'Classify as constant'),
    PatternType('is_increasing', gen_is_increasing, 2, 1, "1E'", 'Classify as increasing'),
    PatternType('is_decreasing', gen_is_decreasing, 2, 1, "1E'", 'Classify as decreasing'),
]

# Year 1.5: Transitional Module (bridges Year 1 sequence prediction to Year 2 reasoning)
YEAR_1_5_PATTERNS = [
    # 1.5A: Multi-Step Operations
    PatternType('apply_twice', gen_apply_twice, 3, 1.5, '1.5A', 'Apply operation twice'),
    PatternType('reverse_operation', gen_reverse_operation, 3, 1.5, '1.5A', 'Inverse operation'),
    PatternType('chain_two_steps', gen_chain_two_steps, 4, 1.5, '1.5A', 'Chain two operations'),

    # 1.5B: Constraint Satisfaction
    PatternType('find_missing_addend', gen_find_missing_addend, 4, 1.5, '1.5B', 'Find missing value'),
    PatternType('conditional_simple', gen_conditional_simple, 3, 1.5, '1.5B', 'Simple if-then'),
    PatternType('min_of_two', gen_min_of_two, 2, 1.5, '1.5B', 'Find minimum'),
    PatternType('max_of_two', gen_max_of_two, 2, 1.5, '1.5B', 'Find maximum'),

    # 1.5C: Working Memory
    PatternType('working_memory_recall', gen_working_memory_recall, 3, 1.5, '1.5C', 'Recall first'),
    PatternType('working_memory_last', gen_working_memory_last, 3, 1.5, '1.5C', 'Recall last'),
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

ALL_PATTERNS = YEAR_1_PATTERNS + YEAR_1_5_PATTERNS + YEAR_2_PATTERNS


def get_patterns_by_year(year: float) -> List[PatternType]:
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
    print("DEVELOPMENTAL CURRICULUM - Years 1, 1.5 & 2")
    print("=" * 70)

    year_info = {
        1: 'SENSORIMOTOR FOUNDATIONS',
        1.5: 'TRANSITIONAL MODULE (Executive Function)',
        2: 'RELATIONAL & PHYSICAL'
    }

    for year in [1, 1.5, 2]:
        year_patterns = get_patterns_by_year(year)
        print(f"\n{'='*50}")
        print(f"YEAR {year}: {year_info[year]}")
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
    print(f"  Year 1.5: {len(YEAR_1_5_PATTERNS)} patterns")
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
