#!/usr/bin/env python3
"""
Run Developmental Curriculum Training

Trains students through Years 0-2 of the developmental curriculum.
Year 0: Quantitative Primitives (number sense - the foundation)
Year 1: Sensorimotor Foundations (patterns)
Year 2: Relational & Physical Understanding

Supports phased training: master each section before adding the next.
"""

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime

from classroom import ClassroomBroker, Student
from developmental_curriculum import (
    DevelopmentalDataset, collate_fn,
    YEAR_0_PATTERNS, YEAR_1_PATTERNS, YEAR_2_PATTERNS, ALL_PATTERNS,
    get_pattern_names, get_patterns_by_section, print_curriculum
)
import random
from systems import ExaminationSystem
from systems.progression import TopicTracker

# Section order for phased training
# YEAR 0: Number sense FIRST - the substrate of all reasoning
# PRINCIPLE: All patterns unambiguous (2+ inputs OR unique single-input mapping)
# - 0A: chains (context-rich: [1,2,3]→4)
# - 0B: two-input arithmetic (add, subtract, remainder)
# - 0C: scaling (double, half - single input but unique mappings)
# - 0D: comparison & algebra (greater_than, less_than, missing_addend)
YEAR_0_SECTIONS = ['0A', '0B', '0C', '0D']

# YEAR 1: Patterns build on number sense
# - 1A, 1B: Pure memory (constancy, repetition)
# - 1C: Position (alternating, ternary - cyclic patterns)
# - 1D: Direction (incrementing, decrementing)
# - 1E: Rate (fixed_offset, variable_step)
# - 1F: Traps (test overconfidence)
# - 1G: Abstract single-number (successor, predecessor - NOW learnable)
# - 1H: Extended arithmetic (compare, add_three, multiply)
YEAR_1_SECTIONS = ['1A', '1B', '1C', '1D', '1E', '1F', '1G', '1H']

YEAR_2_SECTIONS = ['2A', '2B', '2C', '2D', '2E']

ALL_SECTIONS = YEAR_0_SECTIONS + YEAR_1_SECTIONS + YEAR_2_SECTIONS


def identify_tutoring_pairs(broker, pattern_names, pattern_to_idx, level_gap=3):
    """Identify tutor-student pairs for peer teaching."""
    tutoring = {}

    for pt_idx, pt in enumerate(pattern_names):
        tutoring[pt_idx] = {}
        student_levels = {}
        graduates = []

        for name, student in broker.students.items():
            if pt_idx < student.exam_system.n_topics:
                confirmed = student.exam_system.confirmed_level[pt_idx].item()
                graduated = student.exam_system.topic_graduated[pt_idx].item()
                student_levels[name] = confirmed
                if graduated:
                    graduates.append(name)

        for learner_name, learner_level in student_levels.items():
            if learner_name in graduates:
                continue

            best_tutor = None
            best_gap = 0

            for tutor_name, tutor_level in student_levels.items():
                if tutor_name == learner_name:
                    continue

                gap = tutor_level - learner_level
                is_graduate = tutor_name in graduates
                can_tutor = is_graduate or gap >= level_gap

                if can_tutor:
                    tutor_priority = (1 if is_graduate else 0, gap)
                    if best_tutor is None or tutor_priority > (1 if best_tutor in graduates else 0, best_gap):
                        best_tutor = tutor_name
                        best_gap = gap

            if best_tutor:
                tutoring[pt_idx][learner_name] = best_tutor

    return tutoring


def get_tutor_predictions(tutor, tokens, seq_lens, temperature=2.0):
    """Get soft predictions from tutor for knowledge distillation."""
    tutor.eval()
    with torch.no_grad():
        logits, _, _ = tutor(tokens, seq_lens)
        soft_targets = F.softmax(logits / temperature, dim=-1)
    tutor.train()
    return soft_targets


def check_section_mastery(broker, section_patterns, pattern_to_idx, mastery_level=3):
    """
    Check if all students have mastered all patterns in a section.

    Mastery = confirmed level >= mastery_level for all patterns in section.
    Returns: (is_mastered, details_dict)
    """
    details = {}
    all_mastered = True

    for name, student in broker.students.items():
        details[name] = {}
        for pt in section_patterns:
            if pt not in pattern_to_idx:
                continue
            pt_idx = pattern_to_idx[pt]
            if pt_idx >= student.exam_system.n_topics:
                continue

            confirmed = student.exam_system.confirmed_level[pt_idx].item()
            details[name][pt] = confirmed

            if confirmed < mastery_level:
                all_mastered = False

    return all_mastered, details


def get_active_sections(year, current_phase):
    """Get list of active sections based on year and phase."""
    if year == 0:
        sections = YEAR_0_SECTIONS
    elif year == 1:
        sections = YEAR_0_SECTIONS + YEAR_1_SECTIONS  # Year 1 includes Year 0
    elif year == 2:
        sections = YEAR_0_SECTIONS + YEAR_1_SECTIONS + YEAR_2_SECTIONS  # Full curriculum
    else:
        sections = ALL_SECTIONS

    return sections[:current_phase + 1]


def get_patterns_for_sections(sections):
    """Get all patterns for a list of sections."""
    patterns = []
    for section in sections:
        patterns.extend(get_patterns_by_section(section))
    return patterns


def train_epoch(broker, loader, optimizers, criterion, device, pattern_to_idx, tutoring_pairs=None):
    """Train all students for one epoch."""
    broker.train()
    broker.wake_all()

    tutoring_pairs = tutoring_pairs or {}
    SHOW_THRESHOLD = 0.7
    APPROVAL_BONUS = 3.0
    TUTOR_WEIGHT = 0.3
    TUTOR_TEMP = 2.0
    TUTOR_XP_ATTEMPT = 0.3  # Base XP just for trying to help
    TUTOR_XP_SUCCESS = 1.0  # Bonus when tutee gets it right
    MAX_TUTORING_PER_STUDENT = 20  # Cap so tutors still have time for their own learning

    results = {name: {
        'loss': 0.0, 'correct': 0, 'total': 0,
        'show_count': 0, 'approval_count': 0,
        'tutoring_received': 0, 'tutoring_given': 0
    } for name in broker.students.keys()}

    # Track tutoring count this epoch to enforce cap
    tutoring_count = {name: 0 for name in broker.students.keys()}

    for batch in loader:
        tokens = batch['tokens'].to(device)
        targets = batch['target'].to(device)
        seq_lens = batch['seq_len']
        pattern_types = batch['pattern_type']

        tutor_cache = {}

        for name, student in broker.students.items():
            optimizer = optimizers[name]
            optimizer.zero_grad()

            output = student(tokens, seq_lens, return_details=True)
            logits = output['logits']

            main_loss = criterion(logits, targets)

            preds = logits.argmax(dim=-1)
            correct = (preds == targets)
            conf = output['self_state']['emotions']['confidence'].squeeze()

            if conf.dim() == 0:
                conf = conf.unsqueeze(0)
            if correct.dim() == 0:
                correct = correct.unsqueeze(0)

            min_len = min(len(conf), len(correct))
            conf = conf[:min_len]
            correct_float = correct[:min_len].float()

            conf_loss = F.binary_cross_entropy(conf, correct_float)

            # Tutoring
            tutoring_loss = torch.tensor(0.0, device=device)
            tutor_helped = False

            for i, pt in enumerate(pattern_types):
                if pt not in pattern_to_idx:
                    continue
                pt_idx = pattern_to_idx[pt]
                if pt_idx in tutoring_pairs and name in tutoring_pairs[pt_idx]:
                    tutor_name = tutoring_pairs[pt_idx][name]
                    tutor = broker.students[tutor_name]

                    # Check tutoring cap - tutor needs time for their own learning too
                    if tutoring_count[tutor_name] >= MAX_TUTORING_PER_STUDENT:
                        continue

                    if tutor_name not in tutor_cache:
                        tutor_cache[tutor_name] = get_tutor_predictions(tutor, tokens, seq_lens, TUTOR_TEMP)

                    student_log_probs = F.log_softmax(logits[i:i+1] / TUTOR_TEMP, dim=-1)
                    tutor_probs = tutor_cache[tutor_name][i:i+1]

                    kl = F.kl_div(student_log_probs, tutor_probs, reduction='batchmean')
                    tutoring_loss = tutoring_loss + kl
                    tutor_helped = True

                    results[name]['tutoring_received'] += 1
                    results[tutor_name]['tutoring_given'] += 1
                    tutoring_count[tutor_name] += 1

                    # Tutor XP: base for attempting, bonus if tutee got it right
                    if not tutor.exam_system.topic_graduated[pt_idx]:
                        tutor.topic_tracker.award_xp(pt_idx, TUTOR_XP_ATTEMPT)
                        # Success bonus when tutee gets the answer right
                        if i < len(correct) and correct[i]:
                            tutor.topic_tracker.award_xp(pt_idx, TUTOR_XP_SUCCESS)

            loss = main_loss + 0.1 * conf_loss
            if tutor_helped:
                loss = loss + TUTOR_WEIGHT * tutoring_loss

            loss.backward()
            optimizer.step()

            results[name]['loss'] += loss.item() * len(targets)
            results[name]['correct'] += correct.sum().item()
            results[name]['total'] += len(targets)

            # Update topic tracker
            for i, pt in enumerate(pattern_types):
                if pt not in pattern_to_idx:
                    continue
                pt_idx = pattern_to_idx[pt]
                if i < len(correct):
                    student.topic_tracker.update(
                        torch.tensor([pt_idx], device=device),
                        correct[i:i+1],
                        conf[i:i+1] if i < len(conf) else torch.tensor([0.5], device=device)
                    )
                    if correct[i]:
                        student.topic_tracker.award_xp(pt_idx, 1.0)

            # Approval seeking
            for i in range(min(len(targets), len(conf))):
                if conf[i].item() >= SHOW_THRESHOLD:
                    results[name]['show_count'] += 1
                    if correct[i]:
                        results[name]['approval_count'] += 1
                        if pattern_types[i] in pattern_to_idx:
                            pt_idx = pattern_to_idx[pattern_types[i]]
                            student.topic_tracker.award_xp(pt_idx, APPROVAL_BONUS)

    broker.sleep_all()

    for name in results:
        total = results[name]['total']
        if total > 0:
            results[name]['loss'] /= total
            results[name]['accuracy'] = results[name]['correct'] / total
            results[name]['show_rate'] = results[name]['show_count'] / total
            if results[name]['show_count'] > 0:
                results[name]['approval_rate'] = results[name]['approval_count'] / results[name]['show_count']
            else:
                results[name]['approval_rate'] = 0.0
        else:
            results[name]['accuracy'] = 0.0

    return results


def evaluate(broker, loader, device, pattern_to_idx):
    """Evaluate all students."""
    broker.eval()

    results = {name: {'correct': 0, 'total': 0, 'per_pattern': {}}
               for name in broker.students.keys()}

    with torch.no_grad():
        for batch in loader:
            tokens = batch['tokens'].to(device)
            targets = batch['target'].to(device)
            seq_lens = batch['seq_len']
            pattern_types = batch['pattern_type']

            for name, student in broker.students.items():
                logits, _, _ = student(tokens, seq_lens)
                preds = logits.argmax(dim=-1)
                correct = (preds == targets)

                results[name]['correct'] += correct.sum().item()
                results[name]['total'] += len(targets)

                for i, pt in enumerate(pattern_types):
                    if pt not in results[name]['per_pattern']:
                        results[name]['per_pattern'][pt] = {'correct': 0, 'total': 0}
                    results[name]['per_pattern'][pt]['total'] += 1
                    if correct[i]:
                        results[name]['per_pattern'][pt]['correct'] += 1

    for name in results:
        total = results[name]['total']
        results[name]['accuracy'] = results[name]['correct'] / total if total > 0 else 0

        for pt in results[name]['per_pattern']:
            pt_total = results[name]['per_pattern'][pt]['total']
            pt_correct = results[name]['per_pattern'][pt]['correct']
            results[name]['per_pattern'][pt] = pt_correct / pt_total if pt_total > 0 else 0

    return results


def run_exams(broker, pattern_names, pattern_to_idx, device, epoch):
    """Run level-up exams."""
    results = {}

    for name, student in broker.students.items():
        student_results = []
        student.exam_system.tick_cooldowns()

        for pt_idx, pt in enumerate(pattern_names):
            while student.exam_system.check_eligible(pt_idx):
                target_level = student.exam_system.confirmed_level[pt_idx].item() + 1
                exam_size = student.exam_system.get_exam_size(target_level)

                # Generate exam data for this specific pattern
                exam_data = DevelopmentalDataset(
                    n_examples=exam_size,
                    seed=epoch * 1000 + pt_idx * 100 + target_level,
                    patterns=[pt]
                )

                correct_count = 0
                total_count = 0

                student.eval()
                with torch.no_grad():
                    for ex in exam_data:
                        tokens = ex['sequence'].unsqueeze(0).to(device)
                        target = ex['target']
                        seq_len = [ex['seq_len']]

                        logits, _, _ = student(tokens, seq_len)
                        pred = logits.argmax(dim=-1).item()

                        total_count += 1
                        if pred == target:
                            correct_count += 1

                result = student.exam_system.take_exam(pt_idx, correct_count, total_count)
                result['pattern'] = pt
                result['student'] = name
                student_results.append(result)

                if result['passed']:
                    print(f"  ** {student.name} passed L{result['new_level']} exam for {pt}! ({result['score']:.0%})")
                else:
                    print(f"  -- {student.name} failed L{result['new_level']+1} exam for {pt} ({result['score']:.0%} < {result['threshold']:.0%})")

        student.train()
        results[name] = student_results

    return results


def create_datasets(sections, args, seed_offset=0):
    """Create train/val datasets for given sections."""
    train_data = DevelopmentalDataset(
        n_examples=args.train_size,
        seed=42 + seed_offset,
        sections=sections
    )
    val_data = DevelopmentalDataset(
        n_examples=args.val_size,
        seed=123 + seed_offset,
        sections=sections
    )
    return train_data, val_data


def get_mastered_patterns(broker, pattern_to_idx, mastery_level=3):
    """
    Get patterns that ALL students have mastered (confirmed level >= mastery_level).
    These become "unlocked priors" for playday.
    """
    mastered = []
    idx_to_pattern = {v: k for k, v in pattern_to_idx.items()}

    for pt_idx in range(len(pattern_to_idx)):
        all_mastered = True
        for student in broker.students.values():
            if pt_idx >= student.exam_system.n_topics:
                all_mastered = False
                break
            confirmed = student.exam_system.confirmed_level[pt_idx].item()
            if confirmed < mastery_level:
                all_mastered = False
                break

        if all_mastered:
            mastered.append(idx_to_pattern[pt_idx])

    return mastered


def generate_creative_challenge(mastered_patterns, vocab_size=26):
    """
    Teacher generates creative challenges combining mastered skills.

    Challenge types:
    - Longer sequences (test endurance)
    - Mixed patterns (combine two skills)
    - Edge cases (boundary values)
    """
    if not mastered_patterns:
        return None, None

    # Find the pattern objects
    pattern_objs = [p for p in ALL_PATTERNS if p.name in mastered_patterns]
    if not pattern_objs:
        return None, None

    # Pick a challenge type
    challenge_type = random.choice(['longer', 'edge_case', 'reversed'])

    # Pick a pattern to base the challenge on
    pattern = random.choice(pattern_objs)
    base = pattern.generator(vocab_size)

    if challenge_type == 'longer' and len(base['sequence']) < 8:
        # Make a longer version by extending the pattern
        seq = base['sequence']
        target = base['target']
        # Extend by repeating logic (crude but works for demo)
        extended_seq = seq + [target]
        # Generate new target based on pattern
        new_example = pattern.generator(vocab_size)
        challenge = {
            'sequence': extended_seq,
            'target': new_example['target'],
            'challenge_type': 'longer',
            'base_pattern': pattern.name
        }
    elif challenge_type == 'edge_case':
        # Use boundary values
        challenge = {
            'sequence': base['sequence'],
            'target': base['target'],
            'challenge_type': 'edge_case',
            'base_pattern': pattern.name
        }
    else:  # reversed
        # Present sequence backwards (tests flexibility)
        challenge = {
            'sequence': list(reversed(base['sequence'])),
            'target': base['sequence'][0],  # First element becomes target
            'challenge_type': 'reversed',
            'base_pattern': pattern.name
        }

    return challenge, f"Creative {challenge_type} challenge using {pattern.name}"


# Section purpose descriptions (playful for Year 0)
SECTION_PURPOSES = {
    '0A': "Learning to count and see what comes next! Numbers are friends that follow each other.",
    '0B': "Putting numbers together and taking them apart. Addition and subtraction are like giving and sharing!",
    '0C': "Comparing things - which is bigger? Which is smaller? Finding patterns in how numbers relate.",
    '1A': "Sequences get longer and trickier! Patterns can repeat in many ways.",
    '1B': "Numbers can do fancy dances - stepping by 2s, 3s, or even backwards!",
    '1C': "Mixing it up - using all our number skills together.",
}


def run_class_session(broker, new_section, pattern_to_idx, device, vocab_size=26, year=0):
    """
    Run a class session when a new section unlocks.

    Structure (I Do, We Do):
    1. State the PURPOSE of what we're learning (playfully framed for Year 0)
    2. Teacher shows worked examples for new patterns (I Do)
    3. Guided practice - student tries, immediate feedback (We Do)

    This happens BEFORE regular training on new patterns begins.
    """
    print(f"\n  {'📚'*20}")
    print(f"  *** CLASS SESSION: {new_section} ***")

    # 1. PURPOSE - why are we learning this?
    purpose = SECTION_PURPOSES.get(new_section, "Learning new and exciting patterns!")
    print(f"\n  📖 Today's lesson: {purpose}")

    # Get new patterns for this section
    new_patterns = get_patterns_by_section(new_section)
    if not new_patterns:
        print(f"  No new patterns for section {new_section}")
        return

    print(f"  New patterns: {[p.name for p in new_patterns]}")

    # 2. WORKED EXAMPLES (I Do) - Teacher demonstrates each new pattern
    print(f"\n  --- Teacher Demonstrates (I Do) ---")
    for pattern in new_patterns:
        # Show 2 worked examples per pattern
        print(f"\n  {pattern.name}:")
        for i in range(2):
            example = pattern.generator(vocab_size)
            seq_str = ' '.join(map(str, example['sequence']))
            print(f"    Example {i+1}: [{seq_str}] → {example['target']}")

    # 3. GUIDED PRACTICE (We Do) - Students try, teacher gives immediate feedback
    print(f"\n  --- Guided Practice (We Do) ---")
    print(f"  Students try similar problems. Teacher validates immediately.")

    results = {name: {'correct': 0, 'total': 0} for name in broker.students.keys()}

    for pattern in new_patterns:
        print(f"\n  Pattern: {pattern.name}")

        # Generate 3 practice problems per pattern
        for trial in range(3):
            example = pattern.generator(vocab_size)
            max_len = 12
            seq = example['sequence']
            padded = seq + [0] * (max_len - len(seq))
            tokens = torch.tensor(padded[:max_len], dtype=torch.long).unsqueeze(0).to(device)
            target = example['target']
            seq_len = [len(seq)]
            seq_str = ' '.join(map(str, seq))

            for name, student in broker.students.items():
                student.eval()
                with torch.no_grad():
                    logits, conf, _ = student(tokens, seq_len)
                    pred = logits.argmax(dim=-1).item()
                    correct = (pred == target)

                results[name]['total'] += 1

                # Immediate feedback (the key pedagogical moment)
                if correct:
                    results[name]['correct'] += 1
                    # Small reinforcement for correct guided practice
                    if pattern.name in pattern_to_idx:
                        pt_idx = pattern_to_idx[pattern.name]
                        student.topic_tracker.award_xp(pt_idx, 0.3)  # Guided practice XP

            # Show one example of feedback (not all to avoid spam)
            if trial == 0:
                sample_student = list(broker.students.keys())[0]
                sample_pred = results[sample_student]['correct'] > 0  # Simplified
                print(f"    [{seq_str}] → ?")
                print(f"    Teacher: The answer is {target}.")

    # Summary
    print(f"\n  --- Guided Practice Results ---")
    for name in broker.students.keys():
        acc = results[name]['correct'] / max(1, results[name]['total'])
        print(f"    {name}: {results[name]['correct']}/{results[name]['total']} ({acc:.0%})")

    print(f"\n  Class session complete. Ready for independent practice!")
    print(f"  {'📚'*20}\n")

    return results


def run_playday(broker, mastered_patterns, pattern_to_idx, device, epoch, vocab_size=26, year=0, section_playdays=1,
                current_section_patterns=None, all_year_patterns=None):
    """
    Run a playday session - exploration without grades.

    Features:
    - Unlocked priors: access to all mastered patterns
    - Creative challenges: teacher suggests harder variants
    - Peer challenges: students solve patterns from each other
    - No penalties: wrong answers don't hurt XP (exploration mode)
    - Mastery showcase: students who've mastered current section get ALL year patterns
      and others can observe their play

    Teacher involvement scales with section_playdays:
    - 1-2: Light touch, free exploration
    - 3-4: Teacher suggests patterns to focus on
    - 5+: Teacher actively guides, more structured challenges
    """
    # Determine teacher involvement level
    # Scales with how many playdays on this section (more = needs more support)
    if section_playdays == 1:
        teacher_mode = "free"
        teacher_label = "Free Exploration"
    elif section_playdays == 2:
        teacher_mode = "examples"
        teacher_label = "Teacher Shows Examples"
    elif section_playdays <= 4:
        teacher_mode = "guided"
        teacher_label = "Teacher Guided"
    else:
        teacher_mode = "structured"
        teacher_label = "Teacher Structured (focus session)"

    print(f"\n  {'🎮'*20}")
    print(f"  *** PLAYDAY! (Epoch {epoch}) - {teacher_label} ***")
    print(f"  Section playday #{section_playdays}")
    print(f"  Unlocked priors: {mastered_patterns}")
    print(f"  {'🎮'*20}")

    if not mastered_patterns:
        print("  No mastered patterns yet - skipping playday")
        return {}

    results = {name: {
        'creative_correct': 0, 'creative_total': 0,
        'peer_correct': 0, 'peer_total': 0,
        'turns_correct': 0, 'turns_total': 0,
        'patterns_explored': set()
    } for name in broker.students.keys()}

    # === PHASE 1: CREATIVE CHALLENGES FROM TEACHER ===
    print("\n  --- Teacher's Creative Challenges ---")

    # Teacher involvement affects challenge selection
    if teacher_mode == "free":
        # Random exploration
        n_challenges = min(10, len(mastered_patterns) * 3)
        focus_patterns = mastered_patterns
    elif teacher_mode == "examples":
        # Teacher shows worked examples (primaries) then students predict-then-confirm
        n_challenges = min(12, len(mastered_patterns) * 3)
        # Find patterns where students need help
        weak_patterns = []
        for pt in mastered_patterns:
            if pt in pattern_to_idx:
                pt_idx = pattern_to_idx[pt]
                avg_level = sum(
                    s.exam_system.confirmed_level[pt_idx].item()
                    for s in broker.students.values()
                ) / len(broker.students)
                if avg_level < 10:
                    weak_patterns.append((pt, avg_level))
        weak_patterns.sort(key=lambda x: x[1])  # Weakest first
        focus_patterns = [p[0] for p in weak_patterns[:4]] if weak_patterns else mastered_patterns

        # PREDICT-THEN-CONFIRM cycle for each focus pattern
        print(f"\n  --- Predict-Then-Confirm (Example-Problem Pairs) ---")
        for pt_name in focus_patterns[:3]:
            pattern_obj = next((p for p in ALL_PATTERNS if p.name == pt_name), None)
            if pattern_obj is None:
                continue

            # 1. WORKED EXAMPLE - Teacher demonstrates (I Do)
            worked_example = pattern_obj.generator(vocab_size)
            seq_str = ' '.join(map(str, worked_example['sequence']))
            print(f"\n    [{pt_name}] Teacher: 'Watch this pattern...'")
            print(f"      Worked example: [{seq_str}] → {worked_example['target']}")

            # 2. SIMILAR PROBLEM - Student predicts (We Do)
            similar_problem = pattern_obj.generator(vocab_size)
            max_len = 12
            seq = similar_problem['sequence']
            padded = seq + [0] * (max_len - len(seq))
            tokens = torch.tensor(padded[:max_len], dtype=torch.long).unsqueeze(0).to(device)
            target = similar_problem['target']
            seq_len_val = [len(seq)]
            sim_seq_str = ' '.join(map(str, seq))

            print(f"      Now you try: [{sim_seq_str}] → ?")

            # 3. Each student predicts
            predictions = {}
            for name, student in broker.students.items():
                student.eval()
                with torch.no_grad():
                    logits, conf, _ = student(tokens, seq_len_val)
                    pred = logits.argmax(dim=-1).item()
                    predictions[name] = pred

            # 4. CONFIRM - Reveal answer with feedback
            print(f"      Answer: {target}")
            for name, pred in predictions.items():
                correct = (pred == target)
                if correct:
                    print(f"        {name}: ✓ (predicted {pred})")
                    # Prediction success XP (stronger encoding from correct prediction)
                    if pt_name in pattern_to_idx:
                        pt_idx = pattern_to_idx[pt_name]
                        broker.students[name].topic_tracker.award_xp(pt_idx, 0.5)
                    results[name]['creative_correct'] += 1
                else:
                    print(f"        {name}: ✗ (predicted {pred}, was {target})")
                    # Even incorrect predictions help - prediction error drives learning
                    if pt_name in pattern_to_idx:
                        pt_idx = pattern_to_idx[pt_name]
                        broker.students[name].topic_tracker.award_xp(pt_idx, 0.2)
                results[name]['creative_total'] += 1
                results[name]['patterns_explored'].add(pt_name)
        print()
    elif teacher_mode == "guided":
        # Teacher suggests focusing on weaker areas
        n_challenges = min(15, len(mastered_patterns) * 4)
        # Find patterns where students are struggling (not yet L10)
        weak_patterns = []
        for pt in mastered_patterns:
            if pt in pattern_to_idx:
                pt_idx = pattern_to_idx[pt]
                avg_level = sum(
                    s.exam_system.confirmed_level[pt_idx].item()
                    for s in broker.students.values()
                ) / len(broker.students)
                if avg_level < 10:
                    weak_patterns.append(pt)
        focus_patterns = weak_patterns if weak_patterns else mastered_patterns
        print(f"  Teacher suggests focusing on: {focus_patterns[:3]}")
    else:  # structured
        # Teacher actively guides, more challenges, focus on weakest
        n_challenges = min(20, len(mastered_patterns) * 5)
        # Find the weakest pattern across all students
        pattern_scores = []
        for pt in mastered_patterns:
            if pt in pattern_to_idx:
                pt_idx = pattern_to_idx[pt]
                avg_level = sum(
                    s.exam_system.confirmed_level[pt_idx].item()
                    for s in broker.students.values()
                ) / len(broker.students)
                pattern_scores.append((pt, avg_level))
        pattern_scores.sort(key=lambda x: x[1])  # Weakest first
        focus_patterns = [p[0] for p in pattern_scores[:3]] if pattern_scores else mastered_patterns
        print(f"  Teacher focusing session on: {focus_patterns}")
        print(f"  (These need the most work)")

    for _ in range(n_challenges):
        challenge, description = generate_creative_challenge(focus_patterns, vocab_size)
        if challenge is None:
            continue

        # Pad sequence
        max_len = 12
        seq = challenge['sequence']
        padded = seq + [0] * (max_len - len(seq))
        tokens = torch.tensor(padded[:max_len], dtype=torch.long).unsqueeze(0).to(device)
        target = challenge['target']
        seq_len = [len(seq)]

        for name, student in broker.students.items():
            student.eval()
            with torch.no_grad():
                logits, _, _ = student(tokens, seq_len)
                pred = logits.argmax(dim=-1).item()
                correct = (pred == target)

            results[name]['creative_total'] += 1
            results[name]['patterns_explored'].add(challenge['base_pattern'])

            if correct:
                results[name]['creative_correct'] += 1
                # Bonus XP for creative success (but it's playday, so smaller bonus)
                if challenge['base_pattern'] in pattern_to_idx:
                    pt_idx = pattern_to_idx[challenge['base_pattern']]
                    student.topic_tracker.award_xp(pt_idx, 0.5)  # Small exploration bonus

    # === MASTERY SHOWCASE (for students who've mastered current section) ===
    # Students who've mastered current section get to play with ALL year patterns
    # Others can observe them - learning by watching
    if current_section_patterns and all_year_patterns:
        # Identify mastery students (L10 on all current section patterns)
        mastery_students = []
        learning_students = []
        for name, student in broker.students.items():
            all_mastered = True
            for pt in current_section_patterns:
                if pt in pattern_to_idx:
                    pt_idx = pattern_to_idx[pt]
                    level = student.exam_system.confirmed_level[pt_idx].item()
                    if level < 10:
                        all_mastered = False
                        break
            if all_mastered:
                mastery_students.append(name)
            else:
                learning_students.append(name)

        if mastery_students and learning_students:
            print(f"\n  --- Mastery Showcase ---")
            print(f"  Graduated: {mastery_students} (exploring ALL year patterns)")
            print(f"  Observers: {learning_students} (watching and learning)")

            # Mastery students get to explore ALL patterns in the year
            all_pattern_names = [p.name for p in all_year_patterns]
            n_showcase = min(8, len(all_pattern_names))

            for _ in range(n_showcase):
                # Pick a random pattern from the full year
                pattern_name = random.choice(all_pattern_names)
                pattern_obj = next((p for p in ALL_PATTERNS if p.name == pattern_name), None)
                if pattern_obj is None:
                    continue

                example = pattern_obj.generator(vocab_size)
                max_len = 12
                seq = example['sequence']
                padded = seq + [0] * (max_len - len(seq))
                tokens = torch.tensor(padded[:max_len], dtype=torch.long).unsqueeze(0).to(device)
                target = example['target']
                seq_len_val = [len(seq)]

                # Mastery student attempts
                for master_name in mastery_students:
                    master = broker.students[master_name]
                    master.eval()
                    with torch.no_grad():
                        logits, conf, _ = master(tokens, seq_len_val)
                        pred = logits.argmax(dim=-1).item()
                        correct = (pred == target)

                    # Others observe - they see the sequence, the attempt, the result
                    # This is passive learning (observation bonus)
                    for observer_name in learning_students:
                        observer = broker.students[observer_name]
                        # Observers get small XP bonus just for watching mastery play
                        if pattern_name in pattern_to_idx:
                            pt_idx = pattern_to_idx[pattern_name]
                            observer.topic_tracker.award_xp(pt_idx, 0.1)  # Tiny observation bonus

                    results[master_name]['patterns_explored'].add(pattern_name)
                    if correct:
                        # Mastery student XP for exploring new territory
                        if pattern_name in pattern_to_idx:
                            pt_idx = pattern_to_idx[pattern_name]
                            master.topic_tracker.award_xp(pt_idx, 0.3)

    # === PHASE 2: PEER CHALLENGES (Year 1+ only) ===
    # Year 0 is individual exploration - peer interaction comes later
    if year >= 1:
        print("\n  --- Peer Challenges ---")
        student_names = list(broker.students.keys())

        # Each student gets challenged by each other student
        for challenger_name in student_names:
            challenger = broker.students[challenger_name]

            # Challenger picks a pattern they're good at
            challenger_strengths = []
            for pt in mastered_patterns:
                if pt in pattern_to_idx:
                    pt_idx = pattern_to_idx[pt]
                    level = challenger.topic_tracker.get_level(pt_idx)
                    challenger_strengths.append((pt, level))

            if not challenger_strengths:
                continue

            # Pick their strongest pattern
            challenger_strengths.sort(key=lambda x: x[1], reverse=True)
            challenge_pattern = challenger_strengths[0][0]

            # Generate a challenge from that pattern
            pattern_obj = next((p for p in ALL_PATTERNS if p.name == challenge_pattern), None)
            if pattern_obj is None:
                continue

            example = pattern_obj.generator(vocab_size)
            max_len = 12
            seq = example['sequence']
            padded = seq + [0] * (max_len - len(seq))
            tokens = torch.tensor(padded[:max_len], dtype=torch.long).unsqueeze(0).to(device)
            target = example['target']
            seq_len = [len(seq)]

            # Other students try to solve it
            for solver_name, solver in broker.students.items():
                if solver_name == challenger_name:
                    continue

                solver.eval()
                with torch.no_grad():
                    logits, _, _ = solver(tokens, seq_len)
                    pred = logits.argmax(dim=-1).item()
                    correct = (pred == target)

                results[solver_name]['peer_total'] += 1

                if correct:
                    results[solver_name]['peer_correct'] += 1
                    # No XP for peer challenges - just exploration

    # === PHASE 3: TURN-TAKING CHALLENGES (Year 1+ only) ===
    # Students build sequences together, feeling the rhythm of alternating/ternary
    # BUT ONLY if they've learned these patterns - don't test unlearned skills!
    if year >= 1:
        print("\n  --- Turn-Taking Challenges ---")
        student_names = list(broker.students.keys())

        # Only run if alternating/ternary patterns are mastered
        can_do_alternating = 'alternating' in mastered_patterns
        can_do_ternary = 'ternary_cycle' in mastered_patterns

        if not can_do_alternating and not can_do_ternary:
            print("    (Skipped - alternating/ternary patterns not yet learned)")
            print("    Cooperative play continues with mastered patterns only")
        else:
            # Alternating pairs (every other) - all 3 pair combinations
            if can_do_alternating:
                alternating_pairs = [
                    (student_names[0], student_names[1]),  # Nova-Rêve
                    (student_names[1], student_names[2]),  # Rêve-Alex
                    (student_names[2], student_names[0]),  # Alex-Nova
                ]

                print("    [Alternating pairs - 'every other']")
                for pair in alternating_pairs:
                    # Generate an alternating sequence
                    vocab_size = 26
                    a, b = random.randint(1, vocab_size-1), random.randint(1, vocab_size-1)
                    while b == a:
                        b = random.randint(1, vocab_size-1)

                    # Build sequence: A, B, A, B, A, B, A, B (8 tokens, predict positions 2-7)
                    sequence = [a, b, a, b, a, b, a, b]
                    pair_results = {pair[0]: [], pair[1]: []}

                    # Students take turns predicting (starting from position 2)
                    for pos in range(2, 8):
                        whose_turn = pair[pos % 2]  # Alternates between the two students
                        student = broker.students[whose_turn]

                        # Show sequence up to this point
                        context = sequence[:pos]
                        target = sequence[pos]

                        max_len = 12
                        padded = context + [0] * (max_len - len(context))
                        tokens = torch.tensor(padded[:max_len], dtype=torch.long).unsqueeze(0).to(device)
                        seq_len = [len(context)]

                        student.eval()
                        with torch.no_grad():
                            logits, _, _ = student(tokens, seq_len)
                            pred = logits.argmax(dim=-1).item()
                            correct = (pred == target)

                        pair_results[whose_turn].append(correct)
                        results[whose_turn]['turns_total'] += 1
                        if correct:
                            results[whose_turn]['turns_correct'] += 1

                    # Report pair results
                    p0_acc = sum(pair_results[pair[0]]) / len(pair_results[pair[0]]) if pair_results[pair[0]] else 0
                    p1_acc = sum(pair_results[pair[1]]) / len(pair_results[pair[1]]) if pair_results[pair[1]] else 0
                    print(f"      {pair[0]}-{pair[1]}: {pair[0]} {p0_acc:.0%}, {pair[1]} {p1_acc:.0%}")

            # Ternary trio (every third) - all 3 students together
            if can_do_ternary:
                print("    [Ternary trio - 'every third']")
                vocab_size = 26
                a = random.randint(1, vocab_size-1)
                b = random.randint(1, vocab_size-1)
                while b == a:
                    b = random.randint(1, vocab_size-1)
                c = random.randint(1, vocab_size-1)
                while c == a or c == b:
                    c = random.randint(1, vocab_size-1)

                # Build sequence: A, B, C, A, B, C, A, B, C (9 tokens, predict positions 3-8)
                sequence = [a, b, c, a, b, c, a, b, c]
                trio_results = {name: [] for name in student_names}

                # Students take turns in order (starting from position 3)
                for pos in range(3, 9):
                    whose_turn = student_names[pos % 3]  # Rotates through all 3
                    student = broker.students[whose_turn]

                    # Show sequence up to this point
                    context = sequence[:pos]
                    target = sequence[pos]

                    max_len = 12
                    padded = context + [0] * (max_len - len(context))
                    tokens = torch.tensor(padded[:max_len], dtype=torch.long).unsqueeze(0).to(device)
                    seq_len = [len(context)]

                    student.eval()
                    with torch.no_grad():
                        logits, _, _ = student(tokens, seq_len)
                        pred = logits.argmax(dim=-1).item()
                        correct = (pred == target)

                    trio_results[whose_turn].append(correct)
                    results[whose_turn]['turns_total'] += 1
                    if correct:
                        results[whose_turn]['turns_correct'] += 1

                # Report trio results
                trio_report = ", ".join([f"{name} {sum(trio_results[name])/len(trio_results[name]):.0%}"
                                        if trio_results[name] else f"{name} -"
                                        for name in student_names])
                print(f"      All three: {trio_report}")

    # === PLAYDAY SUMMARY ===
    print("\n  --- Playday Results ---")
    for name in broker.students:
        r = results[name]
        creative_acc = r['creative_correct'] / r['creative_total'] if r['creative_total'] > 0 else 0
        peer_acc = r['peer_correct'] / r['peer_total'] if r['peer_total'] > 0 else 0
        turns_acc = r['turns_correct'] / r['turns_total'] if r['turns_total'] > 0 else 0
        explored = len(r['patterns_explored'])
        print(f"    {name:8s}: Creative {creative_acc:.0%}, Peer {peer_acc:.0%}, Turns {turns_acc:.0%}, Explored {explored} patterns")

    print(f"  {'🎮'*20}\n")

    return results


def main(args):
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = Path(args.data_dir)
    data_dir.mkdir(exist_ok=True)

    # Phased training is ON by default
    phased = not args.no_phase
    max_epochs = args.epochs if args.epochs > 0 else 10000  # 0 = unlimited

    print("=" * 70)
    print("DEVELOPMENTAL CURRICULUM TRAINING")
    print(f"Session: {session_id}")
    print(f"Device: {device}")
    print(f"Year(s): {args.year}")
    print(f"Phased: {phased}")
    print(f"Max epochs: {'unlimited' if args.epochs == 0 else args.epochs}")
    print("=" * 70)

    # Print curriculum
    print_curriculum()

    # Determine available sections for this year
    if args.year == 0:
        available_sections = YEAR_0_SECTIONS
    elif args.year == 1:
        available_sections = YEAR_0_SECTIONS + YEAR_1_SECTIONS  # Year 1 includes Year 0
    elif args.year == 2:
        available_sections = YEAR_0_SECTIONS + YEAR_1_SECTIONS + YEAR_2_SECTIONS  # Full
    else:
        available_sections = ALL_SECTIONS

    # For non-phased training, use all sections at once
    if not phased:
        active_sections = available_sections
        current_phase = len(available_sections) - 1
    else:
        # Phased training starts with first section only
        active_sections = [available_sections[0]]
        current_phase = 0

    # Get ALL patterns for topic tracking (need full size for exam system)
    all_year_patterns = get_patterns_for_sections(available_sections)
    all_pattern_names = [p.name for p in all_year_patterns]
    pattern_to_idx = {name: i for i, name in enumerate(all_pattern_names)}
    n_topics = len(all_pattern_names)

    # Get current active patterns
    active_patterns = get_patterns_for_sections(active_sections)
    active_pattern_names = [p.name for p in active_patterns]

    print(f"\nTotal topics: {n_topics} patterns")
    print(f"Active sections: {active_sections}")
    print(f"Active patterns: {active_pattern_names}")

    # Create initial datasets (only active sections)
    train_data, val_data = create_datasets(active_sections, args)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=collate_fn)

    # Create classroom (sized for ALL patterns, not just active)
    broker = ClassroomBroker(
        student_names=['nova', 'rêve', 'alex'],
        d_model=args.d_model,
        n_heads=args.n_heads,
        vocab_size=26,
        n_topics=n_topics
    ).to(device)

    # Create optimizers
    optimizers = {
        name: torch.optim.AdamW(student.parameters(), lr=args.lr)
        for name, student in broker.students.items()
    }

    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in broker.parameters())
    print(f"\nClassroom: {total_params:,} total parameters")
    for name, student in broker.students.items():
        sp = sum(p.numel() for p in student.parameters())
        print(f"  {student.name}: {sp:,} params")

    print(f"\nData: {len(train_data)} train, {len(val_data)} val")
    print("=" * 70)

    # INITIAL CLASS SESSION for first section (before any training)
    initial_section = active_sections[0] if active_sections else '0A'
    print(f"\n{'='*70}")
    print("FIRST DAY OF SCHOOL!")
    print(f"{'='*70}")
    run_class_session(broker, initial_section, pattern_to_idx, device, year=args.year)

    # Training loop
    history = []
    best_acc = 0
    mastery_level = args.mastery_level  # Level required to advance phase
    section_epochs = 0  # Epochs in current section (resets on section change)
    section_playdays = 0  # Playdays in current section (teacher involvement scales with this)

    for epoch in range(1, max_epochs + 1):
        section_epochs += 1  # Count epochs in this section
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}" + (f"/{max_epochs}" if args.epochs > 0 else ""))
        if phased:
            print(f"Phase {current_phase + 1}/{len(available_sections)}: {active_sections}")
        print("=" * 70)

        # PLAYDAY every 5th epoch IN SECTION (4 work, 1 play) - replaces training
        if section_epochs % 5 == 0:
            section_playdays += 1  # Track playdays in this section
            mastered_patterns = get_mastered_patterns(broker, pattern_to_idx, mastery_level=mastery_level)
            playday_patterns = list(set(mastered_patterns + active_pattern_names))
            # Get current section patterns for mastery showcase
            current_section = active_sections[-1] if active_sections else None
            current_section_patterns = [p.name for p in get_patterns_by_section(current_section)] if current_section else []
            run_playday(broker, playday_patterns, pattern_to_idx, device, epoch, year=args.year,
                       section_playdays=section_playdays,
                       current_section_patterns=current_section_patterns,
                       all_year_patterns=all_year_patterns)

            # No exams on playday - it's a real break!
            # Check section mastery though (in case they crossed threshold during play)
            if phased and current_phase < len(available_sections) - 1:
                current_section = active_sections[-1]
                section_pattern_names = [p.name for p in get_patterns_by_section(current_section)]
                mastered, details = check_section_mastery(
                    broker, section_pattern_names, pattern_to_idx, mastery_level
                )
                if mastered:
                    print(f"\n  {'*'*50}")
                    print(f"  *** SECTION {current_section} MASTERED! ***")
                    print(f"  *** All students at L{mastery_level}+ on: {section_pattern_names} ***")
                    current_phase += 1
                    section_epochs = 0  # Reset epoch counter for new section
                    section_playdays = 0  # Reset playday counter for new section
                    active_sections = available_sections[:current_phase + 1]
                    active_patterns = get_patterns_for_sections(active_sections)
                    active_pattern_names = [p.name for p in active_patterns]
                    print(f"  *** Unlocking section {available_sections[current_phase]}! ***")
                    print(f"  *** Active patterns now: {active_pattern_names} ***")
                    print(f"  {'*'*50}")

                    # CLASS SESSION for new section (I Do, We Do before You Do)
                    new_section = available_sections[current_phase]
                    run_class_session(broker, new_section, pattern_to_idx, device, year=args.year)

                    train_data, val_data = create_datasets(active_sections, args, seed_offset=epoch)
                    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
                    val_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=collate_fn)
            continue  # Skip training on playday

        # "Does anyone have any questions?" - check for struggling students
        # Only on follow-up epochs, Year 1+ (Year 0 is parallel play, no classroom Q&A)
        if section_epochs > 1 and args.year >= 1:
            # Check if any student is struggling on current section patterns
            current_section = active_sections[-1] if active_sections else None
            if current_section:
                section_pattern_names = [p.name for p in get_patterns_by_section(current_section)]
                struggling = []
                for name, student in broker.students.items():
                    for pt in section_pattern_names:
                        if pt in pattern_to_idx:
                            pt_idx = pattern_to_idx[pt]
                            level = student.exam_system.confirmed_level[pt_idx].item()
                            if level < 3:  # Still at early levels
                                struggling.append((name, pt, level))

                if struggling and section_epochs % 3 == 0:  # Check every 3rd epoch
                    # Group by pattern
                    patterns_needing_help = set(p for _, p, _ in struggling)
                    if patterns_needing_help:
                        print(f"\n  📋 Teacher: 'Some students have been working on these...'")
                        print(f"  Patterns needing review: {list(patterns_needing_help)[:3]}")
                        # Quick review - show one example per struggling pattern
                        for pt_name in list(patterns_needing_help)[:2]:
                            pattern_obj = next((p for p in ALL_PATTERNS if p.name == pt_name), None)
                            if pattern_obj:
                                example = pattern_obj.generator(26)
                                seq_str = ' '.join(map(str, example['sequence']))
                                print(f"    Let me show this again - {pt_name}: [{seq_str}] → {example['target']}")

        # Tutoring pairs (only for active patterns, Year 1+ only)
        # Year 0 is cooperative exploration - no hierarchy yet
        if args.year >= 1:
            tutoring_pairs = identify_tutoring_pairs(broker, active_pattern_names, pattern_to_idx)
            active_tutoring = sum(len(p) for p in tutoring_pairs.values())
            if active_tutoring > 0:
                print(f"\n  Peer tutoring: {active_tutoring} pairs")
                # Verbose: show who's tutoring whom on what
                tutor_summary = {}  # tutor -> [(learner, pattern), ...]
                for pt_idx, pairs in tutoring_pairs.items():
                    if pairs and pt_idx < len(active_pattern_names):
                        pattern = active_pattern_names[pt_idx]
                        for learner, tutor in pairs.items():
                            if tutor not in tutor_summary:
                                tutor_summary[tutor] = []
                            tutor_summary[tutor].append((learner, pattern))
                for tutor, assignments in tutor_summary.items():
                    learners = ', '.join(f"{l}({p})" for l, p in assignments)
                    print(f"    {tutor} → {learners}")
        else:
            tutoring_pairs = {}  # No tutoring in Year 0

        # Train
        train_results = train_epoch(
            broker, train_loader, optimizers, criterion,
            device, pattern_to_idx, tutoring_pairs
        )

        # Evaluate
        val_results = evaluate(broker, val_loader, device, pattern_to_idx)

        # Print results
        print("\n  Training:")
        for name in broker.students:
            tr = train_results[name]
            print(f"    {name:8s}: loss={tr['loss']:.4f}, acc={tr['accuracy']:.1%}")

        print("\n  Validation:")
        for name in broker.students:
            vr = val_results[name]
            print(f"    {name:8s}: acc={vr['accuracy']:.1%}")

        # Leaderboard - only shown in Year 1+ after cooperative exploration
        # Year 0 is about individual learning and cooperative play, not competition
        if args.year >= 1:
            ranked = sorted(val_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
            print(f"\n  Leaderboard (Epoch {epoch}):")
            for i, (name, data) in enumerate(ranked):
                student = broker.students[name]
                rank = ["1st", "2nd", "3rd"][i] if i < 3 else f"{i+1}th"
                print(f"    {rank}: {name:8s} {data['accuracy']:.1%}  (XP: {student.xp:6.0f})")

        # Per-pattern detail (every 5 epochs or first 3)
        if epoch <= 3 or epoch % 5 == 0:
            nova = broker.students['nova']
            print(f"\n  Per-pattern (Nova):")
            for pt in active_pattern_names[:8]:  # First 8 active patterns
                if pt in val_results['nova']['per_pattern']:
                    acc = val_results['nova']['per_pattern'][pt]
                    pt_idx = pattern_to_idx[pt]
                    if pt_idx < nova.exam_system.n_topics:
                        confirmed = nova.exam_system.confirmed_level[pt_idx].item()
                        xp_level = nova.topic_tracker.get_level(pt_idx)
                        xp = nova.topic_tracker.progression.topic_xp[pt_idx].item()
                        bar = '█' * int(acc * 10) + '·' * (10 - int(acc * 10))
                        print(f"    {pt:20s}: {acc:3.0%} {bar} L{confirmed}(+{xp_level-confirmed}) ({xp:.0f}xp)")

        # Exams (only for active patterns)
        exam_results = run_exams(broker, active_pattern_names, pattern_to_idx, device, epoch)
        total_exams = sum(len(r) for r in exam_results.values())
        if total_exams > 0:
            print(f"\n  Exams this epoch: {total_exams}")

        # Class average
        class_acc = sum(v['accuracy'] for v in val_results.values()) / len(val_results)
        print(f"\n  Class average: {class_acc:.1%}")

        # Track best
        if class_acc > best_acc:
            best_acc = class_acc
            torch.save({
                'epoch': epoch,
                'broker_state': broker.state_dict(),
                'class_accuracy': class_acc,
                'active_sections': active_sections
            }, data_dir / f'developmental_{session_id}_best.pt')

        # PHASED TRAINING: Check if ready to advance
        if phased and current_phase < len(available_sections) - 1:
            # Check mastery on current section (the latest added one)
            current_section = active_sections[-1]
            section_pattern_names = [p.name for p in get_patterns_by_section(current_section)]

            mastered, details = check_section_mastery(
                broker, section_pattern_names, pattern_to_idx, mastery_level
            )

            if mastered:
                print(f"\n  {'*'*50}")
                print(f"  *** SECTION {current_section} MASTERED! ***")
                print(f"  *** All students at L{mastery_level}+ on: {section_pattern_names} ***")

                # Advance to next phase
                current_phase += 1
                section_epochs = 0  # Reset epoch counter for new section
                section_playdays = 0  # Reset playday counter for new section
                active_sections = available_sections[:current_phase + 1]
                active_patterns = get_patterns_for_sections(active_sections)
                active_pattern_names = [p.name for p in active_patterns]

                print(f"  *** Unlocking section {available_sections[current_phase]}! ***")
                print(f"  *** Active patterns now: {active_pattern_names} ***")
                print(f"  {'*'*50}")

                # CLASS SESSION for new section (I Do, We Do before You Do)
                new_section = available_sections[current_phase]
                run_class_session(broker, new_section, pattern_to_idx, device, year=args.year)

                # CELEBRATION PLAYDAY - they earned it!
                # Play with all mastered patterns before tackling new section
                # This is "playday 1" of the new section - fresh start, free exploration
                section_playdays = 1
                mastered_for_play = get_mastered_patterns(broker, pattern_to_idx, mastery_level=mastery_level)
                playday_patterns = list(set(mastered_for_play + active_pattern_names))
                print(f"\n  *** CELEBRATION PLAYDAY for mastering {current_section}! ***")
                # New section patterns for mastery showcase
                new_section = active_sections[-1] if active_sections else None
                new_section_patterns = [p.name for p in get_patterns_by_section(new_section)] if new_section else []
                run_playday(broker, playday_patterns, pattern_to_idx, device, epoch, year=args.year,
                           section_playdays=section_playdays,
                           current_section_patterns=new_section_patterns,
                           all_year_patterns=all_year_patterns)

                # Regenerate datasets with new patterns
                train_data, val_data = create_datasets(active_sections, args, seed_offset=epoch)
                train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
                val_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=collate_fn)

            else:
                # Show progress toward mastery
                print(f"\n  Section {current_section} progress (need L{mastery_level}):")
                for name, levels in details.items():
                    level_str = ", ".join(f"{p}:L{l}" for p, l in levels.items())
                    print(f"    {name}: {level_str}")

        # Graduation check (all patterns at L10)
        all_graduated = True
        for student in broker.students.values():
            for pt_idx in range(n_topics):
                if pt_idx >= student.exam_system.n_topics:
                    continue
                if not student.exam_system.topic_graduated[pt_idx]:
                    all_graduated = False
                    break
            if not all_graduated:
                break

        if all_graduated:
            print(f"\n{'*'*60}")
            print(f"*** CLASS GRADUATED! All patterns mastered! ***")
            print(f"*** Final accuracy: {class_acc:.1%} ***")
            print(f"{'*'*60}")
            break

        history.append({
            'epoch': epoch,
            'class_acc': class_acc,
            'phase': current_phase,
            'active_sections': active_sections.copy()
        })

    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Session: {session_id}")
    print(f"Epochs: {epoch}")
    print(f"Best accuracy: {best_acc:.1%}")
    if phased:
        print(f"Final phase: {current_phase + 1}/{len(available_sections)}")
        print(f"Active sections: {active_sections}")

    print("\nFinal standings:")
    for name, student in broker.students.items():
        acc = val_results[name]['accuracy']
        grads = student.exam_system.topic_graduated[:n_topics].sum().item()
        print(f"  {student.name}: {acc:.1%} | XP: {student.xp:.0f} | Graduated: {int(grads)}/{n_topics}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=1, help='Year to train: 0=number sense only, 1=0+patterns, 2=full curriculum')
    parser.add_argument('--no-phase', action='store_true', dest='no_phase', help='Disable phased training (train all patterns at once)')
    parser.add_argument('--mastery_level', type=int, default=10, help='Level required to advance phase (default: 10)')
    parser.add_argument('--epochs', type=int, default=0, help='Max epochs (0 = unlimited, train until graduation)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--train_size', type=int, default=20000)
    parser.add_argument('--val_size', type=int, default=2000)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--data_dir', type=str, default='data')

    args = parser.parse_args()
    main(args)
