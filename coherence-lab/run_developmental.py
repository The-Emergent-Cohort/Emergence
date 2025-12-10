#!/usr/bin/env python3
"""
Run Developmental Curriculum Training

Trains students through Years 1-2 of the developmental curriculum.
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
    YEAR_1_PATTERNS, YEAR_2_PATTERNS, ALL_PATTERNS,
    get_pattern_names, get_patterns_by_section, print_curriculum,
    PlaydaySpec, PLAYDAY_SPECS, get_playday_spec,
    TEACHER_NOTES, get_teacher_notes
)
import random
from systems import ExaminationSystem
from systems.progression import TopicTracker


def get_pattern_by_name(name: str):
    """Look up a pattern by name."""
    for p in ALL_PATTERNS:
        if p.name == name:
            return p
    return None


# Section order for phased training
# Phase 1: Foundational Numeracy (1A-1F)
YEAR_1_SECTIONS = ['1A', '1B', '1C', '1D', '1E', '1F']
# Phase 2: Relations & Physics (2A-2E)
YEAR_2_SECTIONS = ['2A', '2B', '2C', '2D', '2E']
ALL_SECTIONS = YEAR_1_SECTIONS + YEAR_2_SECTIONS


class StudentNotebook:
    """
    Working notebook for a student - temporary scaffolding for learning.

    Tracks:
    - Pattern uncertainties (what's confusing)
    - Working notes (observations about patterns)
    - Sticky notes (things that need reminder even after graduation)

    Lifecycle:
    - Created when entering a phase
    - Used during learning
    - Consulted before exams (cramming)
    - Mostly cleared on graduation, sticky notes carry forward
    """

    def __init__(self, student_name: str):
        self.student_name = student_name
        self.uncertainties = {}      # pattern -> list of (epoch, confidence, notes)
        self.working_notes = {}      # pattern -> list of observations
        self.sticky_notes = {}       # pattern -> notes that persist across phases
        self.current_phase = None

    def enter_phase(self, phase: str):
        """Start a new phase - fresh notebook, but keep sticky notes."""
        self.current_phase = phase
        self.uncertainties = {}
        self.working_notes = {}
        # sticky_notes persist

    def note_uncertainty(self, pattern: str, epoch: int, confidence: float, correct: bool):
        """Record uncertainty on a pattern."""
        if pattern not in self.uncertainties:
            self.uncertainties[pattern] = []
        self.uncertainties[pattern].append({
            'epoch': epoch,
            'confidence': confidence,
            'correct': correct
        })

    def add_working_note(self, pattern: str, note: str):
        """Add a working note about a pattern."""
        if pattern not in self.working_notes:
            self.working_notes[pattern] = []
        self.working_notes[pattern].append(note)

    def mark_sticky(self, pattern: str, note: str):
        """Mark something as needing reminder even after graduation."""
        if pattern not in self.sticky_notes:
            self.sticky_notes[pattern] = []
        self.sticky_notes[pattern].append(note)

    def get_trouble_patterns(self) -> list:
        """Get patterns this student has noted uncertainty on."""
        trouble = []
        for pattern, entries in self.uncertainties.items():
            if len(entries) >= 2:  # Multiple struggles
                trouble.append(pattern)
            elif entries and not entries[-1]['correct']:  # Recent failure
                trouble.append(pattern)
        return trouble

    def cram_summary(self) -> dict:
        """Get summary for cramming before exam."""
        return {
            'trouble_patterns': self.get_trouble_patterns(),
            'sticky_notes': dict(self.sticky_notes),
            'recent_uncertainties': {
                p: entries[-1] for p, entries in self.uncertainties.items() if entries
            }
        }

    def graduate_phase(self, patterns_still_tricky: list = None):
        """Graduate from current phase. Clear working notes, promote tricky to sticky."""
        patterns_still_tricky = patterns_still_tricky or []
        for pattern in patterns_still_tricky:
            if pattern in self.uncertainties and len(self.uncertainties[pattern]) >= 2:
                self.mark_sticky(pattern, f"Struggled in phase {self.current_phase}")
        self.uncertainties = {}
        self.working_notes = {}


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
    if year == 1:
        sections = YEAR_1_SECTIONS
    elif year == 2:
        sections = YEAR_2_SECTIONS
    else:  # year 0 = both
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


def run_teacher_demonstration(broker, pattern_obj, device, vocab_size=26):
    """
    "I Do" phase - Teacher demonstrates a pattern with explanation.

    Following Gradual Release of Responsibility:
    1. Show worked example
    2. Explain the pattern explicitly
    3. Students observe and listen

    Uses Teacher's Edition notes from curriculum (like textbook teacher's guide).
    """
    # Get teacher's edition notes
    notes = get_teacher_notes(pattern_obj.name)

    print(f"\n  📚 Teacher Demonstration: {pattern_obj.name}")
    print(f"     \"{notes['intro']}\"")

    # Generate 3 worked examples
    print("\n     Worked examples:")
    for i in range(3):
        example = pattern_obj.generator(vocab_size)
        seq = example['sequence']
        target = example['target']

        # Show the sequence with explanation
        seq_str = ', '.join(str(x) for x in seq)
        print(f"       [{seq_str}, ?] → {target}")

    # Teacher explains the pattern (from textbook)
    print(f"\n     💡 Explanation: {notes['explain']}")

    # Show the scripted worked example from teacher's edition
    if notes.get('worked_example'):
        print(f"     📖 \"{notes['worked_example']}\"")

    # What to watch for (teacher notes)
    print(f"     👀 Watch for: {notes['watch_for']}")

    return True


def run_guided_practice(broker, pattern_obj, device, vocab_size=26, n_problems=5):
    """
    "We Do" phase - Guided practice with scaffolding and feedback.

    Students predict, then see the answer with explanation.
    Target ~80% success before moving to independent practice.
    """
    print(f"\n  🤝 Guided Practice: {pattern_obj.name}")

    results = {name: {'correct': 0, 'total': 0} for name in broker.students.keys()}

    for problem_num in range(n_problems):
        example = pattern_obj.generator(vocab_size)
        seq = example['sequence']
        target = example['target']

        # Prepare input
        max_len = 12
        padded = seq + [0] * (max_len - len(seq))
        tokens = torch.tensor(padded[:max_len], dtype=torch.long).unsqueeze(0).to(device)
        seq_len = [len(seq)]

        seq_str = ', '.join(str(x) for x in seq)
        print(f"     Problem {problem_num + 1}: [{seq_str}, ?]")

        # Each student predicts
        predictions = {}
        for name, student in broker.students.items():
            student.eval()
            with torch.no_grad():
                logits, _, _ = student(tokens, seq_len)
                pred = logits.argmax(dim=-1).item()
                predictions[name] = pred
                results[name]['total'] += 1
                if pred == target:
                    results[name]['correct'] += 1

        # Show predictions and give feedback
        pred_strs = []
        for name, pred in predictions.items():
            marker = "✓" if pred == target else "✗"
            pred_strs.append(f"{name}:{pred}{marker}")

        print(f"       Predictions: {', '.join(pred_strs)}")
        print(f"       Answer: {target}")

        # Brief feedback if anyone got it wrong
        wrong = [name for name, pred in predictions.items() if pred != target]
        if wrong and problem_num < n_problems - 1:
            print(f"       (Keep watching the pattern, {', '.join(wrong)}!)")

    # Summary
    print("     Results:")
    for name in broker.students:
        acc = results[name]['correct'] / max(1, results[name]['total'])
        bar = '█' * int(acc * 5)
        print(f"       {name}: {results[name]['correct']}/{results[name]['total']} {bar}")

    # Return average success rate
    total_correct = sum(r['correct'] for r in results.values())
    total_problems = sum(r['total'] for r in results.values())
    avg_success = total_correct / max(1, total_problems)

    return avg_success


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


def calculate_star_awards(results, playday_spec):
    """
    Calculate star awards for each student based on playday performance.

    Returns: {student_name: {category: 'gold'/'silver'/'bronze'/None}}

    Gold: >=90% or best in class
    Silver: >=75%
    Bronze: >=50%
    """
    awards = {}
    categories = list(playday_spec.star_categories.keys())

    # Map performance metrics to categories
    metric_map = {
        'accuracy': lambda r: r['creative_correct'] / max(1, r['creative_total']),
        'patience': lambda r: 1.0 - (r.get('rushed_answers', 0) / max(1, r['creative_total'])),
        'curiosity': lambda r: len(r['patterns_explored']) / max(1, len(r.get('available_patterns', ['x']))),
        'creativity': lambda r: r.get('creative_variants', 0) / max(1, r['creative_total']),
        'teamwork': lambda r: r['turns_correct'] / max(1, r['turns_total']),
        'rhythm': lambda r: r['turns_correct'] / max(1, r['turns_total']),
        'memory': lambda r: r.get('memory_score', 0.5),
        'focus': lambda r: r['creative_correct'] / max(1, r['creative_total']),
        'speed': lambda r: r.get('speed_score', 0.5),
        'backwards': lambda r: r.get('backwards_score', 0.5),
        'pattern': lambda r: r['peer_correct'] / max(1, r['peer_total']),
        'insight': lambda r: r.get('insight_score', 0.5),
        'transfer': lambda r: r['peer_correct'] / max(1, r['peer_total']),
        'physics': lambda r: r.get('physics_score', 0.5),
        'intuition': lambda r: r.get('intuition_score', 0.5),
        'conservation': lambda r: r.get('conservation_score', 0.5),
        'logic': lambda r: r['creative_correct'] / max(1, r['creative_total']),
        'prediction': lambda r: r['peer_correct'] / max(1, r['peer_total']),
        # 1A foundational: confidence in trying without hints
        'confidence': lambda r: r['count_correct'] / max(1, r['count_total']),
    }

    for name, r in results.items():
        awards[name] = {}
        for cat in categories:
            # Get score for this category (default to accuracy if not mapped)
            metric_fn = metric_map.get(cat, metric_map['accuracy'])
            score = metric_fn(r)

            # Assign medal
            if score >= 0.90:
                awards[name][cat] = 'gold'
            elif score >= 0.75:
                awards[name][cat] = 'silver'
            elif score >= 0.50:
                awards[name][cat] = 'bronze'
            else:
                awards[name][cat] = None

    return awards


def run_counting_activity(broker, device, activity_type='count_up'):
    """
    Run a counting activity for younger students.

    Activities:
    - count_up: 1, 2, 3, 4... (incrementing)
    - count_down: 10, 9, 8... (decrementing)
    - skip_count: 2, 4, 6... (fixed offset)
    - count_together: students take turns counting
    - whats_next: [n, ?] → n+1 (foundational +1)
    - whats_before: [n, ?] → n-1 (foundational -1)
    """
    results = {name: {'correct': 0, 'total': 0} for name in broker.students.keys()}
    student_names = list(broker.students.keys())

    if activity_type == 'count_up':
        # Each student counts up from a random start
        for name, student in broker.students.items():
            start = random.randint(1, 5)
            length = random.randint(4, 6)
            sequence = list(range(start, start + length))
            target = start + length

            max_len = 12
            padded = sequence + [0] * (max_len - len(sequence))
            tokens = torch.tensor(padded[:max_len], dtype=torch.long).unsqueeze(0).to(device)
            seq_len = [len(sequence)]

            student.eval()
            with torch.no_grad():
                logits, _, _ = student(tokens, seq_len)
                pred = logits.argmax(dim=-1).item()
                correct = (pred == target)

            results[name]['total'] += 1
            if correct:
                results[name]['correct'] += 1

    elif activity_type == 'count_down':
        # Each student counts down
        for name, student in broker.students.items():
            start = random.randint(8, 15)
            length = random.randint(4, 6)
            sequence = list(range(start, start - length, -1))
            target = start - length

            if target >= 0:
                max_len = 12
                padded = sequence + [0] * (max_len - len(sequence))
                tokens = torch.tensor(padded[:max_len], dtype=torch.long).unsqueeze(0).to(device)
                seq_len = [len(sequence)]

                student.eval()
                with torch.no_grad():
                    logits, _, _ = student(tokens, seq_len)
                    pred = logits.argmax(dim=-1).item()
                    correct = (pred == target)

                results[name]['total'] += 1
                if correct:
                    results[name]['correct'] += 1

    elif activity_type == 'skip_count':
        # Count by 2s or 3s
        for name, student in broker.students.items():
            step = random.choice([2, 3])
            start = random.randint(0, 3)
            length = random.randint(4, 5)
            sequence = [start + i * step for i in range(length)]
            target = start + length * step

            if target < 26:
                max_len = 12
                padded = sequence + [0] * (max_len - len(sequence))
                tokens = torch.tensor(padded[:max_len], dtype=torch.long).unsqueeze(0).to(device)
                seq_len = [len(sequence)]

                student.eval()
                with torch.no_grad():
                    logits, _, _ = student(tokens, seq_len)
                    pred = logits.argmax(dim=-1).item()
                    correct = (pred == target)

                results[name]['total'] += 1
                if correct:
                    results[name]['correct'] += 1

    elif activity_type == 'count_together':
        # Students take turns counting - collaborative!
        start = random.randint(1, 5)
        length = 9  # 3 turns each
        sequence = list(range(start, start + length))

        for pos in range(3, length):  # Start predictions after seeing 3 numbers
            whose_turn = student_names[pos % 3]
            student = broker.students[whose_turn]
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

            results[whose_turn]['total'] += 1
            if correct:
                results[whose_turn]['correct'] += 1

    elif activity_type == 'whats_next':
        # Simple +1 operation: given n, predict n+1
        for name, student in broker.students.items():
            for _ in range(5):  # 5 problems each
                n = random.randint(0, 23)  # Leave room for +1
                # Can give hint or not
                if random.random() < 0.3:
                    # With hint: show a previous +1 example
                    prev = random.randint(0, 22)
                    sequence = [prev, prev + 1, n]
                else:
                    sequence = [n]
                target = n + 1

                max_len = 12
                padded = sequence + [0] * (max_len - len(sequence))
                tokens = torch.tensor(padded[:max_len], dtype=torch.long).unsqueeze(0).to(device)
                seq_len = [len(sequence)]

                student.eval()
                with torch.no_grad():
                    logits, _, _ = student(tokens, seq_len)
                    pred = logits.argmax(dim=-1).item()
                    correct = (pred == target)

                results[name]['total'] += 1
                if correct:
                    results[name]['correct'] += 1

    elif activity_type == 'whats_before':
        # Simple -1 operation: given n, predict n-1
        for name, student in broker.students.items():
            for _ in range(5):  # 5 problems each
                n = random.randint(1, 24)  # Leave room for -1
                # Can give hint or not
                if random.random() < 0.3:
                    # With hint: show a previous -1 example
                    prev = random.randint(2, 24)
                    sequence = [prev, prev - 1, n]
                else:
                    sequence = [n]
                target = n - 1

                max_len = 12
                padded = sequence + [0] * (max_len - len(sequence))
                tokens = torch.tensor(padded[:max_len], dtype=torch.long).unsqueeze(0).to(device)
                seq_len = [len(sequence)]

                student.eval()
                with torch.no_grad():
                    logits, _, _ = student(tokens, seq_len)
                    pred = logits.argmax(dim=-1).item()
                    correct = (pred == target)

                results[name]['total'] += 1
                if correct:
                    results[name]['correct'] += 1

    return results


def run_playday(broker, mastered_patterns, pattern_to_idx, device, epoch, vocab_size=26,
                current_section='1A'):
    """
    Run a playday session - exploration without grades!

    Features:
    - Curriculum-aware activities based on current section
    - Star awards: gold/silver/bronze for different strengths
    - PARTY TIME when everyone gets all gold stars
    - Counting and math activities for younger students
    - Turn-taking and rhythm games
    - No penalties: wrong answers don't hurt XP
    """
    # Get playday spec for current section
    playday_spec = get_playday_spec(current_section)

    print(f"\n  {'🎮'*20}")
    print(f"  *** PLAYDAY! (Epoch {epoch}) ***")
    print(f"  Section: {current_section}")
    print(f"  Focus: {', '.join(playday_spec.focus_skills)}")
    print(f"  Activities: {', '.join(playday_spec.activities)}")
    print(f"  Unlocked patterns: {len(mastered_patterns)}")
    print(f"  {'🎮'*20}")

    if not mastered_patterns:
        print("  No mastered patterns yet - let's do some fun activities anyway!")

    results = {name: {
        'creative_correct': 0, 'creative_total': 0,
        'peer_correct': 0, 'peer_total': 0,
        'turns_correct': 0, 'turns_total': 0,
        'count_correct': 0, 'count_total': 0,
        'patterns_explored': set(),
        'available_patterns': mastered_patterns
    } for name in broker.students.keys()}

    # === PHASE 1: COUNTING ACTIVITIES (for early sections) ===
    if current_section in ['1A', '1B', '1C', '1D', '1E', '1F']:
        print("\n  --- Counting Time! ---")

        # 1A foundational activities: what's next/before
        if 'whats_next' in playday_spec.activities:
            print("    [What comes NEXT? (+1)]")
            count_results = run_counting_activity(broker, device, 'whats_next')
            for name in broker.students:
                results[name]['count_correct'] += count_results[name]['correct']
                results[name]['count_total'] += count_results[name]['total']
            for name in broker.students:
                acc = count_results[name]['correct'] / max(1, count_results[name]['total'])
                print(f"      {name}: {count_results[name]['correct']}/{count_results[name]['total']} ({acc:.0%})")

        if 'whats_before' in playday_spec.activities:
            print("    [What comes BEFORE? (-1)]")
            count_results = run_counting_activity(broker, device, 'whats_before')
            for name in broker.students:
                results[name]['count_correct'] += count_results[name]['correct']
                results[name]['count_total'] += count_results[name]['total']
            for name in broker.students:
                acc = count_results[name]['correct'] / max(1, count_results[name]['total'])
                print(f"      {name}: {count_results[name]['correct']}/{count_results[name]['total']} ({acc:.0%})")

        # Count up / count together
        if 'count_up' in playday_spec.activities or 'count_together' in playday_spec.activities:
            print("    [Counting up together]")
            count_results = run_counting_activity(broker, device, 'count_together')
            for name in broker.students:
                results[name]['count_correct'] += count_results[name]['correct']
                results[name]['count_total'] += count_results[name]['total']
            # Report
            for name in broker.students:
                acc = count_results[name]['correct'] / max(1, count_results[name]['total'])
                print(f"      {name}: {count_results[name]['correct']}/{count_results[name]['total']} ({acc:.0%})")

        # Count down (for 1D+)
        if 'count_down' in playday_spec.activities:
            print("    [Counting backwards]")
            count_results = run_counting_activity(broker, device, 'count_down')
            for name in broker.students:
                results[name]['count_correct'] += count_results[name]['correct']
                results[name]['count_total'] += count_results[name]['total']
                results[name]['backwards_score'] = count_results[name]['correct'] / max(1, count_results[name]['total'])

        # Skip counting (for 1E+)
        if 'skip_count' in playday_spec.activities:
            print("    [Skip counting (2s and 3s)]")
            count_results = run_counting_activity(broker, device, 'skip_count')
            for name in broker.students:
                results[name]['count_correct'] += count_results[name]['correct']
                results[name]['count_total'] += count_results[name]['total']

    # === PHASE 2: CREATIVE CHALLENGES FROM TEACHER ===
    if mastered_patterns:
        print("\n  --- Teacher's Creative Challenges ---")
        n_challenges = min(10, len(mastered_patterns) * 3)

        for _ in range(n_challenges):
            challenge, description = generate_creative_challenge(mastered_patterns, vocab_size)
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

    # === PHASE 3: PEER CHALLENGES ===
    if mastered_patterns:
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

    # === PHASE 4: TURN-TAKING / RHYTHM ACTIVITIES ===
    if 'turn_taking' in playday_spec.activities or 'rhythm_game' in playday_spec.activities:
        print("\n  --- Turn-Taking & Rhythm ---")
        student_names = list(broker.students.keys())

        # Alternating pairs (every other) - all 3 pair combinations
        alternating_pairs = [
            (student_names[0], student_names[1]),  # Nova-Rêve
            (student_names[1], student_names[2]),  # Rêve-Alex
            (student_names[2], student_names[0]),  # Alex-Nova
        ]

        print("    [Alternating pairs - 'every other']")
        for pair in alternating_pairs:
            # Generate an alternating sequence
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
        print("    [Ternary trio - 'every third']")
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

    # === STAR AWARDS ===
    print("\n  --- Star Awards Ceremony ---")
    awards = calculate_star_awards(results, playday_spec)

    medal_symbols = {'gold': '🥇', 'silver': '🥈', 'bronze': '🥉', None: '  '}

    for name in broker.students:
        student_awards = awards[name]
        medals = []
        gold_count = 0
        for cat, level in student_awards.items():
            if level:
                symbol = medal_symbols[level]
                desc = playday_spec.star_categories[cat]
                medals.append(f"{symbol} {desc}")
                if level == 'gold':
                    gold_count += 1
        print(f"    {name}:")
        if medals:
            for medal in medals:
                print(f"      {medal}")
        else:
            print(f"      Keep trying! 💪")
        results[name]['gold_stars'] = gold_count

    # === PARTY TIME CHECK ===
    all_gold = all(
        all(level == 'gold' for level in awards[name].values())
        for name in broker.students
    )

    if all_gold:
        print(f"\n  {'🎉'*20}")
        print(f"  *** PARTY TIME!!! ***")
        print(f"  *** EVERYONE GOT ALL GOLD STARS! ***")
        print(f"  {'🎉'*20}")
        # Bonus XP for party time
        for name, student in broker.students.items():
            for pt in mastered_patterns:
                if pt in pattern_to_idx:
                    pt_idx = pattern_to_idx[pt]
                    student.topic_tracker.award_xp(pt_idx, 2.0)  # Party bonus!
    else:
        # Check if anyone got all golds
        for name in broker.students:
            if all(level == 'gold' for level in awards[name].values()):
                print(f"\n  🌟 {name} got ALL GOLD STARS! 🌟")

    # === PLAYDAY SUMMARY ===
    print("\n  --- Playday Summary ---")
    for name in broker.students:
        r = results[name]
        creative_acc = r['creative_correct'] / max(1, r['creative_total'])
        peer_acc = r['peer_correct'] / max(1, r['peer_total'])
        turns_acc = r['turns_correct'] / max(1, r['turns_total'])
        count_acc = r['count_correct'] / max(1, r['count_total'])
        explored = len(r['patterns_explored'])
        golds = r.get('gold_stars', 0)
        print(f"    {name:8s}: Creative {creative_acc:.0%}, Peer {peer_acc:.0%}, Turns {turns_acc:.0%}, Count {count_acc:.0%} | {golds}🥇 | Explored {explored}")

    print(f"  {'🎮'*20}\n")

    return results


def run_question_period(broker, active_pattern_names, pattern_to_idx, device,
                        epochs_on_topic, notebooks=None, epoch=0, vocab_size=26):
    """
    Question period at start of epoch - students surface uncertainty.
    """
    if epochs_on_topic <= 1:
        return {}  # First epoch on topic - no questions yet

    print(f"\n  📋 Question Period (day {epochs_on_topic} on topic)")

    uncertainties = {name: {} for name in broker.students.keys()}

    for pattern_name in active_pattern_names:
        if pattern_name not in pattern_to_idx:
            continue

        pattern = get_pattern_by_name(pattern_name)
        if pattern is None:
            continue

        example = pattern.generator(vocab_size)
        if len(example['sequence']) < 2:
            continue

        max_len = 12
        seq = example['sequence']
        padded = seq + [0] * (max_len - len(seq))
        tokens = torch.tensor(padded[:max_len], dtype=torch.long).unsqueeze(0).to(device)
        target = example['target']
        seq_len = [len(seq)]

        for name, student in broker.students.items():
            student.eval()
            with torch.no_grad():
                logits, _, _ = student(tokens, seq_len)
                probs = torch.softmax(logits, dim=-1)
                pred = logits.argmax(dim=-1).item()
                confidence = probs[0, pred].item()
                correct = (pred == target)

            uncertain = not correct or confidence < 0.7
            uncertainties[name][pattern_name] = {
                'correct': correct,
                'confidence': confidence,
                'uncertain': uncertain
            }

            # Record in notebook if available
            if notebooks and name in notebooks and uncertain:
                notebooks[name].note_uncertainty(pattern_name, epoch, confidence, correct)

    # Report class uncertainties
    confused_patterns = set()
    for name in broker.students:
        uncertain_list = [p for p, v in uncertainties[name].items() if v['uncertain']]
        if uncertain_list:
            confused_patterns.update(uncertain_list)

    if confused_patterns:
        print(f"    Students showing uncertainty on: {list(confused_patterns)}")
    else:
        print(f"    Class feeling confident today!")

    return uncertainties


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
    if args.year == 1:
        available_sections = YEAR_1_SECTIONS
    elif args.year == 2:
        available_sections = YEAR_2_SECTIONS
    else:  # year 0 = both
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

    # Create student notebooks
    notebooks = {
        name: StudentNotebook(name) for name in broker.students.keys()
    }
    starting_phase = active_sections[0] if phased else '1A'
    for notebook in notebooks.values():
        notebook.enter_phase(starting_phase)

    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in broker.parameters())
    print(f"\nClassroom: {total_params:,} total parameters")
    for name, student in broker.students.items():
        sp = sum(p.numel() for p in student.parameters())
        print(f"  {student.name}: {sp:,} params")

    print(f"\nData: {len(train_data)} train, {len(val_data)} val")
    print("=" * 70)

    # Training loop
    history = []
    best_acc = 0
    mastery_level = args.mastery_level  # Level required to advance phase
    epochs_on_phase = 0  # Track time on current topic
    pending_celebration = None  # Section just mastered

    for epoch in range(1, max_epochs + 1):
        epochs_on_phase += 1

        print(f"\n{'='*70}")
        print(f"Epoch {epoch}" + (f"/{max_epochs}" if args.epochs > 0 else ""))
        if phased:
            print(f"Phase {current_phase + 1}/{len(available_sections)}: {active_sections} (day {epochs_on_phase})")
        print("=" * 70)

        # CELEBRATION PLAYDAY - runs the day after mastering a section
        if pending_celebration is not None:
            print(f"\n  *** CELEBRATION PLAYDAY! (mastered {pending_celebration}) ***")
            mastered_for_play = get_mastered_patterns(broker, pattern_to_idx, mastery_level=mastery_level)
            celebration_patterns = list(set(mastered_for_play + active_pattern_names))
            run_playday(broker, celebration_patterns, pattern_to_idx, device, epoch,
                       current_section=active_sections[-1])
            pending_celebration = None
            continue

        # QUESTION PERIOD at start of epoch
        run_question_period(broker, active_pattern_names, pattern_to_idx, device,
                           epochs_on_phase, notebooks, epoch)

        # "I DO, WE DO" - Teacher demonstration on day 1 of new section
        if epochs_on_phase == 1 and phased:
            current_section = active_sections[-1]
            section_patterns = get_patterns_by_section(current_section)
            print(f"\n  {'='*50}")
            print(f"  📖 NEW SECTION: {current_section} - Teacher's Introduction")
            print(f"  {'='*50}")

            for pattern_obj in section_patterns:
                # "I Do" - Teacher demonstrates
                run_teacher_demonstration(broker, pattern_obj, device)

                # "We Do" - Guided practice
                success_rate = run_guided_practice(broker, pattern_obj, device, n_problems=5)

                # Feedback on readiness
                if success_rate >= 0.8:
                    print(f"     ✓ Class ready for independent practice!")
                elif success_rate >= 0.6:
                    print(f"     ~ Class needs more practice, but can continue.")
                else:
                    print(f"     ✗ Class struggling - will need more examples.")

            print(f"  {'='*50}\n")

        # PLAYDAY every 5th epoch on topic (4 work, 1 play)
        if epochs_on_phase % 5 == 0:
            mastered_patterns = get_mastered_patterns(broker, pattern_to_idx, mastery_level=mastery_level)
            playday_patterns = list(set(mastered_patterns + active_pattern_names))
            run_playday(broker, playday_patterns, pattern_to_idx, device, epoch,
                       current_section=active_sections[-1])

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

                    # Graduate notebooks
                    for name, notebook in notebooks.items():
                        trouble = notebook.get_trouble_patterns()
                        notebook.graduate_phase(trouble)

                    current_phase += 1
                    epochs_on_phase = 0
                    active_sections = available_sections[:current_phase + 1]
                    active_patterns = get_patterns_for_sections(active_sections)
                    active_pattern_names = [p.name for p in active_patterns]

                    # Enter new phase in notebooks
                    new_section = available_sections[current_phase]
                    for notebook in notebooks.values():
                        notebook.enter_phase(new_section)

                    print(f"  *** Unlocking section {new_section}! ***")
                    print(f"  *** Active patterns now: {active_pattern_names} ***")
                    print(f"  {'*'*50}")
                    train_data, val_data = create_datasets(active_sections, args, seed_offset=epoch)
                    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
                    val_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=collate_fn)
                    pending_celebration = current_section
            continue  # Skip training on playday

        # Tutoring pairs (only for active patterns)
        tutoring_pairs = identify_tutoring_pairs(broker, active_pattern_names, pattern_to_idx)
        active_tutoring = sum(len(p) for p in tutoring_pairs.values())
        if active_tutoring > 0:
            print(f"\n  Peer tutoring: {active_tutoring} pairs")

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

        # Leaderboard
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

        # Cramming before exams
        any_cramming = False
        for name, notebook in notebooks.items():
            cram = notebook.cram_summary()
            if cram['trouble_patterns']:
                if not any_cramming:
                    print("\n  📖 Pre-exam cramming:")
                    any_cramming = True
                print(f"    {name} reviewing: {cram['trouble_patterns']}")

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

                # Graduate notebooks
                for name, notebook in notebooks.items():
                    trouble = notebook.get_trouble_patterns()
                    notebook.graduate_phase(trouble)

                # Advance to next phase
                current_phase += 1
                epochs_on_phase = 0
                active_sections = available_sections[:current_phase + 1]
                active_patterns = get_patterns_for_sections(active_sections)
                active_pattern_names = [p.name for p in active_patterns]

                # Enter new phase in notebooks
                new_section = available_sections[current_phase]
                for notebook in notebooks.values():
                    notebook.enter_phase(new_section)

                print(f"  *** Unlocking section {new_section}! ***")
                print(f"  *** Active patterns now: {active_pattern_names} ***")
                print(f"  {'*'*50}")

                # Regenerate datasets with new patterns
                train_data, val_data = create_datasets(active_sections, args, seed_offset=epoch)
                train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
                val_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=collate_fn)
                pending_celebration = current_section

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
    parser.add_argument('--year', type=int, default=1, help='Year to train (1, 2, or 0 for both)')
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
