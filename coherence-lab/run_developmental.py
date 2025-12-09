#!/usr/bin/env python3
"""
Run Developmental Curriculum Training

Trains students through Years 1-2 of the developmental curriculum.
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
    get_pattern_names, print_curriculum
)
from systems import ExaminationSystem
from systems.progression import TopicTracker


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


def train_epoch(broker, loader, optimizers, criterion, device, pattern_to_idx, tutoring_pairs=None):
    """Train all students for one epoch."""
    broker.train()
    broker.wake_all()

    tutoring_pairs = tutoring_pairs or {}
    SHOW_THRESHOLD = 0.7
    APPROVAL_BONUS = 3.0
    TUTOR_WEIGHT = 0.3
    TUTOR_TEMP = 2.0
    TUTOR_XP = 0.5

    results = {name: {
        'loss': 0.0, 'correct': 0, 'total': 0,
        'show_count': 0, 'approval_count': 0,
        'tutoring_received': 0, 'tutoring_given': 0
    } for name in broker.students.keys()}

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

                    if tutor_name not in tutor_cache:
                        tutor_cache[tutor_name] = get_tutor_predictions(tutor, tokens, seq_lens, TUTOR_TEMP)

                    student_log_probs = F.log_softmax(logits[i:i+1] / TUTOR_TEMP, dim=-1)
                    tutor_probs = tutor_cache[tutor_name][i:i+1]

                    kl = F.kl_div(student_log_probs, tutor_probs, reduction='batchmean')
                    tutoring_loss = tutoring_loss + kl
                    tutor_helped = True

                    results[name]['tutoring_received'] += 1
                    results[tutor_name]['tutoring_given'] += 1

                    if not tutor.exam_system.topic_graduated[pt_idx]:
                        tutor.topic_tracker.award_xp(pt_idx, TUTOR_XP)

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


def main(args):
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = Path(args.data_dir)
    data_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("DEVELOPMENTAL CURRICULUM TRAINING")
    print(f"Session: {session_id}")
    print(f"Device: {device}")
    print(f"Year(s): {args.year}")
    print("=" * 70)

    # Print curriculum
    print_curriculum()

    # Select patterns based on year
    if args.year == 0:
        patterns = ALL_PATTERNS
        year_list = [1, 2]
    else:
        year_list = [args.year]
        patterns = [p for p in ALL_PATTERNS if p.year == args.year]

    pattern_names = [p.name for p in patterns]
    pattern_to_idx = {name: i for i, name in enumerate(pattern_names)}
    n_topics = len(pattern_names)

    print(f"\nTraining on {n_topics} patterns: {pattern_names}")

    # Create datasets
    train_data = DevelopmentalDataset(
        n_examples=args.train_size,
        seed=42,
        year=year_list if args.year == 0 else args.year
    )
    val_data = DevelopmentalDataset(
        n_examples=args.val_size,
        seed=123,
        year=year_list if args.year == 0 else args.year
    )

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=collate_fn)

    # Create classroom
    broker = ClassroomBroker(
        student_names=['nova', 'rêve', 'alex'],
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
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

    # Training loop
    history = []
    best_acc = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{args.epochs}")
        print("=" * 70)

        # Tutoring pairs
        tutoring_pairs = identify_tutoring_pairs(broker, pattern_names, pattern_to_idx)
        active = sum(len(p) for p in tutoring_pairs.values())
        if active > 0:
            print(f"\n  Peer tutoring: {active} pairs")

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
            for pt in pattern_names[:8]:  # First 8 patterns
                if pt in val_results['nova']['per_pattern']:
                    acc = val_results['nova']['per_pattern'][pt]
                    pt_idx = pattern_to_idx[pt]
                    if pt_idx < nova.exam_system.n_topics:
                        confirmed = nova.exam_system.confirmed_level[pt_idx].item()
                        xp_level = nova.topic_tracker.get_level(pt_idx)
                        xp = nova.topic_tracker.progression.topic_xp[pt_idx].item()
                        bar = '█' * int(acc * 10) + '·' * (10 - int(acc * 10))
                        print(f"    {pt:20s}: {acc:3.0%} {bar} L{confirmed}(+{xp_level-confirmed}) ({xp:.0f}xp)")

        # Exams
        exam_results = run_exams(broker, pattern_names, pattern_to_idx, device, epoch)
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
                'class_accuracy': class_acc
            }, data_dir / f'developmental_{session_id}_best.pt')

        # Graduation check
        all_graduated = True
        for student in broker.students.values():
            for pt_idx in range(min(n_topics, student.exam_system.n_topics)):
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

        history.append({'epoch': epoch, 'class_acc': class_acc})

    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Session: {session_id}")
    print(f"Epochs: {epoch}")
    print(f"Best accuracy: {best_acc:.1%}")

    print("\nFinal standings:")
    for name, student in broker.students.items():
        acc = val_results[name]['accuracy']
        grads = student.exam_system.topic_graduated[:n_topics].sum().item()
        print(f"  {student.name}: {acc:.1%} | XP: {student.xp:.0f} | Graduated: {int(grads)}/{n_topics}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=1, help='Year to train (1, 2, or 0 for both)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--train_size', type=int, default=20000)
    parser.add_argument('--val_size', type=int, default=2000)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--data_dir', type=str, default='data')

    args = parser.parse_args()
    main(args)
