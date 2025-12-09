"""
Classroom Training - Multi-student parallel curriculum learning
Coherence Lab - Emergence Project

Runs Nova, Rêve, and Alex through the curriculum simultaneously.
Phase 1: Independent parallel learning (no interaction yet)

Usage:
    python run_classroom.py                # Fresh start
    python run_classroom.py --epochs 100   # Custom epoch limit
"""

__version__ = "0.1.0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
from pathlib import Path
import argparse
from datetime import datetime

from relational_model import PatternDataset, collate_fn
from classroom import ClassroomBroker, STUDENT_NAMES


# === THE CURRICULUM ===
# Using basic patterns that PatternDataset supports
# Full curriculum will use the extended pattern generators
CURRICULUM = [
    {
        'name': 'A: Basic Patterns',
        'patterns': ['alternating', 'repeating'],
        'description': 'Memory and cycle patterns'
    },
    {
        'name': 'B: Arithmetic Patterns',
        'patterns': ['incrementing', 'fixed_offset'],
        'description': 'Linear and offset progressions'
    }
]


def get_all_patterns():
    """All patterns from curriculum in order."""
    # Basic patterns that PatternDataset supports
    return ['alternating', 'repeating', 'incrementing', 'fixed_offset']


def run_student_exams(
    broker: ClassroomBroker,
    pattern_types: list,
    pattern_to_idx: dict,
    device: torch.device,
    epoch: int
) -> dict:
    """
    Run level-up exams for any eligible students/topics.

    Returns dict of exam results by student.
    """
    results = {}

    for name, student in broker.students.items():
        student_results = []

        # Tick cooldowns
        student.exam_system.tick_cooldowns()

        for pt_idx, pt in enumerate(pattern_types):
            # Keep taking exams while eligible (can level up multiple times per epoch)
            while student.exam_system.check_eligible(pt_idx):
                # Generate exam dataset for this pattern
                target_level = student.exam_system.confirmed_level[pt_idx].item() + 1
                exam_size = student.exam_system.get_exam_size(target_level)

                # Create mini-dataset for exam (unique seed per level attempt)
                exam_data = PatternDataset(n_examples=exam_size, seed=epoch * 1000 + pt_idx * 100 + target_level)

                # Filter to just this pattern (exam_data has all patterns mixed)
                # For now, run on all patterns but track this one specifically
                correct_count = 0
                total_count = 0

                student.eval()
                with torch.no_grad():
                    for ex in exam_data:
                        if ex['pattern_type'] == pt:
                            tokens = ex['sequence'].unsqueeze(0).to(device)
                            target = ex['target']
                            seq_len = [ex['seq_len']]

                            logits, _, _ = student(tokens, seq_len)
                            pred = logits.argmax(dim=-1).item()

                            total_count += 1
                            if pred == target:
                                correct_count += 1

                            if total_count >= exam_size:
                                break

                # If we didn't get enough samples, generate more targeted
                if total_count < exam_size // 2:
                    # Fallback: use what we have
                    pass

                if total_count > 0:
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


def append_to_log(log_file, record):
    """Append a single record to log file (crash-safe)."""
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'a') as f:
            f.write(json.dumps(record) + '\n')
            f.flush()
    except Exception as e:
        print(f"  [Warning: Failed to write log: {e}]")


def train_epoch_parallel(
    broker: ClassroomBroker,
    train_loader: DataLoader,
    optimizers: dict,
    criterion: nn.Module,
    device: torch.device,
    pattern_to_idx: dict
) -> dict:
    """
    One epoch of parallel training for all students.

    Each student processes the same batches independently.
    Returns per-student metrics.
    """
    broker.train()
    broker.wake_all()

    # Per-student tracking
    results = {name: {
        'loss': 0.0, 'correct': 0, 'total': 0,
        'show_count': 0, 'approval_count': 0,
        'xp_from_shows': 0.0
    } for name in broker.students.keys()}

    # Approval-seeking parameters
    SHOW_CONF_THRESHOLD = 0.7  # Only show when confident
    APPROVAL_XP_BONUS = 3.0    # XP bonus for validated work

    for batch in train_loader:
        tokens = batch['tokens'].to(device)
        targets = batch['target'].to(device)
        seq_lens = batch['seq_len']
        pattern_types = batch['pattern_type']

        # Each student processes the batch
        for name, student in broker.students.items():
            optimizer = optimizers[name]
            optimizer.zero_grad()

            # Forward
            output = student(tokens, seq_lens, return_details=True)
            logits = output['logits']

            # Main loss
            main_loss = criterion(logits, targets)

            # Confidence calibration
            preds = logits.argmax(dim=-1)
            correct = (preds == targets)
            conf = output['self_state']['emotions']['confidence'].squeeze()

            # Handle dimension mismatch
            if conf.dim() == 0:
                conf = conf.unsqueeze(0)
            if correct.dim() == 0:
                correct = correct.unsqueeze(0)

            # Ensure same size
            min_len = min(len(conf), len(correct))
            conf = conf[:min_len]
            correct_float = correct[:min_len].float()

            conf_loss = F.binary_cross_entropy(conf, correct_float)

            # Combined loss
            loss = main_loss + 0.1 * conf_loss
            loss.backward()
            optimizer.step()

            # Track
            results[name]['loss'] += loss.item() * len(targets)
            results[name]['correct'] += correct.sum().item()
            results[name]['total'] += len(targets)

            # Update topic tracker with accuracy and confidence
            pattern_indices = torch.tensor(
                [pattern_to_idx[p] for p in pattern_types], device=device
            )
            student.topic_tracker.update(pattern_indices, correct, conf)

            # Award XP for correct answers
            for i, pt in enumerate(pattern_types):
                if correct[i]:
                    pt_idx = pattern_to_idx[pt]
                    student.topic_tracker.award_xp(pt_idx, 1.0)

            # === APPROVAL-SEEKING BEHAVIOR ===
            # Student decides to "show work" when confident
            # Teacher validates and awards bonus XP
            for i in range(len(targets)):
                conf_val = conf[i].item() if i < len(conf) else 0.5

                # Show work when confident
                if conf_val >= SHOW_CONF_THRESHOLD:
                    results[name]['show_count'] += 1
                    pt_idx = pattern_to_idx[pattern_types[i]]

                    # Teacher approves if correct
                    if correct[i]:
                        results[name]['approval_count'] += 1
                        # Bonus XP for validated confidence
                        student.topic_tracker.award_xp(pt_idx, APPROVAL_XP_BONUS)
                        results[name]['xp_from_shows'] += APPROVAL_XP_BONUS

    broker.sleep_all()

    # Compute averages
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
            results[name]['show_rate'] = 0.0
            results[name]['approval_rate'] = 0.0

    return results


def evaluate_all(
    broker: ClassroomBroker,
    val_loader: DataLoader,
    device: torch.device,
    pattern_to_idx: dict
) -> dict:
    """Evaluate all students on validation set."""
    broker.eval()

    results = {name: {
        'correct': 0, 'total': 0,
        'per_pattern': {p: {'correct': 0, 'total': 0} for p in pattern_to_idx}
    } for name in broker.students.keys()}

    with torch.no_grad():
        for batch in val_loader:
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

                # Per-pattern tracking
                for i, pt in enumerate(pattern_types):
                    results[name]['per_pattern'][pt]['total'] += 1
                    if correct[i]:
                        results[name]['per_pattern'][pt]['correct'] += 1

    # Compute accuracies
    for name in results:
        total = results[name]['total']
        results[name]['accuracy'] = results[name]['correct'] / total if total > 0 else 0

        for pt in results[name]['per_pattern']:
            pt_total = results[name]['per_pattern'][pt]['total']
            pt_correct = results[name]['per_pattern'][pt]['correct']
            results[name]['per_pattern'][pt] = pt_correct / pt_total if pt_total > 0 else 0

    return results


def print_leaderboard(results: dict, broker: ClassroomBroker, epoch: int):
    """Print comparative leaderboard with XP/level info."""
    # Sort by accuracy
    ranked = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

    print(f"\n  Leaderboard (Epoch {epoch}):")
    for i, (name, data) in enumerate(ranked):
        acc = data['accuracy']
        rank_marker = ["1st", "2nd", "3rd"][i] if i < 3 else f"{i+1}th"

        # Get XP/level from student
        student = broker.students[name]
        xp = student.xp
        avg_level = student.current_level

        print(f"    {rank_marker}: {name:8s} {acc:.1%}  (XP: {xp:6.0f}, Avg Lvl: {avg_level:.1f})")


def main(args):
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = Path(args.data_dir)
    data_dir.mkdir(exist_ok=True)

    log_file = data_dir / f'classroom_{session_id}.log'

    print("=" * 70)
    print("CLASSROOM TRAINING - Parallel Students")
    print(f"Session: {session_id}")
    print(f"Device: {device}")
    print("=" * 70)

    # Curriculum setup
    pattern_types = get_all_patterns()
    pattern_to_idx = {p: i for i, p in enumerate(pattern_types)}
    n_patterns = len(pattern_types)

    print(f"\nCurriculum ({n_patterns} patterns, {len(CURRICULUM)} sections):")
    for section in CURRICULUM:
        print(f"  {section['name']}: {section['patterns']}")

    # Create classroom
    broker = ClassroomBroker(
        student_names=['Nova', 'Rêve', 'Alex'],
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_think_steps=args.n_think_steps,
        n_topics=n_patterns
    ).to(device)

    total_params = sum(p.numel() for p in broker.parameters())
    print(f"\nClassroom: {total_params:,} total parameters")
    for name, student in broker.students.items():
        student_params = sum(p.numel() for p in student.parameters())
        print(f"  {student.name}: {student_params:,} params")

    # Per-student optimizers
    optimizers = {
        name: torch.optim.Adam(student.parameters(), lr=args.lr)
        for name, student in broker.students.items()
    }

    criterion = nn.CrossEntropyLoss()

    # Data (PatternDataset generates all 4 basic patterns by default)
    train_data = PatternDataset(n_examples=args.n_train, seed=42)
    val_data = PatternDataset(n_examples=args.n_val, seed=123)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=collate_fn)

    print(f"\nData: {len(train_data)} train, {len(val_data)} val")
    print("=" * 70)

    # Training loop
    history = []
    best_class_acc = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{args.epochs}")
        print("=" * 70)

        # Train
        train_results = train_epoch_parallel(
            broker, train_loader, optimizers, criterion, device, pattern_to_idx
        )

        # Evaluate
        val_results = evaluate_all(broker, val_loader, device, pattern_to_idx)

        # Log
        epoch_record = {
            'session_id': session_id,
            'epoch': epoch,
            'train': {name: {'loss': r['loss'], 'accuracy': r['accuracy']}
                     for name, r in train_results.items()},
            'val': {name: {'accuracy': r['accuracy'], 'per_pattern': r['per_pattern']}
                   for name, r in val_results.items()},
            'timestamp': datetime.now().isoformat()
        }
        history.append(epoch_record)
        append_to_log(log_file, epoch_record)

        # Print progress
        print("\n  Training:")
        for name in broker.students.keys():
            tr = train_results[name]
            show_str = f"show={tr['show_rate']:.0%}" if tr['show_rate'] > 0 else ""
            appr_str = f"appr={tr['approval_rate']:.0%}" if tr['show_count'] > 0 else ""
            print(f"    {name:8s}: loss={tr['loss']:.4f}, acc={tr['accuracy']:.1%} {show_str} {appr_str}")

        print("\n  Validation:")
        for name in broker.students.keys():
            vr = val_results[name]
            print(f"    {name:8s}: acc={vr['accuracy']:.1%}")

        # Leaderboard
        print_leaderboard(val_results, broker, epoch)

        # Per-pattern detail with XP/level (first student only, others similar)
        if epoch % 5 == 0 or epoch <= 3:
            nova = broker.students['nova']
            print("\n  Per-pattern (Nova):")
            for section in CURRICULUM:
                for pt in section['patterns']:
                    pt_idx = pattern_to_idx[pt]
                    acc = val_results['nova']['per_pattern'].get(pt, 0)
                    bar = "█" * int(acc * 10) + "·" * (10 - int(acc * 10))
                    level = nova.topic_tracker.get_level(pt_idx)
                    confirmed = nova.exam_system.confirmed_level[pt_idx].item()
                    xp, _, progress, _ = nova.topic_tracker.get_xp_info(pt_idx)
                    print(f"    {pt:18s}: {acc:.0%} {bar} L{confirmed}(+{level-confirmed}) ({xp:.0f}xp)")

        # Run level-up exams for eligible students
        exam_results = run_student_exams(broker, pattern_types, pattern_to_idx, device, epoch)
        total_exams = sum(len(r) for r in exam_results.values())
        if total_exams > 0:
            print(f"\n  Exams this epoch: {total_exams}")

        # Class average
        class_acc = sum(v['accuracy'] for v in val_results.values()) / len(val_results)
        print(f"\n  Class average: {class_acc:.1%}")

        if class_acc > best_class_acc:
            best_class_acc = class_acc
            # Save checkpoint
            torch.save({
                'session_id': session_id,
                'epoch': epoch,
                'broker_state': broker.state_dict(),
                'class_accuracy': class_acc,
                'per_student': {name: val_results[name]['accuracy'] for name in broker.students},
            }, data_dir / f'classroom_{session_id}_best.pt')

        # Early stopping check
        if class_acc >= 0.90:
            print(f"\n{'*'*60}")
            print(f"*** CLASS GRADUATED! Average accuracy: {class_acc:.1%} ***")
            print(f"{'*'*60}")
            break

        import sys; sys.stdout.flush()

    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Session: {session_id}")
    print(f"Epochs: {len(history)}")
    print(f"Best class accuracy: {best_class_acc:.1%}")
    print(f"\nFinal standings:")

    final_val = evaluate_all(broker, val_loader, device, pattern_to_idx)
    for name in broker.students:
        acc = final_val[name]['accuracy']
        student = broker.students[name]
        xp = student.xp
        avg_level = student.current_level
        # Count confirmed levels and graduations
        confirmed_levels = sum(student.exam_system.confirmed_level[i].item() for i in range(student._n_topics))
        graduations = student.exam_system.topic_graduated.sum().item()
        print(f"  {student.name}: {acc:.1%} | XP: {xp:.0f} | Confirmed Lvls: {confirmed_levels} | Grads: {int(graduations)}")

    print(f"\nLog: {log_file}")
    print("=" * 70)

    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classroom Training')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--n-train', type=int, default=20000)
    parser.add_argument('--n-val', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-think-steps', type=int, default=5)

    args = parser.parse_args()
    main(args)
