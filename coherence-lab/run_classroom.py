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
                # Generate exam dataset for this specific pattern only
                target_level = student.exam_system.confirmed_level[pt_idx].item() + 1
                exam_size = student.exam_system.get_exam_size(target_level)

                # Create pattern-specific dataset for exam (unique seed per level attempt)
                exam_data = PatternDataset(
                    n_examples=exam_size,
                    seed=epoch * 1000 + pt_idx * 100 + target_level,
                    pattern_types=[pt]  # Only generate this pattern type
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

                # Take the exam (always have exam_size examples now)
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


def identify_tutoring_pairs(
    broker,
    pattern_types: list,
    pattern_to_idx: dict,
    level_gap_to_tutor: int = 3  # Offer tutoring if 3+ levels ahead
) -> dict:
    """
    Identify tutor-student pairs for peer teaching.

    Tutoring rules:
    - Graduates (L10) always offer tutoring on their topic
    - Anyone 3+ levels ahead offers tutoring
    - Any non-graduate can ask for help from higher-level peers
    - Prioritize graduates > highest level gap

    Returns:
        Dict[pattern_idx, Dict[learner_name, tutor_name]]
    """
    tutoring = {}

    for pt_idx, pt in enumerate(pattern_types):
        tutoring[pt_idx] = {}

        # Get each student's confirmed level for this pattern
        student_levels = {}
        graduates = []

        for name, student in broker.students.items():
            confirmed = student.exam_system.confirmed_level[pt_idx].item()
            graduated = student.exam_system.topic_graduated[pt_idx].item()
            student_levels[name] = confirmed
            if graduated:
                graduates.append(name)

        # For each non-graduate, find a tutor
        for learner_name, learner_level in student_levels.items():
            if learner_name in graduates:
                continue  # Graduates don't need tutoring on this topic

            # Find best tutor: prefer graduates, then highest level gap
            best_tutor = None
            best_gap = 0

            for tutor_name, tutor_level in student_levels.items():
                if tutor_name == learner_name:
                    continue

                gap = tutor_level - learner_level

                # Graduates always available, or 3+ level gap
                is_graduate = tutor_name in graduates
                can_tutor = is_graduate or gap >= level_gap_to_tutor

                if can_tutor:
                    # Prefer graduates, then largest gap
                    tutor_priority = (1 if is_graduate else 0, gap)
                    if best_tutor is None or tutor_priority > (1 if best_tutor in graduates else 0, best_gap):
                        best_tutor = tutor_name
                        best_gap = gap

            if best_tutor:
                tutoring[pt_idx][learner_name] = best_tutor

    return tutoring


def get_tutor_predictions(
    tutor_student,
    tokens: torch.Tensor,
    seq_lens: list,
    temperature: float = 2.0
) -> torch.Tensor:
    """Get soft predictions from tutor for knowledge distillation."""
    tutor_student.eval()
    with torch.no_grad():
        logits, _, _ = tutor_student(tokens, seq_lens)
        # Soften with temperature for better knowledge transfer
        soft_targets = F.softmax(logits / temperature, dim=-1)
    tutor_student.train()
    return soft_targets


def train_epoch_parallel(
    broker: ClassroomBroker,
    train_loader: DataLoader,
    optimizers: dict,
    criterion: nn.Module,
    device: torch.device,
    pattern_to_idx: dict,
    tutoring_pairs: dict = None  # pattern_idx -> {student: tutor}
) -> dict:
    """
    One epoch of parallel training for all students.

    Each student processes the same batches independently.
    Includes peer teaching via knowledge distillation.
    Returns per-student metrics.
    """
    broker.train()
    broker.wake_all()

    tutoring_pairs = tutoring_pairs or {}

    # Per-student tracking
    results = {name: {
        'loss': 0.0, 'correct': 0, 'total': 0,
        'show_count': 0, 'approval_count': 0,
        'xp_from_shows': 0.0,
        'tutoring_received': 0,  # Times received help
        'tutoring_given': 0,     # Times gave help
        'xp_from_tutoring': 0.0  # XP earned from tutoring others
    } for name in broker.students.keys()}

    # Approval-seeking parameters
    SHOW_CONF_THRESHOLD = 0.7  # Only show when confident
    APPROVAL_XP_BONUS = 3.0    # XP bonus for validated work
    TUTORING_WEIGHT = 0.3      # Weight for knowledge distillation loss
    DISTILL_TEMP = 2.0         # Temperature for soft labels
    TUTOR_XP_BONUS = 0.5       # XP tutors earn for helping (teaching reinforces learning)

    for batch in train_loader:
        tokens = batch['tokens'].to(device)
        targets = batch['target'].to(device)
        seq_lens = batch['seq_len']
        pattern_types = batch['pattern_type']

        # Pre-compute tutor predictions for this batch (cache for efficiency)
        tutor_cache = {}  # tutor_name -> soft_predictions

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

            # === PEER TUTORING (Knowledge Distillation) ===
            tutoring_loss = torch.tensor(0.0, device=device)
            tutor_helped = False

            # Check if this student needs tutoring on any pattern in this batch
            for i, pt in enumerate(pattern_types):
                pt_idx = pattern_to_idx[pt]
                if pt_idx in tutoring_pairs and name in tutoring_pairs[pt_idx]:
                    tutor_name = tutoring_pairs[pt_idx][name]
                    tutor = broker.students[tutor_name]

                    # Get tutor's soft predictions (cached)
                    if tutor_name not in tutor_cache:
                        tutor_cache[tutor_name] = get_tutor_predictions(
                            tutor, tokens, seq_lens, DISTILL_TEMP
                        )

                    # KL divergence from student to tutor distribution
                    student_log_probs = F.log_softmax(logits[i:i+1] / DISTILL_TEMP, dim=-1)
                    tutor_probs = tutor_cache[tutor_name][i:i+1]

                    # KL(tutor || student) - student learns from tutor
                    kl = F.kl_div(student_log_probs, tutor_probs, reduction='batchmean')
                    tutoring_loss = tutoring_loss + kl
                    tutor_helped = True

                    # Track tutoring
                    results[name]['tutoring_received'] += 1
                    results[tutor_name]['tutoring_given'] += 1

                    # Non-graduate tutors earn XP for helping (teaching reinforces learning)
                    # Graduates already at L10 - their reward is helping class move on
                    if not tutor.exam_system.topic_graduated[pt_idx]:
                        tutor.topic_tracker.award_xp(pt_idx, TUTOR_XP_BONUS)
                        results[tutor_name]['xp_from_tutoring'] += TUTOR_XP_BONUS

            # Combined loss
            loss = main_loss + 0.1 * conf_loss
            if tutor_helped:
                loss = loss + TUTORING_WEIGHT * tutoring_loss

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

        # Identify peer tutoring pairs (graduates help struggling students)
        tutoring_pairs = identify_tutoring_pairs(broker, pattern_types, pattern_to_idx)

        # Count active tutoring relationships
        active_tutoring = sum(len(pairs) for pairs in tutoring_pairs.values())
        if active_tutoring > 0:
            print(f"\n  Peer tutoring active: {active_tutoring} pairs")
            for pt_idx, pairs in tutoring_pairs.items():
                for student, tutor in pairs.items():
                    pt = pattern_types[pt_idx]
                    print(f"    {tutor} → {student} on {pt}")

        # Train
        train_results = train_epoch_parallel(
            broker, train_loader, optimizers, criterion, device, pattern_to_idx,
            tutoring_pairs=tutoring_pairs
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
            'tutoring': {name: {'received': r['tutoring_received'], 'given': r['tutoring_given']}
                        for name, r in train_results.items()},
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
            tutor_str = ""
            if tr['tutoring_received'] > 0:
                tutor_str = f"📚got help {tr['tutoring_received']}x"
            if tr['tutoring_given'] > 0:
                tutor_str += f" 🎓tutored {tr['tutoring_given']}x"
            print(f"    {name:8s}: loss={tr['loss']:.4f}, acc={tr['accuracy']:.1%} {show_str} {appr_str} {tutor_str}")

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

        # Early stopping check - ALL students must have ALL patterns at L10 confirmed
        all_graduated = True
        for student in broker.students.values():
            n_patterns = len(pattern_types)
            student_grads = student.exam_system.topic_graduated[:n_patterns].sum().item()
            if student_grads < n_patterns:
                all_graduated = False
                break

        if all_graduated:
            print(f"\n{'*'*60}")
            print(f"*** CLASS GRADUATED! All students mastered all patterns! ***")
            print(f"*** Final accuracy: {class_acc:.1%} ***")
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
