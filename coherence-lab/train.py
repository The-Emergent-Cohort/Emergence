"""
Coherence Lab - Unified Training Script
The ONE training script. Curriculum drives everything.

Usage:
    python train.py                    # Fresh start
    python train.py --resume           # Resume from checkpoint
    python train.py --epochs 500       # Custom epoch limit

Log appends every epoch (crash-safe).
Checkpoint saves after each section.
"""

__version__ = "1.0.0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
from pathlib import Path
import argparse
from datetime import datetime

from relational_model import (
    RelationalSystem, PatternDataset, collate_fn, evaluate, DynamicTopicRegistry
)
from curriculum_sequencer import CurriculumSequencer, create_mixed_dataset


# === THE CURRICULUM ===
# This drives everything. No phases, just sections to master in order.
CURRICULUM = [
    {
        'name': 'A: Position Foundations',
        'patterns': ['counting', 'incrementing'],
        'description': 'Pure position awareness and linear progression'
    },
    {
        'name': 'B: Reverse & Cycles',
        'patterns': ['decrementing', 'modular'],
        'description': 'Countdown and cycle position (i % n)'
    },
    {
        'name': 'C: Position Math',
        'patterns': ['staircase', 'fixed_offset'],
        'description': 'Quantization (i // n) and linear growth'
    },
    {
        'name': 'D: Simple Memory',
        'patterns': ['repeating', 'alternating'],
        'description': 'Remember and cycle 1-2 values'
    },
    {
        'name': 'E: Extended Cycles',
        'patterns': ['periodic_repeat'],
        'description': 'Cycle 3-4 values (builds on alternating)'
    },
    {
        'name': 'F: Indexed Retrieval',
        'patterns': ['indexed_lookup'],
        'description': 'Position-based value lookup from memory'
    },
    {
        'name': 'G: Growth Patterns',
        'patterns': ['geometric', 'triangular'],
        'description': 'Exponential and accumulative growth'
    },
    {
        'name': 'H: Combined Operations',
        'patterns': ['fibonacci_like'],
        'description': 'Combine two lookbacks (sum of previous two)'
    }
]

# Bridges for diagnosis when stuck
BRIDGES = {
    'decrementing': ['incrementing'],
    'modular': ['counting'],
    'staircase': ['counting'],
    'fixed_offset': ['incrementing'],
    'alternating': ['modular'],
    'periodic_repeat': ['alternating'],
    'indexed_lookup': ['modular', 'repeating'],
    'geometric': ['incrementing', 'staircase'],
    'triangular': ['incrementing', 'staircase'],
    'fibonacci_like': ['incrementing', 'alternating'],
}


def get_all_patterns():
    """All patterns from curriculum in order."""
    patterns = []
    for section in CURRICULUM:
        patterns.extend(section['patterns'])
    return patterns


def append_to_log(log_file, record):
    """Append a single record to log file (crash-safe)."""
    try:
        # Ensure parent directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'a') as f:
            f.write(json.dumps(record) + '\n')
            f.flush()  # Force write to disk
    except Exception as e:
        print(f"  [Warning: Failed to write log: {e}]")


def load_log(log_file):
    """Load all records from log file."""
    records = []
    if log_file.exists():
        with open(log_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def train_epoch(model, loader, optimizer, criterion, device, pattern_to_idx):
    """One epoch of training with approval-seeking behavior."""
    model.train()
    model.learner.temporal_model.wake()

    total_loss, total_correct, total_samples = 0, 0, 0
    show_count, approval_count = 0, 0

    for batch in loader:
        tokens = batch['tokens'].to(device)
        targets = batch['target'].to(device)
        seq_lens = batch['seq_len']
        pattern_types = batch['pattern_type']

        optimizer.zero_grad()

        details = model(tokens, seq_lens, targets=targets, return_details=True)

        # Main loss
        main_loss = criterion(details['logits'], targets)

        # Pattern classification auxiliary
        pattern_targets = torch.tensor([pattern_to_idx[p] for p in pattern_types], device=device)
        aux_loss = criterion(details['pattern_logits'], pattern_targets)

        # Confidence calibration
        preds = details['logits'].argmax(dim=-1)
        correct = (preds == targets)
        conf = details['learner_self']['emotions']['confidence'].squeeze()
        conf_loss = F.binary_cross_entropy(conf, correct.float())

        # Combined loss
        loss = main_loss + 0.1 * aux_loss + 0.1 * conf_loss
        loss.backward()
        optimizer.step()

        # Approval-seeking behavior (pass pattern_indices for per-topic streak tracking)
        int_level = torch.sigmoid(model.learner.other_model.internalization_level).item()
        is_creative = torch.zeros(tokens.size(0), dtype=torch.bool, device=device)
        pattern_indices = torch.tensor([pattern_to_idx[p] for p in pattern_types], device=device)

        should_show, reasons = model.learner.self_model.should_show_work(
            correct, is_creative, conf, int_level, pattern_indices=pattern_indices
        )

        if should_show.any():
            show_indices = should_show.nonzero(as_tuple=True)[0]
            for idx in show_indices:
                idx_item = idx.item()
                is_correct = correct[idx_item].item()

                # Teacher responds to shown work
                model.learner.other_model.update_trust(is_correct)

                if is_correct:
                    approval_count += 1
                    # Award XP for correct shown work
                    pt = pattern_types[idx_item]
                    pt_idx = pattern_to_idx[pt]
                    xp_gain = 10 + int(5 * (1 - int_level))
                    model.learner.self_model.topic_tracker.award_xp(pt_idx, xp_gain)

            show_count += len(show_indices)

        total_loss += loss.item() * len(targets)
        total_correct += correct.sum().item()
        total_samples += len(targets)

    model.learner.temporal_model.sleep()

    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples,
        'show_rate': show_count / total_samples if total_samples > 0 else 0,
        'approval_rate': approval_count / show_count if show_count > 0 else 0,
    }


def run_exam(model, topic, n_problems, seed, device, pattern_to_idx):
    """Run exam for single topic."""
    exam_data = PatternDataset(n_examples=n_problems, seed=seed, pattern_types=[topic])
    exam_loader = DataLoader(exam_data, batch_size=n_problems, collate_fn=collate_fn)

    model.eval()
    correct = 0
    for batch in exam_loader:
        tokens = batch['tokens'].to(device)
        targets = batch['target'].to(device)
        seq_lens = batch['seq_len']
        with torch.no_grad():
            details = model(tokens, seq_lens, targets=targets, return_details=True)
            preds = details['logits'].argmax(dim=-1)
            correct += (preds == targets).sum().item()
    model.train()

    return correct, n_problems


def save_checkpoint(model, epoch, section_idx, section_passed, data_dir, session_id, proposals=None, suffix=''):
    """Save model checkpoint with session ID and proposals."""
    checkpoint = {
        'session_id': session_id,
        'state_dict': model.state_dict(),
        'epoch': epoch,
        'section_idx': section_idx,
        'section_passed': section_passed,
        'proposals': proposals or {},  # Teacher proposals, stuck patterns, etc.
        'timestamp': datetime.now().isoformat(),
    }
    path = Path(data_dir) / f'{session_id}_checkpoint{suffix}.pt'
    torch.save(checkpoint, path)
    return path


def main(args):
    # Session ID for this run
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = Path(args.data_dir)
    data_dir.mkdir(exist_ok=True)

    # Log file - session specific, append mode
    log_file = data_dir / f'{session_id}_training.log'

    print("=" * 70)
    print("COHERENCE LAB - Unified Training")
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

    # Model
    model = RelationalSystem(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_think_steps=args.n_think_steps,
        n_topics=n_patterns,
        n_patterns=n_patterns
    ).to(device)

    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Topic registry
    topic_registry = DynamicTopicRegistry(pattern_types)
    model.set_topic_registry(topic_registry)

    # Resume from checkpoint if requested
    start_section = 0
    section_passed = [False] * len(CURRICULUM)
    prior_proposals = {}

    if args.resume:
        # Find checkpoint - either specific file or most recent
        if args.resume_from:
            checkpoint_path = Path(args.resume_from)
        else:
            # Find most recent checkpoint
            checkpoints = list(data_dir.glob('*_checkpoint.pt'))
            checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime) if checkpoints else None

        if checkpoint_path and checkpoint_path.exists():
            print(f"\nResuming from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['state_dict'])
            start_section = checkpoint.get('section_idx', 0)
            section_passed = checkpoint.get('section_passed', section_passed)
            prior_proposals = checkpoint.get('proposals', {})
            prior_session = checkpoint.get('session_id', 'unknown')
            print(f"  Prior session: {prior_session}")
            print(f"  Starting at section {start_section}, passed: {section_passed}")
            if prior_proposals:
                print(f"  Prior proposals: {list(prior_proposals.keys())}")
        else:
            print(f"\nNo checkpoint found, starting fresh")

    # Track proposals during this session
    proposals = {
        'stuck_patterns': {},  # pattern -> count
        'teacher_hints': [],   # list of hints given
        'curriculum_issues': [],  # any detected ordering issues
    }

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    tracker = model.learner.self_model.topic_tracker

    # Validation data (all patterns)
    val_data = PatternDataset(n_examples=args.n_val, seed=123, pattern_types=pattern_types)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=collate_fn)

    # Sequencer
    sequencer = CurriculumSequencer(
        sections=CURRICULUM,
        topic_to_idx=pattern_to_idx,
        tracker=tracker,
        section_exam_level=10,
        section_exam_size=24,
        section_exam_threshold=0.90,
        final_exam_size=32,
        final_exam_threshold=0.90
    )
    sequencer.current_section_idx = start_section
    sequencer.section_passed = section_passed

    # === TRAINING FUNCTIONS ===
    def train_fn(mdl, active_topics, maintenance_topics, epoch):
        all_topics = active_topics + maintenance_topics
        n_active = args.n_train // len(all_topics) * len(active_topics) if all_topics else args.n_train
        n_maint = args.n_train - n_active if maintenance_topics else 0

        train_data = create_mixed_dataset(
            PatternDataset, active_topics, maintenance_topics,
            n_active, n_maint, seed=epoch * 100
        )
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

        return train_epoch(mdl, train_loader, optimizer, criterion, device, pattern_to_idx)

    def eval_fn(mdl):
        return evaluate(mdl, val_loader, device, pattern_to_idx)

    def exam_fn(mdl, topic, n_problems, seed, device):
        return run_exam(mdl, topic, n_problems, seed, device, pattern_to_idx)

    # === CALLBACKS ===
    last_section = [start_section]  # Track for checkpoint triggers

    def on_epoch_start(epoch, info):
        is_play = info.get('is_play_day', False)
        play_str = " [PLAY DAY]" if is_play else ""
        print(f"\n{'='*70}")
        print(f"Day {epoch} - Section {info['section_idx']+1}/{len(CURRICULUM)}: {info['section_name']}{play_str}")
        print(f"  Focus: {info['active_topics']}")
        if info['maintenance_topics']:
            print(f"  Maintenance: {info['maintenance_topics']}")

    def on_epoch_end(epoch, record):
        # Append to log (crash-safe, session-tagged)
        # Capture per-pattern details for analysis
        pattern_details = {}
        eval_m = record.get('eval_metrics', {})
        for pt in pattern_types:
            pt_idx = pattern_to_idx[pt]
            # Bounds check - tracker may have fewer topics than curriculum
            if pt_idx < tracker.n_topics:
                pattern_details[pt] = {
                    'accuracy': eval_m.get('per_pattern', {}).get(pt, 0),
                    'confirmed_level': tracker.confirmed_level[pt_idx].item() if hasattr(tracker, 'confirmed_level') and pt_idx < len(tracker.confirmed_level) else 0,
                    'xp': tracker.topic_xp[pt_idx].item(),
                    'best_streak': tracker.topic_best_streak[pt_idx].item(),
                }
            else:
                pattern_details[pt] = {
                    'accuracy': eval_m.get('per_pattern', {}).get(pt, 0),
                    'confirmed_level': 0,
                    'xp': 0,
                    'best_streak': 0,
                }

        log_record = {
            'session_id': session_id,
            'epoch': epoch,
            'section': record['section'],
            'section_name': record['section_name'],
            'is_play_day': record.get('is_play_day', False),
            'train_acc': record.get('train_metrics', {}).get('accuracy'),
            'train_loss': record.get('train_metrics', {}).get('loss'),
            'eval_acc': eval_m.get('accuracy'),
            'pattern_details': pattern_details,
            'level_exams': record.get('level_exams', []),
            'section_exam': record.get('section_exam'),
            'stuck_topics': record.get('stuck_topics', []),
            'proposals': proposals.copy(),
            'timestamp': datetime.now().isoformat()
        }
        append_to_log(log_file, log_record)

        # Print progress
        if 'train_metrics' in record:
            train = record['train_metrics']
            eval_m = record['eval_metrics']
            int_level = torch.sigmoid(model.learner.other_model.internalization_level).item()
            trust = torch.sigmoid(model.learner.other_model.trust).item()

            print(f"\n  Train: loss={train['loss']:.4f}, acc={train['accuracy']:.1%}")
            print(f"  Val: acc={eval_m['accuracy']:.1%}")
            print(f"  Internalization: {int_level:.1%}, Trust: {trust:.1%}")
            print(f"  XP: total={tracker.get_total_xp():.0f}, avg_level={tracker.get_average_level():.1f}")

            # Section patterns
            active, _ = sequencer.get_active_topics()
            print(f"\n  Section patterns:")
            for pt in active:
                acc = eval_m['per_pattern'].get(pt, 0)
                lvl = tracker.confirmed_level[pattern_to_idx[pt]].item() if hasattr(tracker, 'confirmed_level') else tracker.get_level(pattern_to_idx[pt])
                xp, _, progress, _ = tracker.get_xp_info(pattern_to_idx[pt])
                bar = "█" * int(lvl) + "·" * (10 - int(lvl))
                print(f"    {pt:15s}: {acc:.1%} L{int(lvl):2d} {bar} ({xp:.0f}xp)")

        # Level exams
        if record.get('level_exams'):
            print("  Level exams:")
            for r in record['level_exams']:
                sym = "PASS" if r['passed'] else "FAIL"
                extra = ""
                if not r['passed'] and r.get('consecutive_failures', 0) >= 5:
                    extra = f" [plateau:{r['consecutive_failures']}]"
                print(f"    {r['topic']:15s}: {r['score']:.0%} {sym}{extra}")

        import sys; sys.stdout.flush()

    def on_section_complete(section_idx, next_section):
        print(f"\n  *** SECTION {section_idx+1} PASSED! ***")

        # Checkpoint after section (with session ID and proposals)
        epoch_num = sequencer.history[-1]['epoch'] if sequencer.history else 0
        path = save_checkpoint(
            model, epoch_num,
            sequencer.current_section_idx, sequencer.section_passed.copy(),
            data_dir, session_id, proposals, f'_section{section_idx+1}'
        )
        print(f"  Checkpoint saved: {path}")

        # Also update main checkpoint
        save_checkpoint(
            model, epoch_num,
            sequencer.current_section_idx, sequencer.section_passed.copy(),
            data_dir, session_id, proposals
        )

        if next_section:
            print(f"  Next: {next_section['name']}")

    def on_section_exam(section, results, passed):
        print(f"\n  === SECTION EXAM: {section['name']} ===")
        for r in results:
            if 'score' in r:
                sym = "PASS" if r['passed'] else "FAIL"
                print(f"    {r['topic']:15s}: {r['score']:.0%} {sym}")
            else:
                print(f"    {r['topic']:15s}: {r.get('reason', 'N/A')}")

    def on_final_exam(results, passed):
        print(f"\n  {'='*50}")
        print(f"  === FINAL COMPREHENSIVE EXAM ===")
        for r in results:
            sym = "PASS" if r['passed'] else "FAIL"
            print(f"    {r['topic']:15s}: {r['correct']}/{r['total']} ({r['score']:.0%}) {sym}")

        if passed:
            print(f"\n{'*'*60}")
            print(f"*** CURRICULUM COMPLETE! ***")
            print(f"{'*'*60}")

    def on_stuck_topic(stuck_info, topic_to_idx_map, trk):
        topic = stuck_info['topic']
        reason = stuck_info['reason']
        print(f"\n  [Stuck: {topic} - {reason}]")

        # Track in proposals
        proposals['stuck_patterns'][topic] = proposals['stuck_patterns'].get(topic, 0) + 1

        bridges = BRIDGES.get(topic, [])
        if bridges:
            print(f"    Hint: Practice {bridges}")
            proposals['teacher_hints'].append({
                'epoch': sequencer.history[-1]['epoch'] if sequencer.history else 0,
                'topic': topic,
                'hint': f'Practice {bridges}',
                'reason': reason
            })

    # === RUN ===
    result = sequencer.run(
        model=model,
        train_fn=train_fn,
        eval_fn=eval_fn,
        exam_fn=exam_fn,
        device=device,
        max_epochs=args.epochs,
        callbacks={
            'on_epoch_start': on_epoch_start,
            'on_epoch_end': on_epoch_end,
            'on_section_exam': on_section_exam,
            'on_section_complete': on_section_complete,
            'on_final_exam': on_final_exam,
            'on_stuck_topic': on_stuck_topic,
        }
    )

    # Final checkpoint
    save_checkpoint(
        model, result['epochs'],
        sequencer.current_section_idx, sequencer.section_passed,
        data_dir, session_id, proposals
    )

    print("\n" + "=" * 70)
    print(f"Session: {session_id}")
    print(f"Completed: {result['completed']} in {result['epochs']} epochs")
    print(f"Sections passed: {sum(sequencer.section_passed)}/{len(CURRICULUM)}")
    if proposals['stuck_patterns']:
        print(f"Stuck patterns: {proposals['stuck_patterns']}")
    print(f"Checkpoints: {session_id}_checkpoint*.pt")
    print(f"Log: {log_file}")
    print("=" * 70)

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Coherence Lab Training')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--n-train', type=int, default=20000)
    parser.add_argument('--n-val', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=len(get_all_patterns()) * 100)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-think-steps', type=int, default=5)
    parser.add_argument('--resume', action='store_true', help='Resume from most recent checkpoint')
    parser.add_argument('--resume-from', type=str, help='Resume from specific checkpoint file')

    args = parser.parse_args()
    # --resume-from implies --resume
    if args.resume_from:
        args.resume = True
    main(args)
