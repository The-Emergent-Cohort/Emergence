"""
Curriculum Training: Phase-aware developmental learning

Trains the relational model through curriculum phases:
- Phase 1: Foundation (4-6 tokens) - basic patterns
- Phase 2A: Position patterns (6-8 tokens)
- Phase 2B: Arithmetic patterns (8-10 tokens)
- Phase 3A: Working memory (10-12 tokens)
- Phase 3B: Recursive (8-10 tokens)
- Phase 4: Advanced (12-16 tokens)

Each phase must reach success_threshold before advancing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
from pathlib import Path
import argparse
import json
from datetime import datetime

from relational_model import (
    RelationalSystem, CurriculumDataset, CURRICULUM_CONFIG, collate_fn
)


def evaluate(model, loader, criterion, device, pattern_to_idx):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    per_pattern = {pt: {'correct': 0, 'total': 0} for pt in pattern_to_idx.keys()}

    with torch.no_grad():
        for batch in loader:
            tokens = batch['tokens'].to(device)
            targets = batch['target'].to(device)
            seq_lens = batch['seq_len']
            pattern_types = batch['pattern_type']

            details = model(tokens, seq_lens, targets=targets, return_details=True)
            logits = details['logits']

            loss = criterion(logits, targets)
            preds = logits.argmax(dim=-1)
            correct = (preds == targets)

            total_loss += loss.item() * len(targets)
            total_correct += correct.sum().item()
            total_samples += len(targets)

            for i, pt in enumerate(pattern_types):
                if pt in per_pattern:
                    per_pattern[pt]['total'] += 1
                    per_pattern[pt]['correct'] += correct[i].item()

    accuracy = total_correct / total_samples if total_samples > 0 else 0
    per_pattern_acc = {
        pt: (d['correct'] / d['total'] if d['total'] > 0 else 0)
        for pt, d in per_pattern.items()
    }

    return {
        'loss': total_loss / total_samples,
        'accuracy': accuracy,
        'per_pattern': per_pattern_acc
    }


def train_epoch(model, loader, optimizer, criterion, device, pattern_to_idx):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        tokens = batch['tokens'].to(device)
        targets = batch['target'].to(device)
        seq_lens = batch['seq_len']
        pattern_types = batch['pattern_type']

        optimizer.zero_grad()

        details = model(tokens, seq_lens, targets=targets, return_details=True)
        logits = details['logits']

        # Main loss
        main_loss = criterion(logits, targets)

        # Pattern classification auxiliary (if available)
        if 'pattern_logits' in details:
            pattern_targets = torch.tensor(
                [pattern_to_idx.get(p, 0) for p in pattern_types],
                device=device
            )
            n_pattern_classes = details['pattern_logits'].shape[-1]
            valid_mask = pattern_targets < n_pattern_classes
            if valid_mask.any():
                aux_loss = criterion(
                    details['pattern_logits'][valid_mask],
                    pattern_targets[valid_mask]
                )
                loss = main_loss + 0.1 * aux_loss
            else:
                loss = main_loss
        else:
            loss = main_loss

        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=-1)
        correct = (preds == targets)

        total_loss += main_loss.item() * len(targets)
        total_correct += correct.sum().item()
        total_samples += len(targets)

    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples
    }


def train_phase(model, phase, device, args, checkpoint_dir):
    """Train a single curriculum phase until mastery."""
    config = CURRICULUM_CONFIG[phase]
    pattern_types = config['patterns']
    pattern_to_idx = {p: i for i, p in enumerate(pattern_types)}
    success_threshold = config['success_threshold']

    print(f"\n{'='*70}")
    print(f"PHASE {phase}: {config['description']}")
    print(f"Patterns: {pattern_types}")
    print(f"Sequence length: {config['seq_len']}")
    print(f"Success threshold: {success_threshold:.0%}")
    print('='*70)

    # Generate data for this phase
    train_data = CurriculumDataset(phase=phase, n_examples=args.n_train, seed=42)
    val_data = CurriculumDataset(phase=phase, n_examples=args.n_val, seed=123)

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, collate_fn=collate_fn
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    history = []
    best_acc = 0

    for epoch in range(1, args.max_epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, pattern_to_idx
        )

        # Sleep cycle
        model.learner.temporal_model.sleep()

        val_metrics = evaluate(model, val_loader, criterion, device, pattern_to_idx)

        history.append({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['accuracy'],
            'val_acc': val_metrics['accuracy'],
            'per_pattern': val_metrics['per_pattern']
        })

        # Print progress
        print(f"\nEpoch {epoch:2d}", flush=True)
        print(f"  Train: loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.1%}")
        print(f"  Val: acc={val_metrics['accuracy']:.1%}")
        print(f"  Per-pattern:")
        for pt in pattern_types:
            acc = val_metrics['per_pattern'].get(pt, 0)
            status = "O" if acc >= success_threshold else ("o" if acc >= 0.7 else ".")
            print(f"    {pt:20s}: {acc:.1%} {status}")
        import sys; sys.stdout.flush()

        # Save best
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            torch.save({
                'state_dict': model.state_dict(),
                'phase': phase,
                'epoch': epoch,
                'val_acc': best_acc
            }, checkpoint_dir / f'curriculum_phase{phase}_best.pt')

        # Check mastery
        all_mastered = all(
            val_metrics['per_pattern'].get(pt, 0) >= success_threshold
            for pt in pattern_types
        )

        if all_mastered and val_metrics['accuracy'] >= success_threshold:
            print(f"\n*** Phase {phase} MASTERED! ***")
            print(f"    All patterns >= {success_threshold:.0%}")
            break

    return {
        'phase': phase,
        'epochs': len(history),
        'best_acc': best_acc,
        'history': history,
        'mastered': all_mastered if 'all_mastered' in dir() else False
    }


def main(args):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(args.data_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir = checkpoint_dir / 'runs'
    log_dir.mkdir(exist_ok=True)

    print("="*70)
    print("CURRICULUM TRAINING")
    print(f"Run ID: {run_id}")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Phases to train
    phases = args.phases.split(',') if args.phases else ['1']

    # Model - start fresh or load checkpoint
    # Use larger max_seq_len to accommodate later phases
    model = RelationalSystem(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_think_steps=args.n_think_steps,
        max_seq_len=18,  # Max for Phase 4
        n_topics=len(CURRICULUM_CONFIG['4']['patterns'])  # Max patterns
    ).to(device)

    # Load previous checkpoint if continuing
    if args.resume:
        checkpoint_path = checkpoint_dir / args.resume
        if checkpoint_path.exists():
            print(f"Resuming from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print(f"  Loaded phase {checkpoint.get('phase', '?')} checkpoint")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Train each phase
    results = []
    for phase in phases:
        if phase not in CURRICULUM_CONFIG:
            print(f"Unknown phase: {phase}, skipping")
            continue

        phase_result = train_phase(model, phase, device, args, checkpoint_dir)
        results.append(phase_result)

        if not phase_result.get('mastered', False):
            print(f"\nPhase {phase} not mastered, stopping curriculum")
            break

    # Save final results
    final_log = {
        'script': 'curriculum_train.py',
        'run_id': run_id,
        'phases_trained': [r['phase'] for r in results],
        'results': results,
        'args': vars(args)
    }

    log_file = log_dir / f'curriculum_{run_id}.json'
    with open(log_file, 'w') as f:
        json.dump(final_log, f, indent=2)
    print(f"\nLog saved to: {log_file}")

    print("\n" + "="*70)
    print("CURRICULUM COMPLETE")
    for r in results:
        status = "MASTERED" if r.get('mastered', False) else "incomplete"
        print(f"  Phase {r['phase']}: {r['best_acc']:.1%} ({r['epochs']} epochs) - {status}")
    print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--phases', default='1',
                       help='Comma-separated phases to train: 1,2a,2b,3a,3b,4')
    parser.add_argument('--resume', default=None,
                       help='Checkpoint to resume from')
    parser.add_argument('--n_train', type=int, default=40000)
    parser.add_argument('--n_val', type=int, default=4000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_think_steps', type=int, default=5)

    args = parser.parse_args()
    main(args)
