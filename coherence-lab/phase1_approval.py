"""
Phase 1: Easy Patterns with Approval-Seeking
Coherence Lab - Emergence Project

Building the social learning foundation:
- Student decides when to "show work" to teacher
- Teacher always responds positively (even corrections are encouraging)
- Trust builds through positive interactions
- Approval-seeking calibration develops over time

This is the developmental foundation for later phases.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
from pathlib import Path
import argparse
import json
from datetime import datetime

from relational_model import (
    RelationalSystem, PatternDataset, collate_fn, evaluate
)


def train_day_with_approval(model, loader, optimizer, criterion, device, pattern_to_idx):
    """
    Training loop with approval-seeking behavior.

    After each batch:
    1. Student solves problems
    2. Student decides: should I show this to teacher?
    3. If showing: teacher responds (always positive)
    4. Trust and approval calibration update
    """
    model.train()

    # Wake up
    model.learner.temporal_model.wake()

    total_loss, total_correct, total_samples = 0, 0, 0
    show_count = 0
    approval_count = 0
    show_reasons = {'creative': 0, 'streak': 0, 'validation': 0, 'spontaneous': 0}

    for batch_idx, batch in enumerate(loader):
        tokens = batch['tokens'].to(device)
        targets = batch['target'].to(device)
        seq_lens = batch['seq_len']
        pattern_types = batch['pattern_type']

        optimizer.zero_grad()

        # Forward pass
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

        # === APPROVAL-SEEKING BEHAVIOR ===
        # Student decides whether to show work to teacher

        # Get internalization level for spontaneous show rate
        int_level = torch.sigmoid(model.learner.other_model.internalization_level).item()

        # Detect creativity (simplified - check if process_eval available)
        is_creative = torch.zeros(tokens.size(0), dtype=torch.bool, device=device)
        if details.get('process_eval') is not None and details['process_eval'].get('creativity') is not None:
            is_creative = details['process_eval']['creativity'].squeeze() > 0.5

        # Student decides: should I show this?
        should_show, reasons = model.learner.self_model.should_show_work(
            correct, is_creative, conf, int_level
        )

        # Process shows
        if should_show.any():
            show_indices = should_show.nonzero(as_tuple=True)[0]

            for idx in show_indices:
                idx_item = idx.item()
                reason = reasons[idx_item]

                # Count by reason
                if reason in show_reasons:
                    show_reasons[reason] += 1
                show_count += 1

                # Teacher responds to shown work
                learner_state = details['learner_self']['internal_state'][idx:idx+1]
                was_correct_single = correct[idx:idx+1]

                teacher_response, was_approved = model.teacher.respond_to_shown_work(
                    learner_state, was_correct_single, reason
                )

                # Update trust based on positive interaction
                if was_approved:
                    approval_count += 1
                    model.learner.other_model.update_trust(outcome_was_good=True)

                    # Also update approval calibration in self model
                    model.learner.self_model.update_after_approval(True, reason)

                    # If correct, potentially internalize this as "good pattern"
                    if correct[idx]:
                        model.learner.other_model.internalize(
                            teacher_response,
                            torch.tensor(1.0)
                        )

        # Track metrics
        total_loss += main_loss.item() * tokens.size(0)
        total_correct += correct.sum().item()
        total_samples += tokens.size(0)

        # Update topic tracker
        pattern_indices = torch.tensor([pattern_to_idx[p] for p in pattern_types], device=device)
        model.learner.self_model.topic_tracker.update(pattern_indices, correct, conf)

    # Calculate approval rate
    approval_rate = approval_count / max(1, show_count)

    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples,
        'show_rate': show_count / total_samples,
        'approval_rate': approval_rate,
        'show_reasons': show_reasons
    }


def main(args):
    # Setup logging
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.data_dir) / 'runs'
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f'phase1_approval_{run_id}.json'

    print("=" * 70)
    print("Phase 1: Easy Patterns with Approval-Seeking")
    print(f"Run ID: {run_id}")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Easy patterns for foundation building
    pattern_types = ['alternating', 'repeating', 'incrementing', 'fixed_offset']
    pattern_to_idx = {p: i for i, p in enumerate(pattern_types)}
    print(f"Pattern types: {pattern_types}")

    # Generate data
    print("\nGenerating easy pattern data...")
    train_data = PatternDataset(n_examples=args.n_train, seed=42)
    val_data = PatternDataset(n_examples=args.n_val, seed=123)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=collate_fn)

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Model
    model = RelationalSystem(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_think_steps=args.n_think_steps
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print("\nTraining with approval-seeking behavior...")
    print("-" * 70)

    best_acc = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_day_with_approval(
            model, train_loader, optimizer, criterion, device, pattern_to_idx
        )

        # Sleep to consolidate
        model.learner.temporal_model.sleep()

        val_metrics = evaluate(model, val_loader, device, pattern_to_idx)

        # Developmental state
        day = model.learner.temporal_model.current_day.item()
        int_level = torch.sigmoid(model.learner.other_model.internalization_level).item()
        trust = torch.sigmoid(model.learner.other_model.trust).item()
        show_cal = model.learner.self_model.show_calibration.item()

        history.append({
            'epoch': epoch,
            'train_acc': train_metrics['accuracy'],
            'train_loss': train_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
            'show_rate': train_metrics['show_rate'],
            'approval_rate': train_metrics['approval_rate'],
            'show_reasons': train_metrics['show_reasons'],
            'internalization': int_level,
            'trust': trust,
            'show_calibration': show_cal,
            'per_pattern': val_metrics['per_pattern']
        })

        print(f"\nDay {day} (Epoch {epoch:2d})")
        print(f"  Train: loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.1%}")
        print(f"  Val: acc={val_metrics['accuracy']:.1%}")
        print(f"  Shows: {train_metrics['show_rate']:.1%} of answers shown to teacher")
        print(f"  Approval: {train_metrics['approval_rate']:.1%} (should be ~100%)")
        print(f"  Show reasons: {train_metrics['show_reasons']}")
        print(f"  Internalization: {int_level:.1%}, Trust: {trust:.1%}")
        print(f"  Show calibration: {show_cal:.2f}")
        print("  Per-pattern:")
        for pt in pattern_types:
            acc = val_metrics['per_pattern'].get(pt, 0)
            status = "O" if acc >= 0.95 else ("o" if acc >= 0.85 else ".")
            print(f"    {pt:15s}: {acc:.1%} {status}")

        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': best_acc
            }, Path(args.data_dir) / 'phase1_approval_best.pt')

        # Check mastery - easy patterns should be ~100%
        all_good = all(val_metrics['per_pattern'].get(pt, 0) >= 0.95 for pt in pattern_types)
        if all_good and val_metrics['accuracy'] >= 0.98:
            print(f"\n*** Phase 1 complete! Approval-seeking foundation built. ***")
            print(f"    Trust: {trust:.1%}, Internalization: {int_level:.1%}")
            break

    print("\n" + "=" * 70)
    print(f"Best accuracy: {best_acc:.1%}")
    print(f"Final trust: {trust:.1%}")
    print(f"Final internalization: {int_level:.1%}")
    print("=" * 70)

    # Save log
    run_log = {
        'run_id': run_id,
        'best_acc': best_acc,
        'final_trust': trust,
        'final_internalization': int_level,
        'epochs_completed': len(history),
        'args': vars(args),
        'history': history
    }
    with open(log_file, 'w') as f:
        json.dump(run_log, f, indent=2)
    print(f"Run log saved: {log_file}")

    return best_acc, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--n-train', type=int, default=20000)
    parser.add_argument('--n-val', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-think-steps', type=int, default=5)

    args = parser.parse_args()
    main(args)
