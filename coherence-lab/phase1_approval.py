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
    RelationalSystem, PatternDataset, collate_fn, evaluate, DynamicTopicRegistry
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

        # === APPROVAL-SEEKING BEHAVIOR with RISING BARS ===
        # Student decides whether to show work to teacher
        # As internalization grows, student becomes more selective

        # Get internalization level for rising show bar
        int_level = torch.sigmoid(model.learner.other_model.internalization_level).item()

        # Detect creativity (simplified - check if process_eval available)
        is_creative = torch.zeros(tokens.size(0), dtype=torch.bool, device=device)
        if details.get('process_eval') is not None and details['process_eval'].get('creativity') is not None:
            is_creative = details['process_eval']['creativity'].squeeze() > 0.5

        # Get teacher's current goal for showing work
        teacher_goal = model.teacher.current_goal.item()

        # Student decides: should I show this? (rising bar based on internalization)
        should_show, reasons = model.learner.self_model.should_show_work(
            correct, is_creative, conf, int_level, teacher_goal=teacher_goal
        )

        # Process shows with RISING APPROVAL BAR
        if should_show.any():
            show_indices = should_show.nonzero(as_tuple=True)[0]

            for idx in show_indices:
                idx_item = idx.item()
                reason = reasons[idx_item]

                # Count by reason
                if reason in show_reasons:
                    show_reasons[reason] += 1
                show_count += 1

                # Teacher responds to shown work (4-return API with goal-setting)
                learner_state = details['learner_self']['internal_state'][idx:idx+1]
                was_correct_single = correct[idx:idx+1]

                teacher_response, meets_bar, approval_level, goal_action = model.teacher.respond_to_shown_work(
                    learner_state, was_correct_single, reason
                )

                # Update trust based on positive interaction
                # (always positive, but strength varies)
                approval_count += 1
                model.learner.other_model.update_trust(outcome_was_good=True)

                # Update approval calibration in self model
                model.learner.self_model.update_after_approval(meets_bar, reason)

                # Internalize - weighted by how impressed teacher was
                if correct[idx]:
                    model.learner.other_model.internalize(
                        teacher_response,
                        torch.tensor(1.0),
                        approval_level=approval_level
                    )

                # === GOAL-SETTING NEGOTIATION ===
                # If teacher wants to raise the bar, handle it
                if goal_action is not None:
                    if goal_action['is_negotiation']:
                        # Teacher asks: "How many do you think you should do?"
                        # Student proposes based on their learned estimate
                        student_proposal = model.learner.self_model.propose_show_goal(
                            conf.mean().item()
                        )
                        # Teacher evaluates and counter-offers if needed
                        negotiation_result = model.teacher.evaluate_student_goal_proposal(student_proposal)
                        # Student learns from the feedback
                        model.learner.self_model.update_goal_estimate_from_feedback(negotiation_result)
                    else:
                        # Teacher directive: "Let's do X next time"
                        model.teacher.current_goal.fill_(goal_action['goal'])

        # === TEACHER MONITORING (un-shown work) ===
        # Teacher observes ALL work, notices quiet competence
        should_acknowledge, unshown_streak = model.teacher.monitor_unshown_work(
            correct, should_show
        )
        if should_acknowledge:
            # Teacher gives unsolicited acknowledgment
            # "I noticed you've been getting these right without checking with me!"
            model.learner.other_model.internalize_from_quiet_competence(unshown_streak)

        # === SELF-RESTRAINT INTERNALIZATION ===
        # Correct answers where student chose NOT to show → internal confidence
        correct_count = correct.sum().item()
        shown_count = should_show.sum().item()
        unshown_correct = (correct & ~should_show).sum().item()
        model.learner.other_model.internalize_from_self_restraint(
            was_correct_count=correct_count,
            chose_not_to_show_count=unshown_correct
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

    # Get teacher's rising bar metrics
    approval_metrics = model.teacher.get_approval_metrics()

    # Get goal-setting metrics
    goal_cal_rate = model.learner.self_model.get_goal_calibration_rate()

    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples,
        'show_rate': show_count / total_samples,
        'approval_rate': approval_rate,
        'show_reasons': show_reasons,
        'perceived_competence': approval_metrics['perceived_competence'],
        'approval_threshold': approval_metrics['approval_threshold'],
        'teacher_goal': model.teacher.current_goal.item(),
        'student_goal_estimate': model.learner.self_model.goal_estimate.item(),
        'teacher_impressedness': model.teacher.impressedness.item(),
        'goal_calibration_rate': goal_cal_rate
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
    # periodic_repeat added to prep for long_range (teaches "look back N positions")
    pattern_types = ['alternating', 'repeating', 'incrementing', 'fixed_offset', 'periodic_repeat']
    pattern_to_idx = {p: i for i, p in enumerate(pattern_types)}
    print(f"Pattern types: {pattern_types}")

    # Generate data
    print("\nGenerating easy pattern data...")
    train_data = PatternDataset(n_examples=args.n_train, seed=42, pattern_types=pattern_types)
    val_data = PatternDataset(n_examples=args.n_val, seed=123, pattern_types=pattern_types)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=collate_fn)

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Model - explicit n_topics=10 to match phase 2 checkpoint loading
    model = RelationalSystem(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_think_steps=args.n_think_steps,
        n_topics=10
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # === DYNAMIC TOPIC REGISTRY ===
    # Create registry with Phase 1 curriculum patterns
    registry_path = Path(args.data_dir) / 'topic_registry.json'
    print("\nCreating topic registry with Phase 1 curriculum patterns")
    topic_registry = DynamicTopicRegistry(pattern_types)
    print(f"  Initialized with {len(topic_registry)} curriculum topics")

    # Attach registry to model
    model.set_topic_registry(topic_registry)

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
            'per_pattern': val_metrics['per_pattern'],
            'teacher_goal': train_metrics['teacher_goal'],
            'student_goal_estimate': train_metrics['student_goal_estimate'],
            'teacher_impressedness': train_metrics['teacher_impressedness'],
            'goal_calibration_rate': train_metrics['goal_calibration_rate']
        })

        print(f"\nDay {day} (Epoch {epoch:2d})", flush=True)
        print(f"  Train: loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.1%}")
        print(f"  Val: acc={val_metrics['accuracy']:.1%}")
        print(f"  Shows: {train_metrics['show_rate']:.1%} of answers shown to teacher")
        print(f"  Show reasons: {train_metrics['show_reasons']}")
        print(f"  Internalization: {int_level:.1%}, Trust: {trust:.1%}")
        print(f"  Rising bars: competence={train_metrics['perceived_competence']:.1%}, threshold={train_metrics['approval_threshold']:.1%}")
        print(f"  Goals: teacher={train_metrics['teacher_goal']}, student_est={train_metrics['student_goal_estimate']:.1f}, impressed={train_metrics['teacher_impressedness']:.0%}")
        print("  Per-pattern:")
        for pt in pattern_types:
            acc = val_metrics['per_pattern'].get(pt, 0)
            status = "O" if acc >= 0.95 else ("o" if acc >= 0.85 else ".")
            print(f"    {pt:15s}: {acc:.1%} {status}")
        import sys; sys.stdout.flush()

        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': best_acc
            }, Path(args.data_dir) / 'phase1_approval_best.pt')
            # Save topic registry with checkpoint
            topic_registry.save(registry_path)
            print(f"  [Registry saved: {len(topic_registry)} topics]")

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

    # Final registry save and report
    topic_registry.save(registry_path)
    print(f"\nTopic registry saved: {registry_path}")
    print(f"  Total topics: {len(topic_registry)} curriculum patterns")
    print("=" * 70)

    # Save log
    run_log = {
        'script': 'phase1_approval.py',
        'run_id': run_id,
        'best_acc': best_acc,
        'final_trust': trust,
        'final_internalization': int_level,
        'epochs_completed': len(history),
        'curriculum_topics': [topic_registry.get_name(i) for i in range(len(topic_registry))],
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
