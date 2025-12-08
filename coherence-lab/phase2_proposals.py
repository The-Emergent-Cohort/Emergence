"""
Phase 2: Self-Modification Proposals
Coherence Lab - Emergence Project

Building on phase 1's approval-seeking foundation:
- Learner can now propose modifications to itself
- Shows proposals to teacher for approval
- Approved proposals are applied
- Tracks which proposals lead to improvement

This gives the occupant agency over its own development.
The model learns to reflect on its learning process.
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


# =============================================================================
# HARD PATTERN DATASET (from hard_patterns.py)
# =============================================================================

class HardPatternDataset(Dataset):
    """Hard patterns for phase 2 - compositional, long_range, fibonacci_like."""

    def __init__(self, n_examples=10000, vocab_size=26, seed=None, difficulty='medium'):
        if seed:
            random.seed(seed)
        self.vocab_size = vocab_size
        self.difficulty = difficulty
        self.examples = []
        pattern_types = ['compositional', 'long_range', 'fibonacci_like']
        for _ in range(n_examples):
            pt = random.choice(pattern_types)
            self.examples.append(self._generate(pt))

    def _generate(self, pt):
        if pt == 'compositional':
            length = random.randint(4, 8)
            start = random.randint(0, 10)
            # Alternating increment patterns
            if random.random() < 0.5:
                k = random.randint(1, 3)
                seq = [start + (i // 2) * k + (i % 2) for i in range(length)]
            else:
                # Nested pattern
                k1, k2 = random.randint(1, 2), random.randint(2, 4)
                seq = [start + (i % 2) * k1 + (i // 2) * k2 for i in range(length)]
            target = seq[-1] + (seq[-1] - seq[-2]) if len(seq) >= 2 else seq[-1] + 1

        elif pt == 'long_range':
            length = random.randint(6, 10)
            period = random.randint(2, 4)
            base = [random.randint(0, 10) for _ in range(period)]
            seq = [base[i % period] + (i // period) for i in range(length)]
            target = base[length % period] + (length // period)

        else:  # fibonacci_like
            length = random.randint(4, 7)
            a, b = random.randint(1, 5), random.randint(1, 5)
            seq = [a, b]
            for _ in range(length - 2):
                seq.append(seq[-1] + seq[-2])
            target = seq[-1] + seq[-2]

        # Clamp to vocab
        seq = [min(x, self.vocab_size - 1) for x in seq]
        target = min(target, self.vocab_size - 1)
        return {'sequence': seq, 'target': target, 'pattern_type': pt}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        padded = ex['sequence'] + [0] * (12 - len(ex['sequence']))
        return {
            'sequence': torch.tensor(padded[:12], dtype=torch.long),
            'target': torch.tensor(ex['target'], dtype=torch.long),
            'seq_len': len(ex['sequence']),
            'pattern_type': ex['pattern_type']
        }


# =============================================================================
# TRAINING WITH PROPOSALS
# =============================================================================

def train_day_with_proposals(model, loader, optimizer, criterion, device, pattern_to_idx,
                              val_loader=None, topic_calibration=None):
    """
    Training loop with self-modification proposals.

    After batches, learner may:
    1. Propose a modification to itself
    2. Show the proposal to teacher
    3. If approved, apply the modification
    4. Track outcomes for future proposals
    """
    model.train()
    model.learner.temporal_model.wake()

    total_loss, total_correct, total_samples = 0, 0, 0
    show_count, approval_count = 0, 0
    proposal_count = 0
    proposals_approved = 0
    proposals_modified = 0
    proposals_redirected = 0
    proposal_types = {}

    # Track recent accuracy for proposal triggers
    recent_window = []
    recent_accuracy = 0.5

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

        # Track accuracy
        batch_acc = correct.float().mean().item()
        recent_window.append(batch_acc)
        if len(recent_window) > 50:
            recent_window.pop(0)
        recent_accuracy = sum(recent_window) / len(recent_window)

        # === APPROVAL-SEEKING (from phase 1) ===
        int_level = torch.sigmoid(model.learner.other_model.internalization_level).item()
        is_creative = torch.zeros(tokens.size(0), dtype=torch.bool, device=device)
        if details.get('process_eval') is not None and details['process_eval'].get('creativity') is not None:
            is_creative = details['process_eval']['creativity'].squeeze() > 0.5

        # Get teacher's current goal for showing work
        teacher_goal = model.teacher.current_goal.item()

        should_show, reasons = model.learner.self_model.should_show_work(
            correct, is_creative, conf, int_level, teacher_goal=teacher_goal
        )

        if should_show.any():
            show_indices = should_show.nonzero(as_tuple=True)[0]
            for idx in show_indices:
                idx_item = idx.item()
                reason = reasons[idx_item]
                show_count += 1

                learner_state = details['learner_self']['internal_state'][idx:idx+1]
                was_correct_single = correct[idx:idx+1]

                # Teacher responds to shown work (4-return API with goal-setting)
                teacher_response, meets_bar, approval_level, goal_action = model.teacher.respond_to_shown_work(
                    learner_state, was_correct_single, reason
                )

                # Update trust based on positive interaction (always positive, strength varies)
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
            model.learner.other_model.internalize_from_quiet_competence(unshown_streak)

        # === SELF-RESTRAINT INTERNALIZATION ===
        # Correct answers where student chose NOT to show → internal confidence
        correct_count = correct.sum().item()
        unshown_correct = (correct & ~should_show).sum().item()
        model.learner.other_model.internalize_from_self_restraint(
            was_correct_count=correct_count,
            chose_not_to_show_count=unshown_correct
        )

        # === SELF-MODIFICATION PROPOSALS ===
        # Every 20 batches, consider making a proposal
        if batch_idx > 0 and batch_idx % 20 == 0:
            cognitive_state = details['learner_self']['internal_state'].mean(dim=0, keepdim=True)

            # Generate proposal if conditions are met
            proposal, trigger = model.learner.self_model.generate_self_modification_proposal(
                cognitive_state, topic_calibration, recent_accuracy
            )

            if proposal is not None:
                proposal_count += 1
                proposal_type = proposal['type_name'][0]
                proposal_types[proposal_type] = proposal_types.get(proposal_type, 0) + 1

                # Show proposal to teacher
                evaluation, response = model.teacher.evaluate_self_modification_proposal(
                    proposal, cognitive_state, topic_calibration
                )

                # Track decision
                if evaluation['decision'] == 'approve':
                    proposals_approved += 1
                elif evaluation['decision'] == 'modify':
                    proposals_modified += 1
                else:
                    proposals_redirected += 1

                # Apply if approved
                if evaluation['is_approved']:
                    applied = model.learner.self_model.apply_approved_proposal(
                        proposal, evaluation,
                        topic_names=list(pattern_to_idx.keys())
                    )

                    # Update proposal generator with outcome
                    # (We'll measure success by improvement in next batches)
                    model.learner.self_model.proposal_generator.update_outcome(
                        proposal['type_idx'][0],
                        proposal['topic_idx'][0],
                        was_successful=True  # Approved = success for now
                    )

        # Track metrics
        total_loss += main_loss.item() * tokens.size(0)
        total_correct += correct.sum().item()
        total_samples += tokens.size(0)

        # Update topic tracker
        pattern_indices = torch.tensor([pattern_to_idx[p] for p in pattern_types], device=device)
        model.learner.self_model.topic_tracker.update(pattern_indices, correct, conf)

    # Calculate rates
    approval_rate = approval_count / max(1, show_count)

    # Get teacher's rising bar and goal-setting metrics
    approval_metrics = model.teacher.get_approval_metrics()

    # Get student's goal calibration
    student_goal_calibration = model.learner.self_model.get_goal_calibration_rate()

    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples,
        'show_rate': show_count / total_samples,
        'approval_rate': approval_rate,
        'perceived_competence': approval_metrics['perceived_competence'],
        'approval_threshold': approval_metrics['approval_threshold'],
        'impressedness': approval_metrics['impressedness'],
        'current_goal': approval_metrics['current_goal'],
        'goals_met_count': approval_metrics['goals_met_count'],
        'student_goal_estimate': model.learner.self_model.goal_estimate.item(),
        'student_goal_calibration': student_goal_calibration,
        'proposals': {
            'total': proposal_count,
            'approved': proposals_approved,
            'modified': proposals_modified,
            'redirected': proposals_redirected,
            'by_type': proposal_types
        }
    }


def main(args):
    # Setup logging
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.data_dir) / 'runs'
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f'phase2_proposals_{run_id}.json'

    print("=" * 70)
    print("Phase 2: Self-Modification Proposals")
    print(f"Run ID: {run_id}")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Hard patterns for phase 2
    pattern_types = ['compositional', 'long_range', 'fibonacci_like']
    pattern_to_idx = {p: i for i, p in enumerate(pattern_types)}
    print(f"Pattern types: {pattern_types}")

    # Generate data
    print("\nGenerating hard pattern data...")
    train_data = HardPatternDataset(n_examples=args.n_train, seed=42, difficulty='medium')
    val_data = HardPatternDataset(n_examples=args.n_val, seed=123, difficulty='medium')

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=collate_fn)

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Model - use n_topics=10 to match phase 1 checkpoint
    # (phase 2 only uses 3 topics but needs compatible tensor sizes)
    model = RelationalSystem(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_think_steps=args.n_think_steps,
        n_topics=10
    ).to(device)

    # Load phase 1 checkpoint if available
    phase1_path = Path(args.data_dir) / 'phase1_approval_best.pt'
    if phase1_path.exists() and not args.fresh:
        print(f"\nLoading phase 1 checkpoint: {phase1_path}")
        checkpoint = torch.load(phase1_path, map_location=device)
        # Load with strict=False since we added new components
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f"  Phase 1 val_acc: {checkpoint.get('val_acc', 'N/A')}")
    else:
        print("\nStarting fresh (no phase 1 checkpoint)")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print("\nTraining with self-modification proposals...")
    print("-" * 70)

    best_acc = 0
    history = []
    topic_calibration = {}

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_day_with_proposals(
            model, train_loader, optimizer, criterion, device, pattern_to_idx,
            val_loader, topic_calibration
        )

        # Sleep to consolidate
        model.learner.temporal_model.sleep()

        val_metrics = evaluate(model, val_loader, device, pattern_to_idx)

        # Update topic calibration for next epoch
        for pattern_name, idx in pattern_to_idx.items():
            acc, conf, gap = model.learner.self_model.topic_tracker.get_calibration(idx)
            topic_calibration[pattern_name] = {
                'accuracy': acc,
                'confidence': conf,
                'gap': gap,
                'status': 'guessing' if gap > 0.1 else ('overconfident' if gap < -0.1 else 'calibrated')
            }

        # Developmental state
        day = model.learner.temporal_model.current_day.item()
        int_level = torch.sigmoid(model.learner.other_model.internalization_level).item()
        trust = torch.sigmoid(model.learner.other_model.trust).item()
        show_cal = model.learner.self_model.show_calibration.item()

        # Proposal success rate
        prop_success = model.learner.self_model.proposal_generator.get_proposal_success_rate()

        history.append({
            'epoch': epoch,
            'train_acc': train_metrics['accuracy'],
            'train_loss': train_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
            'show_rate': train_metrics['show_rate'],
            'approval_rate': train_metrics['approval_rate'],
            'perceived_competence': train_metrics['perceived_competence'],
            'approval_threshold': train_metrics['approval_threshold'],
            'impressedness': train_metrics['impressedness'],
            'current_goal': train_metrics['current_goal'],
            'student_goal_estimate': train_metrics['student_goal_estimate'],
            'student_goal_calibration': train_metrics['student_goal_calibration'],
            'proposals': train_metrics['proposals'],
            'internalization': int_level,
            'trust': trust,
            'show_calibration': show_cal,
            'proposal_success_rate': prop_success,
            'per_pattern': val_metrics['per_pattern'],
            'topic_calibration': topic_calibration.copy()
        })

        print(f"\nDay {day} (Epoch {epoch:2d})")
        print(f"  Train: loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.1%}")
        print(f"  Val: acc={val_metrics['accuracy']:.1%}")
        print(f"  Shows: {train_metrics['show_rate']:.1%} shown, {train_metrics['approval_rate']:.1%} approved")
        print(f"  Internalization: {int_level:.1%}, Trust: {trust:.1%}")
        print(f"  Goal-setting: goal={train_metrics['current_goal']:.0f}, student_est={train_metrics['student_goal_estimate']:.1f}, impressed={train_metrics['impressedness']:.0%}")

        # Proposal metrics
        props = train_metrics['proposals']
        print(f"  Proposals: {props['total']} total")
        if props['total'] > 0:
            print(f"    Approved: {props['approved']}, Modified: {props['modified']}, Redirected: {props['redirected']}")
            print(f"    Types: {props['by_type']}")
            print(f"    Success rate: {prop_success:.1%}")

        print(f"  Topic calibration:")
        for pt in pattern_types:
            cal = topic_calibration.get(pt, {})
            acc = val_metrics['per_pattern'].get(pt, 0)
            status = cal.get('status', 'unknown')
            symbol = {'calibrated': 'O', 'guessing': '?', 'overconfident': '!', 'unknown': '.'}[status]
            print(f"    {pt:18s}: {acc:.1%} [{status:12s}] {symbol}")

        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': best_acc
            }, Path(args.data_dir) / 'phase2_proposals_best.pt')

        # Check mastery - requires accuracy AND calibration at high level
        all_accurate = all(val_metrics['per_pattern'].get(pt, 0) >= 0.90 for pt in pattern_types)

        # Calibration requires BOTH status='calibrated' AND actual competence
        # Being "calibrated" at 50% accuracy = knowing you're guessing, not mastery
        def is_truly_calibrated(pt):
            cal = topic_calibration.get(pt, {})
            return (cal.get('status') == 'calibrated' and
                    cal.get('accuracy', 0) >= 0.80)  # Must be competent, not just calibrated

        all_calibrated = all(is_truly_calibrated(pt) for pt in pattern_types)

        # Identify problems: uncalibrated OR calibrated-but-guessing
        problems = []
        for pt in pattern_types:
            cal = topic_calibration.get(pt, {})
            if cal.get('status') != 'calibrated':
                problems.append(f"{pt} ({cal.get('status', 'unknown')})")
            elif cal.get('accuracy', 0) < 0.80:
                problems.append(f"{pt} (calibrated but only {cal.get('accuracy', 0):.0%} - still learning)")

        if all_accurate and val_metrics['accuracy'] >= 0.90:
            if all_calibrated:
                early_stop_reason = "Phase 2 complete - all topics accurate and calibrated"
                print(f"\n*** Phase 2 complete! Self-modification proposals working. ***")
                print(f"    Trust: {trust:.1%}, Internalization: {int_level:.1%}")
                print(f"    Proposal success rate: {prop_success:.1%}")
                break
            else:
                # Accurate but not truly calibrated
                print(f"\n  [Teacher notes: Not ready to graduate - {problems}]")

    print("\n" + "=" * 70)
    print(f"Best accuracy: {best_acc:.1%}")
    print(f"Final trust: {trust:.1%}")
    print(f"Final internalization: {int_level:.1%}")
    print(f"Final proposal success rate: {prop_success:.1%}")
    print("=" * 70)

    # Save log
    run_log = {
        'script': 'phase2_proposals.py',
        'run_id': run_id,
        'phase1_checkpoint': str(phase1_path) if phase1_path.exists() and not args.fresh else None,
        'phase1_val_acc': checkpoint.get('val_acc') if phase1_path.exists() and not args.fresh else None,
        'best_acc': best_acc,
        'final_trust': trust,
        'final_internalization': int_level,
        'final_proposal_success_rate': prop_success,
        'epochs_completed': len(history),
        'early_stop_reason': early_stop_reason if 'early_stop_reason' in dir() else None,
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
    parser.add_argument('--n-train', type=int, default=40000)
    parser.add_argument('--n-val', type=int, default=4000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-think-steps', type=int, default=5)
    parser.add_argument('--fresh', action='store_true', help='Start fresh without phase 1 checkpoint')

    args = parser.parse_args()
    main(args)
