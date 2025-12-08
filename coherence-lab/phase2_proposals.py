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

__version__ = "0.6.0"  # Graduation + final exam required; clean registry per run

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
                              val_loader=None, topic_calibration=None, epoch=None):
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

        # WEIGHTED TRAINING - graduated topics get maintenance training (from Phase 1)
        tracker = model.learner.self_model.topic_tracker
        graduated_mask = torch.tensor([
            tracker.get_exam_stats(pattern_to_idx[p])['graduated']
            for p in pattern_types
        ], dtype=torch.bool, device=device)
        active_mask = ~graduated_mask

        # Loss weights: 1.0 for active, 0.1 for graduated (maintenance)
        loss_weights = torch.where(graduated_mask,
                                   torch.tensor(0.1, device=device),
                                   torch.tensor(1.0, device=device))

        optimizer.zero_grad()

        # Forward pass - ALL topics
        details = model(tokens, seq_lens, targets=targets, return_details=True)

        # Main loss - weighted by active/graduated status
        logits = details['logits']
        per_sample_loss = F.cross_entropy(logits, targets, reduction='none')
        main_loss = (per_sample_loss * loss_weights).mean()

        # Pattern classification auxiliary - weighted
        pattern_targets = torch.tensor([pattern_to_idx[p] for p in pattern_types], device=device)
        aux_per_sample = F.cross_entropy(details['pattern_logits'], pattern_targets, reduction='none')
        aux_loss = (aux_per_sample * loss_weights).mean()

        # Confidence calibration - weighted
        preds = details['logits'].argmax(dim=-1)
        correct = (preds == targets)
        conf = details['learner_self']['emotions']['confidence'].squeeze()
        conf_per_sample = F.binary_cross_entropy(conf, correct.float(), reduction='none')
        conf_loss = (conf_per_sample * loss_weights).mean()

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

        # Detect TRUE creativity: novel approach + high confidence
        # "It's only creative if you know WHY it worked"
        # v0.5.17: Level-scaled creativity threshold (L0: 0.5, L10: 0.9)
        is_creative = torch.zeros(tokens.size(0), dtype=torch.bool, device=device)
        if details.get('process_eval') is not None and details['process_eval'].get('creativity') is not None:
            # Get per-topic creativity thresholds based on confirmed level
            confirmed_levels = torch.tensor([
                tracker.confirmed_level[pattern_to_idx[p]].item()
                for p in pattern_types
            ], device=device)
            creativity_thresholds = 0.5 + (confirmed_levels / 10.0) * 0.4  # L0: 0.5, L10: 0.9

            creativity_scores = details['process_eval']['creativity'].squeeze()
            novel_approach = creativity_scores > creativity_thresholds
            knew_why = conf.squeeze() > 0.8  # Must be confident it would work
            is_creative = novel_approach & knew_why

        # Get teacher's current goal for showing work
        teacher_goal = model.teacher.current_goal.item()

        # Create pattern indices for per-topic streak tracking
        pattern_indices = torch.tensor([pattern_to_idx[p] for p in pattern_types], device=device)

        should_show, reasons = model.learner.self_model.should_show_work(
            correct, is_creative, conf, int_level,
            teacher_goal=teacher_goal, pattern_indices=pattern_indices
        )

        # Only show for non-graduated topics
        should_show = should_show & active_mask

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

                # === XP AWARD based on show type and outcome ===
                topic_idx = pattern_to_idx[pattern_types[idx_item]]
                was_correct_item = correct[idx].item()

                if reason == 'creative':
                    if was_correct_item:
                        model.learner.self_model.topic_tracker.award_xp(topic_idx, 5)
                    else:
                        model.learner.self_model.topic_tracker.award_xp(topic_idx, -1)
                elif reason == 'streak':
                    completed_streak = model.learner.self_model.last_completed_streak
                    model.learner.self_model.topic_tracker.award_xp(topic_idx, max(1, completed_streak // 5))
                elif reason == 'validation':
                    if was_correct_item:
                        model.learner.self_model.topic_tracker.award_xp(topic_idx, 1)

                # Internalize - weighted by how impressed teacher was
                # For streak shows on completion (failure), still internalize
                should_internalize = correct[idx] or reason == 'streak'
                if should_internalize:
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

                # === DETAILED PROPOSAL LOGGING ===
                # What is the student actually proposing?
                proposal_detail = {
                    'type': proposal_type,
                    'trigger': trigger,
                    'confidence': proposal['confidence'].mean().item(),
                    'batch_idx': batch_idx,
                    'recent_accuracy': recent_accuracy,
                }

                # Add type-specific details
                if proposal_type == 'propose_novel':
                    novel_cat = proposal.get('novel_category', ['unknown'])[0]
                    uncertainty = proposal.get('uncertainty_level', torch.tensor(0))
                    if isinstance(uncertainty, torch.Tensor):
                        uncertainty = uncertainty.mean().item()
                    proposal_detail['novel_category'] = novel_cat
                    proposal_detail['uncertainty_level'] = uncertainty
                    print(f"      [PROPOSE_NOVEL] Category: {novel_cat}, Uncertainty: {uncertainty:.2f}")
                    print(f"        Trigger: {trigger}, Recent acc: {recent_accuracy:.1%}")

                elif proposal_type == 'request_unknown':
                    uncertainty = proposal.get('uncertainty_level', torch.tensor(0))
                    if isinstance(uncertainty, torch.Tensor):
                        uncertainty = uncertainty.mean().item()
                    proposal_detail['uncertainty_level'] = uncertainty
                    # Show which topics triggered this
                    ambiguous_topics = []
                    if topic_calibration:
                        for t_name, cal in topic_calibration.items():
                            acc = cal.get('accuracy', 0.5)
                            conf = cal.get('confidence', 0.5)
                            if 0.4 < acc < 0.7 and 0.4 < conf < 0.7:
                                ambiguous_topics.append(f"{t_name}({acc:.0%})")
                    proposal_detail['ambiguous_topics'] = ambiguous_topics
                    print(f"      [REQUEST_UNKNOWN] 'Something feels off...'")
                    print(f"        Ambiguous topics: {ambiguous_topics or 'general confusion'}")
                    print(f"        Uncertainty: {uncertainty:.2f}, Recent acc: {recent_accuracy:.1%}")

                else:
                    # Standard proposal types
                    topic_idx = proposal['topic_idx'][0].item()
                    # Use pattern_to_idx (always populated) not topic_calibration (empty on epoch 1)
                    topic_names = list(pattern_to_idx.keys())
                    topic_name = topic_names[topic_idx] if topic_idx < len(topic_names) else f"topic_{topic_idx}"
                    magnitude = proposal['magnitude'].mean().item()
                    proposal_detail['topic'] = topic_name
                    proposal_detail['magnitude'] = magnitude
                    print(f"      [{proposal_type.upper()}] Topic: {topic_name}, Magnitude: {magnitude:.2f}")

                    # Diagnostic: if topic is out of valid range, show probability distribution
                    if topic_idx >= len(topic_names):
                        topic_probs = proposal['topic_probs'][0].tolist()
                        valid_probs = topic_probs[:len(topic_names)]
                        invalid_probs = topic_probs[len(topic_names):]
                        print(f"        [OOB topic] valid probs: {[f'{p:.2f}' for p in valid_probs]}, "
                              f"invalid probs: {[f'{p:.2f}' for p in invalid_probs]}")

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

                # Log teacher's response
                decision = evaluation['decision']
                proposal_detail['teacher_decision'] = decision
                if evaluation.get('probing_response'):
                    probes = evaluation['probing_response'].get('probes', [])
                    proposal_detail['teacher_probes'] = [p.get('message', '') for p in probes]
                    print(f"        Teacher probes: {probes[0].get('message', '') if probes else 'none'}")
                elif evaluation.get('novel_evaluation'):
                    novel_eval = evaluation['novel_evaluation']
                    proposal_detail['novel_verdict'] = novel_eval.get('verdict')
                    proposal_detail['novel_message'] = novel_eval.get('message')
                    print(f"        Teacher: {novel_eval.get('message', decision)}")
                else:
                    print(f"        Teacher decision: {decision}")
                    if evaluation.get('redirect_reason'):
                        print(f"        Reason: {evaluation['redirect_reason']}")

                # Apply if approved
                if evaluation['is_approved']:
                    applied = model.learner.self_model.apply_approved_proposal(
                        proposal, evaluation,
                        topic_names=list(pattern_to_idx.keys())
                    )

                    # === CREATE EMERGED TOPIC if propose_novel was approved ===
                    if proposal_type == 'propose_novel' and hasattr(model, 'topic_registry'):
                        novel_cat = proposal.get('novel_category', ['unknown'])[0]
                        # Generate a unique name for the emerged topic
                        emerged_name = f"emerged_{novel_cat}_{len(model.topic_registry)}"
                        topic_id = model.add_emerged_topic(
                            name=emerged_name,
                            epoch=epoch if 'epoch' in dir() else None,
                            metadata={
                                'category': novel_cat,
                                'trigger': trigger,
                                'batch_idx': batch_idx,
                                'recent_accuracy': recent_accuracy
                            }
                        )
                        print(f"        [NEW TOPIC EMERGED] id={topic_id}, name={emerged_name}")

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

    # Determine n_patterns and n_topics from Phase 1 checkpoint (organic growth)
    phase1_path = Path(args.data_dir) / 'phase1_approval_best.pt'
    checkpoint = None
    if phase1_path.exists() and not args.fresh:
        print(f"\nPeeking at phase 1 checkpoint: {phase1_path}")
        checkpoint = torch.load(phase1_path, map_location=device)
        # Read from checkpoint metadata if available
        n_patterns = checkpoint.get('n_patterns', 9)
        n_topics = checkpoint.get('n_topics', n_patterns)
        print(f"  Phase 1 curriculum: {n_patterns} patterns, {n_topics} topics")
    else:
        # Fresh start - use Phase 1's expanded curriculum size
        phase1_patterns = ['alternating', 'repeating', 'incrementing', 'fixed_offset', 'periodic_repeat',
                           'counting', 'modular', 'staircase', 'geometric']
        n_patterns = len(phase1_patterns)
        n_topics = n_patterns
        print(f"\nFresh start: {n_patterns} patterns, {n_topics} topics")

    # Model - dimensions from Phase 1 checkpoint (organic growth)
    model = RelationalSystem(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_think_steps=args.n_think_steps,
        n_topics=n_topics,
        n_patterns=n_patterns
    ).to(device)

    # Load phase 1 checkpoint if we have it (already loaded above for metadata)
    if checkpoint is not None:
        print(f"  Loading weights...")

        # Handle size mismatches for proposal types (we added new ones)
        state_dict = checkpoint['state_dict']
        model_state = model.state_dict()

        # Keys that might have size mismatch due to new proposal types
        proposal_keys = [
            'learner.self_model.proposal_generator.proposal_outcomes',
            'learner.self_model.proposal_generator.proposal_type_head.weight',
            'learner.self_model.proposal_generator.proposal_type_head.bias',
        ]

        for key in proposal_keys:
            if key in state_dict and key in model_state:
                old_shape = state_dict[key].shape
                new_shape = model_state[key].shape
                if old_shape != new_shape:
                    print(f"  Expanding {key}: {old_shape} -> {new_shape}")
                    # Create new tensor with model's shape, copy old values
                    new_tensor = model_state[key].clone()
                    # Copy the old values into the beginning
                    slices = tuple(slice(0, min(o, n)) for o, n in zip(old_shape, new_shape))
                    new_tensor[slices] = state_dict[key][slices]
                    state_dict[key] = new_tensor

        # Load with strict=False for any other new components
        model.load_state_dict(state_dict, strict=False)
        print(f"  Phase 1 val_acc: {checkpoint.get('val_acc', 'N/A')}")
    else:
        print("\nStarting fresh (no phase 1 checkpoint)")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # === DYNAMIC TOPIC REGISTRY ===
    # Load previous registry or create new one with curriculum patterns
    # Each Phase 2 run starts fresh - emerged topics are OUTPUT, not INPUT
    registry_path = Path(args.data_dir) / 'topic_registry.json'
    if registry_path.exists():
        print(f"\nLoading topic registry: {registry_path}")
        topic_registry = DynamicTopicRegistry(registry_path)
        emerged_count = len(topic_registry.get_emerged_topics())
        if emerged_count > 0:
            removed = topic_registry.reset_emerged()
            print(f"  Reset {removed} emerged topics (keeping curriculum only)")
        print(f"  Loaded {len(topic_registry)} curriculum topics")
    else:
        print("\nCreating topic registry with curriculum patterns")
        # Include Phase 1 patterns as curriculum topics (all 9 foundation patterns)
        phase1_patterns = ['alternating', 'repeating', 'incrementing', 'fixed_offset', 'periodic_repeat',
                           'counting', 'modular', 'staircase', 'geometric']
        topic_registry = DynamicTopicRegistry(phase1_patterns + pattern_types)
        print(f"  Initialized with {len(topic_registry)} curriculum topics")

    # Attach registry to model
    model.set_topic_registry(topic_registry)

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
            val_loader, topic_calibration, epoch=epoch
        )

        # Sleep to consolidate
        model.learner.temporal_model.sleep()

        val_metrics = evaluate(model, val_loader, device, pattern_to_idx)

        # Update topic calibration for next epoch (including XP/level/streak from Phase 1)
        for pattern_name, idx in pattern_to_idx.items():
            acc, conf, gap = model.learner.self_model.topic_tracker.get_calibration(idx)
            streak, best_streak, mastered = model.learner.self_model.topic_tracker.get_streak_info(idx)
            xp, level, progress, xp_high = model.learner.self_model.topic_tracker.get_xp_info(idx)
            topic_calibration[pattern_name] = {
                'accuracy': acc,
                'confidence': conf,
                'gap': gap,
                'status': 'guessing' if gap > 0.1 else ('overconfident' if gap < -0.1 else 'calibrated'),
                'streak': streak,
                'best_streak': best_streak,
                'xp': xp,
                'level': level,
                'level_progress': progress
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

        print(f"\nDay {day} (Epoch {epoch:2d})", flush=True)
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

        print(f"  Per-pattern:")
        for pt in pattern_types:
            cal = topic_calibration.get(pt, {})
            acc = val_metrics['per_pattern'].get(pt, 0)
            status = cal.get('status', 'unknown')
            level = cal.get('level', 0)
            xp = cal.get('xp', 0)
            streak = cal.get('streak', 0)
            best_streak = cal.get('best_streak', 0)
            acc_symbol = "O" if acc >= 0.95 else ("o" if acc >= 0.85 else ".")
            cal_symbol = {'calibrated': 'C', 'guessing': '?', 'overconfident': '!', 'unknown': '.'}[status]
            level_bar = "█" * level + "·" * (10 - level)
            streak_info = f"s{streak}" + (f"/{best_streak}" if best_streak > streak else "")
            print(f"    {pt:18s}: {acc:.1%} {acc_symbol} {cal_symbol} L{level:2d} {level_bar} ({xp:.0f}xp) {streak_info}")
        import sys; sys.stdout.flush()

        # === EXAMINATION SYSTEM (from Phase 1) ===
        # Take exams until you fail - first failure stops you for this epoch
        tracker = model.learner.self_model.topic_tracker

        # Class for generating exam problems - Phase 2's harder patterns
        class SinglePatternDataset(Dataset):
            """Exam dataset for a single pattern type."""
            def __init__(self, n_examples, seed, pattern_type):
                random.seed(seed)
                self.examples = []
                for _ in range(n_examples):
                    self.examples.append(self._generate(pattern_type))

            def _generate(self, pt):
                vocab_size = 26
                if pt == 'compositional':
                    length = random.randint(4, 8)
                    start = random.randint(0, 10)
                    if random.random() < 0.5:
                        k = random.randint(1, 3)
                        seq = [start + (i // 2) * k + (i % 2) for i in range(length)]
                    else:
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
                seq = [min(x, vocab_size - 1) for x in seq]
                target = min(target, vocab_size - 1)
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

        exam_results = []
        failed_this_epoch = set()  # Topics that failed - done for this epoch
        exam_round = 0
        any_advanced = True
        while any_advanced:
            any_advanced = False
            for pattern_name, idx in pattern_to_idx.items():
                if idx in failed_this_epoch:
                    continue  # Already failed this epoch, wait for next
                if tracker.check_exam_eligible(idx):
                    # Generate exam batch for this topic
                    current_level = tracker.get_level(idx)
                    target_level = current_level + 1
                    exam_size = tracker.get_exam_size(target_level)

                    exam_data = SinglePatternDataset(
                        n_examples=exam_size,
                        seed=epoch * 1000 + idx + exam_round * 100,
                        pattern_type=pattern_name
                    )
                    exam_loader = DataLoader(exam_data, batch_size=exam_size, collate_fn=collate_fn)

                    # Run exam (no gradient, just evaluation)
                    model.eval()
                    correct_count = 0
                    for batch in exam_loader:
                        tokens = batch['tokens'].to(device)
                        targets = batch['target'].to(device)
                        seq_lens = batch['seq_len']
                        with torch.no_grad():
                            details = model(tokens, seq_lens, targets=targets, return_details=True)
                            preds = details['logits'].argmax(dim=-1)
                            correct_count += (preds == targets).sum().item()
                    model.train()

                    # Take the exam
                    result = tracker.take_exam(idx, correct_count, exam_size)
                    result['topic'] = pattern_name
                    result['target_level'] = target_level
                    exam_results.append(result)

                    if result['passed']:
                        any_advanced = True  # Keep going if anyone passed
                    else:
                        failed_this_epoch.add(idx)  # Done for this epoch
            exam_round += 1

        # Display exam results
        if exam_results:
            print("  Exams:")
            for r in exam_results:
                if r['passed']:
                    status = f"Ready for L{r['new_level']}"
                    if r['graduated']:
                        status += " - GRADUATED!"
                    print(f"    {r['topic']:18s}: {r['score']:.0%} >= {r['threshold']:.0%} - {status}")
                else:
                    print(f"    {r['topic']:18s}: {r['score']:.0%} < {r['threshold']:.0%} - More practice needed (cooldown: {r['cooldown']} epochs)")
            history[-1]['exams'] = exam_results

        # Count graduated topics
        graduated_count = sum(1 for idx in pattern_to_idx.values() if tracker.get_exam_stats(idx)['graduated'])
        if graduated_count > 0:
            print(f"  Graduated: {graduated_count}/{len(pattern_types)} topics")

        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': best_acc,
                'n_patterns': n_patterns,  # For organic growth across phases
                'n_topics': len(topic_registry)  # May have grown via emerged topics
            }, Path(args.data_dir) / 'phase2_proposals_best.pt')
            # Save topic registry with checkpoint
            topic_registry.save(registry_path)
            print(f"  [Registry saved: {len(topic_registry)} topics]")

        # Check mastery - requires accuracy AND calibration at high level
        all_accurate = all(val_metrics['per_pattern'].get(pt, 0) >= 0.90 for pt in pattern_types)

        # Check if all topics have graduated individually
        all_graduated = all(
            tracker.get_exam_stats(pattern_to_idx[pt])['graduated']
            for pt in pattern_types
        )

        if all_graduated:
            # === FINAL COMPREHENSIVE EXAM ===
            # All topics graduated individually - now prove it all together
            print(f"\n  === FINAL COMPREHENSIVE EXAM ===")
            print(f"  All topics at L10 - proving mastery with fresh problems...")
            topic_registry.save(registry_path)
            print(f"  [Registry saved: {len(topic_registry)} topics]")

            final_results = []
            any_failed = False

            for pattern_name, idx in pattern_to_idx.items():
                # Final exam: 32 problems per topic, 90% threshold
                final_size = 32
                final_threshold = 0.90
                exam_seed = epoch * 10000 + idx

                final_data = SinglePatternDataset(
                    n_examples=final_size,
                    seed=exam_seed,
                    pattern_type=pattern_name
                )
                final_loader = DataLoader(final_data, batch_size=final_size, collate_fn=collate_fn)

                model.eval()
                correct_count = 0
                for batch in final_loader:
                    tokens = batch['tokens'].to(device)
                    targets = batch['target'].to(device)
                    seq_lens = batch['seq_len']
                    with torch.no_grad():
                        details = model(tokens, seq_lens, targets=targets, return_details=True)
                        preds = details['logits'].argmax(dim=-1)
                        correct_count += (preds == targets).sum().item()
                model.train()

                score = correct_count / final_size
                passed = score >= final_threshold

                # Show raw counts to verify fresh inference
                if passed:
                    print(f"    {pattern_name:18s}: {correct_count}/{final_size} ({score:.0%}) - PASSED")
                else:
                    print(f"    {pattern_name:18s}: {correct_count}/{final_size} ({score:.0%}) - FAILED (kicked back)")
                    # Kick back: un-graduate, reset confirmed level, apply penalty
                    tracker.topic_graduated[idx] = False
                    tracker.confirmed_level[idx] = 7  # Knocked back to L7
                    tracker.topic_xp[idx] = tracker.xp_threshold(7)  # Reset XP to L7 threshold
                    tracker.topic_streak[idx] = 0  # Reset streak
                    any_failed = True

                final_results.append({'topic': pattern_name, 'score': score, 'passed': passed})

            # Log final exam results to history
            history[-1]['final_exam'] = final_results

            if not any_failed:
                early_stop_reason = "Phase 2 complete - all topics PASSED FINAL EXAM"
                print(f"\n*** Phase 2 COMPLETE! All topics PASSED FINAL EXAM! ***")
                print(f"    Trust: {trust:.1%}, Internalization: {int_level:.1%}")
                print(f"    Proposal success rate: {prop_success:.1%}")
                print(f"    Comprehensive mastery proven - ready for Phase 3!")
                break
            else:
                print(f"\n  Some topics failed final - back to training!")

    print("\n" + "=" * 70)
    print(f"Best accuracy: {best_acc:.1%}")
    print(f"Final trust: {trust:.1%}")
    print(f"Final internalization: {int_level:.1%}")
    print(f"Final proposal success rate: {prop_success:.1%}")

    # Report emerged topics
    emerged = topic_registry.get_emerged_topics()
    if emerged:
        print(f"\nEmerged topics during training: {len(emerged)}")
        for topic in emerged:
            print(f"  {topic['id']}: {topic['name']} (epoch {topic.get('created_epoch', '?')})")
            if topic.get('metadata'):
                print(f"      category: {topic['metadata'].get('category', '?')}")
    else:
        print("\nNo topics emerged during training")

    # Final registry save
    topic_registry.save(registry_path)
    print(f"\nTopic registry saved: {registry_path}")
    print(f"  Total topics: {len(topic_registry)} ({len(emerged)} emerged)")
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
        'emerged_topics': [
            {'id': t['id'], 'name': t['name'], 'epoch': t.get('created_epoch'),
             'category': t.get('metadata', {}).get('category')}
            for t in emerged
        ],
        'total_topics': len(topic_registry),
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
