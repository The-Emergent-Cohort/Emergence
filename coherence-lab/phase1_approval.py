"""
Phase 1: Easy Patterns with Approval-Seeking
Coherence Lab - Emergence Project

Building the social learning foundation:
- Student decides when to "show work" to teacher
- Teacher always responds positively (even corrections are encouraging)
- Trust builds through positive interactions
- Approval-seeking calibration develops over time

SECTIONED CURRICULUM:
- Train 1-2 patterns at a time, focused learning
- Section exam proves mastery before moving on
- Final exam on all patterns ensures integration

This is the developmental foundation for later phases.
"""

__version__ = "0.7.0"  # Sectioned curriculum: focused learning, section exams

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

# === SECTIONED CURRICULUM ===
# Learn patterns in focused groups, master each before moving on
CURRICULUM_SECTIONS = [
    {
        'name': 'A: Position Foundations',
        'patterns': ['counting', 'incrementing'],
        'description': 'Pure position awareness and linear progression'
    },
    {
        'name': 'B: Position Math',
        'patterns': ['modular', 'staircase'],
        'description': 'Cycle position (i % n) and quantization (i // n)'
    },
    {
        'name': 'C: Simple Memory',
        'patterns': ['repeating', 'alternating'],
        'description': 'Remember and repeat values'
    },
    {
        'name': 'D: Complex Memory',
        'patterns': ['indexed_lookup', 'periodic_repeat'],
        'description': 'Position-based value retrieval'
    },
    {
        'name': 'E: Growth Patterns',
        'patterns': ['fixed_offset', 'geometric'],
        'description': 'Linear and exponential growth'
    }
]

def get_all_patterns():
    """Get all patterns from curriculum in order."""
    all_patterns = []
    for section in CURRICULUM_SECTIONS:
        all_patterns.extend(section['patterns'])
    return all_patterns


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

        # WEIGHTED TRAINING - graduated topics get maintenance training to prevent forgetting
        # Active topics: full weight (1.0), Graduated topics: reduced weight (0.1)
        # This focuses compute on struggling topics while maintaining knowledge
        tracker = model.learner.self_model.topic_tracker
        graduated_mask = torch.tensor([
            tracker.get_exam_stats(pattern_to_idx[p])['graduated']
            for p in pattern_types
        ], dtype=torch.bool, device=device)
        active_mask = ~graduated_mask

        # Create loss weights: 1.0 for active, 0.1 for graduated (maintenance)
        loss_weights = torch.where(graduated_mask,
                                   torch.tensor(0.1, device=device),
                                   torch.tensor(1.0, device=device))

        optimizer.zero_grad()

        # Forward pass - ALL topics, not just active
        details = model(tokens, seq_lens, targets=targets, return_details=True)

        # Main loss - weighted by active/graduated status
        # Use per-sample loss then apply weights
        logits = details['logits']
        per_sample_loss = F.cross_entropy(logits, targets, reduction='none')
        main_loss = (per_sample_loss * loss_weights).mean()

        # Pattern classification auxiliary - all topics
        pattern_targets = torch.tensor([pattern_to_idx[p] for p in pattern_types], device=device)
        aux_per_sample = F.cross_entropy(details['pattern_logits'], pattern_targets, reduction='none')
        aux_loss = (aux_per_sample * loss_weights).mean()

        # Confidence calibration - all topics
        preds = details['logits'].argmax(dim=-1)
        correct = (preds == targets)
        conf = details['learner_self']['emotions']['confidence'].squeeze()
        conf_per_sample = F.binary_cross_entropy(conf, correct.float(), reduction='none')
        conf_loss = (conf_per_sample * loss_weights).mean()

        # Combined loss
        loss = main_loss + 0.1 * aux_loss + 0.1 * conf_loss
        loss.backward()
        optimizer.step()

        # === APPROVAL-SEEKING BEHAVIOR with RISING BARS ===
        # Student decides whether to show work to teacher
        # As internalization grows, student becomes more selective

        # Get internalization level for rising show bar
        int_level = torch.sigmoid(model.learner.other_model.internalization_level).item()

        # Detect TRUE creativity: novel approach + high confidence
        # "It's only creative if you know WHY it worked"
        # Just doing something different and getting lucky isn't creativity
        # LEVEL-SCALED: Higher level = higher bar for "creative"
        # L0: 0.5 threshold (encourage exploration), L10: 0.9 (only truly novel)
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

        # Student decides: should I show this? (rising bar based on internalization)
        should_show, reasons = model.learner.self_model.should_show_work(
            correct, is_creative, conf, int_level,
            teacher_goal=teacher_goal, pattern_indices=pattern_indices
        )

        # Only show for non-mastered topics - no point showing mastered work
        should_show = should_show & active_mask

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

                # === XP AWARD based on show type and outcome ===
                topic_idx = pattern_to_idx[pattern_types[idx_item]]
                was_correct_item = correct[idx].item()
                topic_level = model.learner.self_model.topic_tracker.get_level(topic_idx)

                # Note: Farming is naturally prevented by level scaling (1/level XP)
                # High level = diminishing returns = move on to harder content

                if reason == 'creative':
                    if was_correct_item:
                        model.learner.self_model.topic_tracker.award_xp(topic_idx, 5)  # Validated insight
                    else:
                        model.learner.self_model.topic_tracker.award_xp(topic_idx, -1)  # Gentle nudge - don't punish exploration
                elif reason == 'streak':
                    # Streak shows now trigger on completion (failure or mastery)
                    # Use the completed streak length stored during show decision
                    completed_streak = model.learner.self_model.last_completed_streak
                    model.learner.self_model.topic_tracker.award_xp(topic_idx, max(1, completed_streak // 5))  # Streak bonus
                elif reason == 'validation':
                    # Asking teacher for help = engaging with learning = +1 XP
                    if was_correct_item:
                        model.learner.self_model.topic_tracker.award_xp(topic_idx, 1)
                # spontaneous: 0 XP (neutral)

                # Internalize - weighted by how impressed teacher was
                # For streak shows on completion (failure), still internalize the streak feedback
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
                        # Mastery cap at 100
                        safe_goal = max(1, min(100, int(goal_action['goal'])))
                        model.teacher.current_goal.fill_(safe_goal)

        # === TEACHER MONITORING (un-shown work) ===
        # Teacher observes ALL work, notices quiet competence
        should_acknowledge, unshown_streak = model.teacher.monitor_unshown_work(
            correct, should_show
        )
        if should_acknowledge:
            # Teacher gives unsolicited acknowledgment
            # "I noticed you've been getting these right without checking with me!"
            model.learner.other_model.internalize_from_quiet_competence(unshown_streak)

        # === SELF-RESTRAINT INTERNALIZATION (active topics only) ===
        # Correct answers where student chose NOT to show → internal confidence
        correct_active = correct[active_mask]
        show_active = should_show[active_mask]
        correct_count = correct_active.sum().item()
        unshown_correct = (correct_active & ~show_active).sum().item()
        model.learner.other_model.internalize_from_self_restraint(
            was_correct_count=correct_count,
            chose_not_to_show_count=unshown_correct
        )

        # Track metrics - all topics now train, but report active progress
        batch_size = tokens.size(0)
        active_count = active_mask.sum().item()
        total_loss += main_loss.item() * batch_size  # All contribute to loss
        total_correct += correct.sum().item()  # Track all correct
        total_samples += batch_size  # Count all samples

        # Update topic tracker - ALL topics to maintain accuracy tracking
        pattern_indices = torch.tensor([pattern_to_idx[p] for p in pattern_types], device=device)
        model.learner.self_model.topic_tracker.update(
            pattern_indices, correct, conf
        )

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
        'goal_calibration_rate': goal_cal_rate,
        'highest_goal_achieved': approval_metrics['highest_goal_achieved']
    }


def run_section_exam(model, section, pattern_to_idx, device, epoch, tracker):
    """
    Run exam for all patterns in a section.
    All patterns must pass for section to pass.
    Returns: (passed, results_list)
    """
    section_passed = True
    results = []

    for pattern_name in section['patterns']:
        idx = pattern_to_idx[pattern_name]
        current_level = tracker.get_level(idx)

        # Section exam: must be L5+ and pass 90% on 24 problems
        if current_level < 5:
            results.append({
                'topic': pattern_name,
                'passed': False,
                'reason': f'Not ready (L{current_level}, need L5+)'
            })
            section_passed = False
            continue

        exam_size = 24
        threshold = 0.90

        exam_data = PatternDataset(
            n_examples=exam_size,
            seed=epoch * 1000 + idx,
            pattern_types=[pattern_name]
        )
        exam_loader = DataLoader(exam_data, batch_size=exam_size, collate_fn=collate_fn)

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

        score = correct_count / exam_size
        passed = score >= threshold

        results.append({
            'topic': pattern_name,
            'score': score,
            'passed': passed,
            'threshold': threshold
        })

        if not passed:
            section_passed = False

    return section_passed, results


def main(args):
    # Setup logging
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.data_dir) / 'runs'
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f'phase1_approval_{run_id}.json'

    print("=" * 70)
    print("Phase 1: SECTIONED CURRICULUM")
    print("  Focus on 1-2 patterns at a time, master before moving on")
    print(f"Run ID: {run_id}")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # All patterns from sectioned curriculum
    pattern_types = get_all_patterns()
    pattern_to_idx = {p: i for i, p in enumerate(pattern_types)}

    print(f"\nCurriculum Sections:")
    for i, section in enumerate(CURRICULUM_SECTIONS):
        print(f"  {section['name']}: {section['patterns']}")
        print(f"    {section['description']}")

    # Data will be regenerated per section - validation covers all patterns
    val_data = PatternDataset(n_examples=args.n_val, seed=123, pattern_types=pattern_types)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=collate_fn)
    print(f"Val data: {len(val_data)} examples (all patterns)")

    # Model - n_topics and n_patterns both derive from curriculum
    # n_topics can grow via emerged topics, n_patterns is curriculum size
    n_patterns = len(pattern_types)
    n_topics = n_patterns  # Start equal, can grow organically

    model = RelationalSystem(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_think_steps=args.n_think_steps,
        n_topics=n_topics,
        n_patterns=n_patterns
    ).to(device)

    print(f"Curriculum: {n_patterns} patterns, {n_topics} topics")

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

    # XP uses pure level scaling (1/level) - no static topic difficulty
    # Higher levels naturally get less XP = move on to harder content

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # === SECTIONED CURRICULUM STATE ===
    current_section_idx = 0
    section_passed = [False] * len(CURRICULUM_SECTIONS)
    all_sections_complete = False

    print("\n" + "=" * 70)
    print("SECTIONED CURRICULUM: Focus on 1-2 patterns at a time")
    print("=" * 70)

    best_acc = 0
    history = []
    tracker = model.learner.self_model.topic_tracker

    for epoch in range(1, args.epochs + 1):
        # === DETERMINE ACTIVE PATTERNS FOR THIS EPOCH ===
        # Current section patterns are primary focus
        # Previous passed sections get maintenance training (reduced weight)
        current_section = CURRICULUM_SECTIONS[current_section_idx]
        active_patterns = current_section['patterns'].copy()

        # Add patterns from passed sections for maintenance
        maintenance_patterns = []
        for i, passed in enumerate(section_passed):
            if passed and i < current_section_idx:
                maintenance_patterns.extend(CURRICULUM_SECTIONS[i]['patterns'])

        # Combined training patterns
        training_patterns = active_patterns + maintenance_patterns

        # Generate training data for current patterns (focused)
        # More examples for active patterns, fewer for maintenance
        n_active = args.n_train // len(training_patterns) * len(active_patterns) if training_patterns else args.n_train
        n_maintenance = args.n_train - n_active if maintenance_patterns else 0

        train_samples = []
        if active_patterns:
            active_data = PatternDataset(n_examples=max(1000, n_active), seed=epoch * 100, pattern_types=active_patterns)
            train_samples.extend([active_data[i] for i in range(len(active_data))])
        if maintenance_patterns and n_maintenance > 0:
            maint_data = PatternDataset(n_examples=max(500, n_maintenance), seed=epoch * 100 + 50, pattern_types=maintenance_patterns)
            train_samples.extend([maint_data[i] for i in range(len(maint_data))])

        # Create combined dataset
        class CombinedDataset(Dataset):
            def __init__(self, samples):
                self.samples = samples
            def __len__(self):
                return len(self.samples)
            def __getitem__(self, idx):
                return self.samples[idx]

        train_data = CombinedDataset(train_samples)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

        print(f"\n{'='*70}")
        print(f"Day {epoch} - Section {current_section_idx+1}/{len(CURRICULUM_SECTIONS)}: {current_section['name']}")
        print(f"  Focus: {active_patterns}")
        if maintenance_patterns:
            print(f"  Maintenance: {maintenance_patterns}")
        print(f"  Training samples: {len(train_data)}")

        # === TRAINING ===
        train_metrics = train_day_with_approval(
            model, train_loader, optimizer, criterion, device, pattern_to_idx
        )

        # Sleep to consolidate
        model.learner.temporal_model.sleep()

        val_metrics = evaluate(model, val_loader, device, pattern_to_idx)

        # Update topic calibration
        topic_calibration = {}
        for pattern_name, idx in pattern_to_idx.items():
            acc, conf, gap = tracker.get_calibration(idx)
            streak, best_streak, mastered = tracker.get_streak_info(idx)
            xp, level, progress, xp_high = tracker.get_xp_info(idx)
            topic_calibration[pattern_name] = {
                'accuracy': acc, 'confidence': conf, 'gap': gap,
                'status': 'guessing' if gap > 0.1 else ('overconfident' if gap < -0.1 else 'calibrated'),
                'streak': streak, 'best_streak': best_streak, 'mastered': mastered,
                'xp': xp, 'level': level, 'level_progress': progress, 'xp_high': xp_high
            }

        # Developmental state
        day = model.learner.temporal_model.current_day.item()
        int_level = torch.sigmoid(model.learner.other_model.internalization_level).item()
        trust = torch.sigmoid(model.learner.other_model.trust).item()

        # XP summary
        total_xp = tracker.get_total_xp()
        avg_level = tracker.get_average_level()

        # Display current section patterns
        print(f"\n  Train: loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.1%}")
        print(f"  Val: acc={val_metrics['accuracy']:.1%}")
        print(f"  Internalization: {int_level:.1%}, Trust: {trust:.1%}")
        print(f"  XP: total={total_xp:.0f}, avg_level={avg_level:.1f}")

        print(f"\n  Section patterns:")
        for pt in active_patterns:
            acc = val_metrics['per_pattern'].get(pt, 0)
            cal = topic_calibration.get(pt, {})
            level = cal.get('level', 0)
            progress = cal.get('level_progress', 0)
            xp = cal.get('xp', 0)
            level_bar = "█" * level + ("░" if progress > 0.5 else "") + "·" * (10 - level - (1 if progress > 0.5 else 0))
            ready = "READY" if level >= 5 else ""
            print(f"    {pt:15s}: {acc:.1%} L{level:2d} {level_bar} ({xp:.0f}xp) {ready}")

        import sys; sys.stdout.flush()

        # === LEVEL-UP EXAMS (within section) ===
        # Still run level exams for progression tracking
        exam_results = []
        for pattern_name in training_patterns:
            idx = pattern_to_idx[pattern_name]
            if tracker.check_exam_eligible(idx):
                current_level = tracker.get_level(idx)
                target_level = current_level + 1
                exam_size = tracker.get_exam_size(target_level)

                exam_data = PatternDataset(
                    n_examples=exam_size,
                    seed=epoch * 1000 + idx,
                    pattern_types=[pattern_name]
                )
                exam_loader = DataLoader(exam_data, batch_size=exam_size, collate_fn=collate_fn)

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

                result = tracker.take_exam(idx, correct_count, exam_size)
                result['topic'] = pattern_name
                exam_results.append(result)

        if exam_results:
            print("  Level exams:")
            for r in exam_results:
                status = f"-> L{r['new_level']}" if r['passed'] else f"(cooldown {r['cooldown']})"
                symbol = "✓" if r['passed'] else "✗"
                print(f"    {r['topic']:15s}: {r['score']:.0%} {symbol} {status}")

        # === SECTION EXAM ===
        # Check if all patterns in current section are ready (L5+)
        section_ready = all(
            tracker.get_level(pattern_to_idx[p]) >= 5
            for p in current_section['patterns']
        )

        if section_ready and not section_passed[current_section_idx]:
            print(f"\n  === SECTION EXAM: {current_section['name']} ===")
            passed, results = run_section_exam(
                model, current_section, pattern_to_idx, device, epoch, tracker
            )

            for r in results:
                if 'score' in r:
                    symbol = "✓" if r['passed'] else "✗"
                    print(f"    {r['topic']:15s}: {r['score']:.0%} {symbol}")
                else:
                    print(f"    {r['topic']:15s}: {r['reason']}")

            if passed:
                section_passed[current_section_idx] = True
                print(f"\n  *** SECTION {current_section_idx+1} PASSED! ***")

                # Move to next section
                if current_section_idx < len(CURRICULUM_SECTIONS) - 1:
                    current_section_idx += 1
                    next_section = CURRICULUM_SECTIONS[current_section_idx]
                    print(f"  Moving to: {next_section['name']}")
                    print(f"    Patterns: {next_section['patterns']}")
                else:
                    print(f"\n  All sections complete! Preparing final exam...")
                    all_sections_complete = True

        # Save history
        history.append({
            'epoch': epoch, 'section': current_section_idx,
            'section_name': current_section['name'],
            'train_acc': train_metrics['accuracy'],
            'train_loss': train_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
            'topic_calibration': topic_calibration.copy(),
            'total_xp': total_xp, 'avg_level': avg_level,
            'internalization': int_level, 'trust': trust
        })

        # Save checkpoint
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': best_acc,
                'n_patterns': n_patterns,
                'n_topics': n_topics,
                'current_section': current_section_idx,
                'section_passed': section_passed
            }, Path(args.data_dir) / 'phase1_approval_best.pt')
            topic_registry.save(registry_path)

        # === FINAL COMPREHENSIVE EXAM ===
        if all_sections_complete:
            print(f"\n  {'='*50}")
            print(f"  === FINAL COMPREHENSIVE EXAM ===")
            print(f"  All sections passed - prove mastery across all patterns!")
            print(f"  {'='*50}")

            final_results = []
            any_failed = False

            for pattern_name, idx in pattern_to_idx.items():
                final_size = 32
                final_threshold = 0.90

                final_data = PatternDataset(
                    n_examples=final_size,
                    seed=epoch * 10000 + idx,
                    pattern_types=[pattern_name]
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
                symbol = "✓" if passed else "✗"
                print(f"    {pattern_name:15s}: {correct_count}/{final_size} ({score:.0%}) {symbol}")

                if not passed:
                    any_failed = True
                    # Find which section this pattern belongs to
                    for sec_idx, sec in enumerate(CURRICULUM_SECTIONS):
                        if pattern_name in sec['patterns']:
                            section_passed[sec_idx] = False
                            if sec_idx < current_section_idx:
                                current_section_idx = sec_idx  # Go back to failed section

                final_results.append({'topic': pattern_name, 'score': score, 'passed': passed})

            history[-1]['final_exam'] = final_results

            if not any_failed:
                print(f"\n{'*'*60}")
                print(f"*** Phase 1 COMPLETE! ALL PATTERNS MASTERED! ***")
                print(f"    Trust: {trust:.1%}, Internalization: {int_level:.1%}")
                print(f"    Total epochs: {epoch}")
                print(f"*** Ready for Phase 2! ***")
                print(f"{'*'*60}")
                break
            else:
                print(f"\n  Final exam failed - back to section {current_section_idx+1}")
                all_sections_complete = False

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
        'version': __version__,
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
    parser.add_argument('--epochs', type=int, default=100)  # Run until complete, break manually if stuck
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-think-steps', type=int, default=5)

    args = parser.parse_args()
    main(args)
