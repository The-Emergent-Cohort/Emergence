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

__version__ = "0.8.0"  # Refactored to use CurriculumSequencer

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
from curriculum_sequencer import CurriculumSequencer, create_mixed_dataset

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

    # Validation data covers all patterns
    val_data = PatternDataset(n_examples=args.n_val, seed=123, pattern_types=pattern_types)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=collate_fn)
    print(f"Val data: {len(val_data)} examples (all patterns)")

    # Model
    n_patterns = len(pattern_types)
    n_topics = n_patterns

    model = RelationalSystem(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_think_steps=args.n_think_steps,
        n_topics=n_topics,
        n_patterns=n_patterns
    ).to(device)

    print(f"Curriculum: {n_patterns} patterns, {n_topics} topics")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Topic registry
    registry_path = Path(args.data_dir) / 'topic_registry.json'
    topic_registry = DynamicTopicRegistry(pattern_types)
    model.set_topic_registry(topic_registry)
    print(f"Topic registry: {len(topic_registry)} curriculum topics")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    tracker = model.learner.self_model.topic_tracker

    # === CREATE CURRICULUM SEQUENCER ===
    sequencer = CurriculumSequencer(
        sections=CURRICULUM_SECTIONS,
        topic_to_idx=pattern_to_idx,
        tracker=tracker,
        section_exam_level=5,
        section_exam_size=24,
        section_exam_threshold=0.90,
        final_exam_size=32,
        final_exam_threshold=0.90
    )

    # Track best accuracy for checkpointing
    best_acc = 0

    # === DEFINE TRAINING FUNCTION ===
    def train_fn(mdl, active_topics, maintenance_topics, epoch):
        """Phase 1 training with approval-seeking behavior."""
        # Generate mixed training data
        all_topics = active_topics + maintenance_topics
        n_active = args.n_train // len(all_topics) * len(active_topics) if all_topics else args.n_train
        n_maint = args.n_train - n_active if maintenance_topics else 0

        train_data = create_mixed_dataset(
            PatternDataset, active_topics, maintenance_topics,
            n_active, n_maint, seed=epoch * 100
        )
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

        # Run Phase 1 approval-seeking training
        metrics = train_day_with_approval(mdl, train_loader, optimizer, criterion, device, pattern_to_idx)

        # Sleep to consolidate
        mdl.learner.temporal_model.sleep()

        return metrics

    # === DEFINE EVAL FUNCTION ===
    def eval_fn(mdl):
        """Evaluate on all patterns."""
        return evaluate(mdl, val_loader, device, pattern_to_idx)

    # === DEFINE EXAM FUNCTION ===
    def exam_fn(mdl, topic_name, n_problems, seed, dev):
        """Generate and run exam for a single topic."""
        exam_data = PatternDataset(n_examples=n_problems, seed=seed, pattern_types=[topic_name])
        exam_loader = DataLoader(exam_data, batch_size=n_problems, collate_fn=collate_fn)

        mdl.eval()
        correct = 0
        for batch in exam_loader:
            tokens = batch['tokens'].to(dev)
            targets = batch['target'].to(dev)
            seq_lens = batch['seq_len']
            with torch.no_grad():
                details = mdl(tokens, seq_lens, targets=targets, return_details=True)
                preds = details['logits'].argmax(dim=-1)
                correct += (preds == targets).sum().item()
        mdl.train()

        return correct, n_problems

    # === DEFINE CALLBACKS ===
    def on_epoch_start(epoch, section_info):
        print(f"\n{'='*70}")
        print(f"Day {epoch} - Section {section_info['section_idx']+1}/{len(CURRICULUM_SECTIONS)}: {section_info['section_name']}")
        print(f"  Focus: {section_info['active_topics']}")
        if section_info['maintenance_topics']:
            print(f"  Maintenance: {section_info['maintenance_topics']}")

    def on_epoch_end(epoch, record):
        nonlocal best_acc

        train_metrics = record['train_metrics']
        eval_metrics = record['eval_metrics']

        # Get developmental state
        int_level = torch.sigmoid(model.learner.other_model.internalization_level).item()
        trust = torch.sigmoid(model.learner.other_model.trust).item()
        total_xp = tracker.get_total_xp()
        avg_level = tracker.get_average_level()

        print(f"\n  Train: loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.1%}")
        print(f"  Val: acc={eval_metrics['accuracy']:.1%}")
        print(f"  Internalization: {int_level:.1%}, Trust: {trust:.1%}")
        print(f"  XP: total={total_xp:.0f}, avg_level={avg_level:.1f}")

        # Display section patterns
        active_topics, _ = sequencer.get_active_topics()
        print(f"\n  Section patterns:")
        for pt in active_topics:
            acc = eval_metrics['per_pattern'].get(pt, 0)
            level = tracker.get_level(pattern_to_idx[pt])
            xp, _, progress, _ = tracker.get_xp_info(pattern_to_idx[pt])
            level_bar = "█" * level + ("░" if progress > 0.5 else "") + "·" * (10 - level - (1 if progress > 0.5 else 0))
            ready = "READY" if level >= 5 else ""
            print(f"    {pt:15s}: {acc:.1%} L{level:2d} {level_bar} ({xp:.0f}xp) {ready}")

        # Display level exams
        if record['level_exams']:
            print("  Level exams:")
            for r in record['level_exams']:
                status = f"-> L{r['new_level']}" if r['passed'] else f"(cooldown {r['cooldown']})"
                symbol = "✓" if r['passed'] else "✗"
                print(f"    {r['topic']:15s}: {r['score']:.0%} {symbol} {status}")

        import sys; sys.stdout.flush()

        # Save checkpoint if best
        if eval_metrics['accuracy'] > best_acc:
            best_acc = eval_metrics['accuracy']
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': best_acc,
                'n_patterns': n_patterns,
                'n_topics': n_topics,
                'current_section': sequencer.current_section_idx,
                'section_passed': sequencer.section_passed.copy()
            }, Path(args.data_dir) / 'phase1_approval_best.pt')
            topic_registry.save(registry_path)

    def on_section_exam(section, results, passed):
        print(f"\n  === SECTION EXAM: {section['name']} ===")
        for r in results:
            if 'score' in r:
                symbol = "✓" if r['passed'] else "✗"
                print(f"    {r['topic']:15s}: {r['score']:.0%} {symbol}")
            else:
                print(f"    {r['topic']:15s}: {r['reason']}")

    def on_section_complete(section_idx, next_section):
        print(f"\n  *** SECTION {section_idx+1} PASSED! ***")
        if next_section:
            print(f"  Moving to: {next_section['name']}")
            print(f"    Patterns: {next_section.get('patterns', next_section.get('topics', []))}")
        else:
            print(f"\n  All sections complete! Preparing final exam...")

    def on_final_exam(results, passed):
        print(f"\n  {'='*50}")
        print(f"  === FINAL COMPREHENSIVE EXAM ===")
        print(f"  {'='*50}")
        for r in results:
            symbol = "✓" if r['passed'] else "✗"
            print(f"    {r['topic']:15s}: {r['correct']}/{r['total']} ({r['score']:.0%}) {symbol}")

        if passed:
            int_level = torch.sigmoid(model.learner.other_model.internalization_level).item()
            trust = torch.sigmoid(model.learner.other_model.trust).item()
            print(f"\n{'*'*60}")
            print(f"*** Phase 1 COMPLETE! ALL PATTERNS MASTERED! ***")
            print(f"    Trust: {trust:.1%}, Internalization: {int_level:.1%}")
            print(f"*** Ready for Phase 2! ***")
            print(f"{'*'*60}")
        else:
            print(f"\n  Final exam failed - back to training!")

    # === RUN CURRICULUM ===
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
            'on_final_exam': on_final_exam
        }
    )

    # Final summary
    int_level = torch.sigmoid(model.learner.other_model.internalization_level).item()
    trust = torch.sigmoid(model.learner.other_model.trust).item()

    print("\n" + "=" * 70)
    print(f"Best accuracy: {best_acc:.1%}")
    print(f"Final trust: {trust:.1%}")
    print(f"Final internalization: {int_level:.1%}")
    print(f"Completed: {result['completed']} in {result['epochs']} epochs")

    # Final registry save
    topic_registry.save(registry_path)
    print(f"\nTopic registry saved: {registry_path}")
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
