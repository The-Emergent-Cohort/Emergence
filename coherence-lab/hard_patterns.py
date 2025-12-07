"""
Hard Patterns: Tasks that require the relational architecture
Coherence Lab - Emergence Project

These tasks should trigger:
- Teacher intervention (learner gets stuck)
- Frustration detection
- Internalization progression
- Memory consolidation benefits

Pattern types:
1. Compositional: alternating + incrementing combined
2. Long-range: dependencies spanning many positions
3. Context-dependent: same sequence, different answer based on context
4. Ambiguous: multiple valid answers, need confidence calibration
5. Novel inference: figure out new pattern from few examples
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

# Import the relational model
from relational_model import (
    RelationalSystem, PatternDataset, collate_fn,
    train_day, evaluate
)


class HardPatternDataset(Dataset):
    """
    Harder patterns that should require teacher intervention.
    """

    def __init__(self, n_examples=50000, vocab_size=26, seed=None,
                 difficulty='medium'):
        if seed:
            random.seed(seed)

        self.vocab_size = vocab_size
        self.difficulty = difficulty

        # Pattern types by difficulty
        if difficulty == 'easy':
            self.pattern_types = ['alternating', 'repeating', 'incrementing', 'fixed_offset']
        elif difficulty == 'medium':
            self.pattern_types = ['compositional', 'long_range', 'fibonacci_like']
        elif difficulty == 'hard':
            self.pattern_types = ['context_dependent', 'ambiguous', 'meta_pattern']
        else:
            self.pattern_types = ['alternating', 'compositional', 'long_range',
                                  'context_dependent', 'fibonacci_like']

        self.examples = []
        for _ in range(n_examples):
            pattern_type = random.choice(self.pattern_types)
            example = self._generate(pattern_type)
            if example:
                self.examples.append(example)

    def _generate(self, pt):
        if pt == 'alternating':
            return self._gen_alternating()
        elif pt == 'repeating':
            return self._gen_repeating()
        elif pt == 'incrementing':
            return self._gen_incrementing()
        elif pt == 'fixed_offset':
            return self._gen_fixed_offset()
        elif pt == 'compositional':
            return self._gen_compositional()
        elif pt == 'long_range':
            return self._gen_long_range()
        elif pt == 'fibonacci_like':
            return self._gen_fibonacci_like()
        elif pt == 'context_dependent':
            return self._gen_context_dependent()
        elif pt == 'ambiguous':
            return self._gen_ambiguous()
        elif pt == 'meta_pattern':
            return self._gen_meta_pattern()
        return None

    def _gen_alternating(self):
        a, b = random.sample(range(self.vocab_size), 2)
        length = random.randint(4, 8)
        seq = [a if i % 2 == 0 else b for i in range(length)]
        target = a if length % 2 == 0 else b
        return {'sequence': seq, 'target': target, 'pattern_type': 'alternating'}

    def _gen_repeating(self):
        a = random.randint(0, self.vocab_size - 1)
        length = random.randint(3, 7)
        seq = [a] * length
        return {'sequence': seq, 'target': a, 'pattern_type': 'repeating'}

    def _gen_incrementing(self):
        length = random.randint(3, 6)
        max_start = self.vocab_size - length - 1
        start = random.randint(0, max(0, max_start))
        seq = [start + i for i in range(length)]
        target = start + length
        return {'sequence': seq, 'target': target, 'pattern_type': 'incrementing'}

    def _gen_fixed_offset(self):
        length = random.randint(3, 5)
        k = random.randint(1, 3)
        max_start = self.vocab_size - k * length - 1
        start = random.randint(0, max(0, max_start))
        seq = [start + i * k for i in range(length)]
        target = start + length * k
        return {'sequence': seq, 'target': target, 'pattern_type': 'fixed_offset'}

    def _gen_compositional(self):
        """
        Compositional: Alternating pairs that also increment
        Example: [1,2, 3,4, 5,6, ?] → 7 (next odd) or 8 (next in pair)
        """
        start = random.randint(0, self.vocab_size - 10)
        # Pairs: (1,2), (3,4), (5,6), ...
        length = random.randint(4, 8)
        seq = []
        for i in range(length):
            pair_num = i // 2
            in_pair = i % 2
            seq.append(start + pair_num * 2 + in_pair)

        # Next is either continuing the pair or starting new pair
        if length % 2 == 0:
            # Just finished a pair, start new pair
            target = start + (length // 2) * 2
        else:
            # In middle of pair, complete it
            target = seq[-1] + 1

        if target >= self.vocab_size:
            return self._gen_alternating()  # fallback

        return {'sequence': seq, 'target': target, 'pattern_type': 'compositional'}

    def _gen_long_range(self):
        """
        Long range: Answer depends on position far back
        Example: [A, x, x, x, x, ?] → A (copy from position 0)
        """
        anchor = random.randint(0, self.vocab_size - 1)
        length = random.randint(5, 9)

        # Fill with distractors
        seq = [random.randint(0, self.vocab_size - 1) for _ in range(length)]
        seq[0] = anchor  # Anchor at position 0

        # Target is the anchor
        target = anchor

        return {'sequence': seq, 'target': target, 'pattern_type': 'long_range'}

    def _gen_fibonacci_like(self):
        """
        Fibonacci-like: Each element is sum of previous two (mod vocab_size)
        Example: [1, 2, 3, 5, 8, ?] → 13
        """
        a = random.randint(0, 5)
        b = random.randint(1, 5)
        length = random.randint(4, 7)

        seq = [a, b]
        for _ in range(length - 2):
            next_val = (seq[-1] + seq[-2]) % self.vocab_size
            seq.append(next_val)

        target = (seq[-1] + seq[-2]) % self.vocab_size

        return {'sequence': seq, 'target': target, 'pattern_type': 'fibonacci_like'}

    def _gen_context_dependent(self):
        """
        Context dependent: First element determines the rule
        If seq[0] < 10: use incrementing rule
        If seq[0] >= 10: use decrementing rule
        """
        if random.random() < 0.5:
            # Incrementing context
            context = random.randint(0, 9)
            length = random.randint(3, 5)
            max_start = min(9, self.vocab_size - length - 1)
            start = random.randint(0, max(0, max_start))
            seq = [context] + [start + i for i in range(length)]
            target = start + length
        else:
            # Decrementing context
            context = random.randint(10, min(19, self.vocab_size - 1))
            length = random.randint(3, 5)
            start = random.randint(length, min(self.vocab_size - 1, 20))
            seq = [context] + [start - i for i in range(length)]
            target = max(0, start - length)

        if target >= self.vocab_size or target < 0:
            return self._gen_alternating()

        return {'sequence': seq, 'target': target, 'pattern_type': 'context_dependent'}

    def _gen_ambiguous(self):
        """
        Ambiguous: Could be multiple patterns, model should have low confidence
        Example: [2, 4, 6, ?] could be +2 (→8) or *2 (→12 if we had that)
        """
        # Create a sequence that fits multiple interpretations
        start = random.randint(1, 4)
        k = random.randint(1, 2)
        length = random.randint(3, 5)

        seq = [start + i * k for i in range(length)]

        # Target is the "obvious" one, but confidence should be lower
        target = start + length * k
        if target >= self.vocab_size:
            return self._gen_fixed_offset()

        return {'sequence': seq, 'target': target, 'pattern_type': 'ambiguous'}

    def _gen_meta_pattern(self):
        """
        Meta pattern: Pattern of patterns
        Example: [1,1, 2,2,2, 3,3,3,3, ?] → pattern is "repeat N times"
        """
        max_n = min(4, (self.vocab_size - 1))
        n = random.randint(2, max_n)

        seq = []
        for i in range(1, n + 1):
            if i >= self.vocab_size:
                break
            seq.extend([i] * i)

        if len(seq) > 10:
            seq = seq[:10]

        # Next element continues the pattern
        # Count current repetitions to determine target
        if len(seq) > 0:
            last = seq[-1]
            count_last = seq.count(last)
            if count_last < last:
                target = last  # Continue current run
            else:
                target = min(last + 1, self.vocab_size - 1)  # Start new run
        else:
            target = 1

        return {'sequence': seq, 'target': target, 'pattern_type': 'meta_pattern'}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        max_len = 12
        seq = ex['sequence']
        padded = seq + [0] * (max_len - len(seq))
        return {
            'sequence': torch.tensor(padded[:max_len], dtype=torch.long),
            'target': torch.tensor(ex['target'], dtype=torch.long),
            'seq_len': len(seq),
            'pattern_type': ex['pattern_type']
        }


def main(args):
    # Setup logging with timestamp
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.data_dir) / 'runs'
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f'hard_patterns_{args.difficulty}_{run_id}.json'

    print("=" * 70)
    print("Hard Patterns: Testing the Relational Architecture")
    print(f"Run ID: {run_id}")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Difficulty: {args.difficulty}")
    print(f"Fresh start: {args.fresh}")

    # Get pattern types for this difficulty
    dummy = HardPatternDataset(n_examples=1, difficulty=args.difficulty)
    pattern_types = dummy.pattern_types
    pattern_to_idx = {p: i for i, p in enumerate(pattern_types)}

    print(f"Pattern types: {pattern_types}")

    # Handle fresh start - remove existing checkpoint
    checkpoint_path = Path(args.data_dir) / f'hard_patterns_{args.difficulty}.pt'
    if args.fresh and checkpoint_path.exists():
        print(f"Fresh start: removing {checkpoint_path}")
        checkpoint_path.unlink()

    # Data - use run-specific seeds when fresh to avoid easy val sets
    train_seed = 42
    val_seed = int(run_id.replace('_', '')[-6:]) if args.fresh else 123
    print(f"\nGenerating hard pattern data (train_seed={train_seed}, val_seed={val_seed})...")
    train_data = HardPatternDataset(
        n_examples=args.n_train,
        seed=train_seed,
        difficulty=args.difficulty
    )
    val_data = HardPatternDataset(
        n_examples=args.n_val,
        seed=val_seed,
        difficulty=args.difficulty
    )

    # Distribution check
    train_counts = {}
    for ex in train_data.examples:
        pt = ex['pattern_type']
        train_counts[pt] = train_counts.get(pt, 0) + 1
    print(f"Train distribution: {train_counts}")

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size,
        collate_fn=collate_fn
    )

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Model - use the relational architecture
    # Need to adjust pattern classifier for different number of patterns
    model = RelationalSystem(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_think_steps=args.n_think_steps
    ).to(device)

    # Replace pattern classifier head for different pattern count
    n_patterns = len(pattern_types)
    model.learner.pattern_head = nn.Linear(args.d_model, n_patterns).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print("\nTraining on hard patterns...")
    print("-" * 70)

    best_acc = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_day(
            model, train_loader, optimizer, criterion, device, pattern_to_idx, val_loader
        )

        # Sleep
        model.learner.temporal_model.sleep()

        val_metrics = evaluate(model, val_loader, device, pattern_to_idx)

        # Developmental state
        day = model.learner.temporal_model.current_day.item()
        int_level = torch.sigmoid(model.learner.other_model.internalization_level).item()
        trust = torch.sigmoid(model.learner.other_model.trust).item()

        history.append({
            'epoch': epoch,
            'train_acc': train_metrics['accuracy'],
            'train_loss': train_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
            'interventions': train_metrics['interventions'],
            'generalization_gap': train_metrics.get('generalization_gap', 0),
            'internalization': int_level,
            'trust': trust,
            'per_pattern': val_metrics['per_pattern'].copy(),
            'topic_calibration': train_metrics.get('topic_calibration', {})
        })

        print(f"\nDay {day} (Epoch {epoch:2d})")
        print(f"  Train: loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.1%}")
        print(f"  Val: acc={val_metrics['accuracy']:.1%}")
        gap = train_metrics.get('generalization_gap', 0)
        if gap > 0.1:
            print(f"  ⚠ Generalization gap: {gap:.1%} (train >> val)")
        print(f"  Interventions: {train_metrics['interventions']:.1%}")
        print(f"  Internalization: {int_level:.1%}, Trust: {trust:.1%}")
        print("  Per-pattern (val_acc | topic_conf → calibration):")
        topic_cal = train_metrics.get('topic_calibration', {})
        for pt in pattern_types:
            acc = val_metrics['per_pattern'].get(pt, 0)
            status = "✓" if acc >= 0.85 else "←"
            # Show topic calibration if available
            if pt in topic_cal:
                tc = topic_cal[pt]
                cal_status = tc['status']
                cal_symbol = "?" if cal_status == 'guessing' else ("!" if cal_status == 'overconfident' else "=")
                print(f"    {pt:20s}: {acc:.1%} {status} | conf={tc['confidence']:.1%} {cal_symbol}")
            else:
                print(f"    {pt:20s}: {acc:.1%} {status}")

        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': best_acc,
                'difficulty': args.difficulty
            }, Path(args.data_dir) / f'hard_patterns_{args.difficulty}.pt')

        # Check mastery - but require certification exam on fresh problems!
        all_good = all(
            val_metrics['per_pattern'].get(pt, 0) >= 0.85
            for pt in pattern_types
        )
        if all_good and val_metrics['accuracy'] >= 0.90:
            print(f"\n  Threshold met ({val_metrics['accuracy']:.1%}) - running certification exam...")

            # Generate completely fresh problems (new seed based on epoch)
            exam_seed = int(run_id.replace('_', '')[-4:]) + epoch * 1000
            exam_data = HardPatternDataset(
                n_examples=2000,  # Smaller but fresh
                seed=exam_seed,
                difficulty=args.difficulty
            )
            exam_loader = DataLoader(
                exam_data, batch_size=args.batch_size,
                collate_fn=collate_fn
            )

            # Evaluate on fresh exam
            exam_metrics = evaluate(model, exam_loader, device, pattern_to_idx)

            print(f"  📝 Certification Exam (seed={exam_seed}):")
            print(f"     Overall: {exam_metrics['accuracy']:.1%}")
            exam_all_good = True
            for pt in pattern_types:
                exam_acc = exam_metrics['per_pattern'].get(pt, 0)
                status = "✓" if exam_acc >= 0.85 else "✗"
                print(f"     {pt:20s}: {exam_acc:.1%} {status}")
                if exam_acc < 0.85:
                    exam_all_good = False

            # Must pass fresh exam to be declared mastered
            if exam_all_good and exam_metrics['accuracy'] >= 0.90:
                print(f"\n*** CERTIFIED: Hard patterns mastered! ***")
                history[-1]['certification'] = {
                    'passed': True,
                    'exam_seed': exam_seed,
                    'exam_acc': exam_metrics['accuracy'],
                    'exam_per_pattern': exam_metrics['per_pattern']
                }
                break
            else:
                print(f"  ❌ Certification failed - continue training")
                history[-1]['certification'] = {
                    'passed': False,
                    'exam_seed': exam_seed,
                    'exam_acc': exam_metrics['accuracy'],
                    'exam_per_pattern': exam_metrics['per_pattern']
                }

    print("\n" + "=" * 70)
    print(f"Difficulty: {args.difficulty}")
    print(f"Best accuracy: {best_acc:.1%}")
    print(f"Final internalization: {int_level:.1%}")
    print("=" * 70)

    # Save run log
    run_log = {
        'run_id': run_id,
        'difficulty': args.difficulty,
        'fresh': args.fresh,
        'train_seed': train_seed,
        'val_seed': val_seed,
        'best_acc': best_acc,
        'final_internalization': int_level,
        'final_trust': trust,
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
    parser.add_argument('--n-train', type=int, default=80000)
    parser.add_argument('--n-val', type=int, default=8000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-think-steps', type=int, default=5)
    parser.add_argument('--difficulty', type=str, default='medium',
                        choices=['easy', 'medium', 'hard', 'all'])
    parser.add_argument('--fresh', action='store_true',
                        help='Start fresh (ignore any existing checkpoints)')

    args = parser.parse_args()
    main(args)
