"""
Phase 2 v3: Training with Unit Probes and Remediation Detection
Coherence Lab - Emergence Project

Implements the human-education-inspired testing hierarchy:
- Unit probes every N batches (detect dips early)
- Track per-pattern trajectories
- Flag skills that plateau or regress
- Support for remediation branching
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
import argparse
from datetime import datetime
from collections import defaultdict

try:
    from phase2_integrated_model import Phase2IntegratedModel
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from phase2_integrated_model import Phase2IntegratedModel


class SequenceDataset(Dataset):
    """Dataset for Phase 2 sequence prediction."""

    def __init__(self, data_path, max_seq_len=20):
        self.max_seq_len = max_seq_len

        with open(data_path, 'r') as f:
            content = f.read().strip()
            if content.startswith('['):
                self.examples = json.loads(content)
            else:
                self.examples = [json.loads(line) for line in content.split('\n') if line.strip()]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        seq = ex['sequence']
        padded = seq + [0] * (self.max_seq_len - len(seq))

        return {
            'tokens': torch.tensor(padded[:self.max_seq_len], dtype=torch.long),
            'target': torch.tensor(ex['target'], dtype=torch.long),
            'seq_len': len(seq),
            'pattern_type': ex['pattern_type']
        }


def collate_fn(batch):
    return {
        'tokens': torch.stack([b['tokens'] for b in batch]),
        'target': torch.stack([b['target'] for b in batch]),
        'seq_len': [b['seq_len'] for b in batch],
        'pattern_type': [b['pattern_type'] for b in batch]
    }


class SkillTracker:
    """
    Tracks per-skill performance over time.
    Detects dips, plateaus, and flags skills needing remediation.
    """

    def __init__(self, skill_names, window_size=5, dip_threshold=0.05, plateau_threshold=0.02):
        self.skill_names = skill_names
        self.window_size = window_size
        self.dip_threshold = dip_threshold  # Flag if accuracy drops by this much
        self.plateau_threshold = plateau_threshold  # Flag if improvement < this for window_size probes

        # History: skill -> list of (probe_num, accuracy)
        self.history = {skill: [] for skill in skill_names}
        self.flags = {skill: [] for skill in skill_names}  # List of (probe_num, flag_type, details)

    def record(self, probe_num, skill_accuracies):
        """Record a unit probe result."""
        for skill, acc in skill_accuracies.items():
            if skill in self.history:
                self.history[skill].append((probe_num, acc))
                self._check_for_issues(skill, probe_num)

    def _check_for_issues(self, skill, probe_num):
        """Check for dips or plateaus in a skill."""
        hist = self.history[skill]
        if len(hist) < 2:
            return

        # Check for dip (current vs previous)
        prev_acc = hist[-2][1]
        curr_acc = hist[-1][1]

        if prev_acc - curr_acc > self.dip_threshold:
            self.flags[skill].append((probe_num, 'DIP', {
                'from': prev_acc,
                'to': curr_acc,
                'drop': prev_acc - curr_acc
            }))

        # Check for plateau (no improvement over window)
        if len(hist) >= self.window_size:
            window = hist[-self.window_size:]
            max_acc = max(a for _, a in window)
            min_acc = min(a for _, a in window)
            improvement = max_acc - min_acc

            if improvement < self.plateau_threshold:
                # Only flag once per plateau
                recent_flags = [f for f in self.flags[skill] if f[0] > probe_num - self.window_size]
                if not any(f[1] == 'PLATEAU' for f in recent_flags):
                    self.flags[skill].append((probe_num, 'PLATEAU', {
                        'window_start': window[0][0],
                        'avg_acc': sum(a for _, a in window) / len(window),
                        'spread': improvement
                    }))

    def get_struggling_skills(self, min_accuracy=0.7):
        """Return skills currently below threshold or recently flagged."""
        struggling = []
        for skill in self.skill_names:
            if self.history[skill]:
                current_acc = self.history[skill][-1][1]
                recent_flags = [f for f in self.flags[skill] if f[0] > len(self.history[skill]) - 5]

                if current_acc < min_accuracy or recent_flags:
                    struggling.append({
                        'skill': skill,
                        'current_acc': current_acc,
                        'recent_flags': recent_flags
                    })
        return struggling

    def get_report(self):
        """Generate a summary report."""
        report = []
        for skill in self.skill_names:
            if self.history[skill]:
                accs = [a for _, a in self.history[skill]]
                report.append({
                    'skill': skill,
                    'probes': len(accs),
                    'current': accs[-1] if accs else 0,
                    'best': max(accs) if accs else 0,
                    'worst': min(accs) if accs else 0,
                    'trend': accs[-1] - accs[0] if len(accs) > 1 else 0,
                    'flags': len(self.flags[skill])
                })
        return report


def run_unit_probe(model, probe_loader, device):
    """Run a quick unit probe on a subset of validation data."""
    model.eval()
    pattern_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    with torch.no_grad():
        for batch in probe_loader:
            tokens = batch['tokens'].to(device)
            targets = batch['target'].to(device)
            pattern_types = batch['pattern_type']

            logits = model(tokens)
            preds = logits.argmax(dim=-1)
            correct = (preds == targets)

            for i, pt in enumerate(pattern_types):
                pattern_stats[pt]['total'] += 1
                if correct[i]:
                    pattern_stats[pt]['correct'] += 1

    return {pt: stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            for pt, stats in pattern_stats.items()}


def train_with_probes(args):
    print("=" * 70)
    print("Phase 2 v3: Training with Unit Probes and Remediation Detection")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)

    # Generate data if needed
    if not (data_dir / 'phase2_train.json').exists():
        print("\nGenerating Phase 2 data...")
        from phase2_data import Phase2DataGenerator
        generator = Phase2DataGenerator(seed=42)
        train_data, val_data = generator.generate_dataset(n_examples=150000)
        held_out_data = generator.generate_held_out(n_examples=2000)

        with open(data_dir / 'phase2_train.json', 'w') as f:
            json.dump(train_data, f)
        with open(data_dir / 'phase2_val.json', 'w') as f:
            json.dump(val_data, f)
        with open(data_dir / 'phase2_held_out.json', 'w') as f:
            json.dump(held_out_data, f)

    train_dataset = SequenceDataset(data_dir / 'phase2_train.json')
    val_dataset = SequenceDataset(data_dir / 'phase2_val.json')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Create probe loader (subset of validation for quick checks)
    probe_indices = list(range(0, len(val_dataset), max(1, len(val_dataset) // 1000)))
    probe_subset = Subset(val_dataset, probe_indices)
    probe_loader = DataLoader(
        probe_subset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=0
    )

    print(f"Train examples: {len(train_dataset)}")
    print(f"Val examples: {len(val_dataset)}")
    print(f"Probe subset: {len(probe_subset)} examples")

    # Model
    phase1_path = data_dir / 'phase1_integrated_checkpoint.pt'
    model = Phase2IntegratedModel(
        phase1_checkpoint_path=phase1_path,
        num_new_layers=4,
        num_heads=4
    ).to(device)

    params = model.count_parameters()
    print(f"\nParameters: {params['frozen']:,} frozen, {params['trainable']:,} trainable")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    criterion = nn.CrossEntropyLoss()

    # Skill tracker
    skill_names = ['alternating', 'incrementing', 'repeating', 'fixed_offset']
    tracker = SkillTracker(skill_names, window_size=5, dip_threshold=0.05)

    # Training state
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_per_pattern': [],
        'unit_probes': []  # List of (batch_num, per_pattern_acc)
    }

    best_val_acc = 0
    patience_counter = 0
    probe_num = 0
    batch_num = 0

    print(f"\nStarting training with unit probes every {args.probe_interval} batches...")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_samples = 0

        for batch in train_loader:
            batch_num += 1
            tokens = batch['tokens'].to(device)
            targets = batch['target'].to(device)

            optimizer.zero_grad()
            logits = model(tokens)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=-1)
            epoch_loss += loss.item() * tokens.size(0)
            epoch_correct += (preds == targets).float().sum().item()
            epoch_samples += tokens.size(0)

            # Unit probe check
            if batch_num % args.probe_interval == 0:
                probe_num += 1
                probe_results = run_unit_probe(model, probe_loader, device)
                tracker.record(probe_num, probe_results)
                history['unit_probes'].append({
                    'batch': batch_num,
                    'probe_num': probe_num,
                    'results': probe_results
                })

                # Check for struggling skills
                struggling = tracker.get_struggling_skills(min_accuracy=args.remediation_threshold)
                if struggling:
                    print(f"\n  [Probe {probe_num} @ batch {batch_num}] ATTENTION NEEDED:")
                    for s in struggling:
                        flags_str = ', '.join(f[1] for f in s['recent_flags']) if s['recent_flags'] else 'low acc'
                        print(f"    {s['skill']}: {s['current_acc']:.1%} ({flags_str})")

                model.train()  # Back to training mode

        # End of epoch
        train_loss = epoch_loss / epoch_samples
        train_acc = epoch_correct / epoch_samples
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # Full validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_samples = 0
        all_preds = []
        all_targets = []
        all_patterns = []

        with torch.no_grad():
            for batch in val_loader:
                tokens = batch['tokens'].to(device)
                targets = batch['target'].to(device)

                logits = model(tokens)
                loss = criterion(logits, targets)

                preds = logits.argmax(dim=-1)
                val_loss += loss.item() * tokens.size(0)
                val_correct += (preds == targets).float().sum().item()
                val_samples += tokens.size(0)

                all_preds.extend(preds.cpu().tolist())
                all_targets.extend(targets.cpu().tolist())
                all_patterns.extend(batch['pattern_type'])

        val_loss /= val_samples
        val_acc = val_correct / val_samples
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Per-pattern accuracy
        pattern_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        for pred, target, pt in zip(all_preds, all_targets, all_patterns):
            pattern_stats[pt]['total'] += 1
            if pred == target:
                pattern_stats[pt]['correct'] += 1

        per_pattern = {pt: s['correct']/s['total'] for pt, s in pattern_stats.items()}
        history['val_per_pattern'].append(per_pattern)

        print(f"\nEpoch {epoch:2d} | Loss: {train_loss:.4f}/{val_loss:.4f} | Acc: {train_acc:.1%}/{val_acc:.1%}")
        print("  Per-pattern (chapter test):")
        for pt in sorted(per_pattern.keys()):
            acc = per_pattern[pt]
            status = "✓" if acc >= args.remediation_threshold else "← needs work"
            print(f"    {pt:15s}: {acc:.1%} {status}")

        # Check best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'val_accuracy': best_val_acc,
                'per_pattern': per_pattern,
                'tracker_report': tracker.get_report()
            }, data_dir / 'phase2_v3_checkpoint.pt')
            print(f"  Checkpoint saved: {best_val_acc:.1%}")
        else:
            patience_counter += 1

        # Phase transition gate check
        all_pass = all(acc >= args.gate_threshold for acc in per_pattern.values())
        if all_pass:
            print(f"\n*** PHASE GATE PASSED: All skills >= {args.gate_threshold:.0%} ***")
            print("Ready to advance to Phase 3!")
            break

        if patience_counter >= args.patience:
            print(f"\nEarly stopping after {args.patience} epochs without improvement")
            break

    # Final report
    print("\n" + "=" * 70)
    print("Training Complete - Skill Tracker Report")
    print("=" * 70)

    report = tracker.get_report()
    for skill in report:
        print(f"\n{skill['skill']}:")
        print(f"  Probes: {skill['probes']}, Current: {skill['current']:.1%}, Best: {skill['best']:.1%}")
        print(f"  Trend: {skill['trend']:+.1%}, Flags: {skill['flags']}")

    # Recommendations
    struggling = tracker.get_struggling_skills(min_accuracy=args.gate_threshold)
    if struggling:
        print("\n" + "-" * 70)
        print("REMEDIATION RECOMMENDATIONS:")
        for s in struggling:
            print(f"\n  {s['skill']} ({s['current_acc']:.1%}):")
            if s['recent_flags']:
                for flag in s['recent_flags']:
                    if flag[1] == 'DIP':
                        print(f"    - DIP detected: {flag[2]['from']:.1%} → {flag[2]['to']:.1%}")
                    elif flag[1] == 'PLATEAU':
                        print(f"    - PLATEAU at ~{flag[2]['avg_acc']:.1%}")
            print(f"    → Consider: position_copy remediation task")
    else:
        print("\nAll skills at acceptable levels!")

    # Save full history
    with open(data_dir / 'phase2_v3_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    return best_val_acc, tracker


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--probe-interval', type=int, default=100,
                        help='Run unit probe every N batches')
    parser.add_argument('--remediation-threshold', type=float, default=0.70,
                        help='Flag skills below this accuracy')
    parser.add_argument('--gate-threshold', type=float, default=0.85,
                        help='All skills must reach this to pass phase gate')

    args = parser.parse_args()
    train_with_probes(args)
