"""
Phase 2 v2: Training with Integrated Phase 1 Foundation
Coherence Lab - Emergence Project

Same training as phase2_train.py but uses the integrated Phase 1 checkpoint
which includes position awareness training.
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import argparse
from datetime import datetime

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


def compute_per_pattern_accuracy(logits, targets, pattern_types):
    preds = logits.argmax(dim=-1)
    correct = (preds == targets)

    pattern_stats = {}
    for i, pt in enumerate(pattern_types):
        if pt not in pattern_stats:
            pattern_stats[pt] = {'correct': 0, 'total': 0}
        pattern_stats[pt]['total'] += 1
        if correct[i]:
            pattern_stats[pt]['correct'] += 1

    return {pt: stats['correct'] / stats['total'] for pt, stats in pattern_stats.items()}


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch in dataloader:
        tokens = batch['tokens'].to(device)
        targets = batch['target'].to(device)

        optimizer.zero_grad()
        logits = model(tokens)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=-1)
        total_loss += loss.item() * tokens.size(0)
        total_correct += (preds == targets).float().sum().item()
        total_samples += tokens.size(0)

    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples
    }


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_pattern_types = []
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            tokens = batch['tokens'].to(device)
            targets = batch['target'].to(device)

            logits = model(tokens)
            loss = criterion(logits, targets)

            preds = logits.argmax(dim=-1)
            total_loss += loss.item() * tokens.size(0)
            total_correct += (preds == targets).float().sum().item()
            total_samples += tokens.size(0)

            all_pattern_types.extend(batch['pattern_type'])
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    per_pattern = compute_per_pattern_accuracy(all_logits, all_targets, all_pattern_types)

    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples,
        'per_pattern': per_pattern
    }


def train(args):
    print("=" * 65)
    print("Phase 2 v2: Sequence Prediction (Integrated Phase 1 Foundation)")
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)

    # Generate data if needed
    if not (data_dir / 'phase2_train.json').exists():
        print("\nGenerating Phase 2 data (150K examples)...")
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
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=0
    )

    print(f"Train examples: {len(train_dataset)}")
    print(f"Val examples: {len(val_dataset)}")

    # Model - using integrated Phase 1
    phase1_path = data_dir / 'phase1_integrated_checkpoint.pt'
    model = Phase2IntegratedModel(
        phase1_checkpoint_path=phase1_path,
        num_new_layers=4,
        num_heads=4
    ).to(device)

    params = model.count_parameters()
    print(f"\nParameters:")
    print(f"  Frozen (Phase 1): {params['frozen']:,}")
    print(f"  Trainable (Phase 2): {params['trainable']:,}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    criterion = nn.CrossEntropyLoss()

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_per_pattern': []
    }

    best_val_acc = 0
    patience_counter = 0

    print("\nStarting training...")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_per_pattern'].append(val_metrics['per_pattern'])

        print(f"Epoch {epoch:3d} | "
              f"Loss: {train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} | "
              f"Acc: {train_metrics['accuracy']:.1%}/{val_metrics['accuracy']:.1%}")

        # Show per-pattern every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            print("  Per-pattern val acc:")
            for pt, acc in sorted(val_metrics['per_pattern'].items()):
                print(f"    {pt:15s}: {acc:.1%}")

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0

            checkpoint_path = data_dir / 'phase2_integrated_checkpoint.pt'
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'val_accuracy': best_val_acc,
                'per_pattern': val_metrics['per_pattern']
            }, checkpoint_path)
            print(f"  Checkpoint saved: {best_val_acc:.1%}")
        else:
            patience_counter += 1

        if val_metrics['accuracy'] >= args.target_acc:
            print(f"\nTarget {args.target_acc:.0%} reached!")
            break

        if patience_counter >= args.patience:
            print(f"\nEarly stopping (no improvement for {args.patience} epochs)")
            break

    # Final results
    print("\n" + "=" * 65)
    print("Training Complete")
    print("=" * 65)
    print(f"Best validation accuracy: {best_val_acc:.1%}")

    print("\nFinal per-pattern accuracy:")
    for pt, acc in sorted(val_metrics['per_pattern'].items()):
        print(f"  {pt:15s}: {acc:.1%}")

    # Held-out test
    if (data_dir / 'phase2_held_out.json').exists():
        print("\nHeld-out (longer sequences 11-15):")
        held_out = SequenceDataset(data_dir / 'phase2_held_out.json')
        held_out_loader = DataLoader(held_out, batch_size=args.batch_size, collate_fn=collate_fn)
        held_metrics = evaluate(model, held_out_loader, criterion, device)
        print(f"  Overall: {held_metrics['accuracy']:.1%}")
        for pt, acc in sorted(held_metrics['per_pattern'].items()):
            print(f"    {pt:15s}: {acc:.1%}")

    # Save history
    with open(data_dir / 'phase2_integrated_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    return best_val_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--target-acc', type=float, default=0.95)

    args = parser.parse_args()
    train(args)
