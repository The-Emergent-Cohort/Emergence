"""
Phase 2: Training Loop for Sequence Prediction
Coherence Lab - Emergence Project

Trains Phase 2 model (frozen Phase 1 + new layers) on next-token prediction.
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import argparse
from datetime import datetime

try:
    from phase2_model import Phase2Transformer
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from phase2_model import Phase2Transformer


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
        # Pad sequence to max length
        padded = seq + [0] * (self.max_seq_len - len(seq))

        return {
            'tokens': torch.tensor(padded[:self.max_seq_len], dtype=torch.long),
            'target': torch.tensor(ex['target'], dtype=torch.long),
            'seq_len': len(seq),
            'pattern_type': ex['pattern_type']
        }


def collate_fn(batch):
    """Collate batch of examples."""
    return {
        'tokens': torch.stack([b['tokens'] for b in batch]),
        'target': torch.stack([b['target'] for b in batch]),
        'seq_len': [b['seq_len'] for b in batch],
        'pattern_type': [b['pattern_type'] for b in batch]
    }


def compute_accuracy(logits, targets):
    """Compute prediction accuracy."""
    preds = logits.argmax(dim=-1)
    correct = (preds == targets).float()
    return correct.mean().item()


def compute_per_pattern_accuracy(logits, targets, pattern_types):
    """Compute accuracy broken down by pattern type."""
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
    """Train for one epoch."""
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

        total_loss += loss.item() * tokens.size(0)
        total_correct += compute_accuracy(logits.detach(), targets) * tokens.size(0)
        total_samples += tokens.size(0)

    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples
    }


def evaluate(model, dataloader, criterion, device):
    """Evaluate on validation set."""
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

            total_loss += loss.item() * tokens.size(0)
            total_correct += compute_accuracy(logits, targets) * tokens.size(0)
            total_samples += tokens.size(0)

            all_pattern_types.extend(batch['pattern_type'])
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

    # Compute per-pattern accuracy
    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    per_pattern = compute_per_pattern_accuracy(all_logits, all_targets, all_pattern_types)

    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples,
        'per_pattern': per_pattern
    }


def train(args):
    """Main training loop."""
    print("=" * 60)
    print("Phase 2: Sequence Prediction Training")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Data
    data_dir = Path(args.data_dir)

    # Check if data exists, generate if not
    if not (data_dir / 'phase2_train.json').exists():
        print("\nGenerating Phase 2 data...")
        from phase2_data import Phase2DataGenerator
        generator = Phase2DataGenerator(seed=42)
        train_data, val_data = generator.generate_dataset(n_examples=50000)
        held_out_data = generator.generate_held_out(n_examples=1000)

        with open(data_dir / 'phase2_train.json', 'w') as f:
            json.dump(train_data, f)
        with open(data_dir / 'phase2_val.json', 'w') as f:
            json.dump(val_data, f)
        with open(data_dir / 'phase2_held_out.json', 'w') as f:
            json.dump(held_out_data, f)
        print("Data generated.")

    train_dataset = SequenceDataset(data_dir / 'phase2_train.json')
    val_dataset = SequenceDataset(data_dir / 'phase2_val.json')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn
    )

    print(f"Train examples: {len(train_dataset)}")
    print(f"Val examples: {len(val_dataset)}")

    # Model
    phase1_path = data_dir / 'phase1_checkpoint.pt'
    model = Phase2Transformer(
        phase1_checkpoint_path=phase1_path,
        num_new_layers=4,
        num_heads=2
    ).to(device)

    params = model.count_parameters()
    print(f"\nParameter counts:")
    print(f"  Frozen (Phase 1): {params['frozen']:,}")
    print(f"  Trainable (Phase 2): {params['trainable']:,}")

    # Training setup
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    criterion = nn.CrossEntropyLoss()

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_per_pattern': []
    }

    best_val_acc = 0
    patience_counter = 0

    print("\nStarting training...")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_per_pattern'].append(val_metrics['per_pattern'])

        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Train Acc: {train_metrics['accuracy']:.3%} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.3%}")

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0

            checkpoint_path = data_dir / 'phase2_checkpoint.pt'
            model.save_checkpoint(checkpoint_path, metadata={
                'epoch': epoch,
                'val_accuracy': best_val_acc,
                'timestamp': datetime.now().isoformat()
            })
        else:
            patience_counter += 1

        if val_metrics['accuracy'] >= args.target_acc:
            print(f"\nTarget accuracy {args.target_acc:.1%} reached!")
            break

        if patience_counter >= args.patience:
            print(f"\nEarly stopping after {args.patience} epochs without improvement")
            break

    # Final evaluation
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Best validation accuracy: {best_val_acc:.3%}")

    print("\nPer-pattern accuracy (final):")
    for pt, acc in sorted(val_metrics['per_pattern'].items()):
        print(f"  {pt:15s}: {acc:.3%}")

    # Test on held-out (longer sequences)
    if (data_dir / 'phase2_held_out.json').exists():
        print("\nHeld-out evaluation (longer sequences 11-15):")
        held_out_dataset = SequenceDataset(data_dir / 'phase2_held_out.json')
        held_out_loader = DataLoader(
            held_out_dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn
        )
        held_out_metrics = evaluate(model, held_out_loader, criterion, device)
        print(f"  Overall accuracy: {held_out_metrics['accuracy']:.3%}")
        for pt, acc in sorted(held_out_metrics['per_pattern'].items()):
            print(f"  {pt:15s}: {acc:.3%}")

    # Save history
    history_path = data_dir / 'phase2_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")

    return best_val_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Phase 2 sequence predictor')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing training data')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')
    parser.add_argument('--target-acc', type=float, default=0.95,
                        help='Target validation accuracy to stop training')

    args = parser.parse_args()
    train(args)
