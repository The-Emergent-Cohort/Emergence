"""
Phase 1: Training Loop for Reflex Classification
Coherence Lab - Emergence Project

Trains minimal transformer on trivially learnable patterns.
Establishes frozen-layer mechanism for curriculum learning.
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import argparse
from datetime import datetime

# Import model (will work when this runs locally or remotely)
try:
    from phase1_model import Phase1Transformer
except ImportError:
    # If run from different directory
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from phase1_model import Phase1Transformer


class ReflexDataset(Dataset):
    """Dataset for Phase 1 reflex classification."""

    def __init__(self, data_path, max_seq_len=20):
        self.max_seq_len = max_seq_len
        self.examples = []

        with open(data_path, 'r') as f:
            for line in f:
                self.examples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        # Pad sequence to max length
        seq = ex['sequence']
        padded = seq + [0] * (self.max_seq_len - len(seq))

        return {
            'tokens': torch.tensor(padded[:self.max_seq_len], dtype=torch.long),
            'target_pos': torch.tensor(ex['target_position'], dtype=torch.long),
            'labels': torch.tensor(ex['multihot'], dtype=torch.float),
            'seq_len': len(seq)
        }


def collate_fn(batch):
    """Collate batch of examples."""
    return {
        'tokens': torch.stack([b['tokens'] for b in batch]),
        'target_pos': torch.stack([b['target_pos'] for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch]),
        'seq_len': [b['seq_len'] for b in batch]
    }


def compute_accuracy(logits, labels, threshold=0.5):
    """Compute per-label and overall accuracy."""
    preds = (torch.sigmoid(logits) > threshold).float()

    # Per-label accuracy
    per_label = (preds == labels).float().mean(dim=0)

    # Exact match (all labels correct)
    exact_match = (preds == labels).all(dim=1).float().mean()

    # Hamming accuracy (average correct labels)
    hamming = (preds == labels).float().mean()

    return {
        'per_label': per_label,
        'exact_match': exact_match.item(),
        'hamming': hamming.item()
    }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch in dataloader:
        tokens = batch['tokens'].to(device)
        target_pos = batch['target_pos'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        logits = model(tokens, target_pos)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * tokens.size(0)

        # Compute accuracy
        acc = compute_accuracy(logits.detach(), labels)
        total_correct += acc['exact_match'] * tokens.size(0)
        total_samples += tokens.size(0)

    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples
    }


def evaluate(model, dataloader, criterion, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    total_samples = 0
    all_per_label = []
    all_exact = 0
    all_hamming = 0

    with torch.no_grad():
        for batch in dataloader:
            tokens = batch['tokens'].to(device)
            target_pos = batch['target_pos'].to(device)
            labels = batch['labels'].to(device)

            logits = model(tokens, target_pos)
            loss = criterion(logits, labels)

            total_loss += loss.item() * tokens.size(0)

            acc = compute_accuracy(logits, labels)
            all_per_label.append(acc['per_label'])
            all_exact += acc['exact_match'] * tokens.size(0)
            all_hamming += acc['hamming'] * tokens.size(0)
            total_samples += tokens.size(0)

    avg_per_label = torch.stack(all_per_label).mean(dim=0)

    return {
        'loss': total_loss / total_samples,
        'exact_match': all_exact / total_samples,
        'hamming': all_hamming / total_samples,
        'per_label': avg_per_label.tolist()
    }


def train(args):
    """Main training loop."""
    print("=" * 60)
    print("Phase 1: Reflex Classification Training")
    print("=" * 60)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Data
    data_dir = Path(args.data_dir)
    train_dataset = ReflexDataset(data_dir / 'phase1_train.json')
    val_dataset = ReflexDataset(data_dir / 'phase1_val.json')

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
    model = Phase1Transformer(
        vocab_size=26,
        max_seq_len=20,
        embed_dim=128,
        num_layers=2,
        num_heads=1,
        ffn_dim=256,
        num_classes=6,
        dropout=0.1
    ).to(device)

    print(f"Model parameters: {model.count_parameters():,}")

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_per_label': []
    }

    best_val_acc = 0
    patience_counter = 0

    print("\nStarting training...")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Log
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['exact_match'])
        history['val_per_label'].append(val_metrics['per_label'])

        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Train Acc: {train_metrics['accuracy']:.3%} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val Acc: {val_metrics['exact_match']:.3%}")

        # Check for improvement
        if val_metrics['exact_match'] > best_val_acc:
            best_val_acc = val_metrics['exact_match']
            patience_counter = 0

            # Save best model
            checkpoint_path = data_dir / 'phase1_checkpoint.pt'
            model.save_checkpoint(checkpoint_path, metadata={
                'epoch': epoch,
                'val_accuracy': best_val_acc,
                'timestamp': datetime.now().isoformat()
            })
        else:
            patience_counter += 1

        # Early stopping
        if val_metrics['exact_match'] >= args.target_acc:
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

    # Per-label breakdown
    label_names = [
        "VOWEL_MATCH", "CONSONANT_MATCH",
        "EVEN_VALUE", "ODD_VALUE",
        "HIGH_HALF", "LOW_HALF"
    ]
    print("\nPer-label accuracy (final):")
    for name, acc in zip(label_names, val_metrics['per_label']):
        print(f"  {name:18s}: {acc:.3%}")

    # Test on held-out tokens (21-25)
    if (data_dir / 'phase1_held_out.json').exists():
        print("\nHeld-out token evaluation (tokens 21-25):")
        held_out_dataset = ReflexDataset(data_dir / 'phase1_held_out.json')
        held_out_loader = DataLoader(
            held_out_dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn
        )
        held_out_metrics = evaluate(model, held_out_loader, criterion, device)
        print(f"  Exact match: {held_out_metrics['exact_match']:.3%}")
        print(f"  Hamming: {held_out_metrics['hamming']:.3%}")

    # Save training history
    history_path = data_dir / 'phase1_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")

    return best_val_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Phase 1 reflex classifier')
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
    parser.add_argument('--target-acc', type=float, default=0.98,
                        help='Target validation accuracy to stop training')

    args = parser.parse_args()
    train(args)
