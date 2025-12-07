"""
Phase 1b: Shape/Structure Invariants Training
Coherence Lab - Emergence Project

Teaches categorical matching using frozen Phase 1 embeddings.
GPU-optimized.
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import argparse
from datetime import datetime

try:
    from phase1_model import Phase1Transformer
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from phase1_model import Phase1Transformer


class PairDataset(Dataset):
    """Dataset for same/different classification."""

    def __init__(self, data_path):
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
        return {
            'tokens': torch.tensor(ex['tokens'], dtype=torch.long),
            'label': torch.tensor(ex['label'], dtype=torch.float),
        }


def collate_fn(batch):
    return {
        'tokens': torch.stack([b['tokens'] for b in batch]),
        'label': torch.stack([b['label'] for b in batch]),
    }


class Phase1bModel(nn.Module):
    """Same/different classifier using frozen Phase 1 embeddings."""

    def __init__(self, phase1_checkpoint_path=None, embed_dim=128):
        super().__init__()

        # Load Phase 1 for embeddings
        if phase1_checkpoint_path and Path(phase1_checkpoint_path).exists():
            self.phase1, _ = Phase1Transformer.load_checkpoint(phase1_checkpoint_path)
            print("Loaded Phase 1 checkpoint")
        else:
            self.phase1 = Phase1Transformer()
            print("Using fresh Phase 1 (no checkpoint)")

        # Freeze Phase 1
        for param in self.phase1.parameters():
            param.requires_grad = False

        self.embed_dim = embed_dim

        # Classification head: compare two embeddings
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),  # Concat two embeddings
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Binary output
        )

    def forward(self, tokens):
        """
        Args:
            tokens: [batch, 2] - pairs of token IDs
        Returns:
            logits: [batch, 1] - same/different prediction
        """
        # Get embeddings for both tokens (frozen)
        emb1 = self.phase1.token_embedding(tokens[:, 0])  # [batch, embed_dim]
        emb2 = self.phase1.token_embedding(tokens[:, 1])  # [batch, embed_dim]

        # Concatenate and classify
        combined = torch.cat([emb1, emb2], dim=1)  # [batch, embed_dim * 2]
        logits = self.classifier(combined)

        return logits.squeeze(-1)

    def count_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {'trainable': trainable, 'frozen': frozen}


def train(args):
    print("=" * 60)
    print("Phase 1b: Shape/Structure Invariants Training")
    print("=" * 60)

    # Device selection - prefer GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        print("Device: cpu (GPU not available)")

    # Data
    data_dir = Path(args.data_dir)
    train_dataset = PairDataset(data_dir / 'phase1b_train.json')
    val_dataset = PairDataset(data_dir / 'phase1b_val.json')

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        collate_fn=collate_fn, num_workers=0, pin_memory=torch.cuda.is_available()
    )

    print(f"Train examples: {len(train_dataset)}")
    print(f"Val examples: {len(val_dataset)}")

    # Model
    phase1_path = data_dir / 'phase1_checkpoint.pt'
    model = Phase1bModel(phase1_checkpoint_path=phase1_path).to(device)

    params = model.count_parameters()
    print(f"Frozen params: {params['frozen']:,}")
    print(f"Trainable params: {params['trainable']:,}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print("\nStarting training...")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0

        for batch in train_loader:
            tokens = batch['tokens'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            logits = model(tokens)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            preds = (torch.sigmoid(logits) > 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_loss += loss.item() * tokens.size(0)
            total_samples += tokens.size(0)

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        # Validate
        model.eval()
        val_loss, val_correct, val_samples = 0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                tokens = batch['tokens'].to(device)
                labels = batch['label'].to(device)

                logits = model(tokens)
                loss = criterion(logits, labels)

                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_loss += loss.item() * tokens.size(0)
                val_samples += tokens.size(0)

        val_loss = val_loss / val_samples
        val_acc = val_correct / val_samples

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3%} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'state_dict': model.state_dict(),
                'val_accuracy': best_val_acc,
                'epoch': epoch,
                'timestamp': datetime.now().isoformat()
            }, data_dir / 'phase1b_checkpoint.pt')
            print(f"Saved checkpoint")
        else:
            patience_counter += 1

        if val_acc >= args.target_acc:
            print(f"\nTarget accuracy {args.target_acc:.1%} reached!")
            break

        if patience_counter >= args.patience:
            print(f"\nEarly stopping after {args.patience} epochs without improvement")
            break

    print("\n" + "=" * 60)
    print("Training Complete")
    print(f"Best validation accuracy: {best_val_acc:.3%}")

    # Save history
    with open(data_dir / 'phase1b_history.json', 'w') as f:
        json.dump(history, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--target-acc', type=float, default=0.95)

    args = parser.parse_args()
    train(args)
