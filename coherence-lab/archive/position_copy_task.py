"""
Position Copy Task: Side Approach for Position Retrieval
Coherence Lab - Emergence Project

Teaches position-based retrieval directly:
- "Copy the token from position N-K"
- "What was at position 2?"

This is the skill that Phase 2 alternating/repeating patterns require
but our position classification (even/odd) didn't teach.

If the model can't do this, it can't do alternating patterns.
If it CAN do this, we know position retrieval is working.
"""

import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import argparse

try:
    from phase1_integrated_model import IntegratedPhase1Model
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from phase1_integrated_model import IntegratedPhase1Model


class PositionCopyDataGenerator:
    """
    Generate position copy tasks:
    - Given a sequence and a query position, output the token at that position
    - Variations: absolute position, relative position (N-K from end)
    """

    def __init__(self, vocab_size=26, seed=None):
        self.vocab_size = vocab_size
        if seed is not None:
            random.seed(seed)

    def generate_example(self, task_type='absolute'):
        """
        Generate one position copy example.

        task_type:
        - 'absolute': Query specific position (0, 1, 2, ...)
        - 'relative_back': Query N positions back from current
        - 'relative_start': Query N positions from start
        """
        seq_len = random.randint(4, 10)
        sequence = [random.randint(0, self.vocab_size - 1) for _ in range(seq_len)]

        if task_type == 'absolute':
            # Query: "What is at position K?"
            query_pos = random.randint(0, seq_len - 1)
            target = sequence[query_pos]
            # Encode query position in the input
            # We'll append the query position as a special token at the end
            return {
                'sequence': sequence,
                'query_pos': query_pos,
                'target': target,
                'task_type': 'absolute'
            }

        elif task_type == 'relative_back':
            # Query: "What was K positions ago?" (from end of sequence)
            max_back = min(4, seq_len - 1)
            k_back = random.randint(1, max_back)
            query_pos = seq_len - 1 - k_back
            target = sequence[query_pos]
            return {
                'sequence': sequence,
                'k_back': k_back,
                'query_pos': query_pos,
                'target': target,
                'task_type': 'relative_back'
            }

        elif task_type == 'alternating_copy':
            # Specifically for alternating pattern training
            # "If positions alternate A-B-A-B, and I'm at odd position, copy from position 1"
            # Simplified: copy from position with same parity
            query_pos = random.randint(2, seq_len - 1)
            # Copy from same-parity position (2 back)
            source_pos = query_pos - 2
            if source_pos >= 0:
                target = sequence[source_pos]
            else:
                target = sequence[query_pos % 2]  # First position of same parity
            return {
                'sequence': sequence,
                'query_pos': query_pos,
                'source_pos': source_pos if source_pos >= 0 else query_pos % 2,
                'target': target,
                'task_type': 'alternating_copy'
            }

    def generate_dataset(self, n_examples=50000, task_mix=None):
        """Generate a mixed dataset of position copy tasks."""
        if task_mix is None:
            task_mix = {
                'absolute': 0.4,
                'relative_back': 0.3,
                'alternating_copy': 0.3
            }

        examples = []
        for _ in range(n_examples):
            task_type = random.choices(
                list(task_mix.keys()),
                weights=list(task_mix.values())
            )[0]
            examples.append(self.generate_example(task_type))

        # Split train/val
        random.shuffle(examples)
        split = int(0.9 * len(examples))
        return examples[:split], examples[split:]


class PositionCopyDataset(Dataset):
    """Dataset for position copy task."""

    def __init__(self, examples, max_seq_len=12, vocab_size=26):
        self.examples = examples
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        seq = ex['sequence']

        # Pad sequence
        padded = seq + [0] * (self.max_seq_len - len(seq))

        # Encode query position as additional input
        # We'll use the last position to hold the query
        query_pos = ex.get('query_pos', ex.get('source_pos', 0))

        return {
            'tokens': torch.tensor(padded[:self.max_seq_len], dtype=torch.long),
            'query_pos': torch.tensor(query_pos, dtype=torch.long),
            'target': torch.tensor(ex['target'], dtype=torch.long),
            'seq_len': len(seq),
            'task_type': ex['task_type']
        }


class PositionCopyModel(nn.Module):
    """
    Model for position copy task.

    Takes sequence + query position, outputs the token at that position.
    Uses frozen Phase 1 encoder + new copy head.
    """

    def __init__(self, phase1_checkpoint_path=None, vocab_size=26, max_seq_len=12):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Load frozen Phase 1
        if phase1_checkpoint_path and Path(phase1_checkpoint_path).exists():
            self.encoder, _ = IntegratedPhase1Model.load_checkpoint(phase1_checkpoint_path)
            self.d_model = self.encoder.d_model
            print(f"Loaded Phase 1 encoder (d_model={self.d_model})")
        else:
            self.encoder = IntegratedPhase1Model()
            self.d_model = self.encoder.d_model

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Position query embedding
        self.query_embed = nn.Embedding(max_seq_len, self.d_model)

        # Copy mechanism: attend to the queried position
        self.query_proj = nn.Linear(self.d_model, self.d_model)
        self.key_proj = nn.Linear(self.d_model, self.d_model)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, vocab_size)
        )

    def forward(self, tokens, query_pos):
        """
        Args:
            tokens: [batch, seq_len]
            query_pos: [batch] - which position to copy from

        Returns:
            logits: [batch, vocab_size]
        """
        batch_size = tokens.size(0)

        # Encode sequence with frozen Phase 1
        x = self.encoder.token_embedding(tokens)
        x = self.encoder.pos_encoding(x)
        x = self.encoder.dropout(x)
        hidden = self.encoder.transformer(x)  # [batch, seq_len, d_model]

        # Get query embedding
        query = self.query_embed(query_pos)  # [batch, d_model]
        query = self.query_proj(query)  # [batch, d_model]

        # Attention over sequence to find the right position
        keys = self.key_proj(hidden)  # [batch, seq_len, d_model]

        # Compute attention scores
        attn_scores = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1)  # [batch, seq_len]
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [batch, seq_len]

        # Weighted sum of hidden states
        context = torch.bmm(attn_weights.unsqueeze(1), hidden).squeeze(1)  # [batch, d_model]

        # Predict token
        logits = self.output_head(context)  # [batch, vocab_size]

        return logits, attn_weights


def train_position_copy(args):
    print("=" * 65)
    print("Position Copy Task: Side Approach for Position Retrieval")
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)

    # Generate data
    print("\nGenerating position copy data...")
    generator = PositionCopyDataGenerator(seed=42)
    train_examples, val_examples = generator.generate_dataset(n_examples=args.n_examples)

    print(f"Train: {len(train_examples)}, Val: {len(val_examples)}")

    # Task distribution
    task_counts = {}
    for ex in train_examples:
        task_counts[ex['task_type']] = task_counts.get(ex['task_type'], 0) + 1
    print(f"Task mix: {task_counts}")

    train_dataset = PositionCopyDataset(train_examples)
    val_dataset = PositionCopyDataset(val_examples)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Model
    phase1_path = data_dir / 'phase1_integrated_checkpoint.pt'
    model = PositionCopyModel(phase1_checkpoint_path=phase1_path).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"\nParameters: {frozen:,} frozen, {trainable:,} trainable")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    criterion = nn.CrossEntropyLoss()

    best_acc = 0

    print("\nTraining...")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            tokens = batch['tokens'].to(device)
            query_pos = batch['query_pos'].to(device)
            targets = batch['target'].to(device)

            optimizer.zero_grad()
            logits, _ = model(tokens, query_pos)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=-1)
            train_loss += loss.item() * tokens.size(0)
            train_correct += (preds == targets).sum().item()
            train_total += tokens.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        task_correct = {}
        task_total = {}

        with torch.no_grad():
            for batch in val_loader:
                tokens = batch['tokens'].to(device)
                query_pos = batch['query_pos'].to(device)
                targets = batch['target'].to(device)
                task_types = batch['task_type']

                logits, attn = model(tokens, query_pos)
                preds = logits.argmax(dim=-1)

                correct = (preds == targets)
                val_correct += correct.sum().item()
                val_total += tokens.size(0)

                for i, tt in enumerate(task_types):
                    task_total[tt] = task_total.get(tt, 0) + 1
                    if correct[i]:
                        task_correct[tt] = task_correct.get(tt, 0) + 1

        val_acc = val_correct / val_total
        per_task = {tt: task_correct.get(tt, 0) / task_total[tt] for tt in task_total}

        print(f"Epoch {epoch:2d} | Loss: {train_loss:.4f} | Acc: {train_acc:.1%}/{val_acc:.1%}")
        print(f"  Per-task: " + ", ".join(f"{k}: {v:.1%}" for k, v in sorted(per_task.items())))

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'per_task': per_task
            }, data_dir / 'position_copy_checkpoint.pt')
            print(f"  Checkpoint saved!")

        # Check if we've mastered the skill
        if val_acc >= 0.95:
            print(f"\nPosition copy skill mastered! ({val_acc:.1%})")
            break

    print("\n" + "=" * 65)
    print(f"Best accuracy: {best_acc:.1%}")

    if best_acc >= 0.90:
        print("\nPosition retrieval is WORKING.")
        print("Model can copy from specific positions.")
        print("→ Safe to retry Phase 2 alternating/repeating patterns.")
    else:
        print("\nPosition retrieval still STRUGGLING.")
        print("→ Need more foundational work before Phase 2 patterns.")

    return best_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--n-examples', type=int, default=50000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=15)

    args = parser.parse_args()
    train_position_copy(args)
