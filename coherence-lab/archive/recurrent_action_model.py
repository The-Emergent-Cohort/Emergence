"""
Recurrent Action Model: Testing Recursion + Explicit Actions
Coherence Lab - Emergence Project

Minimal test of the hypothesis:
- Explicit LOOK action (attend to specific position)
- Recurrent think steps (not single-pass)
- Self-evaluation loop

If this works on position retrieval, we know the direction is right.
If it fails, we learn something about what's actually broken.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
from pathlib import Path
import argparse


class ActionDataset(Dataset):
    """
    Simple position retrieval task with explicit action framing.

    Task: Given sequence and target position, retrieve the token.
    Framed as: "LOOK at position K, what do you see?"
    """

    def __init__(self, n_examples=50000, seq_len_range=(4, 10), vocab_size=26, seed=None):
        if seed:
            random.seed(seed)

        self.examples = []
        for _ in range(n_examples):
            seq_len = random.randint(*seq_len_range)
            sequence = [random.randint(0, vocab_size - 1) for _ in range(seq_len)]
            target_pos = random.randint(0, seq_len - 1)
            target_token = sequence[target_pos]

            self.examples.append({
                'sequence': sequence,
                'target_pos': target_pos,
                'target_token': target_token,
                'seq_len': seq_len
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        # Pad sequence
        max_len = 12
        padded = ex['sequence'] + [0] * (max_len - len(ex['sequence']))

        return {
            'sequence': torch.tensor(padded[:max_len], dtype=torch.long),
            'target_pos': torch.tensor(ex['target_pos'], dtype=torch.long),
            'target_token': torch.tensor(ex['target_token'], dtype=torch.long),
            'seq_len': ex['seq_len']
        }


class RecurrentActionModel(nn.Module):
    """
    Model with:
    - Token embeddings (learnable, not frozen)
    - Position embeddings
    - Recurrent think steps
    - Explicit LOOK action via attention
    """

    def __init__(self, vocab_size=26, d_model=64, max_seq_len=12, n_heads=4, n_think_steps=3):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.n_think_steps = n_think_steps

        # Embeddings (NOT frozen - learning from scratch)
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        # Query position embedding (the "LOOK at position K" instruction)
        self.query_pos_embed = nn.Embedding(max_seq_len, d_model)

        # Single transformer layer for encoding
        self.encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )

        # Recurrent "think" layer - processes state across steps
        self.think_layer = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True
        )

        # Attention mechanism for LOOK action
        self.look_query = nn.Linear(d_model, d_model)
        self.look_key = nn.Linear(d_model, d_model)
        self.look_value = nn.Linear(d_model, d_model)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, vocab_size)
        )

        # Confidence head (self-evaluation: "am I sure?")
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def encode_sequence(self, tokens):
        """Encode the sequence."""
        batch_size, seq_len = tokens.shape

        # Token + position embeddings
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embed(tokens) + self.pos_embed(positions)

        # Encode
        encoded = self.encoder(x)
        return encoded

    def look_action(self, encoded, query_pos, state):
        """
        Explicit LOOK action: attend to the queried position.

        Args:
            encoded: [batch, seq_len, d_model] - encoded sequence
            query_pos: [batch] - which position to look at
            state: [batch, d_model] - current think state

        Returns:
            observation: [batch, d_model] - what we "saw"
            attention: [batch, seq_len] - attention weights (for interpretability)
        """
        batch_size = encoded.size(0)

        # Build query from position embedding + current state
        pos_query = self.query_pos_embed(query_pos)  # [batch, d_model]
        combined_query = pos_query + state  # Incorporate what we've learned so far

        # Attention
        Q = self.look_query(combined_query).unsqueeze(1)  # [batch, 1, d_model]
        K = self.look_key(encoded)  # [batch, seq_len, d_model]
        V = self.look_value(encoded)  # [batch, seq_len, d_model]

        # Scaled dot-product attention
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.d_model ** 0.5)  # [batch, 1, seq_len]
        attention = F.softmax(scores, dim=-1)  # [batch, 1, seq_len]

        # Get observation
        observation = torch.bmm(attention, V).squeeze(1)  # [batch, d_model]

        return observation, attention.squeeze(1)

    def forward(self, tokens, target_pos, return_steps=False):
        """
        Forward pass with recurrent think steps.

        Args:
            tokens: [batch, seq_len] - input sequence
            target_pos: [batch] - position to retrieve from
            return_steps: if True, return intermediate states for gradient visibility

        Returns:
            logits: [batch, vocab_size] - prediction
            confidence: [batch, 1] - self-evaluated confidence
            (optional) steps: list of intermediate states
        """
        batch_size = tokens.size(0)
        device = tokens.device

        # Encode sequence once
        encoded = self.encode_sequence(tokens)

        # Initialize think state
        state = torch.zeros(batch_size, self.d_model, device=device)

        steps = []

        # Recurrent think steps
        for step in range(self.n_think_steps):
            # LOOK action
            observation, attention = self.look_action(encoded, target_pos, state)

            # Update state through GRU (the "thinking")
            state_input = observation.unsqueeze(1)  # [batch, 1, d_model]
            _, hidden = self.think_layer(state_input, state.unsqueeze(0))
            state = hidden.squeeze(0)  # [batch, d_model]

            if return_steps:
                steps.append({
                    'step': step,
                    'attention': attention.detach(),
                    'state_norm': state.norm(dim=-1).mean().item()
                })

        # Final prediction
        logits = self.output_head(state)
        confidence = self.confidence_head(state)

        if return_steps:
            return logits, confidence, steps
        return logits, confidence


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        tokens = batch['sequence'].to(device)
        target_pos = batch['target_pos'].to(device)
        target_token = batch['target_token'].to(device)

        optimizer.zero_grad()
        logits, confidence = model(tokens, target_pos)

        # Main loss: predict the right token
        loss = criterion(logits, target_token)

        # Confidence calibration: confidence should match correctness
        preds = logits.argmax(dim=-1)
        correct = (preds == target_token).float()
        confidence_loss = F.binary_cross_entropy(confidence.squeeze(), correct)

        # Combined loss (weight confidence less)
        total_batch_loss = loss + 0.1 * confidence_loss

        total_batch_loss.backward()
        optimizer.step()

        total_loss += loss.item() * tokens.size(0)
        total_correct += correct.sum().item()
        total_samples += tokens.size(0)

    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples
    }


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    total_confidence = 0

    with torch.no_grad():
        for batch in loader:
            tokens = batch['sequence'].to(device)
            target_pos = batch['target_pos'].to(device)
            target_token = batch['target_token'].to(device)

            logits, confidence = model(tokens, target_pos)
            loss = criterion(logits, target_token)

            preds = logits.argmax(dim=-1)
            correct = (preds == target_token).float()

            total_loss += loss.item() * tokens.size(0)
            total_correct += correct.sum().item()
            total_confidence += confidence.sum().item()
            total_samples += tokens.size(0)

    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples,
        'avg_confidence': total_confidence / total_samples
    }


def visualize_attention(model, loader, device, n_examples=3):
    """Show what the model is attending to."""
    model.eval()

    print("\nAttention Visualization:")
    print("-" * 50)

    with torch.no_grad():
        batch = next(iter(loader))
        tokens = batch['sequence'][:n_examples].to(device)
        target_pos = batch['target_pos'][:n_examples].to(device)
        target_token = batch['target_token'][:n_examples]
        seq_lens = batch['seq_len'][:n_examples]

        logits, confidence, steps = model(tokens, target_pos, return_steps=True)
        preds = logits.argmax(dim=-1).cpu()

        for i in range(n_examples):
            seq = tokens[i, :seq_lens[i]].cpu().tolist()
            pos = target_pos[i].item()
            true_token = target_token[i].item()
            pred_token = preds[i].item()
            conf = confidence[i].item()

            print(f"\nExample {i+1}:")
            print(f"  Sequence: {seq}")
            print(f"  LOOK at position {pos} → should get token {true_token}")
            print(f"  Predicted: {pred_token}, Confidence: {conf:.2f}")
            print(f"  {'✓ CORRECT' if pred_token == true_token else '✗ WRONG'}")

            # Show attention at final step
            attn = steps[-1]['attention'][i, :seq_lens[i]].cpu().numpy()
            print(f"  Attention: {[f'{a:.2f}' for a in attn]}")
            print(f"  Peak at position: {attn.argmax()}")


def main(args):
    print("=" * 65)
    print("Recurrent Action Model: Testing Recursion + Explicit Actions")
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Think steps: {args.n_think_steps}")

    # Data
    print("\nGenerating data...")
    train_data = ActionDataset(n_examples=args.n_train, seed=42)
    val_data = ActionDataset(n_examples=args.n_val, seed=123)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Model
    model = RecurrentActionModel(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_think_steps=args.n_think_steps
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Training
    print("\nTraining...")
    print("-" * 65)

    best_acc = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch:2d} | "
              f"Loss: {train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} | "
              f"Acc: {train_metrics['accuracy']:.1%}/{val_metrics['accuracy']:.1%} | "
              f"Conf: {val_metrics['avg_confidence']:.2f}")

        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            torch.save(model.state_dict(), Path(args.data_dir) / 'recurrent_action_model.pt')

        if val_metrics['accuracy'] >= 0.95:
            print(f"\n*** Position retrieval WORKING! ({val_metrics['accuracy']:.1%}) ***")
            break

    # Final evaluation
    print("\n" + "=" * 65)
    print(f"Best accuracy: {best_acc:.1%}")

    if best_acc >= 0.90:
        print("\nRecursion + explicit actions WORK for position retrieval.")
        print("Direction confirmed. Can build on this foundation.")
    else:
        print("\nStill struggling. Need to investigate further.")

    # Visualize
    visualize_attention(model, val_loader, device)

    return best_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--n-train', type=int, default=50000)
    parser.add_argument('--n-val', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-think-steps', type=int, default=3)

    args = parser.parse_args()
    main(args)
