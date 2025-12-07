"""
Pattern Completion Model: Extending Recurrent Actions to Patterns
Coherence Lab - Emergence Project

Building on the 100% success of recurrent_action_model for position retrieval,
this extends the approach to pattern completion:
- Alternating: [A, B, A, B, ?] → A
- Repeating: [A, A, A, ?] → A
- Incrementing: [1, 2, 3, ?] → 4

Key architecture:
- Multiple LOOK actions to gather evidence
- COMPARE action to detect relationships
- Recurrent think steps to build hypothesis
- Predict based on accumulated observations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
from pathlib import Path
import argparse
import json


class PatternDataset(Dataset):
    """
    Generate pattern completion tasks.

    Pattern types:
    - alternating: [A, B, A, B, ?] → A
    - repeating: [A, A, A, ?] → A
    - incrementing: [1, 2, 3, ?] → 4
    - fixed_offset: [A, A+k, A+2k, ?] → A+3k
    """

    def __init__(self, n_examples=50000, vocab_size=26, seed=None, pattern_types=None):
        if seed:
            random.seed(seed)

        self.vocab_size = vocab_size
        self.pattern_types = pattern_types or ['alternating', 'repeating', 'incrementing', 'fixed_offset']

        self.examples = []
        for _ in range(n_examples):
            pattern_type = random.choice(self.pattern_types)
            example = self._generate_example(pattern_type)
            self.examples.append(example)

    def _generate_example(self, pattern_type):
        if pattern_type == 'alternating':
            # [A, B, A, B, A, ?] → B (or A depending on length)
            a = random.randint(0, self.vocab_size - 1)
            b = random.randint(0, self.vocab_size - 1)
            while b == a:
                b = random.randint(0, self.vocab_size - 1)

            length = random.randint(4, 8)
            sequence = [a if i % 2 == 0 else b for i in range(length)]
            target = a if length % 2 == 0 else b

            return {
                'sequence': sequence,
                'target': target,
                'pattern_type': 'alternating',
                'pattern_params': {'a': a, 'b': b}
            }

        elif pattern_type == 'repeating':
            # [A, A, A, ?] → A
            a = random.randint(0, self.vocab_size - 1)
            length = random.randint(3, 7)
            sequence = [a] * length
            target = a

            return {
                'sequence': sequence,
                'target': target,
                'pattern_type': 'repeating',
                'pattern_params': {'a': a}
            }

        elif pattern_type == 'incrementing':
            # [1, 2, 3, ?] → 4
            start = random.randint(0, self.vocab_size - 5)
            length = random.randint(3, 6)
            sequence = [start + i for i in range(length)]
            target = start + length

            # Ensure target is in vocab
            if target >= self.vocab_size:
                target = self.vocab_size - 1

            return {
                'sequence': sequence,
                'target': target,
                'pattern_type': 'incrementing',
                'pattern_params': {'start': start}
            }

        elif pattern_type == 'fixed_offset':
            # [A, A+k, A+2k, ?] → A+3k
            k = random.randint(1, 4)
            start = random.randint(0, self.vocab_size - 1 - k * 4)
            length = random.randint(3, 5)
            sequence = [start + i * k for i in range(length)]
            target = start + length * k

            if target >= self.vocab_size:
                target = self.vocab_size - 1

            return {
                'sequence': sequence,
                'target': target,
                'pattern_type': 'fixed_offset',
                'pattern_params': {'start': start, 'k': k}
            }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        max_len = 12
        padded = ex['sequence'] + [0] * (max_len - len(ex['sequence']))

        return {
            'sequence': torch.tensor(padded[:max_len], dtype=torch.long),
            'target': torch.tensor(ex['target'], dtype=torch.long),
            'seq_len': len(ex['sequence']),
            'pattern_type': ex['pattern_type']
        }


class PatternCompletionModel(nn.Module):
    """
    Model that uses recurrent LOOK actions to complete patterns.

    Architecture:
    - Token + position embeddings (learnable)
    - Encoder for sequence representation
    - Recurrent think steps with:
      - LOOK action (attend to specific positions)
      - State update via GRU
    - Learnable "where to look" based on think step
    - Output prediction from final state
    """

    def __init__(self, vocab_size=26, d_model=64, max_seq_len=12, n_heads=4, n_think_steps=5):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.n_think_steps = n_think_steps

        # Embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        # Step embedding (which think step we're on)
        self.step_embed = nn.Embedding(n_think_steps, d_model)

        # Encoder
        self.encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )

        # Recurrent think layer
        self.think_layer = nn.GRU(
            input_size=d_model * 2,  # observation + step info
            hidden_size=d_model,
            num_layers=1,
            batch_first=True
        )

        # LOOK action - learns where to attend based on state and step
        self.look_query = nn.Linear(d_model * 2, d_model)  # state + step → query
        self.look_key = nn.Linear(d_model, d_model)
        self.look_value = nn.Linear(d_model, d_model)

        # COMPARE action - detect relationships between observations
        self.compare_layer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, vocab_size)
        )

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

        # Pattern type classifier (auxiliary task for gradient visibility)
        self.pattern_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 4)  # 4 pattern types
        )

    def encode_sequence(self, tokens, seq_lens=None):
        """Encode the sequence."""
        batch_size, seq_len = tokens.shape

        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embed(tokens) + self.pos_embed(positions)

        # Create padding mask if seq_lens provided
        if seq_lens is not None:
            mask = torch.zeros(batch_size, seq_len, device=tokens.device)
            for i, slen in enumerate(seq_lens):
                mask[i, slen:] = float('-inf')
        else:
            mask = None

        encoded = self.encoder(x)
        return encoded

    def look_action(self, encoded, state, step_idx, seq_lens=None):
        """
        Learnable LOOK action - model decides where to attend.

        Args:
            encoded: [batch, seq_len, d_model]
            state: [batch, d_model]
            step_idx: which think step
            seq_lens: actual sequence lengths

        Returns:
            observation: [batch, d_model]
            attention: [batch, seq_len]
        """
        batch_size = encoded.size(0)
        device = encoded.device

        # Get step embedding
        step_tensor = torch.full((batch_size,), step_idx, dtype=torch.long, device=device)
        step_emb = self.step_embed(step_tensor)  # [batch, d_model]

        # Query from state + step
        combined = torch.cat([state, step_emb], dim=-1)  # [batch, d_model*2]
        Q = self.look_query(combined).unsqueeze(1)  # [batch, 1, d_model]

        K = self.look_key(encoded)  # [batch, seq_len, d_model]
        V = self.look_value(encoded)  # [batch, seq_len, d_model]

        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.d_model ** 0.5)  # [batch, 1, seq_len]

        # Mask padding
        if seq_lens is not None:
            for i, slen in enumerate(seq_lens):
                scores[i, 0, slen:] = float('-inf')

        attention = F.softmax(scores, dim=-1)  # [batch, 1, seq_len]
        observation = torch.bmm(attention, V).squeeze(1)  # [batch, d_model]

        return observation, attention.squeeze(1)

    def forward(self, tokens, seq_lens=None, return_steps=False):
        """
        Forward pass with recurrent think steps.

        The model learns to:
        1. Look at different positions across think steps
        2. Build up a representation of the pattern
        3. Predict the next token
        """
        batch_size = tokens.size(0)
        device = tokens.device

        # Encode sequence once
        encoded = self.encode_sequence(tokens, seq_lens)

        # Initialize state
        state = torch.zeros(batch_size, self.d_model, device=device)
        prev_observation = torch.zeros(batch_size, self.d_model, device=device)

        steps = []

        for step in range(self.n_think_steps):
            # LOOK action
            observation, attention = self.look_action(encoded, state, step, seq_lens)

            # Get step embedding
            step_tensor = torch.full((batch_size,), step, dtype=torch.long, device=device)
            step_emb = self.step_embed(step_tensor)

            # Think input: observation + step info
            think_input = torch.cat([observation, step_emb], dim=-1).unsqueeze(1)

            # Update state
            _, hidden = self.think_layer(think_input, state.unsqueeze(0))
            state = hidden.squeeze(0)

            if return_steps:
                steps.append({
                    'step': step,
                    'attention': attention.detach(),
                    'state_norm': state.norm(dim=-1).mean().item()
                })

            prev_observation = observation

        # Final prediction
        logits = self.output_head(state)
        confidence = self.confidence_head(state)

        # Pattern classification (auxiliary)
        pattern_logits = self.pattern_classifier(state)

        if return_steps:
            return logits, confidence, pattern_logits, steps
        return logits, confidence, pattern_logits


def train_epoch(model, loader, optimizer, criterion, device, pattern_to_idx):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    pattern_correct = {p: 0 for p in pattern_to_idx}
    pattern_total = {p: 0 for p in pattern_to_idx}

    for batch in loader:
        tokens = batch['sequence'].to(device)
        targets = batch['target'].to(device)
        seq_lens = batch['seq_len']
        pattern_types = batch['pattern_type']

        optimizer.zero_grad()
        logits, confidence, pattern_logits = model(tokens, seq_lens)

        # Main loss: predict next token
        main_loss = criterion(logits, targets)

        # Auxiliary loss: pattern classification (for gradient visibility)
        pattern_targets = torch.tensor([pattern_to_idx[p] for p in pattern_types], device=device)
        aux_loss = criterion(pattern_logits, pattern_targets)

        # Confidence calibration
        preds = logits.argmax(dim=-1)
        correct = (preds == targets).float()
        conf_loss = F.binary_cross_entropy(confidence.squeeze(), correct)

        # Combined loss
        loss = main_loss + 0.2 * aux_loss + 0.1 * conf_loss

        loss.backward()
        optimizer.step()

        total_loss += main_loss.item() * tokens.size(0)
        total_correct += correct.sum().item()
        total_samples += tokens.size(0)

        # Track per-pattern
        for i, pt in enumerate(pattern_types):
            pattern_total[pt] += 1
            if correct[i]:
                pattern_correct[pt] += 1

    per_pattern = {p: pattern_correct[p] / pattern_total[p] if pattern_total[p] > 0 else 0
                   for p in pattern_to_idx}

    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples,
        'per_pattern': per_pattern
    }


def evaluate(model, loader, criterion, device, pattern_to_idx):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    total_confidence = 0
    pattern_correct = {p: 0 for p in pattern_to_idx}
    pattern_total = {p: 0 for p in pattern_to_idx}

    with torch.no_grad():
        for batch in loader:
            tokens = batch['sequence'].to(device)
            targets = batch['target'].to(device)
            seq_lens = batch['seq_len']
            pattern_types = batch['pattern_type']

            logits, confidence, _ = model(tokens, seq_lens)
            loss = criterion(logits, targets)

            preds = logits.argmax(dim=-1)
            correct = (preds == targets).float()

            total_loss += loss.item() * tokens.size(0)
            total_correct += correct.sum().item()
            total_confidence += confidence.sum().item()
            total_samples += tokens.size(0)

            for i, pt in enumerate(pattern_types):
                pattern_total[pt] += 1
                if correct[i]:
                    pattern_correct[pt] += 1

    per_pattern = {p: pattern_correct[p] / pattern_total[p] if pattern_total[p] > 0 else 0
                   for p in pattern_to_idx}

    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples,
        'avg_confidence': total_confidence / total_samples,
        'per_pattern': per_pattern
    }


def visualize_attention(model, loader, device, n_examples=3):
    """Show what the model is attending to across think steps."""
    model.eval()

    print("\nAttention Visualization:")
    print("-" * 60)

    with torch.no_grad():
        batch = next(iter(loader))
        tokens = batch['sequence'][:n_examples].to(device)
        targets = batch['target'][:n_examples]
        seq_lens = batch['seq_len'][:n_examples]
        pattern_types = batch['pattern_type'][:n_examples]

        logits, confidence, _, steps = model(tokens, seq_lens, return_steps=True)
        preds = logits.argmax(dim=-1).cpu()

        for i in range(n_examples):
            seq = tokens[i, :seq_lens[i]].cpu().tolist()
            true_token = targets[i].item()
            pred_token = preds[i].item()
            conf = confidence[i].item()
            pt = pattern_types[i]

            print(f"\nExample {i+1} [{pt}]:")
            print(f"  Sequence: {seq}")
            print(f"  Target: {true_token}, Predicted: {pred_token}, Confidence: {conf:.2f}")
            print(f"  {'✓ CORRECT' if pred_token == true_token else '✗ WRONG'}")

            print(f"  Think steps (where model looked):")
            for step_info in steps:
                attn = step_info['attention'][i, :seq_lens[i]].cpu().numpy()
                peak_pos = attn.argmax()
                peak_val = attn[peak_pos]
                print(f"    Step {step_info['step']}: peak at pos {peak_pos} ({peak_val:.2f}) → saw {seq[peak_pos]}")


def main(args):
    print("=" * 70)
    print("Pattern Completion Model: Extending Recurrent Actions to Patterns")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Think steps: {args.n_think_steps}")

    # Pattern type mapping
    pattern_types = ['alternating', 'repeating', 'incrementing', 'fixed_offset']
    pattern_to_idx = {p: i for i, p in enumerate(pattern_types)}

    # Data
    print("\nGenerating pattern data...")
    train_data = PatternDataset(n_examples=args.n_train, seed=42, pattern_types=pattern_types)
    val_data = PatternDataset(n_examples=args.n_val, seed=123, pattern_types=pattern_types)

    # Count pattern distribution
    train_counts = {}
    for ex in train_data.examples:
        pt = ex['pattern_type']
        train_counts[pt] = train_counts.get(pt, 0) + 1
    print(f"Train distribution: {train_counts}")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Model
    model = PatternCompletionModel(
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
    print("-" * 70)

    best_acc = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, pattern_to_idx)
        val_metrics = evaluate(model, val_loader, criterion, device, pattern_to_idx)

        history.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics
        })

        print(f"\nEpoch {epoch:2d} | Loss: {train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} | "
              f"Acc: {train_metrics['accuracy']:.1%}/{val_metrics['accuracy']:.1%}")
        print("  Per-pattern accuracy:")
        for pt in pattern_types:
            train_acc = train_metrics['per_pattern'].get(pt, 0)
            val_acc = val_metrics['per_pattern'].get(pt, 0)
            status = "✓" if val_acc >= 0.85 else "←"
            print(f"    {pt:15s}: {train_acc:.1%}/{val_acc:.1%} {status}")

        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': best_acc,
                'per_pattern': val_metrics['per_pattern'],
                'args': vars(args)
            }, Path(args.data_dir) / 'pattern_completion_model.pt')

        # Check if all patterns mastered
        all_good = all(acc >= 0.90 for acc in val_metrics['per_pattern'].values())
        if all_good and val_metrics['accuracy'] >= 0.92:
            print(f"\n*** ALL PATTERNS WORKING! ({val_metrics['accuracy']:.1%}) ***")
            break

    # Final evaluation
    print("\n" + "=" * 70)
    print(f"Best accuracy: {best_acc:.1%}")

    if best_acc >= 0.85:
        print("\nPattern completion with recurrent actions WORKS!")
        print("Recursion + explicit attention + unfrozen learning = success")
    else:
        print("\nNeed more work. Check per-pattern breakdown.")

    # Visualize
    visualize_attention(model, val_loader, device)

    # Save history
    with open(Path(args.data_dir) / 'pattern_completion_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    return best_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--n-train', type=int, default=80000)
    parser.add_argument('--n-val', type=int, default=8000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-think-steps', type=int, default=5)

    args = parser.parse_args()
    main(args)
