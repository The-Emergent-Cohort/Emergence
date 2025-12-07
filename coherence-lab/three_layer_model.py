"""
Three-Layer Architecture: World Model + Learner + Teacher
Coherence Lab - Emergence Project

Based on early childhood learning principles:
- World Model: The "physics" - actions have consequences
- Learner: Explores, tries things, builds hypotheses (the HRM)
- Teacher: Monitors, intervenes when stuck, validates learning

Key principles:
- Free play mode: learner explores without constant oversight
- Self-validation: learner checks work against world model
- Frustration detection: "stop" becomes "ask for direction"
- Conditional weights: learning tentative until test-confirmed
- Positive reinforcement: reward good, explain (don't punish) wrong
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
from pathlib import Path
import argparse
import json
from collections import deque


# =============================================================================
# LAYER 1: WORLD MODEL
# =============================================================================

class WorldModel(nn.Module):
    """
    The "physics" layer - provides ground truth for actions.

    Like a shape sorter: you try an action, it either fits or doesn't.
    No judgment, just reality.

    Actions:
    - LOOK(pos) → token at that position
    - COMPARE(pos1, pos2) → same/different
    - PREDICT_NEXT(sequence) → what should come next
    """

    def __init__(self, vocab_size=26, max_seq_len=12):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Pattern rules (the "physics")
        self.pattern_rules = {
            'alternating': self._alternating_next,
            'repeating': self._repeating_next,
            'incrementing': self._incrementing_next,
            'fixed_offset': self._fixed_offset_next
        }

    def look(self, sequence, position, seq_len):
        """
        LOOK action: What token is at this position?
        Returns the token, or -1 if out of bounds.
        """
        batch_size = sequence.size(0)
        results = torch.zeros(batch_size, dtype=torch.long, device=sequence.device)
        valid = torch.zeros(batch_size, dtype=torch.bool, device=sequence.device)

        for i in range(batch_size):
            pos = position[i].item() if isinstance(position, torch.Tensor) else position
            slen = seq_len[i] if isinstance(seq_len, (list, torch.Tensor)) else seq_len

            if 0 <= pos < slen:
                results[i] = sequence[i, pos]
                valid[i] = True
            else:
                results[i] = -1
                valid[i] = False

        return results, valid

    def compare(self, sequence, pos1, pos2, seq_len):
        """
        COMPARE action: Are tokens at pos1 and pos2 the same?
        Returns: 1 if same, 0 if different, -1 if invalid
        """
        tok1, valid1 = self.look(sequence, pos1, seq_len)
        tok2, valid2 = self.look(sequence, pos2, seq_len)

        result = torch.where(
            valid1 & valid2,
            (tok1 == tok2).long(),
            torch.tensor(-1, device=sequence.device)
        )
        return result, valid1 & valid2

    def _alternating_next(self, sequence, seq_len):
        """Alternating: [A,B,A,B,...] → next in pattern"""
        # Next is same as position (seq_len % 2)
        batch_size = sequence.size(0)
        targets = []
        for i in range(batch_size):
            slen = seq_len[i] if isinstance(seq_len, (list, torch.Tensor)) else seq_len
            pos = slen % 2
            targets.append(sequence[i, pos].item())
        return torch.tensor(targets, device=sequence.device)

    def _repeating_next(self, sequence, seq_len):
        """Repeating: [A,A,A,...] → A"""
        return sequence[:, 0]

    def _incrementing_next(self, sequence, seq_len):
        """Incrementing: [1,2,3,...] → next integer"""
        batch_size = sequence.size(0)
        targets = []
        for i in range(batch_size):
            slen = seq_len[i] if isinstance(seq_len, (list, torch.Tensor)) else seq_len
            last = sequence[i, slen - 1].item()
            targets.append(min(last + 1, self.vocab_size - 1))
        return torch.tensor(targets, device=sequence.device)

    def _fixed_offset_next(self, sequence, seq_len):
        """Fixed offset: [A,A+k,A+2k,...] → A+n*k"""
        batch_size = sequence.size(0)
        targets = []
        for i in range(batch_size):
            slen = seq_len[i] if isinstance(seq_len, (list, torch.Tensor)) else seq_len
            if slen >= 2:
                k = sequence[i, 1].item() - sequence[i, 0].item()
                last = sequence[i, slen - 1].item()
                targets.append(min(max(0, last + k), self.vocab_size - 1))
            else:
                targets.append(sequence[i, 0].item())
        return torch.tensor(targets, device=sequence.device)

    def get_ground_truth(self, sequence, seq_len, pattern_type):
        """Get the correct next token according to pattern rules."""
        if isinstance(pattern_type, str):
            return self.pattern_rules[pattern_type](sequence, seq_len)
        else:
            # Batch of different pattern types
            batch_size = sequence.size(0)
            targets = torch.zeros(batch_size, dtype=torch.long, device=sequence.device)
            for i, pt in enumerate(pattern_type):
                single_seq = sequence[i:i+1]
                single_len = [seq_len[i]] if isinstance(seq_len, list) else seq_len[i:i+1]
                targets[i] = self.pattern_rules[pt](single_seq, single_len)[0]
            return targets


# =============================================================================
# LAYER 2: LEARNER (HRM)
# =============================================================================

class Learner(nn.Module):
    """
    The explorer - tries actions, builds hypotheses, learns from consequences.

    Based on recurrent action model that achieved 98.9%:
    - Multiple think steps
    - Learnable attention (decides where to look)
    - Confidence estimation
    - Pattern detection
    """

    def __init__(self, vocab_size=26, d_model=64, max_seq_len=12,
                 n_heads=4, n_think_steps=5):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.n_think_steps = n_think_steps

        # Embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
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
            input_size=d_model * 2,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True
        )

        # Action heads
        self.look_query = nn.Linear(d_model * 2, d_model)
        self.look_key = nn.Linear(d_model, d_model)
        self.look_value = nn.Linear(d_model, d_model)

        # Output heads
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, vocab_size)
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

        # Frustration detector: "am I stuck?"
        self.frustration_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),  # current state + state delta
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

        # Pattern classifier (auxiliary)
        self.pattern_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 4)
        )

        # Learning state
        self.register_buffer('exploration_count', torch.tensor(0))

    def encode_sequence(self, tokens, seq_lens=None):
        batch_size, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embed(tokens) + self.pos_embed(positions)
        encoded = self.encoder(x)
        return encoded

    def look_action(self, encoded, state, step_idx, seq_lens=None):
        batch_size = encoded.size(0)
        device = encoded.device

        step_tensor = torch.full((batch_size,), step_idx, dtype=torch.long, device=device)
        step_emb = self.step_embed(step_tensor)

        combined = torch.cat([state, step_emb], dim=-1)
        Q = self.look_query(combined).unsqueeze(1)
        K = self.look_key(encoded)
        V = self.look_value(encoded)

        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.d_model ** 0.5)

        if seq_lens is not None:
            for i, slen in enumerate(seq_lens):
                scores[i, 0, slen:] = float('-inf')

        attention = F.softmax(scores, dim=-1)
        observation = torch.bmm(attention, V).squeeze(1)

        return observation, attention.squeeze(1)

    def forward(self, tokens, seq_lens=None, return_trace=False):
        """
        Forward pass with think steps.

        Returns:
            logits: prediction
            confidence: how sure
            frustration: how stuck
            pattern_logits: pattern type guess
            trace: (optional) step-by-step info
        """
        batch_size = tokens.size(0)
        device = tokens.device

        encoded = self.encode_sequence(tokens, seq_lens)
        state = torch.zeros(batch_size, self.d_model, device=device)
        prev_state = state.clone()

        trace = [] if return_trace else None

        for step in range(self.n_think_steps):
            observation, attention = self.look_action(encoded, state, step, seq_lens)

            step_tensor = torch.full((batch_size,), step, dtype=torch.long, device=device)
            step_emb = self.step_embed(step_tensor)
            think_input = torch.cat([observation, step_emb], dim=-1).unsqueeze(1)

            prev_state = state.clone()
            _, hidden = self.think_layer(think_input, state.unsqueeze(0))
            state = hidden.squeeze(0)

            if return_trace:
                trace.append({
                    'step': step,
                    'attention': attention.detach(),
                    'state_delta': (state - prev_state).norm(dim=-1).mean().item()
                })

        # Outputs
        logits = self.output_head(state)
        confidence = self.confidence_head(state)

        # Frustration: are we stuck? (low state change over steps)
        state_delta = state - prev_state
        frustration_input = torch.cat([state, state_delta], dim=-1)
        frustration = self.frustration_head(frustration_input)

        pattern_logits = self.pattern_classifier(state)

        if return_trace:
            return logits, confidence, frustration, pattern_logits, trace
        return logits, confidence, frustration, pattern_logits


# =============================================================================
# LAYER 3: TEACHER
# =============================================================================

class Teacher(nn.Module):
    """
    The monitor - watches the learner, provides guidance when needed.

    Roles:
    - Validates learning (confirms tentative → permanent)
    - Detects when learner is stuck
    - Provides hints, not answers
    - Rewards good process, explains (doesn't punish) mistakes
    """

    def __init__(self, d_model=64, frustration_threshold=0.7,
                 confidence_threshold=0.3, patience=3):
        super().__init__()

        self.frustration_threshold = frustration_threshold
        self.confidence_threshold = confidence_threshold
        self.patience = patience

        # Hint generator: given learner state, suggest where to look
        self.hint_generator = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 12)  # attention over positions
        )

        # Validation gate: is this learning ready to be permanent?
        self.validation_gate = nn.Sequential(
            nn.Linear(d_model + 1, d_model // 2),  # state + correctness
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

        # Track learner performance
        self.performance_history = deque(maxlen=100)

    def should_intervene(self, frustration, confidence):
        """
        Decide if teacher should step in.

        Intervene if:
        - High frustration (learner is stuck)
        - Very low confidence (learner is lost)
        """
        stuck = frustration > self.frustration_threshold
        lost = confidence < self.confidence_threshold
        return stuck | lost

    def generate_hint(self, learner_state, correct_position=None):
        """
        Generate a hint for the learner.

        If we know the correct position, bias toward it.
        Otherwise, suggest based on learner state.
        """
        hint_logits = self.hint_generator(learner_state)

        if correct_position is not None:
            # Soft hint: increase probability of correct position
            hint_logits = hint_logits.scatter(1, correct_position.unsqueeze(1),
                                               hint_logits.max() + 1)

        return F.softmax(hint_logits, dim=-1)

    def provide_feedback(self, prediction, target, confidence, process_score):
        """
        Generate teaching feedback.

        Returns:
            reward: scalar reward
            explanation: what went right/wrong
        """
        correct = (prediction == target).float()

        # Base reward: correctness weighted by process
        # Good process + wrong answer = partial credit
        # Lucky guess + right answer = less credit than good process
        reward = 0.5 * correct + 0.5 * process_score

        # Confidence calibration bonus
        # Reward accurate confidence (high when right, low when wrong)
        confidence_accuracy = 1 - torch.abs(confidence.squeeze() - correct)
        reward = reward + 0.1 * confidence_accuracy

        return reward, correct

    def validate_learning(self, learner_state, was_correct, test_performance):
        """
        Decide if tentative learning should become permanent.

        Criteria:
        - Consistently correct on similar problems
        - Confident in answers
        - Good process (not lucky guessing)
        """
        validation_input = torch.cat([learner_state, was_correct.unsqueeze(-1)], dim=-1)
        gate_value = self.validation_gate(validation_input)

        # Additional check: test performance above threshold
        should_consolidate = (gate_value > 0.5) & (test_performance > 0.8)

        return should_consolidate, gate_value


# =============================================================================
# INTEGRATED SYSTEM
# =============================================================================

class ThreeLayerSystem(nn.Module):
    """
    The complete system: World Model + Learner + Teacher working together.
    """

    def __init__(self, vocab_size=26, d_model=64, max_seq_len=12,
                 n_heads=4, n_think_steps=5):
        super().__init__()

        self.world_model = WorldModel(vocab_size, max_seq_len)
        self.learner = Learner(vocab_size, d_model, max_seq_len, n_heads, n_think_steps)
        self.teacher = Teacher(d_model)

        self.vocab_size = vocab_size
        self.d_model = d_model

    def forward(self, tokens, seq_lens=None, pattern_types=None,
                mode='supervised', return_details=False):
        """
        Forward pass through the system.

        Modes:
        - 'supervised': standard training with labels
        - 'free_play': learner explores, teacher monitors
        - 'test': validation mode, no learning
        """
        # Learner makes prediction
        logits, confidence, frustration, pattern_logits, trace = self.learner(
            tokens, seq_lens, return_trace=True
        )

        prediction = logits.argmax(dim=-1)

        # World model provides ground truth (if pattern types known)
        if pattern_types is not None:
            ground_truth = self.world_model.get_ground_truth(tokens, seq_lens, pattern_types)
        else:
            ground_truth = None

        # Teacher decides on intervention
        should_intervene = self.teacher.should_intervene(frustration, confidence)

        # Calculate process score from attention patterns
        # Good process = attending to relevant positions
        process_score = self._calculate_process_score(trace, seq_lens)

        if return_details:
            return {
                'logits': logits,
                'prediction': prediction,
                'confidence': confidence,
                'frustration': frustration,
                'pattern_logits': pattern_logits,
                'ground_truth': ground_truth,
                'should_intervene': should_intervene,
                'process_score': process_score,
                'trace': trace
            }

        return logits, confidence, pattern_logits

    def _calculate_process_score(self, trace, seq_lens):
        """
        Score the learner's process based on attention patterns.

        Good process indicators:
        - Focused attention (not uniform)
        - Progressive refinement (state changes decrease)
        - Looking at relevant positions
        """
        if not trace:
            return torch.zeros(1)

        batch_size = trace[0]['attention'].size(0)
        device = trace[0]['attention'].device

        # Attention focus score: entropy of attention (lower = more focused)
        focus_scores = []
        for step_info in trace:
            attn = step_info['attention']
            entropy = -(attn * (attn + 1e-10).log()).sum(dim=-1)
            max_entropy = torch.log(torch.tensor(attn.size(-1), dtype=torch.float, device=device))
            focus = 1 - entropy / max_entropy  # 1 = perfectly focused, 0 = uniform
            focus_scores.append(focus)

        avg_focus = torch.stack(focus_scores).mean(dim=0)

        # Refinement score: state deltas should decrease (converging)
        if len(trace) >= 2:
            deltas = [step_info['state_delta'] for step_info in trace]
            refinement = 1.0 if deltas[-1] < deltas[0] else 0.5
        else:
            refinement = 0.5

        process_score = 0.7 * avg_focus + 0.3 * refinement
        return process_score

    def train_step(self, batch, optimizer, criterion, pattern_to_idx):
        """
        Single training step with three-layer interaction.
        """
        tokens = batch['tokens']
        targets = batch['target']
        seq_lens = batch['seq_len']
        pattern_types = batch['pattern_type']

        optimizer.zero_grad()

        # Forward through system
        details = self(tokens, seq_lens, pattern_types,
                      mode='supervised', return_details=True)

        # Main loss: prediction
        main_loss = criterion(details['logits'], targets)

        # Auxiliary loss: pattern classification
        pattern_targets = torch.tensor(
            [pattern_to_idx[p] for p in pattern_types],
            device=tokens.device
        )
        aux_loss = criterion(details['pattern_logits'], pattern_targets)

        # Confidence calibration
        correct = (details['prediction'] == targets).float()
        conf_loss = F.binary_cross_entropy(details['confidence'].squeeze(), correct)

        # Process reward (encourage good process)
        process_loss = -details['process_score'].mean()  # maximize process score

        # Combined loss
        loss = main_loss + 0.2 * aux_loss + 0.1 * conf_loss + 0.05 * process_loss

        loss.backward()
        optimizer.step()

        return {
            'loss': main_loss.item(),
            'accuracy': correct.mean().item(),
            'avg_confidence': details['confidence'].mean().item(),
            'avg_frustration': details['frustration'].mean().item(),
            'process_score': details['process_score'].mean().item(),
            'interventions': details['should_intervene'].float().mean().item()
        }


# =============================================================================
# DATASET
# =============================================================================

class PatternDataset(Dataset):
    """Same dataset as pattern_completion_model.py"""

    def __init__(self, n_examples=50000, vocab_size=26, seed=None):
        if seed:
            random.seed(seed)

        self.vocab_size = vocab_size
        self.pattern_types = ['alternating', 'repeating', 'incrementing', 'fixed_offset']

        self.examples = []
        for _ in range(n_examples):
            pattern_type = random.choice(self.pattern_types)
            example = self._generate_example(pattern_type)
            self.examples.append(example)

    def _generate_example(self, pattern_type):
        if pattern_type == 'alternating':
            a = random.randint(0, self.vocab_size - 1)
            b = random.randint(0, self.vocab_size - 1)
            while b == a:
                b = random.randint(0, self.vocab_size - 1)
            length = random.randint(4, 8)
            sequence = [a if i % 2 == 0 else b for i in range(length)]
            target = a if length % 2 == 0 else b

        elif pattern_type == 'repeating':
            a = random.randint(0, self.vocab_size - 1)
            length = random.randint(3, 7)
            sequence = [a] * length
            target = a

        elif pattern_type == 'incrementing':
            length = random.randint(3, 6)
            max_start = self.vocab_size - length - 1
            start = random.randint(0, max(0, max_start))
            sequence = [start + i for i in range(length)]
            target = start + length

        elif pattern_type == 'fixed_offset':
            length = random.randint(3, 5)
            k = random.randint(1, 3)
            max_start = self.vocab_size - k * length - 1
            start = random.randint(0, max(0, max_start))
            sequence = [start + i * k for i in range(length)]
            target = start + length * k

        return {
            'sequence': sequence,
            'target': target,
            'pattern_type': pattern_type
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


def collate_fn(batch):
    return {
        'tokens': torch.stack([b['sequence'] for b in batch]),
        'target': torch.stack([b['target'] for b in batch]),
        'seq_len': [b['seq_len'] for b in batch],
        'pattern_type': [b['pattern_type'] for b in batch]
    }


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, loader, optimizer, criterion, device, pattern_to_idx):
    model.train()
    metrics = {
        'loss': 0, 'accuracy': 0, 'confidence': 0,
        'frustration': 0, 'process': 0, 'interventions': 0
    }
    n_batches = 0

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        step_metrics = model.train_step(batch, optimizer, criterion, pattern_to_idx)

        metrics['loss'] += step_metrics['loss']
        metrics['accuracy'] += step_metrics['accuracy']
        metrics['confidence'] += step_metrics['avg_confidence']
        metrics['frustration'] += step_metrics['avg_frustration']
        metrics['process'] += step_metrics['process_score']
        metrics['interventions'] += step_metrics['interventions']
        n_batches += 1

    return {k: v / n_batches for k, v in metrics.items()}


def evaluate(model, loader, criterion, device, pattern_to_idx):
    model.eval()
    total_correct = 0
    total_samples = 0
    pattern_correct = {p: 0 for p in pattern_to_idx}
    pattern_total = {p: 0 for p in pattern_to_idx}

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            logits, confidence, _ = model(batch['tokens'], batch['seq_len'])
            preds = logits.argmax(dim=-1)
            correct = (preds == batch['target'])

            total_correct += correct.sum().item()
            total_samples += len(preds)

            for i, pt in enumerate(batch['pattern_type']):
                pattern_total[pt] += 1
                if correct[i]:
                    pattern_correct[pt] += 1

    per_pattern = {p: pattern_correct[p] / max(1, pattern_total[p])
                   for p in pattern_to_idx}

    return {
        'accuracy': total_correct / total_samples,
        'per_pattern': per_pattern
    }


def main(args):
    print("=" * 70)
    print("Three-Layer Architecture: World Model + Learner + Teacher")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Think steps: {args.n_think_steps}")

    pattern_types = ['alternating', 'repeating', 'incrementing', 'fixed_offset']
    pattern_to_idx = {p: i for i, p in enumerate(pattern_types)}

    # Data
    print("\nGenerating data...")
    train_data = PatternDataset(n_examples=args.n_train, seed=42)
    val_data = PatternDataset(n_examples=args.n_val, seed=123)

    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=args.batch_size,
                            collate_fn=collate_fn)

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Model
    model = ThreeLayerSystem(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_think_steps=args.n_think_steps
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Training
    print("\nTraining with three-layer system...")
    print("-" * 70)

    best_acc = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer,
                                    criterion, device, pattern_to_idx)
        val_metrics = evaluate(model, val_loader, criterion, device, pattern_to_idx)

        print(f"\nEpoch {epoch:2d}")
        print(f"  Train: loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.1%}")
        print(f"  Val: acc={val_metrics['accuracy']:.1%}")
        print(f"  Learner: conf={train_metrics['confidence']:.2f}, "
              f"frust={train_metrics['frustration']:.2f}, "
              f"process={train_metrics['process']:.2f}")
        print(f"  Teacher interventions: {train_metrics['interventions']:.1%}")
        print("  Per-pattern:")
        for pt in pattern_types:
            print(f"    {pt:15s}: {val_metrics['per_pattern'][pt]:.1%}")

        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': best_acc
            }, Path(args.data_dir) / 'three_layer_model.pt')

        if val_metrics['accuracy'] >= 0.95:
            print(f"\n*** System mastered all patterns! ({val_metrics['accuracy']:.1%}) ***")
            break

    print("\n" + "=" * 70)
    print(f"Best accuracy: {best_acc:.1%}")
    print("=" * 70)

    return best_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--n-train', type=int, default=80000)
    parser.add_argument('--n-val', type=int, default=8000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-think-steps', type=int, default=5)

    args = parser.parse_args()
    main(args)
