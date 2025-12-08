"""
Integrated Phase 1: Multi-Task Foundation Model
Coherence Lab - Emergence Project

Single model trained on 4 tasks simultaneously:
1. Token properties (6-way multi-label per token)
2. Token relations (binary: same category?)
3. Position properties (4-way multi-label per position)
4. Position relations (binary: same parity?)
"""

import torch
import torch.nn as nn
import math
from pathlib import Path


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class IntegratedPhase1Model(nn.Module):
    """
    Multi-task transformer for integrated Phase 1 training.

    Architecture:
    - Shared token embeddings + positional encoding
    - Shared transformer layers (4 layers)
    - 4 task-specific heads
    """

    def __init__(
        self,
        vocab_size=26,
        d_model=64,
        num_heads=4,
        num_layers=4,
        d_ff=256,
        max_seq_len=20,
        dropout=0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Shared embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)

        # Shared transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Task 1: Token properties head (6-way multi-label per token)
        # Input: per-token hidden states, Output: 6 binary predictions
        self.token_prop_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 6)  # vowel, consonant, even, odd, high, low
        )

        # Task 2: Token relations head (binary: do tokens share category?)
        # Input: concatenated pair of token hidden states
        self.token_rel_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)  # binary: same category?
        )

        # Task 3: Position properties head (4-way multi-label per position)
        # Input: per-position hidden states, Output: 4 binary predictions
        self.pos_prop_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 4)  # even_pos, odd_pos, early, late
        )

        # Task 4: Position relations head (binary: same parity?)
        # Input: concatenated pair of position hidden states
        self.pos_rel_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)  # binary: same parity?
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, tokens, task_info=None):
        """
        Forward pass.

        Args:
            tokens: [batch, seq_len] token IDs
            task_info: dict with task-specific indices/pairs

        Returns:
            dict of task outputs
        """
        # Shared encoding
        x = self.token_embedding(tokens)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Create padding mask
        # (assuming 0 is used for padding, but our vocab includes 0)
        # For simplicity, we'll process all positions
        hidden = self.transformer(x)  # [batch, seq_len, d_model]

        outputs = {}

        # Task 1: Token properties for all positions
        outputs['token_properties'] = self.token_prop_head(hidden)  # [batch, seq_len, 6]

        # Task 3: Position properties for all positions
        outputs['position_properties'] = self.pos_prop_head(hidden)  # [batch, seq_len, 4]

        # Task 2: Token relations (if pairs provided)
        if task_info and 'token_pairs' in task_info:
            # token_pairs: list of (pos1, pos2) tuples per batch item
            # We'll compute for provided pairs
            batch_size = tokens.size(0)
            token_rel_outputs = []

            for b in range(batch_size):
                pairs = task_info['token_pairs'][b]
                if pairs:
                    pair_outputs = []
                    for p1, p2 in pairs:
                        h1 = hidden[b, p1]  # [d_model]
                        h2 = hidden[b, p2]  # [d_model]
                        pair_hidden = torch.cat([h1, h2], dim=-1)  # [2*d_model]
                        pair_out = self.token_rel_head(pair_hidden)  # [1]
                        pair_outputs.append(pair_out)
                    token_rel_outputs.append(torch.stack(pair_outputs).squeeze(-1))
                else:
                    token_rel_outputs.append(torch.tensor([]))

            outputs['token_relations'] = token_rel_outputs

        # Task 4: Position relations (if pairs provided)
        if task_info and 'position_pairs' in task_info:
            batch_size = tokens.size(0)
            pos_rel_outputs = []

            for b in range(batch_size):
                pairs = task_info['position_pairs'][b]
                if pairs:
                    pair_outputs = []
                    for p1, p2 in pairs:
                        h1 = hidden[b, p1]
                        h2 = hidden[b, p2]
                        pair_hidden = torch.cat([h1, h2], dim=-1)
                        pair_out = self.pos_rel_head(pair_hidden)
                        pair_outputs.append(pair_out)
                    pos_rel_outputs.append(torch.stack(pair_outputs).squeeze(-1))
                else:
                    pos_rel_outputs.append(torch.tensor([]))

            outputs['position_relations'] = pos_rel_outputs

        return outputs

    def count_parameters(self):
        """Count trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}

    def save_checkpoint(self, path, metadata=None):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': {
                'd_model': self.d_model,
                'max_seq_len': self.max_seq_len
            }
        }
        if metadata:
            checkpoint['metadata'] = metadata
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    @classmethod
    def load_checkpoint(cls, path, device='cpu'):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint.get('config', {})

        model = cls(
            d_model=config.get('d_model', 64),
            max_seq_len=config.get('max_seq_len', 20)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint.get('metadata', {})


def main():
    """Test the model."""
    print("Integrated Phase 1: Multi-Task Foundation Model")
    print("=" * 55)

    model = IntegratedPhase1Model()
    params = model.count_parameters()
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")

    # Test forward pass
    batch_size = 4
    seq_len = 8
    tokens = torch.randint(0, 26, (batch_size, seq_len))

    # Task info with pairs
    task_info = {
        'token_pairs': [[(0, 1), (2, 3)] for _ in range(batch_size)],
        'position_pairs': [[(0, 2), (1, 3)] for _ in range(batch_size)]
    }

    outputs = model(tokens, task_info)

    print(f"\nTest forward pass (batch={batch_size}, seq_len={seq_len}):")
    print(f"  Token properties shape: {outputs['token_properties'].shape}")
    print(f"  Position properties shape: {outputs['position_properties'].shape}")
    print(f"  Token relations: {len(outputs['token_relations'])} batches, "
          f"{len(outputs['token_relations'][0])} pairs each")
    print(f"  Position relations: {len(outputs['position_relations'])} batches, "
          f"{len(outputs['position_relations'][0])} pairs each")


if __name__ == "__main__":
    main()
