"""
Phase 1: Minimal Transformer for Reflex Classification
Coherence Lab - Emergence Project

~270K parameter model for trivially learnable reflex patterns.
Establishes frozen-layer mechanism for curriculum learning.
"""

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """Single transformer block with configurable attention heads."""

    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
        Returns:
            x: [batch_size, seq_len, embed_dim]
        """
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # FFN with residual
        x = self.norm2(x + self.ffn(x))

        return x


class Phase1Transformer(nn.Module):
    """Minimal transformer for Phase 1 reflex classification."""

    def __init__(
        self,
        vocab_size=26,
        max_seq_len=20,
        embed_dim=128,
        num_layers=2,
        num_heads=1,
        ffn_dim=256,
        num_classes=6,
        dropout=0.1
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.target_marker = nn.Parameter(torch.randn(embed_dim) * 0.02)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        # Output head
        self.output_head = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize embeddings with small random values."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.normal_(self.output_head.weight, std=0.02)
        nn.init.zeros_(self.output_head.bias)

    def forward(self, token_ids, target_pos):
        """
        Args:
            token_ids: [batch_size, seq_len] - token indices (0-25)
            target_pos: [batch_size] - target position per sample

        Returns:
            logits: [batch_size, num_classes] - classification logits
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        # Create position indices
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # Combine embeddings
        x = self.token_embedding(token_ids) + self.position_embedding(positions)

        # Add target marker at target positions
        batch_indices = torch.arange(batch_size, device=device)
        x[batch_indices, target_pos] = x[batch_indices, target_pos] + self.target_marker

        x = self.dropout(x)

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Pool at target position
        pooled = x[batch_indices, target_pos]

        # Classification
        logits = self.output_head(pooled)

        return logits

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_transformer_blocks(self):
        """Freeze transformer blocks for Phase 2 training."""
        for block in self.transformer_blocks:
            for param in block.parameters():
                param.requires_grad = False
        print(f"Frozen {len(self.transformer_blocks)} transformer blocks")

    def save_checkpoint(self, path, metadata=None):
        """Save model checkpoint."""
        checkpoint = {
            'state_dict': self.state_dict(),
            'config': {
                'vocab_size': self.vocab_size,
                'max_seq_len': self.max_seq_len,
                'embed_dim': self.embed_dim,
                'num_classes': self.num_classes
            },
            'metadata': metadata or {}
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    @classmethod
    def load_checkpoint(cls, path):
        """Load model from checkpoint."""
        checkpoint = torch.load(path)
        config = checkpoint['config']
        model = cls(
            vocab_size=config['vocab_size'],
            max_seq_len=config['max_seq_len'],
            embed_dim=config['embed_dim'],
            num_classes=config['num_classes']
        )
        model.load_state_dict(checkpoint['state_dict'])
        return model, checkpoint.get('metadata', {})


def main():
    """Test model creation and forward pass."""
    print("Phase 1 Transformer Model")
    print("=" * 40)

    # Create model
    model = Phase1Transformer()
    print(f"Total parameters: {model.count_parameters():,}")

    # Test forward pass
    batch_size = 4
    seq_len = 10
    token_ids = torch.randint(0, 26, (batch_size, seq_len))
    target_pos = torch.randint(0, seq_len, (batch_size,))

    logits = model(token_ids, target_pos)
    print(f"\nInput shape: {token_ids.shape}")
    print(f"Target positions: {target_pos.tolist()}")
    print(f"Output shape: {logits.shape}")
    print(f"Output (logits): {logits[0].tolist()}")

    # Test freezing
    print("\nTesting freeze mechanism:")
    model.freeze_transformer_blocks()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable after freeze: {trainable:,}")


if __name__ == "__main__":
    main()
