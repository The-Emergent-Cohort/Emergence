"""
Phase 2: Extended Transformer for Sequence Prediction
Coherence Lab - Emergence Project

Loads frozen Phase 1 layers and adds new trainable layers for
action-consequence mapping (next-token prediction).
"""

import torch
import torch.nn as nn
from pathlib import Path

# Import Phase 1 components
try:
    from phase1_model import TransformerBlock, Phase1Transformer
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from phase1_model import TransformerBlock, Phase1Transformer


class Phase2Transformer(nn.Module):
    """
    Transformer for Phase 2: Sequence Prediction.

    Loads frozen Phase 1 layers and adds new trainable layers.
    Task: Given a sequence, predict the next token.
    """

    def __init__(
        self,
        phase1_checkpoint_path=None,
        vocab_size=26,
        max_seq_len=20,
        embed_dim=128,
        num_new_layers=4,
        num_heads=2,
        ffn_dim=256,
        dropout=0.1
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        # Load Phase 1 model (will be frozen)
        if phase1_checkpoint_path and Path(phase1_checkpoint_path).exists():
            self.phase1, metadata = Phase1Transformer.load_checkpoint(phase1_checkpoint_path)
            print(f"Loaded Phase 1 checkpoint: {metadata}")
        else:
            print("No Phase 1 checkpoint found, initializing fresh Phase 1 layers")
            self.phase1 = Phase1Transformer(
                vocab_size=vocab_size,
                max_seq_len=max_seq_len,
                embed_dim=embed_dim,
                num_layers=2,
                num_heads=1,
                ffn_dim=ffn_dim,
                num_classes=6,
                dropout=dropout
            )

        # Freeze all Phase 1 parameters
        self._freeze_phase1()

        # New Phase 2 transformer layers (trainable)
        self.phase2_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_new_layers)
        ])

        # New output head for next-token prediction
        self.next_token_head = nn.Linear(embed_dim, vocab_size)
        nn.init.normal_(self.next_token_head.weight, std=0.02)
        nn.init.zeros_(self.next_token_head.bias)

        self.dropout = nn.Dropout(dropout)

    def _freeze_phase1(self):
        """Freeze all Phase 1 parameters."""
        frozen_count = 0
        for param in self.phase1.parameters():
            param.requires_grad = False
            frozen_count += param.numel()
        print(f"Frozen {frozen_count:,} Phase 1 parameters")

    def forward(self, token_ids):
        """
        Args:
            token_ids: [batch_size, seq_len] - input sequence

        Returns:
            logits: [batch_size, vocab_size] - next token prediction
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        # Use Phase 1 embeddings (frozen)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.phase1.token_embedding(token_ids) + self.phase1.position_embedding(positions)
        x = self.phase1.dropout(x)

        # Pass through frozen Phase 1 transformer blocks
        for block in self.phase1.transformer_blocks:
            x = block(x)

        # Pass through new Phase 2 blocks (trainable)
        for block in self.phase2_blocks:
            x = block(x)

        # Pool: use last position for next-token prediction
        last_hidden = x[:, -1, :]

        # Predict next token
        logits = self.next_token_head(last_hidden)

        return logits

    def count_parameters(self):
        """Count trainable and frozen parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}

    def save_checkpoint(self, path, metadata=None):
        """Save model checkpoint."""
        checkpoint = {
            'state_dict': self.state_dict(),
            'config': {
                'vocab_size': self.vocab_size,
                'max_seq_len': self.max_seq_len,
                'embed_dim': self.embed_dim,
            },
            'metadata': metadata or {}
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")


def main():
    """Test Phase 2 model creation."""
    print("Phase 2 Transformer Model")
    print("=" * 50)

    # Try to load Phase 1 checkpoint
    checkpoint_path = Path(__file__).parent / "data" / "phase1_checkpoint.pt"

    model = Phase2Transformer(
        phase1_checkpoint_path=checkpoint_path,
        num_new_layers=4,
        num_heads=2
    )

    params = model.count_parameters()
    print(f"\nParameter counts:")
    print(f"  Frozen (Phase 1): {params['frozen']:,}")
    print(f"  Trainable (Phase 2): {params['trainable']:,}")
    print(f"  Total: {params['total']:,}")

    # Test forward pass
    batch_size = 4
    seq_len = 8
    token_ids = torch.randint(0, 26, (batch_size, seq_len))

    logits = model(token_ids)
    print(f"\nInput shape: {token_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Sample prediction: token {logits[0].argmax().item()}")


if __name__ == "__main__":
    main()
