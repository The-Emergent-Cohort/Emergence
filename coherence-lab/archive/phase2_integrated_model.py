"""
Phase 2 v2: Sequence Prediction with Integrated Phase 1 Foundation
Coherence Lab - Emergence Project

Uses the integrated Phase 1 checkpoint (with position awareness) as frozen foundation.
"""

import torch
import torch.nn as nn
from pathlib import Path

try:
    from phase1_integrated_model import IntegratedPhase1Model
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from phase1_integrated_model import IntegratedPhase1Model


class Phase2IntegratedModel(nn.Module):
    """
    Phase 2 model built on integrated Phase 1 foundation.

    Freezes the shared encoder from Phase 1 and adds new layers
    for sequence prediction.
    """

    def __init__(
        self,
        phase1_checkpoint_path=None,
        vocab_size=26,
        d_model=64,
        num_heads=4,
        num_new_layers=4,
        d_ff=256,
        max_seq_len=20,
        dropout=0.1
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Load Phase 1 integrated model
        if phase1_checkpoint_path and Path(phase1_checkpoint_path).exists():
            self.phase1, metadata = IntegratedPhase1Model.load_checkpoint(
                phase1_checkpoint_path
            )
            print(f"Loaded integrated Phase 1: {metadata}")
            # Use Phase 1's d_model
            self.d_model = self.phase1.d_model
            d_model = self.d_model
        else:
            print("No Phase 1 checkpoint - initializing fresh")
            self.phase1 = IntegratedPhase1Model(
                vocab_size=vocab_size,
                d_model=d_model,
                num_heads=num_heads,
                num_layers=4,
                d_ff=d_ff,
                max_seq_len=max_seq_len,
                dropout=dropout
            )

        # Freeze Phase 1
        self._freeze_phase1()

        # New Phase 2 transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.phase2_layers = nn.TransformerEncoder(encoder_layer, num_layers=num_new_layers)

        # Output head for next-token prediction
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, vocab_size)
        )

        self._init_phase2_weights()

    def _freeze_phase1(self):
        """Freeze all Phase 1 parameters."""
        frozen = 0
        for param in self.phase1.parameters():
            param.requires_grad = False
            frozen += param.numel()
        print(f"Frozen {frozen:,} Phase 1 parameters")

    def _init_phase2_weights(self):
        """Initialize Phase 2 weights."""
        for module in self.output_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, tokens):
        """
        Forward pass for sequence prediction.

        Args:
            tokens: [batch, seq_len] input tokens

        Returns:
            logits: [batch, vocab_size] next token prediction
        """
        # Get hidden states from frozen Phase 1
        x = self.phase1.token_embedding(tokens)
        x = self.phase1.pos_encoding(x)
        x = self.phase1.dropout(x)
        hidden = self.phase1.transformer(x)  # [batch, seq_len, d_model]

        # Pass through new Phase 2 layers
        hidden = self.phase2_layers(hidden)

        # Use last position for prediction
        last_hidden = hidden[:, -1, :]  # [batch, d_model]

        # Predict next token
        logits = self.output_head(last_hidden)  # [batch, vocab_size]

        return logits

    def count_parameters(self):
        """Count parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {'trainable': trainable, 'frozen': frozen, 'total': trainable + frozen}


def main():
    """Test the model."""
    print("Phase 2 v2: With Integrated Phase 1 Foundation")
    print("=" * 55)

    checkpoint = Path(__file__).parent / "data" / "phase1_integrated_checkpoint.pt"

    model = Phase2IntegratedModel(phase1_checkpoint_path=checkpoint)
    params = model.count_parameters()

    print(f"\nParameters:")
    print(f"  Frozen (Phase 1): {params['frozen']:,}")
    print(f"  Trainable (Phase 2): {params['trainable']:,}")
    print(f"  Total: {params['total']:,}")

    # Test forward pass
    tokens = torch.randint(0, 26, (4, 8))
    logits = model(tokens)
    print(f"\nInput: {tokens.shape}")
    print(f"Output: {logits.shape}")


if __name__ == "__main__":
    main()
