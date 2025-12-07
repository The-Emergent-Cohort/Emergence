# Phase 1 Model Specification: Minimal Transformer for Reflex Classification

*Version 0.1 - Dec 7, 2024*

## 1. Task Summary

**Input:** Variable-length sequence of 5-20 tokens (values 0-25) with one marked target position
**Output:** Multi-hot vector of 6 binary labels (reflex classifications)
**Objective:** Trivially learnable baseline that establishes the frozen-layer checkpoint mechanism

---

## 2. Architecture Configuration

| Component | Value | Rationale |
|-----------|-------|-----------|
| Vocab size | 26 | Tokens 0-25 |
| Max seq length | 20 | Per data spec |
| Embedding dim | 128 | Minimal sufficient |
| Num layers | 2 | Light for Phase 1 |
| Num attention heads | 1 | Per curriculum |
| FFN hidden dim | 256 | 2× embedding dim |
| Dropout | 0.1 | Light regularization |
| Num classes | 6 | Reflex labels |

---

## 3. Parameter Count

```
Token embeddings:       26 × 128 =          3,328
Position embeddings:    20 × 128 =          2,560
Target marker:          128 =                 128

Per Transformer Block:
  Attention (Q/K/V/O):  4 × 128 × 128 =    65,536
  LayerNorm × 2:        256 × 2 =             512
  FFN:                  128×256 + 256×128 = 65,536
  Subtotal:                               131,584

2 blocks:               2 × 131,584 =     263,168
Output head:            128 × 6 =             768

TOTAL: ~270K parameters (well under 0.5M budget)
```

---

## 4. Input Design

**Embedding strategy:**
```
input = token_embed(token_id) + position_embed(position)
input[target_pos] += target_marker  # Mark target position
```

---

## 5. Output Design

**Pooling:** Use target position directly
```
pooled = transformer_output[:, target_pos, :]
```

**Classification:** Multi-label with sigmoid
```
logits = linear(pooled)  # [batch, 6]
loss = BCEWithLogitsLoss(logits, labels)
```

---

## 6. Freezing Mechanism

```python
def freeze_for_phase2(model):
    for name, param in model.named_parameters():
        if 'transformer_blocks' in name:
            param.requires_grad = False
```

---

## 7. Model Implementation

```python
import torch
import torch.nn as nn

class Phase1Transformer(nn.Module):
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

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.target_marker = nn.Parameter(torch.randn(embed_dim))

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        self.output_head = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids, target_pos):
        batch_size, seq_len = token_ids.shape

        # Embeddings
        positions = torch.arange(seq_len, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)

        # Add target marker
        for b in range(batch_size):
            x[b, target_pos[b]] += self.target_marker

        x = self.dropout(x)

        # Transformer
        for block in self.transformer_blocks:
            x = block(x)

        # Pool at target position
        pooled = x[range(batch_size), target_pos]

        return self.output_head(pooled)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
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
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x
```

---

## 8. Training Loop

```python
model = Phase1Transformer()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

for batch in dataloader:
    token_ids, target_pos, labels = batch
    logits = model(token_ids, target_pos)
    loss = criterion(logits, labels.float())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 9. Phase 2 Extension Points

1. Freeze transformer blocks from Phase 1
2. Add new token types (vocabulary extension)
3. Add new transformer layers on top
4. New output heads for expanded task

---

**Status:** Ready for implementation.
