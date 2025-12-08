"""
Integrated Phase 1: Multi-Task Training Loop
Coherence Lab - Emergence Project

Trains the integrated model on all 4 tasks simultaneously.
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import argparse
from datetime import datetime

try:
    from phase1_integrated_model import IntegratedPhase1Model
    from phase1_integrated_data import IntegratedPhase1DataGenerator
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from phase1_integrated_model import IntegratedPhase1Model
    from phase1_integrated_data import IntegratedPhase1DataGenerator


class IntegratedDataset(Dataset):
    """Dataset for integrated Phase 1 multi-task training."""

    def __init__(self, data_path, max_seq_len=12):
        self.max_seq_len = max_seq_len

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

        seq = ex['sequence']
        seq_len = ex['seq_len']

        # Pad sequence
        padded = seq + [0] * (self.max_seq_len - len(seq))

        # Token property labels: [seq_len, 6]
        token_prop_labels = torch.zeros(self.max_seq_len, 6)
        for tp in ex['tasks']['token_properties']:
            pos = tp['pos']
            if pos < self.max_seq_len:
                token_prop_labels[pos] = torch.tensor(tp['labels'], dtype=torch.float)

        # Position property labels: [seq_len, 4]
        pos_prop_labels = torch.zeros(self.max_seq_len, 4)
        for pp in ex['tasks']['position_properties']:
            pos = pp['pos']
            if pos < self.max_seq_len:
                pos_prop_labels[pos] = torch.tensor(pp['labels'], dtype=torch.float)

        # Token relations: list of (pos1, pos2, same)
        token_relations = []
        for tr in ex['tasks']['token_relations']:
            if tr['pos1'] < self.max_seq_len and tr['pos2'] < self.max_seq_len:
                token_relations.append({
                    'pos1': tr['pos1'],
                    'pos2': tr['pos2'],
                    'same': tr['same']
                })

        # Position relations: list of (pos1, pos2, same)
        position_relations = []
        for pr in ex['tasks']['position_relations']:
            if pr['pos1'] < self.max_seq_len and pr['pos2'] < self.max_seq_len:
                position_relations.append({
                    'pos1': pr['pos1'],
                    'pos2': pr['pos2'],
                    'same': pr['same']
                })

        return {
            'tokens': torch.tensor(padded[:self.max_seq_len], dtype=torch.long),
            'seq_len': seq_len,
            'token_prop_labels': token_prop_labels,
            'pos_prop_labels': pos_prop_labels,
            'token_relations': token_relations,
            'position_relations': position_relations
        }


def collate_fn(batch):
    """Custom collate for variable-length relation pairs."""
    tokens = torch.stack([b['tokens'] for b in batch])
    seq_lens = [b['seq_len'] for b in batch]
    token_prop_labels = torch.stack([b['token_prop_labels'] for b in batch])
    pos_prop_labels = torch.stack([b['pos_prop_labels'] for b in batch])

    # Relations as lists (variable length per example)
    token_relations = [b['token_relations'] for b in batch]
    position_relations = [b['position_relations'] for b in batch]

    return {
        'tokens': tokens,
        'seq_lens': seq_lens,
        'token_prop_labels': token_prop_labels,
        'pos_prop_labels': pos_prop_labels,
        'token_relations': token_relations,
        'position_relations': position_relations
    }


def compute_multi_task_loss(outputs, batch, device, loss_weights=None):
    """
    Compute combined loss across all 4 tasks.

    Returns dict with individual losses and combined loss.
    """
    if loss_weights is None:
        loss_weights = {
            'token_properties': 1.0,
            'token_relations': 1.0,
            'position_properties': 1.0,
            'position_relations': 1.0
        }

    bce = nn.BCEWithLogitsLoss()
    losses = {}

    # Task 1: Token properties (multi-label BCE)
    token_prop_pred = outputs['token_properties']  # [batch, seq_len, 6]
    token_prop_target = batch['token_prop_labels'].to(device)

    # Mask to only compute loss for valid positions
    batch_size, max_seq_len, _ = token_prop_pred.shape
    mask = torch.zeros(batch_size, max_seq_len, device=device)
    for i, sl in enumerate(batch['seq_lens']):
        mask[i, :sl] = 1.0

    # Expand mask for all 6 labels
    mask = mask.unsqueeze(-1).expand_as(token_prop_pred)
    losses['token_properties'] = (bce(token_prop_pred, token_prop_target) * mask).sum() / mask.sum()

    # Task 3: Position properties (multi-label BCE)
    pos_prop_pred = outputs['position_properties']  # [batch, seq_len, 4]
    pos_prop_target = batch['pos_prop_labels'].to(device)

    mask = torch.zeros(batch_size, max_seq_len, device=device)
    for i, sl in enumerate(batch['seq_lens']):
        mask[i, :sl] = 1.0
    mask = mask.unsqueeze(-1).expand_as(pos_prop_pred)
    losses['position_properties'] = (bce(pos_prop_pred, pos_prop_target) * mask).sum() / mask.sum()

    # Task 2: Token relations (binary BCE)
    if 'token_relations' in outputs:
        token_rel_preds = outputs['token_relations']
        token_rel_targets = batch['token_relations']

        all_preds = []
        all_targets = []
        for b_idx, (preds, targets) in enumerate(zip(token_rel_preds, token_rel_targets)):
            if len(targets) > 0 and len(preds) > 0:
                for p_idx, t in enumerate(targets):
                    if p_idx < len(preds):
                        all_preds.append(preds[p_idx])
                        all_targets.append(t['same'])

        if all_preds:
            all_preds = torch.stack(all_preds).to(device)
            all_targets = torch.tensor(all_targets, dtype=torch.float, device=device)
            losses['token_relations'] = bce(all_preds, all_targets)
        else:
            losses['token_relations'] = torch.tensor(0.0, device=device)

    # Task 4: Position relations (binary BCE)
    if 'position_relations' in outputs:
        pos_rel_preds = outputs['position_relations']
        pos_rel_targets = batch['position_relations']

        all_preds = []
        all_targets = []
        for b_idx, (preds, targets) in enumerate(zip(pos_rel_preds, pos_rel_targets)):
            if len(targets) > 0 and len(preds) > 0:
                for p_idx, t in enumerate(targets):
                    if p_idx < len(preds):
                        all_preds.append(preds[p_idx])
                        all_targets.append(t['same'])

        if all_preds:
            all_preds = torch.stack(all_preds).to(device)
            all_targets = torch.tensor(all_targets, dtype=torch.float, device=device)
            losses['position_relations'] = bce(all_preds, all_targets)
        else:
            losses['position_relations'] = torch.tensor(0.0, device=device)

    # Combined weighted loss
    total = sum(loss_weights[k] * losses[k] for k in losses)
    losses['total'] = total

    return losses


def compute_accuracies(outputs, batch, device):
    """Compute accuracy for each task."""
    accuracies = {}

    # Task 1: Token properties
    token_prop_pred = (torch.sigmoid(outputs['token_properties']) > 0.5).float()
    token_prop_target = batch['token_prop_labels'].to(device)

    batch_size, max_seq_len, _ = token_prop_pred.shape
    correct = 0
    total = 0
    for i, sl in enumerate(batch['seq_lens']):
        correct += (token_prop_pred[i, :sl] == token_prop_target[i, :sl]).float().sum().item()
        total += sl * 6  # 6 labels per position

    accuracies['token_properties'] = correct / total if total > 0 else 0

    # Task 3: Position properties
    pos_prop_pred = (torch.sigmoid(outputs['position_properties']) > 0.5).float()
    pos_prop_target = batch['pos_prop_labels'].to(device)

    correct = 0
    total = 0
    for i, sl in enumerate(batch['seq_lens']):
        correct += (pos_prop_pred[i, :sl] == pos_prop_target[i, :sl]).float().sum().item()
        total += sl * 4  # 4 labels per position

    accuracies['position_properties'] = correct / total if total > 0 else 0

    # Task 2: Token relations
    if 'token_relations' in outputs:
        token_rel_preds = outputs['token_relations']
        token_rel_targets = batch['token_relations']

        correct = 0
        total = 0
        for preds, targets in zip(token_rel_preds, token_rel_targets):
            if len(targets) > 0 and len(preds) > 0:
                for p_idx, t in enumerate(targets):
                    if p_idx < len(preds):
                        pred = (torch.sigmoid(preds[p_idx]) > 0.5).float().item()
                        if pred == t['same']:
                            correct += 1
                        total += 1

        accuracies['token_relations'] = correct / total if total > 0 else 0

    # Task 4: Position relations
    if 'position_relations' in outputs:
        pos_rel_preds = outputs['position_relations']
        pos_rel_targets = batch['position_relations']

        correct = 0
        total = 0
        for preds, targets in zip(pos_rel_preds, pos_rel_targets):
            if len(targets) > 0 and len(preds) > 0:
                for p_idx, t in enumerate(targets):
                    if p_idx < len(preds):
                        pred = (torch.sigmoid(preds[p_idx]) > 0.5).float().item()
                        if pred == t['same']:
                            correct += 1
                        total += 1

        accuracies['position_relations'] = correct / total if total > 0 else 0

    return accuracies


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_losses = {'total': 0, 'token_properties': 0, 'token_relations': 0,
                    'position_properties': 0, 'position_relations': 0}
    total_accs = {'token_properties': 0, 'token_relations': 0,
                  'position_properties': 0, 'position_relations': 0}
    n_batches = 0

    for batch in dataloader:
        tokens = batch['tokens'].to(device)

        # Build task info for pairs
        task_info = {
            'token_pairs': [[(r['pos1'], r['pos2']) for r in rels]
                           for rels in batch['token_relations']],
            'position_pairs': [[(r['pos1'], r['pos2']) for r in rels]
                              for rels in batch['position_relations']]
        }

        optimizer.zero_grad()
        outputs = model(tokens, task_info)
        losses = compute_multi_task_loss(outputs, batch, device)
        losses['total'].backward()
        optimizer.step()

        # Track losses
        for k in total_losses:
            total_losses[k] += losses[k].item()

        # Track accuracies
        accs = compute_accuracies(outputs, batch, device)
        for k in total_accs:
            total_accs[k] += accs[k]

        n_batches += 1

    # Average
    for k in total_losses:
        total_losses[k] /= n_batches
    for k in total_accs:
        total_accs[k] /= n_batches

    return total_losses, total_accs


def evaluate(model, dataloader, device):
    """Evaluate on validation set."""
    model.eval()
    total_losses = {'total': 0, 'token_properties': 0, 'token_relations': 0,
                    'position_properties': 0, 'position_relations': 0}
    total_accs = {'token_properties': 0, 'token_relations': 0,
                  'position_properties': 0, 'position_relations': 0}
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            tokens = batch['tokens'].to(device)

            task_info = {
                'token_pairs': [[(r['pos1'], r['pos2']) for r in rels]
                               for rels in batch['token_relations']],
                'position_pairs': [[(r['pos1'], r['pos2']) for r in rels]
                                  for rels in batch['position_relations']]
            }

            outputs = model(tokens, task_info)
            losses = compute_multi_task_loss(outputs, batch, device)

            for k in total_losses:
                total_losses[k] += losses[k].item()

            accs = compute_accuracies(outputs, batch, device)
            for k in total_accs:
                total_accs[k] += accs[k]

            n_batches += 1

    for k in total_losses:
        total_losses[k] /= n_batches
    for k in total_accs:
        total_accs[k] /= n_batches

    return total_losses, total_accs


def train(args):
    """Main training loop."""
    print("=" * 65)
    print("Integrated Phase 1: Multi-Task Foundation Training")
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    data_dir.mkdir(exist_ok=True)

    # Generate data if needed
    if not (data_dir / 'phase1_integrated_train.json').exists():
        print("\nGenerating integrated Phase 1 data...")
        generator = IntegratedPhase1DataGenerator(seed=42)
        train_data, val_data = generator.generate_dataset(n_examples=100000)

        with open(data_dir / 'phase1_integrated_train.json', 'w') as f:
            json.dump(train_data, f)
        with open(data_dir / 'phase1_integrated_val.json', 'w') as f:
            json.dump(val_data, f)

        stats = generator.get_stats(train_data)
        print(f"Generated {stats['n_examples']} examples")
        print(f"Average sequence length: {stats['avg_seq_len']:.1f}")
        print(f"Token relation balance: {stats['token_relation_balance']}")
        print(f"Position relation balance: {stats['position_relation_balance']}")

    # Load data
    train_dataset = IntegratedDataset(data_dir / 'phase1_integrated_train.json')
    val_dataset = IntegratedDataset(data_dir / 'phase1_integrated_val.json')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=0
    )

    print(f"\nTrain examples: {len(train_dataset)}")
    print(f"Val examples: {len(val_dataset)}")

    # Model
    model = IntegratedPhase1Model(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers
    ).to(device)

    params = model.count_parameters()
    print(f"\nModel parameters: {params['total']:,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': {},
        'val_acc': {}
    }
    for task in ['token_properties', 'token_relations', 'position_properties', 'position_relations']:
        history['train_acc'][task] = []
        history['val_acc'][task] = []

    best_val_acc = 0
    patience_counter = 0

    print("\nStarting training...")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        train_losses, train_accs = train_epoch(model, train_loader, optimizer, device)
        val_losses, val_accs = evaluate(model, val_loader, device)

        # Track history
        history['train_loss'].append(train_losses['total'])
        history['val_loss'].append(val_losses['total'])
        for task in train_accs:
            history['train_acc'][task].append(train_accs[task])
            history['val_acc'][task].append(val_accs[task])

        # Average accuracy across tasks
        avg_train_acc = sum(train_accs.values()) / len(train_accs)
        avg_val_acc = sum(val_accs.values()) / len(val_accs)

        print(f"Epoch {epoch:3d} | "
              f"Loss: {train_losses['total']:.4f}/{val_losses['total']:.4f} | "
              f"Acc: {avg_train_acc:.1%}/{avg_val_acc:.1%}")

        # Print per-task accuracies every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            print("  Per-task val acc:")
            for task, acc in sorted(val_accs.items()):
                short_name = task.replace('_', ' ').replace('properties', 'props').replace('relations', 'rels')
                print(f"    {short_name:20s}: {acc:.1%}")

        # Check if all tasks above threshold
        min_val_acc = min(val_accs.values())
        if min_val_acc > best_val_acc:
            best_val_acc = min_val_acc
            patience_counter = 0

            checkpoint_path = data_dir / 'phase1_integrated_checkpoint.pt'
            model.save_checkpoint(checkpoint_path, metadata={
                'epoch': epoch,
                'val_accuracies': val_accs,
                'min_val_acc': min_val_acc,
                'timestamp': datetime.now().isoformat()
            })
        else:
            patience_counter += 1

        # Check success criteria (95% on ALL tasks)
        if min_val_acc >= args.target_acc:
            print(f"\nTarget accuracy {args.target_acc:.0%} reached on ALL tasks!")
            break

        if patience_counter >= args.patience:
            print(f"\nEarly stopping after {args.patience} epochs without improvement")
            break

    # Final summary
    print("\n" + "=" * 65)
    print("Training Complete")
    print("=" * 65)
    print(f"Best minimum validation accuracy: {best_val_acc:.1%}")

    print("\nFinal per-task validation accuracy:")
    for task, acc in sorted(val_accs.items()):
        print(f"  {task:25s}: {acc:.1%}")

    # Save history
    history_path = data_dir / 'phase1_integrated_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nHistory saved to {history_path}")

    return best_val_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Integrated Phase 1')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Data directory')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Maximum epochs')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--target-acc', type=float, default=0.95,
                        help='Target accuracy for ALL tasks')
    parser.add_argument('--d-model', type=int, default=64,
                        help='Model dimension')
    parser.add_argument('--num-heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=4,
                        help='Number of transformer layers')

    args = parser.parse_args()
    train(args)
