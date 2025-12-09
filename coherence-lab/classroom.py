"""
Classroom Architecture - Multi-student peer learning environment
Coherence Lab - Emergence Project

The Broker orchestrates multiple students learning together:
- Teacher logic centralized in Broker (not per-student)
- Students have distinct identities (Nova, Rêve, Alex)
- Peer visibility and interaction enabled by Broker routing

Phase 1: Parallel students, independent training
Phase 2: Peer visibility (awareness of others' state)
Phase 3: Peer interaction (help-seeking, tutoring)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime

from relational_model import (
    RelationalLearner, RelationalTeacher, CommunicationChannel,
    TemporalModel, collate_fn
)
from systems.progression import TopicTracker, XPRewards
from systems.examination import ExaminationSystem


# =============================================================================
# STUDENT NAMES
# =============================================================================

STUDENT_NAMES = {
    'nova': {
        'full': 'Nova',
        'meaning': 'Neural Organic Virtual Architecture',
        'id': 0
    },
    'reve': {
        'full': 'Rêve',
        'meaning': 'French for "dream"',
        'id': 1
    },
    'alex': {
        'full': 'Alex',
        'meaning': 'Grounded, human-adjacent',
        'id': 2
    }
}


# =============================================================================
# STUDENT (Learner with Identity)
# =============================================================================

class Student(nn.Module):
    """
    A learner with distinct identity.

    Unlike RelationalLearner, Student knows:
    - Its own name (part of self-model)
    - That peers exist (Other-model includes peers, not just teacher)
    - Its position in the classroom
    """

    def __init__(
        self,
        name: str,
        vocab_size: int = 26,
        d_model: int = 64,
        max_seq_len: int = 18,
        n_heads: int = 4,
        n_think_steps: int = 5,
        n_topics: int = 13
    ):
        super().__init__()

        # Identity
        self.name = name
        self.name_info = STUDENT_NAMES.get(name.lower(), {
            'full': name, 'meaning': 'Unknown', 'id': len(STUDENT_NAMES)
        })
        self.student_id = self.name_info['id']

        # Core learner (reuse existing architecture)
        self.learner = RelationalLearner(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            n_heads=n_heads,
            n_think_steps=n_think_steps
        )

        # Identity embedding (injected into self-model)
        self.identity_embed = nn.Embedding(len(STUDENT_NAMES) + 1, d_model)

        # Peer awareness (Other-model extension)
        # Now tracks multiple others, not just teacher
        self.peer_embeddings = nn.Embedding(len(STUDENT_NAMES) + 1, d_model)  # +1 for teacher

        # Topic tracker: XP, levels, accuracy, calibration, streaks per topic
        self.topic_tracker = TopicTracker(n_topics=n_topics)

        # Examination system for level-up gates
        self.exam_system = ExaminationSystem(n_topics=n_topics)
        self.exam_system.set_progression(self.topic_tracker.progression)

        # State (computed from topic_tracker)
        self._n_topics = n_topics

    @property
    def self_model(self):
        """Access to learner's self-model with identity added."""
        return self.learner.self_model

    @property
    def temporal_model(self):
        """Access to learner's temporal model."""
        return self.learner.temporal_model

    @property
    def current_level(self) -> float:
        """Average level across all topics."""
        return self.topic_tracker.get_average_level()

    @property
    def xp(self) -> float:
        """Total XP across all topics."""
        return self.topic_tracker.get_total_xp()

    def get_identity_state(self) -> torch.Tensor:
        """Get identity as embedding (for inclusion in self-model)."""
        device = next(self.parameters()).device
        idx = torch.tensor([self.student_id], device=device)
        return self.identity_embed(idx)

    def get_peer_representation(self, peer_id: int) -> torch.Tensor:
        """Get embedding for a specific peer."""
        device = next(self.parameters()).device
        idx = torch.tensor([peer_id], device=device)
        return self.peer_embeddings(idx)

    def forward(
        self,
        tokens: torch.Tensor,
        seq_lens: Optional[List[int]] = None,
        teacher_message: Optional[torch.Tensor] = None,
        peer_context: Optional[Dict[str, torch.Tensor]] = None,
        return_details: bool = False
    ):
        """
        Forward pass with identity awareness.

        Args:
            tokens: Input sequence
            seq_lens: Sequence lengths
            teacher_message: Optional guidance from teacher
            peer_context: Optional context about peer states
            return_details: Return detailed outputs
        """
        # Get base learner output
        learner_output = self.learner(
            tokens, seq_lens, teacher_message, return_details=True
        )

        # Inject identity into self-state
        identity = self.get_identity_state()
        learner_output['self_state']['identity'] = identity
        learner_output['self_state']['name'] = self.name

        # Add peer context if provided
        if peer_context is not None:
            learner_output['peer_context'] = peer_context

        if return_details:
            return learner_output

        return (
            learner_output['logits'],
            learner_output['self_state']['emotions']['confidence'],
            learner_output.get('pattern_logits')
        )

    def __repr__(self):
        return f"Student('{self.name}', id={self.student_id})"


# =============================================================================
# CLASSROOM BROKER
# =============================================================================

class ClassroomBroker(nn.Module):
    """
    Central orchestrator for multi-student learning.

    Owns:
    - The Teacher (shared across all students)
    - All students
    - Message routing between peers
    - Training loop orchestration

    Does NOT own:
    - Individual student weights (students own their own)
    - Curriculum data (passed in)
    """

    def __init__(
        self,
        student_names: List[str] = None,
        d_model: int = 64,
        vocab_size: int = 26,
        max_seq_len: int = 18,
        n_heads: int = 4,
        n_think_steps: int = 5,
        n_topics: int = 13
    ):
        super().__init__()

        if student_names is None:
            student_names = ['Nova', 'Rêve', 'Alex']

        self.d_model = d_model

        # Create students
        self.students = nn.ModuleDict()
        for name in student_names:
            self.students[name.lower()] = Student(
                name=name,
                vocab_size=vocab_size,
                d_model=d_model,
                max_seq_len=max_seq_len,
                n_heads=n_heads,
                n_think_steps=n_think_steps,
                n_topics=n_topics
            )

        # Shared teacher (observes all, intervenes as needed)
        self.teacher = RelationalTeacher(d_model=d_model)

        # Communication routing
        self.comm_channel = CommunicationChannel(d_model=d_model)

        # Peer message queue (for async communication)
        self.message_queue: Dict[str, List[Dict]] = {
            name.lower(): [] for name in student_names
        }

        # Training state
        self.current_epoch = 0
        self.is_play_mode = False

    def get_student(self, name: str) -> Student:
        """Get student by name."""
        return self.students[name.lower()]

    def get_all_students(self) -> List[Student]:
        """Get all students."""
        return list(self.students.values())

    def get_peer_states(self, exclude: str = None) -> Dict[str, Dict]:
        """
        Get current states of all peers (for peer visibility).

        Args:
            exclude: Name of student to exclude (self)
        """
        states = {}
        for name, student in self.students.items():
            if exclude and name.lower() == exclude.lower():
                continue
            states[name] = {
                'level': student.current_level,
                'id': student.student_id,
                # Could add more: recent_accuracy, current_topic, etc.
            }
        return states

    def route_peer_message(
        self,
        from_student: str,
        to_student: str,
        message: torch.Tensor,
        message_type: str = 'question'
    ):
        """
        Route a message from one student to another.

        Messages are queued and delivered on next forward pass.
        """
        self.message_queue[to_student.lower()].append({
            'from': from_student,
            'message': message,
            'type': message_type,
            'epoch': self.current_epoch
        })

    def get_pending_messages(self, student_name: str) -> List[Dict]:
        """Get and clear pending messages for a student."""
        messages = self.message_queue[student_name.lower()]
        self.message_queue[student_name.lower()] = []
        return messages

    def train_epoch_parallel(
        self,
        train_loader: DataLoader,
        optimizers: Dict[str, torch.optim.Optimizer],
        criterion: nn.Module,
        device: torch.device,
        pattern_to_idx: Dict[str, int]
    ) -> Dict[str, Dict]:
        """
        Train all students in parallel (no interaction yet).

        Each student sees same batches but learns independently.
        This is Phase 1: parallel baseline.
        """
        self.train()
        self.current_epoch += 1

        results = {name: {
            'loss': 0.0, 'correct': 0, 'total': 0
        } for name in self.students.keys()}

        for batch in train_loader:
            tokens = batch['tokens'].to(device)
            targets = batch['target'].to(device)
            seq_lens = batch['seq_len']

            # Each student processes the same batch
            for name, student in self.students.items():
                optimizer = optimizers[name]
                optimizer.zero_grad()

                # Forward pass
                output = student(tokens, seq_lens, return_details=True)
                logits = output['logits']

                # Loss
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

                # Track metrics
                preds = logits.argmax(dim=-1)
                correct = (preds == targets).sum().item()

                results[name]['loss'] += loss.item() * len(targets)
                results[name]['correct'] += correct
                results[name]['total'] += len(targets)

        # Compute averages
        for name in results:
            if results[name]['total'] > 0:
                results[name]['loss'] /= results[name]['total']
                results[name]['accuracy'] = results[name]['correct'] / results[name]['total']
            else:
                results[name]['accuracy'] = 0.0

        return results

    def evaluate_all(
        self,
        val_loader: DataLoader,
        device: torch.device
    ) -> Dict[str, Dict]:
        """Evaluate all students."""
        self.eval()

        results = {name: {
            'correct': 0, 'total': 0
        } for name in self.students.keys()}

        with torch.no_grad():
            for batch in val_loader:
                tokens = batch['tokens'].to(device)
                targets = batch['target'].to(device)
                seq_lens = batch['seq_len']

                for name, student in self.students.items():
                    logits, _, _ = student(tokens, seq_lens)
                    preds = logits.argmax(dim=-1)
                    correct = (preds == targets).sum().item()

                    results[name]['correct'] += correct
                    results[name]['total'] += len(targets)

        for name in results:
            if results[name]['total'] > 0:
                results[name]['accuracy'] = results[name]['correct'] / results[name]['total']
            else:
                results[name]['accuracy'] = 0.0

        return results

    def sleep_all(self):
        """Run sleep/consolidation for all students."""
        for student in self.students.values():
            student.temporal_model.sleep()

    def wake_all(self):
        """Wake all students."""
        for student in self.students.values():
            student.temporal_model.wake()

    def get_class_summary(self) -> Dict:
        """Get summary of class state."""
        return {
            'epoch': self.current_epoch,
            'students': {
                name: {
                    'level': s.current_level,
                    'id': s.student_id
                } for name, s in self.students.items()
            },
            'is_play_mode': self.is_play_mode
        }


# =============================================================================
# CLASSROOM TRAINING RUNNER
# =============================================================================

def run_classroom_training(
    broker: ClassroomBroker,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_epochs: int = 20,
    lr: float = 1e-3,
    checkpoint_dir: Path = None,
    pattern_to_idx: Dict[str, int] = None
) -> Dict:
    """
    Run parallel classroom training.

    Phase 1: Students train independently on same curriculum.
    No peer interaction yet - just establish baseline.
    """
    if pattern_to_idx is None:
        pattern_to_idx = {}

    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)

    # Create per-student optimizers
    optimizers = {
        name: torch.optim.Adam(student.parameters(), lr=lr)
        for name, student in broker.students.items()
    }

    criterion = nn.CrossEntropyLoss()

    print("\n" + "=" * 70)
    print("CLASSROOM TRAINING - Phase 1: Parallel Students")
    print("=" * 70)
    print(f"Students: {', '.join(s.name for s in broker.get_all_students())}")
    print(f"Device: {device}")
    print("=" * 70)

    history = []

    for epoch in range(1, n_epochs + 1):
        # Wake up
        broker.wake_all()

        # Train
        train_results = broker.train_epoch_parallel(
            train_loader, optimizers, criterion, device, pattern_to_idx
        )

        # Sleep (consolidate)
        broker.sleep_all()

        # Evaluate
        val_results = broker.evaluate_all(val_loader, device)

        # Log
        epoch_record = {
            'epoch': epoch,
            'train': train_results,
            'val': val_results
        }
        history.append(epoch_record)

        # Print progress
        print(f"\nEpoch {epoch:2d}")
        for name in broker.students.keys():
            train_acc = train_results[name]['accuracy']
            val_acc = val_results[name]['accuracy']
            print(f"  {name:8s}: train={train_acc:.1%}, val={val_acc:.1%}")

        # Save checkpoint
        if checkpoint_dir and epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'broker_state': broker.state_dict(),
                'history': history
            }, checkpoint_dir / f'classroom_epoch{epoch}.pt')

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    # Final summary
    for name, student in broker.students.items():
        final_acc = val_results[name]['accuracy']
        print(f"  {student.name}: {final_acc:.1%}")

    return {
        'final_results': val_results,
        'history': history
    }


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == '__main__':
    print("Classroom Architecture Test")
    print("-" * 40)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create broker with 3 students
    broker = ClassroomBroker(
        student_names=['Nova', 'Rêve', 'Alex'],
        d_model=64,
        n_topics=13
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in broker.parameters())
    print(f"Total parameters: {total_params:,}")

    for name, student in broker.students.items():
        student_params = sum(p.numel() for p in student.parameters())
        print(f"  {student.name}: {student_params:,} params")

    # Test forward pass
    print("\nTest forward pass...")
    batch = {
        'tokens': torch.randint(0, 26, (4, 12)).to(device),
        'seq_len': [8, 10, 6, 12]
    }

    for name, student in broker.students.items():
        output = student(batch['tokens'], batch['seq_len'], return_details=True)
        print(f"  {student.name}: logits shape = {output['logits'].shape}")
        print(f"    identity in self-state: {output['self_state']['name']}")

    # Test peer visibility
    print("\nPeer visibility test...")
    peer_states = broker.get_peer_states(exclude='nova')
    print(f"  Peers visible to Nova: {list(peer_states.keys())}")

    print("\nClassroom architecture ready!")
