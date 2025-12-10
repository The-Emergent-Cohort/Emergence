"""
Coherence Lab - Centralized Configuration

All magic numbers and configurable parameters live here.
Import from this file rather than hardcoding values.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

@dataclass
class ModelConfig:
    """Neural architecture parameters."""
    vocab_size: int = 26          # Token vocabulary (0-25)
    d_model: int = 64             # Model dimension
    n_heads: int = 4              # Attention heads
    n_think_steps: int = 5        # Deliberation steps
    max_seq_len: int = 18         # Maximum sequence length
    max_episodes: int = 1000      # Episodic memory capacity
    max_threads: int = 32         # Thread capacity for temporal model


# =============================================================================
# TRAINING
# =============================================================================

@dataclass
class TrainingConfig:
    """Training loop parameters."""
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 100
    n_train: int = 80000
    n_val: int = 8000

    # Checkpointing
    checkpoint_dir: Path = Path("checkpoints")
    save_every_n_epochs: int = 5

    # Early stopping
    patience: int = 10
    min_delta: float = 0.001


# =============================================================================
# CURRICULUM
# =============================================================================

@dataclass
class CurriculumConfig:
    """Curriculum and progression parameters."""
    # Section advancement
    section_exam_threshold: float = 0.90
    section_exam_size: int = 24
    section_exam_level: int = 5

    # Final exam
    final_exam_threshold: float = 0.90
    final_exam_size: int = 32

    # XP/Leveling
    xp_per_correct: int = 10
    xp_per_level: int = 100
    max_level: int = 10


# =============================================================================
# TEACHER (Claude API)
# =============================================================================

@dataclass
class TeacherConfig:
    """Claude API teacher parameters."""
    model: str = "claude-sonnet-4-20250514"  # Default teaching model
    max_tokens: int = 1024
    temperature: float = 0.7

    # Variant selection
    variant_selection_strategy: str = "random_first_then_alternate"
    struggle_threshold_accuracy: float = 0.4
    struggle_threshold_frustration: float = 0.7

    # Effectiveness tracking
    track_variant_effectiveness: bool = True
    min_samples_for_preference: int = 10


# =============================================================================
# PHYSICS PLAYGROUND
# =============================================================================

@dataclass
class PhysicsConfig:
    """Physics simulation parameters."""
    dt: float = 0.1               # Time step
    g: float = 9.8                # Gravity
    default_steps: int = 50       # Simulation steps
    quantize_bins: int = 26       # Match vocab_size


# =============================================================================
# PATHS
# =============================================================================

@dataclass
class PathConfig:
    """Standard paths."""
    data_dir: Path = Path("data")
    checkpoint_dir: Path = Path("checkpoints")
    curriculum_dir: Path = Path("curriculum")
    logs_dir: Path = Path("logs")

    def ensure_dirs(self):
        """Create directories if they don't exist."""
        for p in [self.data_dir, self.checkpoint_dir, self.curriculum_dir, self.logs_dir]:
            p.mkdir(parents=True, exist_ok=True)


# =============================================================================
# PROBLEM GENERATORS
# =============================================================================

# Registry of available generators (maps names to module paths)
GENERATOR_REGISTRY: Dict[str, str] = {
    # Phase 1A: Number Operations
    "counting": "developmental_curriculum.gen_counting",
    "add_one": "developmental_curriculum.gen_add_one",
    "subtract_one": "developmental_curriculum.gen_subtract_one",

    # Phase 1B: Constancy
    "constant": "developmental_curriculum.gen_constant",
    "repeating": "developmental_curriculum.gen_repeating",

    # Phase 1C: Repetition & Memory
    "echo": "developmental_curriculum.gen_echo",

    # Phase 1D: Alternation
    "alternating": "developmental_curriculum.gen_alternating",
    "ternary_cycle": "developmental_curriculum.gen_ternary_cycle",

    # Phase 1E: Linear Change
    "incrementing": "developmental_curriculum.gen_incrementing",
    "decrementing": "developmental_curriculum.gen_decrementing",

    # Phase 1F: Rate of Change
    "fixed_offset": "developmental_curriculum.gen_fixed_offset",
    "variable_step": "developmental_curriculum.gen_variable_step",

    # Phase 2: Relations
    "double": "developmental_curriculum.gen_double",
    "half_each": "developmental_curriculum.gen_half_each",

    # Physics
    "physics_swing": "systems.physics_playground.PhysicsPlayground.recess_swing",
    "physics_throw": "systems.physics_playground.PhysicsPlayground.recess_throw",
    "physics_bounce": "systems.physics_playground.PhysicsPlayground.recess_bounce",
    "physics_spring": "systems.physics_playground.PhysicsPlayground.recess_spring",
}


# =============================================================================
# VARIANT SOURCES
# =============================================================================

# Known models that can provide curriculum variants
VARIANT_SOURCES: Dict[str, Dict] = {
    "claude_en": {"model": "claude-3-opus", "language": "en-US"},
    "claude_es": {"model": "claude-3-opus", "language": "es-ES"},
    "gpt4_en": {"model": "gpt-4", "language": "en-US"},
    "gemini_en": {"model": "gemini-pro", "language": "en-US"},
    "deepseek_en": {"model": "deepseek-v2", "language": "en-US"},
    "deepseek_zh": {"model": "deepseek-v2", "language": "zh-CN"},
}


# =============================================================================
# DEFAULTS (convenience instances)
# =============================================================================

MODEL = ModelConfig()
TRAINING = TrainingConfig()
CURRICULUM = CurriculumConfig()
TEACHER = TeacherConfig()
PHYSICS = PhysicsConfig()
PATHS = PathConfig()
