"""
Coherence Lab - Reusable Systems

Modular components for pedagogically-grounded ML:
- progression: XP, levels, geometric thresholds
- examination: Level-up exams, pass/fail, cooldowns
- logging: Standardized output formatting
- physics_playground: Shared physics simulation for recess
- student_files: Persistent storage for student work
- curriculum_advisor: LLM-guided pedagogical decisions
- school_day: Unified class workflow orchestration
"""

from .progression import ProgressionSystem
from .examination import ExaminationSystem
from .logging import TrainingLogger
from .physics_playground import PhysicsPlayground, PhysicsEpisode
from .student_files import StudentFileSystem, ClassLibrary
from .curriculum_advisor import CurriculumAdvisor, StudentState, TrainingAction
from .school_day import SchoolDay, StudentProfile, Period

__all__ = [
    # Core training systems
    'ProgressionSystem',
    'ExaminationSystem',
    'TrainingLogger',
    # Playground and persistence
    'PhysicsPlayground',
    'PhysicsEpisode',
    'StudentFileSystem',
    'ClassLibrary',
    # Curriculum management
    'CurriculumAdvisor',
    'StudentState',
    'TrainingAction',
    # School day orchestration
    'SchoolDay',
    'StudentProfile',
    'Period',
]
