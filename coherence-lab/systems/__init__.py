"""
Coherence Lab - Reusable Systems

Modular components for pedagogically-grounded ML:
- progression: XP, levels, geometric thresholds
- examination: Level-up exams, pass/fail, cooldowns
- logging: Standardized output formatting
"""

from .progression import ProgressionSystem, TopicTracker, XPRewards
from .examination import ExaminationSystem
from .logging import TrainingLogger

__all__ = ['ProgressionSystem', 'TopicTracker', 'XPRewards', 'ExaminationSystem', 'TrainingLogger']
