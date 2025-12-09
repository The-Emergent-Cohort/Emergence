"""
Coherence Lab - Reusable Systems

Modular components for pedagogically-grounded ML:
- progression: XP, levels, geometric thresholds
- examination: Level-up exams, pass/fail, cooldowns
- curriculum_advisor: LLM-guided training decisions
- logging: Standardized output formatting
"""

from .progression import ProgressionSystem, TopicTracker, XPRewards
from .examination import ExaminationSystem
from .logging import TrainingLogger
from .curriculum_advisor import CurriculumAdvisor, StudentState, TrainingAction, ControlLevel

__all__ = [
    'ProgressionSystem', 'TopicTracker', 'XPRewards',
    'ExaminationSystem',
    'CurriculumAdvisor', 'StudentState', 'TrainingAction', 'ControlLevel',
    'TrainingLogger'
]
