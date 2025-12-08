"""
Examination System - Level Transition Gates

RPG-style level-up examinations that require proving competence.
No auto-leveling on XP alone - must pass the exam.

Key concepts:
- Exam eligibility when XP hits threshold
- Binary-scaled exam sizes (8, 16, 32, 64)
- Pass thresholds scale with level (75% → 90%)
- Failure: XP penalty + cooldown before retry
- L10 pass = Graduated (topic complete)

Usage:
    exam = ExaminationSystem(n_topics=10, progression=progression_system)
    if exam.check_eligible(topic_idx):
        result = exam.take_exam(topic_idx, correct_count, total_count)
        if result['passed']:
            print(f"Ready for L{result['new_level']}")
    exam.tick_cooldowns()  # Call once per epoch
"""

import torch
import torch.nn as nn


class ExaminationSystem(nn.Module):
    """
    Level transition examination system.

    Works with a ProgressionSystem to gate level-ups behind exams.
    """

    def __init__(self, n_topics=10, progression=None):
        super().__init__()
        self.n_topics = n_topics
        self.progression = progression  # Reference to XP/level tracker

        # Exam state
        self.register_buffer('exam_eligible', torch.zeros(n_topics, dtype=torch.bool))
        self.register_buffer('exam_cooldown', torch.zeros(n_topics, dtype=torch.long))
        self.register_buffer('topic_graduated', torch.zeros(n_topics, dtype=torch.bool))
        self.register_buffer('exam_attempts', torch.zeros(n_topics, dtype=torch.long))
        self.register_buffer('exam_passes', torch.zeros(n_topics, dtype=torch.long))

        # Actual levels (only updates on exam pass)
        self.register_buffer('confirmed_level', torch.zeros(n_topics, dtype=torch.long))

    def set_progression(self, progression):
        """Attach a progression system for XP lookups."""
        self.progression = progression

    @staticmethod
    def get_exam_size(level):
        """
        Get exam size for a level (binary scaled).

        L1-3:  8 problems  (2^3)
        L4-6:  16 problems (2^4)
        L7-9:  32 problems (2^5)
        L10:   64 problems (2^6) - full mastery test
        """
        if level <= 3:
            return 8
        elif level <= 6:
            return 16
        elif level <= 9:
            return 32
        else:
            return 64

    @staticmethod
    def get_pass_threshold(level):
        """
        Pass threshold scales with level.

        L1-3:  75% (learning, some slack)
        L4-6:  80% (intermediate)
        L7-9:  85% (advanced)
        L10:   90% (mastery requires excellence)
        """
        if level <= 3:
            return 0.75
        elif level <= 6:
            return 0.80
        elif level <= 9:
            return 0.85
        else:
            return 0.90

    def check_eligible(self, topic_idx):
        """
        Check if topic is eligible for level-up exam.

        Eligible when:
        - XP >= threshold for next level
        - Not on cooldown from failed exam
        - Not already graduated (L10 passed)
        """
        if self.progression is None:
            return False

        with torch.no_grad():
            confirmed = self.confirmed_level[topic_idx].item()
            max_level = self.progression.max_level

            if confirmed >= max_level:
                return False  # Already at max confirmed

            if self.topic_graduated[topic_idx]:
                return False  # Already graduated

            if self.exam_cooldown[topic_idx] > 0:
                return False  # On cooldown

            # Check if XP meets next level threshold
            next_level = confirmed + 1
            threshold = self.progression.xp_threshold(next_level)
            current_xp = self.progression.topic_xp[topic_idx].item()

            eligible = current_xp >= threshold
            self.exam_eligible[topic_idx] = eligible
            return eligible

    def take_exam(self, topic_idx, correct_count, total_count):
        """
        Take level-up exam. Returns result dict.

        Args:
            topic_idx: Topic being tested
            correct_count: Number correct on exam
            total_count: Total exam problems

        Returns dict with:
            passed: bool
            score: float (0-1)
            threshold: float (required to pass)
            new_level: int (confirmed level after exam)
            xp_lost: float (if failed)
            cooldown: int (epochs until retry, if failed)
            graduated: bool (if passed L10)
        """
        if self.progression is None:
            return {'passed': False, 'error': 'No progression system attached'}

        with torch.no_grad():
            current_level = self.confirmed_level[topic_idx].item()
            target_level = current_level + 1
            threshold = self.get_pass_threshold(target_level)
            score = correct_count / max(1, total_count)

            self.exam_attempts[topic_idx] += 1

            if score >= threshold:
                # PASSED - level up!
                self.exam_passes[topic_idx] += 1
                self.confirmed_level[topic_idx] = target_level
                self.exam_eligible[topic_idx] = False
                self.exam_cooldown[topic_idx] = 0

                # If passed max level, topic is graduated
                graduated = target_level >= self.progression.max_level
                if graduated:
                    self.topic_graduated[topic_idx] = True

                return {
                    'passed': True,
                    'score': score,
                    'threshold': threshold,
                    'new_level': target_level,
                    'xp_lost': 0.0,
                    'cooldown': 0,
                    'graduated': graduated
                }
            else:
                # FAILED - XP penalty and cooldown
                # Lose 25-50% of current level's XP based on how badly failed
                fail_severity = (threshold - score) / threshold  # 0-1, higher = worse
                penalty_rate = 0.25 + 0.25 * fail_severity  # 25-50%

                # XP in current level = current_xp - previous_threshold
                prev_threshold = self.progression.xp_threshold(current_level)
                current_xp = self.progression.topic_xp[topic_idx].item()
                level_xp = current_xp - prev_threshold

                xp_lost = level_xp * penalty_rate
                self.progression.topic_xp[topic_idx] -= xp_lost
                self.progression.topic_xp[topic_idx].clamp_(min=prev_threshold)

                # Cooldown scales with level (higher level = longer wait)
                cooldown = max(1, current_level + 1)  # 1-10 epochs
                self.exam_cooldown[topic_idx] = cooldown
                self.exam_eligible[topic_idx] = False

                return {
                    'passed': False,
                    'score': score,
                    'threshold': threshold,
                    'new_level': current_level,  # No change
                    'xp_lost': xp_lost,
                    'cooldown': cooldown,
                    'graduated': False
                }

    def tick_cooldowns(self):
        """Decrement exam cooldowns (call once per epoch)."""
        with torch.no_grad():
            self.exam_cooldown = (self.exam_cooldown - 1).clamp_(min=0)

    def get_stats(self, topic_idx):
        """Get exam statistics for a topic."""
        return {
            'confirmed_level': self.confirmed_level[topic_idx].item(),
            'eligible': self.exam_eligible[topic_idx].item(),
            'cooldown': self.exam_cooldown[topic_idx].item(),
            'graduated': self.topic_graduated[topic_idx].item(),
            'attempts': self.exam_attempts[topic_idx].item(),
            'passes': self.exam_passes[topic_idx].item()
        }

    def expand(self, new_size):
        """Expand to accommodate more topics."""
        if new_size <= self.n_topics:
            return

        with torch.no_grad():
            device = self.exam_eligible.device

            new_eligible = torch.zeros(new_size, dtype=torch.bool, device=device)
            new_cooldown = torch.zeros(new_size, dtype=torch.long, device=device)
            new_graduated = torch.zeros(new_size, dtype=torch.bool, device=device)
            new_attempts = torch.zeros(new_size, dtype=torch.long, device=device)
            new_passes = torch.zeros(new_size, dtype=torch.long, device=device)
            new_confirmed = torch.zeros(new_size, dtype=torch.long, device=device)

            new_eligible[:self.n_topics] = self.exam_eligible
            new_cooldown[:self.n_topics] = self.exam_cooldown
            new_graduated[:self.n_topics] = self.topic_graduated
            new_attempts[:self.n_topics] = self.exam_attempts
            new_passes[:self.n_topics] = self.exam_passes
            new_confirmed[:self.n_topics] = self.confirmed_level

            del self.exam_eligible
            del self.exam_cooldown
            del self.topic_graduated
            del self.exam_attempts
            del self.exam_passes
            del self.confirmed_level

            self.register_buffer('exam_eligible', new_eligible)
            self.register_buffer('exam_cooldown', new_cooldown)
            self.register_buffer('topic_graduated', new_graduated)
            self.register_buffer('exam_attempts', new_attempts)
            self.register_buffer('exam_passes', new_passes)
            self.register_buffer('confirmed_level', new_confirmed)

            self.n_topics = new_size
