"""
Progression System - XP and Level Management

RPG-style experience point system with geometric leveling.
Designed for pedagogical tracking across any curriculum.

Key concepts:
- XP accumulates per-topic
- Levels use geometric thresholds (N² × 10)
- XP gains scale inversely with level (1/level)
- Penalties are NOT scaled (always hurt equally)

Usage:
    progression = ProgressionSystem(n_topics=10)
    progression.award_xp(topic_idx=0, amount=5)  # +5 base XP
    level = progression.get_level(0)
    xp, level, progress, high = progression.get_xp_info(0)
"""

import torch
import torch.nn as nn


class ProgressionSystem(nn.Module):
    """
    Per-topic XP and level tracking with geometric scaling.

    Standalone module that can be attached to any learning system.
    """

    def __init__(self, n_topics=10, max_level=10):
        super().__init__()
        self.n_topics = n_topics
        self.max_level = max_level

        # Per-topic XP tracking
        self.register_buffer('topic_xp', torch.zeros(n_topics))
        self.register_buffer('topic_xp_high', torch.zeros(n_topics))  # High water mark

    @staticmethod
    def xp_threshold(level):
        """
        XP required to reach a level.

        Geometric scaling: level N requires N² × 10 total XP
        L1=10, L2=40, L3=90, L4=160, L5=250, L6=360, L7=490, L8=640, L9=810, L10=1000

        Early levels are quick, mastery requires sustained proof.
        """
        return level * level * 10

    def get_level(self, topic_idx):
        """Get current level for a topic based on XP."""
        xp = self.topic_xp[topic_idx].item()
        for level in range(self.max_level, 0, -1):
            if xp >= self.xp_threshold(level):
                return level
        return 0

    def get_xp_info(self, topic_idx):
        """
        Get detailed XP info for a topic.

        Returns: (current_xp, level, progress_to_next, high_water_mark)
        """
        xp = self.topic_xp[topic_idx].item()
        xp_high = self.topic_xp_high[topic_idx].item()
        level = self.get_level(topic_idx)

        # Progress to next level
        if level >= self.max_level:
            progress = 1.0  # Maxed out
        else:
            current_threshold = self.xp_threshold(level)
            next_threshold = self.xp_threshold(level + 1)
            range_size = next_threshold - current_threshold
            progress = (xp - current_threshold) / range_size if range_size > 0 else 0

        return xp, level, progress, xp_high

    def award_xp(self, topic_idx, amount):
        """
        Award (or deduct) XP for a topic, scaled by level.

        Formula: actual_xp = base_amount / max(1, current_level)

        Pure RPG-style level scaling:
        - L1 on any topic: full XP (it's hard for you)
        - L5 on any topic: 1/5th XP (it's trivial now)
        - Penalties are NOT scaled (always hurt the same)

        XP cannot go below 0.
        """
        with torch.no_grad():
            level = max(1, self.get_level(topic_idx))

            # Scale positive XP by 1/level; penalties stay fixed
            if amount > 0:
                scaled_amount = amount / level
            else:
                scaled_amount = amount  # Penalties not scaled

            self.topic_xp[topic_idx] += scaled_amount
            self.topic_xp[topic_idx].clamp_(min=0)

            # Track high water mark
            if self.topic_xp[topic_idx] > self.topic_xp_high[topic_idx]:
                self.topic_xp_high[topic_idx] = self.topic_xp[topic_idx].clone()

    def get_total_xp(self):
        """Get total XP across all topics."""
        return self.topic_xp.sum().item()

    def get_average_level(self, active_mask=None):
        """
        Get average level across topics.

        Args:
            active_mask: Optional bool tensor. If provided, only average active topics.
        """
        if active_mask is None:
            active_mask = torch.ones(self.n_topics, dtype=torch.bool, device=self.topic_xp.device)

        if not active_mask.any():
            return 0.0

        levels = torch.tensor([self.get_level(i) for i in range(self.n_topics)],
                              device=self.topic_xp.device)
        return levels[active_mask].float().mean().item()

    def expand(self, new_size):
        """Expand to accommodate more topics."""
        if new_size <= self.n_topics:
            return

        with torch.no_grad():
            device = self.topic_xp.device

            new_xp = torch.zeros(new_size, device=device)
            new_xp_high = torch.zeros(new_size, device=device)

            new_xp[:self.n_topics] = self.topic_xp
            new_xp_high[:self.n_topics] = self.topic_xp_high

            del self.topic_xp
            del self.topic_xp_high

            self.register_buffer('topic_xp', new_xp)
            self.register_buffer('topic_xp_high', new_xp_high)

            self.n_topics = new_size


# Alias for backwards compatibility
TopicTracker = ProgressionSystem


# XP award constants - standardized across phases
class XPRewards:
    """Standard XP reward amounts."""
    CORRECT_BASE = 1.0        # Basic correct answer
    CREATIVE_CORRECT = 5.0    # Validated creative insight
    CREATIVE_WRONG = -3.0     # Overconfidence penalty
    STREAK_DIVISOR = 5        # streak_length // 5 = bonus XP
    VALIDATION = 0.0          # Asking for help is neutral
    SPONTANEOUS = 0.0         # Spontaneous share is neutral
