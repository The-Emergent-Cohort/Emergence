"""
Student State Bridge - Extract student state for the teacher.

Bridges the RelationalSystem's internal state to human-readable
context that the Claude teacher can use for decisions.

Provides:
- Emotional state (confidence, frustration, curiosity)
- Topic progress (XP, levels, accuracy)
- Struggle detection
- Internalization level
- Formatted summaries for teacher context

Usage:
    bridge = StudentStateBridge(model, pattern_to_idx)
    state = bridge.get_full_state()
    struggling, reason = bridge.is_struggling("counting")
    context = bridge.format_for_teacher()
"""

try:
    import torch
except ImportError:
    torch = None  # Allow running without torch for testing

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from config import CURRICULUM as CURRICULUM_CONFIG


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EmotionalState:
    """Student's current emotional/motivational state."""
    confidence: float = 0.5
    frustration: float = 0.0
    curiosity: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        return {
            "confidence": self.confidence,
            "frustration": self.frustration,
            "curiosity": self.curiosity,
        }

    def __str__(self) -> str:
        return (
            f"confidence={self.confidence:.0%}, "
            f"frustration={self.frustration:.0%}, "
            f"curiosity={self.curiosity:.0%}"
        )


@dataclass
class TopicProgress:
    """Progress on a specific topic/pattern."""
    topic_name: str
    xp: int = 0
    level: int = 0
    confirmed_level: int = 0
    recent_accuracy: float = 0.0
    streak: int = 0
    epochs_on_topic: int = 0
    exam_attempts: int = 0
    exam_passes: int = 0

    @property
    def mastery_percent(self) -> float:
        """Rough mastery estimate (0-100%)."""
        return min(100, (self.confirmed_level / 10) * 100)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_name": self.topic_name,
            "xp": self.xp,
            "level": self.level,
            "confirmed_level": self.confirmed_level,
            "recent_accuracy": self.recent_accuracy,
            "streak": self.streak,
            "epochs_on_topic": self.epochs_on_topic,
            "exam_attempts": self.exam_attempts,
            "exam_passes": self.exam_passes,
            "mastery_percent": self.mastery_percent,
        }


@dataclass
class StruggleInfo:
    """Information about a detected struggle."""
    is_struggling: bool = False
    reason: str = ""
    severity: str = "none"  # none | mild | moderate | severe
    suggested_action: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_struggling": self.is_struggling,
            "reason": self.reason,
            "severity": self.severity,
            "suggested_action": self.suggested_action,
        }


# =============================================================================
# STUDENT STATE BRIDGE
# =============================================================================

class StudentStateBridge:
    """
    Extract and format student state for the teacher.

    Wraps RelationalSystem and provides clean interfaces
    for the teacher to understand the student's state.
    """

    def __init__(
        self,
        model,  # RelationalSystem
        pattern_to_idx: Dict[str, int],
        recent_window: int = 20,
    ):
        """
        Args:
            model: The RelationalSystem (student model)
            pattern_to_idx: Mapping from pattern names to indices
            recent_window: Number of recent results to consider for accuracy
        """
        self.model = model
        self.pattern_to_idx = pattern_to_idx
        self.idx_to_pattern = {v: k for k, v in pattern_to_idx.items()}
        self.recent_window = recent_window

        # Track recent results per topic for accuracy calculation
        self.recent_results: Dict[str, List[bool]] = {
            p: [] for p in pattern_to_idx
        }

    # =========================================================================
    # EMOTIONAL STATE
    # =========================================================================

    def get_emotional_state(self) -> EmotionalState:
        """
        Extract emotional state from the model's SelfModel.

        Falls back to neutral values if model doesn't have state.
        """
        try:
            # Get from SelfModel if available
            if hasattr(self.model, 'learner') and hasattr(self.model.learner, 'self_model'):
                self_model = self.model.learner.self_model

                # SelfModel stores these as nn.Linear layers that output scalars
                # We need a dummy input to get current state
                # For now, use last known values if tracked, else defaults
                if hasattr(self_model, '_last_emotions'):
                    emotions = self_model._last_emotions
                    def get_val(key, default):
                        v = emotions.get(key)
                        if v is None:
                            return default
                        return v.item() if hasattr(v, 'item') else float(v)
                    return EmotionalState(
                        confidence=get_val('confidence', 0.5),
                        frustration=get_val('frustration', 0.0),
                        curiosity=get_val('curiosity', 0.5),
                    )

            return EmotionalState()  # Defaults

        except Exception:
            return EmotionalState()

    def update_emotional_state(self, emotions: Dict[str, Any]):
        """
        Update cached emotional state (called after forward pass).

        Args:
            emotions: Dict with 'confidence', 'frustration', 'curiosity' tensors
        """
        try:
            if hasattr(self.model, 'learner') and hasattr(self.model.learner, 'self_model'):
                self.model.learner.self_model._last_emotions = emotions
        except Exception:
            pass

    # =========================================================================
    # TOPIC PROGRESS
    # =========================================================================

    def get_topic_progress(self, topic_name: str) -> TopicProgress:
        """
        Get progress for a specific topic.

        Pulls from TopicTracker if available.
        """
        progress = TopicProgress(topic_name=topic_name)

        try:
            if topic_name not in self.pattern_to_idx:
                return progress

            idx = self.pattern_to_idx[topic_name]

            # Try to get from TopicTracker
            if (hasattr(self.model, 'learner') and
                hasattr(self.model.learner, 'self_model') and
                hasattr(self.model.learner.self_model, 'topic_tracker')):

                tracker = self.model.learner.self_model.topic_tracker

                if hasattr(tracker, 'xp') and idx < len(tracker.xp):
                    progress.xp = int(tracker.xp[idx].item())

                if hasattr(tracker, 'levels') and idx < len(tracker.levels):
                    progress.level = int(tracker.levels[idx].item())

                if hasattr(tracker, 'confirmed_level') and idx < len(tracker.confirmed_level):
                    progress.confirmed_level = int(tracker.confirmed_level[idx].item())

                if hasattr(tracker, 'streak') and idx < len(tracker.streak):
                    progress.streak = int(tracker.streak[idx].item())

            # Calculate recent accuracy from our tracking
            recent = self.recent_results.get(topic_name, [])
            if recent:
                progress.recent_accuracy = sum(recent) / len(recent)

            progress.epochs_on_topic = len(recent)

        except Exception:
            pass

        return progress

    def get_all_progress(self) -> Dict[str, TopicProgress]:
        """Get progress for all topics."""
        return {
            topic: self.get_topic_progress(topic)
            for topic in self.pattern_to_idx
        }

    def record_result(self, topic_name: str, correct: bool):
        """
        Record a result for accuracy tracking.

        Args:
            topic_name: Which topic/pattern
            correct: Whether the response was correct
        """
        if topic_name not in self.recent_results:
            self.recent_results[topic_name] = []

        self.recent_results[topic_name].append(correct)

        # Keep only recent window
        if len(self.recent_results[topic_name]) > self.recent_window:
            self.recent_results[topic_name] = self.recent_results[topic_name][-self.recent_window:]

    # =========================================================================
    # STRUGGLE DETECTION
    # =========================================================================

    def is_struggling(self, topic_name: str) -> StruggleInfo:
        """
        Detect if student is struggling with a topic.

        Considers:
        - Recent accuracy
        - Frustration level
        - Plateau detection
        - Exam failures
        """
        progress = self.get_topic_progress(topic_name)
        emotions = self.get_emotional_state()

        # Check multiple indicators
        struggles = []
        severity_score = 0

        # Low accuracy + high frustration
        if progress.recent_accuracy < 0.4 and emotions.frustration > 0.6:
            struggles.append("low_accuracy_high_frustration")
            severity_score += 2

        # Very low accuracy alone
        if progress.recent_accuracy < 0.25:
            struggles.append("very_low_accuracy")
            severity_score += 2

        # Plateau (many attempts, no level progress)
        if progress.epochs_on_topic > 15 and progress.level < 3:
            struggles.append("plateau")
            severity_score += 1

        # Repeated exam failures
        if progress.exam_attempts > 2 and progress.exam_passes == 0:
            struggles.append("repeated_exam_failure")
            severity_score += 2

        # High frustration alone
        if emotions.frustration > 0.8:
            struggles.append("high_frustration")
            severity_score += 1

        # Losing streak
        recent = self.recent_results.get(topic_name, [])
        if len(recent) >= 5 and not any(recent[-5:]):
            struggles.append("losing_streak")
            severity_score += 2

        # Determine severity
        if severity_score == 0:
            severity = "none"
        elif severity_score <= 1:
            severity = "mild"
        elif severity_score <= 3:
            severity = "moderate"
        else:
            severity = "severe"

        # Suggest action
        if not struggles:
            action = "continue"
        elif "high_frustration" in struggles:
            action = "take_break_or_switch_to_play"
        elif "losing_streak" in struggles or "very_low_accuracy" in struggles:
            action = "try_different_variant"
        elif "plateau" in struggles:
            action = "revisit_prerequisites"
        else:
            action = "increase_scaffolding"

        return StruggleInfo(
            is_struggling=len(struggles) > 0,
            reason=", ".join(struggles) if struggles else "",
            severity=severity,
            suggested_action=action,
        )

    # =========================================================================
    # INTERNALIZATION
    # =========================================================================

    def get_internalization_level(self) -> float:
        """
        Get how internalized the teacher has become.

        Higher = student has stronger "inner guide"
        """
        try:
            if (hasattr(self.model, 'learner') and
                hasattr(self.model.learner, 'other_model')):
                other_model = self.model.learner.other_model
                if hasattr(other_model, 'internalization_level'):
                    return torch.sigmoid(other_model.internalization_level).item()
        except Exception:
            pass
        return 0.0

    def get_trust_level(self) -> float:
        """Get current trust in teacher."""
        try:
            if (hasattr(self.model, 'learner') and
                hasattr(self.model.learner, 'other_model')):
                other_model = self.model.learner.other_model
                if hasattr(other_model, 'trust'):
                    return torch.sigmoid(other_model.trust).item()
        except Exception:
            pass
        return 0.5

    # =========================================================================
    # FULL STATE
    # =========================================================================

    def get_full_state(self, topic_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get complete student state.

        Args:
            topic_name: If specified, include topic-specific info
        """
        state = {
            "emotional_state": self.get_emotional_state().to_dict(),
            "internalization_level": self.get_internalization_level(),
            "trust_level": self.get_trust_level(),
        }

        if topic_name:
            state["topic_progress"] = self.get_topic_progress(topic_name).to_dict()
            state["struggle_info"] = self.is_struggling(topic_name).to_dict()

        return state

    # =========================================================================
    # TEACHER CONTEXT FORMATTING
    # =========================================================================

    def format_for_teacher(
        self,
        topic_name: Optional[str] = None,
        include_all_topics: bool = False,
    ) -> str:
        """
        Format student state as text for Claude teacher context.

        Args:
            topic_name: Current topic being taught
            include_all_topics: Include progress on all topics
        """
        lines = ["## Student State\n"]

        # Emotional state
        emotions = self.get_emotional_state()
        lines.append("### Emotional State")
        lines.append(f"- Confidence: {emotions.confidence:.0%}")
        lines.append(f"- Frustration: {emotions.frustration:.0%}")
        lines.append(f"- Curiosity: {emotions.curiosity:.0%}")
        lines.append("")

        # Internalization
        int_level = self.get_internalization_level()
        trust = self.get_trust_level()
        lines.append("### Relationship")
        lines.append(f"- Trust in teacher: {trust:.0%}")
        lines.append(f"- Internalization: {int_level:.0%}")
        if int_level > 0.7:
            lines.append("  (Strong inner guide developing)")
        lines.append("")

        # Current topic progress
        if topic_name:
            progress = self.get_topic_progress(topic_name)
            struggle = self.is_struggling(topic_name)

            lines.append(f"### Current Topic: {topic_name}")
            lines.append(f"- Level: {progress.level}/10 (confirmed: {progress.confirmed_level})")
            lines.append(f"- XP: {progress.xp}")
            lines.append(f"- Recent accuracy: {progress.recent_accuracy:.0%}")
            if progress.streak > 0:
                lines.append(f"- Current streak: {progress.streak}")
            lines.append("")

            if struggle.is_struggling:
                lines.append("### ⚠️ Struggle Detected")
                lines.append(f"- Severity: {struggle.severity}")
                lines.append(f"- Reasons: {struggle.reason}")
                lines.append(f"- Suggested: {struggle.suggested_action}")
                lines.append("")

        # All topics if requested
        if include_all_topics:
            lines.append("### All Topic Progress")
            for name in sorted(self.pattern_to_idx.keys()):
                p = self.get_topic_progress(name)
                lines.append(f"- {name}: L{p.level} ({p.recent_accuracy:.0%} recent)")
            lines.append("")

        return "\n".join(lines)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    # Test without actual model
    pattern_to_idx = {
        "counting": 0,
        "add_one": 1,
        "alternating": 2,
    }

    # Create bridge with None model (uses defaults)
    bridge = StudentStateBridge(None, pattern_to_idx)

    # Simulate some results
    for _ in range(10):
        bridge.record_result("counting", True)
    for _ in range(8):
        bridge.record_result("counting", False)
    for _ in range(3):
        bridge.record_result("add_one", False)

    # Test outputs
    print("=== Emotional State ===")
    print(bridge.get_emotional_state())
    print()

    print("=== Topic Progress ===")
    print(bridge.get_topic_progress("counting").to_dict())
    print()

    print("=== Struggle Detection ===")
    print(bridge.is_struggling("counting").to_dict())
    print(bridge.is_struggling("add_one").to_dict())
    print()

    print("=== Formatted for Teacher ===")
    print(bridge.format_for_teacher("counting", include_all_topics=True))
