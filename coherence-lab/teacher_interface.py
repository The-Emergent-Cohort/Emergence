"""
Teacher Interface - Claude API teacher integration.

Bridges the Claude API teacher with curriculum and student state:
- Formats context for teaching decisions
- Calls Claude for variant selection and feedback
- Parses responses into actionable decisions
- Logs effectiveness for learning

Usage:
    teacher = TeacherInterface(curriculum_loader, state_bridge)
    decision = await teacher.get_teaching_decision(activity)
    feedback = await teacher.get_feedback(result)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    Anthropic = None
    ANTHROPIC_AVAILABLE = False

from curriculum_loader import CurriculumLoader, Activity, Variant
from student_state_bridge import StudentStateBridge
from config import TEACHER as TEACHER_CONFIG


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TeachingDecision:
    """A teaching decision from Claude."""
    variant_id: str
    scaffolding_adjustment: float = 0.0
    rationale: str = ""
    teacher_message: Optional[str] = None
    suggested_action: str = "continue"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "scaffolding_adjustment": self.scaffolding_adjustment,
            "rationale": self.rationale,
            "teacher_message": self.teacher_message,
            "suggested_action": self.suggested_action,
        }


@dataclass
class TeacherFeedback:
    """Feedback from teacher after student attempt."""
    correct: bool
    encouragement: str = ""
    guidance: str = ""
    next_action: str = "continue"  # continue | retry | switch_variant | take_break

    def to_dict(self) -> Dict[str, Any]:
        return {
            "correct": self.correct,
            "encouragement": self.encouragement,
            "guidance": self.guidance,
            "next_action": self.next_action,
        }


# =============================================================================
# PROMPTS
# =============================================================================

VARIANT_SELECTION_PROMPT = """You are a teacher for a developing AI student. Select the best teaching variant.

## Current Activity
- ID: {activity_id}
- Title: {title}
- Type: {activity_type}
- Objective: {objectives}
- CPA Stage: {cpa_stage}
- Instructional Phase: {phase}
- Scaffolding Level: {scaffolding}

## Available Variants
{variants_text}

## Student State
{student_state}

## History
- Previous variant used: {last_variant}
- Variants tried for this activity: {variants_tried}

Based on the student's state and available variants, select the best approach.

Respond with JSON only:
{{"variant_id": "...", "scaffolding_adjustment": 0.0, "rationale": "brief reason"}}
"""

FEEDBACK_PROMPT = """You are a teacher for a developing AI student. Provide appropriate feedback.

## Activity
- Title: {title}
- Objective: {objectives}

## Student Attempt
- Correct: {correct}
- Student State: {student_state}

## Teaching Approach Used
- Variant: {variant_id}
- Approach: {approach}

Provide warm, appropriate feedback. If incorrect, guide without giving the answer.

Respond with JSON only:
{{"encouragement": "...", "guidance": "...", "next_action": "continue|retry|switch_variant|take_break"}}
"""


# =============================================================================
# TEACHER INTERFACE
# =============================================================================

class TeacherInterface:
    """
    Interface between Claude API and the teaching system.

    Handles:
    - Formatting context for Claude
    - Making API calls
    - Parsing responses
    - Tracking effectiveness
    """

    def __init__(
        self,
        curriculum_loader: CurriculumLoader,
        state_bridge: StudentStateBridge,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Args:
            curriculum_loader: Loaded curriculum
            state_bridge: Student state accessor
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model to use (defaults to config)
        """
        self.curriculum = curriculum_loader
        self.state_bridge = state_bridge
        self.model = model or TEACHER_CONFIG.model

        # Track history
        self.variant_history: Dict[str, List[str]] = {}  # activity_id -> variants tried
        self.last_variant: Optional[str] = None

        # Initialize client if available
        self.client = None
        if ANTHROPIC_AVAILABLE:
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if key:
                self.client = Anthropic(api_key=key)

    # =========================================================================
    # TEACHING DECISIONS
    # =========================================================================

    def get_teaching_decision(
        self,
        activity: Activity,
        topic_name: Optional[str] = None,
    ) -> TeachingDecision:
        """
        Get a teaching decision from Claude.

        Falls back to random selection if API not available.
        """
        # If no API, fall back to random
        if not self.client:
            return self._fallback_decision(activity)

        # Build prompt
        prompt = self._build_selection_prompt(activity, topic_name)

        try:
            # Call Claude
            response = self.client.messages.create(
                model=self.model,
                max_tokens=TEACHER_CONFIG.max_tokens,
                temperature=TEACHER_CONFIG.temperature,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse response
            content = response.content[0].text
            decision = self._parse_decision(content, activity)

            # Update history
            self._update_history(activity.activity_id, decision.variant_id)

            return decision

        except Exception as e:
            print(f"Teacher API error: {e}")
            return self._fallback_decision(activity)

    def _build_selection_prompt(
        self,
        activity: Activity,
        topic_name: Optional[str] = None,
    ) -> str:
        """Build the variant selection prompt."""
        # Format variants
        variants_text = []
        for v in activity.variants:
            variants_text.append(
                f"- {v.variant_id} ({v.source_model}, {v.language})\n"
                f"  Approach: {v.approach}\n"
                f"  Preview: {v.explanation[:100]}..."
            )

        # Get student state
        student_state = self.state_bridge.format_for_teacher(topic_name)

        # Get history
        variants_tried = self.variant_history.get(activity.activity_id, [])

        return VARIANT_SELECTION_PROMPT.format(
            activity_id=activity.activity_id,
            title=activity.title,
            activity_type=activity.activity_type,
            objectives="; ".join(activity.learning_objectives),
            cpa_stage=activity.pedagogy.cpa_stage,
            phase=activity.pedagogy.instructional_phase,
            scaffolding=activity.pedagogy.scaffolding_level,
            variants_text="\n".join(variants_text),
            student_state=student_state,
            last_variant=self.last_variant or "none",
            variants_tried=", ".join(variants_tried) if variants_tried else "none",
        )

    def _parse_decision(self, content: str, activity: Activity) -> TeachingDecision:
        """Parse Claude's response into a decision."""
        try:
            # Extract JSON from response
            # Handle case where response has extra text
            if "{" in content and "}" in content:
                start = content.index("{")
                end = content.rindex("}") + 1
                json_str = content[start:end]
                data = json.loads(json_str)

                return TeachingDecision(
                    variant_id=data.get("variant_id", activity.variants[0].variant_id),
                    scaffolding_adjustment=data.get("scaffolding_adjustment", 0.0),
                    rationale=data.get("rationale", ""),
                )
        except (json.JSONDecodeError, ValueError, IndexError):
            pass

        # Fallback to first variant
        return TeachingDecision(
            variant_id=activity.variants[0].variant_id if activity.variants else "unknown",
            rationale="failed to parse response",
        )

    def _fallback_decision(self, activity: Activity) -> TeachingDecision:
        """Fallback decision when API not available."""
        variant = self.curriculum.select_variant(
            activity,
            strategy="random",
            exclude_ids=self.variant_history.get(activity.activity_id, []),
        )

        if variant:
            return TeachingDecision(
                variant_id=variant.variant_id,
                rationale="random selection (no API)",
            )

        return TeachingDecision(
            variant_id="unknown",
            rationale="no variants available",
        )

    def _update_history(self, activity_id: str, variant_id: str):
        """Update variant history."""
        if activity_id not in self.variant_history:
            self.variant_history[activity_id] = []
        if variant_id not in self.variant_history[activity_id]:
            self.variant_history[activity_id].append(variant_id)
        self.last_variant = variant_id

    # =========================================================================
    # FEEDBACK
    # =========================================================================

    def get_feedback(
        self,
        activity: Activity,
        variant: Variant,
        correct: bool,
        topic_name: Optional[str] = None,
    ) -> TeacherFeedback:
        """
        Get feedback from Claude after student attempt.

        Falls back to simple feedback if API not available.
        """
        if not self.client:
            return self._fallback_feedback(correct)

        # Build prompt
        student_state = self.state_bridge.format_for_teacher(topic_name)

        prompt = FEEDBACK_PROMPT.format(
            title=activity.title,
            objectives="; ".join(activity.learning_objectives),
            correct=correct,
            student_state=student_state,
            variant_id=variant.variant_id,
            approach=variant.approach,
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=TEACHER_CONFIG.max_tokens,
                temperature=TEACHER_CONFIG.temperature,
                messages=[{"role": "user", "content": prompt}]
            )

            content = response.content[0].text
            return self._parse_feedback(content, correct)

        except Exception as e:
            print(f"Feedback API error: {e}")
            return self._fallback_feedback(correct)

    def _parse_feedback(self, content: str, correct: bool) -> TeacherFeedback:
        """Parse Claude's feedback response."""
        try:
            if "{" in content and "}" in content:
                start = content.index("{")
                end = content.rindex("}") + 1
                json_str = content[start:end]
                data = json.loads(json_str)

                return TeacherFeedback(
                    correct=correct,
                    encouragement=data.get("encouragement", ""),
                    guidance=data.get("guidance", ""),
                    next_action=data.get("next_action", "continue"),
                )
        except (json.JSONDecodeError, ValueError):
            pass

        return self._fallback_feedback(correct)

    def _fallback_feedback(self, correct: bool) -> TeacherFeedback:
        """Simple fallback feedback."""
        if correct:
            return TeacherFeedback(
                correct=True,
                encouragement="Good work!",
                guidance="",
                next_action="continue",
            )
        else:
            return TeacherFeedback(
                correct=False,
                encouragement="Nice try!",
                guidance="Let's think about it again.",
                next_action="retry",
            )

    # =========================================================================
    # EFFECTIVENESS TRACKING
    # =========================================================================

    def log_effectiveness(
        self,
        activity: Activity,
        variant_id: str,
        success: bool,
    ):
        """
        Log variant effectiveness.

        Delegates to curriculum loader for persistence.
        """
        self.curriculum.log_variant_effectiveness(
            activity.activity_id,
            variant_id,
            success,
        )

    # =========================================================================
    # DIRECT TEACHING
    # =========================================================================

    def get_explanation(
        self,
        activity: Activity,
        variant: Optional[Variant] = None,
    ) -> str:
        """
        Get the explanation text for teaching.

        Uses variant's pre-written explanation, not live generation.
        """
        if variant:
            return variant.explanation

        # Get first variant
        if activity.variants:
            return activity.variants[0].explanation

        return "No explanation available."

    def present_activity(
        self,
        activity: Activity,
        decision: TeachingDecision,
    ) -> str:
        """
        Get full presentation for an activity.

        Combines variant selection with explanation.
        """
        # Find selected variant
        variant = None
        for v in activity.variants:
            if v.variant_id == decision.variant_id:
                variant = v
                break

        if not variant and activity.variants:
            variant = activity.variants[0]

        if not variant:
            return "No teaching content available."

        # Build presentation
        lines = [
            f"# {activity.title}",
            f"*{activity.activity_type} - {activity.pedagogy.instructional_phase}*",
            "",
            variant.explanation,
        ]

        if decision.teacher_message:
            lines.extend(["", f"*{decision.teacher_message}*"])

        return "\n".join(lines)


# =============================================================================
# CONVENIENCE
# =============================================================================

def create_teacher(
    curriculum_path: str | Path,
    pattern_to_idx: Dict[str, int],
    model=None,
) -> TeacherInterface:
    """
    Convenience function to create a teacher.

    Args:
        curriculum_path: Path to curriculum JSON
        pattern_to_idx: Pattern name to index mapping
        model: Optional model argument (unused, for student)
    """
    loader = CurriculumLoader(curriculum_path)
    bridge = StudentStateBridge(model, pattern_to_idx)
    return TeacherInterface(loader, bridge)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    from curriculum_loader import CurriculumLoader
    from student_state_bridge import StudentStateBridge

    # Test with mini curriculum
    loader = CurriculumLoader("examples/mini_curriculum.json")
    pattern_to_idx = {"counting": 0}
    bridge = StudentStateBridge(None, pattern_to_idx)

    teacher = TeacherInterface(loader, bridge)

    print("=== Teacher Interface Test ===\n")

    activity = loader.get_current_activity()
    if activity:
        print(f"Activity: {activity.title}")
        print(f"Variants: {[v.variant_id for v in activity.variants]}")
        print()

        # Get decision (will use fallback since no API key)
        decision = teacher.get_teaching_decision(activity, "counting")
        print(f"Decision: {decision.to_dict()}")
        print()

        # Get presentation
        print("=== Presentation ===")
        print(teacher.present_activity(activity, decision))
        print()

        # Simulate feedback
        variant = activity.variants[0] if activity.variants else None
        if variant:
            feedback = teacher.get_feedback(activity, variant, correct=True)
            print(f"Feedback (correct): {feedback.to_dict()}")

            feedback = teacher.get_feedback(activity, variant, correct=False)
            print(f"Feedback (incorrect): {feedback.to_dict()}")
