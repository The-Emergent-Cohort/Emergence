"""
Curriculum Advisor - LLM-guided training for neural network students

Uses Gemini (or other LLM) to make pedagogical decisions about student training.
Explores whether an LLM can effectively teach smaller neural networks.

Control levels:
  - OBSERVER: Reports on student progress, no interventions
  - ADVISOR: Suggests curriculum changes, human/Teacher approves
  - CONTROLLER: Directly modifies training parameters
  - TRAINER: Full autonomy over pedagogical decisions

Usage:
    advisor = CurriculumAdvisor(control_level="CONTROLLER")
    advisor.configure_api(api_key)

    # Each epoch, let advisor observe and potentially intervene
    decisions = advisor.evaluate_and_decide(student_states)
    for student_name, actions in decisions.items():
        apply_actions(student, actions)
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class ControlLevel(Enum):
    OBSERVER = "observer"      # Watch and report only
    ADVISOR = "advisor"        # Suggest, don't act
    CONTROLLER = "controller"  # Modify parameters directly
    TRAINER = "trainer"        # Full pedagogical autonomy


@dataclass
class StudentState:
    """Snapshot of a student's learning state for advisor analysis."""
    name: str
    epoch: int

    # Performance metrics
    train_acc: float
    val_acc: float
    loss: float

    # Per-pattern breakdown
    pattern_accuracies: Dict[str, float]  # pattern_type -> accuracy
    pattern_xp: Dict[str, float]          # pattern_type -> XP
    pattern_levels: Dict[str, int]        # pattern_type -> confirmed level

    # Behavioral metrics
    confidence_avg: float
    calibration_gap: float  # |confidence - accuracy|
    approval_rate: float    # % of "show work" that was correct

    # History (last N epochs)
    acc_history: List[float] = field(default_factory=list)
    loss_history: List[float] = field(default_factory=list)

    # Exam history
    recent_exams: List[Dict] = field(default_factory=list)  # [{pattern, level, passed, score}]


@dataclass
class TrainingAction:
    """An action the advisor wants to take on a student."""
    action_type: str  # "set_lr", "focus_pattern", "skip_pattern", "run_exam", "adjust_difficulty"
    target: str       # student name or "all"
    params: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


class CurriculumAdvisor:
    """
    LLM-powered curriculum advisor for neural network students.

    Analyzes student performance and makes pedagogical decisions
    at various levels of autonomy.
    """

    def __init__(
        self,
        control_level: str = "ADVISOR",
        model_id: str = "gemini-3-pro-preview",
        query_interval: int = 5,  # Epochs between queries
        thinking_level: str = "high"
    ):
        self.control_level = ControlLevel(control_level.lower())
        self.model_id = model_id
        self.query_interval = query_interval
        self.thinking_level = thinking_level

        self.client = None
        self.history: List[Dict] = []  # Conversation/decision history
        self.last_query_epoch = -999

        # System prompt establishing the advisor's role
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build the system prompt based on control level."""

        base = """You are a curriculum advisor for a classroom of small neural networks learning pattern completion tasks.

CONTEXT:
- Students are transformer-based models (~50K parameters each)
- They learn to complete sequences: [A,B,A,B,?] → A (alternating), [3,3,3,?] → 3 (repeating), etc.
- Pattern types: alternating, repeating, incrementing, fixed_offset
- Students earn XP for correct answers, take exams to confirm level-ups
- Levels 1-10, with increasing exam difficulty (75% → 90% pass threshold)

YOUR ROLE:
Analyze student performance data and make pedagogical decisions to optimize their learning.
Consider:
- Which patterns each student struggles with
- Whether students are overconfident or underconfident (calibration)
- Learning rate adjustments based on loss curves
- When to focus on weak areas vs. reinforce strengths
- Peer learning opportunities (students teaching each other)

"""

        level_specific = {
            ControlLevel.OBSERVER: """
CONTROL LEVEL: OBSERVER
You observe and report only. Provide analysis and insights but no action recommendations.
Format your response as a JSON object with:
{
  "observations": {"student_name": "analysis..."},
  "patterns_noted": ["pattern1", "pattern2"],
  "concerns": ["concern1", "concern2"]
}
""",
            ControlLevel.ADVISOR: """
CONTROL LEVEL: ADVISOR
You analyze and suggest, but a human approves actions.
Format your response as a JSON object with:
{
  "analysis": {"student_name": "analysis..."},
  "suggestions": [
    {"student": "name", "action": "description", "reasoning": "why"}
  ],
  "priority": "highest priority suggestion"
}
""",
            ControlLevel.CONTROLLER: """
CONTROL LEVEL: CONTROLLER
You can directly modify training parameters. Your decisions will be executed.
Format your response as a JSON object with:
{
  "analysis": {"student_name": "brief analysis..."},
  "actions": [
    {
      "action_type": "set_lr|focus_pattern|skip_pattern|adjust_batch|cool_down",
      "target": "student_name|all",
      "params": {"lr": 0.001, "pattern": "alternating", ...},
      "reasoning": "why this action"
    }
  ]
}

Available action_types:
- set_lr: Set learning rate. params: {"lr": float}
- focus_pattern: Increase sampling weight for pattern. params: {"pattern": str, "weight": float}
- skip_pattern: Temporarily skip a mastered pattern. params: {"pattern": str, "epochs": int}
- adjust_batch: Change batch size. params: {"batch_size": int}
- cool_down: Reduce training intensity for overfitting student. params: {"factor": float}
""",
            ControlLevel.TRAINER: """
CONTROL LEVEL: TRAINER
You have full pedagogical autonomy. Design the training strategy.
Format your response as a JSON object with:
{
  "strategy": "overall approach description",
  "per_student_plans": {
    "student_name": {
      "diagnosis": "what's happening",
      "prescription": "what to do",
      "actions": [{"action_type": ..., "params": ...}]
    }
  },
  "classroom_actions": [
    {"action_type": "peer_teach", "teacher": "student1", "learner": "student2", "topic": "pattern"},
    {"action_type": "group_drill", "pattern": "...", "epochs": N}
  ],
  "next_review_in": N  # epochs until you want to reassess
}

Additional TRAINER action_types:
- peer_teach: Have one student's outputs guide another
- group_drill: Focused practice for all students on one pattern
- introduce_pattern: Add a new pattern type to curriculum
- exam_override: Force or skip an exam
- celebration: Acknowledge milestone (affects "morale" tracking)
"""
        }

        return base + level_specific[self.control_level]

    def configure_api(self, api_key: Optional[str] = None):
        """Configure the Gemini API client."""
        try:
            from google import genai
            from google.genai import types

            if api_key:
                import os
                os.environ["GOOGLE_API_KEY"] = api_key

            self.client = genai.Client()
            self._types = types
            print(f"[CurriculumAdvisor] Configured with {self.model_id}, control={self.control_level.value}")
            return True
        except ImportError:
            print("[CurriculumAdvisor] google-genai not installed. Run: pip install google-genai")
            return False
        except Exception as e:
            print(f"[CurriculumAdvisor] API configuration failed: {e}")
            return False

    def should_query(self, epoch: int) -> bool:
        """Check if we should query the advisor this epoch."""
        return (epoch - self.last_query_epoch) >= self.query_interval

    def format_student_states(self, states: List[StudentState]) -> str:
        """Format student states for the LLM prompt."""
        lines = [f"=== EPOCH {states[0].epoch} STATUS ===\n"]

        for s in states:
            lines.append(f"## {s.name.upper()}")
            lines.append(f"Train: {s.train_acc:.1%} | Val: {s.val_acc:.1%} | Loss: {s.loss:.4f}")
            lines.append(f"Confidence: {s.confidence_avg:.1%} | Calibration gap: {s.calibration_gap:+.1%}")
            lines.append(f"Approval rate (show work): {s.approval_rate:.0%}")
            lines.append("")
            lines.append("Per-pattern breakdown:")
            for pt in s.pattern_accuracies:
                acc = s.pattern_accuracies[pt]
                xp = s.pattern_xp.get(pt, 0)
                lvl = s.pattern_levels.get(pt, 0)
                lines.append(f"  {pt}: {acc:.0%} (L{lvl}, {xp:.0f} XP)")

            if s.acc_history:
                lines.append(f"\nRecent accuracy trend: {' → '.join(f'{a:.0%}' for a in s.acc_history[-5:])}")

            if s.recent_exams:
                lines.append("Recent exams:")
                for ex in s.recent_exams[-3:]:
                    status = "✓ PASS" if ex.get('passed') else "✗ FAIL"
                    lines.append(f"  {ex['pattern']} L{ex['level']}: {status} ({ex['score']:.0%})")

            lines.append("")

        return "\n".join(lines)

    def query(self, states: List[StudentState]) -> Dict:
        """
        Query the LLM advisor with current student states.
        Returns parsed response with analysis/actions.
        """
        if not self.client:
            return {"error": "API not configured"}

        # Build the prompt
        state_text = self.format_student_states(states)

        prompt = f"""{state_text}

Based on this data, provide your {self.control_level.value}-level response.
Respond with valid JSON only, no markdown code blocks."""

        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=self._types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    thinking_config=self._types.ThinkingConfig(
                        thinking_level=self.thinking_level
                    ),
                    temperature=0.7,
                )
            )

            # Parse JSON response
            text = response.text.strip()
            # Handle potential markdown wrapping
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]

            result = json.loads(text)
            result["_raw"] = response.text
            result["_epoch"] = states[0].epoch

            self.history.append(result)
            self.last_query_epoch = states[0].epoch

            return result

        except json.JSONDecodeError as e:
            return {"error": f"JSON parse failed: {e}", "_raw": response.text}
        except Exception as e:
            return {"error": str(e)}

    def evaluate_and_decide(
        self,
        states: List[StudentState],
        force: bool = False
    ) -> Dict[str, List[TrainingAction]]:
        """
        Main entry point: evaluate students and return decisions.

        Returns dict mapping student names to lists of actions.
        Empty dict if not querying this epoch (unless force=True).
        """
        if not force and not self.should_query(states[0].epoch):
            return {}

        response = self.query(states)

        if "error" in response:
            print(f"[CurriculumAdvisor] Error: {response['error']}")
            return {}

        # Parse actions based on control level
        actions_by_student: Dict[str, List[TrainingAction]] = {}

        if self.control_level == ControlLevel.OBSERVER:
            # Just print observations, no actions
            if "observations" in response:
                print("\n[Advisor Observations]")
                for name, obs in response["observations"].items():
                    print(f"  {name}: {obs}")
            return {}

        elif self.control_level == ControlLevel.ADVISOR:
            # Print suggestions for human review
            if "suggestions" in response:
                print("\n[Advisor Suggestions]")
                for sug in response["suggestions"]:
                    print(f"  → {sug['student']}: {sug['action']}")
                    print(f"    Reasoning: {sug['reasoning']}")
            if "priority" in response:
                print(f"  Priority: {response['priority']}")
            return {}  # Human must implement

        elif self.control_level == ControlLevel.CONTROLLER:
            # Parse and return executable actions
            if "actions" in response:
                for action_dict in response["actions"]:
                    action = TrainingAction(
                        action_type=action_dict.get("action_type", "unknown"),
                        target=action_dict.get("target", "all"),
                        params=action_dict.get("params", {}),
                        reasoning=action_dict.get("reasoning", "")
                    )
                    target = action.target
                    if target not in actions_by_student:
                        actions_by_student[target] = []
                    actions_by_student[target].append(action)

        elif self.control_level == ControlLevel.TRAINER:
            # Full autonomy - parse comprehensive plan
            if "per_student_plans" in response:
                for name, plan in response["per_student_plans"].items():
                    actions_by_student[name] = []
                    for action_dict in plan.get("actions", []):
                        action = TrainingAction(
                            action_type=action_dict.get("action_type", "unknown"),
                            target=name,
                            params=action_dict.get("params", {}),
                            reasoning=plan.get("diagnosis", "")
                        )
                        actions_by_student[name].append(action)

            if "classroom_actions" in response:
                actions_by_student["_classroom"] = [
                    TrainingAction(
                        action_type=a.get("action_type"),
                        target="classroom",
                        params=a
                    )
                    for a in response["classroom_actions"]
                ]

        return actions_by_student

    def get_history_summary(self) -> str:
        """Get a summary of advisor decisions over time."""
        if not self.history:
            return "No advisor history yet."

        lines = [f"Advisor History ({len(self.history)} queries):"]
        for entry in self.history[-5:]:  # Last 5
            epoch = entry.get("_epoch", "?")
            if "actions" in entry:
                n_actions = len(entry["actions"])
                lines.append(f"  Epoch {epoch}: {n_actions} actions")
            elif "suggestions" in entry:
                n_sug = len(entry["suggestions"])
                lines.append(f"  Epoch {epoch}: {n_sug} suggestions")
            elif "observations" in entry:
                lines.append(f"  Epoch {epoch}: observations recorded")

        return "\n".join(lines)


# =============================================================================
# Action Executors - Apply advisor decisions to students
# =============================================================================

def apply_action(student, action: TrainingAction, optimizer=None) -> bool:
    """
    Apply a single training action to a student.

    Returns True if action was applied successfully.
    """
    action_type = action.action_type
    params = action.params

    if action_type == "set_lr":
        if optimizer is not None:
            new_lr = params.get("lr", 0.001)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"    [Action] Set {action.target} LR → {new_lr}")
            return True

    elif action_type == "focus_pattern":
        # This would modify sampling weights in the data loader
        pattern = params.get("pattern")
        weight = params.get("weight", 2.0)
        print(f"    [Action] Focus {action.target} on {pattern} (weight={weight})")
        # Implementation depends on data loader design
        return True

    elif action_type == "skip_pattern":
        pattern = params.get("pattern")
        epochs = params.get("epochs", 5)
        print(f"    [Action] Skip {pattern} for {action.target} ({epochs} epochs)")
        return True

    elif action_type == "cool_down":
        factor = params.get("factor", 0.5)
        if optimizer is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= factor
            print(f"    [Action] Cool down {action.target} (LR × {factor})")
            return True

    elif action_type == "peer_teach":
        teacher = params.get("teacher")
        learner = params.get("learner")
        topic = params.get("topic")
        print(f"    [Action] Peer teaching: {teacher} → {learner} on {topic}")
        # Implementation: use teacher's outputs as soft labels
        return True

    else:
        print(f"    [Action] Unknown action type: {action_type}")
        return False

    return False


# =============================================================================
# Helper to build StudentState from classroom
# =============================================================================

def build_student_state(
    student,  # Student from classroom.py
    epoch: int,
    train_results: Dict,
    val_results: Dict,
    pattern_types: List[str]
) -> StudentState:
    """Build a StudentState from classroom training results."""

    name = student.name

    # Per-pattern data from topic tracker
    pattern_accs = {}
    pattern_xp = {}
    pattern_levels = {}

    for i, pt in enumerate(pattern_types):
        # Get from topic tracker
        if hasattr(student, 'topic_tracker'):
            acc, conf, gap = student.topic_tracker.get_calibration(i)
            pattern_accs[pt] = acc
            pattern_xp[pt] = student.topic_tracker.progression.topic_xp[i].item()
            pattern_levels[pt] = student.exam_system.confirmed_level[i].item()
        else:
            pattern_accs[pt] = 0.0
            pattern_xp[pt] = 0.0
            pattern_levels[pt] = 0

    # Overall metrics
    train_acc = train_results.get(name, {}).get('accuracy', 0.0)
    val_acc = val_results.get(name, {}).get('accuracy', 0.0)
    loss = train_results.get(name, {}).get('loss', 0.0)

    # Behavioral metrics (from train results if available)
    confidence_avg = train_results.get(name, {}).get('avg_confidence', 0.5)
    approval_rate = train_results.get(name, {}).get('approval_rate', 0.0)
    calibration_gap = confidence_avg - train_acc

    return StudentState(
        name=name,
        epoch=epoch,
        train_acc=train_acc,
        val_acc=val_acc,
        loss=loss,
        pattern_accuracies=pattern_accs,
        pattern_xp=pattern_xp,
        pattern_levels=pattern_levels,
        confidence_avg=confidence_avg,
        calibration_gap=calibration_gap,
        approval_rate=approval_rate,
        acc_history=[],  # Would need to track across epochs
        loss_history=[],
        recent_exams=[]  # Would populate from exam results
    )
