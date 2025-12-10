"""
School Day - Unified class workflow integrating all systems.

A school day rotates through subjects, gives students recess time,
and maintains persistent storage for their work. This creates a more
complete educational experience than isolated pattern training.

School Day Schedule:
    1. Morning Circle - Review yesterday, set goals
    2. Math Period - Pattern completion curriculum
    3. Recess - Physics playground activities
    4. Language Arts - Reading/writing (future)
    5. Art/Music Period - Creative expression (future)
    6. Closing Circle - Reflect, save notes, journal

This file integrates:
    - developmental_curriculum.py (Math)
    - physics_playground.py (Recess)
    - student_files.py (Persistence)
    - curriculum_advisor.py (Teacher guidance)
"""

import random
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

# Our systems
from .physics_playground import PhysicsPlayground
from .student_files import StudentFileSystem, ClassLibrary


@dataclass
class StudentProfile:
    """
    Complete student profile across all subjects.

    Tracks progress in multiple domains, not just math patterns.
    """
    name: str

    # Academic progress by subject
    math_level: int = 0
    math_xp: int = 0
    reading_level: int = 0
    physics_intuition: float = 0.0  # 0-1 score on physics predictions

    # Behavioral/social
    participation_streak: int = 0
    questions_asked: int = 0
    peers_helped: int = 0

    # Interests discovered through exploration
    favorite_subjects: List[str] = field(default_factory=list)
    favorite_activities: List[str] = field(default_factory=list)

    # Today's state
    energy_level: float = 1.0  # Decreases through day, resets each morning
    attention_span: int = 10   # Batches before needing a break
    mood: str = "neutral"      # neutral, engaged, frustrated, curious

    def needs_break(self, batches_since_break: int) -> bool:
        """Check if student needs a break."""
        return batches_since_break >= self.attention_span

    def adjust_energy(self, delta: float):
        """Adjust energy level (bounded 0-1)."""
        self.energy_level = max(0.0, min(1.0, self.energy_level + delta))


@dataclass
class Period:
    """A single period in the school day."""
    name: str
    subject: str
    duration_batches: int
    activities: List[str]
    is_break: bool = False


class SchoolDay:
    """
    Orchestrates a complete school day for the classroom.

    Manages the schedule, transitions between subjects, and ensures
    students get appropriate variety and breaks.
    """

    def __init__(
        self,
        student_names: List[str],
        base_path: str = "data",
        physics_playground: PhysicsPlayground = None
    ):
        self.student_names = student_names

        # Initialize systems
        self.physics = physics_playground or PhysicsPlayground()
        self.library = ClassLibrary(f"{base_path}/shared_library")

        # Per-student filesystems
        self.filesystems: Dict[str, StudentFileSystem] = {
            name: StudentFileSystem(name, f"{base_path}/student_files")
            for name in student_names
        }

        # Student profiles
        self.profiles: Dict[str, StudentProfile] = {
            name: StudentProfile(name=name)
            for name in student_names
        }

        # Day tracking
        self.current_day = 0
        self.current_period_idx = 0
        self.batches_in_period = 0
        self.batches_since_break = 0

        # Default schedule
        self.schedule = self._default_schedule()

    def _default_schedule(self) -> List[Period]:
        """Create the default school day schedule."""
        return [
            Period(
                name="Morning Circle",
                subject="social",
                duration_batches=2,
                activities=["review_yesterday", "set_goals", "attendance"]
            ),
            Period(
                name="Math Period",
                subject="math",
                duration_batches=50,
                activities=["pattern_training", "guided_practice", "independent_work"]
            ),
            Period(
                name="Morning Recess",
                subject="physics",
                duration_batches=10,
                activities=["swing", "throw", "bounce", "spring"],
                is_break=True
            ),
            Period(
                name="Language Arts",
                subject="reading",
                duration_batches=30,
                activities=["letter_patterns", "word_building", "story_time"]
            ),
            Period(
                name="Lunch & Free Play",
                subject="break",
                duration_batches=15,
                activities=["eat", "socialize", "free_choice"],
                is_break=True
            ),
            Period(
                name="Art & Music",
                subject="creative",
                duration_batches=20,
                activities=["drawing", "rhythm", "composition"]
            ),
            Period(
                name="Afternoon Recess",
                subject="physics",
                duration_batches=10,
                activities=["swing", "throw", "bounce", "spring"],
                is_break=True
            ),
            Period(
                name="Closing Circle",
                subject="social",
                duration_batches=3,
                activities=["reflect", "share", "journal"]
            )
        ]

    @property
    def current_period(self) -> Period:
        """Get the current period."""
        return self.schedule[self.current_period_idx]

    def start_new_day(self):
        """Start a new school day."""
        self.current_day += 1
        self.current_period_idx = 0
        self.batches_in_period = 0
        self.batches_since_break = 0

        # Reset student energy
        for profile in self.profiles.values():
            profile.energy_level = 1.0

        # Clear scratch pads from yesterday
        for fs in self.filesystems.values():
            fs.scratch_clear()

        print(f"\n{'='*60}")
        print(f"SCHOOL DAY {self.current_day}")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
        print(f"Students: {', '.join(self.student_names)}")
        print(f"{'='*60}\n")

    def advance_batch(self) -> Dict[str, Any]:
        """
        Advance one batch and return context for training.

        Returns dict with:
            - period: current period info
            - subject: what to train on
            - activities: available activities
            - students_need_break: list of students who need breaks
        """
        self.batches_in_period += 1
        self.batches_since_break += 1

        period = self.current_period

        # Check if period is done
        if self.batches_in_period >= period.duration_batches:
            return self._transition_period()

        # Check for break needs
        students_need_break = [
            name for name, profile in self.profiles.items()
            if profile.needs_break(self.batches_since_break)
        ]

        # Drain energy during non-break periods
        if not period.is_break:
            for profile in self.profiles.values():
                profile.adjust_energy(-0.01)

        return {
            'period': period.name,
            'subject': period.subject,
            'activities': period.activities,
            'batch_in_period': self.batches_in_period,
            'students_need_break': students_need_break,
            'is_break': period.is_break,
            'day': self.current_day
        }

    def _transition_period(self) -> Dict[str, Any]:
        """Handle transition to next period."""
        old_period = self.current_period
        self.current_period_idx += 1
        self.batches_in_period = 0

        # Check if day is done
        if self.current_period_idx >= len(self.schedule):
            return self._end_day()

        new_period = self.current_period

        # Reset break counter if entering a break
        if new_period.is_break:
            self.batches_since_break = 0
            # Restore energy during breaks
            for profile in self.profiles.values():
                profile.adjust_energy(0.2)

        print(f"\n--- Transitioning: {old_period.name} → {new_period.name} ---\n")

        return {
            'period': new_period.name,
            'subject': new_period.subject,
            'activities': new_period.activities,
            'batch_in_period': 0,
            'transition_from': old_period.name,
            'is_break': new_period.is_break,
            'day': self.current_day
        }

    def _end_day(self) -> Dict[str, Any]:
        """End the school day."""
        print(f"\n{'='*60}")
        print(f"END OF DAY {self.current_day}")
        print(f"{'='*60}\n")

        # Auto-journal for each student
        for name, profile in self.profiles.items():
            self._auto_journal(name, profile)

        return {
            'period': 'end_of_day',
            'subject': None,
            'day_complete': True,
            'day': self.current_day
        }

    def _auto_journal(self, name: str, profile: StudentProfile):
        """Create automatic end-of-day journal entry."""
        fs = self.filesystems[name]

        entry = f"""# Day {self.current_day} Reflection

## Progress Today
- Math XP: {profile.math_xp}
- Physics Intuition: {profile.physics_intuition:.1%}
- Questions Asked: {profile.questions_asked}
- Peers Helped: {profile.peers_helped}

## Energy & Mood
- Ending Energy: {profile.energy_level:.0%}
- Ending Mood: {profile.mood}
- Participation Streak: {profile.participation_streak}

## Notes
(Auto-generated summary)
"""
        fs.journal_entry(entry, tags=['daily_summary', f'day_{self.current_day}'])

    # =========================================================================
    # SUBJECT-SPECIFIC METHODS
    # =========================================================================

    def get_recess_activity(self) -> Dict[str, Any]:
        """Get a physics playground activity for recess."""
        return self.physics.random_recess()

    def record_recess_prediction(self, student_name: str, activity: Dict, prediction: int):
        """Record a student's physics prediction and update intuition."""
        correct = (prediction == activity['target'])
        profile = self.profiles[student_name]

        # Update physics intuition with exponential moving average
        old_intuition = profile.physics_intuition
        profile.physics_intuition = 0.9 * old_intuition + 0.1 * (1.0 if correct else 0.0)

        # Save to scratch for later analysis
        fs = self.filesystems[student_name]
        fs.scratch_write('last_recess', {
            'activity': activity['pattern_type'],
            'predicted': prediction,
            'actual': activity['target'],
            'correct': correct
        })

        return correct

    def save_to_portfolio(self, student_name: str, work: Any, name: str, category: str):
        """Save student work to their portfolio."""
        fs = self.filesystems[student_name]
        success = fs.portfolio_save(name, work, category)
        if success:
            print(f"  {student_name} saved '{name}' to {category} portfolio")
        return success

    def share_with_class(self, student_name: str, work: Any, name: str, category: str):
        """Share student work with the class library."""
        success = self.library.share(work, name, category, student_name)
        if success:
            print(f"  {student_name} shared '{name}' with the class!")
        return success

    def take_notes(self, student_name: str, topic: str, content: str, section: str = "math"):
        """Student takes notes on a topic."""
        fs = self.filesystems[student_name]
        return fs.note_write(topic, content, section)

    def ask_question(self, student_name: str, question: str):
        """Record a student asking a question."""
        profile = self.profiles[student_name]
        profile.questions_asked += 1
        profile.mood = "curious"

        # Save to journal
        fs = self.filesystems[student_name]
        fs.journal_entry(f"I asked: {question}", tags=["question", "curious"])

    def help_peer(self, helper_name: str, helped_name: str, topic: str):
        """Record peer helping interaction."""
        helper = self.profiles[helper_name]
        helper.peers_helped += 1

        # Both students benefit
        for name in [helper_name, helped_name]:
            profile = self.profiles[name]
            profile.mood = "engaged"

    # =========================================================================
    # REPORTING
    # =========================================================================

    def daily_report(self) -> str:
        """Generate end-of-day report."""
        lines = [
            f"\n{'='*60}",
            f"DAILY REPORT - Day {self.current_day}",
            f"{'='*60}\n"
        ]

        for name, profile in self.profiles.items():
            fs = self.filesystems[name]
            summary = fs.summary()

            lines.append(f"{name.upper()}:")
            lines.append(f"  Math: Level {profile.math_level} ({profile.math_xp} XP)")
            lines.append(f"  Physics Intuition: {profile.physics_intuition:.1%}")
            lines.append(f"  Questions: {profile.questions_asked} | Helped: {profile.peers_helped}")
            lines.append(f"  Files: {summary['notes']} notes, {summary['portfolio_pieces']} portfolio items")
            lines.append(f"  Mood: {profile.mood} | Energy: {profile.energy_level:.0%}")
            lines.append("")

        return "\n".join(lines)

    def class_summary(self) -> Dict:
        """Get summary statistics for the class."""
        return {
            'day': self.current_day,
            'students': len(self.student_names),
            'avg_math_level': sum(p.math_level for p in self.profiles.values()) / len(self.profiles),
            'avg_physics_intuition': sum(p.physics_intuition for p in self.profiles.values()) / len(self.profiles),
            'total_questions': sum(p.questions_asked for p in self.profiles.values()),
            'total_peer_help': sum(p.peers_helped for p in self.profiles.values()),
        }


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def create_training_batch_with_context(
    school_day: SchoolDay,
    math_dataset,  # DevelopmentalDataset
    batch_size: int = 32
) -> Dict[str, Any]:
    """
    Create a training batch with school day context.

    This wraps the standard training batch with schedule awareness,
    inserting recess activities and maintaining student state.
    """
    context = school_day.advance_batch()

    if context.get('day_complete'):
        return context

    subject = context['subject']

    if subject == 'math':
        # Standard pattern training batch
        return {
            **context,
            'batch_type': 'pattern_training',
            'data_source': 'developmental_curriculum'
        }

    elif subject == 'physics':
        # Physics playground activity
        activity = school_day.get_recess_activity()
        return {
            **context,
            'batch_type': 'physics_prediction',
            'activity': activity,
            'sequence': activity['sequence'],
            'target': activity['target'],
            'hint': activity.get('hint', '')
        }

    elif subject == 'reading':
        # Placeholder for reading curriculum
        return {
            **context,
            'batch_type': 'reading_placeholder',
            'note': 'Reading curriculum not yet implemented'
        }

    elif subject == 'creative':
        # Placeholder for art/music
        return {
            **context,
            'batch_type': 'creative_placeholder',
            'note': 'Creative curriculum not yet implemented'
        }

    elif subject == 'social':
        # Circle time - no training, just state management
        return {
            **context,
            'batch_type': 'circle_time',
            'activities': context['activities']
        }

    elif subject == 'break':
        # Free time - restore energy, no training
        for profile in school_day.profiles.values():
            profile.adjust_energy(0.1)
        return {
            **context,
            'batch_type': 'break'
        }

    return context


# =============================================================================
# DEMO
# =============================================================================

if __name__ == '__main__':
    print("=== School Day Demo ===\n")

    # Create a school day for our students
    school = SchoolDay(
        student_names=['nova', 'reve', 'alex'],
        base_path='/tmp/coherence_school'
    )

    # Start the day
    school.start_new_day()

    # Simulate a few batches
    print("Simulating school day batches...\n")

    batch_count = 0
    while batch_count < 20:
        context = school.advance_batch()

        if context.get('day_complete'):
            break

        if 'transition_from' in context:
            # Period transition happened
            pass
        elif context['batch_in_period'] == 1:
            # First batch of period
            print(f"  [{context['period']}] Starting {context['subject']}...")

        batch_count += 1

        # Simulate some interactions
        if context['subject'] == 'physics' and batch_count % 3 == 0:
            activity = school.get_recess_activity()
            for name in school.student_names:
                # Random prediction for demo
                pred = random.choice(activity['sequence'])
                correct = school.record_recess_prediction(name, activity, pred)

    # Daily report
    print(school.daily_report())

    # Class summary
    print("Class Summary:", school.class_summary())

    print("\n=== Demo Complete ===")
