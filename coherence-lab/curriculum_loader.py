"""
Curriculum Loader - Parse and navigate JSON curriculum files.

Bridges the curriculum format spec with the training system:
- Loads hierarchical curriculum JSON
- Maps activities to problem generators
- Tracks position and provides navigation
- Extracts metadata for teacher decisions

Usage:
    loader = CurriculumLoader("curriculum/math_k.json")
    activity = loader.get_current_activity()
    generator = loader.get_generator_for_activity(activity)
    variants = loader.get_variants(activity)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Iterator
from dataclasses import dataclass, field
import importlib
import random

from config import GENERATOR_REGISTRY, MODEL


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Variant:
    """A single teaching variant for an activity."""
    variant_id: str
    source_model: str
    language: str
    approach: str
    explanation: str
    instructions_teacher: Optional[str] = None
    instructions_learner: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict) -> "Variant":
        return cls(
            variant_id=data.get("variant_id", "unknown"),
            source_model=data.get("source_model", "unknown"),
            language=data.get("language", "en-US"),
            approach=data.get("approach", "procedural"),
            explanation=data.get("explanation", ""),
            instructions_teacher=data.get("instructions_teacher"),
            instructions_learner=data.get("instructions_learner"),
        )


@dataclass
class ProblemGenerator:
    """Reference to a problem generator function."""
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict) -> "ProblemGenerator":
        return cls(
            name=data.get("name", "counting"),
            params=data.get("params", {}),
        )


@dataclass
class Pedagogy:
    """Pedagogical metadata for an activity."""
    cpa_stage: str = "concrete"  # concrete | pictorial | abstract
    instructional_phase: str = "i_do"  # i_do | we_do | you_do_together | you_do_alone
    scaffolding_level: float = 1.0  # 0.0 (none) to 1.0 (full)

    @classmethod
    def from_dict(cls, data: Dict) -> "Pedagogy":
        return cls(
            cpa_stage=data.get("cpa_stage", "concrete"),
            instructional_phase=data.get("instructional_phase", "i_do"),
            scaffolding_level=data.get("scaffolding_level", 1.0),
        )


@dataclass
class Activity:
    """A single learning activity."""
    activity_id: str
    title: str
    activity_type: str  # demonstration | guided_practice | independent_practice
    sequence: int
    learning_objectives: List[str]
    variants: List[Variant]
    pedagogy: Pedagogy
    problem_generator: Optional[ProblemGenerator]
    raw_data: Dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict) -> "Activity":
        # Parse variants
        variants = [Variant.from_dict(v) for v in data.get("variants", [])]

        # Parse extensions
        extensions = data.get("extensions", {})
        pedagogy_data = extensions.get("pedagogy", {})
        di_training = extensions.get("di_training", {})

        # Parse problem generator if present
        gen_data = di_training.get("problem_generator")
        problem_generator = ProblemGenerator.from_dict(gen_data) if gen_data else None

        return cls(
            activity_id=data.get("activity_id", "unknown"),
            title=data.get("title", "Untitled"),
            activity_type=data.get("activity_type", "demonstration"),
            sequence=data.get("sequence", 0),
            learning_objectives=data.get("learning_objectives", []),
            variants=variants,
            pedagogy=Pedagogy.from_dict(pedagogy_data),
            problem_generator=problem_generator,
            raw_data=data,
        )


@dataclass
class Lesson:
    """A lesson containing multiple activities."""
    lesson_id: str
    title: str
    sequence: int
    activities: List[Activity]
    raw_data: Dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict) -> "Lesson":
        activities = [Activity.from_dict(a) for a in data.get("activities", [])]
        return cls(
            lesson_id=data.get("lesson_id", "unknown"),
            title=data.get("title", "Untitled"),
            sequence=data.get("sequence", 0),
            activities=activities,
            raw_data=data,
        )


# =============================================================================
# CURRICULUM LOADER
# =============================================================================

class CurriculumLoader:
    """
    Load and navigate a curriculum JSON file.

    Provides:
    - Hierarchical navigation (subject → course → unit → lesson → activity)
    - Activity-to-generator mapping
    - Variant access for teacher
    - Position tracking
    """

    def __init__(self, curriculum_path: Path | str):
        self.path = Path(curriculum_path)
        self.curriculum: Dict = {}
        self.activities: List[Activity] = []
        self.activity_index: Dict[str, Activity] = {}

        # Position tracking
        self.current_idx: int = 0

        # Effectiveness tracking
        self.variant_effectiveness: Dict[str, Dict[str, float]] = {}

        # Load the curriculum
        self._load()

    def _load(self):
        """Load and parse the curriculum JSON."""
        with open(self.path, "r", encoding="utf-8") as f:
            self.curriculum = json.load(f)

        # Flatten activities for easy iteration
        self._flatten_activities()

    def _flatten_activities(self):
        """Extract all activities into a flat list, preserving order."""
        self.activities = []
        self.activity_index = {}

        for subject in self.curriculum.get("subjects", []):
            for course in subject.get("courses", []):
                for unit in course.get("units", []):
                    for lesson_data in unit.get("lessons", []):
                        lesson = Lesson.from_dict(lesson_data)
                        for activity in lesson.activities:
                            self.activities.append(activity)
                            self.activity_index[activity.activity_id] = activity

    # =========================================================================
    # NAVIGATION
    # =========================================================================

    def get_current_activity(self) -> Optional[Activity]:
        """Get the activity at current position."""
        if 0 <= self.current_idx < len(self.activities):
            return self.activities[self.current_idx]
        return None

    def advance(self) -> Optional[Activity]:
        """Move to next activity and return it."""
        self.current_idx += 1
        return self.get_current_activity()

    def go_to(self, activity_id: str) -> Optional[Activity]:
        """Jump to a specific activity by ID."""
        if activity_id in self.activity_index:
            activity = self.activity_index[activity_id]
            self.current_idx = self.activities.index(activity)
            return activity
        return None

    def reset(self):
        """Go back to the beginning."""
        self.current_idx = 0

    def __iter__(self) -> Iterator[Activity]:
        """Iterate through all activities."""
        return iter(self.activities)

    def __len__(self) -> int:
        return len(self.activities)

    # =========================================================================
    # ACTIVITY ACCESS
    # =========================================================================

    def get_activity(self, activity_id: str) -> Optional[Activity]:
        """Get a specific activity by ID."""
        return self.activity_index.get(activity_id)

    def get_activities_by_type(self, activity_type: str) -> List[Activity]:
        """Get all activities of a specific type."""
        return [a for a in self.activities if a.activity_type == activity_type]

    def get_activities_by_cpa_stage(self, stage: str) -> List[Activity]:
        """Get all activities at a specific CPA stage."""
        return [a for a in self.activities if a.pedagogy.cpa_stage == stage]

    # =========================================================================
    # VARIANTS
    # =========================================================================

    def get_variants(self, activity: Activity | str) -> List[Variant]:
        """Get all variants for an activity."""
        if isinstance(activity, str):
            activity = self.get_activity(activity)
        return activity.variants if activity else []

    def select_variant(
        self,
        activity: Activity,
        strategy: str = "random",
        exclude_ids: Optional[List[str]] = None,
        prefer_approach: Optional[str] = None,
    ) -> Optional[Variant]:
        """
        Select a variant based on strategy.

        Strategies:
        - random: Random selection
        - effective: Weight by effectiveness (if tracked)
        - approach: Prefer specific approach type
        """
        variants = activity.variants
        if not variants:
            return None

        # Filter out excluded
        if exclude_ids:
            variants = [v for v in variants if v.variant_id not in exclude_ids]

        if not variants:
            return None

        # Filter by approach if specified
        if prefer_approach:
            approach_variants = [v for v in variants if v.approach == prefer_approach]
            if approach_variants:
                variants = approach_variants

        # Apply strategy
        if strategy == "random":
            return random.choice(variants)

        elif strategy == "effective":
            # Weight by effectiveness if we have data
            activity_eff = self.variant_effectiveness.get(activity.activity_id, {})
            if activity_eff:
                # Simple weighted selection
                weights = [activity_eff.get(v.variant_id, 0.5) for v in variants]
                total = sum(weights)
                if total > 0:
                    weights = [w / total for w in weights]
                    return random.choices(variants, weights=weights, k=1)[0]

            return random.choice(variants)

        elif strategy == "approach" and prefer_approach:
            return variants[0] if variants else None

        return random.choice(variants)

    def log_variant_effectiveness(
        self,
        activity_id: str,
        variant_id: str,
        success: bool,
        alpha: float = 0.1,
    ):
        """
        Update effectiveness tracking for a variant.

        Uses exponential moving average.
        """
        if activity_id not in self.variant_effectiveness:
            self.variant_effectiveness[activity_id] = {}

        current = self.variant_effectiveness[activity_id].get(variant_id, 0.5)
        new_value = 1.0 if success else 0.0
        self.variant_effectiveness[activity_id][variant_id] = (
            (1 - alpha) * current + alpha * new_value
        )

    # =========================================================================
    # PROBLEM GENERATORS
    # =========================================================================

    def get_generator_for_activity(
        self,
        activity: Activity | str,
        generator_map_path: Optional[Path] = None,
    ) -> Optional[Callable]:
        """
        Get the problem generator function for an activity.

        Checks in order:
        1. Activity's embedded problem_generator field
        2. External activity_generator_map.json (explicit mappings)
        3. Activity ID pattern matching from map
        4. Keyword inference from map

        Returns a callable that generates practice problems.
        """
        if isinstance(activity, str):
            activity = self.get_activity(activity)

        if not activity:
            return None

        # Try embedded generator first
        gen_info = None
        if activity.problem_generator:
            gen_info = {
                "name": activity.problem_generator.name,
                "params": activity.problem_generator.params,
            }

        # Try external mapping if no embedded generator
        if not gen_info:
            gen_info = self._lookup_generator_from_map(
                activity.activity_id,
                generator_map_path,
            )

        if not gen_info:
            return None

        gen_name = gen_info.get("name")
        gen_path = GENERATOR_REGISTRY.get(gen_name)

        if not gen_path:
            return None

        # Import and return the generator
        try:
            module_path, func_name = gen_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            generator = getattr(module, func_name)

            # Wrap with params from gen_info
            params = gen_info.get("params", {})

            def wrapped_generator(vocab_size: int = MODEL.vocab_size):
                return generator(vocab_size=vocab_size, **params)

            return wrapped_generator
        except (ImportError, AttributeError) as e:
            print(f"Warning: Could not load generator {gen_name}: {e}")
            return None

    def _lookup_generator_from_map(
        self,
        activity_id: str,
        map_path: Optional[Path] = None,
    ) -> Optional[Dict]:
        """
        Look up generator from external mapping file.

        Checks:
        1. explicit_mappings (exact activity_id match)
        2. activity_id_patterns (prefix match)
        3. patterns (keyword in activity_id)
        """
        # Default map location
        if map_path is None:
            map_path = Path("curriculum/activity_generator_map.json")

        if not map_path.exists():
            return None

        try:
            with open(map_path, "r") as f:
                gen_map = json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

        # 1. Check explicit mappings
        explicit = gen_map.get("explicit_mappings", {})
        if activity_id in explicit:
            return explicit[activity_id]

        # 2. Check activity_id_patterns (prefix match)
        id_patterns = gen_map.get("activity_id_patterns", {})
        for pattern, gen_name in id_patterns.items():
            if activity_id.startswith(pattern):
                patterns = gen_map.get("patterns", {})
                if gen_name in patterns:
                    return patterns[gen_name]

        # 3. Check patterns (keyword in activity_id)
        patterns = gen_map.get("patterns", {})
        for keyword, gen_info in patterns.items():
            if keyword in activity_id.lower():
                return gen_info

        return None

    def generate_problem(
        self,
        activity: Activity | str,
        vocab_size: Optional[int] = None,
    ) -> Optional[Dict]:
        """
        Generate a practice problem for an activity.

        Returns dict with 'sequence', 'target', etc.
        """
        generator = self.get_generator_for_activity(activity)
        if not generator:
            return None

        vocab_size = vocab_size or MODEL.vocab_size
        return generator(vocab_size)

    # =========================================================================
    # METADATA
    # =========================================================================

    def get_curriculum_metadata(self) -> Dict:
        """Get top-level curriculum metadata."""
        return {
            "curriculum_id": self.curriculum.get("curriculum_id"),
            "title": self.curriculum.get("title"),
            "version": self.curriculum.get("version"),
            "locale": self.curriculum.get("locale"),
            "education_levels": self.curriculum.get("education_levels", []),
        }

    def get_progress(self) -> Dict:
        """Get current progress through curriculum."""
        return {
            "current_index": self.current_idx,
            "total_activities": len(self.activities),
            "percent_complete": (
                self.current_idx / len(self.activities) * 100
                if self.activities else 0
            ),
            "current_activity_id": (
                self.activities[self.current_idx].activity_id
                if self.activities and self.current_idx < len(self.activities)
                else None
            ),
        }

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def save_state(self, path: Path | str):
        """Save loader state (position, effectiveness tracking)."""
        state = {
            "curriculum_path": str(self.path),
            "current_idx": self.current_idx,
            "variant_effectiveness": self.variant_effectiveness,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: Path | str):
        """Restore loader state."""
        with open(path, "r") as f:
            state = json.load(f)
        self.current_idx = state.get("current_idx", 0)
        self.variant_effectiveness = state.get("variant_effectiveness", {})


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_curriculum(path: str | Path) -> CurriculumLoader:
    """Convenience function to load a curriculum."""
    return CurriculumLoader(path)


def list_available_generators() -> List[str]:
    """List all registered problem generators."""
    return list(GENERATOR_REGISTRY.keys())


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    # Quick test with mini curriculum
    from pathlib import Path

    example_path = Path("examples/mini_curriculum.json")
    if example_path.exists():
        loader = CurriculumLoader(example_path)

        print(f"Loaded curriculum: {loader.get_curriculum_metadata()['title']}")
        print(f"Total activities: {len(loader)}")
        print()

        for activity in loader:
            print(f"Activity: {activity.activity_id}")
            print(f"  Type: {activity.activity_type}")
            print(f"  CPA Stage: {activity.pedagogy.cpa_stage}")
            print(f"  Phase: {activity.pedagogy.instructional_phase}")
            print(f"  Variants: {len(activity.variants)}")
            if activity.problem_generator:
                print(f"  Generator: {activity.problem_generator.name}")
            print()

        # Test variant selection
        activity = loader.get_current_activity()
        if activity:
            variant = loader.select_variant(activity, strategy="random")
            if variant:
                print(f"Selected variant: {variant.variant_id}")
                print(f"  Approach: {variant.approach}")
                print(f"  Language: {variant.language}")
    else:
        print(f"Example curriculum not found at {example_path}")
