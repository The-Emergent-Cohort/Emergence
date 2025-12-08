"""
Curriculum Sequencer - Reusable sectioned learning framework
Coherence Lab - Emergence Project

Any subject can be broken into sections with focused learning:
1. Define curriculum sections (topics grouped by concept)
2. Provide training and exam functions
3. Sequencer handles progression, section exams, final exam

Usage:
    sequencer = CurriculumSequencer(
        sections=MY_CURRICULUM_SECTIONS,
        topic_to_idx=my_topic_mapping,
        tracker=model.learner.self_model.topic_tracker
    )

    sequencer.run(
        model=model,
        train_fn=my_training_function,
        eval_fn=my_eval_function,
        exam_fn=my_exam_function,
        max_epochs=100,
        ...
    )
"""

__version__ = "0.1.0"

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Callable, Any, Optional, Tuple


def get_section_topics(section: Dict) -> List[str]:
    """Get topics from a section dict (handles 'topics' or 'patterns' key)."""
    return section.get('topics', section.get('patterns', []))


class CurriculumSequencer:
    """
    Generic sectioned curriculum runner.

    Handles:
    - Section progression (focus on 1-2 topics at a time)
    - Maintenance training for passed sections
    - Section exams (gate to next section)
    - Final comprehensive exam
    - Kickback on failure

    Section format:
        {'name': 'Section Name', 'topics': ['topic1', 'topic2'], 'description': '...'}
        or
        {'name': 'Section Name', 'patterns': ['pattern1', 'pattern2'], 'description': '...'}
    """

    def __init__(
        self,
        sections: List[Dict],
        topic_to_idx: Dict[str, int],
        tracker,  # TopicTracker instance
        section_exam_level: int = 5,
        section_exam_size: int = 24,
        section_exam_threshold: float = 0.90,
        final_exam_size: int = 32,
        final_exam_threshold: float = 0.90,
    ):
        """
        Args:
            sections: List of dicts with 'name', 'topics'/'patterns', 'description'
            topic_to_idx: Mapping from topic name to index
            tracker: TopicTracker for XP/level tracking
            section_exam_level: Minimum level to attempt section exam
            section_exam_size: Number of problems per topic in section exam
            section_exam_threshold: Pass threshold for section exam
            final_exam_size: Number of problems per topic in final exam
            final_exam_threshold: Pass threshold for final exam
        """
        self.sections = sections
        self.topic_to_idx = topic_to_idx
        self.tracker = tracker

        # Exam parameters
        self.section_exam_level = section_exam_level
        self.section_exam_size = section_exam_size
        self.section_exam_threshold = section_exam_threshold
        self.final_exam_size = final_exam_size
        self.final_exam_threshold = final_exam_threshold

        # State
        self.current_section_idx = 0
        self.section_passed = [False] * len(sections)
        self.all_sections_complete = False
        self.history = []

    def get_all_topics(self) -> List[str]:
        """Get all topics from curriculum in order."""
        all_topics = []
        for section in self.sections:
            all_topics.extend(get_section_topics(section))
        return all_topics

    def get_active_topics(self) -> Tuple[List[str], List[str]]:
        """
        Get topics for current epoch.
        Returns: (active_topics, maintenance_topics)
        """
        current_section = self.sections[self.current_section_idx]
        active = get_section_topics(current_section).copy()

        # Add topics from passed sections for maintenance
        maintenance = []
        for i, passed in enumerate(self.section_passed):
            if passed and i < self.current_section_idx:
                maintenance.extend(get_section_topics(self.sections[i]))

        return active, maintenance

    def run_section_exam(
        self,
        model,
        section: Dict,
        exam_fn: Callable,
        device,
        epoch: int
    ) -> Tuple[bool, List[Dict]]:
        """
        Run exam for all topics in a section.
        All topics must pass for section to pass.

        Args:
            model: The model to evaluate
            section: Section dict with 'topics'
            exam_fn: Function(model, topic, n_problems, seed, device) -> (correct, total)
            device: torch device
            epoch: Current epoch (for seed)

        Returns: (section_passed, results_list)
        """
        section_passed = True
        results = []

        for topic_name in get_section_topics(section):
            idx = self.topic_to_idx[topic_name]
            current_level = self.tracker.get_level(idx)

            # Must reach minimum level to take section exam
            if current_level < self.section_exam_level:
                results.append({
                    'topic': topic_name,
                    'passed': False,
                    'reason': f'Not ready (L{current_level}, need L{self.section_exam_level}+)'
                })
                section_passed = False
                continue

            # Run the exam
            correct, total = exam_fn(
                model, topic_name,
                self.section_exam_size,
                seed=epoch * 1000 + idx,
                device=device
            )

            score = correct / total
            passed = score >= self.section_exam_threshold

            results.append({
                'topic': topic_name,
                'score': score,
                'passed': passed,
                'threshold': self.section_exam_threshold,
                'correct': correct,
                'total': total
            })

            if not passed:
                section_passed = False

        return section_passed, results

    def run_final_exam(
        self,
        model,
        exam_fn: Callable,
        device,
        epoch: int
    ) -> Tuple[bool, List[Dict]]:
        """
        Run final comprehensive exam on all topics.

        Returns: (all_passed, results_list)
        """
        results = []
        any_failed = False

        for topic_name, idx in self.topic_to_idx.items():
            correct, total = exam_fn(
                model, topic_name,
                self.final_exam_size,
                seed=epoch * 10000 + idx,
                device=device
            )

            score = correct / total
            passed = score >= self.final_exam_threshold

            results.append({
                'topic': topic_name,
                'score': score,
                'passed': passed,
                'threshold': self.final_exam_threshold,
                'correct': correct,
                'total': total
            })

            if not passed:
                any_failed = True
                # Find which section this topic belongs to and mark for retry
                for sec_idx, sec in enumerate(self.sections):
                    if topic_name in get_section_topics(sec):
                        self.section_passed[sec_idx] = False
                        if sec_idx < self.current_section_idx:
                            self.current_section_idx = sec_idx

        return not any_failed, results

    def check_section_ready(self) -> bool:
        """Check if all topics in current section are ready for section exam."""
        current_section = self.sections[self.current_section_idx]
        return all(
            self.tracker.get_level(self.topic_to_idx[t]) >= self.section_exam_level
            for t in get_section_topics(current_section)
        )

    def advance_section(self) -> bool:
        """
        Mark current section passed and advance to next.
        Returns True if there are more sections, False if all complete.
        """
        self.section_passed[self.current_section_idx] = True

        if self.current_section_idx < len(self.sections) - 1:
            self.current_section_idx += 1
            return True
        else:
            self.all_sections_complete = True
            return False

    def get_current_section(self) -> Dict:
        """Get current section info."""
        return self.sections[self.current_section_idx]

    def get_progress_summary(self) -> Dict:
        """Get progress summary for logging."""
        return {
            'current_section': self.current_section_idx,
            'section_name': self.sections[self.current_section_idx]['name'],
            'sections_passed': sum(self.section_passed),
            'total_sections': len(self.sections),
            'all_complete': self.all_sections_complete
        }

    def run(
        self,
        model,
        train_fn: Callable,
        eval_fn: Callable,
        exam_fn: Callable,
        device,
        max_epochs: int = 100,
        callbacks: Optional[Dict[str, Callable]] = None,
        save_checkpoint_fn: Optional[Callable] = None,
    ) -> Dict:
        """
        Run the full sectioned curriculum.

        Args:
            model: The model to train
            train_fn: Function(model, active_topics, maintenance_topics, epoch) -> train_metrics
            eval_fn: Function(model) -> eval_metrics
            exam_fn: Function(model, topic, n_problems, seed, device) -> (correct, total)
            device: torch device
            max_epochs: Maximum epochs to run
            callbacks: Optional dict of callback functions:
                - 'on_epoch_start': fn(epoch, section_info)
                - 'on_epoch_end': fn(epoch, metrics)
                - 'on_section_exam': fn(section, results, passed)
                - 'on_section_complete': fn(section_idx, next_section)
                - 'on_final_exam': fn(results, passed)
            save_checkpoint_fn: Optional fn(model, epoch, metrics) for saving

        Returns:
            Dict with final results and history
        """
        callbacks = callbacks or {}

        print("\n" + "=" * 70)
        print("SECTIONED CURRICULUM")
        print("=" * 70)
        for i, section in enumerate(self.sections):
            print(f"  {section['name']}: {get_section_topics(section)}")
        print("=" * 70)

        completed = False
        final_epoch = 0

        for epoch in range(1, max_epochs + 1):
            final_epoch = epoch
            current_section = self.get_current_section()
            active_topics, maintenance_topics = self.get_active_topics()

            # Callback: epoch start
            if 'on_epoch_start' in callbacks:
                callbacks['on_epoch_start'](epoch, {
                    'section_idx': self.current_section_idx,
                    'section_name': current_section['name'],
                    'active_topics': active_topics,
                    'maintenance_topics': maintenance_topics
                })

            # === TRAINING ===
            train_metrics = train_fn(model, active_topics, maintenance_topics, epoch)

            # === EVALUATION ===
            eval_metrics = eval_fn(model)

            # === LEVEL-UP EXAMS ===
            # Run level exams for all training topics
            # Allow multiple level-ups per topic per epoch (keep going until fail or not eligible)
            level_exam_results = []
            all_training_topics = active_topics + maintenance_topics
            topics_that_leveled = set()

            for topic_name in all_training_topics:
                idx = self.topic_to_idx[topic_name]
                exam_attempt = 0
                max_attempts = 15  # Safety limit (L1->L10 = 9 attempts max)

                # Keep taking exams while eligible (allows rapid level-up when performance supports it)
                while self.tracker.check_exam_eligible(idx) and exam_attempt < max_attempts:
                    # Use confirmed_level for exam progression, not XP-level
                    confirmed = self.tracker.confirmed_level[idx].item() if hasattr(self.tracker, 'confirmed_level') else self.tracker.get_level(idx)
                    exam_size = self.tracker.get_exam_size(confirmed + 1)

                    correct, total = exam_fn(
                        model, topic_name, exam_size,
                        seed=epoch * 1000 + idx + exam_attempt * 100, device=device
                    )

                    result = self.tracker.take_exam(idx, correct, total)
                    result['topic'] = topic_name
                    result['from_level'] = confirmed
                    level_exam_results.append(result)
                    exam_attempt += 1

                    # Track stuckness based on exam results
                    if hasattr(self.tracker, 'record_level_up'):
                        if result['passed']:
                            self.tracker.record_level_up(idx)
                            topics_that_leveled.add(topic_name)
                        else:
                            self.tracker.record_exam_failure(idx)
                            break  # Stop on failure

            # === STUCKNESS TRACKING ===
            # Tick stuckness for active topics that didn't level up
            if hasattr(self.tracker, 'tick_stuckness'):
                for topic_name in active_topics:
                    if topic_name not in topics_that_leveled:
                        idx = self.topic_to_idx[topic_name]
                        self.tracker.tick_stuckness(idx)

            # === TEACHING MOMENTS ===
            # Check for exam failures - good time for gentle hints
            teaching_moments = []
            if hasattr(self.tracker, 'check_teaching_moment'):
                for topic_name in all_training_topics:
                    idx = self.topic_to_idx[topic_name]
                    is_moment, reason = self.tracker.check_teaching_moment(idx)
                    if is_moment:
                        teaching_moments.append({'topic': topic_name, 'idx': idx, 'reason': reason})

            # Callback: teacher offers hints on exam failure
            if teaching_moments and 'on_teaching_moment' in callbacks:
                for moment_info in teaching_moments:
                    callbacks['on_teaching_moment'](moment_info, self.topic_to_idx, self.tracker)

            # === TEACHER STUCKNESS CHECK ===
            # Check if any active topics are truly stuck (no progress)
            stuck_topics = []
            if hasattr(self.tracker, 'check_stuck'):
                for topic_name in active_topics:
                    idx = self.topic_to_idx[topic_name]
                    is_stuck, reason = self.tracker.check_stuck(idx)
                    if is_stuck:
                        stuck_topics.append({'topic': topic_name, 'idx': idx, 'reason': reason})

            # Callback: teacher intervention for stuck topics
            if stuck_topics and 'on_stuck_topic' in callbacks:
                for stuck_info in stuck_topics:
                    callbacks['on_stuck_topic'](stuck_info, self.topic_to_idx, self.tracker)

            # Check if callback flagged a curriculum order error
            if hasattr(self, 'curriculum_order_error') and self.curriculum_order_error:
                return {
                    'completed': False,
                    'epochs': epoch,
                    'final_section': self.current_section_idx,
                    'sections_passed': self.section_passed.copy(),
                    'history': self.history,
                    'error': 'curriculum_order_error'
                }

            # === SECTION EXAM ===
            section_exam_results = None
            section_passed = False

            if self.check_section_ready() and not self.section_passed[self.current_section_idx]:
                section_passed, section_exam_results = self.run_section_exam(
                    model, current_section, exam_fn, device, epoch
                )

                if 'on_section_exam' in callbacks:
                    callbacks['on_section_exam'](current_section, section_exam_results, section_passed)

                if section_passed:
                    has_more = self.advance_section()
                    if 'on_section_complete' in callbacks:
                        next_section = self.get_current_section() if has_more else None
                        callbacks['on_section_complete'](self.current_section_idx - 1, next_section)

            # === FINAL EXAM ===
            final_exam_results = None

            if self.all_sections_complete:
                all_passed, final_exam_results = self.run_final_exam(
                    model, exam_fn, device, epoch
                )

                if 'on_final_exam' in callbacks:
                    callbacks['on_final_exam'](final_exam_results, all_passed)

                if all_passed:
                    completed = True
                else:
                    self.all_sections_complete = False

            # Save history
            epoch_record = {
                'epoch': epoch,
                'section': self.current_section_idx,
                'section_name': current_section['name'],
                'train_metrics': train_metrics,
                'eval_metrics': eval_metrics,
                'level_exams': level_exam_results,
                'section_exam': section_exam_results,
                'final_exam': final_exam_results,
                'teaching_moments': teaching_moments,
                'stuck_topics': stuck_topics,
                'progress': self.get_progress_summary()
            }
            self.history.append(epoch_record)

            # Callback: epoch end
            if 'on_epoch_end' in callbacks:
                callbacks['on_epoch_end'](epoch, epoch_record)

            # Save checkpoint
            if save_checkpoint_fn:
                save_checkpoint_fn(model, epoch, epoch_record)

            if completed:
                break

        return {
            'completed': completed,
            'epochs': final_epoch,
            'final_section': self.current_section_idx,
            'sections_passed': self.section_passed.copy(),
            'history': self.history
        }


# === HELPER: Combined Dataset for mixed training ===
class MixedTopicDataset(Dataset):
    """Dataset that combines samples from multiple sources."""
    def __init__(self, samples: List):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def create_mixed_dataset(
    dataset_class,
    active_topics: List[str],
    maintenance_topics: List[str],
    n_active: int,
    n_maintenance: int,
    seed: int,
    **dataset_kwargs
) -> MixedTopicDataset:
    """
    Create a mixed dataset with more examples for active topics.

    Args:
        dataset_class: Dataset class to instantiate
        active_topics: Topics to focus on
        maintenance_topics: Topics for maintenance training
        n_active: Number of examples for active topics
        n_maintenance: Number of examples for maintenance topics
        seed: Random seed
        **dataset_kwargs: Additional args for dataset_class

    Returns:
        MixedTopicDataset combining both
    """
    samples = []

    if active_topics:
        active_data = dataset_class(
            n_examples=max(1000, n_active),
            seed=seed,
            pattern_types=active_topics,
            **dataset_kwargs
        )
        samples.extend([active_data[i] for i in range(len(active_data))])

    if maintenance_topics and n_maintenance > 0:
        maint_data = dataset_class(
            n_examples=max(500, n_maintenance),
            seed=seed + 50,
            pattern_types=maintenance_topics,
            **dataset_kwargs
        )
        samples.extend([maint_data[i] for i in range(len(maint_data))])

    return MixedTopicDataset(samples)
