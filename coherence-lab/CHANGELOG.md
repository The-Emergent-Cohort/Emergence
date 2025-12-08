# Coherence Lab Changelog

## [0.3.0] - 2024-12-08
### Fixed
- **Overflow protection**: Added defensive clamping throughout goal-setting code
  - `evaluate_student_goal_proposal`: competence (0-1), goals (1-1000)
  - `set_new_goal`: same clamping + sanity cap at 1000
  - `phase1_approval.py`: safety net on direct goal fills
- **Goal dynamics**: Goals now start reasonable (~9) instead of jumping to 62

### Added
- **Version tracking**: `__version__` in phase1_approval.py, included in run logs
- **Per-topic streak tracking** in TopicConfidenceTracker:
  - `topic_streak`: current consecutive correct per topic
  - `topic_best_streak`: best ever per topic
  - `topic_mastered`: flag when streak hits 100
- **Scaled goal increments**: 25-50% increase based on impressedness (was flat +1)
- **Topic calibration logging**: accuracy vs confidence gap per topic

### Changed
- `perceived_competence` now updated from ALL items (in `monitor_unshown_work`), not just shown items
- Streak counter only resets on streak-shows, not creative/validation shows

## [0.2.0] - 2024-12-07
### Added
- Dynamic Topic Registry for tracking curriculum patterns
- Show reason priority: streak > creative > validation > spontaneous
- Creativity confidence gate (requires confidence > 0.8)
- Teacher goal-raising based on impressedness

### Fixed
- Teacher goal logic - was raising goals incorrectly
- Streak tracking - creative shows were breaking streaks

## [0.1.0] - 2024-12-06
### Initial Implementation
- RelationalSystem with Self/Other/World + Temporality architecture
- Phase 1 approval-seeking training loop
- Teacher-student dynamics with trust building
- Basic curriculum: alternating, repeating, incrementing, fixed_offset patterns
