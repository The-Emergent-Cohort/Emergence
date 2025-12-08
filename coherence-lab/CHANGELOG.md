# Coherence Lab Changelog

## [0.3.2] - 2024-12-08
### Fixed
- **Boredom re-triggering**: Teacher boredom (`impressedness < 0.5`) was causing
  `should_set_new_goal()` to fire on every shown item, escalating 3→8→13→...→100 instantly
  - Setting a new goal now boosts `impressedness` to at least 0.6 (teacher becomes engaged)
  - This prevents the boredom trigger from repeatedly firing

## [0.3.1] - 2024-12-08
### Fixed
- **Goal runaway bug**: Compound scaling (1.25-1.50x) caused goals to explode 3→8→12→18→...→1000
  - Changed to additive increments: +2 to +5 based on impressedness
  - Now: 3→8→13→18→23→... (linear, controllable)
- **Missing reset**: `set_new_goal` now resets `goals_met_count` to prevent immediate re-triggering
- **Mastery cap**: Changed from 1000 to 100 everywhere (100 consecutive = mastery)

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
