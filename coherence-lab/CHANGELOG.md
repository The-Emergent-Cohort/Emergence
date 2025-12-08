# Coherence Lab Changelog

## [0.5.15] - 2024-12-08
### Added
- **Streak prerequisite for exams**: Must prove consistency before testing
  - L1 exam requires 10 best_streak, L5 requires 50, L10 requires 100
  - Uses best_streak (not current) - once you've proven it, you're eligible
  - Struggling topics need to build streaks before advancing
  - Strong topics naturally meet requirement as they train

## [0.5.14] - 2024-12-08
### Changed
- **Per-topic streak tracking**: Streaks now tracked per-topic, not globally
  - 5 correct on alternating + 5 on repeating is NOT a 10-streak
  - Each topic maintains its own streak independently
  - Show decisions use topic-specific streaks
  - Display now shows `s{current}/{best}` per topic

## [0.5.13] - 2024-12-08
### Changed
- **Streak shows on completion**: Streaks only shown when they END, not mid-run
  - Was: show when streak hits goal (interrupts potentially longer streak)
  - Now: show when streak BREAKS (on failure) or hits mastery (100)
  - Completed streak length stored for XP calculation
  - Enables longer natural streaks without premature interruption
  - Streak shows still internalize feedback (about the completed streak)
### Added
- **Final exam logging**: Final comprehensive exam results now saved to history JSON
- **Raw counts in final exam output**: Shows `32/32 (100%)` to verify fresh inference

## [0.5.12] - 2024-12-08
### Fixed
- **Anti-forgetting maintenance training**: Graduated topics no longer excluded entirely
  - Was: graduated topics completely masked out of training → catastrophic forgetting
  - Now: graduated topics get 10% loss weight (maintenance), active get 100%
  - Prevents model from forgetting patterns while focusing compute on struggling topics
  - Final exam will actually test retained knowledge, not forgotten skills
  - Topic tracker now updates ALL topics to maintain accuracy tracking

## [0.5.11] - 2024-12-08
### Changed
- **No artificial epoch limit**: Default epochs raised from 10 to 100
  - Let the system run until complete, break manually if stuck
  - Some topics genuinely need more time - don't cut them off arbitrarily
  - User can always Ctrl+C if it looks looped

## [0.5.10] - 2024-12-08
### Added
- **Final comprehensive exam**: After all topics graduate individually (L10), must pass final
  - Tests all topics together (32 problems each, 90% threshold)
  - Any topic that fails gets kicked back to L7, must rebuild
  - Phase 1 only completes when ALL topics pass final in same epoch
  - No more "graduate individually and done" - prove it all together

## [0.5.9] - 2024-12-08
### Changed
- **No XP cap**: XP accumulates freely during epoch
  - Removed the cap that limited XP to next exam threshold
  - Exams verify what you've earned - fail penalty drops you back
  - Enables true multi-level advancement per epoch
  - Earn 500 XP? Take exams for all levels up to that threshold

## [0.5.8] - 2024-12-08
### Changed
- **Simplified exam flow**: No more cooldowns, just fail-and-retry
  - XP accumulates during epoch, exams at end
  - First failure = done for this epoch, 25% penalty
  - Failed exam drops you to target threshold minus 25% of excess
  - You don't keep XP above what you couldn't prove
  - Try again next epoch if XP allows

## [0.5.7] - 2024-12-08
### Changed
- **Multiple level-ups per epoch**: Topics can advance as far as they can prove
  - Exam loop continues until topics fail or graduate
  - Focuses compute on topics that need it (fast topics get out of the way)
  - A topic at 100% accuracy can go L1→L10 in one epoch if it passes every exam

## [0.5.6] - 2024-12-08
### Fixed
- **Phase 1 completion requires graduation**: Was ending at 95% accuracy
  - Now requires all topics to PASS L10 EXAM (true graduation)
  - Removes old accuracy/calibration/goal checks
  - Training continues until exam-proven mastery, not just high accuracy

## [0.5.5] - 2024-12-08
### Fixed
- **Streak-mastered topics can now earn XP**: Was blocking XP on `topic_mastered`
  - Changed to `topic_graduated` (exam-proven L10)
  - Streak mastery = sign of readiness, but still need to prove via exams
  - Fixes `repeating` stuck at L1 with 100% accuracy but 10xp

## [0.5.4] - 2024-12-08
### Fixed
- **XP capped at next exam threshold**: No more runaway XP
  - XP cannot exceed next level threshold until you pass the exam
  - Prevents weird "L10 XP but L2 confirmed" states
  - Must earn each level through exams, not just accumulate XP
### Changed
- **Streak reset on failed exam**: Failed exam = back to practice
  - Streak resets to 0 on exam failure
  - XP penalty (25%) still applies
  - Must rebuild streak AND XP to try again

## [0.5.3] - 2024-12-08
### Fixed
- **Only graduated topics excluded from training**: Fixes catastrophic forgetting
  - Changed from `topic_mastered` (streak=100) to `exam_graduated` (passed L10)
  - Failed exams now resume training - topics stay active until truly proven
  - Streak mastery = sign, Exam graduation = proof
### Changed
- **Gentler exploration penalty**: Wrong creative shows now -1 XP (was -3)
  - Don't punish curiosity too harshly - exploration is how you learn
  - Still provides signal that guess was wrong, without discouraging risk-taking

## [0.5.2] - 2024-12-08
### Fixed
- **Exam system uses confirmed_level**: Exams now gate on exam-verified level, not XP-level
  - XP gets you eligible to test, but must pass to advance confirmed_level
  - Fixes bug where XP-level hitting L10 caused "already at max" false negative
- **Flat 25% exam failure penalty**: Simplified from 25-50% variable
### Changed
- **Mastered topics excluded from training**: Compute goes where learning is needed
  - Topics with `topic_mastered=True` skip loss computation, shows, and tracker updates
  - Batches with all mastered topics are skipped entirely
  - No more XP accumulation on graduated/mastered topics
- **Systems module extraction**: Reusable components in `systems/`
  - `progression.py`: XP/level tracking (standalone)
  - `examination.py`: Exam logic (standalone)
  - `logging.py`: Standardized output formatting

## [0.5.1] - 2024-12-08
### Changed
- **Teacher-validated XP**: Removed auto-XP from practice
  - Practice alone doesn't level you up - must engage with teacher
  - XP only comes from shows: creative, streak, validation
- **Calibration modifier**: Uncalibrated topics get 75% XP
  - If teacher thinks you're "guessing" or "overconfident", XP reduced
  - Not a punishment, just "let's make sure you understand"
- **Validation shows now +1 XP**: Asking for help = engaging = rewarded

## [0.5.0] - 2024-12-08
### Added
- **Examination System**: Level transitions require passing an exam
  - No auto-leveling on XP threshold - must prove competence
  - Binary-scaled exam sizes: L1-3=8, L4-6=16, L7-9=32, L10=64 problems
  - Pass thresholds scale with level: 75%→80%→85%→90%
  - Failure penalty: 25-50% of current level XP (based on how badly failed)
  - Cooldown: 1-10 epochs before retry (scales with level)
  - L10 pass = **GRADUATED** - topic done until final comprehensive
- **Formative assessment framing**: "Ready" vs "More practice needed" (not pass/fail)
- Exam stats tracking: attempts, passes, graduated status
- Enables future instructor/class models with standardized skill measurement

## [0.4.2] - 2024-12-08
### Changed
- **Pure level scaling**: Simplified XP formula to `actual_xp = base / level`
  - Removed static topic difficulty - difficulty is now relative to YOUR level
  - RPG-style: what's hard at L1 is trivial at L10 (same content, different XP)
  - Penalties still NOT scaled (always hurt the same)
- **Removed farming detection**: Level scaling naturally prevents farming
  - High level = diminishing returns = move on to harder content
  - No need for explicit penalties

## [0.4.1] - 2024-12-08
### Added
- **XP Scaling Formula**: `actual_xp = base × difficulty / level`
  - Prevents low-level farming: grinding easy content gives diminishing returns
  - Topic difficulties: alternating=1.0, repeating=1.0, incrementing=1.5, fixed_offset=2.0, periodic_repeat=2.5
  - Level scaling: L5 on easy topic gets 1/5th base XP
- **Farming Detection**: Teacher penalizes wasting time on mastered easy topics
  - If level >= 4 AND difficulty <= 1.5 AND showing validation/spontaneous: -2 XP
  - "You already know this. Move on."
  - Note: Penalties are NOT scaled (always hurt the same)

## [0.4.0] - 2024-12-08
### Added
- **XP (Experience Points) System**: Per-topic skill tracking with geometric leveling
  - Correct answers: +1 XP (base accumulation)
  - Creative shows (correct): +5 XP (validated insight)
  - Creative shows (wrong): -3 XP (overconfidence cost - self-competition)
  - Streak shows: +streak_length/5 XP (consistency bonus)
  - Validation shows: 0 XP (neutral - asking for help is fine)
  - Level thresholds: N² × 10 (L1=10, L2=40, L3=90, L4=160... L10=1000)
  - Early levels are quick, mastery requires sustained proof
  - XP can go down (but not below 0) - honest self-assessment
- **Visual level display**: `L 5 █████░····` shows level + progress to next
- **Total XP and average level** displayed per epoch

## [0.3.5] - 2024-12-08
### Fixed
- **Student goal estimate clamp**: Was capped at 10, now caps at 100 (mastery)
  - Student couldn't learn goals above 10, but teacher could set up to 100
  - This caused 32/10 disconnects where student fell behind teacher's expectations
  - Now student can grow to match teacher throughout the learning journey

## [0.3.4] - 2024-12-08
### Added
- **Per-topic streak display**: Epoch output now shows `streak:current/best` for each topic
  - 'M' symbol for mastered topics (100 consecutive correct)
  - Streak info also saved to run logs for analysis

## [0.3.3] - 2024-12-08
### Fixed
- **Goals only on streaks**: Goal increases now ONLY trigger on streak shows
  - Creative, validation, and spontaneous shows don't raise the bar
  - Streaks are the actual goal achievement - "get X correct in a row, then show"
  - This matches the pedagogical intent: prove consistent competence, not just enthusiasm

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
