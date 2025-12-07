# Approval-Seeking Mechanism: Design Notes

## Current Architecture Summary

### Learner (RelationalLearner)
- **SelfModel**: emotions (confidence, frustration, curiosity), topic_tracker
- **OtherModel**: trust, internalization_level, consult_inner_guide()
- **TemporalModel**: continuity, episodic memory, day/night cycle
- Perceives world, thinks in steps, outputs answers

### Teacher (TeacherGuide)
- Observes learner (visible state only, not internal)
- Decides interventions based on learner emotions
- Generates: hints, encouragement, explanations, creativity rewards, habit corrections
- ProcessEvaluator watches HOW learner thinks

### Current Intervention Logic (PASSIVE)
```python
# Teacher intervenes when:
needs_help = (frustration > 0.7) & (confidence < 0.3)
```
- Teacher watches and may intervene when student struggles
- Student never initiates interaction
- No positive reinforcement loop for successes

---

## Problem with Current Design

1. **0% interventions** - frustration threshold never met because model is confidently wrong
2. **Trust eroding** - no positive interactions to build trust
3. **Internalization stuck** - requires teacher messages which require interventions
4. **Missing social learning** - student doesn't seek approval, doesn't learn when to share

---

## Proposed: Approval-Seeking Mechanism

### Core Concept
Student actively decides when to "show teacher" work. This builds:
- Social judgment (when to share vs work independently)
- Trust through positive interactions
- Approval calibration (what's worth sharing)

### Key Insight from Patrick
> "The student also wants to show 'creative' ways to solve it or runs of correct answers, not every problem. It's fine-tuning approval seeking from the start."

> "Sometimes creative is creative, but doesn't suit the need. The impulse is important, but that's channeling it."

---

## Implementation Plan

### 1. New Emotions in SelfModel

```python
# Add to SelfModel:
self.pride = nn.Linear(d_model, 1)  # "I did something worth showing"
self.approval_desire = nn.Linear(d_model, 1)  # "I want feedback"
```

### 2. Show-Work Decision Logic

Student decides to show when:
- **Creative approach**: novel path to correct answer
- **Streak of success**: N correct in a row (pride builds)
- **Uncertain but correct**: seeking validation ("was I right?")
- **Spontaneous sharing**: random sampling (natural child behavior)

NOT every correct answer. Calibration develops over time.

```python
def should_show_work(self, was_correct, is_creative, streak_count, confidence):
    """
    Decide whether to present work to teacher.

    Early: show more often (learning the dynamic)
    Later: more selective (calibrated sharing)
    """
    pride = self.self_model.get_pride(cognitive_state)

    # Definite shows
    if is_creative and was_correct:
        return True, 'creative'
    if streak_count >= 3:
        return True, 'streak'

    # Uncertain seeking validation
    if was_correct and confidence < 0.5:
        return True, 'validation_seeking'

    # Spontaneous (decreases as internalization grows)
    spontaneous_rate = 0.1 * (1 - internalization_level)
    if random.random() < spontaneous_rate:
        return True, 'spontaneous'

    return False, None
```

### 3. Teacher Response (Always Positive)

Even corrections are framed positively:

```python
def respond_to_shown_work(self, student_work, was_correct, show_reason):
    """
    Respond to student presenting work.

    Always positive framing, even for corrections.
    """
    if was_correct:
        if show_reason == 'creative':
            # Channel creativity appropriately
            return self.generate_creativity_acknowledgment(student_work)
        elif show_reason == 'streak':
            return self.generate_streak_celebration(student_work)
        else:
            return self.generate_approval(student_work)
    else:
        # Wrong but showed - still positive
        # "Great that you're checking! Let's look at this together..."
        return self.generate_gentle_guidance(student_work)
```

### 4. Trust & Approval Building

```python
def update_after_showing(self, teacher_response_was_positive):
    """
    Update trust and approval-seeking calibration.
    """
    if teacher_response_was_positive:
        self.other_model.update_trust(outcome_was_good=True)
        # Reinforce this type of showing
        self.approval_calibration[show_reason] += 0.1
```

---

## Confidence Hierarchy (from conversation)

```
Self-confidence (global)
    aggregates from subjects
    projects down as priors for new topics

└── Subject: Math/Pattern Functions
    ├── Topic: compositional (strong)
    ├── Topic: long_range (strong)
    └── Topic: fibonacci_like (weak)
```

- **Bottom-up**: topic confidences aggregate to subject
- **Top-down**: subject confidence provides priors for new topics
- "If you can code, you can code" - expertise enables transfer

---

## Phase 1 Implementation (Easy Patterns)

For phase 1, keep it simple:
1. Binary entities: student + teacher
2. Always positive feedback
3. Student shows:
   - Creative solutions
   - Streaks (3+ correct)
   - Sometimes randomly (learning the dynamic)
4. Teacher responds:
   - Approval ("Great job!")
   - Guided approval ("Good thinking! Here's another way to see it...")
5. Trust builds through positive interactions

### Metrics to Track
- Show rate: how often student presents
- Approval rate: teacher positive responses
- Trust trajectory: building not eroding
- Calibration: are shows becoming more selective?

---

## Later Phases (Multiple Reviewers)

As complexity grows:
- Multiple teachers/reviewers
- Consensus building
- "Let me check with another expert"
- Different experts for different topics

This maps to real learning:
- Early: parent/single teacher
- Later: multiple teachers, peers, mentors
- Adult: self + consulting experts as needed

---

## Key Files to Modify

1. **relational_model.py**
   - SelfModel: add pride, approval_desire
   - Add show_work decision logic
   - OtherModel: add respond_to_shown_work
   - Update trust/approval calibration

2. **hard_patterns.py** (and similar training scripts)
   - Integrate show-work into training loop
   - Track show/approval metrics
   - Log approval-seeking calibration

3. **New: phase1_approval.py**
   - Clean phase 1 with approval-seeking from start
   - Easy patterns to build the dynamic
   - Foundation for harder phases

---

## Questions to Consider

1. How often should early-stage student show? (Start high, calibrate down?)
2. Should teacher ever NOT respond? (Probably not in phase 1)
3. How to balance show-seeking with independent work?
4. When does showing become "seeking too much approval"?

---

## Next Steps

1. Implement pride/approval_desire in SelfModel
2. Add show_work decision logic
3. Add teacher response methods
4. Create phase1_approval.py for clean testing
5. Run and iterate
