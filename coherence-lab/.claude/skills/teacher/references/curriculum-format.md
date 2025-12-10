# Curriculum Format Quick Reference

Full spec at `research/curriculum_format_spec.md`. Here's what you need in practice.

## Hierarchy

```
Curriculum → Subject → Course → Unit → Lesson → Activity
```

Each Activity has what you need to teach that moment.

## Activity Structure

```json
{
  "activity_id": "math.counting.unit01.lesson01.act01",
  "activity_type": "demonstration | guided_practice | play | ...",
  "variants": [
    {
      "variant_id": "claude_en",
      "source_model": "claude",
      "language": "en-US",
      "approach": "visual-spatial | procedural | narrative | ...",
      "explanation": "The actual teaching content"
    }
  ],
  "extensions": {
    "pedagogy": {
      "cpa_stage": "concrete | pictorial | abstract",
      "instructional_phase": "i_do | we_do | you_do",
      "scaffolding_level": 0.0-1.0
    }
  }
}
```

## Picking Variants

- First time: random (no single model's bias)
- On struggle: pick different `approach` tag
- Track what works for this student

## CPA Stages

| Stage | What it means | Examples |
|-------|---------------|----------|
| Concrete | Physical/tangible | Blocks, counting objects |
| Pictorial | Visual representation | Diagrams, bar models |
| Abstract | Symbolic | Numbers, equations |

Progress through these in order within a topic.

## Instructional Phases

| Phase | You do | They do |
|-------|--------|---------|
| I Do | Demonstrate fully | Watch |
| We Do | Guide with prompts | Try with support |
| You Do | Monitor | Work independently |

Scaffolding decreases as you move through these.

## Play Activities

Different from instruction - no "correct" answer:

```json
{
  "activity_type": "play",
  "play_type": "free_exploration | guided_discovery | structured_play",
  "environment": {
    "type": "physics_playground",
    "entities": ["block", "ball", "ramp"]
  },
  "discovery_targets": ["objects_fall_down", "stacking_requires_balance"]
}
```

Your role during play: mostly observe, occasional gentle questions.
