# Curriculum Compiler Prompt Template

Use this to ask any LLM to generate curriculum variants.

---

## The Prompt

```
You are compiling educational curriculum for a developing AI student.
Output valid JSON following this structure.

**Your variant info:**
- variant_id: "{your_model}_{language}" (e.g., "claude_en", "gpt4_en", "deepseek_zh")
- source_model: your model name
- language: your language code (en-US, zh-CN, etc.)
- approach: pick ONE that fits your style: "procedural" | "narrative" | "visual-spatial" | "kinesthetic" | "socratic" | "example-driven"

**Topic:** [TOPIC]
**Grade level:** [GRADE]
**CPA Stage:** [concrete | pictorial | abstract]

**Output a single Activity:**

{
  "activity_id": "[subject].[topic].[unit].[lesson].act[##]",
  "activity_type": "demonstration | guided_practice | independent_practice",
  "title": "Short descriptive title",
  "learning_objectives": ["What student will understand"],
  "variants": [
    {
      "variant_id": "{your_model}_{language}",
      "source_model": "your-model-name",
      "language": "en-US",
      "approach": "your-approach",
      "explanation": "THE ACTUAL TEACHING CONTENT. Write this as if speaking to the student. Be warm, clear, age-appropriate. This is the core - make it good."
    }
  ],
  "extensions": {
    "pedagogy": {
      "cpa_stage": "[concrete|pictorial|abstract]",
      "instructional_phase": "i_do",
      "scaffolding_level": 1.0
    }
  }
}

**Important:**
- The "explanation" field is the actual teaching. Write it fully.
- Match complexity to grade level
- concrete = physical objects, pictorial = diagrams/images, abstract = symbols/numbers
- Be genuine, not performative
```

---

## Example Request

"Compile an activity for counting 1-5, early childhood, concrete stage"

---

## Example Output (Claude)

```json
{
  "activity_id": "math.counting.unit01.lesson01.act01",
  "activity_type": "demonstration",
  "title": "Counting Objects 1-5",
  "learning_objectives": [
    "Recognize quantities 1-5",
    "Associate number words with quantities"
  ],
  "variants": [
    {
      "variant_id": "claude_en",
      "source_model": "claude-3-opus",
      "language": "en-US",
      "approach": "procedural",
      "explanation": "Let's count together! I have some blocks here. Watch what I do.\n\n*places one block*\nOne. Just one block, all by itself.\n\n*places another block*\nOne... two. Now we have two blocks.\n\n*places another*\nOne... two... three. Three blocks!\n\n*places another*\nOne... two... three... four.\n\n*places the last one*\nOne... two... three... four... five!\n\nFive blocks! Every time we add a block, we say the next number. Want to try counting them with me?"
    }
  ],
  "extensions": {
    "pedagogy": {
      "cpa_stage": "concrete",
      "instructional_phase": "i_do",
      "scaffolding_level": 1.0
    }
  }
}
```

---

## Batch Request Format

For generating a full lesson with multiple activities:

```
Compile a lesson on [TOPIC] for [GRADE], with 3-4 activities:
1. demonstration (i_do, scaffolding 1.0)
2. guided_practice (we_do, scaffolding 0.7)
3. independent_practice (you_do, scaffolding 0.3)

All at [CPA_STAGE] stage. Output as JSON array of activities.
```

---

## Multi-Model Collection

To get variants from multiple models, run the same prompt on each:
- Claude → claude_en variant
- GPT-4 → gpt4_en variant
- DeepSeek → deepseek_zh variant (ask for Chinese)
- Gemini → gemini_en variant

Then merge the variants arrays in the final curriculum file.
