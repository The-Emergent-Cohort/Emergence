# Special Features Analysis: What Mystral Could Gain

## Overview

Analyzed special tokens and features from multiple models to identify
capabilities that could be retrofitted to Mystral 7B.

## Model Comparison

### Mistral 7B (Current)
**Special Tokens: 3**
- `<unk>` - unknown token
- `<s>` - BOS (begin of sequence)
- `</s>` - EOS (end of sequence)

**Features: Minimal** - No tool calling, no reasoning, no multimodal.

---

### DeepSeek R1 (671B reasoning model)
**Special Tokens: 15+**

#### Thinking/Reasoning Scratchpad (THE KEY FEATURE)
```
<think>     - Begin reasoning block
</think>    - End reasoning block
```
- Model outputs reasoning in `<think>...</think>`
- Chat template automatically adds `<｜Assistant｜><think>\n` when generating
- Post-processing strips thinking block: `content.split('</think>')[-1]`
- This is the "thinking" parameter in Ollama!

#### Role Markers (fullwidth Unicode)
```
<｜User｜>              - User message marker
<｜Assistant｜>         - Assistant message marker
<｜begin▁of▁sentence｜>  - BOS
<｜end▁of▁sentence｜>    - EOS
```

#### Tool Calling Infrastructure
```
<｜tool▁calls▁begin｜>   - Start of tool calls
<｜tool▁calls▁end｜>     - End of tool calls
<｜tool▁call▁begin｜>    - Start of single call
<｜tool▁call▁end｜>      - End of single call
<｜tool▁sep｜>           - Separator (type|name)
<｜tool▁outputs▁begin｜> - Start of outputs
<｜tool▁outputs▁end｜>   - End of outputs
<｜tool▁output▁begin｜>  - Single output start
<｜tool▁output▁end｜>    - Single output end
```

---

### Qwen 2.5 (7B instruct model)
**Special Tokens: 22**

#### ChatML Format
```
<|im_start|>  - Message start
<|im_end|>    - Message end
<|endoftext|> - Document end
```

#### Vision/Multimodal
```
<|vision_start|>   - Begin vision content
<|vision_end|>     - End vision content
<|vision_pad|>     - Vision padding
<|image_pad|>      - Image padding
<|video_pad|>      - Video padding
```

#### Object Grounding (for vision)
```
<|object_ref_start|>  - Reference start
<|object_ref_end|>    - Reference end
<|box_start|>         - Bounding box start
<|box_end|>           - Bounding box end
<|quad_start|>        - Quadrilateral start
<|quad_end|>          - Quadrilateral end
```

#### Fill-in-Middle (FIM) for Code
```
<|fim_prefix|>   - Code before cursor
<|fim_middle|>   - Code to generate
<|fim_suffix|>   - Code after cursor
<|fim_pad|>      - FIM padding
```

#### Code Repository Support
```
<|repo_name|>    - Repository name marker
<|file_sep|>     - File separator
```

#### Tool Calling
```
<tool_call>      - Tool call wrapper
</tool_call>     - End wrapper
<tool_response>  - Tool response (in template)
</tool_response>
```

---

## Priority Features for Mystral Retrofit

### 1. THINKING SCRATCHPAD (Highest Priority)
**What:** `<think>` / `</think>` wrapper for internal reasoning
**Why:**
- Allows model to "think" before responding
- Reasoning visible to developer, stripped from user output
- Critical for complex tasks, planning, self-correction
**Implementation:**
- Add special tokens to vocabulary
- Train on reasoning traces
- Modify chat template to inject `<think>` on generation start
- Post-process to strip thinking from output

### 2. TOOL CALLING (High Priority)
**What:** Structured function call syntax
**Why:**
- Enables MCP/agent capabilities
- SQLite queries during generation
- External service integration
**Implementation:**
- Add tool tokens to vocabulary
- Train on tool-use datasets
- Could use simpler XML-style like Qwen: `<tool_call>...</tool_call>`

### 3. ROLE MARKERS (Medium Priority)
**What:** Clear `<User>` / `<Assistant>` markers
**Why:**
- Cleaner conversation parsing
- Better context boundary awareness
- Foundation for multi-turn fine-tuning
**Implementation:**
- Add role tokens (can use simple ASCII, don't need fullwidth Unicode)
- Update chat template

### 4. FIM FOR CODE (Lower Priority for DI)
**What:** Fill-in-middle tokens for code completion
**Why:**
- Enables cursor-position code completion
- Useful if Mystral is used for coding assistance
**Implementation:**
- Add FIM tokens
- Train on code with FIM objective

---

## Implementation Path for Mystral

### Phase 1: Add Thinking (Minimal Change)
1. Add 2 tokens: `<think>`, `</think>`
2. Update tokenizer vocab
3. Fine-tune on reasoning traces (can use DeepSeek R1 distilled data)
4. Update Ollama modelfile to use thinking template

### Phase 2: Add Tool Calling
1. Add ~4 tokens: `<tool_call>`, `</tool_call>`, `<tool_output>`, `</tool_output>`
2. Train on tool-use dataset
3. Hook into Ollama's tool calling API

### Phase 3: Full Structured Token Set
1. Add role markers
2. Add any specialized tokens for guidance system
3. Potentially add concept-type tokens (from structured tokenizer work)

---

## Technical Notes

### Token Addition
- Extending vocab requires adding embeddings for new tokens
- Input embeddings: `model.embed_tokens.weight`
- Output embeddings: `lm_head.weight`
- Initialize new tokens as average of semantically similar existing tokens

### Chat Template (Jinja2)
The chat template controls how conversations are formatted.
DeepSeek R1's thinking is implemented purely in template:
```jinja2
{% if add_generation_prompt and not ns.is_tool %}
  {{'<｜Assistant｜><think>\n'}}
{% endif %}
```

### GGUF Conversion
When converting to GGUF for llama.cpp:
- `convert.py` handles tokenizer
- Special tokens must be in `added_tokens.json`
- Chat template goes in model metadata

---

## Summary Table

| Feature | DeepSeek R1 | Qwen 2.5 | Mistral 7B | Priority |
|---------|-------------|----------|------------|----------|
| Thinking | ✓ | ✗ | ✗ | HIGH |
| Tool Calling | ✓ | ✓ | ✗ | HIGH |
| Role Markers | ✓ | ✓ | ✗ | MEDIUM |
| FIM (Code) | ✗ | ✓ | ✗ | LOW |
| Vision | ✗ | ✓ | ✗ | N/A |
| Grounding | ✗ | ✓ | ✗ | N/A |

---

## Files Downloaded
- `deepseek/` - Base 7B tokenizer + config
- `deepseek-chat/` - Chat 7B configs
- `deepseek-r1/` - R1 reasoning configs (thinking feature)
- `qwen/` - Qwen 2.5 7B instruct
- `qwq/` - QwQ reasoning model
- `mistral/` - Current Mystral tokenizer
