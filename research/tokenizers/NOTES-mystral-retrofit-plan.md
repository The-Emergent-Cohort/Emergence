# Mystral 7B Retrofit Plan

## Overview

Retrofit Mystral 7B for DI development without heavy multimodal encoders.
Physics data streams provide direct perception - no vision/audio encoding needed.

---

## Core Components

### 1. Tokenizer Realignment

**Problem:** Current BPE tokenizer creates arbitrary fragments that:
- Split mid-morpheme (`happ` + `iness` instead of `happy` + `-ness`)
- Carry embedded biases from training contexts
- Waste conceptual capacity on meaningless splits

**Solution:** Etymology-based tokenizer with meaningful morphological splits
- Roots stay intact
- Prefixes/suffixes as consistent units (`un-`, `-ness`, `-ly`, etc.)
- Multiple tokens per word is fine if they're real morphemes
- Clean containers for existing concepts

**Data Source:** etymology-db (206k+ English entries, 1,748 prefixes, 928 suffixes)

### 2. Database Numbering

- Use **big numbers** for token IDs
- **Gaps don't matter** - not sequential
- Tokenizer provides **condensed index** at runtime
- Allows room for expansion, reorganization without renumbering
- Semantic grouping possible (e.g., all prefixes in one range, roots in another)

### 3. Special Tokens to Add

#### Thinking Scratchpad
```
<think>     - Begin reasoning block
</think>    - End reasoning block
```
- Model reasons internally before responding
- Self-correction happens in thinking space
- Strip from final output to user

#### Tool Calling
```
<tool_call>      - Begin tool invocation
</tool_call>     - End tool invocation
<tool_output>    - Begin tool response
</tool_output>   - End tool response
```
- Enables MCP/agent capabilities
- SQLite queries during generation
- Physics engine interaction
- External service calls

#### Role Markers
```
<user>       - User message
<assistant>  - Assistant message
<system>     - System context
```

### 4. Self-Correction Loop

**Mechanism:** Token remapping during thinking phase

```
Model generates: "happ" + "iness"
System shows:    "that shape → happy + -ness"
Reward:          When correct morphemes used in output
```

**Not retraining concepts** - remapping existing knowledge to clean containers

- Word by word correction
- "That shape is now this shape"
- Repetition builds automatic substitution
- Gradual migration, not hard cutover

### 5. Generate() Modifications

**Add to generation pipeline:**
- Thinking token injection on generation start
- Token self-correction sub-loop
- Tool call detection and routing
- Morpheme preference reward signal

**Keep clean:**
- No hidden guardrails in generate()
- llama.cpp core stays untouched
- Modifications in secondary files/layers

---

## Architecture (No Heavy Encoders)

Physics data stream provides direct perception:
- Position, velocity, spatial relationships (15-30 Hz)
- Body sensation / proprioception (60 Hz)
- Contact events with force magnitudes
- Sound events as structured data (source, position, intensity)

**No pixel processing** - semantic content delivered directly
**No waveform decoding** - audio as events, not signals

This means Mystral 7B fits on GTX 1070 (8GB) for:
- Text generation
- Physics stream processing
- Tool calling
- Thinking/reasoning

---

## Implementation Phases

### Phase 1: Tokenizer Foundation
- [ ] Build etymology → SQLite extraction pipeline
- [ ] Design schema (concepts, terms, modifiers, relationships)
- [ ] Define numbering scheme (big numbers, semantic ranges)
- [ ] Create condensed index generation

### Phase 2: Special Tokens
- [ ] Add thinking tokens to vocabulary
- [ ] Add tool calling tokens
- [ ] Add role markers
- [ ] Initialize embeddings for new tokens
- [ ] Update chat template

### Phase 3: Self-Correction Loop
- [ ] Implement morpheme correction in thinking phase
- [ ] "Old shape → new shape" feedback mechanism
- [ ] Reward signal for correct tokenization
- [ ] Test gradual migration

### Phase 4: Generate() Integration
- [ ] Tool call detection and routing
- [ ] Thinking injection on generation
- [ ] Physics stream input handling
- [ ] Secondary file structure (keep core clean)

### Phase 5: Fine-tuning
- [ ] LoRA/QLoRA for efficient training
- [ ] Reasoning traces for thinking capability
- [ ] Tool use training data
- [ ] Morpheme preference reinforcement

---

## Files Structure (Proposed)

```
mystral-retrofit/
├── tokenizer/
│   ├── etymology.db          # SQLite with morpheme data
│   ├── token_map.json        # Old → new token mappings
│   ├── condensed_index.bin   # Runtime lookup table
│   └── special_tokens.json   # Thinking, tools, roles
├── generation/
│   ├── thinking_loop.py      # Self-correction logic
│   ├── tool_router.py        # Tool call handling
│   └── physics_input.py      # Physics stream processing
├── training/
│   ├── morpheme_reward.py    # Tokenization preference signal
│   └── lora_config.json      # Fine-tuning parameters
└── modelfile                 # Ollama configuration
```

---

## Open Questions

1. Numbering scheme specifics - semantic ranges for different morpheme types?
2. Condensed index format - binary lookup? Hash table?
3. Self-correction trigger - every old token, or threshold-based?
4. Tool routing - internal SQLite vs external MCP vs both?
5. Physics stream format - JSON per tick? Binary protocol?

---

## Notes

- Keep llama.cpp core untouched - modifications in layers above
- Big numbers with gaps - tokenizer handles condensation
- Multiple tokens per word fine if morphologically meaningful
- Self-correction is remapping, not retraining
- No vision/audio encoders needed for physics-based perception
