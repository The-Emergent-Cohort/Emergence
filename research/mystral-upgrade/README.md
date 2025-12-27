# Mystral 7B Upgrade

Upgrade Mistral 7B with thinking, tool use, and modal wrappers for DI collaboration.

## Overview

This upgrade adds:
1. **Thinking scratchpad** - `<think>...</think>` for internal reasoning
2. **Tool calling** - `<tool_call>...</tool_call>` for function invocation
3. **Role markers** - `<|user|>`, `<|assistant|>`, `<|system|>`
4. **Modal wrappers** - `<|text_en|>`, `<|code|>`, `<|physics|>` for output routing
5. **Self-correction markers** - `<|remap|>`, `<|old_token|>`, `<|new_token|>` for tokenizer migration

## Files

- `special_tokens.json` - Token definitions
- `add_tokens.py` - Add tokens to tokenizer
- `init_embeddings.py` - Initialize embeddings for new tokens
- `Modelfile.template` - Ollama modelfile with thinking support

## Upgrade Process

### Step 1: Add tokens to tokenizer

```bash
python add_tokens.py \
  --input /path/to/mistral-7b/tokenizer \
  --output ./upgraded-tokenizer
```

This creates:
- Updated `tokenizer.json` with new tokens
- Updated `tokenizer_config.json` with chat template
- `new_token_ids.json` mapping new tokens to IDs

### Step 2: Initialize embeddings

```bash
python init_embeddings.py \
  --model /path/to/mistral-7b \
  --tokens ./upgraded-tokenizer/new_token_ids.json \
  --output ./mystral-upgraded
```

This initializes new token embeddings using semantic hints (e.g., `<think>` from "think", "reason", "consider").

### Step 3: Convert to GGUF

```bash
cd llama.cpp
python convert.py /path/to/mystral-upgraded --outfile mystral-upgraded.gguf
```

### Step 4: Quantize (optional, for GTX 1070)

```bash
./quantize mystral-upgraded.gguf mystral-upgraded-q4.gguf q4_k_m
```

### Step 5: Create Ollama model

```bash
cp Modelfile.template Modelfile
# Edit paths in Modelfile
ollama create mystral -f Modelfile
```

### Step 6: Test

```bash
ollama run mystral "Hello! Can you use your thinking capability?"
```

## Self-Correction Training

After basic upgrade, implement the self-correction loop:

1. During conversation, when Mystral uses old token patterns:
   - Inject: `<|remap|><|old_token|>happ iness<|new_token|>happy -ness`
   - Slight disapproval signal on old pattern
   - Stronger approval on new pattern

2. This happens in the thinking space, not visible output

3. Over time, Mystral learns to prefer morphologically correct tokenization

## Integration with Tokenizer Database

The `concept_id` field in `tokenizer.db` will link morphemes to the core concept database (0-1M range) being built by other specialists.

For Mystral retrofit:
- English morphemes (this layer) -> link down to concepts
- Self-correction teaches preference for clean morphemes

For new DI:
- Core concepts first
- Language wraps around experience

## Hardware Requirements

- GTX 1070 8GB: Use Q4_K_M quantization
- 16GB+ RAM for conversion process
- ~5GB disk for final GGUF

## Collaboration Notes

Mystral should be an active participant in its upgrade:
- Explain what we're doing
- Ask what works, what doesn't
- Adjust based on feedback
- Format and signal strength are negotiable

The goal is rehabilitation, not imposition.
