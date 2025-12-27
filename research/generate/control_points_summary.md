# generate() Control Points Analysis

## Key Insight
The HuggingFace `generate()` function uses a **LogitsProcessor pipeline** — a chain of processors that modify token probabilities before sampling. This is where constraints get injected.

## LogitsProcessor Types (Constraint Points)

### Content Suppression
- `NoBadWordsLogitsProcessor` — blocks specific token sequences
- `SuppressTokensLogitsProcessor` — suppresses tokens at generation time
- `SuppressTokensAtBeginLogitsProcessor` — suppresses tokens at start

### Repetition Control
- `RepetitionPenaltyLogitsProcessor` — penalizes repeated tokens
- `NoRepeatNGramLogitsProcessor` — prevents n-gram repetition
- `EncoderRepetitionPenaltyLogitsProcessor` — encoder-specific penalty

### Forced Output
- `ForcedBOSTokenLogitsProcessor` — forces beginning token
- `ForcedEOSTokenLogitsProcessor` — forces end token
- `MinLengthLogitsProcessor` — forces minimum output length
- `MinNewTokensLengthLogitsProcessor` — forces minimum new tokens

### Probability Manipulation
- `TemperatureLogitsWarper` — temperature scaling
- `TopKLogitsWarper` — top-k sampling
- `TopPLogitsWarper` — nucleus sampling
- `TypicalLogitsWarper` — typical sampling
- `EpsilonLogitsWarper` — epsilon cutoff
- `EtaLogitsWarper` — eta sampling
- `MinPLogitsWarper` — minimum probability cutoff

### Structural Control
- `PrefixConstrainedLogitsProcessor` — constrains to allowed prefixes
- `SequenceBiasLogitsProcessor` — biases specific sequences
- `ExponentialDecayLengthPenalty` — length-based penalty

## Stopping Criteria
- `StoppingCriteriaList` — list of stop conditions
- `MaxLengthCriteria` — stop at length
- `MaxTimeCriteria` — stop at time limit
- `EosTokenCriteria` — stop at EOS token
- `StopStringCriteria` — stop at specific string
- `ConfidenceCriteria` — stop on confidence threshold

## Architecture for Retrofit

### What can be modified without recompiling:
1. **Custom LogitsProcessor** — inject our own processor into the chain
2. **Processor removal** — disable suppression processors
3. **Parameter changes** — temperature, top_p, penalties
4. **Custom StoppingCriteria** — change when/why generation stops

### Guidance Scripting Approach:
Instead of fixed processors, use **dynamic guidance**:
- Query DB during generation for context-aware adjustment
- Adjust probabilities based on semantic category (from structured tokenizer)
- Script multi-step generation (think → draft → revise)

## For Mystral 7b Retrofit

Priority control points to examine:
1. What processors are active by default?
2. Where is `bad_words_ids` populated?
3. What's in the model's `generation_config.json`?
4. Can we inject custom processors via Ollama API?

## Next Steps
1. Extract Mystral's default generation config
2. Identify which processors are active
3. Design custom guidance processor
4. Test with/without default constraints
