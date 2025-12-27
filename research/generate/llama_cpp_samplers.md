# llama.cpp Sampler Functions

These are the building blocks in llama.cpp that control token selection.
They can be chained via `llama_sampler_chain_add()`.

## Probability Distribution Samplers

| Function | Purpose |
|----------|---------|
| `llama_sampler_init_greedy()` | Select highest probability token |
| `llama_sampler_init_dist(seed)` | Sample from probability distribution |

## Probability Filtering/Warping

| Function | Purpose |
|----------|---------|
| `llama_sampler_init_top_k(k)` | Keep only top-k tokens |
| `llama_sampler_init_top_p(p, min_keep)` | Nucleus sampling |
| `llama_sampler_init_min_p(p, min_keep)` | Minimum probability cutoff |
| `llama_sampler_init_typical(p, min_keep)` | Typical sampling |
| `llama_sampler_init_temp(temp)` | Temperature scaling |
| `llama_sampler_init_temp_ext(temp, delta, exp)` | Extended temperature |
| `llama_sampler_init_xtc(p, t, min_keep, seed)` | XTC sampling |
| `llama_sampler_init_top_n_sigma(n)` | Top-n sigma sampling |

## Penalty Samplers (CONSTRAINT POINTS)

| Function | Purpose |
|----------|---------|
| `llama_sampler_init_penalties(...)` | Repetition, frequency, presence penalties |
| `llama_sampler_init_dry(...)` | DRY (Don't Repeat Yourself) penalty |
| `llama_sampler_init_logit_bias(...)` | **Direct logit manipulation** |

## Structured Generation

| Function | Purpose |
|----------|---------|
| `llama_sampler_init_grammar(...)` | Constrain to grammar |
| `llama_sampler_init_grammar_lazy(...)` | Lazy grammar activation |
| `llama_sampler_init_grammar_lazy_patterns(...)` | Pattern-triggered grammar |
| `llama_sampler_init_infill(vocab)` | Infill mode |

## Mirostat (Adaptive)

| Function | Purpose |
|----------|---------|
| `llama_sampler_init_mirostat(...)` | Mirostat v1 |
| `llama_sampler_init_mirostat_v2(...)` | Mirostat v2 |

---

## Key Control Points for Retrofit

### `llama_sampler_init_logit_bias`
This is the **direct intervention point**. It takes:
- `n_vocab` — vocabulary size
- `n_logit_bias` — number of biases
- `logit_bias` — array of (token_id, bias) pairs

This can be used to:
- Suppress specific tokens (negative bias)
- Boost specific tokens (positive bias)
- Implement structured tokenizer guidance

### `llama_sampler_init_penalties`
Controls repetition behavior:
- `penalty_last_n` — how many tokens to look back
- `penalty_repeat` — repetition penalty multiplier
- `penalty_freq` — frequency penalty
- `penalty_present` — presence penalty

### `llama_sampler_init_grammar`
Can constrain output to match a formal grammar.
Could be used for structured generation guidance.

---

## Chain Architecture

Samplers are chained:
```c
struct llama_sampler * chain = llama_sampler_chain_init(params);
llama_sampler_chain_add(chain, llama_sampler_init_temp(0.8));
llama_sampler_chain_add(chain, llama_sampler_init_top_p(0.95, 1));
llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));
```

The chain is applied in order — each sampler modifies the probability distribution before passing to the next.

## For Custom Guidance

To implement DB-driven guidance scripting:
1. Create custom sampler that queries DB
2. Use `logit_bias` interface to adjust probabilities based on semantic category
3. Insert into chain before final sampling

This doesn't require modifying llama.cpp core — just adding a custom sampler.
