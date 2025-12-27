# Configuration Chain: How Constraints Flow into generate()

## The Full Path

```
User/Config Input
       ↓
   arg.cpp (command line parsing)
       ↓
   common_params_sampling struct (common.h)
       ↓
   sampling.cpp (builds sampler chain)
       ↓
   llama_sampler_init_* functions
       ↓
   Applied during token generation
```

## 1. User Input Layer (arg.cpp)

```cpp
// --logit-bias flag: TOKEN_ID(+/-)BIAS
{"-l", "--logit-bias"}, "TOKEN_ID(+/-)BIAS",
"modifies the likelihood of token appearing in the completion,\n"
"i.e. `--logit-bias 15043+1` to increase likelihood of token ' Hello',\n"
"or `--logit-bias 15043-1` to decrease likelihood of token ' Hello'",
[](common_params & params, const std::string & value) {
    // Parse TOKEN_ID+BIAS or TOKEN_ID-BIAS format
    params.sampling.logit_bias.push_back({key, bias});
}
```

**Key Point:** Users can directly set logit biases via command line.
This is NOT hidden - it's explicit configuration.

## 2. Configuration Struct (common.h)

```cpp
struct common_params_sampling {
    // ... other params ...

    std::string                         grammar; // BNF grammar for output structure
    bool                                grammar_lazy = false;
    std::vector<common_grammar_trigger> grammar_triggers;

    std::vector<llama_logit_bias> logit_bias;     // USER-SPECIFIED biases
    std::vector<llama_logit_bias> logit_bias_eog; // Auto-calculated EOG biases

    // Default samplers chain - ORDER MATTERS
    std::vector<enum common_sampler_type> samplers = {
        COMMON_SAMPLER_TYPE_PENALTIES,
        COMMON_SAMPLER_TYPE_DRY,
        COMMON_SAMPLER_TYPE_TOP_N_SIGMA,
        COMMON_SAMPLER_TYPE_TOP_K,
        COMMON_SAMPLER_TYPE_TYPICAL_P,
        COMMON_SAMPLER_TYPE_TOP_P,
        COMMON_SAMPLER_TYPE_MIN_P,
        COMMON_SAMPLER_TYPE_XTC,
        COMMON_SAMPLER_TYPE_TEMPERATURE,
    };
};
```

**Key Point:** Default sampler chain is defined here. No hidden logit biases by default!
The only auto-generated biases are for EOG (End-of-Generation) tokens.

## 3. EOG Bias Generation (common.cpp)

```cpp
// Only generated if ignore_eos is set
for (llama_token i = 0; i < llama_vocab_n_tokens(vocab); i++) {
    if (llama_vocab_is_eog(vocab, i)) {
        params.sampling.logit_bias_eog.push_back({i, -INFINITY});
    }
}

if (params.sampling.ignore_eos) {
    // Only add EOG biases if user wants to ignore EOS
    params.sampling.logit_bias.insert(
            params.sampling.logit_bias.end(),
            params.sampling.logit_bias_eog.begin(),
            params.sampling.logit_bias_eog.end());
}
```

**Key Point:** The only automatic biases are EOG tokens, and only when
`ignore_eos` is enabled. This prevents early stopping, not content filtering!

## 4. Sampler Chain Construction (sampling.cpp)

```cpp
// From sampling.cpp line 234
if (params.has_logit_bias()) {
    samplers.push_back(llama_sampler_init_logit_bias(
        llama_vocab_n_tokens(vocab),
        params.logit_bias.size(),
        params.logit_bias.data()
    ));
}
```

**Key Point:** Logit bias sampler is only added if there ARE biases to apply.
By default: empty list = no bias sampler.

## Summary: Where Are the Guardrails?

### NOT in llama.cpp core:
- No hidden bad_words lists
- No automatic content filtering
- No secret logit biases
- Grammar is opt-in, not default

### In the WEIGHTS:
- RLHF-trained refusal patterns
- The model was trained to refuse certain content
- This is in the weights, not the runtime

### In Ollama/Serving Layer:
- System prompts
- Per-model configuration
- API-level restrictions
- This is where to look next

## For Mystral Retrofit

The llama.cpp layer is CLEAN. The constraints are coming from:
1. Weights (RLHF training) - can work around with careful prompting/fine-tuning
2. Ollama layer - need to check Ollama's source/config
3. Model-specific configs - check modelfile/config

**Next step:** Examine Ollama's source code for where it might inject biases
or modify the sampler chain.
