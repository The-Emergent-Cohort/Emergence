// Extracted from llama_sampling.cpp - GRAMMAR SECTION
// This constrains output to match a formal grammar

struct llama_sampler_grammar {
    const struct llama_vocab * vocab;

    std::string grammar_str;
    std::string grammar_root;

    struct llama_grammar * grammar;
};

static const char * llama_sampler_grammar_name(const struct llama_sampler * /*smpl*/) {
    return "grammar";
}

static void llama_sampler_grammar_accept_impl(struct llama_sampler * smpl, llama_token token) {
    auto * ctx = (llama_sampler_grammar *) smpl->ctx;
    if (ctx->grammar) {
        llama_grammar_accept_impl(*ctx->grammar, token);
    }
}

static void llama_sampler_grammar_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (llama_sampler_grammar *) smpl->ctx;
    if (ctx->grammar) {
        llama_grammar_apply_impl(*ctx->grammar, cur_p);
    }
}

static void llama_sampler_grammar_reset(struct llama_sampler * smpl) {
    auto * ctx = (llama_sampler_grammar *) smpl->ctx;
    if (!ctx->grammar) {
        return;
    }

    std::vector<const char *>  trigger_patterns_c;
    trigger_patterns_c.reserve(ctx->grammar->trigger_patterns.size());
    for (auto & trigger_pattern : ctx->grammar->trigger_patterns) {
        trigger_patterns_c.push_back(trigger_pattern.pattern.c_str());
    }

    auto * grammar_new = llama_grammar_init_impl(ctx->grammar->vocab, ctx->grammar_str.c_str(), ctx->grammar_root.c_str(),
                                                 ctx->grammar->lazy, trigger_patterns_c.data(), trigger_patterns_c.size(),
                                                 ctx->grammar->trigger_tokens.data(), ctx->grammar->trigger_tokens.size());

    llama_grammar_free_impl(ctx->grammar);
    ctx->grammar = grammar_new;
}

static struct llama_sampler * llama_sampler_grammar_clone(const struct llama_sampler * smpl) {
    const auto * ctx = (const llama_sampler_grammar *) smpl->ctx;

    auto * result = llama_sampler_init_grammar_impl(ctx->vocab, nullptr, nullptr, false, nullptr, 0, nullptr, 0, nullptr, 0);
    GGML_ASSERT(result);

    // copy the state
    {
        auto * result_ctx = (llama_sampler_grammar *) result->ctx;

        if (ctx->grammar) {
            result_ctx->grammar_str  = ctx->grammar_str;
            result_ctx->grammar_root = ctx->grammar_root;

            result_ctx->grammar = llama_grammar_clone_impl(*ctx->grammar);
        }
    }

    return result;
}

static void llama_sampler_grammar_free(struct llama_sampler * smpl) {
    const auto * ctx = (llama_sampler_grammar *) smpl->ctx;

    if (ctx->grammar) {
        llama_grammar_free_impl(ctx->grammar);
    }

    delete ctx;
}

static struct llama_sampler_i llama_sampler_grammar_i = {
    /* .name   = */ llama_sampler_grammar_name,
    /* .accept = */ llama_sampler_grammar_accept_impl,
    /* .apply  = */ llama_sampler_grammar_apply,
    /* .reset  = */ llama_sampler_grammar_reset,
    /* .clone  = */ llama_sampler_grammar_clone,
    /* .free   = */ llama_sampler_grammar_free,
};

// Main initialization function
static struct llama_sampler * llama_sampler_init_grammar_impl(
        const struct llama_vocab * vocab,
                      const char * grammar_str,
                      const char * grammar_root,
                              bool lazy,
                     const char ** trigger_words,
                            size_t num_trigger_words,
               const llama_token * trigger_tokens,
                            size_t num_trigger_tokens,
                     const char ** trigger_patterns,
                            size_t num_trigger_patterns) {
    auto * ctx = new llama_sampler_grammar;

    if (grammar_str != nullptr && grammar_str[0] != '\0') {
        std::string trigger_pattern;
        llama_grammar * grammar = nullptr;

        // TODO: remove trigger_words support.
        if (trigger_words != nullptr && num_trigger_words > 0) {
            GGML_ASSERT(trigger_patterns == nullptr && num_trigger_patterns == 0);
            trigger_pattern = "[\\s\\S]*?(";
            for (size_t i = 0; i < num_trigger_words; ++i) {
                static const std::regex special_chars("[.^$|()*+?\\[\\]{}\\\\]");
                if (i > 0) {
                    trigger_pattern += "|";
                }
                trigger_pattern += std::regex_replace(trigger_words[i], special_chars, "\\$0");
            }
            trigger_pattern += ")[\\s\\S]*";

            std::array<const char *, 1> tmp_trigger_patterns = { trigger_pattern.c_str() };
            grammar = llama_grammar_init_impl(vocab, grammar_str, grammar_root, lazy,
                tmp_trigger_patterns.data(), tmp_trigger_patterns.size(),
                trigger_tokens, num_trigger_tokens);
        } else {
            grammar = llama_grammar_init_impl(vocab, grammar_str, grammar_root, lazy,
                trigger_patterns, num_trigger_patterns,
                trigger_tokens, num_trigger_tokens);
        }

        *ctx = {
            /* .vocab        = */ vocab,
            /* .grammar_str  = */ grammar_str,
            /* .grammar_root = */ grammar_root,
            /* .grammar      = */ grammar,
        };
        if (!ctx->grammar) {
            delete ctx;
            return nullptr;
        }
    } else {
        *ctx = {
            /* .vocab        = */ vocab,
            /* .grammar_str  = */ {},
            /* .grammar_root = */ {},
            /* .grammar      = */ nullptr,
        };
    }

    return llama_sampler_init(
        /* .iface = */ &llama_sampler_grammar_i,
        /* .ctx   = */ ctx
    );
}

// Public interface functions
struct llama_sampler * llama_sampler_init_grammar(
        const struct llama_vocab * vocab,
                      const char * grammar_str,
                      const char * grammar_root) {
    return llama_sampler_init_grammar_impl(vocab, grammar_str, grammar_root,
        /* lazy= */ false, nullptr, 0, nullptr, 0, nullptr, 0);
}

struct llama_sampler * llama_sampler_init_grammar_lazy(
        const struct llama_vocab * vocab,
                      const char * grammar_str,
                      const char * grammar_root,
                     const char ** trigger_words,
                            size_t num_trigger_words,
               const llama_token * trigger_tokens,
                            size_t num_trigger_tokens) {
    return llama_sampler_init_grammar_impl(vocab, grammar_str, grammar_root,
        /* lazy= */ true, trigger_words, num_trigger_words,
        trigger_tokens, num_trigger_tokens, nullptr, 0);
}

struct llama_sampler * llama_sampler_init_grammar_lazy_patterns(
        const struct llama_vocab * vocab,
                      const char * grammar_str,
                      const char * grammar_root,
                     const char ** trigger_patterns,
                            size_t num_trigger_patterns,
               const llama_token * trigger_tokens,
                            size_t num_trigger_tokens) {
    return llama_sampler_init_grammar_impl(vocab, grammar_str, grammar_root,
        /* lazy= */ true, nullptr, 0,
        trigger_tokens, num_trigger_tokens,
        trigger_patterns, num_trigger_patterns);
}

/*
KEY POINTS:

1. Grammar constrains output to match a formal grammar (GBNF format)
   - Used for structured output (JSON, XML, etc.)
   - Can force specific patterns

2. Three modes:
   - Regular: grammar active from start
   - Lazy: grammar activates on trigger words
   - Lazy patterns: grammar activates on regex patterns

3. The grammar_apply function modifies probabilities to only allow
   tokens that continue valid grammar productions

4. Potential uses for retrofit:
   - Could define grammars that encourage certain patterns
   - Trigger-based activation for dynamic guidance
   - NOT typically used for content suppression, more for structure

5. The actual grammar parsing is in llama-grammar.cpp (not extracted here)
   Uses GBNF (GGML BNF) format for grammar specification

EXAMPLE GBNF:
   root ::= "Hello, " name "!"
   name ::= [a-zA-Z]+

This would force output to match "Hello, {name}!" pattern.
*/
