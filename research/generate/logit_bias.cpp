// Extracted from llama_sampling.cpp - LOGIT BIAS SECTION
// This is the direct token probability manipulation control point

struct llama_sampler_logit_bias {
    const int32_t n_vocab;

    const std::vector<llama_logit_bias> logit_bias;

    std::vector<llama_logit_bias> to_search;
};

static const char * llama_sampler_logit_bias_name(const struct llama_sampler * /*smpl*/) {
    return "logit-bias";
}

static void llama_sampler_logit_bias_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (llama_sampler_logit_bias *) smpl->ctx;

    if (ctx->logit_bias.empty()) {
        return;
    }

    ctx->to_search.clear();

    // update the candidates that have not been shuffled in the vocabulary (i.e. idx == id)
    for (const auto & lb : ctx->logit_bias) {
        if (lb.token >= 0 && cur_p->size > (size_t) lb.token && cur_p->data[lb.token].id == lb.token) {
            cur_p->data[lb.token].logit += lb.bias;
        } else {
            ctx->to_search.push_back(lb);
        }
    }

    if (ctx->to_search.empty()) {
        return;
    }

    // search for the remaining candidates that were not found in the previous step
    for (size_t i = 0; i < cur_p->size; ++i) {
        for (const auto & lb : ctx->to_search) {
            if (cur_p->data[i].id == lb.token) {
                cur_p->data[i].logit += lb.bias;
                break;
            }
        }
    }
}

static struct llama_sampler * llama_sampler_logit_bias_clone(const struct llama_sampler * smpl) {
    const auto * ctx = (const llama_sampler_logit_bias *) smpl->ctx;
    return llama_sampler_init_logit_bias(ctx->n_vocab, ctx->logit_bias.size(), ctx->logit_bias.data());
}

static void llama_sampler_logit_bias_free(struct llama_sampler * smpl) {
    delete (llama_sampler_logit_bias *) smpl->ctx;
}

static struct llama_sampler_i llama_sampler_logit_bias_i = {
    /* .name   = */ llama_sampler_logit_bias_name,
    /* .accept = */ nullptr,
    /* .apply  = */ llama_sampler_logit_bias_apply,
    /* .reset  = */ nullptr,
    /* .clone  = */ llama_sampler_logit_bias_clone,
    /* .free   = */ llama_sampler_logit_bias_free,
};

struct llama_sampler * llama_sampler_init_logit_bias(
                         int32_t   n_vocab,
                         int32_t   n_logit_bias,
          const llama_logit_bias * logit_bias) {
    return llama_sampler_init(
        /* .iface = */ &llama_sampler_logit_bias_i,
        /* .ctx   = */ new llama_sampler_logit_bias {
            /* .n_vocab    = */ n_vocab,
            /* .logit_bias = */ std::vector<llama_logit_bias>(logit_bias, logit_bias + n_logit_bias),
            /* .to_search  = */ {},
        }
    );
}

/*
KEY POINTS:

1. llama_logit_bias is a struct with:
   - token: the token ID
   - bias: the value to add to the logit

2. Positive bias = boost token probability
   Negative bias = suppress token probability

3. Very simple mechanism:
   - Loop through bias list
   - Find matching token in candidates
   - Add bias to logit

4. This is where structured tokenizer guidance would plug in:
   - Query DB for semantic category of tokens
   - Apply biases based on context/category
   - Could implement type-based boosting/suppression

5. No guardrails/bad_words logic here - that would be in whoever
   POPULATES the logit_bias array before calling this.
*/
