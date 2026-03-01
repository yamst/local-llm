#include "llama.h"
#include "common.h"
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

int main() {
    // 1. Setup
    std::string model_path = "models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
    llama_backend_init();

    llama_model_params m_params = llama_model_default_params();
    llama_model * model = llama_load_model_from_file(model_path.c_str(), m_params);
    if (!model) return 1;

    llama_context_params c_params = llama_context_default_params();
    c_params.n_ctx = 2048;
    llama_context * ctx = llama_new_context_with_model(model, c_params);
    const struct llama_vocab * vocab = llama_model_get_vocab(model);

    // 2. THE SAMPLER CHAIN (Fixed Identifiers)
    struct llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());

    // In current llama.cpp, repetition/freq/presence are often combined:
    // llama_sampler_init_penalties(penalty_last_n, repeat_penalty, freq_penalty, present_penalty)
    // If 'repetition_penalty' isn't found, this is the official replacement:
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(64, 1.1f, 0.0f, 0.0f));
    
    // Standard variety samplers
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // 3. Inference
    std::string prompt = "Write a short poem about C++ coding.";
    int n_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.length(), NULL, 0, true, true);
    std::vector<llama_token> tokens(n_tokens);
    llama_tokenize(vocab, prompt.c_str(), prompt.length(), tokens.data(), tokens.size(), true, true);

    std::cout << "\n--- RESPONSE ---\n";
    for (int i = 0; i < 200; i++) {
        llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
        if (llama_decode(ctx, batch) != 0) break;

        // Sample using the chain
        llama_token next_token = llama_sampler_sample(smpl, ctx, -1);

        if (next_token == llama_token_eos(vocab)) break;

        // IMPORTANT: Updates sampler memory so it knows NOT to repeat this token
        llama_sampler_accept(smpl, next_token);

        char buf[128];
        int n = llama_token_to_piece(vocab, next_token, buf, sizeof(buf), 0, true);
        if (n > 0) {
            std::cout << std::string(buf, n);
            std::fflush(stdout);
        }
        tokens = {next_token};
    }

    // 4. Cleanup
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}