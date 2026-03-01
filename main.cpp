#include "llama.h"
#include "common.h"
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm> // Required for std::count

int main() {
    std::string model_path = "models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
    std::string prompt;

    if (!std::filesystem::exists(model_path)) {
        std::cerr << "[Error] Model file not found." << std::endl;
        return 1;
    }

    llama_backend_init();

    llama_model_params m_params = llama_model_default_params();
    llama_model * model = llama_load_model_from_file(model_path.c_str(), m_params);
    if (!model) return 1;

    llama_context_params c_params = llama_context_default_params();
    c_params.n_ctx = 2048; 
    llama_context * ctx = llama_new_context_with_model(model, c_params);
    if (!ctx) return 1;

    const struct llama_vocab * vocab = llama_model_get_vocab(model);

    std::cout << "\n--- PROMPT ---\n";
    std::getline(std::cin, prompt);
    std::cout << "\n--- RESPONSE ---\n";

    int n_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.length(), NULL, 0, true, true);
    std::vector<llama_token> tokens(n_tokens);
    llama_tokenize(vocab, prompt.c_str(), prompt.length(), tokens.data(), tokens.size(), true, true);

    // --- REPETITION TRACKING (Pure C++) ---
    std::vector<llama_token> history;
    const int window_size = 20; // Look back at the last 20 tokens
    const int max_repeats = 3;  // Stop if a token appears more than 3 times in that window

    for (int i = 0; i < 200; i++) {
        llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
        
        if (llama_decode(ctx, batch) != 0) break;

        // Your original Greedy Sampling logic (This we know works!)
        auto logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
        llama_token next_token = 0;
        float max_logit = -1e10;
        for (int id = 0; id < llama_n_vocab(vocab); id++) {
            if (logits[id] > max_logit) {
                max_logit = logits[id];
                next_token = id;
            }
        }

        if (next_token == llama_token_eos(vocab)) break;

        // --- THE FIX: MANUAL REPETITION CHECK ---
        history.push_back(next_token);
        if (history.size() > window_size) {
            history.erase(history.begin());
        }

        // Count how many times this specific token has appeared recently
        if (std::count(history.begin(), history.end(), next_token) > max_repeats) {
            std::cout << "\n[Stopping: Repetition Detected]";
            break;
        }

        // Convert to string and print
        char buf[128];
        int n = llama_token_to_piece(vocab, next_token, buf, sizeof(buf), 0, true);
        if (n > 0) {
            std::cout << std::string(buf, n);
            std::fflush(stdout);
        }

        tokens = {next_token};
    }

    std::cout << "\n--- DONE ---\n";
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}