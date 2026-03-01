#include "llama.h"
#include "common.h"
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

int main() {
    // 1. Path Configuration
    std::string model_path = "models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
    std::string prompt;

    std::cout << "[Info] Working Directory: " << std::filesystem::current_path() << std::endl;

    // 2. Physical File Check
    if (!std::filesystem::exists(model_path)) {
        std::cerr << "[Error] Model file not found at: " << model_path << std::endl;
        return 1;
    }

    // 3. Initialize Backend
    llama_backend_init();

    // 4. Load Model
    llama_model_params m_params = llama_model_default_params();
    llama_model * model = llama_load_model_from_file(model_path.c_str(), m_params);
    
    if (!model) {
        std::cerr << "[Error] Failed to load model from " << model_path << std::endl;
        llama_backend_free();
        return 1;
    }

    // 5. Create Context & Vocab
    llama_context_params c_params = llama_context_default_params();
    c_params.n_ctx = 2048; 
    llama_context * ctx = llama_new_context_with_model(model, c_params);
    
    if (!ctx) {
        std::cerr << "[Error] Failed to create context." << std::endl;
        llama_free_model(model);
        llama_backend_free();
        return 1;
    }

    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    std::cout << "[Info] Model loaded. Starting generation..." << std::endl;

    // --- REPETITION SAMPLER SETUP ---
    struct llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    // This is the specific identifier usually found in the common/sampling headers
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(64, 1.1f, 0.0f, 0.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(1234));

    // 7. Prediction Loop
    std::cout << "\n--- PROMPT ---\n";
    std::getline(std::cin, prompt);
    std::cout << "\n--- RESPONSE ---\n";

    // 6. Tokenize Prompt
    int n_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.length(), NULL, 0, true, true);
    std::vector<llama_token> tokens(n_tokens);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.length(), tokens.data(), tokens.size(), true, true) < 0) {
        std::cerr << "[Error] Failed to tokenize prompt." << std::endl;
        return 1;
    }

    for (int i = 0; i < 200; i++) {
        llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
        
        if (llama_decode(ctx, batch) != 0) {
            std::cerr << "\n[Error] Decode failed." << std::endl;
            break;
        }

        // --- UPDATED SAMPLING ---
        llama_token next_token = llama_sampler_sample(smpl, ctx, -1);
        llama_sampler_accept(smpl, next_token); 

        if (next_token == llama_token_eos(vocab)) {
            std::cout << " [EOS]";
            break;
        }

        char buf[128];
        int n = llama_token_to_piece(vocab, next_token, buf, sizeof(buf), 0, true);
        if (n > 0) {
            std::string piece(buf, n);
            std::cout << piece;
            std::fflush(stdout);
        }

        tokens = {next_token};
    }

    std::cout << "\n--- DONE ---\n";

    // 8. Cleanup
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return 0;
}