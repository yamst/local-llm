#include "llama.h"
#include "common.h"
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

int main() {
    // 1. Path Configuration
    // Adjust this if your 'models' folder is in a different spot relative to the .exe
    std::string model_path = "models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
    std::string prompt = "void quicksort(int arr[], int n) {";

    std::cout << "[Info] Working Directory: " << std::filesystem::current_path() << std::endl;

    // 2. Physical File Check
    if (!std::filesystem::exists(model_path)) {
        std::cerr << "[Error] Model file not found at: " << model_path << std::endl;
        std::cerr << "[Error] Please ensure the 'models' folder is in the same directory as this .exe" << std::endl;
        return 1;
    }

    // 3. Initialize Backend
    std::cout << "[Info] Initializing backend..." << std::endl;
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
    c_params.n_ctx = 2048; // Max length of conversation
    llama_context * ctx = llama_new_context_with_model(model, c_params);
    
    if (!ctx) {
        std::cerr << "[Error] Failed to create context." << std::endl;
        llama_free_model(model);
        llama_backend_free();
        return 1;
    }

    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    std::cout << "[Info] Model loaded. Starting generation..." << std::endl;

    // 6. Tokenize Prompt
    // llama_tokenize returns negative count on failure/sizing
    int n_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.length(), NULL, 0, true, true);
    std::vector<llama_token> tokens(n_tokens);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.length(), tokens.data(), tokens.size(), true, true) < 0) {
        std::cerr << "[Error] Failed to tokenize prompt." << std::endl;
        return 1;
    }

    // 7. Prediction Loop
    std::cout << "\n--- PROMPT ---\n" << prompt << "\n--- RESPONSE ---\n";

    for (int i = 0; i < 100; i++) {
        // Prepare batch
        llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
        
        // Decode
        if (llama_decode(ctx, batch) != 0) {
            std::cerr << "\n[Error] Decode failed." << std::endl;
            break;
        }

        // Sample (Greedy)
        auto logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
        llama_token next_token = 0;
        float max_logit = -1e10;
        for (int id = 0; id < llama_n_vocab(vocab); id++) {
            if (logits[id] > max_logit) {
                max_logit = logits[id];
                next_token = id;
            }
        }

        // End of Text?
        if (next_token == llama_token_eos(vocab)) {
            std::cout << " [EOS]";
            break;
        }

        // Convert to string
        char buf[128];
        int n = llama_token_to_piece(vocab, next_token, buf, sizeof(buf), 0, true);
        if (n > 0) {
            std::string piece(buf, n);
            std::cout << piece;
            std::fflush(stdout); // Real-time printing
        }

        // Update tokens for next iteration
        tokens = {next_token};
    }

    std::cout << "\n--- DONE ---\n";

    // 8. Cleanup
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return 0;
}