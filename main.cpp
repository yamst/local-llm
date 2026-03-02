#include "llama.h"
#include "common.h"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <thread>    
#include <algorithm> 

int main() {
    int n_threads = std::max(1u, std::thread::hardware_concurrency() / 2);
    std::string model_path = "models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
    
    llama_backend_init();
    llama_model_params m_params = llama_model_default_params();
    m_params.n_gpu_layers = 5; 

    llama_model * model = llama_load_model_from_file(model_path.c_str(), m_params);
    if (!model) return 1;

    llama_context_params c_params = llama_context_default_params();
    c_params.n_ctx = 2048; 
    c_params.n_threads = n_threads;
    llama_context * ctx = llama_new_context_with_model(model, c_params);
    
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);

    std::string prompt;
    std::cout << "\n--- ENTER PROMPT TO ANALYZE ---\n";
    std::getline(std::cin, prompt);
    
    // Tokenize with BOS
    int n_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.length(), NULL, 0, true, true);
    std::vector<llama_token> tokens(n_tokens);
    llama_tokenize(vocab, prompt.c_str(), prompt.length(), tokens.data(), tokens.size(), true, true);

    std::cout << "\n--- PROMPT PROBABILITY ANALYSIS ---\n";
    std::cout << std::left << std::setw(20) << "Token" << " | " << "Probability" << "\n";
    std::cout << "------------------------------------------\n";

    // Loop through the prompt
    for (int i = 0; i < n_tokens - 1; i++) {
        // Decode token i to predict token i+1
        llama_batch batch = llama_batch_get_one(&tokens[i], 1);
        if (llama_decode(ctx, batch) != 0) break;

        float * logits = llama_get_logits_ith(ctx, 0);
        llama_token actual_next = tokens[i+1];

        // Softmax for probability
        float max_l = logits[0];
        for (int v = 1; v < n_vocab; v++) if (logits[v] > max_l) max_l = logits[v];
        
        double sum = 0.0;
        for (int v = 0; v < n_vocab; v++) sum += std::exp((double)logits[v] - max_l);
        double prob = std::exp((double)logits[actual_next] - max_l) / sum;

        // Display
        char buf[128];
        int n = llama_token_to_piece(vocab, actual_next, buf, sizeof(buf), 0, true);
        std::string piece = (n > 0) ? std::string(buf, n) : "???";
        
        // Escape newlines for console
        size_t p = 0;
        while ((p = piece.find("\n", p)) != std::string::npos) {
            piece.replace(p, 1, "\\n");
            p += 2;
        }

        std::cout << std::left << std::setw(20) << piece 
                  << " | " << std::fixed << std::setprecision(4) << (prob * 100.0) << "%" << std::endl;
    }

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}