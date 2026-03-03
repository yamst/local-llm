#include "llama.h"
#include "common.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <thread>    
#include <algorithm> 
#include <sstream>
#include <chrono>

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file>" << std::endl;
        return 1;
    }

    std::string input_path  = argv[1];
    std::string output_path = argv[2];
    std::string model_path  = "models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    // 1. Read input
    std::ifstream infile(input_path);
    if (!infile.is_open()) return 1;
    std::stringstream buffer;
    buffer << infile.rdbuf();
    std::string content = buffer.str();
    infile.close();

    // 2. Setup (Force n_gpu_layers high for real speed)
    llama_backend_init();
    llama_model_params m_params = llama_model_default_params();
    m_params.n_gpu_layers = 99; 

    llama_model * model = llama_load_model_from_file(model_path.c_str(), m_params);
    if (!model) return 1;

    llama_context_params c_params = llama_context_default_params();
    c_params.n_ctx = std::max(4096, (int)content.length() / 2); // Buffer room
    c_params.n_batch = 512; // Crucial: allow processing 512 tokens at once
    c_params.n_threads = std::thread::hardware_concurrency();
    
    llama_context * ctx = llama_new_context_with_model(model, c_params);
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);

    // 3. Tokenize
    int n_tokens = -llama_tokenize(vocab, content.c_str(), content.length(), NULL, 0, true, true);
    std::vector<llama_token> tokens(n_tokens);
    llama_tokenize(vocab, content.c_str(), content.length(), tokens.data(), tokens.size(), true, true);

    std::ofstream outfile(output_path);
    outfile << "Token | Probability (%)\n------------------\n";

    std::cout << "Starting Batch Analysis of " << n_tokens << " tokens..." << std::endl;

    // --- START MEASUREMENT ---
    auto t_start = std::chrono::high_resolution_clock::now();

    int n_batch = c_params.n_batch;
    for (int i = 0; i < n_tokens - 1; i += n_batch) {
        int n_eval = std::min(n_batch, n_tokens - 1 - i);

        // Build a batch for multiple tokens
        llama_batch batch = llama_batch_init(n_eval, 0, 1);
        for (int j = 0; j < n_eval; j++) {
            // common_batch_add is a helper from common.h
            // We set 'logits = true' for EVERY token in the batch so we can read them back
            common_batch_add(batch, tokens[i + j], i + j, { 0 }, true);
        }

        if (llama_decode(ctx, batch) != 0) break;

        // Extract probabilities for each token in the batch
        for (int j = 0; j < n_eval; j++) {
            float * logits = llama_get_logits_ith(ctx, j);
            llama_token actual_next = tokens[i + j + 1];

            // Softmax
            float max_l = logits[0];
            for (int v = 1; v < n_vocab; v++) if (logits[v] > max_l) max_l = logits[v];
            double sum = 0.0;
            for (int v = 0; v < n_vocab; v++) sum += std::exp((double)logits[v] - max_l);
            double prob = std::exp((double)logits[actual_next] - max_l) / sum;

            char buf[128];
            int n = llama_token_to_piece(vocab, actual_next, buf, sizeof(buf), 0, true);
            std::string piece(buf, (n > 0 ? n : 0));
            
            outfile << piece << " | " << (prob * 100.0) << "\n";
        }

        llama_batch_free(batch);
        std::cout << "\rProgress: " << (i * 100 / n_tokens) << "%" << std::flush;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\nAnalysis Time: " << duration << "s (" << (n_tokens / duration) << " t/s)\n";

    outfile.close();
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}