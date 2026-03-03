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
#include <chrono> // For measurement

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file>" << std::endl;
        return 1;
    }

    std::string input_path  = argv[1];
    std::string output_path = argv[2];
    std::string model_path  = "models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    // 1. Read input file
    std::ifstream infile(input_path);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open input file " << input_path << std::endl;
        return 1;
    }
    std::stringstream buffer;
    buffer << infile.rdbuf();
    std::string content = buffer.str();
    infile.close();

    // 2. Setup Llama
    llama_backend_init();
    llama_model_params m_params = llama_model_default_params();
    m_params.n_gpu_layers = 10; 

    llama_model * model = llama_load_model_from_file(model_path.c_str(), m_params);
    if (!model) return 1;

    llama_context_params c_params = llama_context_default_params();
    c_params.n_ctx = std::max(2048, (int)content.length());
    c_params.n_threads = std::max(1u, std::thread::hardware_concurrency() / 2);
    
    llama_context * ctx = llama_new_context_with_model(model, c_params);
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);

    // 3. Tokenize
    int n_tokens = -llama_tokenize(vocab, content.c_str(), content.length(), NULL, 0, true, true);
    std::vector<llama_token> tokens(n_tokens);
    llama_tokenize(vocab, content.c_str(), content.length(), tokens.data(), tokens.size(), true, true);

    // 4. Open Output File
    std::ofstream outfile(output_path);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open output file " << output_path << std::endl;
        return 1;
    }

    outfile << std::left << std::setw(25) << "Token" << " | " << "Probability (%)" << "\n";
    outfile << "--------------------------------------------------\n";

    std::cout << "Starting analysis of " << n_tokens << " tokens..." << std::endl;

    // --- START MEASUREMENT ---
    auto t_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n_tokens - 1; i++) {
        llama_batch batch = llama_batch_get_one(&tokens[i], 1);
        if (llama_decode(ctx, batch) != 0) break;

        float * logits = llama_get_logits_ith(ctx, 0);
        llama_token actual_next = tokens[i+1];

        // Softmax
        float max_l = logits[0];
        for (int v = 1; v < n_vocab; v++) if (logits[v] > max_l) max_l = logits[v];
        
        double sum = 0.0;
        for (int v = 0; v < n_vocab; v++) sum += std::exp((double)logits[v] - max_l);
        double prob = std::exp((double)logits[actual_next] - max_l) / sum;

        char buf[128];
        int n = llama_token_to_piece(vocab, actual_next, buf, sizeof(buf), 0, true);
        std::string piece = (n > 0) ? std::string(buf, n) : "???";
        
        size_t p = 0;
        while ((p = piece.find("\n", p)) != std::string::npos) { piece.replace(p, 1, "\\n"); p += 2; }

        outfile << std::left << std::setw(25) << piece 
                << " | " << std::fixed << std::setprecision(6) << (prob * 100.0) << "\n";
        
        if (i % 20 == 0) std::cout << "\rProgress: " << (i * 100 / n_tokens) << "%" << std::flush;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    // --- END MEASUREMENT ---

    double duration = std::chrono::duration<double>(t_end - t_start).count();
    double tps = (n_tokens - 1) / duration;

    std::cout << "\rAnalysis complete!                                \n";
    std::cout << "----------------------------------\n";
    std::cout << "Analysis Time: " << std::fixed << std::setprecision(3) << duration << " seconds\n";
    std::cout << "Analysis Speed: " << std::fixed << std::setprecision(2) << tps << " tokens/sec\n";
    std::cout << "Results saved to: " << output_path << "\n";
    std::cout << "----------------------------------\n";

    outfile.close();
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}