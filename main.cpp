#include "llama.h"
#include "common.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <chrono>

int main(int argc, char ** argv) {
    if (argc < 3) return 1;

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    std::string model_path = "models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    // 1. Fast File Read
    std::ifstream infile(input_path, std::ios::binary);
    std::vector<char> content((std::istreambuf_iterator<char>(infile)), std::istreambuf_iterator<char>());
    infile.close();

    llama_backend_init();
    llama_model_params m_params = llama_model_default_params();
    m_params.n_gpu_layers = 99; // Put it all on GPU

    llama_model * model = llama_load_model_from_file(model_path.c_str(), m_params);
    if (!model) return 1;

    llama_context_params c_params = llama_context_default_params();
    c_params.n_ctx = 4096; 
    c_params.n_batch = 2048; // Physical batch size
    c_params.flash_attn = true;

    llama_context * ctx = llama_new_context_with_model(model, c_params);
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);

    // 2. Tokenize
    int n_tokens = -llama_tokenize(vocab, content.data(), content.size(), NULL, 0, true, true);
    std::vector<llama_token> tokens(n_tokens);
    llama_tokenize(vocab, content.data(), content.size(), tokens.data(), tokens.size(), true, true);

    // FIX: Ensure BOS is present so the first word (tokens[1]) can be predicted from tokens[0]
    if (tokens.empty() || tokens[0] != llama_token_bos(vocab)) {
        tokens.insert(tokens.begin(), llama_token_bos(vocab));
        n_tokens = tokens.size();
    }

    std::ofstream outfile(output_path);
    
    // 3. The "Sweet Spot" Loop
    // We use a smaller eval size to prevent bus choking
    int n_eval_size = 32; 
    auto t_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n_tokens - 1; i += n_eval_size) {
        int n_batch = std::min(n_eval_size, n_tokens - 1 - i);

        llama_batch batch = llama_batch_init(n_batch, 0, 1);
        for (int j = 0; j < n_batch; j++) {
            // Only request logits for the tokens we are actually evaluating
            batch.token[j] = tokens[i+j];
            batch.pos[j] = i + j;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j] = true; // Still need these for prob calculation
        }
        batch.n_tokens = n_batch;

        if (llama_decode(ctx, batch) != 0) break;

        for (int j = 0; j < n_batch; j++) {
            float * logits = llama_get_logits_ith(ctx, j);
            llama_token actual_next = tokens[i + j + 1];

            // Optimized Softmax
            float max_l = -1e10f;
            for (int v = 0; v < n_vocab; v++) if (logits[v] > max_l) max_l = logits[v];
            
            double sum = 0.0;
            for (int v = 0; v < n_vocab; v++) sum += std::exp((double)logits[v] - max_l);
            double prob = std::exp((double)logits[actual_next] - max_l) / sum;

            char piece[128];
            int len = llama_token_to_piece(vocab, actual_next, piece, sizeof(piece), 0, true);
            outfile.write(piece, len);
            outfile << " | " << (prob * 100.0) << "%\n";
        }
        llama_batch_free(batch);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << "Analysis Time: " << duration << "s (" << (n_tokens / duration) << " t/s)\n";

    outfile.close();
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}