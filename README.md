## A cpp environment for running an llm locally.

## Usage
Copy the qwen model into models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf.
Compile with `cmake -B build . && cmake --build build -j8`
Run `build/main <input_text_file> <output_text_file>`

## Details & Implementation
The inference engine used is llama.cpp, an llm inference in cpp.
We link against and compile with cmake.
The current main finds token predictions for a provided text, and returns probabilities.

## Future work
The current goal is to improve the runtime.
Some directions marked for exploration:
 - Use a bitset model.
 - Recompile the model with less tokens (Remove irrelevant tokens such as other languages).
 - Pruning and other model optimizations.
