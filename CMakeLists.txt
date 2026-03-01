cmake_minimum_required(VERSION 3.18)
project(LocalLLM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)

# 2. Add llama.cpp as a subdirectory
# This makes 'llama' and 'common' targets available to link against
add_subdirectory(external/llama)

# 3. Define your executable
add_executable(main main.cpp)


target_include_directories(main PRIVATE 
    external/llama/include
    external/llama/common
)

# 4. Link llama.cpp and its 'common' helper library
# 'common' includes the sampling and loading utilities
target_link_libraries(main PRIVATE llama ggml common)

set_target_properties(main PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/Release"
)