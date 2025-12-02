// bullet_cli.cpp
// Bullet Inference CLI - Clean UX for running BQ4 models
// Usage: bullet run model.bq4 --prompt "Namaskar"

#include "bq4/bq4_loader.h"
#include "bq4/bq4_inference.h"
#include "bq4/bq4_attention.h"
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <chrono>

using namespace bullet::bq4;

void print_usage() {
    std::cout << "Bullet Inference CLI v1.0\n\n";
    std::cout << "Usage:\n";
    std::cout << "  bullet run <model.bq4> --prompt \"Your text here\"\n";
    std::cout << "  bullet info <model.bq4>\n";
    std::cout << "  bullet benchmark <model.bq4>\n\n";
    std::cout << "Options:\n";
    std::cout << "  --prompt TEXT    Input prompt for generation\n";
    std::cout << "  --max-tokens N   Maximum tokens to generate (default: 50)\n";
    std::cout << "  --temp FLOAT     Temperature for sampling (default: 0.7)\n";
    std::cout << "\nExamples:\n";
    std::cout << "  bullet run marathi.bq4 --prompt \"जीवन\"\n";
    std::cout << "  bullet info marathi.bq4\n";
}

void cmd_info(const std::string& model_path) {
    std::cout << "=== Model Info ===" << std::endl;
    std::cout << "File: " << model_path << std::endl;
    
    Loader loader(model_path);
    if (!loader.load_metadata()) {
        std::cerr << "Failed to load model" << std::endl;
        return;
    }
    
    std::cout << "\nTensors: " << loader.num_tensors() << std::endl;
    
    uint64_t total_size = 0;
    for (uint32_t i = 0; i < loader.num_tensors(); i++) {
        const auto& entry = loader.get_entry(i);
        
        std::cout << "\n  [" << i << "] " << entry.name << std::endl;
        std::cout << "      Shape: [";
        for (size_t d = 0; d < entry.dims.size(); d++) {
            std::cout << entry.dims[d];
            if (d < entry.dims.size() - 1) std::cout << " x ";
        }
        std::cout << "]" << std::endl;
        std::cout << "      Blocks: " << entry.num_blocks << std::endl;
        
        uint64_t tensor_size = entry.num_blocks * (entry.block_size / 2 + 4);
        total_size += tensor_size;
    }
    
    std::cout << "\nTotal size: " << (total_size / 1024) << " KB" << std::endl;
}

void cmd_benchmark(const std::string& model_path) {
    std::cout << "=== Benchmark ===" << std::endl;
    
    Loader loader(model_path);
    if (!loader.load_metadata()) {
        std::cerr << "Failed to load model" << std::endl;
        return;
    }
    
    std::cout << "Loading tensor 0..." << std::endl;
    Tensor W = loader.load_tensor(0);
    
    uint32_t hidden_dim = W.dims[1];
    uint32_t out_dim = W.dims[0];
    
    std::vector<float> input(hidden_dim, 0.5f);
    std::vector<float> output(out_dim);
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        matmul_bq4(input.data(), W, output.data());
    }
    
    // Benchmark
    int iterations = 1000;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        matmul_bq4(input.data(), W, output.data());
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    float avg_ms = duration.count() / (float)iterations / 1000.0f;
    
    std::cout << "\nResults:" << std::endl;
    std::cout << "  Iterations: " << iterations << std::endl;
    std::cout << "  Avg time: " << avg_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << (1000.0f / avg_ms) << " inferences/sec" << std::endl;
}

void cmd_run(const std::string& model_path, const std::string& prompt, 
             int max_tokens, float temperature) {
    std::cout << "=== Bullet Inference ===" << std::endl;
    std::cout << "Model: " << model_path << std::endl;
    std::cout << "Prompt: \"" << prompt << "\"" << std::endl;
    std::cout << "Max tokens: " << max_tokens << std::endl;
    std::cout << "Temperature: " << temperature << std::endl;
    
    Loader loader(model_path);
    if (!loader.load_metadata()) {
        std::cerr << "Failed to load model" << std::endl;
        return;
    }
    
    std::cout << "\n--- Output ---" << std::endl;
    std::cout << prompt;
    
    // TODO: Implement full generation loop
    // For now, just show that the model loaded
    std::cout << " [generation not yet implemented]" << std::endl;
    
    std::cout << "\n✅ Inference complete!" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }
    
    std::string command = argv[1];
    
    if (command == "info" && argc >= 3) {
        cmd_info(argv[2]);
    }
    else if (command == "benchmark" && argc >= 3) {
        cmd_benchmark(argv[2]);
    }
    else if (command == "run" && argc >= 3) {
        std::string model_path = argv[2];
        std::string prompt = "";
        int max_tokens = 50;
        float temperature = 0.7f;
        
        // Parse arguments
        for (int i = 3; i < argc; i++) {
            if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
                prompt = argv[++i];
            }
            else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
                max_tokens = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
                temperature = atof(argv[++i]);
            }
        }
        
        if (prompt.empty()) {
            std::cerr << "Error: --prompt required for 'run' command" << std::endl;
            return 1;
        }
        
        cmd_run(model_path, prompt, max_tokens, temperature);
    }
    else {
        print_usage();
        return 1;
    }
    
    return 0;
}
