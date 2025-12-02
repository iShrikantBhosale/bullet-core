// demo_bq4_inference.cpp
// BQ4 Inference Demo & Benchmark
// Demonstrates loading and running inference with BQ4 models

#include "bq4/bq4_loader.h"
#include "bq4/bq4_inference.h"
#include <iostream>
#include <chrono>
#include <random>

using namespace bullet::bq4;

void benchmark_matmul(const Tensor& W, int num_iterations = 100) {
    std::cout << "\n=== Benchmarking BQ4 MatMul ===" << std::endl;
    
    uint32_t hidden_dim = W.dims[1];
    uint32_t out_dim = W.dims[0];
    
    // Create random input
    std::vector<float> input(hidden_dim);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (auto& v : input) v = dist(gen);
    
    std::vector<float> output(out_dim);
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        matmul_bq4(input.data(), W, output.data());
    }
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; i++) {
        matmul_bq4(input.data(), W, output.data());
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    float avg_time_us = duration.count() / (float)num_iterations;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    std::cout << "  Input dim: " << hidden_dim << std::endl;
    std::cout << "  Output dim: " << out_dim << std::endl;
    std::cout << "  Iterations: " << num_iterations << std::endl;
    std::cout << "  Avg time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << (1000.0f / avg_time_ms) << " inferences/sec" << std::endl;
}

void test_accuracy(const Tensor& W) {
    std::cout << "\n=== Testing BQ4 Accuracy ===" << std::endl;
    
    uint32_t hidden_dim = W.dims[1];
    uint32_t out_dim = W.dims[0];
    
    // Create test input
    std::vector<float> input(hidden_dim, 0.5f);
    std::vector<float> output(out_dim);
    
    // Run inference
    matmul_bq4(input.data(), W, output.data());
    
    // Show first few outputs
    std::cout << "  First 10 outputs: ";
    for (int i = 0; i < std::min(10u, out_dim); i++) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;
    
    // Compute statistics
    float mean = 0.0f, min_val = output[0], max_val = output[0];
    for (auto v : output) {
        mean += v;
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }
    mean /= output.size();
    
    std::cout << "  Output stats:" << std::endl;
    std::cout << "    Mean: " << mean << std::endl;
    std::cout << "    Min: " << min_val << std::endl;
    std::cout << "    Max: " << max_val << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <model.bq4>" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    
    std::cout << "======================================" << std::endl;
    std::cout << "BQ4 Inference Demo & Benchmark" << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << "\nLoading model: " << model_path << std::endl;
    
    // Load model
    Loader loader(model_path);
    
    if (!loader.load_metadata()) {
        std::cerr << "Failed to load model metadata" << std::endl;
        return 1;
    }
    
    std::cout << "\n--- Loaded " << loader.num_tensors() << " tensors ---" << std::endl;
    
    // Load first tensor for testing
    std::cout << "\nLoading tensor 0..." << std::endl;
    Tensor W = loader.load_tensor(0);
    
    std::cout << "  Name: " << W.name << std::endl;
    std::cout << "  Shape: [";
    for (size_t i = 0; i < W.dims.size(); i++) {
        std::cout << W.dims[i];
        if (i < W.dims.size() - 1) std::cout << " x ";
    }
    std::cout << "]" << std::endl;
    std::cout << "  Blocks: " << W.blocks.size() << std::endl;
    std::cout << "  Block size: " << W.block_size << std::endl;
    
    // Test accuracy
    test_accuracy(W);
    
    // Benchmark
    benchmark_matmul(W, 1000);
    
    std::cout << "\n======================================" << std::endl;
    std::cout << "Demo complete!" << std::endl;
    std::cout << "======================================" << std::endl;
    
    return 0;
}
