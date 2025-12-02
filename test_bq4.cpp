// test_bq4.cpp
// Unit tests for BQ4 quantization kernels

#include "bq4/bq4_kernels.h"
#include <iostream>
#include <random>

using namespace bullet::bq4;

void test_roundtrip() {
    std::cout << "Testing BQ4 round-trip (quantize → dequantize)...\n";
    
    // Test with known values
    float orig[32];
    for (int i = 0; i < 32; i++) {
        orig[i] = (float)(i - 16) / 10.0f;  // Range: -1.6 to 1.5
    }
    
    float err = test_roundtrip(orig, 32);
    
    std::cout << "  Total error: " << err << "\n";
    
    if (err < 0.05) {
        std::cout << "  ✅ PASS: Error < 0.05\n";
    } else {
        std::cout << "  ❌ FAIL: Error too high!\n";
    }
}

void test_random() {
    std::cout << "\nTesting BQ4 with random weights...\n";
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    
    float orig[64];
    for (int i = 0; i < 64; i++) {
        orig[i] = dist(gen);
    }
    
    float err = test_roundtrip(orig, 64);
    
    std::cout << "  Total error: " << err << "\n";
    
    if (err < 0.1) {
        std::cout << "  ✅ PASS: Error < 0.1\n";
    } else {
        std::cout << "  ❌ FAIL: Error too high!\n";
    }
}

void test_edge_cases() {
    std::cout << "\nTesting BQ4 edge cases...\n";
    
    // All zeros
    float zeros[32] = {0};
    float err1 = test_roundtrip(zeros, 32);
    std::cout << "  Zeros error: " << err1 << (err1 < 0.001 ? " ✅\n" : " ❌\n");
    
    // Very small values
    float small[32];
    for (int i = 0; i < 32; i++) small[i] = 0.0001f;
    float err2 = test_roundtrip(small, 32);
    std::cout << "  Small values error: " << err2 << (err2 < 0.01 ? " ✅\n" : " ❌\n");
    
    // Large values
    float large[32];
    for (int i = 0; i < 32; i++) large[i] = 100.0f;
    float err3 = test_roundtrip(large, 32);
    std::cout << "  Large values error: " << err3 << (err3 < 5.0 ? " ✅\n" : " ❌\n");
}

int main() {
    std::cout << "======================================\n";
    std::cout << "BQ4 Kernel Unit Tests\n";
    std::cout << "======================================\n\n";
    
    test_roundtrip();
    test_random();
    test_edge_cases();
    
    std::cout << "\n======================================\n";
    std::cout << "All tests complete!\n";
    std::cout << "======================================\n";
    
    return 0;
}
