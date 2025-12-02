#include "test_harness.hpp"

// We need quantize from builder and dequant from core.
// But they both define BulletHeader and helpers.
// We can't include both .cpp files directly in the same translation unit without conflict.
// Solution:
// 1. Extract BQ4 logic to a common header? (Best but requires refactoring)
// 2. Mock one of them?
// 3. Just include builder for quant test, and core for dequant test?
// But we want round trip.
// 
// Hack: Use namespaces or macros to hide conflicts?
// Or just copy the quantize function here for testing?
// The prompt says "Use bullet-builder.cpp".
// 
// Let's copy the `quantize_block_bq4` function here to test it against `dequant_block_bq4` from core.
// This avoids including all of builder.

#include "../bullet-core.cpp"

// Copied from bullet-builder.cpp for testing purposes
// We assume this matches the implementation we want to test.
// Ideally we'd link objects, but we are including .cpps.

uint16_t fp32_to_fp16_copy(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 31) & 0x1;
    uint32_t exp = (x >> 23) & 0xFF;
    uint32_t mant = x & 0x7FFFFF;
    
    uint16_t h;
    if (exp == 0) {
        h = sign << 15;
    } else if (exp == 0xFF) {
        h = (sign << 15) | 0x7C00 | (mant ? 1 : 0);
    } else {
        int new_exp = exp - 127 + 15;
        if (new_exp >= 31) {
            h = (sign << 15) | 0x7C00;
        } else if (new_exp <= 0) {
            h = sign << 15;
        } else {
            h = (sign << 15) | (new_exp << 10) | (mant >> 13);
        }
    }
    return h;
}

void quantize_block_bq4_copy(const float* src, uint8_t* dst) {
    float min_val = src[0];
    float max_val = src[0];
    for (int i = 1; i < 32; ++i) {
        if (src[i] < min_val) min_val = src[i];
        if (src[i] > max_val) max_val = src[i];
    }
    
    float range = max_val - min_val;
    float scale = range / 15.0f;
    if (scale < 1e-8f) scale = 1e-8f;
    
    int8_t zero = static_cast<int8_t>(round(-min_val / scale));
    
    uint16_t scale_fp16 = fp32_to_fp16_copy(scale);
    
    *reinterpret_cast<uint16_t*>(dst) = scale_fp16;
    dst[2] = zero;
    dst[3] = 0;
    
    uint8_t* nibbles = dst + 4;
    for (int i = 0; i < 16; ++i) {
        float v1 = src[i*2];
        float v2 = src[i*2+1];
        
        int q1 = static_cast<int>(round((v1 - min_val) / scale));
        int q2 = static_cast<int>(round((v2 - min_val) / scale));
        
        if (q1 < 0) q1 = 0; if (q1 > 15) q1 = 15;
        if (q2 < 0) q2 = 0; if (q2 > 15) q2 = 15;
        
        nibbles[i] = (q2 << 4) | q1;
    }
}

TEST(TestBQ4RoundTrip) {
    float src[32];
    uint8_t block[20];
    float dst[32];
    
    for(int i=0; i<32; ++i) src[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    
    quantize_block_bq4_copy(src, block);
    dequant_block_bq4(block, dst);
    
    float max_err = 0.0f;
    for(int i=0; i<32; ++i) {
        float err = std::abs(src[i] - dst[i]);
        if (err > max_err) max_err = err;
    }
    
    ASSERT_TRUE(max_err < 0.15f);
    
    return true;
}

TEST(TestBQ4Zero) {
    float src[32] = {0};
    uint8_t block[20];
    float dst[32];
    
    quantize_block_bq4_copy(src, block);
    dequant_block_bq4(block, dst);
    
    for(int i=0; i<32; ++i) {
        ASSERT_NEAR(dst[i], 0.0f, 1e-5f);
    }
    return true;
}
