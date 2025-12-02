// bq4_kernels.h
// Core BQ4 Quantization/Dequantization Kernels
// Bullet OS - 4-bit Quantization System

#ifndef BQ4_KERNELS_H
#define BQ4_KERNELS_H

#include <cstdint>
#include <cmath>
#include <cstring>

namespace bullet {
namespace bq4 {

// ============================================================================
// BQ4 Dequantization Kernel
// ============================================================================
// Converts packed int4 weights → FP32
// Each byte contains two int4 values (high nibble, low nibble)
// Signed int4 range: [-8, +7]

inline void dequantize(
    const uint8_t* w_in,     // packed int4 data
    float scale,             // scale factor
    float* w_out,            // output floats
    int num_weights          // usually 32 or 64
) {
    for (int i = 0; i < num_weights / 2; i++) {
        // Read packed byte
        uint8_t packed = w_in[i];

        // Extract two signed int4 values
        int8_t lo = (packed & 0x0F);
        int8_t hi = (packed >> 4);

        // Convert from unsigned nibble to signed [-8, +7]
        if (lo > 7) lo -= 16;
        if (hi > 7) hi -= 16;

        // Dequantize to fp32
        w_out[2*i + 0] = lo * scale;
        w_out[2*i + 1] = hi * scale;
    }
}

// ============================================================================
// BQ4 Quantization Kernel
// ============================================================================
// Converts FP32 weights → packed int4 + scale
// Uses symmetric quantization with per-block scaling

inline void quantize(
    const float* w_in,
    uint8_t* w_out,
    float& scale_out,
    int num_weights   // usually 32 or 64
) {
    // 1. Compute scale
    float max_abs = 0.0f;
    for (int i = 0; i < num_weights; i++) {
        float x = fabsf(w_in[i]);
        if (x > max_abs) max_abs = x;
    }

    // Avoid division by zero
    if (max_abs < 1e-8f) max_abs = 1e-8f;

    scale_out = max_abs / 7.0f;
    float inv_scale = 1.0f / scale_out;

    // 2. Quantize + pack two int4 per byte
    for (int i = 0; i < num_weights / 2; i++) {
        // Quantize two floats
        int8_t q0 = (int8_t) roundf(w_in[2*i + 0] * inv_scale);
        int8_t q1 = (int8_t) roundf(w_in[2*i + 1] * inv_scale);

        // Clamp to int4 valid range [-8, +7]
        if (q0 < -8) q0 = -8;
        if (q0 >  7) q0 =  7;
        if (q1 < -8) q1 = -8;
        if (q1 >  7) q1 =  7;

        // Convert signed int4 → unsigned nibble
        uint8_t lo = (uint8_t)(q0 & 0x0F);
        uint8_t hi = (uint8_t)(q1 & 0x0F);

        // Pack two int4 values
        w_out[i] = (hi << 4) | lo;
    }
}

// ============================================================================
// Round-trip Test
// ============================================================================
// Returns total absolute error after quantize → dequantize

inline float test_roundtrip(const float* orig, int num_weights) {
    uint8_t* packed = new uint8_t[num_weights / 2];
    float* deq = new float[num_weights];
    float scale;

    quantize(orig, packed, scale, num_weights);
    dequantize(packed, scale, deq, num_weights);

    float err = 0.0f;
    for (int i = 0; i < num_weights; i++)
        err += fabsf(orig[i] - deq[i]);

    delete[] packed;
    delete[] deq;

    return err;
}

} // namespace bq4
} // namespace bullet

#endif // BQ4_KERNELS_H
