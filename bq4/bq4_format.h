// bq4_format.h
// BQ4 File Format Specification
// Bullet OS - 4-bit Quantization System

#ifndef BQ4_FORMAT_H
#define BQ4_FORMAT_H

#include <cstdint>
#include <string>
#include <vector>

namespace bullet {
namespace bq4 {

// ============================================================================
// BQ4 File Header
// ============================================================================

struct Header {
    char magic[4];          // "BQ4F"
    uint32_t version;       // 1
    uint32_t num_tensors;   // number of weight tensors
};

// ============================================================================
// Tensor Directory Entry
// ============================================================================

struct TensorEntry {
    std::string name;
    std::vector<uint32_t> dims;
    uint32_t block_size;     // e.g., 32 or 64
    uint32_t num_blocks;     // total blocks for this tensor
    uint64_t data_offset;    // offset of quantized blocks from start of file
};

// ============================================================================
// BQ4 Block (in-memory representation)
// ============================================================================

struct Block {
    std::vector<uint8_t> data;  // packed int4 (size = block_size/2)
    float scale;                // scale factor
};

// ============================================================================
// BQ4 Tensor (in-memory representation)
// ============================================================================

struct Tensor {
    std::string name;
    std::vector<uint32_t> dims;
    uint32_t block_size;
    std::vector<Block> blocks;
    
    // Helper: get total number of weights
    uint32_t total_weights() const {
        uint32_t total = 1;
        for (auto d : dims) total *= d;
        return total;
    }
};

} // namespace bq4
} // namespace bullet

#endif // BQ4_FORMAT_H
