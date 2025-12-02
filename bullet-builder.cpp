#ifndef BULLET_BUILDER_H
#define BULLET_BUILDER_H
#include <iostream>
#include <cstdint>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <map>
#include <cassert>
#include <sstream>
#include <iomanip>

namespace bullet {
namespace builder {

// ==================================================================================
// 🟢 CONFIG & CONSTANTS
// ==================================================================================

constexpr uint32_t MAGIC_TOKENIZER = 0x4B4C5542; // "BULK" (Little Endian: K, L, U, B)
constexpr uint32_t MAGIC_WEIGHTS = 0x30545742;   // "BWT0" (Little Endian: 0, T, W, B)

enum class QuantType : uint8_t {
    BQ4 = 0,
    BQ5 = 1,
    BQ8 = 2,
    FP16 = 3
};

// ==================================================================================
// 🟢 UTILS
// ==================================================================================

// FNV-1a 64-bit hash
constexpr uint64_t fnv1a64(const char* str, size_t len) {
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < len; ++i) {
        hash ^= static_cast<uint8_t>(str[i]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

uint64_t hash_name(const std::string& name) {
    return fnv1a64(name.c_str(), name.size());
}

// Align offset to N bytes
size_t align_to(size_t offset, size_t alignment) {
    size_t remainder = offset % alignment;
    if (remainder == 0) return offset;
    return offset + (alignment - remainder);
}

// ==================================================================================
// 🟢 DATA STRUCTURES
// ==================================================================================

struct TensorInfo {
    std::string name;
    std::vector<uint16_t> shape;
    std::vector<float> tensor_data; // FP32 source data
};

struct BulletHeader {
    std::string version = "1.0";
    std::string model_name = "bullet-model";
    std::string architecture = "bullet-hybrid-transformer";
    int hidden_size = 256;
    int num_layers = 8;
    int num_heads = 4;
    int vocab_size = 0;
    int max_context = 2048;
    uint64_t tokenizer_offset = 0;
    uint64_t weights_offset = 0;
    
    std::string to_json() const {
        std::stringstream ss;
        ss << "{\n";
        ss << "  \"bullet_version\": \"" << version << "\",\n";
        ss << "  \"model_name\": \"" << model_name << "\",\n";
        ss << "  \"architecture\": \"" << architecture << "\",\n";
        ss << "  \"dimensions\": {\n";
        ss << "      \"hidden_size\": " << hidden_size << ",\n";
        ss << "      \"num_layers\": " << num_layers << ",\n";
        ss << "      \"num_heads\": " << num_heads << "\n";
        ss << "  },\n";
        ss << "  \"quantization\": {\n";
        ss << "      \"type\": \"BQ4\"\n";
        ss << "  },\n";
        ss << "  \"vocab_size\": " << vocab_size << ",\n";
        ss << "  \"max_context\": " << max_context << ",\n";
        ss << "  \"file_offsets\": {\n";
        ss << "      \"tokenizer_start\": " << tokenizer_offset << ",\n";
        ss << "      \"weights_start\": " << weights_offset << "\n";
        ss << "  }\n";
        ss << "}";
        return ss.str();
    }
};

// ==================================================================================
// 🟢 BQ4 QUANTIZATION KERNEL
// ==================================================================================

// FP32 to FP16 helper (IEEE 754)
uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 31) & 0x1;
    uint32_t exp = (x >> 23) & 0xFF;
    uint32_t mant = x & 0x7FFFFF;
    
    uint16_t h;
    if (exp == 0) {
        h = sign << 15; // Zero
    } else if (exp == 0xFF) {
        h = (sign << 15) | 0x7C00 | (mant ? 1 : 0); // Inf/NaN
    } else {
        int new_exp = exp - 127 + 15;
        if (new_exp >= 31) {
            h = (sign << 15) | 0x7C00; // Overflow to Inf
        } else if (new_exp <= 0) {
            h = sign << 15; // Underflow to Zero (simplified)
        } else {
            h = (sign << 15) | (new_exp << 10) | (mant >> 13);
        }
    }
    return h;
}

// Block: [scale(fp16), zero(int8), pad(u8), 16 bytes data] = 20 bytes
void quantize_block_bq4(const float* src, uint8_t* dst) {
    // 1. Find min/max
    float min_val = src[0];
    float max_val = src[0];
    for (int i = 1; i < 32; ++i) {
        if (src[i] < min_val) min_val = src[i];
        if (src[i] > max_val) max_val = src[i];
    }
    
    // 2. Compute scale & zero
    // We want to map [min, max] to [0, 15]
    // val = (real / scale) + zero
    // real = (val - zero) * scale
    
    float range = max_val - min_val;
    float scale = range / 15.0f;
    if (scale < 1e-8f) scale = 1e-8f; // Avoid div by zero
    
    int8_t zero = static_cast<int8_t>(round(-min_val / scale));
    
    // Store scale as fp16
    uint16_t scale_fp16 = fp32_to_fp16(scale);
    
    *reinterpret_cast<uint16_t*>(dst) = scale_fp16;
    dst[2] = zero;
    dst[3] = 0; // Padding
    
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

// ==================================================================================
// 🟢 BUILDER CLASS
// ==================================================================================

class BulletBuilder {
    std::string out_path;
    BulletHeader header;
    struct Token {
        std::string text;
        float score;
    };
    std::vector<Token> vocab;
    struct Merge {
        std::string t1;
        std::string t2;
    };
    std::vector<Merge> merges;
    std::vector<TensorInfo> tensors;
    
public:
    BulletBuilder(const std::string& path) : out_path(path) {}
    
    void set_metadata(int hidden, int layers, int heads, int ctx) {
        header.hidden_size = hidden;
        header.num_layers = layers;
        header.num_heads = heads;
        header.max_context = ctx;
    }
    
    void load_vocab(const std::string& vocab_path) {
        std::ifstream f(vocab_path);
        std::string line;
        while (std::getline(f, line)) {
            // Trim newline
            if (!line.empty() && line.back() == '\n') line.pop_back();
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            
            // Parse "token score" or just "token" (default score 0)
            // If space exists, split. But token might contain space?
            // Usually vocab files are: "token" or "token score"
            // Let's assume last space separates score if it's a number.
            
            // Simple approach: look for last space
            size_t last_space = line.find_last_of(' ');
            float score = 0.0f;
            std::string text = line;
            
            if (last_space != std::string::npos) {
                std::string score_str = line.substr(last_space + 1);
                try {
                    size_t processed;
                    score = std::stof(score_str, &processed);
                    if (processed == score_str.size()) {
                        text = line.substr(0, last_space);
                    } else {
                        // Not a pure number, treat as part of token
                        score = 0.0f;
                    }
                } catch (...) {
                    // Conversion failed, treat as part of token
                    score = 0.0f;
                }
            }
            
            // Handle special replacement for space (e.g. u2581) if needed
            // For now, raw text.
            
            vocab.push_back({text, score});
        }
        header.vocab_size = vocab.size();
        std::cout << "Loaded " << vocab.size() << " tokens.\n";
    }
    
    void load_merges(const std::string& merges_path) {
        std::ifstream f(merges_path);
        std::string line;
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\n') line.pop_back();
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            
            size_t space = line.find(' ');
            if (space != std::string::npos) {
                std::string t1 = line.substr(0, space);
                std::string t2 = line.substr(space + 1);
                merges.push_back({t1, t2});
            }
        }
        std::cout << "Loaded " << merges.size() << " merges.\n";
    }
    
    void add_tensor(const std::string& name, const std::vector<uint16_t>& shape, const std::vector<float>& input_data) {
        TensorInfo t;
        t.name = name;
        t.shape = shape;
        t.tensor_data = input_data;
        tensors.push_back(t);
    }
    
    void build() {
        std::ofstream out(out_path, std::ios::binary);
        if (!out) throw std::runtime_error("Cannot open output file");
        
        // 1. Write Placeholder Header
        // We don't know offsets yet. Write enough 0s.
        std::string json = header.to_json();
        // Pad JSON to 4KB for alignment? Or just write it.
        // Spec says: JSON terminated by 4 null bytes.
        // We'll write it later. For now, reserve space.
        // Actually, we can compute offsets sequentially if we buffer.
        // But easier to write placeholders and seek back.
        
        // Let's assume header fits in 4096 bytes.
        std::vector<char> header_pad(4096, 0);
        out.write(header_pad.data(), 4096);
        
        // 2. Write Tokenizer
        uint64_t tokenizer_pos = out.tellp();
        // Align to 4KB?
        // Spec: Tokenizer MUST start on 4KB boundary.
        // We already wrote 4096 bytes, so we are at 4096. Good.
        
        header.tokenizer_offset = tokenizer_pos;
        
        // Magic BULK
        uint32_t magic_tok = MAGIC_TOKENIZER; 
        // Note: MAGIC_TOKENIZER is 0x4B4C5542.
        // If we write as uint32 on Little Endian machine, it writes 42 55 4C 4B ('B','U','L','K'). Correct.
        out.write(reinterpret_cast<const char*>(&magic_tok), 4);
        
        uint32_t v_size = vocab.size();
        out.write(reinterpret_cast<const char*>(&v_size), 4);
        
        for (const auto& token : vocab) {
            // Write Score
            out.write(reinterpret_cast<const char*>(&token.score), 4);
            
            uint16_t len = token.text.size();
            out.write(reinterpret_cast<const char*>(&len), 2);
            out.write(token.text.data(), len);
        }
        
        // 2b. Write Merges
        uint32_t num_merges = merges.size();
        out.write(reinterpret_cast<const char*>(&num_merges), 4);
        
        for (const auto& m : merges) {
            uint16_t l1 = m.t1.size();
            out.write(reinterpret_cast<const char*>(&l1), 2);
            out.write(m.t1.data(), l1);
            
            uint16_t l2 = m.t2.size();
            out.write(reinterpret_cast<const char*>(&l2), 2);
            out.write(m.t2.data(), l2);
        }
        
        // 3. Write Weights
        // Align to 32 bytes?
        // Spec: Weights block starts... doesn't strictly say alignment for block start, 
        // but tensors inside must be 32-byte aligned.
        // Let's align weights block start to 64 bytes as per spec "All blocks MUST start on 64-byte alignment"
        
        long pos = out.tellp();
        long aligned_pos = align_to(pos, 64);
        std::vector<char> pad(aligned_pos - pos, 0);
        if (!pad.empty()) out.write(pad.data(), pad.size());
        
        header.weights_offset = out.tellp();
        
        // Magic BWT0
        uint32_t magic_w = MAGIC_WEIGHTS;
        out.write(reinterpret_cast<const char*>(&magic_w), 4);
        
        uint32_t num_tensors = tensors.size();
        std::cout << "[DEBUG] BulletBuilder: Writing " << num_tensors << " tensors.\n";
        out.write(reinterpret_cast<const char*>(&num_tensors), 4);
        
        for (const auto& t : tensors) {
            // Metadata
            uint64_t hash = hash_name(t.name);
            out.write(reinterpret_cast<const char*>(&hash), 8);
            
            uint8_t rank = t.shape.size();
            out.write(reinterpret_cast<const char*>(&rank), 1);
            
            // Shape (always 4 * uint16 in struct, but here we write rank * uint16? 
            // Spec says: [shape: uint16 * rank]. 
            // Core implementation read 4 * uint16. 
            // Let's follow Spec: "uint16 * rank".
            // WAIT. Core implementation: `memcpy(tv.shape, ptr, sizeof(uint16_t) * 4);`
            // This implies Core expects fixed 4 dims.
            // Spec text: `[shape: uint16 * rank]`
            // There is a conflict.
            // If Core reads 8 bytes (4 dims), we MUST write 8 bytes.
            // Let's write 4 dims, padding with 1s.
            
            uint16_t shape_fixed[4] = {1, 1, 1, 1};
            for(int i=0; i<rank && i<4; ++i) shape_fixed[i] = t.shape[i];
            out.write(reinterpret_cast<const char*>(shape_fixed), 8);
            
            uint8_t q_type = static_cast<uint8_t>(QuantType::BQ4);
            out.write(reinterpret_cast<const char*>(&q_type), 1);
            
            // Compress Data
            // BQ4: 32 floats -> 20 bytes.
            size_t numel = t.tensor_data.size();
            size_t num_blocks = (numel + 31) / 32;
            uint32_t compressed_size = num_blocks * 20;
            
            out.write(reinterpret_cast<const char*>(&compressed_size), 4);
            
            // Padding for data alignment (32 bytes)
            long curr = out.tellp();
            long next_aligned = align_to(curr, 32);
            std::vector<char> dpad(next_aligned - curr, 0);
            if (!dpad.empty()) out.write(dpad.data(), dpad.size());
            
            // Write Blocks
            std::vector<uint8_t> block_buf(20);
            for (size_t b = 0; b < num_blocks; ++b) {
                float src_buf[32];
                std::memset(src_buf, 0, sizeof(src_buf));
                
                size_t start = b * 32;
                size_t count = std::min((size_t)32, numel - start);
                for(size_t i=0; i<count; ++i) src_buf[i] = t.tensor_data[start+i];
                
                quantize_block_bq4(src_buf, block_buf.data());
                out.write(reinterpret_cast<const char*>(block_buf.data()), 20);
            }
        }
        
        // Footer
        out.write("END!", 4);
        
        // 4. Rewrite Header
        out.seekp(0);
        json = header.to_json();
        out.write(json.c_str(), json.size());
        // Terminator
        char term[4] = {0, 0, 0, 0};
        out.write(term, 4);
        
        std::cout << "Successfully built " << out_path << "\n";
        std::cout << "Tokenizer Offset: " << header.tokenizer_offset << "\n";
        std::cout << "Weights Offset: " << header.weights_offset << "\n";
    }
};

// ==================================================================================
// 🟢 MAIN
// ==================================================================================

} // namespace builder
} // namespace bullet

using namespace bullet::builder;

#ifndef BULLET_TEST
int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: ./bullet-builder <vocab.txt> <merges.txt> <out.bullet>\n";
        std::cout << "Note: This is a dummy builder. Real one needs PyTorch bindings.\n";
        std::cout << "It will generate a random model for testing.\n";
        return 1;
    }
    
    try {
        std::string vocab_path = argv[1];
        std::string merges_path = argv[2];
        std::string out_path = argv[3];
        
        BulletBuilder builder(out_path);
        builder.load_vocab(vocab_path);
        builder.load_merges(merges_path);
        
        // Dummy Metadata
        builder.set_metadata(256, 2, 4, 128);
        
        // Generate Dummy Tensors
        // 1. Token Embeddings: [vocab, 256]
        // We need vocab size from builder, but it's private.
        // Let's just assume small vocab for test or read file again.
        // We'll just add a few tensors.
        
        std::vector<float> dummy_data(256 * 32); // Some data
        for(auto& f : dummy_data) f = ((float)rand() / RAND_MAX) - 0.5f;
        
        builder.add_tensor("tok_embeddings.weight", std::vector<uint16_t>{32, 256}, dummy_data);
        
        for(int l=0; l<2; ++l) {
            std::string prefix = "layers." + std::to_string(l) + ".";
            builder.add_tensor(prefix + "norm.attn.weight", std::vector<uint16_t>{256}, dummy_data); // Should be 1D, but using BQ4 so needs to be multiple of 32? 
            // rms_norm code: `for (int i = 0; i < size; i += 32) dequant_block_bq4...`
            // It expects BQ4 blocks. So size 256 is fine (8 blocks).
            
            builder.add_tensor(prefix + "attention.query.weight", std::vector<uint16_t>{256, 256}, dummy_data);
            builder.add_tensor(prefix + "attention.key.weight", std::vector<uint16_t>{256, 256}, dummy_data);
            builder.add_tensor(prefix + "attention.value.weight", std::vector<uint16_t>{256, 256}, dummy_data);
            builder.add_tensor(prefix + "attention.output.weight", std::vector<uint16_t>{256, 256}, dummy_data);
            
            builder.add_tensor(prefix + "norm.ffn.weight", std::vector<uint16_t>{256}, dummy_data);
            
            builder.add_tensor(prefix + "ffn.w1.weight", std::vector<uint16_t>{256, 1024}, dummy_data); // Gate
            builder.add_tensor(prefix + "ffn.w3.weight", std::vector<uint16_t>{256, 1024}, dummy_data); // Up
            builder.add_tensor(prefix + "ffn.w2.weight", std::vector<uint16_t>{1024, 256}, dummy_data); // Down
        }
        
        builder.add_tensor("final_layernorm.weight", std::vector<uint16_t>{256}, dummy_data);
        builder.add_tensor("output_head.weight", std::vector<uint16_t>{32, 256}, dummy_data);
        
        builder.build();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
#endif
#endif
