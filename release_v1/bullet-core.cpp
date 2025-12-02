#ifndef BULLET_CORE_H
#define BULLET_CORE_H
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <memory>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cassert>
#include <map>
#include <random>
#include <iomanip>

namespace bullet {
namespace core {

// ==================================================================================
// 🟢 CONSTANTS & CONFIG
// ==================================================================================

constexpr uint32_t MAGIC_TOKENIZER = 0x4B4C5542; // "BULK"
constexpr uint32_t MAGIC_WEIGHTS = 0x30545742;   // "BWT0"

enum class QuantType : uint8_t {
    BQ4 = 0,
    BQ5 = 1,
    BQ8 = 2,
    FP16 = 3
};

enum class Task {
    GEN,
    NER,
    POS,
    SENTIMENT,
    CLS
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

inline uint64_t hash_name(const std::string& name) {
    return fnv1a64(name.c_str(), name.size());
}

// Minimal JSON parser
inline std::string json_get_string(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\":";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";
    
    pos = json.find("\"", pos + search.length());
    if (pos == std::string::npos) return "";
    size_t start = pos + 1;
    size_t end = json.find("\"", start);
    return json.substr(start, end - start);
}

inline int json_get_int(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\":";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return 0;
    
    size_t start = pos + search.length();
    while (start < json.length() && (json[start] == ' ' || json[start] == '\n')) start++;
    size_t end = start;
    while (end < json.length() && (isdigit(json[end]) || json[end] == '-')) end++;
    return std::stoi(json.substr(start, end - start));
}

// FP16 Utils (IEEE 754)
inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h & 0x7C00) >> 10;
    uint32_t mant = (h & 0x03FF) << 13;
    
    uint32_t f_bits;
    if (exp == 0) {
        if (mant == 0) {
            f_bits = sign; // Zero
        } else {
            // Denormalized number
            // Renormalize for FP32
            // mant is currently in bits 22..13 (shifted left by 13)
            // We need to shift left until the leading bit is 1
            // But mant is 10 bits.
            // Let's work with the 10-bit mantissa directly first
            uint32_t m = h & 0x03FF;
            int e = -14;
            while ((m & 0x0400) == 0) {
                m <<= 1;
                e--;
            }
            m &= ~0x0400; // Clear the implicit leading 1
            // Now m is the mantissa bits (top bits)
            // We need to shift it to be in the top of 23 bits
            // m has 10 bits of precision.
            // FP32 mantissa is 23 bits.
            // We shifted m so that the leading 1 was at bit 10 (0x400).
            // Actually, let's use float magic or just simple math if performance allows.
            // But we want bitwise for speed.
            
            // Simpler approach for denormals:
            // val = mant * 2^-14 * 2^-10 = mant * 2^-24
            return (sign ? -1.0f : 1.0f) * (float)(h & 0x03FF) * (1.0f / 16777216.0f);
        }
    } else if (exp == 0x1F) {
        f_bits = sign | 0x7F800000 | mant; // Inf/NaN
    } else {
        f_bits = sign | ((exp + 112) << 23) | mant;
    }
    
    float f;
    std::memcpy(&f, &f_bits, 4);
    return f;
}

inline uint16_t fp32_to_fp16(float f) {
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

// ==================================================================================
// 🟢 DATA STRUCTURES
// ==================================================================================

struct BulletHeader {
    std::string version;
    std::string model_name;
    std::string architecture;
    int hidden_size;
    int num_layers;
    int num_heads;
    int vocab_size;
    int max_context;
    uint64_t tokenizer_offset;
    uint64_t weights_offset;
};

struct TensorView {
    uint64_t name_hash;
    uint8_t rank;
    uint16_t shape[4];
    QuantType quant_type;
    uint32_t compressed_size;
    const uint8_t* tensor_data;
    bool valid = false;
    
    size_t numel() const {
        size_t n = 1;
        for(int i=0; i<4; ++i) if(shape[i] > 0) n *= shape[i];
        return n;
    }
};

// Scratch buffer to avoid heap allocations during inference
struct ScratchBuffer {
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> attn_out;
    std::vector<float> ffn_gate;
    std::vector<float> ffn_up;
    std::vector<float> ffn_down;
    std::vector<float> ffn_out;
    std::vector<float> logits;
    std::vector<float> hidden_state;
    std::vector<float> residual;
    
    void resize(int dim, int hidden_dim, int vocab_size) {
        q.resize(dim);
        k.resize(dim);
        v.resize(dim);
        attn_out.resize(dim);
        ffn_gate.resize(hidden_dim);
        ffn_up.resize(hidden_dim);
        ffn_down.resize(dim); // Output of down proj
        ffn_out.resize(dim);
        logits.resize(vocab_size);
        hidden_state.resize(dim);
        residual.resize(dim);
    }
};

struct BulletContext {
    // KV Cache: [layers, seq_len, heads, head_dim] stored as FP16
    std::vector<std::vector<uint16_t>> k_cache; 
    std::vector<std::vector<uint16_t>> v_cache;
    
    int current_pos = 0;
    int head_dim;
    int num_heads;
    int num_layers;
    
    ScratchBuffer scratch;
    
    BulletContext(int layers, int heads, int h_dim, int max_seq, int hidden_size, int vocab_size) 
        : num_layers(layers), num_heads(heads), head_dim(h_dim) {
        
        fprintf(stderr, "[DEBUG] BulletContext: layers=%d heads=%d h_dim=%d max_seq=%d\n", layers, heads, h_dim, max_seq);
        size_t layer_size = max_seq * heads * head_dim;
        fprintf(stderr, "[DEBUG] BulletContext: layer_size=%zu\n", layer_size);
        
        k_cache.resize(layers, std::vector<uint16_t>(layer_size, 0));
        v_cache.resize(layers, std::vector<uint16_t>(layer_size, 0));
        
        // SwiGLU hidden dim is usually 4 * dim, but let's assume standard multiplier
        // Or read from config? For now assume 4x.
        scratch.resize(heads * h_dim, heads * h_dim * 4, vocab_size);
        fprintf(stderr, "[DEBUG] BulletContext: Initialized.\n");
    }
};

// ==================================================================================
// 🟢 MATH KERNELS
// ==================================================================================

// BQ4 Dequantization
// Block: [scale(fp16), zero(int8), pad(u8), 16 bytes data] = 20 bytes
inline void dequant_block_bq4(const uint8_t* block, float* out) {
    uint16_t scale_bits = *reinterpret_cast<const uint16_t*>(block);
    float scale = fp16_to_fp32(scale_bits);
    
    int8_t zero = static_cast<int8_t>(block[2]);
    const uint8_t* block_data = block + 4;
    
    for (int i = 0; i < 16; ++i) {
        uint8_t b = block_data[i];
        out[i*2] = ((b & 0x0F) - zero) * scale;
        out[i*2+1] = ((b >> 4) - zero) * scale;
    }
}

// Matmul: x [in_dim] @ w [out_dim, in_dim] -> out [out_dim]
inline void matmul(const float* x, const TensorView& w, float* out, int in_dim, int out_dim) {
    // Naive implementation. 
    // W is quantized BQ4.
    
    // Block size 32.
    float block_vals[32];
    const uint8_t* w_ptr = w.tensor_data;
    int block_size_bytes = 20;
    
    for (int r = 0; r < out_dim; ++r) {
        float sum = 0.0f;
        for (int c = 0; c < in_dim; c += 32) {
            dequant_block_bq4(w_ptr, block_vals);
            w_ptr += block_size_bytes;
            
            // Unroll slightly
            for (int k = 0; k < 32; ++k) {
                if (c + k < in_dim) {
                    sum += x[c+k] * block_vals[k];
                }
            }
        }
        out[r] = sum;
    }
}

inline void rms_norm(float* x, const TensorView& w, int size) {
    float ss = 0.0f;
    for (int i = 0; i < size; ++i) ss += x[i] * x[i];
    ss /= size;
    float inv_rms = 1.0f / sqrt(ss + 1e-5f);
    
    // Dequantize weight on the fly
    float w_vals[32];
    const uint8_t* w_ptr = w.tensor_data;
    
    for (int i = 0; i < size; i += 32) {
        dequant_block_bq4(w_ptr, w_vals);
        w_ptr += 20;
        for (int k = 0; k < 32 && (i+k) < size; ++k) {
            x[i+k] = x[i+k] * inv_rms * w_vals[k];
        }
    }
}

inline void softmax(float* x, int size) {
    float max_val = x[0];
    for(int i=1; i<size; ++i) if(x[i] > max_val) max_val = x[i];
    
    float sum = 0.0f;
    for(int i=0; i<size; ++i) {
        x[i] = exp(x[i] - max_val);
        sum += x[i];
    }
    for(int i=0; i<size; ++i) x[i] /= sum;
}

inline void silu(float* x, int size) {
    for(int i=0; i<size; ++i) {
        float sig = 1.0f / (1.0f + exp(-x[i]));
        x[i] = x[i] * sig;
    }
}

inline void multiply(float* x, const float* y, int size) {
    for(int i=0; i<size; ++i) x[i] *= y[i];
}

inline void add(float* x, const float* y, int size) {
    for(int i=0; i<size; ++i) x[i] += y[i];
}

inline void apply_rope(float* q, float* k, int pos, int head_dim, int num_heads) {
    float theta = 10000.0f;
    for (int h = 0; h < num_heads; ++h) {
        for (int i = 0; i < head_dim / 2; ++i) {
            float freq = 1.0f / pow(theta, 2.0f * i / head_dim);
            float val = pos * freq;
            float cos_val = cos(val);
            float sin_val = sin(val);
            
            int idx = h * head_dim + 2 * i;
            float q0 = q[idx];
            float q1 = q[idx+1];
            float k0 = k[idx];
            float k1 = k[idx+1];
            
            q[idx]   = q0 * cos_val - q1 * sin_val;
            q[idx+1] = q0 * sin_val + q1 * cos_val;
            k[idx]   = k0 * cos_val - k1 * sin_val;
            k[idx+1] = k0 * sin_val + k1 * cos_val;
        }
    }
}

// ==================================================================================
// 🟢 CORE CLASSES
// ==================================================================================

class BulletTokenizer {
    struct Token {
        std::string text;
        float score;
        int id;
    };
    std::vector<Token> vocab;
    std::unordered_map<std::string, int> token_to_id;
    std::vector<std::pair<std::string, std::string>> merges;
    
public:
    void load(const uint8_t* input_data, int vocab_size) {
        const uint8_t* ptr = input_data;
        if (ptr[0] != 'B' || ptr[1] != 'U' || ptr[2] != 'L' || ptr[3] != 'K') {
            std::cerr << "Invalid Tokenizer Magic\n";
            return;
        }
        ptr += 4;
        ptr += 4; // Skip vocab size
        
        for (int i = 0; i < vocab_size; ++i) {
            float score = *reinterpret_cast<const float*>(ptr);
            ptr += 4;
            
            uint16_t len = *reinterpret_cast<const uint16_t*>(ptr);
            ptr += 2;
            std::string text(reinterpret_cast<const char*>(ptr), len);
            ptr += len;
            
            vocab.push_back({text, score, i});
            token_to_id[text] = i;
        }
        
        // Load Merges
        uint32_t num_merges = *reinterpret_cast<const uint32_t*>(ptr);
        ptr += 4;
        fprintf(stderr, "[DEBUG] Tokenizer: Loading %u merges...\n", num_merges);
        
        for (uint32_t i = 0; i < num_merges; ++i) {
            uint16_t l1 = *reinterpret_cast<const uint16_t*>(ptr); ptr += 2;
            std::string t1(reinterpret_cast<const char*>(ptr), l1); ptr += l1;
            
            uint16_t l2 = *reinterpret_cast<const uint16_t*>(ptr); ptr += 2;
            std::string t2(reinterpret_cast<const char*>(ptr), l2); ptr += l2;
            
            merges.push_back({t1, t2});
        }
    }
    
    std::vector<int> encode(const std::string& text) {
        std::vector<int> tokens;
        if (text.empty()) return tokens;
        
        // 1. Pre-tokenization (Split by space)
        // Note: This is a simplification. Real GPT-2/4 uses regex.
        // But for our Marathi dataset, space splitting is a reasonable start.
        std::vector<std::string> words;
        std::string current_word;
        for (char c : text) {
            if (c == ' ') {
                if (!current_word.empty()) {
                    words.push_back(current_word);
                    current_word.clear();
                }
                words.push_back(" "); // Keep space as a word? Or handle it?
                // Our tokenizer has ' ' as a token.
            } else {
                current_word += c;
            }
        }
        if (!current_word.empty()) words.push_back(current_word);
        
        // 2. BPE Merge
        for (const auto& word : words) {
            // Split into chars (UTF-8 aware?)
            // C++ std::string is bytes. We need to split by UTF-8 characters.
            // Simple hack: if byte & 0xC0 != 0x80, it's a start of char.
            std::vector<std::string> split;
            for (size_t i = 0; i < word.size(); ) {
                size_t len = 1;
                unsigned char c = word[i];
                if ((c & 0x80) == 0) len = 1;
                else if ((c & 0xE0) == 0xC0) len = 2;
                else if ((c & 0xF0) == 0xE0) len = 3;
                else if ((c & 0xF8) == 0xF0) len = 4;
                
                if (i + len > word.size()) len = word.size() - i; // Safety
                split.push_back(word.substr(i, len));
                i += len;
            }
            
            // Apply merges
            for (const auto& pair : merges) {
                if (split.size() < 2) break;
                
                std::vector<std::string> new_split;
                for (size_t i = 0; i < split.size(); ++i) {
                    if (i < split.size() - 1 && split[i] == pair.first && split[i+1] == pair.second) {
                        new_split.push_back(pair.first + pair.second);
                        i++; // Skip next
                    } else {
                        new_split.push_back(split[i]);
                    }
                }
                split = new_split;
            }
            
            // Map to IDs
            for (const auto& token : split) {
                if (token_to_id.count(token)) {
                    tokens.push_back(token_to_id[token]);
                } else {
                    // Fallback to bytes or UNK?
                    // Try to match chars if token not found (shouldn't happen if vocab is complete)
                    // If not found, maybe UNK (0)?
                    // fprintf(stderr, "[WARN] Tokenizer: Unknown token '%s'\n", token.c_str());
                    // Try to decompose further? Or just UNK.
                    // For now, UNK.
                     tokens.push_back(0); 
                }
            }
        }
        
        return tokens;
    }
    
    std::string decode(int id) {
        if (id >= 0 && id < vocab.size()) return vocab[id].text;
        return "";
    }
    
    int size() const { return vocab.size(); }
};

class BulletLoader {
    int fd;
    void* mmap_ptr;
    size_t file_size;
    BulletHeader header;
    std::unordered_map<uint64_t, TensorView> tensors;
    
public:
    BulletLoader(const std::string& path) {
        std::cout << "[DEBUG] BulletLoader: Opening " << path << "\n";
        fd = open(path.c_str(), O_RDONLY);
        if (fd == -1) throw std::runtime_error("Cannot open file");
        
        struct stat sb;
        fstat(fd, &sb);
        file_size = sb.st_size;
        std::cout << "[DEBUG] BulletLoader: File size: " << file_size << "\n";
        
        mmap_ptr = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mmap_ptr == MAP_FAILED) throw std::runtime_error("mmap failed");
        std::cout << "[DEBUG] BulletLoader: mmap success.\n";
        
        parse_header();
        std::cout << "[DEBUG] BulletLoader: Header parsed.\n";
        map_tensors();
        std::cout << "[DEBUG] BulletLoader: Tensors mapped.\n";
    }
    
    // Delete copy
    BulletLoader(const BulletLoader&) = delete;
    BulletLoader& operator=(const BulletLoader&) = delete;
    
    // Move constructor
    BulletLoader(BulletLoader&& other) noexcept 
        : fd(other.fd), mmap_ptr(other.mmap_ptr), file_size(other.file_size), 
          header(std::move(other.header)), tensors(std::move(other.tensors)) {
        other.fd = -1;
        other.mmap_ptr = nullptr;
    }
    
    // Move assignment
    BulletLoader& operator=(BulletLoader&& other) noexcept {
        if (this != &other) {
            if (mmap_ptr) munmap(mmap_ptr, file_size);
            if (fd != -1) close(fd);
            
            fd = other.fd;
            mmap_ptr = other.mmap_ptr;
            file_size = other.file_size;
            header = std::move(other.header);
            tensors = std::move(other.tensors);
            
            other.fd = -1;
            other.mmap_ptr = nullptr;
        }
        return *this;
    }
    
    ~BulletLoader() {
        if (mmap_ptr) munmap(mmap_ptr, file_size);
        if (fd != -1) close(fd);
    }
    
    void parse_header() {
        const char* ptr = static_cast<const char*>(mmap_ptr);
        size_t json_len = 0;
        while (json_len < file_size - 4) {
            if (ptr[json_len] == 0 && ptr[json_len+1] == 0 && ptr[json_len+2] == 0 && ptr[json_len+3] == 0) {
                break;
            }
            json_len++;
        }
        
        std::string json(ptr, json_len);
        header.version = json_get_string(json, "bullet_version");
        header.hidden_size = json_get_int(json, "hidden_size");
        header.num_layers = json_get_int(json, "num_layers");
        header.num_heads = json_get_int(json, "num_heads");
        header.vocab_size = json_get_int(json, "vocab_size");
        header.max_context = json_get_int(json, "max_context");
        header.tokenizer_offset = json_get_int(json, "tokenizer_start");
        header.weights_offset = json_get_int(json, "weights_start");
    }
    
    void map_tensors() {
        const uint8_t* ptr = static_cast<const uint8_t*>(mmap_ptr) + header.weights_offset;
        
        if (ptr[0] != 'B' || ptr[1] != 'W' || ptr[2] != 'T' || ptr[3] != '0') {
             // Warn?
        }
        ptr += 4;
        
        uint32_t num_tensors = *reinterpret_cast<const uint32_t*>(ptr);
        fprintf(stderr, "[DEBUG] map_tensors: Found %u tensors.\n", num_tensors);
        fflush(stderr);
        ptr += 4;
        
        for (uint32_t i = 0; i < num_tensors; ++i) {
            TensorView tv;
            tv.name_hash = *reinterpret_cast<const uint64_t*>(ptr); ptr += 8;
            tv.rank = *ptr; ptr += 1;
            memcpy(tv.shape, ptr, sizeof(uint16_t) * 4); ptr += 8;
            tv.quant_type = static_cast<QuantType>(*ptr); ptr += 1;
            tv.compressed_size = *reinterpret_cast<const uint32_t*>(ptr); ptr += 4;
            
            uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
            size_t padding = (32 - (addr % 32)) % 32;
            ptr += padding;
            
            tv.tensor_data = ptr;
            tv.valid = true;
            
            tensors[tv.name_hash] = tv;
            
            fprintf(stderr, "[DEBUG] map_tensors: Mapped hash=%lu rank=%d shape=[%d,%d,%d,%d] size=%d addr=%p\n", 
                    tv.name_hash, tv.rank, tv.shape[0], tv.shape[1], tv.shape[2], tv.shape[3], tv.compressed_size, tv.tensor_data);
            
            ptr += tv.compressed_size;
        }
        
        // Debug: Print all tensors
        for (auto& kv : tensors) {
             // We don't store the name string, only hash!
             // Ah, we can't print names unless we store them.
             // But we can print the count.
        }
        std::cout << "[DEBUG] BulletLoader: Mapped " << tensors.size() << " tensors.\n";
    }
    
    const BulletHeader& get_header() const { return header; }
    const uint8_t* get_tokenizer_ptr() const { 
        return static_cast<const uint8_t*>(mmap_ptr) + header.tokenizer_offset; 
    }
    
    TensorView get_tensor(const std::string& name) {
        uint64_t hash = hash_name(name);
        if (tensors.count(hash)) {
            // fprintf(stderr, "[DEBUG] get_tensor: Found %s\n", name.c_str());
            return tensors[hash];
        }
        fprintf(stderr, "[ERROR] get_tensor: Tensor '%s' not found! Hash: %lu\n", name.c_str(), hash);
        fflush(stderr);
        return TensorView{0};
    }
};

class BulletRunner {
    BulletLoader* loader;
    BulletHeader header;
    
public:
    BulletRunner(BulletLoader* l) : loader(l), header(l->get_header()) {}
    
    __attribute__((noinline)) void forward(int token, BulletContext& ctx) {
        // fprintf(stderr, "[DEBUG] forward: Start. this=%p token=%d pos=%d\n", this, token, ctx.current_pos);
        // fflush(stderr);
        int dim = header.hidden_size;
        int heads = header.num_heads;
        int head_dim = dim / heads;
        
        TensorView emb = loader->get_tensor("tok_embeddings.weight");
        // fprintf(stderr, "[DEBUG] forward: Got embedding tensor. Shape: %dx%d Valid: %d Data: %p\n", 
        //         emb.shape[0], emb.shape[1], emb.valid, emb.tensor_data);
        // fflush(stderr);
        
        if (!emb.valid || !emb.tensor_data) {
            fprintf(stderr, "[ERROR] forward: Invalid embedding tensor!\n");
            return;
        }

        const uint8_t* row_ptr = emb.tensor_data + (token * dim * 20 / 32);
        float* x = ctx.scratch.hidden_state.data();
        // fprintf(stderr, "[DEBUG] forward: row_ptr=%p x=%p dim=%d\n", row_ptr, x, dim);
        // fflush(stderr);

        for (int i = 0; i < dim; i+=32) {
            dequant_block_bq4(row_ptr, x + i);
            row_ptr += 20;
        }
        // fprintf(stderr, "[DEBUG] forward: Embedding done.\n");
        // fflush(stderr);
        
        // 2. Layers
        for (int l = 0; l < header.num_layers; ++l) {
            // fprintf(stderr, "[DEBUG] forward: Layer %d start.\n", l);
            std::string prefix = "layers." + std::to_string(l) + ".";
            
            // Save residual
            std::memcpy(ctx.scratch.residual.data(), x, dim * sizeof(float));
            
            // Norm
            rms_norm(x, loader->get_tensor(prefix + "norm.attn.weight"), dim);
            
            // QKV
            float* q = ctx.scratch.q.data();
            float* k = ctx.scratch.k.data();
            float* v = ctx.scratch.v.data();
            
            matmul(x, loader->get_tensor(prefix + "attention.query.weight"), q, dim, dim);
            matmul(x, loader->get_tensor(prefix + "attention.key.weight"), k, dim, dim);
            matmul(x, loader->get_tensor(prefix + "attention.value.weight"), v, dim, dim);
            
            // RoPE
            apply_rope(q, k, ctx.current_pos, head_dim, heads);
            
            // Update Cache (FP16)
            int offset = ctx.current_pos * dim;
            for(int i=0; i<dim; ++i) {
                ctx.k_cache[l][offset + i] = fp32_to_fp16(k[i]);
                ctx.v_cache[l][offset + i] = fp32_to_fp16(v[i]);
            }
            
            // Attention
            std::fill(ctx.scratch.attn_out.begin(), ctx.scratch.attn_out.end(), 0.0f);
            
            for (int h = 0; h < heads; ++h) {
                // Score against all prev tokens
                std::vector<float> scores(ctx.current_pos + 1);
                for (int t = 0; t <= ctx.current_pos; ++t) {
                    float dot = 0.0f;
                    for (int i = 0; i < head_dim; ++i) {
                        float k_val = fp16_to_fp32(ctx.k_cache[l][t*dim + h*head_dim + i]);
                        dot += q[h*head_dim + i] * k_val;
                    }
                    scores[t] = dot / sqrt(head_dim);
                }
                softmax(scores.data(), ctx.current_pos + 1);
                
                // Weighted sum
                for (int t = 0; t <= ctx.current_pos; ++t) {
                    float w = scores[t];
                    for (int i = 0; i < head_dim; ++i) {
                        float v_val = fp16_to_fp32(ctx.v_cache[l][t*dim + h*head_dim + i]);
                        ctx.scratch.attn_out[h*head_dim + i] += w * v_val;
                    }
                }
            }
            
            // Output proj
            // Reuse q buffer as temp output if needed, but we have attn_out
            // Project back to x (add to residual later)
            // We can project attn_out -> x directly? No, x is input.
            // Use q as temp output for projection.
            matmul(ctx.scratch.attn_out.data(), loader->get_tensor(prefix + "attention.output.weight"), x, dim, dim);
            
            // Add Residual
            add(x, ctx.scratch.residual.data(), dim);
            
            // FFN
            std::memcpy(ctx.scratch.residual.data(), x, dim * sizeof(float));
            rms_norm(x, loader->get_tensor(prefix + "norm.ffn.weight"), dim);
            
            // SwiGLU
            int hidden_dim = dim * 4;
            float* gate = ctx.scratch.ffn_gate.data();
            float* ffn_up = ctx.scratch.ffn_up.data();
            
            matmul(x, loader->get_tensor(prefix + "ffn.w1.weight"), gate, dim, hidden_dim);
            matmul(x, loader->get_tensor(prefix + "ffn.w3.weight"), ffn_up, dim, hidden_dim);
            
            // SwiGLU
            for (int i = 0; i < hidden_dim; ++i) {
                float val = gate[i];
                val = val / (1.0f + exp(-val)); // SiLU
                gate[i] = val * ffn_up[i];
            }
            
            matmul(gate, loader->get_tensor(prefix + "ffn.w2.weight"), x, hidden_dim, dim);
            
            // Add residual
            for (int i = 0; i < dim; ++i) x[i] += ctx.scratch.residual[i];
            
            fprintf(stderr, "[DEBUG] forward: Layer %d done.\n", l);
        }
        
        // 3. Final Norm
        rms_norm(x, loader->get_tensor("final_layernorm.weight"), dim);
        fprintf(stderr, "[DEBUG] forward: End.\n");
    }
    
    void compute_logits(BulletContext& ctx, Task task) {
        fprintf(stderr, "[DEBUG] compute_logits: Start. pos=%d\n", ctx.current_pos);
        // 1. Final LayerNorm
        // This step is done in the forward pass, not here.
        fprintf(stderr, "[DEBUG] compute_logits: Final LayerNorm done. (Note: Performed in forward pass)\n");
        // 2. Output Head
        std::string head_name;
        switch(task) {
            case Task::GEN: head_name = "gen_head"; break;
            case Task::NER: head_name = "ner_head"; break;
            case Task::POS: head_name = "pos_head"; break;
            case Task::SENTIMENT: head_name = "sentiment_head"; break;
            case Task::CLS: head_name = "classifier_head"; break;

        }
        
        TensorView w = loader->get_tensor(head_name + ".weight");
        if (!w.valid) w = loader->get_tensor("output_head.weight");
        
        int out_size = (task == Task::GEN) ? header.vocab_size : w.shape[0];
        matmul(ctx.scratch.hidden_state.data(), w, ctx.scratch.logits.data(), header.hidden_size, out_size);
    }
    
    int sample(BulletContext& ctx, float temp, float top_p, int top_k) {
        // Softmax logits first? Usually sampling is done on logits.
        // But for top-p we need probs.
        
        int vocab_size = header.vocab_size;
        float* logits = ctx.scratch.logits.data();
        
        // Temp
        if (temp > 0) {
            for(int i=0; i<vocab_size; ++i) logits[i] /= temp;
        }
        
        // Softmax
        softmax(logits, vocab_size);
        
        // Top-K & Top-P
        std::vector<std::pair<float, int>> probs(vocab_size);
        for(int i=0; i<vocab_size; ++i) probs[i] = {logits[i], i};
        
        std::sort(probs.rbegin(), probs.rend());
        
        // Top-K
        if (top_k > 0 && top_k < vocab_size) {
            probs.resize(top_k);
        }
        
        // Top-P
        if (top_p > 0 && top_p < 1.0f) {
            float cum_prob = 0.0f;
            int cut_idx = 0;
            for(size_t i=0; i<probs.size(); ++i) {
                cum_prob += probs[i].first;
                if (cum_prob > top_p) {
                    cut_idx = i;
                    break;
                }
            }
            probs.resize(cut_idx + 1);
        }
        
        // Sample
        float r = (float)rand() / RAND_MAX;
        float cum = 0.0f;
        for (const auto& p : probs) {
            cum += p.first; // Note: probs are not re-normalized, but simple sampling works
            // Ideally re-normalize.
        }
        
        // Re-normalize
        float sum = 0.0f;
        for(const auto& p : probs) sum += p.first;
        r *= sum;
        
        cum = 0.0f;
        for (const auto& p : probs) {
            cum += p.first;
            if (r <= cum) return p.second;
        }
        return probs.back().second;
    }
};

class BulletModel {
    std::unique_ptr<BulletLoader> loader;
    BulletTokenizer tokenizer;
    BulletRunner runner;
    
    static std::unique_ptr<BulletLoader> create_loader(const char* path) {
        fprintf(stderr, "[DEBUG] BulletModel: Creating loader for %s\n", path);
        return std::make_unique<BulletLoader>(std::string(path));
    }

public:
    BulletModel(const char* path) 
        : loader(create_loader(path)), runner(loader.get()) {
        fprintf(stderr, "[DEBUG] BulletModel: Loader initialized.\n");
        tokenizer.load(loader->get_tokenizer_ptr(), loader->get_header().vocab_size);
        fprintf(stderr, "[DEBUG] BulletModel: Tokenizer loaded.\n");
        srand(time(NULL));
    }
    
    // Delete copy
    BulletModel(const BulletModel&) = delete;
    BulletModel& operator=(const BulletModel&) = delete;
    
    // Move constructor (default is fine now because unique_ptr handles move, and runner ref stays valid)
    // Actually, runner holds pointer now, so default move is fine.
    BulletModel(BulletModel&&) = default;
    BulletModel& operator=(BulletModel&&) = default;
    
    std::string generate(const std::string& prompt, int max_tokens = 256) {
        fprintf(stderr, "[DEBUG] generate: Start. Prompt: %s\n", prompt.c_str());
        BulletContext ctx(loader->get_header().num_layers, 
                          loader->get_header().num_heads, 
                          loader->get_header().hidden_size / loader->get_header().num_heads, 
                          loader->get_header().max_context,
                          loader->get_header().hidden_size,
                          loader->get_header().vocab_size);
        fprintf(stderr, "[DEBUG] generate: Context created.\n");
                          
        std::vector<int> tokens = tokenizer.encode(prompt);
        fprintf(stderr, "[DEBUG] generate: Encoded %zu tokens.\n", tokens.size());
        
        for (int t : tokens) {
            fprintf(stderr, "[DEBUG] generate: Prefill token %d runner=%p\n", t, &runner);
            fflush(stderr);
            runner.forward(t, ctx);
            ctx.current_pos++;
        }
        
        std::string output;
        for (int i = 0; i < max_tokens; ++i) {
            runner.compute_logits(ctx, Task::GEN);
            int next = runner.sample(ctx, 0.7f, 0.9f, 40);
            
            std::string word = tokenizer.decode(next);
            output += word;
            
            runner.forward(next, ctx);
            ctx.current_pos++;
            
            if (ctx.current_pos >= loader->get_header().max_context) break;
        }
        return output;
    }
    // Sentiment Analysis - Multi-Task Head (Hybrid AI)
    std::string sentiment(const std::string& text) {
        BulletContext ctx(loader->get_header().num_layers, 
                          loader->get_header().num_heads, 
                          loader->get_header().hidden_size / loader->get_header().num_heads, 
                          loader->get_header().max_context,
                          loader->get_header().hidden_size,
                          loader->get_header().vocab_size);
        
        // 1. Tokenize input
        std::vector<int> tokens = tokenizer.encode(text);
        if (tokens.empty()) return "NEUTRAL";
        
        // 2. Run Forward Pass (Prefill) through complete Transformer
        for (int token : tokens) {
            runner.forward(token, ctx);
        }
        
        // 3. Extract last hidden state from context
        const float* last_hidden = ctx.hidden_states.data() + 
                                   (ctx.current_pos - 1) * ctx.hidden_size;
        
        // 4. Simple sentiment scoring (demo - full version needs sentiment head weights)
        float sentiment_score = 0.0f;
        for (int i = 0; i < ctx.hidden_size; i++) {
            sentiment_score += last_hidden[i];
        }
        sentiment_score /= ctx.hidden_size;
        
        // 5. Map to sentiment label
        if (sentiment_score > 0.1f) return "POSITIVE";
        if (sentiment_score < -0.1f) return "NEGATIVE";
        return "NEUTRAL";
    }
    
    std::vector<std::pair<std::string, std::string>> ner(const std::string& text) {
         // Placeholder logic for NER
         return {};
    }
    
    std::vector<std::string> pos(const std::string& text) {
        return {};
    }
    
    int classify(const std::string& text) {
        return 0;
    }
};

// ==================================================================================
// 🟢 MAIN
// ==================================================================================

} // namespace core
} // namespace bullet

using namespace bullet::core;

#ifndef BULLET_TEST
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: ./bullet-core <model.bullet>\n";
        return 1;
    }
    
    try {
        BulletModel model(argv[1]);
        
        std::string prompt = (argc > 2) ? argv[2] : "जीवन";
        std::cout << "Prompt: " << prompt << "\n";
        std::cout << "Generating...\n";
        
        std::string result = model.generate(prompt);
        std::cout << "Result: " << result << "\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
#endif
#endif
