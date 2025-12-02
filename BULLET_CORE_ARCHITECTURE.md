# 🏗️ BULLET-CORE: Reference Implementation Architecture

> **Target System:** 1GB RAM, No GPU, Mobile CPU
> **Goal:** Zero-dependency, single-header-style C++17 runtime for `.bullet` files.

---

## 1. 🧩 High-Level Architecture

The architecture is designed as a **read-only memory-mapped pipeline**. We do not load the model into RAM; we map the file and point to data.

```mermaid
graph TD
    UserApp[User Application] --> API[Bullet Inference API]
    
    subgraph "Bullet Core Runtime"
        API --> Loader[Model Loader (mmap)]
        API --> Runner[Inference Runner]
        
        Loader --> Header[Header Parser (JSON)]
        Loader --> Tokenizer[Tokenizer (Binary)]
        Loader --> Weights[Weight Map (Pointers)]
        
        Runner --> Context[Context / KV Cache]
        Runner --> Transformer[Transformer Block]
        Runner --> Heads[Multi-Task Heads]
        
        Transformer --> Attn[Self-Attention]
        Transformer --> FFN[Feed Forward]
        
        Weights -.-> Attn
        Weights -.-> FFN
        Weights -.-> Heads
    end
    
    File[(".bullet File (Disk)")] -.-> Loader
```

---

## 2. 🏛️ C++ Class Structure

We use a **flat class hierarchy** to minimize vtable overhead.

### Core Classes

```cpp
// 1. The Main Interface
class BulletModel {
public:
    BulletModel(const std::string& path);
    ~BulletModel();

    // Main Inference
    std::string generate(const std::string& prompt, int max_tokens);
    
    // Task-Specific Heads
    std::vector<std::string> ner(const std::string& text);
    float sentiment(const std::string& text);

private:
    BulletLoader loader;
    BulletTokenizer tokenizer;
    BulletRunner runner;
};

// 2. The File Loader (Memory Mapping)
class BulletLoader {
public:
    void load(const std::string& path);
    const BulletHeader& get_header() const;
    const uint8_t* get_weights_ptr() const;
    
private:
    void* mmap_ptr;
    size_t file_size;
    BulletHeader header;
    std::unordered_map<uint64_t, TensorView> tensor_map; // Hash -> Pointer
};

// 3. The Tokenizer
class BulletTokenizer {
public:
    std::vector<int> encode(const std::string& text);
    std::string decode(int token_id);
    
private:
    std::vector<std::string> vocab;
    std::unordered_map<std::string, int> token_to_id;
};

// 4. The Inference Engine
class BulletRunner {
public:
    void forward(const std::vector<int>& input_ids, BulletContext& ctx);
    int sample_next_token();
    
private:
    void attention_layer(int layer_idx, BulletContext& ctx);
    void ffn_layer(int layer_idx, BulletContext& ctx);
    void rms_norm(float* x, const TensorView& weight);
};

// 5. Context / State (KV Cache)
struct BulletContext {
    std::vector<float> k_cache; // [layers, seq_len, heads, head_dim]
    std::vector<float> v_cache;
    std::vector<float> current_logits;
    int current_pos;
};
```

---

## 3. 📂 File Parsing Workflow

The loader follows the spec strictly:

1.  **Open File:** `mmap()` the entire `.bullet` file (read-only).
2.  **Parse Header:**
    *   Read from offset 0 until `00 00 00 00`.
    *   Parse JSON to `BulletHeader` struct.
    *   Validate `magic` and `version`.
3.  **Load Tokenizer:**
    *   Jump to `header.file_offsets.tokenizer_start`.
    *   Verify `BULK` magic.
    *   Read `vocab_size`.
    *   Loop `vocab_size` times to read `(len, bytes)` and build `std::vector<string>`.
4.  **Map Weights:**
    *   Jump to `header.file_offsets.weights_start`.
    *   Verify `BWT0` magic.
    *   Read `num_tensors`.
    *   Loop `num_tensors` times:
        *   Read `hash`, `rank`, `shape`, `quant_type`.
        *   Store a `TensorView` pointing to the data block in the mmap.
        *   **Crucial:** Do NOT copy data. Just store pointers.
5.  **Verify Footer:**
    *   Check last 4 bytes for `END!`.

---

## 4. 📉 BQ4 Loading & Dequantization

**BQ4 (BulletQuant 4-bit)** is designed for SIMD.

### Data Layout (20 bytes per block)
*   `scale` (float16, 2 bytes)
*   `zero` (int8, 1 byte)
*   `padding` (1 byte, for alignment if needed, usually packed tight)
*   `data` (16 bytes = 32 x 4-bit nibbles)

### Dequantization Kernel (SIMD-friendly)

```cpp
// Dequantize one block of 32 weights
void dequant_block_bq4(const uint8_t* block_ptr, float* out_ptr) {
    // 1. Read metadata
    float16_t scale = *reinterpret_cast<const float16_t*>(block_ptr);
    int8_t zero = static_cast<int8_t>(block_ptr[2]);
    const uint8_t* nibbles = block_ptr + 3; // Start of packed data (assuming tight packing)

    // 2. Unpack and scale
    for (int i = 0; i < 16; ++i) {
        uint8_t byte = nibbles[i];
        
        // Lower nibble
        int8_t val1 = (byte & 0x0F);
        out_ptr[2*i] = (val1 - zero) * scale;
        
        // Upper nibble
        int8_t val2 = (byte >> 4);
        out_ptr[2*i + 1] = (val2 - zero) * scale;
    }
}
```

**Optimization:** In the forward pass, we often dequantize *on the fly* into a small L1 cache buffer to perform matrix multiplication, rather than dequantizing the whole tensor.

---

## 5. 🧠 Transformer Forward Pass

Standard LLaMA-like architecture loop:

```cpp
void BulletRunner::forward(const std::vector<int>& tokens, BulletContext& ctx) {
    // 1. Embedding
    float* x = ctx.scratch_buffer;
    lookup_embedding(tokens.back(), x); // Only process last token for generation

    for (int l = 0; l < model.layers; ++l) {
        float* residual = x;
        
        // 2. RMS Norm
        rms_norm(x, get_tensor(l, "norm.attn"));
        
        // 3. Attention (Q, K, V)
        // Dequantize weights on fly -> MatMul
        matmul(x, get_tensor(l, "attn.q"), q_buf);
        matmul(x, get_tensor(l, "attn.k"), k_buf);
        matmul(x, get_tensor(l, "attn.v"), v_buf);
        
        // 4. RoPE (Rotary Positional Embeddings)
        apply_rope(q_buf, k_buf, ctx.current_pos);
        
        // 5. Update KV Cache
        update_cache(l, k_buf, v_buf, ctx);
        
        // 6. Scaled Dot Product Attention
        attention_score(q_buf, ctx.k_cache[l], scores);
        softmax(scores);
        weighted_sum(scores, ctx.v_cache[l], attn_out);
        
        // 7. Output Projection
        matmul(attn_out, get_tensor(l, "attn.out"), x);
        
        // 8. Residual Add
        add(x, residual);
        
        // --- FFN Block ---
        residual = x;
        rms_norm(x, get_tensor(l, "norm.ffn"));
        
        // SwiGLU: (W1(x) * Sigmoid(W1(x))) * W2(x) ... simplified
        matmul(x, get_tensor(l, "ffn.w1"), w1_out);
        matmul(x, get_tensor(l, "ffn.w3"), w3_out); // Gate
        silu(w1_out);
        multiply(w1_out, w3_out);
        matmul(w1_out, get_tensor(l, "ffn.w2"), x);
        
        add(x, residual);
    }
    
    // 9. Final Norm
    rms_norm(x, get_tensor("final_norm"));
    
    // 10. To Logits (Head Selection happens here)
    // See Section 6
}
```

---

## 6. 🔀 Multi-Task Head Switching

We use an `enum` to select which final linear layer to use.

```cpp
enum class Task { GEN, NER, POS, SENTIMENT, CLS };

// In BulletRunner
void compute_logits(float* hidden_state, Task task) {
    std::string head_name;
    switch(task) {
        case Task::GEN: head_name = "gen_head"; break;
        case Task::NER: head_name = "ner_head"; break;
        case Task::SENTIMENT: head_name = "sentiment_head"; break;
        // ...
    }
    
    TensorView w = get_tensor(head_name + ".weight");
    // Optional bias
    TensorView b = get_tensor(head_name + ".bias"); 
    
    matmul(hidden_state, w, logits_out);
    if (b.valid) add(logits_out, b);
}
```

This allows the same backbone to drive multiple outputs.

---

## 7. 💾 Memory Optimization Plan

1.  **mmap() Everything:**
    *   OS manages paging. We never `malloc()` for weights.
    *   If system RAM is low, OS swaps out unused pages automatically.

2.  **Arena Allocator for Inference:**
    *   Allocate one big `scratch_buffer` (e.g., 2MB) at startup.
    *   All intermediate activations (Q, K, V, FFN outputs) live here.
    *   Zero `malloc` calls during generation loop.

3.  **Static KV Cache:**
    *   Pre-allocate KV cache for `max_context` (e.g., 2048).
    *   Size = `layers * 2 * context * dim * sizeof(float16)`.
    *   Use `float16` for KV cache to halve RAM usage.

4.  **Alignment:**
    *   Ensure `mmap` pointer is aligned.
    *   Use `__builtin_assume_aligned` for SIMD auto-vectorization.

---

## 8. 🔌 Minimal Inference API

```cpp
int main() {
    // 1. Initialize (Zero copy load)
    BulletModel model("models/tiny-v1.bullet");
    
    // 2. Task: Chat
    std::cout << "User: Hello!\n";
    std::string reply = model.generate("Hello!", 50);
    std::cout << "AI: " << reply << "\n";
    
    // 3. Task: Sentiment Analysis
    float score = model.sentiment("I love this OS!");
    std::cout << "Sentiment: " << (score > 0.5 ? "Positive" : "Negative") << "\n";
    
    return 0;
}
```

---

## 9. 📜 Pseudocode Pipeline

```text
FUNCTION RunInference(prompt, task):
    ctx = InitializeContext()
    tokens = Tokenizer.Encode(prompt)
    
    # Prefill Phase
    FOR token IN tokens:
        ForwardTransformer(token, ctx)
    
    # Generation Phase
    WHILE len(tokens) < max_len:
        last_hidden = ctx.last_hidden_state
        
        # Switch Head
        logits = Head(task).Forward(last_hidden)
        
        # Sample
        next_token = Sample(logits)
        tokens.append(next_token)
        
        if next_token == EOS: BREAK
        
        # Feed back
        ForwardTransformer(next_token, ctx)
        
    RETURN Tokenizer.Decode(tokens)
```
