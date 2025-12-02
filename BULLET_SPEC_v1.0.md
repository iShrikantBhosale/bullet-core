# 🔵 .bullet v1.0 — The "Tiny Giant" Model Specification

> **Status:** FINAL | **Version:** 1.0 | **Type:** Open Standard

## 👋 What is a `.bullet` file?
Imagine a `.bullet` file as a **smart zip file** for AI. Instead of having messy folders with 10 different files (tokenizer, weights, config, etc.), everything a tiny AI needs to think and speak is packed into **one single, highly optimized file**.

It is designed to be:
- **Tiny**: Perfect for 5MB - 50MB models.
- **Fast**: Your computer can read it instantly without "unpacking" it.
- **Smart**: It knows how to do multiple jobs (chat, summarize, classify) at once.

---

## 🏗️ The Blueprint (File Structure)
A `.bullet` file has three main parts, like a sandwich:

1.  **The Header (The Menu):** Tells the computer what's inside (ingredients, size, recipe).
2.  **The Tokenizer (The Dictionary):** Helps the AI understand words and letters.
3.  **The Weights (The Brain):** The actual neural network that does the thinking.

```mermaid
graph TD
    File[".bullet File"] --> Header[1. HEADER (JSON)]
    File --> Tokenizer[2. TOKENIZER (Binary)]
    File --> Weights[3. WEIGHTS (Binary)]
    
    Header -->|Offset 0| Config[Configuration & Metadata]
    Tokenizer -->|Offset 4096| Vocab[Vocabulary & Tokens]
    Weights -->|Offset 14336| Tensors[Neural Network Layers]
```

---

## 1. 🟦 The Header (Metadata)
**Location:** Start of file (Offset 0)
**Format:** JSON Text

This is the ID card of the model. It's written in plain text (JSON) so you can easily read it.

### Example Header
```json
{
  "bullet_version": "1.0",
  "model_name": "bullet-hybrid-v1",
  "architecture": "bullet-hybrid-transformer",
  "dimensions": {
      "hidden_size": 256,   // How complex the thought process is
      "num_layers": 8,      // How deep the thinking goes
      "num_heads": 4        // How many things it can focus on at once
  },
  "quantization": {
      "type": "BQ4"         // Compression type (4-bit = very small)
  },
  "file_offsets": {
      "tokenizer_start": 4096, // Where the dictionary starts
      "weights_start": 14336   // Where the brain starts
  }
}
```
**Rule:** The header must end with four "null bytes" (`00 00 00 00`) to tell the computer "Stop reading text, binary data coming next!".

---

## 2. 🟩 The Tokenizer (The Dictionary)
**Location:** Defined by `tokenizer_start`
**Format:** Binary (Machine Code)

Computers don't understand words like "Apple"; they understand numbers. The tokenizer converts "Apple" into numbers like `[104, 220]`.

- **Internal & Fast:** Unlike other models, the dictionary is *inside* the file. No missing files, ever.
- **Structure:**
  - Starts with magic word: `BULK`
  - Then the list of all words and their number codes.
  - **Each Token:**
    - `score` (float32): Priority for merging (higher is better).
    - `length` (uint16): How long the word is.
    - `bytes`: The actual word characters.

---

## 3. 🟨 The Weights (The Brain)
**Location:** Defined by `weights_start`
**Format:** Binary (Optimized)

This is the heavy part. It contains millions of numbers (parameters) that make up the AI's intelligence.

- **Magic Prefix:** `BWT0`
- **Organization:** The brain is split into "Tensors" (blocks of numbers).
- **Quantization (Compression):**
  - We use **BQ4 (BulletQuant 4-bit)**.
  - It shrinks numbers down so they take up 4x less space but still work 99% as well.
  - **32 weights** are packed into a tiny **20-byte block**.
  - **Shape:** Fixed 4 dimensions (`uint16[4]`). Unused dimensions are 1.

### Multi-Tasking (Hybrid AI)
A unique feature of `.bullet` is that it can have multiple "Heads". Think of it like a Swiss Army Knife:
- **Gen Head:** For writing stories.
- **NER Head:** For finding names in text.
- **Sentiment Head:** For detecting emotions.

All these heads share the same "body" (layers), making it incredibly efficient.

---

## 4. 🛡️ Safety & Speed Features

### Alignment (Speed)
The file is organized so that data lines up perfectly with your computer's memory (RAM).
- **4KB Alignment:** Large chunks start at 4096-byte intervals. This matches how operating systems manage memory, making loading **instant**.

### Integrity (Safety) [NEW]
To ensure the file isn't broken or hacked, we add a footer.
- **Magic End:** The file ends with `END!` (`45 4E 44 21`).
- **Checksum (Optional):** A SHA-256 hash can be added before the end tag to verify the file is 100% correct.

---

## 🚀 Summary for Developers
If you are building a reader for this file:
1.  **Read the JSON Header** first (stop at `00 00 00 00`).
2.  **Jump** to `tokenizer_start` to load the vocab.
3.  **Jump** to `weights_start` to map the tensors.
4.  **Verify** the `END!` tag exists.
5.  **Run** inference using the BQ4 dequantization method.

**This is the future of Tiny AI.**
