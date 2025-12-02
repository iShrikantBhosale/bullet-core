# Bullet OS 🇮🇳

**The AI Operating System for the Edge**

[![Bullet OS Website](https://img.shields.io/badge/Website-Live-FF9933?style=for-the-badge)](https://iShrikantBhosale.github.io/bullet-core/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

> **Created by Shrikant Bhosale** | **Mentored by [Hintson.com](https://hintson.com)**

Bullet OS is a revolutionary, zero-dependency AI runtime designed to run powerful Transformer models on edge devices with minimal resources.

## 🚀 Features

*   **BQ4 Quantization**: 6.4x compression (4-bit) with near-FP32 accuracy.
*   **Zero-Copy Runtime**: Instant startup (<10ms) via memory mapping.
*   **Hybrid AI**: Single pass generation and sentiment analysis.
*   **Cross-Platform**: Runs on Linux, Android, and Web (WASM).

## 📦 Quick Start

### 1. Download V1.0 Model
Get the production-ready Marathi Philosophy model:
[Download production_v1.bullet](release_v1/production_v1.bullet)

### 2. Run with C++
```bash
g++ -O3 -o bullet bullet-core.cpp
./bullet production_v1.bullet -p "Life is"
```

### 3. Run on Web
Open `docs/index.html` or visit our [Website](https://bullet-os.github.io/bullet-core/).

## 🛠️ Training Dashboard

Train your own BQ4 models using our visual dashboard:

1.  Run `./install_dashboard.sh`
2.  Start with `./dashboard/start_dashboard.sh`
3.  Open `http://localhost:8000`

## 📂 Repository Structure

*   `bullet-core.cpp`: The core runtime engine.
*   `bq4/`: BQ4 quantization headers.
*   `dashboard/`: Training dashboard source.
*   `mobile/`: Android SDK.
*   `wasm_build/`: WebAssembly artifacts.
*   `docs/`: Website source.

## 🇮🇳 Made in India
Proudly developed to democratize AI access for everyone.

---
© 2025 Bullet OS. MIT License.
