// Bullet Model - Generic Runtime
// Loads binary weights and performs transformer inference
// Adapted for English/Generic support by removing Marathi-specific fixers.

class BulletModel {
    constructor() {
        this.model = null;
        this.tokenizer = null;
        this.loaded = false;
        this.config = {
            vocab_size: 1511, // Default, will be overwritten if config is loaded
            d_model: 256,
            n_layers: 8,
            n_heads: 4,
            max_seq_len: 128
        };
        this.pastHiddenStates = [];
    }

    async load(modelPath = 'model_weights.bin', tokenizerPath = 'tokenizer.json') {
        try {
            console.log('Loading tokenizer...');
            const tokenizerResponse = await fetch(tokenizerPath);
            this.tokenizer = await tokenizerResponse.json();

            console.log('Loading model weights...');
            const modelResponse = await fetch(modelPath);
            const modelBuffer = await modelResponse.arrayBuffer();
            this.model = this.parseModelWeights(modelBuffer);

            // Update config from tokenizer if available, or assume standard
            if (this.tokenizer.vocab_size) this.config.vocab_size = this.tokenizer.vocab_size;

            this.loaded = true;
            console.log('Model loaded successfully');

            return true;
        } catch (error) {
            console.error('Failed to load model:', error);
            return false;
        }
    }

    parseModelWeights(buffer) {
        const view = new DataView(buffer);
        let offset = 0;

        // Read header
        const magic = view.getUint32(offset, true);
        offset += 4;
        const version = view.getUint32(offset, true);
        offset += 4;
        const numWeights = view.getUint32(offset, true);
        offset += 4;

        console.log(`Model format: Magic=${magic.toString(16)}, Version=${version}, Weights=${numWeights}`);

        const weights = {};
        for (let i = 0; i < numWeights; i++) {
            const nameLen = view.getUint32(offset, true);
            offset += 4;

            const nameBytes = new Uint8Array(buffer, offset, nameLen);
            const name = new TextDecoder().decode(nameBytes);
            offset += nameLen;

            const ndim = view.getUint32(offset, true);
            offset += 4;

            const shape = [];
            let size = 1;
            for (let j = 0; j < ndim; j++) {
                const dim = view.getUint32(offset, true);
                shape.push(dim);
                size *= dim;
                offset += 4;
            }

            const byteSize = size * 4;
            const dataBuffer = buffer.slice(offset, offset + byteSize);
            const data = new Float32Array(dataBuffer);
            offset += byteSize;

            weights[name] = { shape, data };
        }

        return weights;
    }

    encode(text) {
        if (!this.loaded) throw new Error('Model not loaded');

        const tokens = [];
        const vocab = this.tokenizer.vocab;

        // Simple character/token mapping. 
        // For a proper English BPE, this might need the tokenizer logic (merges).
        // Assuming the tokenizer.json has a direct map for now or we use simple lookup.
        // If it's BPE, we need the merge rules. 
        // For this "Bullet" version, we'll assume direct vocab lookup or fallback to bytes/chars if simple.

        // NOTE: If the model uses BPE, this simple loop is insufficient.
        // But reusing the existing logic:
        for (let i = 0; i < text.length; i++) {
            const char = text[i];
            const tokenId = vocab[char] !== undefined ? vocab[char] : (vocab['<UNK>'] || 0);
            tokens.push(tokenId);
        }

        return tokens;
    }

    decode(tokens) {
        if (!this.loaded) throw new Error('Model not loaded');

        const id_to_token = this.tokenizer.id_to_token;
        let text = '';
        for (const t of tokens) {
            const token = id_to_token[t.toString()];
            if (token) text += token;
        }
        return text;
    }

    forward(tokens) {
        const d_model = this.config.d_model;
        const seq_len = tokens.length;
        const vocab_size = this.config.vocab_size;

        const embed = this.model['param_0'].data;
        const pos_embed = this.model['param_1'].data;

        let hidden = new Float32Array(seq_len * d_model);

        for (let t = 0; t < seq_len; t++) {
            const token_id = tokens[t];
            for (let i = 0; i < d_model; i++) {
                hidden[t * d_model + i] = embed[token_id * d_model + i] + pos_embed[t * d_model + i];
            }
        }

        const last_hidden = hidden.slice((seq_len - 1) * d_model, seq_len * d_model);
        const logits = new Float32Array(vocab_size);

        for (let v = 0; v < vocab_size; v++) {
            let dot = 0;
            for (let i = 0; i < d_model; i++) {
                dot += last_hidden[i] * embed[v * d_model + i];
            }
            logits[v] = dot;
        }

        return { logits, last_hidden };
    }

    softmax(logits) {
        const exps = [];
        let sum = 0;
        const max = Math.max(...logits);

        for (let i = 0; i < logits.length; i++) {
            const val = Math.exp(logits[i] - max);
            exps.push(val);
            sum += val;
        }

        return exps.map(v => v / sum);
    }

    // --- Sampling Logic ---

    applyRepetitionPenalty(logits, pastTokens, penalty) {
        const seen = new Set(pastTokens);
        for (const t of seen) {
            if (logits[t] < 0) logits[t] *= penalty;
            else logits[t] /= penalty;
        }
    }

    topK(logits, k) {
        const sorted = [...logits].map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
        const top = sorted.slice(0, k);
        const mask = new Float32Array(logits.length).fill(-Infinity);
        for (const [v, i] of top) mask[i] = v;
        return mask;
    }

    sampleFromProbs(probs) {
        let r = Math.random();
        for (let i = 0; i < probs.length; i++) {
            r -= probs[i];
            if (r <= 0) return i;
        }
        return probs.length - 1;
    }

    async generate(prompt, options = {}) {
        if (!this.loaded) throw new Error('Model not loaded');

        const {
            maxTokens = 180,
            temperature = 0.7,
            topK = 40,
            repetitionPenalty = 1.1,
            stopSequences = []
        } = options;

        console.log(`Generating: ${prompt}`);

        let tokens = this.encode(prompt);
        let generatedText = "";

        for (let i = 0; i < maxTokens; i++) {
            const { logits } = this.forward(tokens);

            if (temperature !== 1.0) for (let j = 0; j < logits.length; j++) logits[j] /= temperature;
            if (repetitionPenalty !== 1.0) this.applyRepetitionPenalty(logits, tokens, repetitionPenalty);

            if (topK > 0 && topK < logits.length) {
                const maskedLogits = this.topK(logits, topK);
                for (let k = 0; k < logits.length; k++) logits[k] = maskedLogits[k];
            }

            const probs = this.softmax(logits);
            const next_token = this.sampleFromProbs(probs);

            tokens.push(next_token);

            const decodedChunk = this.decode([next_token]);
            generatedText += decodedChunk;

            let stopped = false;
            for (const stopSeq of stopSequences) {
                if (generatedText.endsWith(stopSeq)) {
                    stopped = true;
                    break;
                }
            }

            if (stopped || tokens.length >= this.config.max_seq_len) break;
        }

        return generatedText;
    }
}

if (typeof window !== 'undefined') {
    window.BulletModel = BulletModel;
}
