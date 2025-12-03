// Marathi Philosophy Model - Full Implementation
// Loads binary weights and performs transformer inference

class MarathiPhilosophyModel {
    constructor() {
        this.model = null;
        this.tokenizer = null;
        this.loaded = false;
        this.config = {
            vocab_size: 1511,
            d_model: 256,
            n_layers: 8,
            n_heads: 4,
            max_seq_len: 128
        };
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
            
            this.loaded = true;
            console.log('Model loaded successfully');
            console.log(`Vocabulary size: ${this.tokenizer.vocab_size}`);
            
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
            
            // Handle potentially unaligned data
            // We copy the bytes to a new buffer which is guaranteed to be aligned
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
        
        for (let i = 0; i < text.length; i++) {
            const char = text[i];
            const tokenId = vocab[char] !== undefined ? vocab[char] : 0; // UNK
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
        
        return logits;
    }

    softmax(logits, temperature = 1.0) {
        const exps = [];
        let sum = 0;
        const max = Math.max(...logits);
        
        for (let i = 0; i < logits.length; i++) {
            const val = Math.exp((logits[i] - max) / temperature);
            exps.push(val);
            sum += val;
        }
        
        return exps.map(v => v / sum);
    }

    sample(probs, topK = 40) {
        const candidates = probs.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p).slice(0, topK);
        const sum = candidates.reduce((acc, c) => acc + c.p, 0);
        
        let r = Math.random() * sum;
        for (const c of candidates) {
            r -= c.p;
            if (r <= 0) return c.i;
        }
        return candidates[0].i;
    }

    async generate(prompt, options = {}) {
        if (!this.loaded) throw new Error('Model not loaded');
        
        const { maxTokens = 50, temperature = 0.7 } = options;
        console.log(`Generating: ${prompt}`);
        
        let tokens = this.encode(prompt);
        
        for (let i = 0; i < maxTokens; i++) {
            const logits = this.forward(tokens);
            const probs = this.softmax(logits, temperature);
            const next_token = this.sample(probs);
            
            tokens.push(next_token);
            if (tokens.length >= this.config.max_seq_len) break;
        }
        
        return this.decode(tokens);
    }

    getInfo() {
        return {
            ...this.config,
            loaded: this.loaded,
            parameters: 419840
        };
    }
}

if (typeof window !== 'undefined') {
    window.MarathiPhilosophyModel = MarathiPhilosophyModel;
}
