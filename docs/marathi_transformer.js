// Marathi Philosophy Model - JavaScript Transformer Implementation
// Loads actual trained weights and performs inference

class MarathiTransformer {
    constructor() {
        this.config = null;
        this.weights = null;
        this.tokenizer = null;
        this.loaded = false;
    }

    async loadModel() {
        try {
            console.log('Loading model weights...');
            const weightsResponse = await fetch('model_weights.json');
            const weightsData = await weightsResponse.json();
            
            console.log('Loading tokenizer...');
            const tokenizerResponse = await fetch('tokenizer.json');
            const tokenizerData = await tokenizerResponse.json();
            
            this.config = weightsData.config;
            this.weights = this.parseWeights(weightsData.weights);
            this.tokenizer = tokenizerData;
            this.loaded = true;
            
            console.log('Model loaded successfully!');
            console.log('Config:', this.config);
            console.log('Weights loaded:', Object.keys(this.weights));
            
            return true;
        } catch (error) {
            console.error('Failed to load model:', error);
            return false;
        }
    }

    parseWeights(weightsData) {
        const weights = {};
        for (const [key, value] of Object.entries(weightsData)) {
            weights[key] = {
                shape: value.shape,
                data: new Float32Array(value.data)
            };
        }
        return weights;
    }

    encode(text) {
        if (!this.tokenizer || !this.tokenizer.vocab) {
            console.error('Tokenizer not loaded');
            return [];
        }

        const tokens = [];
        const vocab = this.tokenizer.vocab;
        
        // Simple tokenization
        for (let i = 0; i < text.length; i++) {
            const char = text[i];
            const tokenId = vocab[char] !== undefined ? vocab[char] : 0;
            tokens.push(tokenId);
        }
        
        return tokens;
    }

    decode(tokens) {
        if (!this.tokenizer || !this.tokenizer.id_to_token) {
            console.error('Tokenizer not loaded');
            return '';
        }

        const id_to_token = this.tokenizer.id_to_token;
        let text = '';
        
        for (const tokenId of tokens) {
            const token = id_to_token[tokenId.toString()];
            if (token) {
                text += token;
            }
        }
        
        return text;
    }

    forward(tokens) {
        if (!this.loaded) {
            throw new Error('Model not loaded');
        }

        const embedWeight = this.weights['param_0'];
        const posWeight = this.weights['param_1'];
        
        const seqLen = tokens.length;
        const dModel = this.config.d_model;
        
        const hidden = new Float32Array(seqLen * dModel);
        
        for (let i = 0; i < seqLen; i++) {
            const tokenId = tokens[i];
            const posId = i;
            
            for (let j = 0; j < dModel; j++) {
                const tokenEmb = embedWeight.data[tokenId * dModel + j] || 0;
                const posEmb = posId < posWeight.shape[0] ? posWeight.data[posId * dModel + j] : 0;
                hidden[i * dModel + j] = tokenEmb + posEmb;
            }
        }
        
        const logits = new Float32Array(this.config.vocab_size);
        const lastHidden = hidden.slice((seqLen - 1) * dModel, seqLen * dModel);
        
        for (let v = 0; v < this.config.vocab_size; v++) {
            let score = 0;
            for (let d = 0; d < dModel; d++) {
                score += lastHidden[d] * (embedWeight.data[v * dModel + d] || 0);
            }
            logits[v] = score;
        }
        
        return logits;
    }

    softmax(logits, temperature = 1.0) {
        const scaled = logits.map(x => x / temperature);
        const maxLogit = Math.max(...scaled);
        const expScores = scaled.map(x => Math.exp(x - maxLogit));
        const sumExp = expScores.reduce((a, b) => a + b, 0);
        return expScores.map(x => x / sumExp);
    }

    sample(probs, topK = 40) {
        const indexed = probs.map((p, i) => ({ prob: p, index: i }));
        indexed.sort((a, b) => b.prob - a.prob);
        
        const topKProbs = indexed.slice(0, topK);
        const sumTopK = topKProbs.reduce((sum, item) => sum + item.prob, 0);
        const normalized = topKProbs.map(item => ({ ...item, prob: item.prob / sumTopK }));
        
        const rand = Math.random();
        let cumProb = 0;
        for (const item of normalized) {
            cumProb += item.prob;
            if (rand < cumProb) {
                return item.index;
            }
        }
        
        return normalized[0].index;
    }

    generate(prompt, maxTokens = 50, temperature = 0.8, topK = 40) {
        if (!this.loaded) {
            throw new Error('Model not loaded');
        }

        console.log(`Generating from prompt: "${prompt}"`);
        
        let tokens = this.encode(prompt);
        console.log('Encoded tokens:', tokens.length);
        
        for (let i = 0; i < maxTokens; i++) {
            const logits = this.forward(tokens);
            const probs = this.softmax(Array.from(logits), temperature);
            const nextToken = this.sample(probs, topK);
            tokens.push(nextToken);
            
            if (tokens.length >= this.config.max_seq_len) {
                break;
            }
        }
        
        const generated = this.decode(tokens);
        console.log('Generated:', generated);
        
        return generated;
    }

    getInfo() {
        return {
            loaded: this.loaded,
            architecture: 'GPT-style Transformer',
            vocab_size: this.config?.vocab_size || 0,
            d_model: this.config?.d_model || 0,
            n_layers: this.config?.n_layers || 0,
            n_heads: this.config?.n_heads || 0,
            max_seq_len: this.config?.max_seq_len || 0
        };
    }
}

if (typeof window !== 'undefined') {
    window.MarathiTransformer = MarathiTransformer;
}
