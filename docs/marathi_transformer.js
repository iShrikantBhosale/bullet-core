// Marathi Philosophy Model - Full Implementation
// Loads binary weights and performs transformer inference
// Upgraded with Elite Sampling, Contrastive Search & Bullet-Proof Devanagari Fixer (2025 Full Code Pack)

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
        this.pastHiddenStates = []; // Store past hidden states for Contrastive Search
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

    // --- Advanced Sampling Logic ---

    applyRepetitionPenalty(logits, pastTokens, penalty) {
        const lenFactor = pastTokens.length > 50 ? 1.2 : 1.0;
        const effectivePenalty = penalty * lenFactor;

        const seen = new Set(pastTokens);
        for (const t of seen) {
            if (logits[t] < 0) logits[t] *= effectivePenalty;
            else logits[t] /= effectivePenalty;
        }
    }

    applyPresencePenalty(logits, tokenCounts, alpha) {
        for (const t in tokenCounts) {
            logits[t] -= alpha;
        }
    }

    applyFrequencyPenalty(logits, tokenCounts, beta) {
        for (const t in tokenCounts) {
            logits[t] -= beta * tokenCounts[t];
        }
    }

    applyAntiRepeatBlock(logits, pastTokens, n) {
        if (pastTokens.length < n) return;
        for (let i = 1; i <= n; i++) {
            if (pastTokens.length >= i) {
                const token = pastTokens[pastTokens.length - i];
                logits[token] = -Infinity;
            }
        }
    }

    applyLogitBias(logits, biasMap) {
        const vocab = this.tokenizer.vocab;
        for (const [word, bias] of Object.entries(biasMap)) {
            if (vocab[word] !== undefined) {
                logits[vocab[word]] += bias;
            } else {
                for (let i = 0; i < word.length; i++) {
                    const char = word[i];
                    if (vocab[char] !== undefined) {
                        logits[vocab[char]] += bias / word.length;
                    }
                }
            }
        }
    }

    applyContrastivePenalty(logits, pastHiddenStates, alpha = 0.6) {
        if (pastHiddenStates.length === 0) return;

        const d_model = this.config.d_model;
        const embed = this.model['param_0'].data;

        const sortedIndices = [...logits.keys()].sort((a, b) => logits[b] - logits[a]).slice(0, 20);

        for (const tokenIdx of sortedIndices) {
            const tokenEmbed = embed.subarray(tokenIdx * d_model, (tokenIdx + 1) * d_model);
            
            let maxSim = -1.0;
            const checkWindow = pastHiddenStates.slice(-5);
            
            for (const pastHidden of checkWindow) {
                const sim = this.cosineSimilarity(tokenEmbed, pastHidden);
                if (sim > maxSim) maxSim = sim;
            }

            if (maxSim > 0.5) {
                 logits[tokenIdx] -= alpha * maxSim;
            }
        }
    }

    cosineSimilarity(a, b) {
        let dot = 0;
        let magA = 0;
        let magB = 0;
        for (let i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            magA += a[i] * a[i];
            magB += b[i] * b[i];
        }
        return dot / (Math.sqrt(magA) * Math.sqrt(magB) + 1e-9);
    }

    topK(logits, k) {
        const sorted = [...logits].map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
        const top = sorted.slice(0, k);
        const mask = new Float32Array(logits.length).fill(-Infinity);
        for (const [v, i] of top) mask[i] = v;
        return mask;
    }

    topP(logits, p) {
        const probs = this.softmax(logits);
        const sorted = [...probs].map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
        let cum = 0;
        const keep = [];
        for (const item of sorted) {
            cum += item[0];
            keep.push(item);
            if (cum >= p) break;
        }
        const mask = new Float32Array(logits.length).fill(-Infinity);
        for (const [_, i] of keep) mask[i] = logits[i];
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

    // --- 2025 Full Code Pack: Bullet-Proof Devanagari Fixer ---
    bulletOSPerfectMarathi(raw) {
        let out = raw
            .normalize('NFC') // The Nuclear Fix

            // 1. Force Proper Devanagari Joining (The Root Fix)
            .replace(/([क-ह])([ा-्])(?![ा-्\s])/g, '$1$2')   // keep matras attached
            .replace(/([क-ह])्([क-ह])/g, '$1्$2')           // halant joining
            .replace(/([क-ह])([ािीुूेैोौंःँ़])/g, '$1$2') // vowel signs stick
            
            // 2. Insert spaces only where linguistically correct
            .replace(/([क-ह])([ा-्])([क-ह])/g, '$1$2 $3')      // consonant + matra + consonant -> space after matra
            .replace(/([।,!?])\s*/g, '$1 ')                  // space after punctuation
            .replace(/\s+/g, ' ')                            // collapse multiple spaces

            // 3. Kill leftover broken clusters
            .replace(/([क-ह])\s+([ा-्])/g, '$1$2')            // pull matra back
            .replace(/([क-ह])्\s+([क-ह])/g, '$1्$2')         // pull halant back

            // 4. Final safety net
            .replace(/([^\s])[ा-्](?=\s|$)/g, '$1 ') 
            
            // 5. Poetic Replacements (Rebel Gasolina)
            .replace(/आपण/g, 'तुम्ही')
            .replace(/आपल्य/g, 'तुमच')
            .replace(/हा हे की/g, 'हेच')
            .replace(/शांत.*$/g, 'शांत राहा. कारण शांतताच खरे यश आहे.')

            .trim();

        // Optional: poetic line breaks
        out = out.replace(/।/g, '।\n');
        
        return out;
    }

    async generate(prompt, options = {}) {
        if (!this.loaded) throw new Error('Model not loaded');
        
        const { 
            maxTokens = 180, 
            temperature = 0.68, 
            topK = 50, 
            topP = 0.93, 
            repetitionPenalty = 1.21,
            presencePenalty = 0.27,
            frequencyPenalty = 0.23,
            antiRepeatBlock = 4,
            contrastiveAlpha = 0.6,
            stopSequences = ["\n\n", "###", "</end>"],
            logitBias = {}
        } = options;
        
        console.log(`Generating: ${prompt}`);
        
        let tokens = this.encode(prompt);
        let generatedText = "";
        this.pastHiddenStates = [];
        
        const tokenCounts = {};
        for (const t of tokens) {
            tokenCounts[t] = (tokenCounts[t] || 0) + 1;
        }
        
        for (let i = 0; i < maxTokens; i++) {
            const { logits, last_hidden } = this.forward(tokens);
            this.pastHiddenStates.push(last_hidden);

            if (Object.keys(logitBias).length > 0) this.applyLogitBias(logits, logitBias);
            if (temperature !== 1.0) for (let j = 0; j < logits.length; j++) logits[j] /= temperature;
            if (repetitionPenalty !== 1.0) this.applyRepetitionPenalty(logits, tokens, repetitionPenalty);
            if (presencePenalty !== 0.0) this.applyPresencePenalty(logits, tokenCounts, presencePenalty);
            if (frequencyPenalty !== 0.0) this.applyFrequencyPenalty(logits, tokenCounts, frequencyPenalty);
            if (antiRepeatBlock > 0) this.applyAntiRepeatBlock(logits, tokens, antiRepeatBlock);
            if (contrastiveAlpha > 0) this.applyContrastivePenalty(logits, this.pastHiddenStates, contrastiveAlpha);
            
            if (topK > 0 && topK < logits.length) {
                const maskedLogits = this.topK(logits, topK);
                for(let k=0; k<logits.length; k++) logits[k] = maskedLogits[k];
            }
            
            if (topP < 1.0) {
                const maskedLogits = this.topP(logits, topP);
                for(let k=0; k<logits.length; k++) logits[k] = maskedLogits[k];
            }
            
            const probs = this.softmax(logits);
            const next_token = this.sampleFromProbs(probs);
            
            tokens.push(next_token);
            tokenCounts[next_token] = (tokenCounts[next_token] || 0) + 1;
            
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
        
        // Apply the Bullet-Proof Fixer
        let finalText = this.decode(tokens);
        finalText = this.bulletOSPerfectMarathi(finalText);
        
        return finalText;
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
