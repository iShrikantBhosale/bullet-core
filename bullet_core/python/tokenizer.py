"""
Byte-Pair Encoding (BPE) Tokenizer for Marathi
Implements a simple BPE algorithm to create subword vocabulary
"""

import json
import re
from collections import Counter, defaultdict
import numpy as np

class BPETokenizer:
    def __init__(self, vocab_size=2000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []
        self.char_to_id = {}
        self.id_to_char = {}
        self.cache = {}  # Cache for word tokenization
        
    def train(self, text, verbose=True):
        """Train BPE tokenizer on text corpus"""
        if verbose:
            print(f"Training BPE tokenizer (target vocab: {self.vocab_size})...")
        
        # Step 1: Initialize with character vocabulary
        chars = sorted(list(set(text)))
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(chars)}
        
        if verbose:
            print(f"  Initial character vocab: {len(chars)}")
        
        # Step 2: Tokenize text into characters
        words = text.split()
        word_freqs = Counter(words)
        
        # Represent each word as list of characters
        splits = {word: list(word) for word in word_freqs.keys()}
        
        # Step 3: Iteratively merge most frequent pairs
        num_merges = self.vocab_size - len(chars)
        
        for i in range(num_merges):
            # Count all pairs
            pair_freqs = defaultdict(int)
            for word, freq in word_freqs.items():
                split = splits[word]
                if len(split) == 1:
                    continue
                for j in range(len(split) - 1):
                    pair = (split[j], split[j + 1])
                    pair_freqs[pair] += freq
            
            if not pair_freqs:
                break
            
            # Find most frequent pair
            best_pair = max(pair_freqs, key=pair_freqs.get)
            
            # Merge this pair in all words
            for word in word_freqs:
                split = splits[word]
                if len(split) == 1:
                    continue
                
                new_split = []
                j = 0
                while j < len(split):
                    if j < len(split) - 1 and (split[j], split[j + 1]) == best_pair:
                        new_split.append(split[j] + split[j + 1])
                        j += 2
                    else:
                        new_split.append(split[j])
                        j += 1
                splits[word] = new_split
            
            # Record this merge
            self.merges.append(best_pair)
            
            if verbose and (i + 1) % 100 == 0:
                print(f"  Merges: {i + 1}/{num_merges}")
        
        # Step 4: Build final vocabulary
        self.vocab = set(chars)
        for pair in self.merges:
            self.vocab.add(pair[0] + pair[1])
        
        # Create token to ID mapping
        self.vocab = sorted(list(self.vocab))
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}
        
        if verbose:
            print(f"  Final vocab size: {len(self.vocab)}")
            print(f"  Number of merges: {len(self.merges)}")
        
        return self
    
    def encode(self, text):
        """Encode text to token IDs with caching"""
        # Split into words
        words = text.split()
        
        token_ids = []
        for word in words:
            # Check cache first
            if word in self.cache:
                token_ids.extend(self.cache[word])
                if ' ' in self.token_to_id:
                    token_ids.append(self.token_to_id[' '])
                continue
                
            # Start with character split
            split = list(word)
            
            # Apply merges
            for pair in self.merges:
                if len(split) == 1:
                    break
                
                new_split = []
                j = 0
                while j < len(split):
                    if j < len(split) - 1 and (split[j], split[j + 1]) == pair:
                        new_split.append(split[j] + split[j + 1])
                        j += 2
                    else:
                        new_split.append(split[j])
                        j += 1
                split = new_split
            
            # Convert tokens to IDs
            word_ids = []
            for token in split:
                if token in self.token_to_id:
                    word_ids.append(self.token_to_id[token])
                else:
                    # Fallback to character encoding
                    for char in token:
                        if char in self.token_to_id:
                            word_ids.append(self.token_to_id[char])
            
            # Update cache
            self.cache[word] = word_ids
            
            token_ids.extend(word_ids)
            
            # Add space token if not last word
            if ' ' in self.token_to_id:
                token_ids.append(self.token_to_id[' '])
        
        return token_ids
    
    def decode(self, token_ids):
        """Decode token IDs to text"""
        tokens = [self.id_to_token.get(id, '') for id in token_ids]
        return ''.join(tokens)
    
    def save(self, path):
        """Save tokenizer to file"""
        data = {
            'vocab_size': self.vocab_size,
            'vocab': self.vocab,
            'merges': self.merges,
            'token_to_id': self.token_to_id,
            'id_to_token': {str(k): v for k, v in self.id_to_token.items()}
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved to {path}")
    
    def load(self, path):
        """Load tokenizer from file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab_size = data['vocab_size']
        self.vocab = data['vocab']
        self.merges = [tuple(pair) for pair in data['merges']]
        self.token_to_id = data['token_to_id']
        self.id_to_token = {int(k): v for k, v in data['id_to_token'].items()}
        
        print(f"Tokenizer loaded from {path}")
        print(f"  Vocab size: {len(self.vocab)}")
        return self


def train_marathi_tokenizer(data_path, vocab_size=2000, save_path=None):
    """Train BPE tokenizer on Marathi dataset"""
    print("=" * 70)
    print("TRAINING BPE TOKENIZER FOR MARATHI")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    text_data = ""
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            text_data += item['instruction'] + " " + item['response'] + " "
    
    print(f"  Total characters: {len(text_data):,}")
    print(f"  Total words: {len(text_data.split()):,}")
    
    # Train tokenizer
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(text_data, verbose=True)
    
    # Test encoding/decoding
    print("\n" + "=" * 70)
    print("TESTING TOKENIZER")
    print("=" * 70)
    
    test_texts = [
        "जीवनाचा अर्थ काय आहे?",
        "आत्मा आणि शरीर यांचा संबंध",
        "ध्यान म्हणजे काय?"
    ]
    
    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        
        print(f"\nOriginal:  {text}")
        print(f"Encoded:   {encoded[:20]}... ({len(encoded)} tokens)")
        print(f"Decoded:   {decoded}")
        print(f"Match:     {'✅' if text == decoded else '❌'}")
    
    # Statistics
    print("\n" + "=" * 70)
    print("TOKENIZER STATISTICS")
    print("=" * 70)
    
    sample_text = text_data[:10000]
    char_tokens = len(sample_text)
    bpe_tokens = len(tokenizer.encode(sample_text))
    compression = char_tokens / bpe_tokens if bpe_tokens > 0 else 0
    
    print(f"\nCompression ratio: {compression:.2f}x")
    print(f"  Character-level: {char_tokens} tokens")
    print(f"  BPE-level: {bpe_tokens} tokens")
    
    # Save
    if save_path:
        tokenizer.save(save_path)
    
    return tokenizer


if __name__ == "__main__":
    # Train tokenizer
    tokenizer = train_marathi_tokenizer(
        data_path="/home/shri/Desktop/bulletOs/marathi_philosophy_dataset.jsonl",
        vocab_size=2000,
        save_path="/home/shri/Desktop/bulletOs/bullet_core/marathi_tokenizer.json"
    )
    
    print("\n✅ Tokenizer training complete!")
