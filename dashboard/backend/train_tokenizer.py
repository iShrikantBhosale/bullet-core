#!/usr/bin/env python3
"""
SentencePiece Tokenizer Training Script for Bullet OS
Trains a BPE tokenizer on Marathi+English mixed corpus
"""

import sentencepiece as spm
import json
import os
import sys
from pathlib import Path

def extract_text_from_jsonl(jsonl_path):
    """Extract all text from JSONL dataset"""
    texts = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # Extract instruction, input, and response
                instruction = data.get('instruction', '').strip()
                input_text = data.get('input', '').strip()
                response = data.get('response', '').strip()
                
                if instruction:
                    texts.append(instruction)
                if input_text:
                    texts.append(input_text)
                if response:
                    texts.append(response)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON at line {line_num}: {e}")
                continue
    
    return texts

def train_sentencepiece_tokenizer(
    data_path,
    vocab_size=4000,
    model_prefix='bullet_tokenizer',
    model_type='bpe',
    character_coverage=0.9995,
    output_dir='.'
):
    """
    Train SentencePiece tokenizer
    
    Args:
        data_path: Path to training data (.jsonl or .txt)
        vocab_size: Target vocabulary size
        model_prefix: Prefix for output model files
        model_type: 'bpe' or 'unigram'
        character_coverage: Character coverage (0.9995 for mixed scripts)
        output_dir: Directory to save model files
    """
    
    print(f"🔵 Training SentencePiece Tokenizer")
    print(f"   Data: {data_path}")
    print(f"   Vocab Size: {vocab_size}")
    print(f"   Model Type: {model_type}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract text from JSONL
    if data_path.endswith('.jsonl'):
        print("📄 Extracting text from JSONL...")
        texts = extract_text_from_jsonl(data_path)
        print(f"   Extracted {len(texts)} text segments")
        
        # Write to temporary corpus file
        corpus_path = os.path.join(output_dir, 'temp_corpus.txt')
        with open(corpus_path, 'w', encoding='utf-8') as f:
            for text in texts:
                if text:  # Skip empty strings
                    f.write(text + '\n')
        print(f"   Wrote corpus to {corpus_path}")
    else:
        # Use text file directly
        corpus_path = data_path
    
    # Train SentencePiece model
    model_path = os.path.join(output_dir, model_prefix)
    
    print(f"🚀 Training tokenizer...")
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_path,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        # Special tokens
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        # Additional settings
        split_by_whitespace=True,
        split_by_unicode_script=True,
        byte_fallback=True,  # Handle unknown characters
        normalization_rule_name='nmt_nfkc_cf',  # Normalize text
        remove_extra_whitespaces=True,
        max_sentence_length=4192,
        num_threads=os.cpu_count(),
        train_extremely_large_corpus=False,
        # Control vocabulary
        user_defined_symbols=[],
        control_symbols=[],
    )
    
    print(f"✅ Tokenizer trained successfully!")
    print(f"   Model: {model_path}.model")
    print(f"   Vocab: {model_path}.vocab")
    
    # Test the tokenizer
    print(f"\n🧪 Testing tokenizer...")
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_path}.model")
    
    test_sentences = [
        "How to control anger?",
        "रागावर नियंत्रण कसे मिळवावे?",
        "What is the meaning of Karma?",
        "कर्माचा सिद्धांत काय आहे?"
    ]
    
    for sentence in test_sentences:
        tokens = sp.encode(sentence, out_type=str)
        ids = sp.encode(sentence, out_type=int)
        decoded = sp.decode(ids)
        print(f"\n   Input: {sentence}")
        print(f"   Tokens: {tokens[:10]}..." if len(tokens) > 10 else f"   Tokens: {tokens}")
        print(f"   IDs: {ids[:10]}..." if len(ids) > 10 else f"   IDs: {ids}")
        print(f"   Decoded: {decoded}")
    
    # Clean up temporary corpus if created
    if data_path.endswith('.jsonl'):
        try:
            os.remove(corpus_path)
            print(f"\n🗑️  Cleaned up temporary corpus file")
        except:
            pass
    
    return f"{model_path}.model", f"{model_path}.vocab"

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train SentencePiece tokenizer for Bullet OS')
    parser.add_argument('--data', type=str, required=True, help='Path to training data (.jsonl or .txt)')
    parser.add_argument('--vocab-size', type=int, default=4000, help='Vocabulary size (default: 4000)')
    parser.add_argument('--model-prefix', type=str, default='bullet_tokenizer', help='Model prefix (default: bullet_tokenizer)')
    parser.add_argument('--model-type', type=str, default='bpe', choices=['bpe', 'unigram'], help='Model type (default: bpe)')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory (default: current dir)')
    
    args = parser.parse_args()
    
    train_sentencepiece_tokenizer(
        data_path=args.data,
        vocab_size=args.vocab_size,
        model_prefix=args.model_prefix,
        model_type=args.model_type,
        output_dir=args.output_dir
    )
