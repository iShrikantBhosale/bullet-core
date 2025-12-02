"""
Dataset utilities with auto-cleaning
"""

import json
import unicodedata
from pathlib import Path
from typing import List

def sanitize_text(text: str) -> str:
    """
    Clean text data to prevent crashes
    
    - Normalize unicode
    - Remove bad characters
    - Remove blank lines
    - Auto-split long text
    """
    # Normalize unicode (NFKC: compatibility decomposition + canonical composition)
    text = unicodedata.normalize('NFKC', text)
    
    # Remove control characters except newline/tab
    cleaned_chars = []
    for char in text:
        category = unicodedata.category(char)
        if char in '\n\t' or not category.startswith('C'):
            cleaned_chars.append(char)
    text = ''.join(cleaned_chars)
    
    # Remove multiple blank lines
    lines = text.split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    text = '\n'.join(lines)
    
    # Auto-split very long lines (> 1000 chars)
    lines = text.split('\n')
    split_lines = []
    for line in lines:
        if len(line) > 1000:
            # Split at sentence boundaries
            sentences = line.replace('।', '।\n').split('\n')
            split_lines.extend(sentences)
        else:
            split_lines.append(line)
    
    text = '\n'.join(split_lines)
    
    return text

def load_and_clean_dataset(path: str) -> List[str]:
    """
    Load JSONL dataset and auto-clean all text
    """
    print(f"📂 Loading dataset from {path}...")
    
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                # Try different field names
                text = data.get('text', '') or data.get('response', '') or data.get('output', '')
                
                # Auto-clean
                text = sanitize_text(text)
                
                if text:  # Only add non-empty
                    texts.append(text)
            except Exception as e:
                print(f"⚠️  Skipping line {i}: {e}")
    
    print(f"✅ Loaded {len(texts)} clean texts")
    return texts

def validate_dataset(texts: List[str], min_length: int = 10):
    """Validate dataset quality"""
    if not texts:
        raise ValueError("Dataset is empty!")
    
    # Check average length
    avg_len = sum(len(t) for t in texts) / len(texts)
    if avg_len < min_length:
        raise ValueError(f"Average text length too short: {avg_len:.1f} < {min_length}")
    
    # Check for unicode issues
    for i, text in enumerate(texts[:100]):  # Sample first 100
        try:
            text.encode('utf-8')
        except UnicodeEncodeError as e:
            raise ValueError(f"Unicode error in text {i}: {e}")
    
    print(f"✅ Dataset validated: {len(texts)} texts, avg length {avg_len:.1f} chars")
