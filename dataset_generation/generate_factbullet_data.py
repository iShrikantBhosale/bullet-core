import json
import random
import os

# Configuration
OUTPUT_FILE = "factbullet_dataset.jsonl"
TOKENIZER_FILE = "factbullet_tokenizer.json"
NUM_SAMPLES = 1000

# Templates
SYSTEM_PROMPT = """You are the FACTBULLET MODEL.
Purpose: Extract facts, eliminate noise, produce verified answers with citations.
Reasoning:
1. Understand Query
2. Extract Info
3. Verify Sources
4. Detect Contradictions
5. Compress Facts
6. Map Citations
7. Generate Answer
Rule: Never output raw text. Compress into clean points. Maintain precision."""

TOPICS = ["Space", "History", "Science", "Technology", "Nature", "Geography"]
ENTITIES = {
    "Space": ["Mars", "Jupiter", "The Moon", "Black Holes", "NASA"],
    "History": ["Rome", "World War II", "The Industrial Revolution", "Ancient Egypt"],
    "Science": ["Photosynthesis", "Gravity", "Atoms", "DNA"],
    "Technology": ["AI", "Blockchain", "The Internet", "Computers"],
    "Nature": ["Lions", "Oceans", "Rainforests", "Volcanoes"],
    "Geography": ["Mount Everest", "The Nile", "Antarctica", "The Amazon"]
}

ACTIONS = ["discovered", "invented", "located in", "known for", "made of"]

def generate_sample():
    topic = random.choice(TOPICS)
    entity = random.choice(ENTITIES[topic])
    action = random.choice(ACTIONS)
    
    query = f"What is {entity} {action}?"
    if action == "located in":
        query = f"Where is {entity} located?"
    elif action == "made of":
        query = f"What is {entity} made of?"

    # Synthetic Context
    fact1 = f"{entity} is a significant subject in {topic}."
    fact2 = f"It is widely {action} by experts."
    fact3 = f"Research shows {entity} has unique properties."
    
    context = f"[1] {fact1} It is important.\n[2] {fact2} This was confirmed in 2023.\n[3] {fact3}"
    
    # Synthetic Answer
    answer = f"{entity} is primarily {action} in the field of {topic} [1]. Recent studies from 2023 confirm its status [2]. Additionally, it is known for its unique properties [3]."

    # Full Text
    text = f"{SYSTEM_PROMPT}\n\nFacts:\n{context}\nQuestion: {query}\nAnswer (summarize facts with citations [1], [2]...):\n{answer}"
    
    return {"text": text}

def build_tokenizer(texts):
    print("Building tokenizer...")
    # Very simple char-level + common words tokenizer for demo
    # In production, use a real BPE library like tokenizers
    
    vocab = set()
    for text in texts:
        for char in text:
            vocab.add(char)
            
    # Add some special tokens
    special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    vocab_list = special_tokens + sorted(list(vocab))
    
    vocab_map = {token: i for i, token in enumerate(vocab_list)}
    
    return {"vocab": vocab_map, "merges": []}

def main():
    print(f"Generating {NUM_SAMPLES} samples...")
    data = []
    for _ in range(NUM_SAMPLES):
        data.append(generate_sample())
        
    print(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
            
    # Build Tokenizer
    texts = [d['text'] for d in data]
    tokenizer_data = build_tokenizer(texts)
    
    print(f"Saving tokenizer to {TOKENIZER_FILE}...")
    with open(TOKENIZER_FILE, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_data, f, indent=2)
        
    print("Done!")

if __name__ == "__main__":
    main()
