import json
import os

TOKENIZER_PATH = "marathi_tokenizer.json"
VOCAB_OUT = "vocab.txt"
MERGES_OUT = "merges.txt"

def main():
    print(f"Loading {TOKENIZER_PATH}...")
    with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 1. Export Vocab
    # bullet-builder expects "token score" or "token"
    # We need to ensure the order matches token IDs
    # data['vocab'] is a list of tokens, but is it sorted by ID?
    # data['id_to_token'] maps ID to token. This is the source of truth.
    
    id_to_token = {int(k): v for k, v in data['id_to_token'].items()}
    max_id = max(id_to_token.keys())
    
    print(f"Exporting {len(id_to_token)} tokens to {VOCAB_OUT}...")
    with open(VOCAB_OUT, 'w', encoding='utf-8') as f:
        for i in range(max_id + 1):
            if i in id_to_token:
                token = id_to_token[i]
                # Escape spaces? bullet-builder reads line by line.
                # If token contains newline, we are in trouble.
                # BPE tokens might contain spaces (as U+2581 or just space).
                # Our tokenizer uses space ' ' as a token.
                # bullet-builder logic:
                # size_t last_space = line.find_last_of(' ');
                # If token is " ", last_space is 0. score_str is empty.
                # It treats it as token " ".
                # If token is "hello world", last_space is 5. score_str is "world".
                # "world" is not float. So it treats "hello world" as token.
                # Seems robust enough for now.
                f.write(f"{token}\n")
            else:
                f.write(f"[UNK]\n") # Placeholder
                
    # 2. Export Merges
    # data['merges'] is list of [token1, token2]
    print(f"Exporting {len(data['merges'])} merges to {MERGES_OUT}...")
    with open(MERGES_OUT, 'w', encoding='utf-8') as f:
        for pair in data['merges']:
            f.write(f"{pair[0]} {pair[1]}\n")
            
    print("Done.")

if __name__ == "__main__":
    main()
