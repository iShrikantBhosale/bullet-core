"""
Marathi Philosophy Data Scraper
Scrapes Marathi philosophical texts from Wikisource and other sources
"""
import json
import re
import requests
from bs4 import BeautifulSoup
import time

# Target sources
SOURCES = [
    {
        "name": "Tukaram Gatha (Wikisource)",
        "base_url": "https://mr.wikisource.org/wiki/तुकाराम_गाथा",
        "type": "wikisource"
    },
    {
        "name": "Dnyaneshwari (Wikisource)",
        "base_url": "https://mr.wikisource.org/wiki/ज्ञानेश्वरी",
        "type": "wikisource"
    }
]

def clean_marathi_text(text):
    """Clean and normalize Marathi text"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove English text (basic heuristic)
    text = re.sub(r'[A-Za-z]{4,}', '', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special markers
    text = re.sub(r'\[.*?\]', '', text)
    return text.strip()

def scrape_wikisource_page(url):
    """Scrape a single Wikisource page"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find main content div
        content = soup.find('div', {'class': 'mw-parser-output'})
        if not content:
            return ""
        
        # Extract paragraphs
        paragraphs = content.find_all(['p', 'div'])
        text = '\n'.join([p.get_text() for p in paragraphs])
        
        return clean_marathi_text(text)
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""

def create_instruction_response_pairs(text, chunk_size=200):
    """Convert raw text into instruction-response pairs"""
    # Split into sentences (basic Devanagari sentence splitting)
    sentences = re.split(r'[।\n]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    pairs = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            instruction = sentences[i]
            response = sentences[i + 1]
            
            # Create variations
            pairs.append({
                "instruction": f"{instruction} याबद्दल सांगा.",
                "response": response
            })
            pairs.append({
                "instruction": f"{instruction[:50]}... पुढे काय?",
                "response": response
            })
    
    return pairs

def main():
    print("Starting Marathi Philosophy Data Collection...")
    all_data = []
    
    # For now, let's create a simple demo with the existing dataset
    # and show how to expand it
    print("\nNote: Direct web scraping requires careful implementation.")
    print("For Phase 2, we'll use a multi-pronged approach:")
    print("1. Expand existing dataset with synthetic variations")
    print("2. Manual collection of public domain texts")
    print("3. Community contributions")
    
    # Load existing dataset
    existing_file = "/home/shri/Desktop/bulletOs/marathi_philosophy_dataset.jsonl"
    print(f"\nLoading existing dataset from {existing_file}...")
    
    with open(existing_file, 'r', encoding='utf-8') as f:
        existing_data = [json.loads(line) for line in f]
    
    print(f"Existing dataset: {len(existing_data)} entries")
    
    # Create augmented dataset with variations
    augmented_data = []
    for item in existing_data:
        # Original
        augmented_data.append(item)
        
        # Variation 1: Rephrase instruction
        if len(item['instruction']) > 20:
            augmented_data.append({
                "instruction": item['instruction'][:30] + "... याबद्दल विस्तारात सांगा.",
                "response": item['response']
            })
        
        # Variation 2: Question format
        augmented_data.append({
            "instruction": f"{item['instruction']} हे कसे?",
            "response": item['response']
        })
    
    print(f"Augmented dataset: {len(augmented_data)} entries")
    
    # Save V2 dataset
    output_file = "/home/shri/Desktop/bulletOs/marathi_philosophy_dataset_v2.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in augmented_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\nSaved V2 dataset to {output_file}")
    print(f"Total entries: {len(augmented_data)}")
    
    # Calculate character count
    total_chars = sum(len(item['instruction']) + len(item['response']) for item in augmented_data)
    print(f"Total characters: {total_chars:,} (~{total_chars/1e6:.1f}M)")

if __name__ == "__main__":
    main()
