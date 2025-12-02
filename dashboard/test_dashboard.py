import requests
import time
import os
import sys

BASE_URL = "http://localhost:8000"

def test_dashboard():
    print("1. Testing /train endpoint...")
    file_path = "tiny_data.txt"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    with open(file_path, "rb") as f:
        files = {"file": f}
        data = {
            "vocab_size": 100,
            "dim": 32,
            "num_heads": 2,
            "num_layers": 1,
            "max_seq_len": 16,
            "epochs": 2,
            "batch_size": 2
        }
        try:
            response = requests.post(f"{BASE_URL}/train", files=files, data=data)
            response.raise_for_status()
            print("Training started:", response.json())
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to backend. Is it running?")
            return

    print("\n2. Polling /logs...")
    model_name = None
    for i in range(20): # Timeout after 20s
        response = requests.get(f"{BASE_URL}/logs")
        data = response.json()
        
        # Print new logs (simplified)
        if data["logs"]:
            print(f"Logs: {data['logs'][-1]}")
            
        if data["model_available"]:
            print("Training complete!")
            model_name = data["model_name"]
            break
        
        time.sleep(1)
    
    if not model_name:
        print("Error: Training timed out or failed.")
        return

    print(f"\n3. Downloading {model_name}...")
    response = requests.get(f"{BASE_URL}/download/{model_name}")
    if response.status_code == 200:
        with open(model_name, "wb") as f:
            f.write(response.content)
        print(f"Downloaded {model_name} ({len(response.content)} bytes).")
    else:
        print(f"Error downloading model: {response.status_code}")

if __name__ == "__main__":
    test_dashboard()
