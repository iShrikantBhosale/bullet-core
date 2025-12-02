import requests
import time
import os

BASE_URL = "http://localhost:8000"
DATASET_PATH = "../marathi_philosophy_dataset.jsonl"

def trigger_training():
    print("1. Triggering training via Dashboard API...")
    if not os.path.exists(DATASET_PATH):
        print(f"Error: {DATASET_PATH} not found.")
        return

    with open(DATASET_PATH, "rb") as f:
        files = {"file": f}
        # Config from BULLET_SPEC_v1.0.md example
        data = {
            "vocab_size": 5000, 
            "dim": 256,
            "num_heads": 4,
            "num_layers": 8,
            "max_seq_len": 64, 
            "epochs": 1, 
            "batch_size": 8, # Increased batch size for better CPU utilization
            "learning_rate": 0.001
        }
        try:
            response = requests.post(f"{BASE_URL}/train", files=files, data=data)
            response.raise_for_status()
            print("Training started:", response.json())
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to backend. Is it running?")
            return
        except Exception as e:
            print(f"Error: {e}")
            return

    print("\n2. Monitoring Logs...")
    model_name = None
    last_log_index = 0
    
    while True:
        try:
            response = requests.get(f"{BASE_URL}/logs")
            data = response.json()
            
            # Print new logs
            logs = data.get("logs", [])
            if len(logs) > last_log_index:
                for i in range(last_log_index, len(logs)):
                    print(f"[LOG] {logs[i]}")
                last_log_index = len(logs)
                
            if data["model_available"]:
                print("\nTraining complete!")
                model_name = data["model_name"]
                break
            
            if not data["is_training"] and not data["model_available"] and data["progress"] < 1.0:
                 print("\nTraining stopped unexpectedly.")
                 break
                 
            time.sleep(1) # Poll faster
        except Exception as e:
            print(f"\nPolling error: {e}")
            break

    if model_name:
        print(f"\n3. Downloading {model_name}...")
        response = requests.get(f"{BASE_URL}/download/{model_name}")
        if response.status_code == 200:
            with open(model_name, "wb") as f:
                f.write(response.content)
            print(f"Downloaded {model_name} ({len(response.content)} bytes).")
        else:
            print(f"Error downloading model: {response.status_code}")

if __name__ == "__main__":
    trigger_training()
