import requests

BASE_URL = "http://localhost:8000"

def trigger_stop():
    print("Requesting stop...")
    try:
        response = requests.post(f"{BASE_URL}/stop")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    trigger_stop()
