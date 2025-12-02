import sys
import os
from bullet_py_api import Bullet

def main():
    print("Testing inference only (no torch)...")
    if not os.path.exists("tiny.bullet"):
        print("tiny.bullet not found. Run export first.")
        return

    try:
        print("Initializing Bullet...")
        b = Bullet("tiny.bullet")
        print("Bullet initialized.")
        res = b.chat("token_1")
        print(f"Result: {res}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
