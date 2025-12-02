"""
V3.0 Diagnostic Training Script
Progressive scaling approach: Start with 256-context baseline
"""

# Start with KNOWN STABLE configuration
BLOCK_SIZE = 256      # V2.0 stable baseline
D_MODEL = 256         
N_HEAD = 4            
N_LAYER = 6           # V2.0 stable baseline
FFN_HIDDEN = 1024     

# Conservative training
MAX_STEPS = 1000      # Short diagnostic run
BATCH_SIZE = 1        
LEARNING_RATE = 1e-4  # Conservative LR
WEIGHT_DECAY = 1e-2

# Paths
DATA_PATH = "/home/shri/Desktop/bulletOs/marathi_philosophy_dataset_v2.jsonl"
TOKENIZER_PATH = "/home/shri/Desktop/bulletOs/bullet_core/marathi_tokenizer.json"
CHECKPOINT_DIR = "./marathi_checkpoints_v3_diagnostic"

print("""
======================================================================
V3.0 DIAGNOSTIC RUN - Phase 1: Baseline Validation
======================================================================

Configuration:
  Context: 256 tokens (V2.0 baseline - KNOWN STABLE)
  Layers: 6 (V2.0 baseline - KNOWN STABLE)
  Steps: 1000 (diagnostic run)
  LR: 1e-4 (conservative)

Goal: Confirm V2.0 config still works before scaling up
======================================================================
""")

# This validates that the training stack itself is sound
# If this fails, the issue is not the architecture but the training code
