"""
V3.0 Training Configuration
Expanded context (512 tokens) + Deeper model (8 layers)
"""

# Copy from train_marathi_upgraded.py and modify for V3.0

# Model Architecture (V3.0: Larger context, deeper)
BLOCK_SIZE = 512      # was 256 (2x larger)
D_MODEL = 256         # same
N_HEAD = 4            # same
N_LAYER = 8           # was 6 (deeper for more data)
FFN_HIDDEN = 1024     # same

# Training (V3.0: More steps for convergence)
MAX_STEPS = 20000     # was 15000
BATCH_SIZE = 1        # same
LEARNING_RATE = 5e-4  # same
WEIGHT_DECAY = 1e-2

# Evaluation & Checkpointing
EVAL_INTERVAL = 500
SAVE_INTERVAL = 1000
LOG_INTERVAL = 10

# Sampling
TEMPERATURE = 0.7
TOP_K = 40
TOP_P = 0.9

# Paths
DATA_PATH = "/home/shri/Desktop/bulletOs/marathi_philosophy_dataset_v2.jsonl"
TOKENIZER_PATH = "/home/shri/Desktop/bulletOs/bullet_core/marathi_tokenizer.json"
CHECKPOINT_DIR = "./marathi_checkpoints_v3"

# Expected Results
# - Model params: ~600K (vs 452K in V2)
# - BQ4 size: ~400KB (vs 276KB in V2)
# - Training time: 3-4 hours
# - Quality: Significantly better coherence
