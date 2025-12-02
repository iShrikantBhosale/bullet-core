# 🚀 Training Bullet Models on Google Colab

This guide explains how to train `.bullet` models (like the Marathi Philosophy LLM) using the Bullet Core engine on Google Colab.

## 1. Setup Environment

Open a new Google Colab notebook and run the following cells.

### Step 1: Clone the Repository
```python
!git clone https://github.com/iShrikantBhosale/bullet-core.git
%cd bullet-core
```

### Step 2: Install Dependencies
The Bullet Engine is extremely lightweight and mostly depends on `numpy`. We also use `psutil` for monitoring.
```python
!pip install numpy psutil
```
*(Note: PyTorch is optional and only used for GPU memory logging if available. The training engine itself is pure Python/Numpy + Custom Autograd).*

## 2. Prepare Dataset

You need to upload your dataset (e.g., `marathi_philosophy_dataset_v2.jsonl`) to the Colab environment.

**Option A: Upload directly**
1. Click the "Files" icon on the left sidebar.
2. Click the "Upload" button.
3. Select your `.jsonl` file.

**Option B: Download from URL (if hosted)**
```python
!wget https://your-url.com/marathi_philosophy_dataset_v2.jsonl -O marathi_philosophy_dataset_v2.jsonl
```

**Option C: Mount Google Drive (Recommended for Checkpoints)**
If you want to save checkpoints to your Drive so you don't lose them if Colab disconnects:
```python
from google.colab import drive
drive.mount('/content/drive')

# Create a symlink or copy dataset
!cp /content/drive/MyDrive/path/to/dataset.jsonl ./marathi_philosophy_dataset_v2.jsonl
```

## 3. Configure Training

The configuration is defined in `bullet_core/configs/marathi_small.yaml`. You can edit it directly in Colab (double-click the file in the sidebar) or modify it via code:

```python
# Example: Modify config to use Google Drive for checkpoints
import yaml

config_path = 'bullet_core/configs/marathi_small.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Update checkpoint directory to Drive
config['checkpoint_dir'] = '/content/drive/MyDrive/bullet_checkpoints'

# Save back
with open(config_path, 'w') as f:
    yaml.dump(config, f)
```

## 4. Start Training

Run the production training script. This script uses the **FLASHING Protocol** (FlashAttention CPU Kernel) for stable, efficient training.

```python
!python3 bullet_core/train_production.py
```

## 5. Monitoring

The training script prints professional logs with ETA and tokens/sec.
```
[Step    870] loss=3.8249 lr=2.00e-04 eta=21.0m tokens/s=65 mem=39.9%
```

## 6. Exporting the Model

Once training is complete (or you want to test a checkpoint), the model is saved as a `.pkl` file (checkpoint) or you can export it to the `.bullet` format (if exporter is implemented).

To download files to your local machine:
```python
from google.colab import files
files.download('marathi_checkpoints_stable/checkpoint_step_10000.pkl')
```
