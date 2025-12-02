"""
End-to-end training example using Bullet-Core
Demonstrates complete training pipeline with all Phase 4 features
"""

import numpy as np
from bullet_core import Tensor, nn
from bullet_core.optim import AdamW
from bullet_core.scheduler import WarmupCosineAnnealing
from bullet_core.trainer import Trainer

# ===== 1. CREATE SYNTHETIC DATASET =====

def generate_data(num_samples=1000, input_dim=128, output_dim=64):
    """Generate synthetic regression data"""
    X = np.random.randn(num_samples, input_dim).astype(np.float32)
    # True weights for synthetic task
    W_true = np.random.randn(input_dim, output_dim).astype(np.float32) * 0.1
    y = X @ W_true + np.random.randn(num_samples, output_dim).astype(np.float32) * 0.01
    return X, y

# Generate train and validation data
X_train, y_train = generate_data(num_samples=800, input_dim=128, output_dim=64)
X_val, y_val = generate_data(num_samples=200, input_dim=128, output_dim=64)

# Create simple data loaders (batch iterators)
def create_batches(X, y, batch_size=32):
    """Create batches from data"""
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    batches = []
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        batches.append((X[batch_indices], y[batch_indices]))
    
    return batches

train_loader = create_batches(X_train, y_train, batch_size=32)
val_loader = create_batches(X_val, y_val, batch_size=32)

# ===== 2. BUILD MODEL =====

model = nn.Sequential(
    nn.Linear(128, 256),
    nn.RMSNorm(256),
    nn.Linear(256, 128),
    nn.RMSNorm(128),
    nn.Linear(128, 64)
)

print("=" * 60)
print("Bullet-Core Training Example")
print("=" * 60)
print(f"Model: {model}")
print(f"Parameters: {sum(p.data.size for p in model.parameters())}")
print()

# ===== 3. SETUP TRAINING =====

# Optimizer: AdamW with weight decay
optimizer = AdamW(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=0.01
)

# Scheduler: Warmup + Cosine Annealing
scheduler = WarmupCosineAnnealing(
    optimizer,
    warmup_steps=100,
    T_max=1000,
    eta_min=1e-6
)

# Loss function: MSE
def mse_loss(pred, target):
    """Mean squared error loss"""
    diff = pred - target
    return (diff ** 2).mean()

# ===== 4. CREATE TRAINER =====

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=mse_loss,
    train_data=train_loader,
    val_data=val_loader,
    scheduler=scheduler,
    max_epochs=50,
    gradient_clip=1.0,
    checkpoint_dir='./checkpoints',
    early_stopping_patience=10,
    log_interval=5
)

# ===== 5. TRAIN! =====

results = trainer.train()

# ===== 6. EVALUATE =====

print("\n" + "=" * 60)
print("Training Results")
print("=" * 60)
print(f"Final train loss: {results['train_losses'][-1]:.6f}")
print(f"Final val loss:   {results['val_losses'][-1]:.6f}")
print(f"Best val loss:    {results['best_val_loss']:.6f}")
print()

# Test on a few samples
print("Sample Predictions:")
print("-" * 60)
test_samples = 5
for i in range(test_samples):
    x_test = Tensor(X_val[i:i+1], requires_grad=False)
    y_test = y_val[i:i+1]
    
    pred = model(x_test)
    
    print(f"Sample {i+1}:")
    print(f"  Predicted: {pred.data[0, :3]} ...")  # First 3 values
    print(f"  Actual:    {y_test[0, :3]} ...")
    print(f"  Error:     {np.abs(pred.data[0] - y_test[0]).mean():.6f}")
    print()

print("=" * 60)
print("✅ Training complete!")
print("=" * 60)
