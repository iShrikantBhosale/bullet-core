"""
Simple training example using Bullet-Core autograd
Demonstrates end-to-end training of a tiny MLP
"""

import numpy as np
from bullet_core import Tensor, nn

# Create simple dataset (XOR problem)
X_train = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=np.float32)

y_train = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=np.float32)

# Create model
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.Linear(8, 1)
)

# Training loop
learning_rate = 0.1
epochs = 1000

print("Training XOR with Bullet-Core autograd...")
print("=" * 50)

for epoch in range(epochs):
    # Convert to tensors
    x = Tensor(X_train, requires_grad=False)
    y = Tensor(y_train, requires_grad=False)
    
    # Forward pass
    pred = model(x)
    
    # MSE loss
    loss = ((pred - y) ** 2).mean()
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # SGD update
    for param in model.parameters():
        param.data -= learning_rate * param.grad
    
    # Print progress
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1:4d} | Loss: {loss.data.item():.6f}")

print("=" * 50)
print("\nFinal predictions:")
pred_final = model(Tensor(X_train))
for i in range(4):
    print(f"Input: {X_train[i]} -> Pred: {pred_final.data[i, 0]:.4f}, Target: {y_train[i, 0]}")

print("\n✅ Training complete!")
