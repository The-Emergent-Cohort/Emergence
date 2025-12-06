"""
Experiment 001: Sanity Check

Goal: Verify the training pipeline works end-to-end.
Task: Train a tiny model to learn a simple pattern.

This is not testing any hypothesis — just confirming:
- PyTorch + CUDA work
- Training loop runs
- Loss decreases
- Model can be saved/loaded
"""

import torch
import torch.nn as nn
import torch.optim as optim

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Tiny model: just learn to output the input
class TinyModel(nn.Module):
    def __init__(self, size=16):
        super().__init__()
        self.layer1 = nn.Linear(size, 32)
        self.layer2 = nn.Linear(32, size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# Initialize
model = TinyModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training loop
print("\nTraining...")
for epoch in range(100):
    # Generate random input
    x = torch.randn(32, 16).to(device)

    # Forward pass (try to reconstruct input)
    output = model(x)
    loss = criterion(output, x)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.6f}")

print("\n✓ Training complete")

# Save model
torch.save(model.state_dict(), "tiny_model.pt")
print("✓ Model saved")

# Load and verify
model2 = TinyModel().to(device)
model2.load_state_dict(torch.load("tiny_model.pt"))
print("✓ Model loaded")

# Final test
with torch.no_grad():
    test_input = torch.randn(1, 16).to(device)
    test_output = model2(test_input)
    test_loss = criterion(test_output, test_input)
    print(f"✓ Test loss: {test_loss.item():.6f}")

print("\n🎉 Sanity check passed! Pipeline is working.")
