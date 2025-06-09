import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np # You might need this if not already imported by Loader

# 1. Import your model definition
from classifier import SimpleCNN  # This imports the SimpleCNN class

# 2. Import your data
# When Loader.py is imported, its code runs, and X and Y become available as module-level variables.
import loader as Loader
X_numpy = Loader.X  # Get the X array from the Loader module
Y_numpy = Loader.Y  # Get the Y array from the Loader module

# 3. Convert data to PyTorch Tensors and reshape X
# Assuming X_numpy has shape (num_images, height, width) e.g., (10, 128, 128)
X_tensor = torch.tensor(X_numpy, dtype=torch.float32).unsqueeze(1)  # Adds channel dim: (N, 1, H, W)

# Assuming Y_numpy has shape (num_images,) e.g., (10,)
Y_tensor = torch.tensor(Y_numpy, dtype=torch.float32).unsqueeze(1)  # Reshape to (N, 1)

print(f"Shape of X_tensor: {X_tensor.shape}")
print(f"Shape of Y_tensor: {Y_tensor.shape}")

# 4. Instantiate your model
# It's better to instantiate the model here rather than in classifier.py directly
# if classifier.py is meant to be a library of definitions.
model = SimpleCNN()
print("Model created:")
print(model)

# 5. Define Loss Function and Optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. The Training Loop
epochs = 1000 # Let's try a few more epochs

print("\nStarting training...")
for epoch in range(epochs):
    model.train() # Set the model to training mode

    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, Y_tensor)

    # Backward pass and optimization
    optimizer.zero_grad() # Clear previous gradients
    loss.backward()       # Compute gradients
    optimizer.step()        # Update weights

    # Calculate accuracy (simple version for binary case)
    # Ensure outputs and Y_tensor are on the same device if using GPU later
    predicted = (outputs > 0.5).float() 
    correct = (predicted == Y_tensor).sum().item()
    total = Y_tensor.size(0)
    accuracy = correct / total

    if (epoch + 1) % 2 == 0: # Print every 2 epochs
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

print("Training finished.")

# Optional: You could add code here to evaluate or save the model
torch.save(model, 'output/model.pth')
print("Model saved")