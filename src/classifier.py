import torch
import torch.nn as nn
import torch.optim as optim

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Input image channel = 1 (grayscale)
        # Output channels (number of filters) = 16 (you can choose)
        # Kernel size = 3 (meaning 3x3)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        # Calculate the input features to the linear layer carefully!
        # If input is (128x128), after one conv (no size change due to padding)
        # and one pool (128/2 = 64), so feature map is 16x64x64
        # So, flattened size = 16 * 64 * 64
        # For an input of HxW (e.g., 128x128):
        # After Conv2d with padding=1, size remains HxW (128x128)
        # After MaxPool2d(2,2), size becomes (H/2)x(W/2) (64x64)
        # Number of features = out_channels_of_last_conv * (H/2) * (W/2)
        # e.g. 16 * (128/2) * (128/2) = 16 * 64 * 64 = 65536
        self.fc1 = nn.Linear(16 * 64 * 64, 1) # Output 1 for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x

# Instantiate the model
model = SimpleCNN()
print(model) # See your model architecture