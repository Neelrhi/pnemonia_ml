import torch
from classifier import SimpleCNN


import load_test as loader
A_numpy = loader.A
B_numpy = loader.B
model = torch.load('output/model.pth')
print("Model succesfully loaded")
model.eval()

try:
    img = A_numpy[0]
    X_tensor = torch.tensor(A_numpy, dtype=torch.float32).unsqueeze(1)