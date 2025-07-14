import torch

device = 0 if torch.cuda.is_available() else "cpu"

print(device)