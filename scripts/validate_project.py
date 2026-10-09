import os
import sys
import nbformat
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

print("==================================================")
print("     CNN-CIFAR10 PROJECT VALIDATION SUITE         ")
print("==================================================")

# 1. Validate Notebook Format
notebook_path = r"D:\CNN-CIFAR10-Project\CNN_CIFAR10.ipynb"
assert os.path.exists(notebook_path), "Notebook file not found!"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)
nbformat.validate(nb)
print(f"[CHECK 1 PASSED] Notebook is valid format ({len(nb.cells)} cells found).")

# 2. Check Required Files
required_files = ["README.md", "requirements.txt", ".gitignore"]
for fname in required_files:
    fpath = os.path.join(r"D:\CNN-CIFAR10-Project", fname)
    assert os.path.exists(fpath), f"Missing file: {fname}"
    assert os.path.getsize(fpath) > 0, f"File is empty: {fname}"
print(f"[CHECK 2 PASSED] Required files present: {required_files}")

# 3. Check Directory Structure
results_dir = r"D:\CNN-CIFAR10-Project\results"
assert os.path.exists(results_dir) and os.path.isdir(results_dir), "results/ directory missing!"
print("[CHECK 3 PASSED] results/ directory verified.")

# 4. Hardware Verification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[CHECK 4 PASSED] Hardware acceleration: {device}")
if device.type == "cuda":
    print(f"                 GPU: {torch.cuda.get_device_name(0)}")

# 5. Architecture and Tensor Dimensions Verification
class CIFAR10_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR10_CNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.25)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.30)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.40)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.50),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.block3(self.block2(self.block1(x))))

model = CIFAR10_CNN(num_classes=10).to(device)
param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[CHECK 5 PASSED] CNN instantiated with {param_count:,} trainable parameters.")

# 6. Dummy Batch Forward Pass
dummy_batch = torch.randn(8, 3, 32, 32, device=device)
model.train()
output = model(dummy_batch)
assert output.shape == (8, 10), f"Expected shape (8, 10), got {output.shape}"
print(f"[CHECK 6 PASSED] Dummy forward pass: Input {list(dummy_batch.shape)} -> Output {list(output.shape)}")

# 7. Backward Pass Gradient Check
loss_fn = nn.CrossEntropyLoss()
dummy_targets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
optimizer.zero_grad()
loss = loss_fn(output, dummy_targets)
loss.backward()
optimizer.step()
print(f"[CHECK 7 PASSED] Backward pass gradient calculation & optimizer step verified (Dummy Loss: {loss.item():.4f}).")

print("==================================================")
print("     ALL 7 INITIAL PROJECT CHECKS PASSED!         ")
print("==================================================")

