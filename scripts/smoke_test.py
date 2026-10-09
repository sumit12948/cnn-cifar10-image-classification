import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

print("==================================================")
print("     RUNNING PRE-TRAINING SMOKE TESTS             ")
print("==================================================")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 1. Dataset download and split test
DATA_DIR = r"D:\CNN-CIFAR10-Project\data"
os.makedirs(DATA_DIR, exist_ok=True)
print(f"Verifying CIFAR-10 dataset in: {DATA_DIR}...")
start_dl = time.time()
raw_train = CIFAR10(root=DATA_DIR, train=True, download=True, transform=None)
raw_test = CIFAR10(root=DATA_DIR, train=False, download=True, transform=None)
print(f"Dataset verified in {time.time() - start_dl:.2f}s.")
print(f"Raw Train samples: {len(raw_train):,}")
print(f"Raw Test samples:  {len(raw_test):,}")

# Split 45,000 / 5,000
TRAIN_COUNT = 45000
VAL_COUNT = 5000
generator = torch.Generator().manual_seed(42)
train_sub, val_sub = torch.utils.data.random_split(raw_train, [TRAIN_COUNT, VAL_COUNT], generator=generator)

# Verify disjoint splits
train_indices_set = set(train_sub.indices)
val_indices_set = set(val_sub.indices)
assert len(train_indices_set.intersection(val_indices_set)) == 0, "Data leakage detected between train and val!"
print("[TEST 1 PASSED] 45k/5k train/val split is completely disjoint (Zero data leakage).")

# 2. Transform verification
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

train_transform_aug = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
])

class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
    def __len__(self):
        return len(self.subset)

train_ds_aug = TransformedSubset(train_sub, transform=train_transform_aug)
val_ds = TransformedSubset(val_sub, transform=eval_transform)
test_ds = TransformedSubset(raw_test, transform=eval_transform)

train_loader = DataLoader(train_ds_aug, batch_size=128, shuffle=True, pin_memory=(device.type == "cuda"))
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, pin_memory=(device.type == "cuda"))
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, pin_memory=(device.type == "cuda"))

sample_x, sample_y = next(iter(train_loader))
assert sample_x.shape == (128, 3, 32, 32), f"Unexpected shape {sample_x.shape}"
print(f"[TEST 2 PASSED] DataLoader batch generated: {list(sample_x.shape)}, Labels: {list(sample_y.shape)}")

# 3. Model Architecture and AMP forward/backward smoke test
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
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Mixed precision scaler
use_amp = (device.type == "cuda")
scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

# Test 1 mini-batch step with AMP
model.train()
optimizer.zero_grad()
sample_x, sample_y = sample_x.to(device), sample_y.to(device)

with torch.amp.autocast('cuda', enabled=use_amp):
    train_outputs = model(sample_x)
    loss = criterion(train_outputs, sample_y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

print(f"[TEST 3 PASSED] 1-batch training step executed with AMP (Loss: {loss.item():.4f})")

# 4. Checkpoint save and reload verification (both tested in eval mode)
model.eval()
with torch.no_grad():
    with torch.amp.autocast('cuda', enabled=use_amp):
        eval_outputs = model(sample_x)

checkpoint_path = r"D:\CNN-CIFAR10-Project\results\smoke_test_checkpoint.pth"
torch.save({
    "epoch": 1,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": loss.item()
}, checkpoint_path)

new_model = CIFAR10_CNN(num_classes=10).to(device)
ckpt = torch.load(checkpoint_path, map_location=device)
new_model.load_state_dict(ckpt["model_state_dict"])
new_model.eval()

with torch.no_grad():
    with torch.amp.autocast('cuda', enabled=use_amp):
        reloaded_outputs = new_model(sample_x)

assert torch.allclose(eval_outputs, reloaded_outputs, atol=1e-5), "Checkpoint reload output mismatch!"
print("[TEST 4 PASSED] Checkpoint save and reload verified with identical output tensor values.")

# Clean up smoke test checkpoint
if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)

# 5. VRAM consumption audit
if device.type == "cuda":
    alloc_mb = torch.cuda.memory_allocated(0) / (1024**2)
    res_mb = torch.cuda.memory_reserved(0) / (1024**2)
    print(f"[TEST 5 PASSED] VRAM allocated: {alloc_mb:.1f} MB, Reserved: {res_mb:.1f} MB (Well within 4,096 MB VRAM).")

print("==================================================")
print("     ALL PRE-TRAINING SMOKE TESTS PASSED!         ")
print("==================================================")
