import os
import sys
import time
import json
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

print("=" * 65)
print("     CNN-CIFAR10: TRAINING PROPOSED MODEL (WITH AUGMENTATION)     ")
print("=" * 65)

# Set random seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Initial VRAM Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

# Data preparation
DATA_DIR = r"D:\CNN-CIFAR10-Project\data"
RESULTS_DIR = r"D:\CNN-CIFAR10-Project\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

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

# Load dataset
raw_train = CIFAR10(root=DATA_DIR, train=True, download=False, transform=None)

# 45,000 Train / 5,000 Validation split (official test set untouched)
TRAIN_COUNT = 45000
VAL_COUNT = 5000
generator = torch.Generator().manual_seed(SEED)
train_sub, val_sub = torch.utils.data.random_split(raw_train, [TRAIN_COUNT, VAL_COUNT], generator=generator)

train_ds_aug = TransformedSubset(train_sub, transform=train_transform_aug)
val_ds = TransformedSubset(val_sub, transform=eval_transform)

BATCH_SIZE = 128
PIN_MEM = (device.type == "cuda")
train_loader = DataLoader(train_ds_aug, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=PIN_MEM)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=PIN_MEM)

print(f"Training set:   {len(train_ds_aug):,} images ({len(train_loader)} batches/epoch)")
print(f"Validation set: {len(val_ds):,} images ({len(val_loader)} batches/epoch)")

# Model architecture
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

# Training configurations
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)

use_amp = (device.type == "cuda")
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

history = {
    "train_loss": [], "train_acc": [],
    "val_loss": [], "val_acc": []
}

best_val_acc = 0.0
best_epoch = 0
checkpoint_path = os.path.join(RESULTS_DIR, "best_model_augmented.pth")

print("\n" + "=" * 65)
print(f"Starting 20-Epoch Training Run on {device}...")
print("=" * 65)

total_start_time = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_start = time.time()
    
    # Training phase
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(x)
            loss = criterion(outputs, y)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item() * x.size(0)
        _, preds = outputs.max(1)
        train_total += y.size(0)
        train_correct += preds.eq(y).sum().item()
        
    ep_train_loss = train_loss / train_total
    ep_train_acc = 100.0 * train_correct / train_total
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(x)
                loss = criterion(outputs, y)
                
            val_loss += loss.item() * x.size(0)
            _, preds = outputs.max(1)
            val_total += y.size(0)
            val_correct += preds.eq(y).sum().item()
            
    ep_val_loss = val_loss / val_total
    ep_val_acc = 100.0 * val_correct / val_total
    
    scheduler.step()
    
    history["train_loss"].append(round(ep_train_loss, 4))
    history["train_acc"].append(round(ep_train_acc, 2))
    history["val_loss"].append(round(ep_val_loss, 4))
    history["val_acc"].append(round(ep_val_acc, 2))
    
    epoch_duration = time.time() - epoch_start
    
    # Checkpoint saving on best validation performance
    saved_tag = ""
    if ep_val_acc > best_val_acc:
        best_val_acc = ep_val_acc
        best_epoch = epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": ep_val_acc,
            "val_loss": ep_val_loss
        }, checkpoint_path)
        saved_tag = " -> [SAVED BEST CHECKPOINT]"
        
    if device.type == "cuda":
        vram_mb = torch.cuda.memory_allocated(0) / (1024**2)
    else:
        vram_mb = 0.0
        
    print(f"Epoch [{epoch:02d}/{NUM_EPOCHS:02d}] ({epoch_duration:.1f}s, VRAM: {vram_mb:.0f}MB) | "
          f"Train Loss: {ep_train_loss:.4f} | Train Acc: {ep_train_acc:6.2f}% | "
          f"Val Loss: {ep_val_loss:.4f} | Val Acc: {ep_val_acc:6.2f}%{saved_tag}", flush=True)

total_duration = time.time() - total_start_time

print("=" * 65)
print(f"Training completed in: {total_duration / 60:.2f} minutes ({total_duration:.1f} seconds).")
print(f"Best Validation Accuracy: {best_val_acc:.2f}% (Achieved at Epoch {best_epoch}).")
print(f"Best Model Checkpoint saved to: {checkpoint_path}")
print("=" * 65)

# Save history JSON
history_json_path = os.path.join(RESULTS_DIR, "history_augmented.json")
with open(history_json_path, "w", encoding="utf-8") as f:
    json.dump({
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "total_duration_sec": round(total_duration, 1),
        "history": history
    }, f, indent=2)
print(f"Training history saved to: {history_json_path}")

# Plot loss and accuracy curves
epochs_range = range(1, NUM_EPOCHS + 1)
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, history["train_loss"], "o-", label="Train Loss", color="royalblue")
plt.plot(epochs_range, history["val_loss"], "s-", label="Val Loss", color="darkorange")
plt.title("Augmented Model: Cross-Entropy Loss vs. Epochs", fontsize=13)
plt.xlabel("Epoch", fontsize=11)
plt.ylabel("Loss", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=11)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, history["train_acc"], "o-", label="Train Accuracy", color="royalblue")
plt.plot(epochs_range, history["val_acc"], "s-", label="Val Accuracy", color="darkorange")
plt.axhline(y=85.0, color="green", linestyle="--", label="85% Target Threshold")
plt.title(f"Augmented Model: Accuracy vs. Epochs (Peak Val: {best_val_acc:.2f}%)", fontsize=13)
plt.xlabel("Epoch", fontsize=11)
plt.ylabel("Accuracy (%)", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=11)

plt.tight_layout()
curves_path = os.path.join(RESULTS_DIR, "augmented_model_curves.png")
plt.savefig(curves_path, dpi=300)
plt.close()
print(f"Accuracy & loss curves saved to: {curves_path}")

