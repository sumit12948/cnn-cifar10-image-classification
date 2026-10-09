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

from sklearn.metrics import classification_report, confusion_matrix

print("=" * 70)
print("     CNN-CIFAR10: BASELINE TRAINING & COMPREHENSIVE EVALUATION     ")
print("=" * 70)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

DATA_DIR = r"D:\CNN-CIFAR10-Project\data"
RESULTS_DIR = r"D:\CNN-CIFAR10-Project\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# Normalization-only transform (No Augmentation)
clean_transform = transforms.Compose([
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

# Load raw datasets
raw_train = CIFAR10(root=DATA_DIR, train=True, download=False, transform=None)
raw_test = CIFAR10(root=DATA_DIR, train=False, download=False, transform=None)

# Exactly identical 45,000 / 5,000 split using seed 42
TRAIN_COUNT = 45000
VAL_COUNT = 5000
generator = torch.Generator().manual_seed(SEED)
train_sub, val_sub = torch.utils.data.random_split(raw_train, [TRAIN_COUNT, VAL_COUNT], generator=generator)

# Baseline training set has NO augmentation
train_ds_base = TransformedSubset(train_sub, transform=clean_transform)
val_ds = TransformedSubset(val_sub, transform=clean_transform)
test_ds = TransformedSubset(raw_test, transform=clean_transform)

BATCH_SIZE = 128
PIN_MEM = (device.type == "cuda")
train_loader_base = DataLoader(train_ds_base, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=PIN_MEM)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=PIN_MEM)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=PIN_MEM)

print(f"Baseline Train Samples: {len(train_ds_base):,}")
print(f"Validation Samples:     {len(val_ds):,}")
print(f"Held-out Test Samples:  {len(test_ds):,}")

# Identical CNN architecture
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

# =====================================================================
# TASK 1: BASELINE MODEL TRAINING (WITHOUT AUGMENTATION)
# =====================================================================
print("\n" + "=" * 70)
print("TASK 1: TRAINING BASELINE CNN (WITHOUT DATA AUGMENTATION)")
print("=" * 70)

model_baseline = CIFAR10_CNN(num_classes=10).to(device)
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

criterion = nn.CrossEntropyLoss()
optimizer_base = optim.AdamW(model_baseline.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler_base = CosineAnnealingLR(optimizer_base, T_max=NUM_EPOCHS, eta_min=1e-5)

use_amp = (device.type == "cuda")
scaler_base = torch.amp.GradScaler("cuda", enabled=use_amp)

history_baseline = {
    "train_loss": [], "train_acc": [],
    "val_loss": [], "val_acc": []
}

best_val_acc_base = 0.0
best_epoch_base = 0
checkpoint_path_base = os.path.join(RESULTS_DIR, "best_model_baseline.pth")

base_start_time = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    ep_start = time.time()
    
    # Train
    model_baseline.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in train_loader_base:
        x, y = x.to(device), y.to(device)
        optimizer_base.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model_baseline(x)
            loss = criterion(out, y)
        scaler_base.scale(loss).backward()
        scaler_base.step(optimizer_base)
        scaler_base.update()
        
        running_loss += loss.item() * x.size(0)
        _, preds = out.max(1)
        total += y.size(0)
        correct += preds.eq(y).sum().item()
        
    ep_train_loss = running_loss / total
    ep_train_acc = 100.0 * correct / total
    
    # Val
    model_baseline.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model_baseline(x)
                loss = criterion(out, y)
            val_loss += loss.item() * x.size(0)
            _, preds = out.max(1)
            val_total += y.size(0)
            val_correct += preds.eq(y).sum().item()
            
    ep_val_loss = val_loss / val_total
    ep_val_acc = 100.0 * val_correct / val_total
    
    scheduler_base.step()
    
    history_baseline["train_loss"].append(round(ep_train_loss, 4))
    history_baseline["train_acc"].append(round(ep_train_acc, 2))
    history_baseline["val_loss"].append(round(ep_val_loss, 4))
    history_baseline["val_acc"].append(round(ep_val_acc, 2))
    
    saved_tag = ""
    if ep_val_acc > best_val_acc_base:
        best_val_acc_base = ep_val_acc
        best_epoch_base = epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": model_baseline.state_dict(),
            "optimizer_state_dict": optimizer_base.state_dict(),
            "val_acc": ep_val_acc,
            "val_loss": ep_val_loss
        }, checkpoint_path_base)
        saved_tag = " -> [SAVED BEST CHECKPOINT]"
        
    print(f"Epoch [{epoch:02d}/{NUM_EPOCHS:02d}] ({time.time() - ep_start:.1f}s) | "
          f"Train Loss: {ep_train_loss:.4f} | Train Acc: {ep_train_acc:6.2f}% | "
          f"Val Loss: {ep_val_loss:.4f} | Val Acc: {ep_val_acc:6.2f}%{saved_tag}", flush=True)

base_duration = time.time() - base_start_time
print("-" * 70)
print(f"Baseline Training Finished in: {base_duration / 60:.2f} mins.")
print(f"Baseline Best Validation Accuracy: {best_val_acc_base:.2f}% (Epoch {best_epoch_base})")
print(f"Saved Checkpoint: {checkpoint_path_base}")

# Save baseline history JSON
history_base_json_path = os.path.join(RESULTS_DIR, "history_baseline.json")
with open(history_base_json_path, "w", encoding="utf-8") as f:
    json.dump({
        "best_epoch": best_epoch_base,
        "best_val_acc": best_val_acc_base,
        "total_duration_sec": round(base_duration, 1),
        "history": history_baseline
    }, f, indent=2)

# =====================================================================
# TASK 2: FINAL TEST SET EVALUATION (10,000 IMAGES)
# =====================================================================
print("\n" + "=" * 70)
print("TASK 2: EVALUATION ON UNTOUCHED HELD-OUT TEST SET (10,000 IMAGES)")
print("=" * 70)

# Load Augmented Model Checkpoint
aug_ckpt_path = os.path.join(RESULTS_DIR, "best_model_augmented.pth")
assert os.path.exists(aug_ckpt_path), f"Missing checkpoint: {aug_ckpt_path}"

model_augmented = CIFAR10_CNN(num_classes=10).to(device)
ckpt_aug = torch.load(aug_ckpt_path, map_location=device)
model_augmented.load_state_dict(ckpt_aug["model_state_dict"])
model_augmented.eval()
print(f"Successfully loaded best Augmented model (Validation Acc: {ckpt_aug['val_acc']:.2f}% from Epoch {ckpt_aug['epoch']}).")

# Load Baseline Model Checkpoint
ckpt_base = torch.load(checkpoint_path_base, map_location=device)
model_baseline.load_state_dict(ckpt_base["model_state_dict"])
model_baseline.eval()
print(f"Successfully loaded best Baseline model (Validation Acc: {ckpt_base['val_acc']:.2f}% from Epoch {ckpt_base['epoch']}).")

def evaluate_test_set(model, dataloader, criterion, device):
    all_preds, all_labels = [], []
    total_loss, total_count = 0.0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(x)
                loss = criterion(outputs, y)
            total_loss += loss.item() * x.size(0)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            total_count += y.size(0)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = 100.0 * np.mean(all_preds == all_labels)
    avg_loss = total_loss / total_count
    return all_preds, all_labels, acc, avg_loss

# Run inference on all 10,000 test images
aug_preds, true_labels, aug_test_acc, aug_test_loss = evaluate_test_set(model_augmented, test_loader, criterion, device)
base_preds, _, base_test_acc, base_test_loss = evaluate_test_set(model_baseline, test_loader, criterion, device)

print("\n" + "=" * 70)
print(f"OFFICIAL TEST SET RESULTS (10,000 Held-Out Images):")
print(f"  Proposed Model (With Augmentation):    Acc = {aug_test_acc:.2f}% | Loss = {aug_test_loss:.4f}")
print(f"  Baseline Model (Without Augmentation): Acc = {base_test_acc:.2f}% | Loss = {base_test_loss:.4f}")
print(f"  Improvement from Data Augmentation:    +{aug_test_acc - base_test_acc:.2f}%")
print(f"  College Goal Target (>= 85.00%):       {'ACHIEVED!' if aug_test_acc >= 85.0 else 'NOT ACHIEVED'}")
print("=" * 70)

# Full Classification Report
clf_report_str = classification_report(true_labels, aug_preds, target_names=CLASSES, digits=4)
print("\nClassification Report (Proposed Model with Augmentation):")
print(clf_report_str)

report_txt_path = os.path.join(RESULTS_DIR, "classification_report.txt")
with open(report_txt_path, "w", encoding="utf-8") as f:
    f.write(clf_report_str)

clf_dict = classification_report(true_labels, aug_preds, target_names=CLASSES, output_dict=True)
report_json_path = os.path.join(RESULTS_DIR, "classification_report.json")
with open(report_json_path, "w", encoding="utf-8") as f:
    json.dump(clf_dict, f, indent=2)

# =====================================================================
# TASK 3: VISUALIZATIONS
# =====================================================================
print("\n" + "=" * 70)
print("TASK 3: GENERATING HIGH-RESOLUTION PLOTS")
print("=" * 70)

# 1. Augmentation Comparison Curves
plt.figure(figsize=(14, 5))

# Load augmented history
with open(os.path.join(RESULTS_DIR, "history_augmented.json"), "r", encoding="utf-8") as f:
    hist_aug_data = json.load(f)
history_augmented = hist_aug_data["history"]

# Validation Loss Comparison
plt.subplot(1, 2, 1)
plt.plot(range(1, NUM_EPOCHS + 1), history_augmented["val_loss"], "o-", label=f"With Augmentation (Best Val Loss: {min(history_augmented['val_loss']):.4f})", color="royalblue", linewidth=2)
plt.plot(range(1, NUM_EPOCHS + 1), history_baseline["val_loss"], "s--", label=f"Without Augmentation (Best Val Loss: {min(history_baseline['val_loss']):.4f})", color="crimson", linewidth=2)
plt.title("Validation Loss Comparison: Overfitting Analysis", fontsize=13)
plt.xlabel("Epoch", fontsize=11)
plt.ylabel("Cross-Entropy Loss", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=10)

# Validation Accuracy Comparison
plt.subplot(1, 2, 2)
plt.plot(range(1, NUM_EPOCHS + 1), history_augmented["val_acc"], "o-", label=f"With Augmentation (Val: {hist_aug_data['best_val_acc']:.2f}%, Test: {aug_test_acc:.2f}%)", color="royalblue", linewidth=2)
plt.plot(range(1, NUM_EPOCHS + 1), history_baseline["val_acc"], "s--", label=f"Without Augmentation (Val: {best_val_acc_base:.2f}%, Test: {base_test_acc:.2f}%)", color="crimson", linewidth=2)
plt.axhline(y=85.0, color="green", linestyle=":", linewidth=2, label="85% Project Target Threshold")
plt.title("Validation Accuracy Comparison", fontsize=13)
plt.xlabel("Epoch", fontsize=11)
plt.ylabel("Accuracy (%)", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=10)

plt.tight_layout()
comp_path = os.path.join(RESULTS_DIR, "augmentation_comparison.png")
plt.savefig(comp_path, dpi=300)
plt.close()
print(f"Saved: {comp_path}")

# 2. Confusion Matrix
cm = confusion_matrix(true_labels, aug_preds)
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(CLASSES))
plt.xticks(tick_marks, CLASSES, rotation=45, ha="right", fontsize=10)
plt.yticks(tick_marks, CLASSES, fontsize=10)
plt.title(f"Confusion Matrix on 10,000 Test Images (Actual Test Acc: {aug_test_acc:.2f}%)", fontsize=13)
plt.ylabel("True Class", fontsize=11)
plt.xlabel("Predicted Class", fontsize=11)

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], "d"),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=9)

plt.tight_layout()
cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=300)
plt.close()
print(f"Saved: {cm_path}")

# 3. Misclassified Images
def denormalize(tensor_img):
    img = tensor_img.clone().cpu()
    for c in range(3):
        img[c] = img[c] * CIFAR10_STD[c] + CIFAR10_MEAN[c]
    return torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()

misclassified_indices = np.where(aug_preds != true_labels)[0]
print(f"Total Misclassified Test Images: {len(misclassified_indices)} out of 10,000 ({100.0 - aug_test_acc:.2f}% error rate)")

plt.figure(figsize=(12, 12))
for i, idx in enumerate(misclassified_indices[:16]):
    img_tensor, _ = test_ds[idx]
    true_name = CLASSES[true_labels[idx]]
    pred_name = CLASSES[aug_preds[idx]]
    
    plt.subplot(4, 4, i + 1)
    plt.imshow(denormalize(img_tensor))
    plt.title(f"True: {true_name}\nPred: {pred_name}", fontsize=11, color="darkred")
    plt.axis("off")

plt.suptitle("Sample Misclassified Test Images (Error Analysis)", fontsize=14, y=0.95)
plt.tight_layout()
misc_path = os.path.join(RESULTS_DIR, "misclassified_examples.png")
plt.savefig(misc_path, dpi=300)
plt.close()
print(f"Saved: {misc_path}")

# Save comprehensive final results summary json
summary_results = {
    "target_accuracy": 85.0,
    "augmented_model": {
        "best_val_accuracy": hist_aug_data["best_val_acc"],
        "final_test_accuracy": round(float(aug_test_acc), 2),
        "final_test_loss": round(float(aug_test_loss), 4),
        "checkpoint": aug_ckpt_path,
        "target_achieved": bool(aug_test_acc >= 85.0)
    },
    "baseline_model": {
        "best_val_accuracy": best_val_acc_base,
        "final_test_accuracy": round(float(base_test_acc), 2),
        "final_test_loss": round(float(base_test_loss), 4),
        "checkpoint": checkpoint_path_base,
        "target_achieved": bool(base_test_acc >= 85.0)
    },
    "augmentation_gain_percentage": round(float(aug_test_acc - base_test_acc), 2)
}

final_summary_path = os.path.join(RESULTS_DIR, "final_summary.json")
with open(final_summary_path, "w", encoding="utf-8") as f:
    json.dump(summary_results, f, indent=2)
print(f"Saved summary metrics to: {final_summary_path}")

print("\n" + "=" * 70)
print("     ALL TASKS 1, 2, AND 3 COMPLETED SUCCESSFULLY!      ")
print("=" * 70)

