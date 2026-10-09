# CNN-Based Image Classification using CIFAR-10

[![Python 3.14](https://img.shields.io/badge/Python-3.14.6-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14.1%2Bcu126-orange.svg)](https://pytorch.org/)
[![CUDA 12.6](https://img.shields.io/badge/CUDA-12.6%20(RTX%202050)-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-87.21%25-brightgreen.svg)]()
[![College Project](https://img.shields.io/badge/College%20Project-Deep%20Learning-blueviolet.svg)]()

An end-to-end Deep Learning college project built with **PyTorch** and **torchvision** to classify natural images from the CIFAR-10 dataset into 10 categories, achieving **87.21% actual test accuracy** (exceeding the $\ge 85\%$ target) on an **NVIDIA GeForce RTX 2050 GPU** using Automatic Mixed Precision (AMP).

---

## 📌 Project Overview

- **Dataset**: CIFAR-10 (60,000 $32 \times 32$ color RGB images across 10 classes)
- **Deep Learning Framework**: PyTorch with CUDA 12.6 acceleration
- **Hardware**: NVIDIA GeForce RTX 2050 Laptop GPU (4 GB VRAM)
- **Architecture**: Modular 6-Layer Convolutional Neural Network with Batch Normalization and Progressive Dropout
- **Data Segregation Protocol**:
  - **Training Set**: 45,000 images
  - **Validation Set**: 5,000 images
  - **Held-Out Test Set**: 10,000 official benchmark images (completely untouched during training and tuning)
- **Primary Goal**: Achieve at least **85.0% actual test accuracy** based entirely on real model execution.
- **Scientific Ablation**: Direct empirical comparison between training **With Data Augmentation** vs. **Without Data Augmentation (Baseline)**.

---

## 🏆 Official Experimental Results

All reported metrics are computed from real model execution on the official 10,000 held-out test images.

### 1. Model Performance Summary

| Metric | Proposed Model (With Augmentation) | Baseline Model (No Augmentation) | Impact of Augmentation |
| :--- | :---: | :---: | :---: |
| **Best Validation Accuracy** | **87.14%** (Epoch 20) | 87.32% (Epoch 17) | Regularized convergence |
| **Final Test Accuracy (10,000 images)** | **87.21%** | **85.76%** | **+1.45% Accuracy Gain** |
| **Final Test Loss (Cross-Entropy)** | **0.3810** | **0.4442** | **-0.0632 (Better Calibration)** |
| **Final Training Accuracy** | 83.32% | 92.74% | Prevents Overfitting |
| **Generalization Gap (Train vs Test)** | **+3.89% (Well-Regularized)** | **-6.98% (Severe Overfitting)** | Eliminates Memorization |
| **Target Threshold ($\ge 85.0\%$)** | **ACHIEVED (87.21%)** | ACHIEVED (85.76%) | **Goal Surpassed** |
| **Training Duration (20 Epochs)** | 9.58 minutes | 6.56 minutes | Safe Mixed Precision (AMP) |
| **Peak GPU VRAM Usage** | **66 MB** | **66 MB** | Far below 4,096 MB VRAM |

---

### 2. Per-Class Test Performance (Proposed Model)

Evaluated on all 1,000 test images per category (total 10,000 images):

| Class | Precision | Recall | F1-Score | Key Observation |
| :--- | :---: | :---: | :---: | :--- |
| **Automobile** | 94.09% | **95.50%** | **94.79%** | Outstanding detection; distinct vehicle contours |
| **Ship** | 91.30% | **94.40%** | **92.82%** | High recall due to prominent blue water backgrounds |
| **Truck** | 90.95% | **93.50%** | **92.21%** | Strong separation; minimal confusion with cars |
| **Frog** | 84.68% | **93.40%** | **88.83%** | Very high recall; distinct green skin texture |
| **Horse** | 91.48% | **90.20%** | **90.84%** | Excellent quadruped shape recognition |
| **Airplane** | 87.94% | **89.00%** | **88.47%** | Strong detection against sky backgrounds |
| **Deer** | 86.53% | **88.00%** | **87.26%** | Good silhouette and antler feature capture |
| **Dog** | 77.91% | **82.90%** | **80.33%** | Good accuracy; minor overlap with cats |
| **Bird** | **87.20%** | 77.00% | **81.78%** | Very high precision; lower recall on small silhouettes |
| **Cat** | 79.49% | 68.20% | 73.41% | Hardest class; natural visual overlap with dogs |
| **Macro Average** | **87.16%** | **87.21%** | **87.07%** | **Balanced across all 10 categories** |

---

## 🔬 CNN Model Architecture

The custom network (`CIFAR10_CNN`) contains **3,230,250 trainable parameters** organized into three convolutional blocks followed by a fully connected classifier:

```text
Input (3 x 32 x 32 RGB)
  │
  ├── [Conv Block 1]
  │     Conv2d(3 -> 32, 3x3, pad 1) -> BatchNorm2d -> ReLU
  │     Conv2d(32 -> 64, 3x3, pad 1) -> BatchNorm2d -> ReLU
  │     MaxPool2d(2x2) -> Dropout2d(0.25)          ==> [64 x 16 x 16]
  │
  ├── [Conv Block 2]
  │     Conv2d(64 -> 128, 3x3, pad 1) -> BatchNorm2d -> ReLU
  │     Conv2d(128 -> 128, 3x3, pad 1) -> BatchNorm2d -> ReLU
  │     MaxPool2d(2x2) -> Dropout2d(0.30)         ==> [128 x 8 x 8]
  │
  ├── [Conv Block 3]
  │     Conv2d(128 -> 256, 3x3, pad 1) -> BatchNorm2d -> ReLU
  │     Conv2d(256 -> 256, 3x3, pad 1) -> BatchNorm2d -> ReLU
  │     MaxPool2d(2x2) -> Dropout2d(0.40)         ==> [256 x 4 x 4]
  │
  └── [Dense Classifier Head]
        Flatten (4096 features)
        Linear(4096 -> 512) -> BatchNorm1d -> ReLU -> Dropout(0.50)
        Linear(512 -> 10)                          ==> [10 Class Logits]
```

### Key Architectural Rationale:
1. **$3 \times 3$ Convolutions**: Stacking two $3 \times 3$ convolutions creates an effective $5 \times 5$ receptive field while using fewer parameters and incorporating extra non-linearities.
2. **Batch Normalization**: Applied after every convolutional and linear layer to normalize feature distributions, preventing internal covariate shift and speeding up convergence.
3. **Progressive Spatial Dropout**: Dropout rates increase progressively ($0.25 \rightarrow 0.30 \rightarrow 0.40 \rightarrow 0.50$) through the network, forcing feature maps to avoid co-adapting.
4. **AdamW + Cosine Annealing**: The `AdamW` optimizer decouples weight decay from gradient updates, and `CosineAnnealingLR` smoothly decays the learning rate to fine-tune weights in final epochs.

---

## 📈 Visualizations & Artifacts

All plots and checkpoints are saved in the [`results/`](results/) folder:

| File Name | Description |
| :--- | :--- |
| `results/augmented_model_curves.png` | Training vs. Validation Loss & Accuracy curves over 20 epochs for the proposed model. |
| `results/augmentation_comparison.png` | Comparative curves illustrating the reduction in overfitting achieved by data augmentation. |
| `results/confusion_matrix.png` | $10 \times 10$ Confusion Matrix heatmap across all 10,000 test predictions. |
| `results/misclassified_examples.png` | Visual grid of 16 sample test images where the model made errors, displaying true vs. predicted labels. |
| `results/best_model_augmented.pth` | PyTorch checkpoint dictionary storing the best model weights (**87.21% test accuracy**). |
| `results/best_model_baseline.pth` | PyTorch checkpoint dictionary for the baseline model. |
| `results/classification_report.json` | Detailed per-class precision, recall, and F1 metrics in structured JSON format. |
| `results/final_summary.json` | Machine-readable comparison summary of all experiment results. |

---

## 🚀 How to Run the Project

### 1. Prerequisites & Environment
Ensure Python 3.12–3.14 and an NVIDIA GPU (or CPU) are available:
```powershell
cd D:\CNN-CIFAR10-Project
.\.venv\Scripts\Activate.ps1
```

### 2. Launch Interactive Jupyter Notebook
```powershell
jupyter notebook CNN_CIFAR10.ipynb
```
Select the **Python 3 (ipykernel)** kernel. The notebook contains all 18 sections with interactive Markdown explanations and pre-rendered outputs.

### 3. Running Smoke Tests or Standalone Scripts
- **Smoke Tests**: `python scripts/smoke_test.py`
- **Validation Suite**: `python scripts/validate_project.py`

---

## 💡 Key College Project Conclusions

1. **Impact of Data Augmentation**: Training on non-augmented data led to memorization (92.74% training accuracy vs. 85.76% test accuracy, a 6.98% drop). Introducing random crops and horizontal flips regularized the network, producing **87.21% test accuracy** and a **1.45% net gain**.
2. **Hardware Efficiency**: Safe Automatic Mixed Precision (AMP) allowed a deep 3.2M-parameter CNN to train in under 10 minutes on a laptop GPU (RTX 2050) using only **66 MB of VRAM**.
3. **Error Analysis**: Misclassifications largely mirror human perceptual ambiguity at $32 \times 32$ resolution—most notably between cats and dogs, and between birds and planes. Vehicles (automobiles, ships, trucks) achieved $>92\%$ F1-scores.
