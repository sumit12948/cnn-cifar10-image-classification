"""
Verification test script for app.py
Tests:
1. Model loading & state_dict verification
2. Preprocessing pipeline
3. Inference on sample images
4. Probability distribution validity
"""

import os
import sys

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, os.path.abspath("."))
from app import CIFAR10_CNN, load_trained_model, run_inference, CLASSES

print("=" * 65)
print("TESTING APP.PY MODEL & INFERENCE PIPELINE")
print("=" * 65)

# 1. Test model loading
checkpoint_path = os.path.join("results", "best_model_augmented.pth")
model, device, err, meta = load_trained_model(checkpoint_path)

if err is not None:
    print(f"FAILED: Model loading error: {err}")
    sys.exit(1)

print(f"PASS: Model loaded successfully on device: {device}")
print(f"      Metadata: {meta}")

# 2. Test inference on synthetic RGB image
synthetic_img = Image.new("RGB", (256, 256), color=(100, 150, 200))
pred_idx, conf, probs, downscaled = run_inference(model, synthetic_img, device)

assert 0 <= pred_idx < 10, f"Invalid predicted index: {pred_idx}"
assert 0.0 <= conf <= 1.0, f"Invalid confidence: {conf}"
assert len(probs) == 10, f"Invalid probability vector length: {len(probs)}"
assert np.isclose(np.sum(probs), 1.0, atol=1e-4), f"Probabilities do not sum to 1.0: {np.sum(probs)}"
assert downscaled.size == (32, 32), f"Downscaled image size is not (32, 32): {downscaled.size}"

print("PASS: Synthetic image inference verified.")

# 3. Test inference on all 10 extracted sample images
samples_dir = os.path.join("assets", "samples")
if os.path.exists(samples_dir):
    print(f"\nEvaluating sample test images in {samples_dir}:")
    correct_count = 0
    total_samples = 0
    for sample_name in sorted(os.listdir(samples_dir)):
        if not sample_name.endswith(".png"):
            continue
        expected_class = os.path.splitext(sample_name)[0]
        sample_path = os.path.join(samples_dir, sample_name)
        img = Image.open(sample_path)
        
        idx, conf, p, _ = run_inference(model, img, device)
        pred_name = CLASSES[idx]
        total_samples += 1
        is_match = (pred_name == expected_class)
        if is_match:
            correct_count += 1
        match_str = "[CORRECT]" if is_match else "[MISCLASSIFIED]"
        print(f"  - Ground Truth: {expected_class:10s} | Predicted: {pred_name:10s} | Conf: {conf*100:5.1f}% {match_str}")
    print(f"\nSample Classification Accuracy: {correct_count}/{total_samples} ({100.0*correct_count/total_samples:.1f}%)")

print("\n" + "=" * 65)
print("ALL APP PIPELINE TESTS PASSED SUCCESSFULLY!")
print("=" * 65)
