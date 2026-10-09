"""
=============================================================================
CNN CIFAR-10 Image Classification - Interactive Web Application
Built with Streamlit & PyTorch
=============================================================================
"""

import os
import sys
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import streamlit as st

# ---------------------------------------------------------------------------
# 1. Page Configuration & Theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CIFAR-10 Image Classifier | PyTorch CNN",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Navy & Blue Theme
st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background-color: #0b132b;
        color: #f0f4f8;
    }
    
    /* Main card container */
    .metric-card {
        background: linear-gradient(135deg, #1c2541 0%, #1f2d52 100%);
        border: 1px solid #2b3a67;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    
    /* Top prediction highlight card */
    .prediction-card {
        background: linear-gradient(135deg, #1b2a4a 0%, #203a70 100%);
        border: 2px solid #3a86ff;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 24px rgba(58, 134, 255, 0.25);
        margin-bottom: 1.5rem;
    }
    
    .prediction-title {
        color: #8da9c4;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    
    .prediction-class {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .prediction-confidence {
        color: #00e5ff;
        font-size: 1.4rem;
        font-weight: 700;
    }
    
    /* Notice info box */
    .resolution-notice {
        background-color: rgba(30, 41, 59, 0.85);
        border-left: 4px solid #3a86ff;
        border-radius: 6px;
        padding: 0.85rem 1rem;
        font-size: 0.88rem;
        color: #cbd5e1;
        margin-bottom: 1.25rem;
        line-height: 1.45;
    }
    
    /* Class probability row */
    .prob-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.35rem;
        font-size: 0.92rem;
    }
    
    /* Custom headers */
    h1, h2, h3 {
        color: #ffffff !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #111a33;
        border-right: 1px solid #1c2541;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 2. CIFAR-10 Constants & Exact Class Ordering
# ---------------------------------------------------------------------------
CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)

CLASS_ICONS = {
    "airplane": "✈️",
    "automobile": "🚗",
    "bird": "🐦",
    "cat": "🐱",
    "deer": "🦌",
    "dog": "🐶",
    "frog": "🐸",
    "horse": "🐴",
    "ship": "🚢",
    "truck": "🚚"
}

# Exact normalization parameters used during model training
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# Exact inference preprocessing pipeline
inference_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
])


# ---------------------------------------------------------------------------
# 3. Exact CNN Model Architecture (Matches Checkpoint Weights)
# ---------------------------------------------------------------------------
class CIFAR10_CNN(nn.Module):
    """
    6-Layer Convolutional Neural Network with Batch Normalization,
    Progressive Spatial Dropout, and Dense Classifier Head.
    Exact architecture trained in CNN_CIFAR10.ipynb and train_augmented.py.
    """
    def __init__(self, num_classes: int = 10):
        super(CIFAR10_CNN, self).__init__()
        # Conv Block 1: 3 -> 32 -> 64 filters
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
        # Conv Block 2: 64 -> 128 -> 128 filters
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
        # Conv Block 3: 128 -> 256 -> 256 filters
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
        # Classifier Head: 4096 -> 512 -> 10
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.50),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        return self.classifier(out)


# ---------------------------------------------------------------------------
# 4. Cached Model Loader with CPU Fallback
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_trained_model(checkpoint_path: str = "results/best_model_augmented.pth") -> Tuple[Optional[nn.Module], torch.device, Optional[str], Dict[str, Any]]:
    """
    Loads model weights once using Streamlit resource caching.
    Gracefully falls back to CPU if CUDA is not available.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata: Dict[str, Any] = {}

    if not os.path.exists(checkpoint_path):
        error_msg = f"Model checkpoint not found at '{checkpoint_path}'. Please verify the path."
        return None, device, error_msg, metadata

    try:
        model = CIFAR10_CNN(num_classes=10).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            metadata["epoch"] = checkpoint.get("epoch", 20)
            metadata["val_acc"] = checkpoint.get("val_acc", 87.14)
            metadata["val_loss"] = checkpoint.get("val_loss", 0.4034)
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        return model, device, None, metadata
    except Exception as e:
        return None, device, f"Error loading model weights: {str(e)}", metadata


# ---------------------------------------------------------------------------
# 5. Image Preprocessing & Prediction Helper
# ---------------------------------------------------------------------------
def run_inference(model: nn.Module, image: Image.Image, device: torch.device) -> Tuple[int, float, np.ndarray, Image.Image]:
    """
    Converts image to RGB, resizes to 32x32, normalizes with CIFAR-10 constants,
    and returns predicted class index, confidence, full probability distribution,
    and the 32x32 downscaled PIL image for user inspection.
    """
    rgb_img = image.convert("RGB")
    downscaled_32 = rgb_img.resize((32, 32), Image.Resampling.BILINEAR)

    # Tensor shape [1, 3, 32, 32]
    tensor = inference_transform(rgb_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    predicted_idx = int(np.argmax(probs))
    confidence = float(probs[predicted_idx])

    return predicted_idx, confidence, probs, downscaled_32


# ---------------------------------------------------------------------------
# 6. Sidebar Controls & Model Metadata
# ---------------------------------------------------------------------------
def render_sidebar(device: torch.device, metadata: Dict[str, Any]):
    with st.sidebar:
        st.markdown("## ⚙️ Model Information")
        
        # Device badge
        if device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            st.success(f"🟢 **Hardware**: GPU Acceleration\n\n*{gpu_name}*")
        else:
            st.info("⚪ **Hardware**: Running on CPU")

        # Official metrics summary
        st.markdown("### 📊 Benchmark Metrics")
        st.markdown("""
        - **Model Architecture**: 6-Layer Deep CNN
        - **Total Parameters**: 3,230,250
        - **Final Test Accuracy**: **87.21%**
        - **Best Val Accuracy**: **87.14%**
        - **Test Loss**: 0.3810
        - **Dataset**: CIFAR-10 (10 Classes)
        """)

        st.markdown("---")
        st.markdown("### 🏷️ 10 Target Categories")
        for cls_name, icon in CLASS_ICONS.items():
            st.markdown(f"- {icon} **{cls_name.capitalize()}**")

        st.markdown("---")
        st.caption("College Deep Learning Project | PyTorch & Streamlit")


# ---------------------------------------------------------------------------
# 7. Main Application UI
# ---------------------------------------------------------------------------
def main():
    # Model Loading
    checkpoint_file = os.path.join("results", "best_model_augmented.pth")
    model, device, load_error, metadata = load_trained_model(checkpoint_file)

    render_sidebar(device, metadata)

    # Header section
    st.markdown("""
        <div style="text-align: left; margin-bottom: 1.5rem;">
            <h1 style="margin-bottom: 0.2rem; font-size: 2.3rem;">
                🧠 CNN Image Classifier <span style="color: #3a86ff;">(CIFAR-10)</span>
            </h1>
            <p style="color: #8da9c4; font-size: 1.05rem; margin-top: 0;">
                Trained with PyTorch, Automatic Mixed Precision (AMP), and Data Augmentation • <strong>87.21% Test Accuracy</strong>
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Resolution Notice
    st.markdown("""
        <div class="resolution-notice">
            <strong>⚠️ Important Note on Image Resolution:</strong><br>
            The CIFAR-10 benchmark consists strictly of low-resolution <strong>32×32 pixel</strong> thumbnail images. 
            When uploading ordinary high-resolution photos, the image is automatically resized down to 32×32 pixels. 
            For best prediction accuracy, ensure the subject is prominently centered, well-lit, and matches one of the 10 target categories.
        </div>
    """, unsafe_allow_html=True)

    # Check for loading errors
    if load_error is not None:
        st.error(f"❌ **Failed to initialize model**: {load_error}")
        st.info("Ensure `results/best_model_augmented.pth` is present in the project directory.")
        return

    # Image Input Selection
    st.markdown("### 📤 Choose an Image to Classify")
    
    input_mode = st.radio(
        "Select Image Source:",
        options=["Upload Custom Image", "Try Sample CIFAR-10 Image"],
        horizontal=True
    )

    selected_image: Optional[Image.Image] = None
    image_caption: str = ""

    samples_dir = os.path.join("assets", "samples")

    if input_mode == "Upload Custom Image":
        uploaded_file = st.file_uploader(
            "Upload an image (JPG, JPEG, or PNG):",
            type=["jpg", "jpeg", "png"],
            help="Select an image featuring one of the 10 CIFAR-10 classes."
        )
        if uploaded_file is not None:
            try:
                selected_image = Image.open(uploaded_file)
                image_caption = f"Uploaded File: {uploaded_file.name}"
            except Exception as e:
                st.error(f"❌ Could not decode uploaded file: {str(e)}. Please try another valid image.")
                return

    else:
        # Sample images mode
        if os.path.exists(samples_dir):
            sample_files = [f for f in os.listdir(samples_dir) if f.endswith(".png")]
            sample_classes = [os.path.splitext(f)[0] for f in sample_files]
            
            selected_sample_class = st.selectbox(
                "Choose a sample test class:",
                options=sample_classes,
                format_func=lambda c: f"{CLASS_ICONS.get(c, '📷')} {c.capitalize()}"
            )
            
            if selected_sample_class:
                sample_path = os.path.join(samples_dir, f"{selected_sample_class}.png")
                try:
                    selected_image = Image.open(sample_path)
                    image_caption = f"CIFAR-10 Sample: {selected_sample_class.capitalize()}"
                except Exception as e:
                    st.error(f"Error loading sample image: {e}")
        else:
            st.warning("Sample images folder not found. Please upload a custom image.")

    # Prediction & Visual Display
    if selected_image is not None:
        st.markdown("---")
        
        with st.spinner("Analyzing image through 6-Layer CNN..."):
            try:
                pred_idx, confidence, probs, downscaled_32 = run_inference(model, selected_image, device)
            except Exception as e:
                st.error(f"❌ Error during model inference: {str(e)}")
                return

        pred_class = CLASSES[pred_idx]
        pred_icon = CLASS_ICONS.get(pred_class, "🎯")

        # Layout: 2 Columns (Left: Images, Right: Prediction & Probabilities)
        col_img, col_preds = st.columns([1, 1.4], gap="large")

        with col_img:
            st.markdown("#### 🖼️ Image Preview")
            st.image(selected_image, caption=image_caption, use_container_width=True)
            
            # Show downscaled 32x32 view
            st.markdown("##### 🔍 Model View (Resized to 32×32)")
            st.image(downscaled_32, caption="What the CNN Actually Analyzes (32x32 px)", width=128)
            st.caption(f"Original Resolution: {selected_image.size[0]} × {selected_image.size[1]} px")

        with col_preds:
            # Highlight Card for Top Prediction
            st.markdown(f"""
                <div class="prediction-card">
                    <div class="prediction-title">Top Predicted Category</div>
                    <div class="prediction-class">{pred_icon} {pred_class.capitalize()}</div>
                    <div class="prediction-confidence">{confidence * 100:.2f}% Confidence</div>
                </div>
            """, unsafe_allow_html=True)

            # Class Probability Distribution
            st.markdown("#### 📊 All 10 Class Probabilities")
            
            # Prepare DataFrame sorted by probability descending
            prob_data = [
                {
                    "Icon": CLASS_ICONS.get(cls_name, ""),
                    "Class": cls_name.capitalize(),
                    "Probability (%)": round(probs[i] * 100, 2),
                    "Raw Prob": probs[i]
                }
                for i, cls_name in enumerate(CLASSES)
            ]
            prob_df = pd.DataFrame(prob_data).sort_values(by="Raw Prob", ascending=False).reset_index(drop=True)

            # Display progress bars
            for _, row in prob_df.iterrows():
                is_winner = (row["Class"].lower() == pred_class)
                label_color = "#00e5ff" if is_winner else "#f0f4f8"
                weight = "bold" if is_winner else "normal"
                
                st.markdown(
                    f"<div class='prob-row'>"
                    f"<span style='color: {label_color}; font-weight: {weight};'>{row['Icon']} {row['Class']}</span>"
                    f"<span style='color: {label_color}; font-weight: {weight};'>{row['Probability (%)']:.2f}%</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                # Streamlit progress bar (clamped 0.0 to 1.0)
                st.progress(min(max(float(row["Raw Prob"]), 0.0), 1.0))

            # Expandable interactive chart view
            with st.expander("📈 View Interactive Probability Chart"):
                chart_df = prob_df.rename(columns={"Class": "Category", "Probability (%)": "Confidence"})
                st.bar_chart(chart_df.set_index("Category")["Confidence"], color="#3a86ff")

    else:
        st.info("👆 Upload an image or select a sample image above to start classification.")

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 1rem 0;">
            CNN-CIFAR10 College Deep Learning Project • PyTorch & Streamlit • Trained on NVIDIA GeForce RTX 2050
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

