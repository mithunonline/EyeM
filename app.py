"""
EyeM — Animal Classifier: CNN vs Vision Transformer
Streamlit app that classifies animal images using two ML models and compares their predictions.
Supports both general ImageNet-1k models and fine-tuned 525 Bird Species models.
"""

import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import EfficientNet_B0_Weights
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import entropy as scipy_entropy
import json
import io
import os
import warnings
warnings.filterwarnings("ignore")

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EyeM — Animal Classifier",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #2196F3, #4CAF50);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #666;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #2196F3;
        margin-bottom: 0.6rem;
    }
    .metric-card.vit {
        border-left-color: #4CAF50;
    }
    .metric-card.compare {
        border-left-color: #FF9800;
    }
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-agree { background: #e8f5e9; color: #2e7d32; }
    .badge-disagree { background: #fce4ec; color: #c62828; }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
CNN_MODEL_ID    = "efficientnet_b0"
VIT_MODEL_ID    = "google/vit-base-patch16-224"
TOP_K           = 10
MODELS_DIR      = os.path.join(os.path.dirname(__file__), "models")
CNN_BIRD_CKPT   = os.path.join(MODELS_DIR, "cnn_birds_finetuned.pth")
VIT_BIRD_CKPT   = os.path.join(MODELS_DIR, "vit_birds_finetuned.pth")
BIRD_LABELS_PATH = os.path.join(MODELS_DIR, "bird_class_names.json")

ANIMAL_CLASS_RANGES = [(0, 397), (407, 411), (665, 668)]
ANIMAL_INDICES = set()
for s, e in ANIMAL_CLASS_RANGES:
    ANIMAL_INDICES.update(range(s, e + 1))

# ─── Model loading (cached) ──────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_cnn_model():
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    labels = weights.meta["categories"]
    return model, preprocess, labels


@st.cache_resource(show_spinner=False)
def load_vit_model():
    processor = ViTImageProcessor.from_pretrained(VIT_MODEL_ID)
    model = ViTForImageClassification.from_pretrained(VIT_MODEL_ID)
    model.eval()
    labels = list(model.config.id2label.values())
    return model, processor, labels


@st.cache_resource(show_spinner=False)
def load_cnn_bird_model(num_classes, class_names):
    """Load fine-tuned EfficientNet-B0 for 525 bird species."""
    import torch.nn as nn
    base = models.efficientnet_b0(weights=None)
    in_features = base.classifier[1].in_features
    base.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 512),
        nn.SiLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes),
    )
    ckpt = torch.load(CNN_BIRD_CKPT, map_location="cpu")
    base.load_state_dict(ckpt["model_state_dict"])
    base.eval()
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return base, preprocess


@st.cache_resource(show_spinner=False)
def load_vit_bird_model(num_classes, class_names):
    """Load fine-tuned ViT-B/16 for 525 bird species."""
    label2id = {c: i for i, c in enumerate(class_names)}
    id2label = {i: c for i, c in enumerate(class_names)}
    model = ViTForImageClassification.from_pretrained(
        VIT_MODEL_ID,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    ckpt = torch.load(VIT_BIRD_CKPT, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    processor = ViTImageProcessor.from_pretrained(VIT_MODEL_ID)
    return model, processor


# ─── Bird model availability ─────────────────────────────────────────────────
BIRD_MODELS_AVAILABLE = (
    os.path.exists(CNN_BIRD_CKPT) and
    os.path.exists(VIT_BIRD_CKPT) and
    os.path.exists(BIRD_LABELS_PATH)
)

if BIRD_MODELS_AVAILABLE:
    with open(BIRD_LABELS_PATH) as f:
        BIRD_CLASS_NAMES = json.load(f)
    NUM_BIRD_CLASSES = len(BIRD_CLASS_NAMES)


# ─── Inference ───────────────────────────────────────────────────────────────
@torch.no_grad()
def run_cnn(image: Image.Image, model, preprocess, labels):
    tensor = preprocess(image.convert("RGB")).unsqueeze(0)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=-1).squeeze().numpy()
    top_idx = probs.argsort()[::-1][:TOP_K]
    return {
        "top": [(labels[i], float(probs[i])) for i in top_idx],
        "all_probs": probs,
        "top1_label": labels[probs.argmax()],
        "top1_conf": float(probs.max()),
        "top1_idx": int(probs.argmax()),
        "is_animal": int(probs.argmax()) in ANIMAL_INDICES,
    }


@torch.no_grad()
def run_vit(image: Image.Image, model, processor, labels):
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze().numpy()
    top_idx = probs.argsort()[::-1][:TOP_K]
    return {
        "top": [(labels[i], float(probs[i])) for i in top_idx],
        "all_probs": probs,
        "top1_label": labels[probs.argmax()],
        "top1_conf": float(probs.max()),
        "top1_idx": int(probs.argmax()),
        "is_animal": int(probs.argmax()) in ANIMAL_INDICES,
    }


@torch.no_grad()
def run_cnn_bird(image: Image.Image, model, preprocess, class_names):
    tensor = preprocess(image.convert("RGB")).unsqueeze(0)
    logits = model(tensor)
    probs  = torch.softmax(logits, dim=-1).squeeze().numpy()
    top_idx = probs.argsort()[::-1][:TOP_K]
    return {
        "top": [(class_names[i], float(probs[i])) for i in top_idx],
        "all_probs": probs,
        "top1_label": class_names[probs.argmax()],
        "top1_conf": float(probs.max()),
    }


@torch.no_grad()
def run_vit_bird(image: Image.Image, model, processor, class_names):
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    logits = model(**inputs).logits
    probs  = torch.softmax(logits, dim=-1).squeeze().numpy()
    top_idx = probs.argsort()[::-1][:TOP_K]
    return {
        "top": [(class_names[i], float(probs[i])) for i in top_idx],
        "all_probs": probs,
        "top1_label": class_names[probs.argmax()],
        "top1_conf": float(probs.max()),
    }


# ─── Comparison metrics ──────────────────────────────────────────────────────
def compute_comparison(cnn_res, vit_res):
    p = cnn_res["all_probs"].astype(np.float64) + 1e-10
    q = vit_res["all_probs"].astype(np.float64) + 1e-10
    p /= p.sum(); q /= q.sum()

    kl_pq = float(scipy_entropy(p, q))
    kl_qp = float(scipy_entropy(q, p))
    js_div = float(0.5 * scipy_entropy(p, 0.5 * (p + q)) +
                   0.5 * scipy_entropy(q, 0.5 * (p + q)))

    cos_sim = float(np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q)))

    cnn_top5 = {lbl for lbl, _ in cnn_res["top"][:5]}
    vit_top5 = {lbl for lbl, _ in vit_res["top"][:5]}
    top5_overlap = len(cnn_top5 & vit_top5)
    top5_jaccard = len(cnn_top5 & vit_top5) / len(cnn_top5 | vit_top5)

    top1_agree = cnn_res["top1_label"].lower() == vit_res["top1_label"].lower()

    conf_gap = abs(cnn_res["top1_conf"] - vit_res["top1_conf"])

    return {
        "kl_cnn_vit": kl_pq,
        "kl_vit_cnn": kl_qp,
        "js_divergence": js_div,
        "cosine_similarity": cos_sim,
        "top5_overlap": top5_overlap,
        "top5_jaccard": top5_jaccard,
        "top1_agree": top1_agree,
        "confidence_gap": conf_gap,
    }


# ─── Plots ───────────────────────────────────────────────────────────────────
def plot_top_k_comparison(cnn_res, vit_res):
    cnn_labels = [r[0] for r in cnn_res["top"]]
    cnn_probs  = [r[1] * 100 for r in cnn_res["top"]]
    vit_labels = [r[0] for r in vit_res["top"]]
    vit_probs  = [r[1] * 100 for r in vit_res["top"]]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "<b>CNN — EfficientNet-B0</b>",
            "<b>ViT — vit-base-patch16-224</b>"
        ],
        horizontal_spacing=0.12,
    )

    fig.add_trace(go.Bar(
        x=cnn_probs[::-1], y=cnn_labels[::-1],
        orientation="h",
        marker_color=["#1565C0" if i == len(cnn_probs) - 1 else "#90CAF9"
                      for i in range(len(cnn_probs))],
        text=[f"{p:.2f}%" for p in cnn_probs[::-1]],
        textposition="outside",
        name="CNN",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=vit_probs[::-1], y=vit_labels[::-1],
        orientation="h",
        marker_color=["#1B5E20" if i == len(vit_probs) - 1 else "#A5D6A7"
                      for i in range(len(vit_probs))],
        text=[f"{p:.2f}%" for p in vit_probs[::-1]],
        textposition="outside",
        name="ViT",
    ), row=1, col=2)

    fig.update_xaxes(title_text="Confidence (%)", row=1, col=1)
    fig.update_xaxes(title_text="Confidence (%)", row=1, col=2)
    fig.update_layout(
        height=420,
        showlegend=False,
        margin=dict(l=10, r=80, t=50, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def plot_probability_overlay(cnn_res, vit_res, top_n=30):
    """Overlay CNN and ViT probability distributions for top-N combined classes."""
    cnn_probs = cnn_res["all_probs"]
    vit_probs = vit_res["all_probs"]

    # Union of top-N indices from each model
    cnn_top_idx = set(cnn_probs.argsort()[::-1][:top_n])
    vit_top_idx = set(vit_probs.argsort()[::-1][:top_n])
    combined_idx = sorted(cnn_top_idx | vit_top_idx,
                          key=lambda i: -(cnn_probs[i] + vit_probs[i]))[:top_n]

    # Use ViT labels (same ImageNet-1k)
    labels_list = []
    for i, model_res in [(0, cnn_res)]:
        pass
    # Get labels from whichever is available
    all_labels = [r[0] for r in cnn_res["top"]] + [r[0] for r in vit_res["top"]]

    # Build from CNN labels list (loaded globally)
    from torchvision.models import EfficientNet_B0_Weights
    _labels = EfficientNet_B0_Weights.IMAGENET1K_V1.meta["categories"]
    idx_labels = [_labels[i] for i in combined_idx]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="CNN (EfficientNet-B0)",
        x=idx_labels,
        y=[cnn_probs[i] * 100 for i in combined_idx],
        marker_color="rgba(33, 150, 243, 0.75)",
    ))
    fig.add_trace(go.Bar(
        name="ViT (vit-base-patch16-224)",
        x=idx_labels,
        y=[vit_probs[i] * 100 for i in combined_idx],
        marker_color="rgba(76, 175, 80, 0.75)",
    ))

    fig.update_layout(
        barmode="group",
        title="<b>Probability Distribution Comparison — Top Combined Classes</b>",
        xaxis_title="Class",
        yaxis_title="Confidence (%)",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(tickangle=-35),
        margin=dict(l=10, r=10, t=60, b=100),
    )
    return fig


def plot_radar(metrics: dict):
    categories = [
        "Cosine Sim", "Top-5 Jaccard",
        "Top-5 Overlap (norm)", "Agreement",
        "Low Conf Gap (inv)"
    ]
    vals = [
        min(metrics["cosine_similarity"], 1.0),
        metrics["top5_jaccard"],
        metrics["top5_overlap"] / 5.0,
        1.0 if metrics["top1_agree"] else 0.0,
        max(0.0, 1.0 - metrics["confidence_gap"]),
    ]

    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(255, 152, 0, 0.25)",
        line=dict(color="#FF9800", width=2),
        name="Model Agreement",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="<b>Model Agreement Radar</b>",
        height=350,
        margin=dict(l=40, r=40, t=60, b=40),
        paper_bgcolor="white",
    )
    return fig


def plot_diff_heatmap(cnn_res, vit_res, top_n=20):
    """Bar chart showing prediction probability differences."""
    cnn_p = cnn_res["all_probs"]
    vit_p = vit_res["all_probs"]
    diff = cnn_p - vit_p

    top_diff_idx = np.abs(diff).argsort()[::-1][:top_n]
    from torchvision.models import EfficientNet_B0_Weights
    _labels = EfficientNet_B0_Weights.IMAGENET1K_V1.meta["categories"]
    idx_labels = [_labels[i] for i in top_diff_idx]
    diff_vals = [diff[i] * 100 for i in top_diff_idx]
    colors = ["#1565C0" if d > 0 else "#2E7D32" for d in diff_vals]

    fig = go.Figure(go.Bar(
        x=idx_labels,
        y=diff_vals,
        marker_color=colors,
        text=[f"{d:+.2f}%" for d in diff_vals],
        textposition="outside",
    ))
    fig.update_layout(
        title="<b>Prediction Difference (CNN − ViT) for Top Divergent Classes</b>",
        xaxis_title="Class",
        yaxis_title="Probability Difference (%)",
        height=380,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(tickangle=-35),
        margin=dict(l=10, r=10, t=60, b=120),
        shapes=[dict(type="line", x0=-0.5, x1=len(idx_labels) - 0.5,
                     y0=0, y1=0, line=dict(color="gray", width=1, dash="dot"))],
    )
    blue_patch = dict(x=0, y=0, text="Blue = CNN higher", showarrow=False,
                      xref="paper", yref="paper", font=dict(color="#1565C0", size=11))
    green_patch = dict(x=1, y=0, text="Green = ViT higher", showarrow=False,
                       xref="paper", yref="paper", xanchor="right",
                       font=dict(color="#2E7D32", size=11))
    fig.update_layout(annotations=[blue_patch, green_patch])
    return fig


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Settings")

    # ── Mode selector ─────────────────────────────────────────────────────────
    mode_options = ["General Animal Classifier (ImageNet-1k)"]
    if BIRD_MODELS_AVAILABLE:
        mode_options.append("Bird Specialist (525 Species — Fine-tuned)")
    else:
        mode_options.append("Bird Specialist — Not available (train first)")

    mode = st.radio(
        "Classification Mode",
        options=mode_options,
        index=0,
        help="Bird Specialist uses models fine-tuned on 89,885 bird images from 525 species.",
    )
    BIRD_MODE = "Bird Specialist" in mode and BIRD_MODELS_AVAILABLE

    if not BIRD_MODELS_AVAILABLE:
        st.info(
            "Bird Specialist models not found.\n\n"
            "Train them on Google Colab:\n"
            "- `notebooks/Train_CNN_Birds_Colab.ipynb`\n"
            "- `notebooks/Train_ViT_Birds_Colab.ipynb`\n\n"
            "Then place `cnn_birds_finetuned.pth`, `vit_birds_finetuned.pth`, "
            "and `bird_class_names.json` into the `models/` folder.",
            icon="🐦",
        )

    st.markdown("---")
    show_gradcam = st.checkbox("Show GradCAM (CNN)", value=False,
                               help="Compute GradCAM saliency map for CNN prediction")
    show_attn = st.checkbox("Show Attention Map (ViT)", value=False,
                            help="Extract ViT attention from last transformer layer")
    top_k_show = st.slider("Top-K predictions to show", 3, 10, 5)

    st.markdown("---")
    st.markdown("### About the Models")
    if BIRD_MODE:
        st.markdown(f"""
**CNN — EfficientNet-B0 (Bird Fine-tuned)**
- Fine-tuned on {NUM_BIRD_CLASSES} bird species
- ~89,885 training images
- Transfer learning from ImageNet-1k

**ViT — vit-base-patch16-224 (Bird Fine-tuned)**
- Fine-tuned on {NUM_BIRD_CLASSES} bird species
- ~89,885 training images
- Transfer learning from ImageNet-21k
""")
    else:
        st.markdown("""
**CNN — EfficientNet-B0**
- ~5.3M parameters
- MBConv blocks + Squeeze-Excitation
- Pretrained: ImageNet-1k
- Top-1 accuracy: 77.7%

**ViT — vit-base-patch16-224**
- ~86M parameters
- 12-layer Transformer
- 16×16 patch tokenization
- Pretrained: ImageNet-21k → 1k
- Top-1 accuracy: 81.8%
""")
    st.markdown("---")
    st.markdown("### Metrics Explained")
    st.markdown("""
- **KL Divergence** — How different the probability distributions are (lower = more similar)
- **JS Divergence** — Symmetric version of KL (0 = identical, 1 = completely different)
- **Cosine Similarity** — Direction similarity of probability vectors (1 = identical)
- **Top-5 Jaccard** — Overlap of top-5 class predictions (1 = same 5 classes)
""")

# ─── Header ──────────────────────────────────────────────────────────────────
if BIRD_MODE:
    st.markdown('<div class="main-title">EyeM — Bird Classifier</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="subtitle">CNN vs ViT fine-tuned on <b>{NUM_BIRD_CLASSES} bird species</b> '
        f'(~89,885 training images) — side-by-side comparison</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown('<div class="main-title">EyeM — Animal Classifier</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">CNN (EfficientNet-B0) vs Vision Transformer (ViT-B/16) '
        '— side-by-side prediction comparison</div>',
        unsafe_allow_html=True,
    )

# ─── Load models ─────────────────────────────────────────────────────────────
if BIRD_MODE:
    with st.spinner("Loading fine-tuned CNN Bird model..."):
        cnn_bird_model, cnn_bird_preprocess = load_cnn_bird_model(NUM_BIRD_CLASSES, BIRD_CLASS_NAMES)
    with st.spinner("Loading fine-tuned ViT Bird model..."):
        vit_bird_model, vit_bird_processor = load_vit_bird_model(NUM_BIRD_CLASSES, BIRD_CLASS_NAMES)
    st.success(f"Bird Specialist models loaded — {NUM_BIRD_CLASSES} species.", icon="🐦")
else:
    with st.spinner("Loading CNN model (EfficientNet-B0)..."):
        cnn_model, cnn_preprocess, cnn_labels = load_cnn_model()
    with st.spinner("Loading ViT model (vit-base-patch16-224)..."):
        vit_model, vit_processor, vit_labels = load_vit_model()
    st.success("Both models loaded and ready.", icon="✅")

# ─── Image upload ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Upload an Animal Image")

col_upload, col_preview = st.columns([1, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Supported formats: JPG, PNG, WEBP",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )
    st.caption("Upload a clear photo of an animal for best results.")

with col_preview:
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=f"Uploaded: {uploaded_file.name} ({image.size[0]}×{image.size[1]}px)",
                 use_container_width=True)

# ─── Inference & Results ──────────────────────────────────────────────────────
if uploaded_file is not None:
    st.markdown("---")

    with st.spinner("Running inference..."):
        if BIRD_MODE:
            cnn_result = run_cnn_bird(image, cnn_bird_model, cnn_bird_preprocess, BIRD_CLASS_NAMES)
            vit_result = run_vit_bird(image, vit_bird_model, vit_bird_processor, BIRD_CLASS_NAMES)
        else:
            cnn_result = run_cnn(image, cnn_model, cnn_preprocess, cnn_labels)
            vit_result = run_vit(image, vit_model, vit_processor, vit_labels)
        metrics = compute_comparison(cnn_result, vit_result)

    # ── Top predictions side by side ──────────────────────────────────────────
    st.markdown("### Predictions")
    col_cnn, col_vit = st.columns(2)

    with col_cnn:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f"**CNN — EfficientNet-B0**")
        st.markdown(f"**Top prediction:** `{cnn_result['top1_label']}`")
        st.progress(cnn_result["top1_conf"],
                    text=f"Confidence: {cnn_result['top1_conf']*100:.1f}%")
        if BIRD_MODE:
            st.markdown('<span class="badge badge-agree">Bird Specialist Model</span>', unsafe_allow_html=True)
        else:
            animal_tag = "Animal" if cnn_result.get("is_animal") else "Non-animal"
            tag_color  = "badge-agree" if cnn_result.get("is_animal") else "badge-disagree"
            st.markdown(f'<span class="badge {tag_color}">{animal_tag} class</span>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("**Top predictions:**")
        cnn_df = pd.DataFrame(
            [(lbl, f"{p*100:.2f}%") for lbl, p in cnn_result["top"][:top_k_show]],
            columns=["Class", "Confidence"]
        )
        st.dataframe(cnn_df, hide_index=True, use_container_width=True)

    with col_vit:
        st.markdown('<div class="metric-card vit">', unsafe_allow_html=True)
        st.markdown(f"**ViT — vit-base-patch16-224**")
        st.markdown(f"**Top prediction:** `{vit_result['top1_label']}`")
        st.progress(vit_result["top1_conf"],
                    text=f"Confidence: {vit_result['top1_conf']*100:.1f}%")
        if BIRD_MODE:
            st.markdown('<span class="badge badge-agree">Bird Specialist Model</span>', unsafe_allow_html=True)
        else:
            animal_tag = "Animal" if vit_result.get("is_animal") else "Non-animal"
            tag_color  = "badge-agree" if vit_result.get("is_animal") else "badge-disagree"
            st.markdown(f'<span class="badge {tag_color}">{animal_tag} class</span>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("**Top predictions:**")
        vit_df = pd.DataFrame(
            [(lbl, f"{p*100:.2f}%") for lbl, p in vit_result["top"][:top_k_show]],
            columns=["Class", "Confidence"]
        )
        st.dataframe(vit_df, hide_index=True, use_container_width=True)

    # ── Prediction bar charts ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Prediction Comparison Charts")
    st.plotly_chart(plot_top_k_comparison(cnn_result, vit_result), use_container_width=True)

    # ── Comparison metrics ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Model Agreement Metrics")

    agree_badge = (
        '<span class="badge badge-agree">Models AGREE on top-1</span>'
        if metrics["top1_agree"] else
        '<span class="badge badge-disagree">Models DISAGREE on top-1</span>'
    )
    st.markdown(agree_badge, unsafe_allow_html=True)
    st.markdown("")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("JS Divergence", f"{metrics['js_divergence']:.4f}",
                  help="0 = identical distributions, 1 = completely different")
    with m2:
        st.metric("Cosine Similarity", f"{metrics['cosine_similarity']:.4f}",
                  help="1 = same direction, 0 = orthogonal")
    with m3:
        st.metric("Top-5 Jaccard", f"{metrics['top5_jaccard']:.2f}",
                  help="Overlap of top-5 predictions (1 = same 5 classes)")
    with m4:
        st.metric("Confidence Gap", f"{metrics['confidence_gap']*100:.1f}%",
                  help="Absolute difference in top-1 confidence scores")

    m5, m6 = st.columns(2)
    with m5:
        st.metric("KL(CNN ∥ ViT)", f"{metrics['kl_cnn_vit']:.4f}",
                  help="KL divergence from CNN to ViT distribution")
    with m6:
        st.metric("KL(ViT ∥ CNN)", f"{metrics['kl_vit_cnn']:.4f}",
                  help="KL divergence from ViT to CNN distribution")

    # Detailed metrics table
    with st.expander("Full metrics table"):
        metrics_df = pd.DataFrame([
            ("KL Divergence (CNN → ViT)", f"{metrics['kl_cnn_vit']:.6f}", "Lower = distributions more similar"),
            ("KL Divergence (ViT → CNN)", f"{metrics['kl_vit_cnn']:.6f}", "Lower = distributions more similar"),
            ("JS Divergence",             f"{metrics['js_divergence']:.6f}", "Symmetric. 0=same, 1=different"),
            ("Cosine Similarity",         f"{metrics['cosine_similarity']:.6f}", "Higher = more similar"),
            ("Top-5 Overlap",             str(metrics["top5_overlap"]) + " / 5", "Shared classes in top-5"),
            ("Top-5 Jaccard Index",       f"{metrics['top5_jaccard']:.4f}", "Intersection / Union of top-5"),
            ("Top-1 Agreement",           "Yes" if metrics["top1_agree"] else "No", "Same top prediction?"),
            ("Confidence Gap",            f"{metrics['confidence_gap']*100:.2f}%", "Absolute diff in confidence"),
        ], columns=["Metric", "Value", "Interpretation"])
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)

    # ── Distribution overlay ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Probability Distribution Overlay")
    st.plotly_chart(plot_probability_overlay(cnn_result, vit_result), use_container_width=True)

    # ── Difference chart ──────────────────────────────────────────────────────
    st.markdown("### Prediction Difference Breakdown")
    st.caption("Classes where CNN and ViT differ most in their assigned probabilities.")
    st.plotly_chart(plot_diff_heatmap(cnn_result, vit_result), use_container_width=True)

    # ── Radar chart ───────────────────────────────────────────────────────────
    col_radar, col_summary = st.columns([1, 1])
    with col_radar:
        st.markdown("### Agreement Radar")
        st.plotly_chart(plot_radar(metrics), use_container_width=True)

    with col_summary:
        st.markdown("### Summary Interpretation")
        st.markdown("---")
        cnn_conf_pct = cnn_result["top1_conf"] * 100
        vit_conf_pct = vit_result["top1_conf"] * 100
        more_confident = "CNN" if cnn_conf_pct > vit_conf_pct else "ViT"

        if metrics["top1_agree"]:
            st.success(f"Both models agree: **{cnn_result['top1_label']}**")
        else:
            st.warning(f"CNN predicts **{cnn_result['top1_label']}**, ViT predicts **{vit_result['top1_label']}**")

        st.markdown(f"- **{more_confident}** is more confident ({max(cnn_conf_pct, vit_conf_pct):.1f}% vs {min(cnn_conf_pct, vit_conf_pct):.1f}%)")
        st.markdown(f"- Top-5 prediction overlap: **{metrics['top5_overlap']}/5 classes**")

        if metrics["js_divergence"] < 0.05:
            st.markdown("- Distribution similarity: **Very high** — models largely agree")
        elif metrics["js_divergence"] < 0.15:
            st.markdown("- Distribution similarity: **Moderate** — some divergence")
        else:
            st.markdown("- Distribution similarity: **Low** — models see the image differently")

        if metrics["cosine_similarity"] > 0.9:
            st.markdown("- Probability vectors are **nearly parallel** (cosine ≈ 1)")
        elif metrics["cosine_similarity"] > 0.7:
            st.markdown("- Probability vectors have **moderate alignment**")
        else:
            st.markdown("- Probability vectors are **divergent** — fundamentally different outputs")

    # ── Optional: GradCAM & Attention maps ────────────────────────────────────
    if show_gradcam or show_attn:
        st.markdown("---")
        st.markdown("### Interpretability Maps")
        interp_cols = st.columns(2 if (show_gradcam and show_attn) else 1)

        if show_gradcam:
            import torch.nn.functional as F

            col = interp_cols[0] if show_attn else interp_cols[0]
            with col:
                with st.spinner("Computing GradCAM..."):
                    try:
                        import matplotlib.pyplot as plt

                        cnn_model.eval()
                        tensor = cnn_preprocess(image.convert("RGB")).unsqueeze(0)
                        tensor.requires_grad_(True)

                        grads_list, acts_list = [], []

                        def bwd_hook(m, gi, go): grads_list.append(go[0])
                        def fwd_hook(m, i, o):   acts_list.append(o)

                        fh = cnn_model.features[-1].register_forward_hook(fwd_hook)
                        bh = cnn_model.features[-1].register_backward_hook(bwd_hook)

                        logits = cnn_model(tensor)
                        cnn_model.zero_grad()
                        logits[0, logits.argmax()].backward()

                        fh.remove(); bh.remove()

                        grads = grads_list[0].squeeze().detach().numpy()
                        acts  = acts_list[0].squeeze().detach().numpy()
                        weights = grads.mean(axis=(1, 2))
                        cam = (weights[:, None, None] * acts).sum(axis=0)
                        cam = np.maximum(cam, 0)
                        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                        cam_up = F.interpolate(
                            torch.tensor(cam).unsqueeze(0).unsqueeze(0),
                            size=(224, 224), mode="bilinear", align_corners=False
                        ).squeeze().numpy()

                        img224 = np.array(image.resize((224, 224)))
                        heatmap_rgb = plt.cm.jet(cam_up)[:, :, :3]
                        blended = np.clip(0.6 * img224 / 255.0 + 0.4 * heatmap_rgb, 0, 1)

                        fig_cam, axes_cam = plt.subplots(1, 2, figsize=(10, 4))
                        axes_cam[0].imshow(img224); axes_cam[0].set_title("Original"); axes_cam[0].axis("off")
                        axes_cam[1].imshow(blended); axes_cam[1].set_title(f"GradCAM — {cnn_result['top1_label']}"); axes_cam[1].axis("off")
                        plt.tight_layout()

                        buf = io.BytesIO()
                        fig_cam.savefig(buf, format="png", dpi=120, bbox_inches="tight")
                        buf.seek(0)
                        st.image(buf, caption="CNN GradCAM — Regions the model focused on", use_container_width=True)
                        plt.close(fig_cam)
                    except Exception as e:
                        st.error(f"GradCAM failed: {e}")

        if show_attn:
            col = interp_cols[1] if show_gradcam else interp_cols[0]
            with col:
                with st.spinner("Extracting ViT attention maps..."):
                    try:
                        import matplotlib.pyplot as plt

                        inputs = vit_processor(images=image.convert("RGB"), return_tensors="pt")
                        with torch.no_grad():
                            out = vit_model.vit(**inputs, output_attentions=True, return_dict=True)

                        attn = out.attentions[-1].squeeze(0)   # (12, 197, 197)
                        cls_attn = attn[:, 0, 1:].mean(dim=0).numpy()
                        grid = int(cls_attn.shape[0] ** 0.5)
                        attn_map = cls_attn.reshape(grid, grid)
                        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
                        attn_up  = F.interpolate(
                            torch.tensor(attn_map).unsqueeze(0).unsqueeze(0),
                            size=(224, 224), mode="bilinear", align_corners=False
                        ).squeeze().numpy()

                        img224   = np.array(image.resize((224, 224)))
                        heatmap  = plt.cm.hot(attn_up)[:, :, :3]
                        blended  = np.clip(0.55 * img224 / 255.0 + 0.45 * heatmap, 0, 1)

                        fig_attn, axes_attn = plt.subplots(1, 2, figsize=(10, 4))
                        axes_attn[0].imshow(img224); axes_attn[0].set_title("Original"); axes_attn[0].axis("off")
                        axes_attn[1].imshow(blended); axes_attn[1].set_title(f"ViT Attention — {vit_result['top1_label']}"); axes_attn[1].axis("off")
                        plt.tight_layout()

                        buf2 = io.BytesIO()
                        fig_attn.savefig(buf2, format="png", dpi=120, bbox_inches="tight")
                        buf2.seek(0)
                        st.image(buf2, caption="ViT Attention Map — Patches the model attended to", use_container_width=True)
                        plt.close(fig_attn)
                    except Exception as e:
                        st.error(f"Attention map failed: {e}")

    st.markdown("---")
    st.caption("EyeM — Built with Streamlit · PyTorch · HuggingFace Transformers")
