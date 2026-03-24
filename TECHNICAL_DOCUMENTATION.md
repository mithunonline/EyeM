# EyeM — Animal Classifier: Technical Documentation

**CNN vs Vision Transformer — A Complete, Line-by-Line Guide**

> This document explains everything from first principles. Whether you are a complete beginner or an experienced engineer, every concept is explained in plain language first, followed by the technical detail.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [What Is Machine Learning?](#2-what-is-machine-learning)
3. [What Is a Neural Network?](#3-what-is-a-neural-network)
4. [Convolutional Neural Networks (CNN)](#4-convolutional-neural-networks-cnn)
5. [Vision Transformers (ViT)](#5-vision-transformers-vit)
6. [How the Models Are Trained](#6-how-the-models-are-trained)
7. [The EfficientNet-B0 Model — Deep Dive](#7-the-efficientnet-b0-model--deep-dive)
8. [The ViT-B/16 Model — Deep Dive](#8-the-vit-b16-model--deep-dive)
9. [ImageNet — The Training Dataset](#9-imagenet--the-training-dataset)
10. [Inference — Making Predictions](#10-inference--making-predictions)
11. [Comparing the Two Models](#11-comparing-the-two-models)
12. [The Streamlit Application](#12-the-streamlit-application)
13. [Line-by-Line Code Walkthrough](#13-line-by-line-code-walkthrough)
14. [Comparison Metrics Explained](#14-comparison-metrics-explained)
15. [Project File Structure](#15-project-file-structure)
16. [How to Run the Project](#16-how-to-run-the-project)

---

## 1. Project Overview

**What does this project do?**

EyeM is an application that takes a photo of an animal and tells you what animal it is — using two completely different artificial intelligence (AI) approaches:

1. **CNN (Convolutional Neural Network)** — the traditional deep learning approach, inspired by how the human visual cortex processes images in layers, each detecting increasingly complex patterns.

2. **ViT (Vision Transformer)** — a newer approach that treats an image like a sentence, splitting it into small blocks (like words) and reading the entire image at once using an "attention" mechanism.

The application then **compares the two models** — do they agree? Which one is more confident? Where do they differ? This comparison is both educational and practical, helping users understand how different AI architectures "see" the world.

---

## 2. What Is Machine Learning?

**In plain language:**
Normally when we write a program, we tell the computer *exactly* what to do with rules (if it's orange and has four legs, it's probably a cat). Machine learning takes the opposite approach — we show the computer thousands of examples (photos labeled "cat", "dog", "tiger", etc.) and let the computer *figure out the rules itself*.

**Technically:**
Machine learning is a subset of artificial intelligence where algorithms learn patterns from data. The learning process involves:

1. **Data** — labeled examples (input → correct output)
2. **Model** — a mathematical function with millions of adjustable parameters (called "weights")
3. **Loss function** — a measure of how wrong the model's predictions are
4. **Optimizer** — an algorithm that adjusts the weights to reduce the loss
5. **Training loop** — repeating steps 3–4 thousands of times

After training, the model can generalize — making predictions on new images it has never seen before.

---

## 3. What Is a Neural Network?

**In plain language:**
A neural network is a system loosely inspired by the human brain. It consists of layers of simple "neurons" — mathematical units that each take some numbers as input, multiply them by weights (how important each input is), add them up, and apply a transformation to produce an output.

```
Input → [Neuron Layer 1] → [Neuron Layer 2] → ... → [Output Layer] → Prediction
```

**Technically:**
Each neuron computes:

```
output = activation_function( weight₁×input₁ + weight₂×input₂ + ... + bias )
```

- **Weights** — learned parameters that determine how much each input matters
- **Bias** — a learnable offset, like the y-intercept in `y = mx + b`
- **Activation function** — a non-linear transformation (e.g., ReLU: `max(0, x)`) that allows the network to learn complex, non-linear patterns

Deep neural networks stack many such layers. Each layer learns progressively more abstract features:
- Layer 1: edges, corners
- Layer 2: textures, patterns
- Layer 3: parts (ears, eyes, fur)
- Layer 4+: full objects (cat, tiger)

---

## 4. Convolutional Neural Networks (CNN)

### 4.1 The Core Idea

**In plain language:**
When you look at a photo of a cat, you don't analyze every pixel independently. You look for *local patterns* — an ear shape here, a whisker pattern there. CNNs do the same thing. They slide a small "detector" (called a filter or kernel) across the image, looking for specific patterns everywhere.

**Technically:**
A convolution operation slides a small matrix (the kernel, e.g., 3×3 pixels) across the entire input image. At each position, it computes a dot product between the kernel and the corresponding image patch, producing a single output value. Repeating this across the whole image produces a **feature map** — a new image where each pixel represents how strongly that pattern appeared at that location.

### 4.2 The Convolution Operation

For a 3×3 kernel:
```
Input patch:          Kernel:           Output value:
[ 1  2  1 ]           [ 1  0 -1 ]
[ 0  1  0 ]    *      [ 2  0 -2 ]   =   1+0-1+0+0+0-1+0+1 = 0
[ 1  2  1 ]           [ 1  0 -1 ]
```

The network learns *what kernels to use* during training — it automatically discovers edge detectors, color detectors, texture detectors, etc.

### 4.3 Pooling

After convolution, pooling reduces the spatial size of feature maps:
- **Max pooling** — keeps the maximum value in each 2×2 region
- This makes the model robust to small shifts in position (a slightly rotated tiger is still a tiger)

### 4.4 Key CNN Properties

| Property | What it means |
|---|---|
| **Local connectivity** | Each neuron looks at a small patch, not the whole image |
| **Weight sharing** | The same filter is applied everywhere — fewer parameters |
| **Translation equivariance** | If you shift the input, the output shifts the same way |
| **Hierarchical features** | Early layers detect simple patterns; later layers combine them |

---

## 5. Vision Transformers (ViT)

### 5.1 The Core Idea

**In plain language:**
Before Transformers existed in vision, they were used for language. When reading text, the model looks at every word in relation to every other word — "the tiger [that lives in the jungle] ran fast" — the word "tiger" relates to "jungle", "ran", and "fast" simultaneously. Vision Transformers apply this same "look at everything at once" approach to images.

Instead of treating an image as pixels processed by filters, ViT:
1. Cuts the image into 16×16 pixel squares (called "patches") — like cutting a photo into 196 small tiles
2. Converts each tile into a vector of numbers (a "token")
3. Processes all 196 tokens simultaneously, letting each one "attend" to every other

### 5.2 Patches and Tokens

A 224×224 pixel image divided by 16×16 patches:
```
224 / 16 = 14 patches per row
14 × 14  = 196 patches total
```

Each 16×16×3 patch (3 = RGB channels) = 768 numbers → linearly projected to a 768-dimensional vector.

### 5.3 The [CLS] Token

A special "classification token" ([CLS]) is added at the beginning of the sequence. After processing through all transformer layers, the [CLS] token has "attended" to all patches — its final state is used to make the prediction.

### 5.4 Self-Attention

**In plain language:**
Self-attention is how each patch decides which other patches are relevant. When identifying a tiger, the stripe patches might pay attention to the tail patches, and the face patches might attend to the eye patches — even though they are spatially far apart.

**Technically:**
For each token, three vectors are computed:
- **Query (Q)** — "What am I looking for?"
- **Key (K)** — "What do I represent?"
- **Value (V)** — "What information do I carry?"

Attention score between patch i and patch j:
```
attention(i,j) = softmax( Q_i · K_j / √d_k )
```

The output for patch i is a weighted sum of all Value vectors, weighted by the attention scores.

### 5.5 Multi-Head Attention

**In plain language:**
Instead of one attention mechanism, ViT uses 12 parallel attention "heads". Each head can focus on a different aspect — one might focus on shape, another on color, another on spatial relationships.

**Technically:**
The input is split into 12 subspaces of dimension 64 each (768 / 12 = 64). Each head independently computes attention, then all 12 outputs are concatenated and projected back to 768 dimensions.

### 5.6 Key ViT Properties

| Property | What it means |
|---|---|
| **Global receptive field** | Every patch can attend to every other patch from layer 1 |
| **No translation equivariance** | Must learn position from positional embeddings |
| **Large parameter count** | ~86M vs ~5M for EfficientNet-B0 |
| **Requires more training data** | Less inductive bias means it needs more examples |
| **Attention interpretability** | Can visualize what patches the model focuses on |

---

## 6. How the Models Are Trained

### 6.1 Training Overview

Both models are **pretrained** — they were trained by Google/PyTorch teams on massive datasets before being released publicly. We use them "as-is" (inference only, no fine-tuning).

**The training process (simplified):**

```
1. Start with random weights
2. Show the model an image (e.g., a tiger photo)
3. Model predicts probabilities for all 1,000 classes
4. Calculate the loss (how wrong was the prediction?)
5. Backpropagation: compute how much each weight contributed to the error
6. Update every weight slightly to reduce the error (gradient descent)
7. Repeat for millions of images, thousands of times each
```

### 6.2 Loss Function: Cross-Entropy Loss

**In plain language:**
We want the model to give a high probability to the correct class and low probability to everything else. The cross-entropy loss penalizes the model when it assigns low probability to the correct answer.

**Technically:**
```
L = -log(p_correct_class)
```

If the model says "tiger" has 95% probability and that's correct: `L = -log(0.95) = 0.051` (small loss, good)
If the model says "tiger" has 1% probability and that's wrong: `L = -log(0.01) = 4.6` (large loss, bad)

### 6.3 Backpropagation

**In plain language:**
Backpropagation is the algorithm that figures out how to blame each weight for the model's mistake. It works backwards from the output, using the chain rule of calculus to compute how much each weight contributed to the error.

**Technically:**
Using the chain rule:
```
∂L/∂w = ∂L/∂output × ∂output/∂w
```

This gradient tells us: "if we increase weight w by a tiny amount, how much does the loss change?"

### 6.4 Gradient Descent

**In plain language:**
Once we know which direction increases the error, we move every weight slightly in the *opposite* direction.

**Technically:**
```
w_new = w_old - learning_rate × ∂L/∂w
```

The **learning rate** (typically 0.001 to 0.0001) controls how big each step is:
- Too large: the model "overshoots" and oscillates
- Too small: training takes too long

Modern optimizers like **Adam** adapt the learning rate per-parameter, which works much better in practice.

### 6.5 Batch Training

Instead of updating weights after every single image, we process a **batch** (e.g., 256 images at once) and average the gradients. This:
- Provides a more stable gradient estimate
- Allows efficient GPU parallelism
- Reduces noise in the weight updates

### 6.6 Epochs and Convergence

One **epoch** = one pass through the entire training dataset.
Training typically runs for 90–300 epochs for ImageNet.

The model is considered **converged** when the validation loss stops improving — a sign that it has learned the underlying patterns rather than just memorizing the training data.

### 6.7 Regularization (Preventing Overfitting)

**Overfitting** is when a model memorizes the training data but fails on new images. To prevent it:

| Technique | How it works |
|---|---|
| **Dropout** | Randomly zeros out neurons during training, forcing the model to learn redundant representations |
| **Data augmentation** | Random crops, flips, color jitter — artificially increases dataset diversity |
| **Weight decay (L2)** | Penalizes large weights, keeping the model "simple" |
| **Label smoothing** | Instead of hard 0/1 labels, uses 0.1/0.9 — prevents overconfidence |
| **Stochastic depth** | EfficientNet randomly skips entire blocks during training |

---

## 7. The EfficientNet-B0 Model — Deep Dive

### 7.1 Background

EfficientNet was published by Google Brain in 2019. The key insight: previous CNNs scaled networks either wider (more channels), deeper (more layers), or at higher resolution — but never all three together in a principled way. EfficientNet introduces **compound scaling** — a mathematical formula to scale all three dimensions simultaneously.

### 7.2 Architecture

```
Input (224×224×3)
│
├─ Stem: Conv2d(3→32, k=3, stride=2) + BN + SiLU     → 112×112×32
│
├─ MBConv1 stage (32→16, k=3, ×1 block)               → 112×112×16
├─ MBConv6 stage (16→24, k=3, ×2 blocks)              →  56×56×24
├─ MBConv6 stage (24→40, k=5, ×2 blocks)              →  28×28×40
├─ MBConv6 stage (40→80, k=3, ×3 blocks)              →  14×14×80
├─ MBConv6 stage (80→112, k=5, ×3 blocks)             →  14×14×112
├─ MBConv6 stage (112→192, k=5, ×4 blocks)            →   7×7×192
├─ MBConv6 stage (192→320, k=3, ×1 block)             →   7×7×320
│
├─ Head: Conv2d(320→1280) + BN + SiLU                 →   7×7×1280
├─ AdaptiveAvgPool2d → GlobalAvgPool                   →       1280
├─ Dropout(0.2)
└─ Linear(1280→1000)                                   →       1000
```

### 7.3 MBConv Block (Mobile Inverted Bottleneck)

This is the fundamental building block. For an `expand_ratio` of 6:

```
Input: (B, C, H, W)
│
├─ Expand: Conv2d(C → C*6, k=1) + BN + SiLU          # 1×1 convolution, expand channels
│
├─ Depthwise: Conv2d(C*6, k=3or5, groups=C*6) + BN + SiLU  # each channel convolved separately
│
├─ SE Block:
│   ├─ AvgPool → (B, C*6, 1, 1)
│   ├─ FC(C*6 → C/4) + SiLU                          # squeeze
│   └─ FC(C/4 → C*6) + Sigmoid                       # excitation
│   └─ output = input × SE_output                     # channel-wise scaling
│
├─ Project: Conv2d(C*6 → C_out, k=1) + BN            # no activation — maintain gradient flow
│
└─ Residual connection (if input_channels == output_channels and stride == 1)
```

**Why "inverted"?** Traditional bottlenecks compress channels then expand. MBConv *expands* first, applies depthwise conv in the expanded space, then *compresses* back.

**Why depthwise separable?** A regular 3×3 convolution on 96 channels costs 96×96×3×3 = 82,944 operations per output position. Depthwise separable = 96×3×3 + 96×96×1×1 = 864 + 9,216 = 10,080 — about 8× fewer operations.

### 7.4 Squeeze-and-Excitation (SE) Module

**In plain language:**
The SE module asks: "Which channels (feature maps) are important for this image?" It squeezes each feature map to a single number (global average), learns a weight for each channel, and rescales the feature maps accordingly.

```python
# SE: channel attention
x_se = x.mean(dim=[2,3])           # (B, C) — squeeze spatial info
x_se = self.fc1(x_se)              # (B, C/4) — compress
x_se = F.silu(x_se)
x_se = self.fc2(x_se)              # (B, C) — expand back
x_se = torch.sigmoid(x_se)        # (B, C) — values in [0, 1]
return x * x_se.unsqueeze(-1).unsqueeze(-1)  # channel-wise scaling
```

### 7.5 SiLU Activation

EfficientNet uses SiLU (Sigmoid Linear Unit) instead of ReLU:
```
SiLU(x) = x × sigmoid(x) = x / (1 + e^(-x))
```
SiLU is smooth and non-monotonic, which has been found to improve performance over ReLU on large models.

### 7.6 GradCAM (Gradient-weighted Class Activation Mapping)

**In plain language:**
GradCAM shows which parts of the image the CNN used to make its decision. It asks: "Which pixels, if I nudged them slightly, would most change the prediction?" The result is a heatmap overlaid on the original image.

**Technically:**
1. Run a forward pass, save the feature maps from the target layer
2. Backpropagate the gradient of the predicted class score w.r.t. those feature maps
3. Average the gradients across the spatial dimensions to get per-channel importance weights
4. Compute a weighted sum of the feature maps
5. Apply ReLU (keep only positive activations) and upsample to image size

```python
# Gradient-weighted class activation map
grads = gradients[0]            # (B, C, H, W) — gradient of loss w.r.t. feature maps
acts  = activations[0]          # (B, C, H, W) — feature maps themselves

weights = grads.mean(dim=[2,3]) # (C,) — global average pooling of gradients

cam = (weights[:, None, None] * acts).sum(dim=0)  # weighted sum of feature maps
cam = F.relu(cam)               # keep positive (class-discriminative) regions
cam = (cam - cam.min()) / (cam.max() - cam.min())  # normalize to [0, 1]
```

---

## 8. The ViT-B/16 Model — Deep Dive

### 8.1 Background

Vision Transformer (ViT) was published by Google Research in 2020. It was the first demonstration that pure Transformer architectures (without any convolutions) could match or exceed CNNs on image classification — when trained on sufficient data.

### 8.2 Architecture

```
Input image: 224×224×3
│
├─ Patch Embedding:
│   ├─ Split into 196 patches of 16×16×3 = 768 numbers each
│   ├─ Linear projection: 768 → 768 (learnable matrix)
│   └─ Prepend [CLS] token: sequence = (197, 768)
│
├─ Add Positional Embeddings:
│   └─ 197 learnable vectors of size 768 are added (not concatenated)
│
├─ Transformer Encoder (×12 layers):
│   Each layer contains:
│   ├─ LayerNorm
│   ├─ Multi-Head Self-Attention (12 heads, d_head=64)
│   ├─ Residual connection: output = input + attention_output
│   ├─ LayerNorm
│   ├─ MLP: Linear(768→3072) + GELU + Linear(3072→768)
│   └─ Residual connection: output = input + mlp_output
│
├─ Extract [CLS] token output: (768,)
│
└─ Linear classifier: 768 → 1000
```

### 8.3 Patch Embedding — Line by Line

```python
# In transformers library (simplified pseudocode)

# Image: (B, 3, 224, 224)
# Unfold into patches: (B, 196, 768)
patches = image.unfold(2, 16, 16).unfold(3, 16, 16)
# patches shape: (B, 14, 14, 3, 16, 16)
patches = patches.reshape(B, 196, 3*16*16)  # (B, 196, 768)

# Linear projection to hidden dim
patch_tokens = linear(patches)  # (B, 196, 768) where linear has shape (768, 768)

# Prepend CLS token
cls_token = self.cls_token.expand(B, 1, 768)  # learnable parameter
tokens = torch.cat([cls_token, patch_tokens], dim=1)  # (B, 197, 768)

# Add positional embeddings (also learnable)
tokens = tokens + self.position_embeddings  # (B, 197, 768)
```

### 8.4 Multi-Head Self-Attention — Line by Line

```python
# Input: tokens (B, 197, 768)
# Split into 12 heads of dimension 64 each

# Compute Q, K, V by linear projection
Q = self.W_q(tokens)  # (B, 197, 768)
K = self.W_k(tokens)  # (B, 197, 768)
V = self.W_v(tokens)  # (B, 197, 768)

# Reshape for multi-head: (B, num_heads, seq_len, head_dim)
Q = Q.reshape(B, 197, 12, 64).transpose(1, 2)  # (B, 12, 197, 64)
K = K.reshape(B, 197, 12, 64).transpose(1, 2)  # (B, 12, 197, 64)
V = V.reshape(B, 197, 12, 64).transpose(1, 2)  # (B, 12, 197, 64)

# Scaled dot-product attention
scale  = 64 ** 0.5                             # √d_k = 8
scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, 12, 197, 197)
attn_weights = torch.softmax(scores, dim=-1)  # (B, 12, 197, 197)

# Weighted sum of values
context = torch.matmul(attn_weights, V)       # (B, 12, 197, 64)

# Concatenate heads
context = context.transpose(1, 2).reshape(B, 197, 768)  # (B, 197, 768)

# Output projection
output = self.W_o(context)  # (B, 197, 768)
```

### 8.5 Why Scale by √d_k?

Without scaling, for large d_k, the dot products grow large → softmax becomes very peaked → gradients become very small (vanishing gradient problem). Dividing by √d_k keeps the dot products in a reasonable range.

### 8.6 Layer Normalization

Unlike Batch Normalization (which normalizes across the batch dimension), LayerNorm normalizes across the feature dimension:
```python
# For each token independently:
mean = x.mean(dim=-1, keepdim=True)
std  = x.std(dim=-1, keepdim=True)
x_normalized = (x - mean) / (std + 1e-5)
output = gamma * x_normalized + beta  # gamma, beta are learned parameters
```

### 8.7 MLP Block

```python
# Two linear layers with GELU activation
x = self.linear1(x)          # (B, 197, 768) → (B, 197, 3072) — 4× expansion
x = F.gelu(x)                # GELU(x) = x × Φ(x) where Φ is the Gaussian CDF
x = self.dropout(x)
x = self.linear2(x)          # (B, 197, 3072) → (B, 197, 768)
```

### 8.8 Residual Connections

Each sub-block uses a residual (skip) connection:
```python
x = x + self.attention(self.norm1(x))   # LayerNorm → Attention → Add
x = x + self.mlp(self.norm2(x))         # LayerNorm → MLP → Add
```

**Why residuals?** They allow gradients to flow directly from the output back to early layers during backpropagation, enabling training of very deep networks (12 layers here, but Transformers can go to hundreds of layers).

### 8.9 Attention Map Extraction

```python
# The attention matrix (B, heads, 197, 197) tells us:
# "How much does token i attend to token j?"

# To visualize what image regions the model focuses on:
attn = outputs.attentions[-1]  # last layer: (B, 12, 197, 197)
attn = attn.squeeze(0)         # (12, 197, 197)

# Attention FROM [CLS] token TO all patches
cls_attn = attn[:, 0, 1:]      # (12, 196) — exclude CLS-to-CLS
avg_attn = cls_attn.mean(0)    # (196,) — average across heads

# Reshape to 14×14 spatial grid
attn_map = avg_attn.reshape(14, 14)
```

---

## 9. ImageNet — The Training Dataset

### 9.1 What Is ImageNet?

ImageNet is a massive labeled image dataset assembled by Stanford researchers, first released in 2009. It is the benchmark that transformed deep learning and made modern AI vision possible.

| Property | Value |
|---|---|
| Total images | ~14 million |
| Labeled classes | 21,841 (ImageNet-21k) or 1,000 (ILSVRC subset) |
| Training images (1k) | ~1.28 million |
| Validation images (1k) | 50,000 |
| Image size | Variable (224×224 after preprocessing) |

### 9.2 Animal Classes in ImageNet-1k

Of the 1,000 ImageNet-1k classes, approximately 398 are animals:
- **Dogs**: 120 breeds (classes 151–268), the largest single category
- **Birds**: ~59 species (classes 7–24, 80–100, etc.)
- **Fish**: ~30 species (classes 0–6, 389–397)
- **Reptiles & Amphibians**: ~30 classes
- **Wild mammals**: lions, tigers, elephants, bears, etc.
- **Insects**: bees, butterflies, beetles, etc.

### 9.3 Data Preprocessing for Training

Before feeding images to the model during training:

```python
transforms.Compose([
    transforms.RandomResizedCrop(224),   # Random crop of 224×224 — augmentation
    transforms.RandomHorizontalFlip(),   # 50% chance to mirror — augmentation
    transforms.ColorJitter(              # Random color changes — augmentation
        brightness=0.4, contrast=0.4,
        saturation=0.4, hue=0.1
    ),
    transforms.ToTensor(),               # Convert PIL image to PyTorch tensor [0, 1]
    transforms.Normalize(               # Normalize to mean=0, std=1
        mean=[0.485, 0.456, 0.406],     # ImageNet mean (RGB)
        std=[0.229, 0.224, 0.225]       # ImageNet std (RGB)
    ),
])
```

**Why normalize?** Neural networks work best when inputs are zero-centered with unit variance. Using the dataset's mean and standard deviation ensures the first layer receives well-conditioned inputs.

**Why augment?** Augmentation artificially multiplies the effective dataset size. A model that sees a tiger in different crops, flips, and lighting conditions learns a more robust representation.

---

## 10. Inference — Making Predictions

### 10.1 The Forward Pass

When you upload an animal photo:

```
1. Load image → PIL Image (any size, RGB)
2. Preprocess:
   - Resize to 256px on shorter side
   - Center crop to 224×224
   - Normalize pixels
3. Add batch dimension: (1, 3, 224, 224)
4. Forward pass through model
5. Output: (1, 1000) — raw logits for each class
6. Apply softmax: convert logits to probabilities (all sum to 1.0)
7. Take argmax: the highest probability is the prediction
```

### 10.2 Softmax — Converting Logits to Probabilities

**In plain language:**
The model outputs raw numbers (logits) — larger number = model is more sure about that class. Softmax converts these into proper probabilities that sum to 1.

```python
# Raw logits might be: [2.3, 5.1, -1.2, 0.8, ...]
# Softmax:
exp_scores = np.exp(logits)          # [9.97, 164.0, 0.30, 2.23, ...]
probabilities = exp_scores / exp_scores.sum()  # [0.05, 0.82, 0.001, 0.011, ...]
```

The class with the highest probability (0.82 → ~82% confident) is the prediction.

### 10.3 Temperature and Confidence Calibration

Well-calibrated models mean: when a model says "80% tiger", it should be correct about 80% of the time. Both EfficientNet and ViT can sometimes be overconfident (saying 99.9% when they're actually only 80% accurate on challenging images).

---

## 11. Comparing the Two Models

### 11.1 Architecture Comparison

| Property | EfficientNet-B0 (CNN) | ViT-B/16 |
|---|---|---|
| Type | Convolutional | Transformer |
| Parameters | ~5.3M | ~86M |
| Input size | 224×224 | 224×224 |
| Tokens/features | Feature maps (spatial) | 197 tokens (196 patches + CLS) |
| Receptive field | Local → global | Global from layer 1 |
| Inductive biases | Translation equivariance, locality | None (learned from data) |
| Training data needed | Moderate | Large (benefits from ImageNet-21k) |
| ImageNet-1k Top-1 | 77.7% | 81.8% |
| Inference speed | Fast | Slower (larger model) |

### 11.2 Why Do They Predict Differently?

**CNNs** build features hierarchically:
- They "see" local patterns first and combine them into global understanding
- A tiger is recognized because: stripes → striped texture → striped body shape → tiger
- They are highly sensitive to textures

**ViTs** capture global context immediately:
- Each patch can directly "see" every other patch from the first layer
- A tiger patch near the tail can attend to a patch near the face from the start
- They capture long-range dependencies better (useful when context matters)

Research has shown:
- CNNs are biased toward **texture** (a photo of a cat with elephant texture might be classified as elephant)
- ViTs are biased toward **shape** (more robust to texture changes)

This fundamental difference causes them to disagree on some images — especially those with unusual textures or poses.

---

## 12. The Streamlit Application

### 12.1 Overview

Streamlit is a Python library that turns Python scripts into interactive web applications without any web development knowledge.

Our app (`app.py`) works as follows:

```
1. User opens the app in a browser
2. Models are loaded once and cached (not reloaded on every interaction)
3. User uploads an image
4. Both models run inference simultaneously
5. Results are displayed: predictions, confidence, comparison metrics, charts
6. Optional: GradCAM and Attention maps for interpretability
```

### 12.2 Model Caching

```python
@st.cache_resource(show_spinner=False)
def load_cnn_model():
    ...
```

`@st.cache_resource` ensures the 86MB ViT model is only downloaded and loaded into memory **once**, even if the user uploads multiple images. Without caching, every image upload would reload the model — taking ~30 seconds each time.

### 12.3 The Inference Pipeline in the App

```python
@torch.no_grad()  # Disable gradient computation — saves memory, speeds up inference
def run_cnn(image, model, preprocess, labels):
    tensor = preprocess(image.convert("RGB")).unsqueeze(0)  # (1, 3, 224, 224)
    logits = model(tensor)                                    # (1, 1000)
    probs  = torch.softmax(logits, dim=-1).squeeze().numpy() # (1000,)
    top_idx = probs.argsort()[::-1][:TOP_K]                  # indices of top predictions
    return {
        "top": [(labels[i], float(probs[i])) for i in top_idx],
        "all_probs": probs,       # full distribution — needed for comparison metrics
        "top1_label": ...,
        "top1_conf": ...,
    }
```

---

## 13. Line-by-Line Code Walkthrough

### 13.1 CNN Notebook Key Lines

```python
# Load pretrained EfficientNet-B0
weights = EfficientNet_B0_Weights.IMAGENET1K_V1
# ^ This tells PyTorch to use weights trained on ImageNet-1k (1,000 class version)
# The weights file (~21MB) is automatically downloaded from the internet

model = models.efficientnet_b0(weights=weights)
# ^ Creates the model architecture AND loads the pretrained weights into it

model.eval()
# ^ Switches the model to "evaluation mode"
# This disables Dropout (randomly zeroing neurons) and BatchNorm behaves differently
# CRITICAL: always call this before inference, or predictions will be random

tensor = preprocess(img).unsqueeze(0)
# ^ preprocess: resize → crop → normalize → tensor
# .unsqueeze(0): adds a batch dimension
# Image goes from (3, 224, 224) → (1, 3, 224, 224)
# The "1" is the batch size — we're processing 1 image at a time

with torch.no_grad():
# ^ Disables gradient tracking — we're not training, just predicting
# Saves ~50% memory and speeds up inference

logits = model(tensor)
# ^ Forward pass through all EfficientNet layers
# Output shape: (1, 1000) — one score per ImageNet class

probs = torch.softmax(logits, dim=-1)
# ^ Convert raw scores to probabilities (0 to 1, summing to 1)
# dim=-1 means: apply softmax across the last dimension (the 1000 classes)
```

### 13.2 ViT Notebook Key Lines

```python
MODEL_ID = 'google/vit-base-patch16-224'
# ^ This is the HuggingFace model identifier:
# "google" = published by Google Research
# "vit" = Vision Transformer
# "base" = base-size model (not large or huge)
# "patch16" = 16×16 pixel patches
# "224" = expects 224×224 input images

processor = ViTImageProcessor.from_pretrained(MODEL_ID)
# ^ Downloads and loads the image processor
# This handles: resizing, normalizing, converting to the format ViT expects

vit_model = ViTForImageClassification.from_pretrained(MODEL_ID)
# ^ Downloads weights (~330MB) and loads the full ViT model
# "ForImageClassification" = includes the final linear classification head

inputs = processor(images=img, return_tensors='pt')
# ^ Preprocess the image using ViT's own processor
# return_tensors='pt' = return PyTorch tensors (not numpy or tensorflow)
# Output: dict with key "pixel_values" of shape (1, 3, 224, 224)

outputs = vit_model(**inputs)
# ^ ** unpacks the dict: equivalent to vit_model(pixel_values=tensor)
# Forward pass through all 12 transformer layers

logits = outputs.logits
# ^ The raw class scores before softmax: (1, 1000)
```

### 13.3 Comparison Metrics Lines

```python
# KL Divergence
kl_pq = scipy.stats.entropy(p, q)
# ^ Kullback-Leibler divergence from distribution p (CNN) to q (ViT)
# Measures "how many bits of extra information are needed to encode q using p"
# 0 = identical, larger = more different
# NOT symmetric: KL(p,q) ≠ KL(q,p)

# Jensen-Shannon Divergence
m = 0.5 * (p + q)                                  # mixture distribution
js_div = 0.5 * entropy(p, m) + 0.5 * entropy(q, m) # symmetric KL
# ^ JS divergence is the symmetric version of KL
# Uses the average distribution m as reference
# Bounded between 0 and log(2) ≈ 0.693

# Cosine Similarity
cos_sim = np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q))
# ^ Measures the angle between the two probability vectors
# 1.0 = pointing in the same direction (same prediction patterns)
# 0.0 = orthogonal (completely unrelated predictions)

# Top-5 Jaccard
cnn_top5 = set([label for label, _ in cnn_res["top"][:5]])
vit_top5 = set([label for label, _ in vit_res["top"][:5]])
jaccard = len(cnn_top5 & vit_top5) / len(cnn_top5 | vit_top5)
# ^ Jaccard index = intersection / union
# 1.0 = exact same 5 classes in top-5
# 0.0 = no classes in common
```

---

## 14. Comparison Metrics Explained

### KL Divergence

**In plain language:**
Imagine CNN gives 90% probability to "tiger" and ViT gives 10%. KL divergence quantifies how surprised the ViT would be if it tried to use CNN's distribution to describe what it sees. Large KL = models are "thinking very differently" about this image.

**Formula:** `KL(P ∥ Q) = Σ P(x) × log(P(x) / Q(x))`

**Interpretation:**
- KL < 0.1: Very similar distributions
- KL 0.1–0.5: Moderate difference
- KL > 0.5: Significantly different predictions

### Jensen-Shannon Divergence

A symmetric, bounded version of KL divergence. The square root of JS divergence is a true mathematical distance metric.

**Formula:** `JS(P ∥ Q) = 0.5 × KL(P ∥ M) + 0.5 × KL(Q ∥ M)` where `M = 0.5(P + Q)`

**Interpretation:**
- 0: Identical distributions
- ~0.693: Maximum divergence (one predicts class A with 100%, other predicts class B with 100%)

### Cosine Similarity

**In plain language:**
Think of the probability distribution as a vector in 1000-dimensional space. Cosine similarity measures whether the two vectors point in the same general direction — even if one is much longer (more confident) than the other.

**Formula:** `cos(θ) = (P · Q) / (|P| × |Q|)`

### Top-5 Jaccard Index

**In plain language:**
Do the models agree on *which classes are plausible*? Jaccard measures the overlap between the two models' top-5 prediction lists.

**Formula:** `Jaccard = |A ∩ B| / |A ∪ B|`

Example: CNN top-5 = {tiger, leopard, cheetah, jaguar, lion}, ViT top-5 = {tiger, jaguar, snow_leopard, cat, lynx}
- Intersection: {tiger, jaguar} = 2 classes
- Union: {tiger, leopard, cheetah, jaguar, lion, snow_leopard, cat, lynx} = 8 classes
- Jaccard = 2/8 = 0.25

---

## 15. Project File Structure

```
EyeM/
│
├── app.py                          # Main Streamlit application
│
├── notebooks/
│   ├── CNN_Animal_Classifier.ipynb # CNN (EfficientNet-B0) — full walkthrough
│   └── ViT_Animal_Classifier.ipynb # ViT (vit-base-patch16-224) — full walkthrough
│
├── models/
│   ├── cnn_config.json             # CNN model configuration
│   ├── vit_config.json             # ViT model configuration
│   ├── cnn_sample_output.png       # Generated by CNN notebook
│   ├── cnn_gradcam.png             # GradCAM visualization
│   ├── cnn_feature_maps.png        # CNN feature map visualization
│   ├── vit_sample_output.png       # Generated by ViT notebook
│   ├── vit_attention_map.png       # ViT attention visualization
│   └── vit_all_heads.png           # All 12 attention heads
│
├── requirements.txt                # Python dependencies
└── TECHNICAL_DOCUMENTATION.md     # This document
```

---

## 16. How to Run the Project

### 16.1 Prerequisites

- Python 3.8 or higher
- ~2GB free disk space (for model weights)
- Internet connection (first run downloads model weights)

### 16.2 Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/EyeM.git
cd EyeM

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate.bat       # Windows

# Install dependencies
pip install -r requirements.txt
```

### 16.3 Running the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

**First launch:** Both model weights will be downloaded automatically:
- EfficientNet-B0: ~21MB
- ViT-B/16: ~330MB

### 16.4 Running the Notebooks

```bash
# Install Jupyter
pip install jupyter

# Start Jupyter
jupyter notebook notebooks/
```

Open either notebook and run all cells (Kernel → Restart & Run All).

### 16.5 Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| RAM | 4 GB | 8 GB |
| CPU | Any modern | Multi-core |
| GPU | Optional | NVIDIA GPU with 4GB VRAM |
| Storage | 1 GB | 2 GB |

Both models automatically use GPU if available (via `torch.cuda.is_available()`).

---

*This documentation covers the complete technical stack of the EyeM project — from first principles of machine learning through to the specific mathematical operations inside each model and how they are deployed in the Streamlit interface.*
