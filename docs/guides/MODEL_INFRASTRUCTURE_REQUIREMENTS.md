# Model Infrastructure Requirements Guide

**Purpose:** Computational, dependency, and deployment requirements for each model family
**Audience:** DevOps, ML engineers, infrastructure architects
**Last Updated:** 2025-12-30

---

## Table of Contents

1. [Overview](#overview)
2. [Boosting Models](#boosting-models)
3. [Neural Sequence Models](#neural-sequence-models)
4. [Transformers](#transformers)
5. [CNN Models](#cnn-models)
6. [Foundation Models](#foundation-models)
7. [Classical Models](#classical-models)
8. [Ensemble Models](#ensemble-models)
9. [Infrastructure Recommendations](#infrastructure-recommendations)
10. [Data Pipeline Consumption](#data-pipeline-consumption)
11. [Memory Estimation Utilities](#memory-estimation-utilities)

---

## Overview

### Requirements Matrix

| Model Family | GPU | Min RAM | Training Time | Inference Latency | Disk Space | Python Deps |
|--------------|-----|---------|---------------|-------------------|------------|-------------|
| **Boosting** | No | 4GB | 2-5 min | <1ms | 50MB | xgboost, lightgbm, catboost |
| **Neural (LSTM/GRU/TCN)** | Recommended | 8GB (16GB GPU) | 20-60 min | 5-10ms (GPU) | 20MB | pytorch |
| **Transformers** | Required | 16GB (16GB+ GPU) | 1-3 hours | 10-50ms (GPU) | 50-200MB | pytorch, timm |
| **CNN** | Recommended | 8GB (8GB GPU) | 30-90 min | 5-15ms (GPU) | 30MB | pytorch, sktime |
| **Foundation** | Required | 32GB (16GB+ GPU) | Inference-only | 50-200ms (GPU) | 200MB-2GB | transformers, chronos |
| **Classical** | No | 2GB | 1-10 min | <1ms | 10-50MB | scikit-learn |
| **Ensemble** | Varies | Varies | 2x base models | Same as base | 100MB+ | Depends on base |

---

## Boosting Models

### XGBoost, LightGBM, CatBoost

**Hardware Requirements:**

- **CPU:** 4+ cores recommended (16+ for large datasets)
- **GPU:** Not required (CPU-only training)
- **RAM:** 4GB minimum, 8-16GB for large datasets
  - Memory scales with: `n_samples × n_features × 8 bytes`
  - Example: 20k samples × 150 features = 24MB (plus overhead)
- **Disk:** 50MB per model (saved artifacts)

**Training Time:**

| Dataset Size | Features | Time (4 cores) | Time (16 cores) |
|--------------|----------|----------------|-----------------|
| 10k samples | 150 | 1-2 min | 30-60s |
| 50k samples | 150 | 5-8 min | 2-3 min |
| 200k samples | 150 | 15-20 min | 6-8 min |

**Inference Latency:**

- **Single sample:** <1ms (CPU)
- **Batch (1000 samples):** 5-10ms (CPU)
- **Production latency:** <1ms (acceptable for real-time trading)

**Memory Scaling:**

```python
# Estimate XGBoost memory usage
def estimate_xgboost_memory(n_samples, n_features, max_depth=6):
    """
    Estimate XGBoost training memory.

    Args:
        n_samples: Number of training samples
        n_features: Number of features
        max_depth: Maximum tree depth

    Returns:
        Memory in MB
    """
    # Data matrix
    data_memory = n_samples * n_features * 8  # 8 bytes per float64

    # Tree storage (approximate)
    num_trees = 1000  # Typical num_boost_round
    nodes_per_tree = 2 ** max_depth
    tree_memory = num_trees * nodes_per_tree * 64  # 64 bytes per node

    # Gradient/Hessian storage
    gradient_memory = n_samples * 8 * 2  # grad + hessian

    total_memory_mb = (data_memory + tree_memory + gradient_memory) / (1024 ** 2)

    return total_memory_mb

# Example
memory_mb = estimate_xgboost_memory(20000, 150, max_depth=8)
print(f"Estimated memory: {memory_mb:.1f} MB")  # ~250 MB
```

**Dependencies:**

```bash
# XGBoost
pip install xgboost==2.0.3

# LightGBM
pip install lightgbm==4.1.0

# CatBoost
pip install catboost==1.2.2
```

**Production Deployment:**

```python
# Serve XGBoost with FastAPI
from fastapi import FastAPI
import xgboost as xgb
import numpy as np

app = FastAPI()
model = xgb.Booster()
model.load_model('models/xgboost_model.json')

@app.post("/predict")
def predict(features: list[float]):
    X = np.array(features).reshape(1, -1)
    dmatrix = xgb.DMatrix(X)
    proba = model.predict(dmatrix)
    return {"probabilities": proba.tolist()}
```

**Latency: <1ms**

---

## Neural Sequence Models

### LSTM, GRU, TCN

**Hardware Requirements:**

- **CPU:** 8+ cores (for data loading, augmentation)
- **GPU:** Strongly recommended (10-50x speedup vs CPU)
  - **Minimum:** 8GB VRAM (e.g., RTX 3070, GTX 1080)
  - **Recommended:** 16GB+ VRAM (e.g., RTX 4080, A4000)
  - **Ideal:** 24GB+ VRAM (e.g., RTX 3090, A5000)
- **RAM:** 16GB minimum, 32GB recommended
- **Disk:** 20MB per model

**GPU Memory Scaling:**

```python
def estimate_lstm_gpu_memory(seq_len, batch_size, hidden_size, num_layers, n_features):
    """
    Estimate LSTM GPU memory usage.

    Args:
        seq_len: Sequence length (60, 120, etc.)
        batch_size: Batch size (32, 256, 512)
        hidden_size: Hidden state dimension (64, 128, 256)
        num_layers: Number of LSTM layers (1, 2, 4)
        n_features: Number of input features

    Returns:
        Memory in MB
    """
    # Input tensor
    input_memory = batch_size * seq_len * n_features * 4  # 4 bytes per float32

    # LSTM hidden states (per layer)
    hidden_memory_per_layer = batch_size * seq_len * hidden_size * 4
    hidden_memory_total = hidden_memory_per_layer * num_layers

    # LSTM weights (per layer)
    # Each LSTM cell has 4 gates: input, forget, cell, output
    # Weights: (input_size + hidden_size) * hidden_size * 4
    input_size_first_layer = n_features
    weight_memory_first_layer = (input_size_first_layer + hidden_size) * hidden_size * 4 * 4

    input_size_other_layers = hidden_size
    weight_memory_per_layer = (input_size_other_layers + hidden_size) * hidden_size * 4 * 4
    weight_memory_total = weight_memory_first_layer + weight_memory_per_layer * (num_layers - 1)

    # Gradients (same size as weights)
    gradient_memory = weight_memory_total

    # Classifier head
    classifier_memory = hidden_size * 3 * 4  # 3 output classes

    total_memory_bytes = (
        input_memory +
        hidden_memory_total +
        weight_memory_total +
        gradient_memory +
        classifier_memory
    )

    # Add 30% overhead for PyTorch internals
    total_memory_mb = total_memory_bytes / (1024 ** 2) * 1.3

    return total_memory_mb

# Examples
print("LSTM Memory Estimates:")
print(f"  Small:  {estimate_lstm_gpu_memory(60, 256, 64, 2, 26):.1f} MB")   # ~800 MB
print(f"  Medium: {estimate_lstm_gpu_memory(60, 256, 128, 2, 26):.1f} MB")  # ~1.5 GB
print(f"  Large:  {estimate_lstm_gpu_memory(120, 512, 256, 4, 26):.1f} MB") # ~8 GB
```

**Training Time:**

| Config | GPU | Time per Epoch | Total Time (100 epochs) |
|--------|-----|----------------|-------------------------|
| Small (hidden=64, layers=2) | RTX 3070 | 10s | 15 min |
| Medium (hidden=128, layers=2) | RTX 3070 | 20s | 30 min |
| Large (hidden=256, layers=4) | RTX 3090 | 40s | 60 min |

**Inference Latency:**

- **GPU (RTX 3070):** 5-10ms per sample
- **CPU (16 cores):** 50-100ms per sample
- **Production:** GPU required for real-time (10ms budget)

**Dependencies:**

```bash
# PyTorch
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA (if using GPU)
# Requires CUDA 12.1+ drivers installed
nvidia-smi  # Check CUDA availability
```

**Production Deployment:**

```python
# Serve LSTM with TorchServe
import torch
from src.models.neural.lstm_model import LSTMModel

# Load model
model = LSTMModel.load('models/lstm_model/')
model.eval()
model.to('cuda')

@app.post("/predict")
def predict(sequence: list[list[float]]):  # (seq_len, n_features)
    X = torch.FloatTensor(sequence).unsqueeze(0).to('cuda')  # (1, seq_len, n_features)

    with torch.no_grad():
        output = model.predict(X.cpu().numpy())

    return {"probabilities": output.probabilities.tolist()}
```

**Latency: 5-10ms (GPU)**

**Batch Processing for Throughput:**

```python
# Batch inference for higher throughput
def predict_batch(sequences: list):  # List of (seq_len, n_features)
    X = torch.FloatTensor(sequences).to('cuda')  # (batch, seq_len, n_features)

    with torch.no_grad():
        outputs = model.predict_batch(X.cpu().numpy())

    return outputs

# Throughput: 100-200 samples/second (batch_size=32, GPU)
```

---

## Transformers

### PatchTST, iTransformer, TFT

**Hardware Requirements:**

- **CPU:** 16+ cores (data preprocessing bottleneck)
- **GPU:** Required (100x+ speedup vs CPU)
  - **Minimum:** 16GB VRAM (A4000, RTX 4090)
  - **Recommended:** 24GB+ VRAM (RTX 3090, A5000)
  - **Ideal:** 40GB+ VRAM (A100, A6000) for large models
- **RAM:** 32GB minimum, 64GB recommended
- **Disk:** 50-200MB per model

**GPU Memory Scaling:**

```python
def estimate_transformer_gpu_memory(seq_len, batch_size, d_model, nhead, num_layers):
    """
    Estimate Transformer GPU memory.

    Args:
        seq_len: Sequence length (512 typical for PatchTST)
        batch_size: Batch size (16, 32, 64)
        d_model: Model dimension (128, 256, 512)
        nhead: Number of attention heads (4, 8, 16)
        num_layers: Number of transformer layers (2, 4, 8)

    Returns:
        Memory in MB
    """
    # Input embeddings
    input_memory = batch_size * seq_len * d_model * 4

    # Self-attention (per layer)
    # Q, K, V projections: seq_len × d_model × 3
    qkv_memory_per_layer = batch_size * seq_len * d_model * 3 * 4

    # Attention scores: seq_len × seq_len (quadratic!)
    attention_memory_per_layer = batch_size * nhead * seq_len * seq_len * 4

    # Feed-forward (per layer)
    # Typically 4x expansion: d_model → 4*d_model → d_model
    ff_memory_per_layer = batch_size * seq_len * d_model * 4 * 4

    # Total per layer
    memory_per_layer = qkv_memory_per_layer + attention_memory_per_layer + ff_memory_per_layer

    # All layers
    total_layer_memory = memory_per_layer * num_layers

    # Model weights (approximate)
    weight_memory = num_layers * d_model * d_model * 12 * 4  # Approximate

    # Gradients
    gradient_memory = weight_memory

    total_memory_bytes = input_memory + total_layer_memory + weight_memory + gradient_memory

    # Add 50% overhead for PyTorch + attention mechanisms
    total_memory_mb = total_memory_bytes / (1024 ** 2) * 1.5

    return total_memory_mb

# Examples
print("Transformer Memory Estimates:")
print(f"  Small:  {estimate_transformer_gpu_memory(256, 32, 128, 4, 2):.1f} MB")   # ~3 GB
print(f"  Medium: {estimate_transformer_gpu_memory(512, 16, 256, 8, 4):.1f} MB")   # ~12 GB
print(f"  Large:  {estimate_transformer_gpu_memory(512, 8, 512, 16, 8):.1f} MB")   # ~28 GB
```

**Key Insight:** Attention memory scales **quadratically** with `seq_len`! Doubling `seq_len` quadruples memory.

**Training Time:**

| Config | GPU | Time per Epoch | Total Time (100 epochs) |
|--------|-----|----------------|-------------------------|
| Small (d_model=128, layers=2) | RTX 4090 | 30s | 45 min |
| Medium (d_model=256, layers=4) | RTX 4090 | 90s | 2.5 hours |
| Large (d_model=512, layers=8) | A100 40GB | 180s | 5 hours |

**Inference Latency:**

- **GPU (RTX 4090):** 10-50ms per sample (depends on seq_len)
- **CPU:** 500ms+ per sample (not practical)
- **Production:** GPU required

**Dependencies:**

```bash
# PyTorch
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Timm (for PatchTST)
pip install timm==0.9.12

# Optional: Flash Attention 2 (2-4x speedup)
pip install flash-attn==2.5.0
```

**Production Deployment:**

```python
# Serve with mixed precision (float16) for 2x speedup
import torch
from src.models.transformers.patchtst import PatchTSTModel

model = PatchTSTModel.load('models/patchtst/')
model.eval()
model.to('cuda')
model.half()  # Convert to float16

@app.post("/predict")
def predict(sequence: list[list[float]]):
    X = torch.FloatTensor(sequence).unsqueeze(0).half().to('cuda')

    with torch.no_grad():
        output = model.predict(X.cpu().float().numpy())

    return {"probabilities": output.probabilities.tolist()}
```

**Latency: 10-50ms (GPU, mixed precision)**

---

## CNN Models

### InceptionTime, ResNet

**Hardware Requirements:**

- **CPU:** 8+ cores
- **GPU:** Recommended (5-20x speedup)
  - **Minimum:** 8GB VRAM (RTX 3070)
  - **Recommended:** 16GB VRAM (RTX 4080)
- **RAM:** 16GB
- **Disk:** 30MB per model

**GPU Memory Scaling:**

```python
def estimate_cnn_gpu_memory(seq_len, batch_size, num_filters, num_blocks, n_features):
    """
    Estimate CNN (InceptionTime) GPU memory.

    Args:
        seq_len: Sequence length (60, 120)
        batch_size: Batch size (32, 64, 128)
        num_filters: Number of filters per layer (32, 64, 128)
        num_blocks: Number of inception blocks (2, 4, 8)
        n_features: Number of input features (5, 25)

    Returns:
        Memory in MB
    """
    # Input
    input_memory = batch_size * seq_len * n_features * 4

    # Convolutional layers (per block)
    # InceptionTime has multiple kernel sizes: [3, 5, 8, 11, 23]
    conv_memory_per_block = batch_size * seq_len * num_filters * 5 * 4  # 5 kernels

    # All blocks
    conv_memory_total = conv_memory_per_block * num_blocks

    # Bottleneck layers
    bottleneck_memory = batch_size * seq_len * (num_filters // 4) * num_blocks * 4

    # Global pooling + classifier
    classifier_memory = batch_size * num_filters * 4

    # Weights (approximate)
    weight_memory = num_blocks * num_filters * n_features * 23 * 4  # Max kernel size

    # Gradients
    gradient_memory = weight_memory

    total_memory_bytes = (
        input_memory +
        conv_memory_total +
        bottleneck_memory +
        classifier_memory +
        weight_memory +
        gradient_memory
    )

    total_memory_mb = total_memory_bytes / (1024 ** 2) * 1.3

    return total_memory_mb

# Examples
print(f"  Small:  {estimate_cnn_gpu_memory(60, 64, 32, 2, 5):.1f} MB")   # ~1.5 GB
print(f"  Medium: {estimate_cnn_gpu_memory(60, 64, 128, 4, 5):.1f} MB")  # ~4 GB
print(f"  Large:  {estimate_cnn_gpu_memory(120, 128, 256, 8, 5):.1f} MB") # ~12 GB
```

**Training Time:**

| Config | GPU | Time per Epoch | Total Time (100 epochs) |
|--------|-----|----------------|-------------------------|
| Small (filters=32, blocks=2) | RTX 3070 | 15s | 25 min |
| Medium (filters=128, blocks=4) | RTX 3070 | 40s | 60 min |
| Large (filters=256, blocks=8) | RTX 3090 | 90s | 2.5 hours |

**Inference Latency:**

- **GPU:** 5-15ms per sample
- **CPU:** 50-150ms per sample
- **Production:** GPU recommended

**Dependencies:**

```bash
pip install torch==2.1.0
pip install sktime==0.25.0  # For InceptionTime
```

---

## Foundation Models

### Chronos, TimesFM, Moirai, TimeGPT

**Hardware Requirements:**

- **CPU:** 16+ cores (for preprocessing)
- **GPU:** Required (models are 200M-1B parameters)
  - **Chronos-t5-small (20M params):** 4GB VRAM
  - **Chronos-t5-base (200M params):** 8GB VRAM
  - **Chronos-t5-large (710M params):** 16GB VRAM
  - **TimesFM (200M params):** 12GB VRAM
  - **Moirai (311M params):** 12GB VRAM
- **RAM:** 32GB minimum
- **Disk:** 200MB-2GB (pre-trained weights)

**Inference-Only (No Training):**

Foundation models are used **inference-only** in this pipeline. Fine-tuning is possible but not recommended for zero-shot performance.

**GPU Memory Scaling:**

```python
def estimate_foundation_model_memory(model_name, batch_size=1):
    """
    Estimate foundation model GPU memory.

    Args:
        model_name: 'chronos-small', 'chronos-base', 'chronos-large', 'timesfm'
        batch_size: Batch size for inference

    Returns:
        Memory in MB
    """
    model_sizes = {
        'chronos-small': 20e6,    # 20M parameters
        'chronos-base': 200e6,    # 200M parameters
        'chronos-large': 710e6,   # 710M parameters
        'timesfm': 200e6,         # 200M parameters
    }

    num_params = model_sizes.get(model_name, 200e6)

    # Model weights (float32)
    weight_memory = num_params * 4

    # Activations (approximate)
    context_len = 512
    d_model = 1024  # Typical for T5-base
    activation_memory = batch_size * context_len * d_model * 4

    # KV cache for autoregressive generation
    kv_cache_memory = batch_size * context_len * d_model * 2 * 12  # 12 layers typical

    total_memory_bytes = weight_memory + activation_memory + kv_cache_memory

    total_memory_mb = total_memory_bytes / (1024 ** 2)

    return total_memory_mb

# Examples
print("Foundation Model Memory Estimates:")
print(f"  Chronos-small:  {estimate_foundation_model_memory('chronos-small'):.0f} MB")  # ~4 GB
print(f"  Chronos-base:   {estimate_foundation_model_memory('chronos-base'):.0f} MB")   # ~8 GB
print(f"  Chronos-large:  {estimate_foundation_model_memory('chronos-large'):.0f} MB")  # ~16 GB
```

**Inference Latency:**

- **Chronos-small:** 50-100ms (GPU)
- **Chronos-base:** 100-150ms (GPU)
- **Chronos-large:** 150-200ms (GPU)
- **TimesFM:** 100-150ms (GPU)

**Dependencies:**

```bash
# Chronos
pip install chronos-forecasting==1.0.0
pip install transformers==4.36.0
pip install accelerate==0.25.0

# TimesFM (Google)
pip install timesfm==1.0.0

# Hugging Face hub for model downloads
pip install huggingface-hub==0.19.4
```

**Production Deployment:**

```python
# Serve Chronos with HuggingFace Transformers
from transformers import AutoModelForSeq2SeqLM, AutoConfig
import torch

# Load pre-trained Chronos
model = AutoModelForSeq2SeqLM.from_pretrained("amazon/chronos-t5-base")
model.eval()
model.to('cuda')

@app.post("/predict")
def predict(sequence: list[float]):  # Context window: 512 bars
    # Normalize (zero-mean, unit-variance)
    seq_tensor = torch.FloatTensor(sequence).unsqueeze(0).unsqueeze(-1)  # (1, 512, 1)
    mean = seq_tensor.mean()
    std = seq_tensor.std()
    seq_norm = (seq_tensor - mean) / (std + 1e-8)

    # Generate forecast
    with torch.no_grad():
        forecast = model.generate(seq_norm.to('cuda'))

    # Denormalize
    forecast_denorm = forecast.cpu() * std + mean

    return {"forecast": forecast_denorm.tolist()}
```

**Latency: 100-200ms (GPU)**

---

## Classical Models

### Random Forest, Logistic Regression, SVM

**Hardware Requirements:**

- **CPU:** 4+ cores (16+ for SVM on large datasets)
- **GPU:** Not supported (CPU-only)
- **RAM:** 2GB minimum, 8GB for SVM
- **Disk:** 10-50MB per model

**Training Time:**

| Model | Dataset Size | Time (4 cores) | Time (16 cores) |
|-------|--------------|----------------|-----------------|
| **Random Forest** | 20k × 150 | 2-5 min | 1-2 min |
| **Logistic** | 20k × 150 | 10-30s | 5-10s |
| **SVM (RBF kernel)** | 20k × 150 | 5-10 min | 2-4 min |
| **SVM (Linear kernel)** | 20k × 150 | 1-2 min | 30-60s |

**Inference Latency:**

- **Random Forest:** <1ms per sample
- **Logistic:** <0.5ms per sample
- **SVM:** <1ms per sample

**Dependencies:**

```bash
pip install scikit-learn==1.3.2
pip install numpy==1.24.3
```

**Production Deployment:**

```python
# Serve with scikit-learn
import joblib
from sklearn.ensemble import RandomForestClassifier

model = joblib.load('models/random_forest.pkl')

@app.post("/predict")
def predict(features: list[float]):
    X = np.array(features).reshape(1, -1)
    proba = model.predict_proba(X)
    return {"probabilities": proba.tolist()}
```

**Latency: <1ms (CPU)**

---

## Ensemble Models

### Voting, Stacking, Blending

**Hardware Requirements:**

Depends on base models:

- **All boosting ensemble:** CPU-only, 8GB RAM
- **All neural ensemble:** GPU required, 24GB VRAM
- **Mixed (boosting + neural):** GPU required (for neural), 32GB RAM

**Training Time:**

- **Voting:** Sum of base model training times
- **Stacking:** Sum of base models + meta-learner (~5 min)
- **Blending:** Same as stacking

**Inference Latency:**

- **Voting:** Max latency of base models
- **Stacking:** Sum of base model latencies + meta-learner (<1ms)

**Example: Boosting Ensemble**

```python
# Ensemble of XGBoost + LightGBM + CatBoost
# Hardware: CPU-only, 8GB RAM
# Training time: 3 models × 3 min = 9 min
# Inference: 3 models × <1ms = <3ms
```

**Example: Neural Ensemble**

```python
# Ensemble of LSTM + GRU + TCN
# Hardware: GPU required (24GB VRAM), 32GB RAM
# Training time: 3 models × 30 min = 90 min
# Inference: 3 models × 10ms = 30ms (GPU)
```

**Storage:**

- **Voting:** Sum of base model sizes
- **Stacking:** Sum of base models + OOF predictions (~100MB per base) + meta-learner (~10MB)

**Example Stacking Storage:**

```
models/stacking/
├── base_models/
│   ├── xgboost/         50MB
│   ├── lightgbm/        50MB
│   └── catboost/        50MB
├── oof_predictions/
│   ├── xgboost_oof.npy  100MB
│   ├── lightgbm_oof.npy 100MB
│   └── catboost_oof.npy 100MB
└── meta_learner/
    └── logistic.pkl     10MB

Total: ~510MB
```

---

## Infrastructure Recommendations

### Small Dataset (1 year, 1 symbol, ~50k bars)

**Hardware:**

- **CPU:** 8 cores (Intel i7, AMD Ryzen 7)
- **RAM:** 16GB
- **GPU:** 1× RTX 3070 (8GB VRAM)
- **Disk:** 100GB SSD

**Models supported:**

- All boosting models
- All classical models
- LSTM/GRU/TCN (small configs)
- InceptionTime/ResNet (small configs)
- **NOT supported:** Large transformers, foundation models

**Pipeline runtime:** 10-15 min

---

### Medium Dataset (3 years, 1 symbol, ~150k bars)

**Hardware:**

- **CPU:** 16 cores (Intel i9, AMD Ryzen 9, Threadripper)
- **RAM:** 32GB
- **GPU:** 1× RTX 4080 (16GB VRAM) or 2× RTX 3070
- **Disk:** 500GB SSD

**Models supported:**

- All boosting, classical models
- All neural sequence models
- PatchTST (medium configs)
- Chronos-base, TimesFM
- **NOT supported:** Chronos-large, very large transformers

**Pipeline runtime:** 30-45 min

---

### Large Dataset (5+ years, multiple symbols, 500k+ bars)

**Hardware:**

- **CPU:** 32+ cores (Threadripper, Xeon)
- **RAM:** 64-128GB
- **GPU:** 2× RTX 3090 (24GB each) or 1× A100 (40GB)
- **Disk:** 1TB NVMe SSD

**Models supported:**

- All models
- Chronos-large
- Large transformers (d_model=512, layers=8)
- Multi-symbol ensembles

**Pipeline runtime:** 1-2 hours

---

### Production Deployment

**Inference Server:**

- **CPU:** 8+ cores (for request handling, preprocessing)
- **RAM:** 16GB
- **GPU:** Optional (depends on model)
  - **Boosting/Classical:** No GPU needed
  - **Neural:** 1× RTX 3070 (8GB) minimum
  - **Transformers/Foundation:** 1× RTX 4090 (24GB) recommended

**Latency Requirements:**

| Use Case | Max Latency | Recommended Models |
|----------|-------------|-------------------|
| **Real-time trading (1min bars)** | <10ms | Boosting, Classical, LSTM (GPU) |
| **Near-real-time (5min bars)** | <50ms | All neural models (GPU) |
| **Batch inference (end-of-day)** | <1s | All models |

**Serving Framework:**

```bash
# Option 1: FastAPI (simple, flexible)
pip install fastapi uvicorn

# Option 2: TorchServe (production-grade for PyTorch)
pip install torchserve torch-model-archiver

# Option 3: ONNX Runtime (fastest inference)
pip install onnx onnxruntime-gpu
```

---

## Data Pipeline Consumption

All models consume standardized outputs from Phase 1 via `TimeSeriesDataContainer`:

### Tabular Models (2D)

```python
# src/phase1/stages/datasets/container.py
from pathlib import Path

# Load container
container = TimeSeriesDataContainer.load(Path("data/splits/scaled/run_20250101"))

# Get tabular data (2D arrays)
X_train, y_train, weights_train = container.get_tabular_data(split='train')
X_val, y_val, weights_val = container.get_tabular_data(split='val')
X_test, y_test, weights_test = container.get_tabular_data(split='test')

# Shapes
# X_train: (n_train_samples, n_features)  e.g., (12000, 150)
# y_train: (n_train_samples,)             e.g., (12000,)
# weights_train: (n_train_samples,)       e.g., (12000,)
```

### Sequence Models (3D)

```python
# Get sequence data (3D arrays)
X_train, y_train, weights_train = container.get_sequence_data(
    split='train',
    seq_len=60  # Last 60 bars for each prediction
)

# Shapes
# X_train: (n_train_samples, seq_len, n_features)  e.g., (11940, 60, 25)
# Note: n_samples reduced by seq_len due to lookback requirement
```

### Multi-Resolution Models (4D)

```python
# Get multi-resolution tensors (dict of 3D arrays)
X_multi_train, y_train, weights_train = container.get_multi_resolution_tensors(
    split='train',
    input_timeframes=['1min', '5min', '15min']
)

# Returns dict of tensors
# X_multi_train = {
#     '1min': (n_samples, 15, 5),   # Last 15 minutes at 1min
#     '5min': (n_samples, 3, 5),    # Last 15 minutes at 5min
#     '15min': (n_samples, 1, 5),   # Current 15min bar
# }

# Stack or concatenate as needed
from src.phase1.stages.datasets.tensor_utils import stack_multi_resolution

X_train = stack_multi_resolution(X_multi_train, timeframes=['1min', '5min', '15min'])
# Shape: (n_samples, 3, max_seq_len, 5)  e.g., (11940, 3, 15, 5)
```

---

## Memory Estimation Utilities

### GPU Memory Estimator

```python
# src/models/device.py
import torch

def get_available_gpu_memory() -> float:
    """
    Get available GPU memory in MB.

    Returns:
        Available memory in MB, or 0.0 if no GPU
    """
    if not torch.cuda.is_available():
        return 0.0

    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    available_memory_mb = (total_memory - allocated_memory) / (1024 ** 2)

    return available_memory_mb

def recommend_batch_size(
    model_type: str,
    seq_len: int,
    hidden_size: int,
    available_memory_mb: float
) -> int:
    """
    Recommend batch size based on available GPU memory.

    Args:
        model_type: 'lstm', 'transformer', 'cnn'
        seq_len: Sequence length
        hidden_size: Hidden dimension
        available_memory_mb: Available GPU memory

    Returns:
        Recommended batch size
    """
    if model_type == 'lstm':
        # Estimate memory per sample
        memory_per_sample = seq_len * hidden_size * 4 * 4 / (1024 ** 2)  # Approximate

    elif model_type == 'transformer':
        # Attention is quadratic!
        memory_per_sample = seq_len * seq_len * hidden_size * 4 / (1024 ** 2)

    elif model_type == 'cnn':
        memory_per_sample = seq_len * hidden_size * 4 / (1024 ** 2)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Reserve 20% for PyTorch overhead
    usable_memory = available_memory_mb * 0.8

    # Compute max batch size
    max_batch_size = int(usable_memory / memory_per_sample)

    # Clamp to reasonable range
    recommended_batch_size = min(max(max_batch_size, 8), 512)

    return recommended_batch_size

# Example usage
available_mem = get_available_gpu_memory()
batch_size = recommend_batch_size('lstm', seq_len=60, hidden_size=128, available_memory_mb=available_mem)
print(f"Available GPU memory: {available_mem:.0f} MB")
print(f"Recommended batch size: {batch_size}")
```

### Gradient Accumulation for Limited Memory

```python
# If GPU memory is limited, use gradient accumulation
def fit_with_gradient_accumulation(
    model, X_train, y_train,
    effective_batch_size: int = 256,
    physical_batch_size: int = 32  # Fits in GPU
):
    """
    Train with gradient accumulation to simulate larger batch size.

    Args:
        effective_batch_size: Desired batch size
        physical_batch_size: Actual batch size that fits in GPU
    """
    accumulation_steps = effective_batch_size // physical_batch_size

    optimizer.zero_grad()

    for i, (X_batch, y_batch) in enumerate(get_batches(X_train, y_train, physical_batch_size)):
        # Forward pass
        loss = model.forward(X_batch, y_batch)

        # Backward pass (accumulate gradients)
        loss = loss / accumulation_steps
        loss.backward()

        # Update weights every accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
```

### Mixed Precision Training (Float16)

```python
# Use mixed precision to reduce memory 2x
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(max_epochs):
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()

        # Forward in float16
        with autocast():
            output = model(X_batch)
            loss = criterion(output, y_batch)

        # Backward with scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

**Memory savings: ~2x, with minimal accuracy loss**

---

## Summary

### Model Infrastructure Matrix

| Model Family | Min RAM | GPU Required | Min VRAM | Training Time | Inference Latency | Recommended GPU |
|--------------|---------|--------------|----------|---------------|-------------------|-----------------|
| **Boosting** | 4GB | No | - | 2-5 min | <1ms | None (CPU) |
| **Classical** | 2GB | No | - | 1-10 min | <1ms | None (CPU) |
| **LSTM/GRU/TCN** | 16GB | Yes | 8GB | 20-60 min | 5-10ms | RTX 3070+ |
| **CNN** | 16GB | Recommended | 8GB | 30-90 min | 5-15ms | RTX 3070+ |
| **Transformers** | 32GB | Yes | 16GB+ | 1-3 hours | 10-50ms | RTX 4090, A100 |
| **Foundation** | 32GB | Yes | 16GB+ | Inference-only | 50-200ms | RTX 4090, A100 |
| **Ensemble** | Varies | Varies | Varies | 2x base | Sum of base | Depends |

### Key Takeaways

1. **Boosting/Classical:** CPU-only, fast, low latency (<1ms)
2. **Neural:** GPU strongly recommended (10-50x speedup)
3. **Transformers:** GPU required, high memory (16GB+ VRAM)
4. **Foundation models:** Inference-only, GPU required, 200MB-2GB disk
5. **Attention is quadratic:** Transformer memory scales O(seq_len²)
6. **Use mixed precision:** 2x memory savings for neural models
7. **Gradient accumulation:** Simulate large batches on small GPUs

### File Paths Reference

- Device utilities: `src/models/device.py`
- Data container: `src/phase1/stages/datasets/container.py`
- Tensor utilities: `src/phase1/stages/datasets/tensor_utils.py`

---

**Production deployment checklist:**

- ✅ Estimate GPU memory for your model config
- ✅ Choose batch size based on available memory
- ✅ Use mixed precision (float16) for 2x memory savings
- ✅ Monitor latency requirements (real-time vs batch)
- ✅ Use CPU for boosting/classical (no GPU needed)
- ✅ Reserve 20% GPU memory for PyTorch overhead
- ✅ Test inference latency before production deployment
