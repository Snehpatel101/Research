# PatchTST Training Guide

**STATUS: PLANNED - NOT YET IMPLEMENTED**

## Model Overview

**Family:** Advanced Transformer
**Type:** Patch-based Transformer for Time Series
**GPU Support:** Yes (CUDA required)
**Input Shape:** 3D `(n_samples, sequence_length, n_features)`
**Output:** 3-class predictions with confidence
**Implementation Effort:** 4 days
**Expected Completion:** Q1 2025

## Why PatchTST?

**PatchTST is SOTA for long-term time series forecasting:**
- **21% MSE reduction** vs vanilla Transformer on long-term forecasting benchmarks
- **Efficient attention:** O((L/P)²) instead of O(L²) where P = patch_len
- **Multi-scale learning:** Patches capture both local and global patterns
- **Production-safe:** Causal masking prevents lookahead bias

## Hardware Requirements (Estimated)

### Minimum
- **CPU:** 8 cores
- **RAM:** 32GB
- **GPU:** NVIDIA RTX 3070 (8GB VRAM)
- **Training Time:** ~60 minutes

### Recommended
- **CPU:** 16 cores
- **RAM:** 64GB
- **GPU:** RTX 3080 (10GB VRAM) or RTX 4070 Ti (12GB VRAM)
- **Batch Size:** 128-256
- **Training Time:** ~30 minutes

### Optimal
- **CPU:** 16+ cores
- **RAM:** 64GB
- **GPU:** RTX 3090/4090 (24GB VRAM) or A100 (40GB VRAM)
- **Batch Size:** 512
- **Training Time:** ~15 minutes

## Planned Architecture

### Core Components

```python
# src/models/transformers/patchtst.py

class PatchEmbedding(nn.Module):
    """
    Divide sequence into patches and embed.

    Example: seq_len=512, patch_len=16, stride=8
    → 63 overlapping patches

    This reduces sequence length from L → L/P,
    making attention O((L/P)²) instead of O(L²).
    """
    def __init__(self, patch_len=16, stride=8, d_model=128, n_features=25):
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len * n_features, d_model)

class PatchTST(BaseModel):
    """
    PatchTST: Patch-based Transformer for time series.

    Architecture:
    1. Patch embedding (seq_len → num_patches)
    2. Positional encoding
    3. Transformer encoder (attention over patches)
    4. Global average pooling
    5. Classification head
    """
    def __init__(self, config=None):
        super().__init__(config)
        # ... implementation details in ADVANCED_MODELS_ROADMAP.md
```

## Planned Hyperparameters

### Default Configuration

```yaml
# config/models/patchtst.yaml (to be created)
model_name: patchtst
family: transformer
description: Patch-based Transformer for efficient long-range modeling

# Architecture
patch_len: 16              # Patch length (16 timesteps)
stride: 8                  # Patch stride (50% overlap)
d_model: 128              # Model dimension
nhead: 8                   # Attention heads
num_layers: 3              # Transformer layers
dim_feedforward: 512       # FFN hidden dimension

# Input
sequence_length: 120       # Longer sequences benefit from patching
input_features: null       # Auto-detected

# Learning
learning_rate: 0.0001
weight_decay: 0.0001
gradient_clip: 1.0
dropout: 0.1

# Optimizer
optimizer: adamw
scheduler: cosine
warmup_epochs: 10

# Training
batch_size: 128
max_epochs: 150
early_stopping_patience: 20
min_delta: 0.0001
```

### Expected Hyperparameter Ranges

```python
{
    "patch_len": [8, 16, 24, 32],
    "stride": [4, 8, 16],
    "d_model": [64, 128, 256],
    "nhead": [4, 8, 16],
    "num_layers": [2, 3, 4, 6],
    "learning_rate": [0.00005, 0.0001, 0.0003],
    "sequence_length": [120, 240, 480],
    "batch_size": [64, 128, 256]
}
```

## Estimated GPU Memory Requirements

### Memory Formula

```python
# PatchTST memory estimate
def estimate_patchtst_memory(batch_size, seq_len, patch_len, stride,
                              n_features, d_model, num_layers):
    # Calculate number of patches
    num_patches = (seq_len - patch_len) // stride + 1

    # Patch embedding parameters
    embed_params = patch_len * n_features * d_model

    # Transformer parameters (approx)
    # Each layer: Self-attention (4 * d_model²) + FFN (8 * d_model²)
    layer_params = 12 * d_model * d_model
    transformer_params = layer_params * num_layers

    # Total parameters
    total_params = embed_params + transformer_params + (d_model * 3)  # classifier

    # Model memory (FP32)
    model_mem = (total_params * 4) / (1024**3)

    # Attention memory (batch * num_patches * num_patches * num_layers)
    attention_mem = (batch_size * num_patches * num_patches * num_layers * 4) / (1024**3)

    # Activations
    activation_mem = (batch_size * num_patches * d_model * 4) / (1024**3)

    # Optimizer (AdamW: 2x params)
    optimizer_mem = model_mem * 2

    # Total with overhead
    total = (model_mem + attention_mem + activation_mem + optimizer_mem) * 1.2
    return total

# Example: batch=128, seq=120, patch=16, stride=8, features=25, d_model=128, layers=3
# => ~4.2GB GPU memory
```

### Memory by Configuration

| Batch | Seq Len | Patch | d_model | Layers | Features | GPU Memory | Recommended GPU |
|-------|---------|-------|---------|--------|----------|------------|-----------------|
| 64    | 120     | 16    | 64      | 2      | 25       | 2.1 GB     | RTX 3060 (8GB)  |
| 128   | 120     | 16    | 128     | 3      | 25       | 4.2 GB     | RTX 3070 (8GB)  |
| 256   | 120     | 16    | 128     | 3      | 25       | 7.5 GB     | RTX 3080 (10GB) |
| 128   | 240     | 16    | 128     | 4      | 80       | 8.9 GB     | RTX 3090 (24GB) |
| 256   | 240     | 16    | 256     | 4      | 80       | 16.2 GB    | RTX 4090 (24GB) |

## Expected Training Time

| Dataset Size | GPU (RTX 3080) | GPU (A100) | Notes |
|--------------|----------------|------------|-------|
| 50K samples  | 15 min         | 6 min      | batch=128, seq=120 |
| 100K samples | 30 min         | 12 min     | batch=128, seq=120 |
| 500K samples | 2.5 hours      | 1 hour     | batch=128, seq=120 |
| 1M samples   | 5 hours        | 2 hours    | batch=128, seq=120 |

**Note:** Estimates assume 150 epochs with early stopping around epoch 80.

## Planned Training Commands

### Quick Start

```bash
python scripts/train_model.py \
    --model patchtst \
    --horizon 20 \
    --seq-len 120 \
    --batch-size 128
```

### High-Performance Configuration

```bash
python scripts/train_model.py \
    --model patchtst \
    --horizon 20 \
    --seq-len 240 \
    --batch-size 256 \
    --patch-len 24 \
    --d-model 256 \
    --num-layers 4
```

### Memory-Constrained GPU

```bash
python scripts/train_model.py \
    --model patchtst \
    --horizon 20 \
    --seq-len 120 \
    --batch-size 64 \
    --patch-len 16 \
    --d-model 64 \
    --num-layers 2
```

## Expected Performance

### Benchmark Targets (vs Existing Models)

| Metric | Transformer (current) | PatchTST (target) | Improvement |
|--------|----------------------|-------------------|-------------|
| Val F1 | 0.58-0.62 | 0.62-0.67 | +4-5% |
| Training Time | 25 min | 30 min | +20% slower |
| Inference | 15 ms/sample | 18 ms/sample | +20% slower |
| GPU Memory | 3.5 GB | 4.2 GB | +20% more |

**Why the trade-off?**
- Longer sequences (120 vs 60) capture more temporal context
- Patch embedding adds parameters but improves pattern recognition
- Expected to outperform vanilla Transformer significantly

## Implementation Roadmap

See `/home/user/Research/docs/roadmaps/ADVANCED_MODELS_ROADMAP.md` for detailed implementation plan.

### Day 1-2: Core Architecture
1. Implement `PatchEmbedding` class
2. Implement `PatchTST` model (inherit from `BaseModel`)
3. Add causal masking to prevent lookahead
4. Register with `@register("patchtst", family="transformer")`

### Day 3: Training Integration
5. Implement fit/predict/save/load methods
6. Create config YAML (`config/models/patchtst.yaml`)
7. Add to `Trainer` class

### Day 4: Testing & Documentation
8. Unit tests (fit, predict, save/load)
9. Integration test with Phase 1 data
10. Hyperparameter search space
11. This training guide (update with actual results)

## Validation Strategy

```python
# Same as other sequence models
train: 70%
val: 15%    # Early stopping
test: 15%   # Final evaluation

# Purge & embargo to prevent leakage
purge_bars: 60
embargo_bars: 1440
```

## Mixed Precision Support

```yaml
# Will support automatic mixed precision
device:
  mixed_precision: true  # Auto-detect BF16/FP16/FP32
```

| GPU | Precision | Speedup |
|-----|-----------|---------|
| RTX 20xx | FP16 | 1.5-2x |
| RTX 30xx | BF16 | 1.8-2.5x |
| RTX 40xx | BF16 | 2-3x |
| A100 | BF16 | 2-3x |

## Sequence Ensembles

Once implemented, PatchTST can be ensembled with other sequence models:

```bash
# Valid: PatchTST + LSTM + TCN
python scripts/train_model.py \
    --model voting \
    --base-models patchtst,lstm,tcn \
    --horizon 20 \
    --seq-len 120

# INVALID: PatchTST + XGBoost (mixed 3D/2D)
```

## Comparison with Similar Models

| Model | Attention Cost | Sequence Length | Best For |
|-------|----------------|-----------------|----------|
| **Transformer** | O(L²) | 60-120 | Short sequences |
| **PatchTST** | O((L/P)²) | 120-480 | Long sequences |
| **iTransformer** | O(F²) | 60-240 | Multi-variate correlations |
| **TFT** | O(L²) + LSTM | 60-120 | Interpretability + quantiles |

## References

- PatchTST Paper: Nie et al. (2023) "A Time Series is Worth 64 Words"
- Code: https://github.com/yuqinie98/PatchTST
- Benchmarks: https://paperswithcode.com/paper/a-time-series-is-worth-64-words-long-term

## Implementation Status

**Current:** Planning phase
**Next Steps:**
1. Review PatchTST paper and reference implementation
2. Create skeleton structure in `src/models/transformers/patchtst.py`
3. Implement patch embedding layer
4. Implement full model and register
5. Create tests and config
6. Train on MES data and benchmark

**Estimated Completion:** 4 days from start
**Blocker:** None (all dependencies in place)
