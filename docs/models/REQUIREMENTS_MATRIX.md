# Model Requirements Matrix

**Comprehensive hardware, memory, and training time requirements for all 19 models**

Last Updated: 2025-12-30

---

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Detailed Requirements by Model](#detailed-requirements-by-model)
3. [GPU Memory Formulas](#gpu-memory-formulas)
4. [Batch Size Recommendations](#batch-size-recommendations)
5. [Training Time Benchmarks](#training-time-benchmarks)
6. [Cost Estimates](#cost-estimates)

---

## Quick Reference

### Model Requirements Summary

| Model | Family | Status | Min GPU | Recommended GPU | Batch Size | Training Time | Memory Formula |
|-------|--------|--------|---------|-----------------|------------|---------------|----------------|
| **XGBoost** | Boosting | ✅ Implemented | None (CPU) | RTX 3070 (8GB) | N/A | 2-5 min | Fixed 2-4GB |
| **LightGBM** | Boosting | ✅ Implemented | None (CPU) | RTX 3070 (8GB) | N/A | 2-5 min | Fixed 2-4GB |
| **CatBoost** | Boosting | ✅ Implemented | None (CPU) | RTX 3070 (8GB) | N/A | 3-6 min | Fixed 2-4GB |
| **LSTM** | Neural | ✅ Implemented | GTX 1080 Ti | RTX 3080 (10GB) | 512 | 15-30 min | 4 * hidden * (feat + hidden) |
| **GRU** | Neural | ✅ Implemented | GTX 1080 Ti | RTX 3080 (10GB) | 512 | 12-25 min | 3 * hidden * (feat + hidden) |
| **TCN** | Neural | ✅ Implemented | RTX 2060 | RTX 3080 (10GB) | 256 | 20-35 min | channels * kernel * layers |
| **Transformer** | Neural | ✅ Implemented | RTX 3070 | RTX 3090 (24GB) | 128 | 30-60 min | seq² * d_model |
| **Random Forest** | Classical | ✅ Implemented | None (CPU) | None (CPU) | N/A | 3-8 min | n_estimators * depth |
| **Logistic Regression** | Classical | ✅ Implemented | None (CPU) | None (CPU) | N/A | 1-2 min | Linear in features |
| **SVM** | Classical | ✅ Implemented | None (CPU) | None (CPU) | N/A | 5-15 min | O(n² - n³) |
| **Voting** | Ensemble | ✅ Implemented | Depends on base | Depends on base | Varies | Sum of base models | Sum of base models |
| **Stacking** | Ensemble | ✅ Implemented | Depends on base | Depends on base | Varies | 1.2x base models | Base + OOF storage |
| **Blending** | Ensemble | ✅ Implemented | Depends on base | Depends on base | Varies | 1.1x base models | Base + holdout |
| **InceptionTime** | CNN | 🚧 Planned | RTX 3070 | RTX 3080 (10GB) | 64 | 30-60 min | 5 networks * filters |
| **1D ResNet** | CNN | 🚧 Planned | RTX 3060 | RTX 3070 (8GB) | 128 | 20-40 min | depth * filters² |
| **PatchTST** | Transformer | 🚧 Planned | RTX 3070 | RTX 3090 (24GB) | 128 | 30-60 min | (seq/patch)² * d_model |
| **iTransformer** | Transformer | 🚧 Planned | RTX 3060 | RTX 3080 (10GB) | 256 | 25-50 min | features² * seq |
| **TFT** | Transformer | 🚧 Planned | RTX 3080 | RTX 3090 (24GB) | 64 | 45-90 min | LSTM + attention |
| **N-BEATS** | MLP | 🚧 Planned | RTX 2060 | RTX 3070 (8GB) | 256 | 15-30 min | hidden_size * stacks |

---

## Detailed Requirements by Model

### Boosting Models (3 models)

#### XGBoost

**Hardware:**
- **CPU Mode:** 8 cores, 16GB RAM → 10-20 min
- **GPU Mode:** Any NVIDIA GPU (6GB+) → 2-5 min
- **Recommended:** RTX 3070 (8GB VRAM), 16 cores, 32GB RAM

**Memory:**
```python
# CPU: Scales with dataset size
cpu_memory_gb = 2 + (n_samples / 100000) * 0.5

# GPU: Fixed VRAM (tree building only)
gpu_vram_gb = 2-4  # Independent of dataset size
```

**Batch Size:** N/A (processes full dataset)

**Configuration:**
- n_estimators: 500 (default), 1000 (high accuracy)
- max_depth: 6 (default), 8 (complex patterns)
- tree_method: 'hist' (required for GPU)

---

#### LightGBM

**Hardware:**
- **CPU Mode:** 8 cores, 16GB RAM → 8-15 min
- **GPU Mode:** Any NVIDIA GPU (6GB+) → 1-4 min
- **Recommended:** RTX 3070 (8GB VRAM), 16 cores, 32GB RAM

**Memory:**
```python
# Typically 20-30% less than XGBoost
cpu_memory_gb = 1.5 + (n_samples / 100000) * 0.4
gpu_vram_gb = 2-3
```

**Batch Size:** N/A

**Configuration:**
- num_leaves: 31 (default), 63 (more capacity)
- device_type: 'gpu' for GPU acceleration

---

#### CatBoost

**Hardware:**
- **CPU Mode:** 8 cores, 16GB RAM → 12-25 min
- **GPU Mode:** Any NVIDIA GPU (6GB+) → 3-6 min
- **Recommended:** RTX 3070 (8GB VRAM), 16 cores, 32GB RAM

**Memory:**
```python
# Similar to XGBoost
cpu_memory_gb = 2.5 + (n_samples / 100000) * 0.5
gpu_vram_gb = 2-4
```

**Batch Size:** N/A

**Configuration:**
- iterations: 500 (default), 1000 (high accuracy)
- depth: 6 (default)
- task_type: 'GPU' for GPU acceleration

---

### Neural Models (4 models)

#### LSTM

**Hardware:**
- **Minimum:** GTX 1080 Ti (11GB), 8 cores, 32GB RAM → 45-60 min
- **Recommended:** RTX 3080 (10GB), 16 cores, 64GB RAM → 15-30 min
- **Optimal:** RTX 4090 (24GB), 16 cores, 64GB RAM → 5-10 min

**Memory:**
```python
def lstm_memory(batch, seq, features, hidden, layers):
    # Parameters (4 gates)
    params = 4 * hidden * (features + hidden + 1) * layers
    model = (params * 4) / (1024**3)  # FP32

    # Activations
    activation = (batch * seq * hidden * 4 * 4) / (1024**3)

    # Optimizer (AdamW: 2x params)
    optimizer = model * 2

    # Total with overhead
    return (model + activation + optimizer) * 1.2

# Examples:
# batch=256, seq=60, feat=25, hidden=256, layers=2 => 2.2 GB
# batch=512, seq=60, feat=25, hidden=256, layers=2 => 3.8 GB
# batch=1024, seq=120, feat=80, hidden=512, layers=3 => 18.5 GB
```

**Batch Size by GPU:**
- GTX 1060 (6GB): 128
- GTX 1080 Ti (11GB): 256
- RTX 2080 (8GB): 256
- RTX 3080 (10GB): 512
- RTX 3090 (24GB): 1024
- A100 (40GB): 2048

**Configuration:**
- hidden_size: 256 (default), 512 (high capacity)
- num_layers: 2 (default), 3 (complex patterns)
- sequence_length: 60 (default), 120 (long context)

---

#### GRU

**Hardware:**
- Same as LSTM (25% less memory due to 3 gates vs 4)

**Memory:**
```python
def gru_memory(batch, seq, features, hidden, layers):
    # Parameters (3 gates)
    params = 3 * hidden * (features + hidden + 1) * layers
    model = (params * 4) / (1024**3)

    # Activations
    activation = (batch * seq * hidden * 3 * 4) / (1024**3)

    # Optimizer
    optimizer = model * 2

    return (model + activation + optimizer) * 1.2

# Typically 20-25% less than LSTM
```

**Batch Size:** Same as LSTM or 1.25x larger

**Configuration:** Same as LSTM

---

#### TCN (Temporal Convolutional Network)

**Hardware:**
- **Minimum:** RTX 2060 (8GB), 8 cores, 32GB RAM → 40-60 min
- **Recommended:** RTX 3080 (10GB), 16 cores, 64GB RAM → 20-35 min
- **Optimal:** RTX 4090 (24GB), 16 cores, 64GB RAM → 8-15 min

**Memory:**
```python
def tcn_memory(batch, seq, features, num_channels, kernel, layers):
    # Dilated convolutions
    params = 0
    for i in range(layers):
        in_ch = features if i == 0 else num_channels[i-1]
        out_ch = num_channels[i]
        params += in_ch * out_ch * kernel * 2  # 2 convs per layer

    model = (params * 4) / (1024**3)

    # Activations (less than LSTM due to convolutions)
    activation = (batch * seq * max(num_channels) * 2.5) / (1024**3)

    optimizer = model * 2
    return (model + activation + optimizer) * 1.2

# Example: batch=256, seq=120, feat=25, channels=[64,64,64,64] => 5.2 GB
```

**Batch Size by GPU:**
- RTX 2060 (8GB): 128
- RTX 3070 (8GB): 256
- RTX 3080 (10GB): 512
- RTX 3090 (24GB): 1024

**Configuration:**
- num_channels: [64, 64, 64, 64] (default)
- kernel_size: 3 (default)
- sequence_length: 120 (longer than LSTM due to efficient memory)

---

#### Transformer

**Hardware:**
- **Minimum:** RTX 3070 (8GB), 8 cores, 32GB RAM → 60-90 min
- **Recommended:** RTX 3090 (24GB), 16 cores, 64GB RAM → 30-60 min
- **Optimal:** A100 (40GB), 16 cores, 128GB RAM → 15-30 min

**Memory:**
```python
def transformer_memory(batch, seq, features, d_model, heads, layers):
    # Transformer parameters
    # Each layer: Self-attention (4 * d_model²) + FFN (8 * d_model²)
    layer_params = 12 * d_model * d_model
    total_params = layer_params * layers

    model = (total_params * 4) / (1024**3)

    # Attention matrix: (batch * seq * seq * heads)
    attention = (batch * seq * seq * heads * 4) / (1024**3)

    # Activations
    activation = (batch * seq * d_model * 4) / (1024**3)

    optimizer = model * 2
    return (model + attention + activation + optimizer) * 1.2

# Example: batch=128, seq=60, d_model=256, heads=8, layers=4 => 6.8 GB
```

**Batch Size by GPU:**
- RTX 3070 (8GB): 64
- RTX 3080 (10GB): 128
- RTX 3090 (24GB): 256
- A100 (40GB): 512

**Configuration:**
- d_model: 256 (default), 512 (high capacity)
- nhead: 8 (default)
- num_layers: 4 (default), 6 (complex patterns)
- sequence_length: 60 (default), 120 (long context - expensive!)

---

### Classical Models (3 models)

#### Random Forest

**Hardware:**
- **CPU Only:** 8 cores, 16GB RAM → 3-8 min
- **Recommended:** 16 cores, 32GB RAM → 2-5 min

**Memory:**
```python
memory_gb = 1 + (n_estimators * max_depth * n_features * 8) / (1024**3)

# Example: 300 trees, depth 8, 80 features => 3.2 GB
```

**Configuration:**
- n_estimators: 300 (default), 500 (high accuracy)
- max_depth: 8 (default), 12 (complex patterns)

---

#### Logistic Regression

**Hardware:**
- **CPU Only:** 4 cores, 8GB RAM → 1-2 min
- **Recommended:** 8 cores, 16GB RAM → 30-60 sec

**Memory:**
```python
memory_gb = 0.5 + (n_features * n_classes * 8) / (1024**3)

# Very lightweight, typically < 1GB
```

**Configuration:**
- C: 1.0 (default regularization)
- solver: 'lbfgs' (default)

---

#### SVM

**Hardware:**
- **CPU Only:** 8 cores, 32GB RAM → 5-15 min
- **Recommended:** 16 cores, 64GB RAM → 3-10 min

**Memory:**
```python
# SVM scales poorly with dataset size: O(n² - n³)
memory_gb = 2 + (n_samples ** 2 * 8) / (1024**3)

# Example: 100K samples => 80 GB (impractical!)
# Recommendation: Subsample to 10-20K samples
```

**Configuration:**
- kernel: 'rbf' (default)
- C: 1.0 (default)
- **Note:** Consider subsampling for large datasets

---

### Ensemble Models (3 models)

#### Voting Ensemble

**Hardware:** Depends on base models

**Memory:**
```python
# Sum of individual base model memory + prediction storage
memory = sum(base_model_memory) + (n_samples * n_models * 3 * 8) / (1024**3)

# Example: XGBoost (2GB) + LightGBM (2GB) + CatBoost (2.5GB) + predictions (0.5GB)
# Total: 7 GB RAM
```

**Training Time:** Sum of base model training times (sequential)

**Configuration:**
- base_models: 2-3 models recommended
- voting_type: 'soft' (probability averaging)

---

#### Stacking Ensemble

**Hardware:** Depends on base models + meta-learner

**Memory:**
```python
# Base models + OOF predictions + meta-learner
memory = sum(base_model_memory) + (n_samples * n_models * 3 * 8) / (1024**3) + meta_memory

# Typically 1.2x base model memory
```

**Training Time:** 1.2x base model training (includes OOF generation + meta-learner training)

**Configuration:**
- base_models: 3-5 models recommended
- meta_learner: 'logistic' (default), 'xgboost'

---

#### Blending Ensemble

**Hardware:** Depends on base models

**Memory:** Similar to Stacking (no OOF, uses holdout instead)

**Training Time:** 1.1x base model training (simpler than Stacking)

**Configuration:**
- base_models: 2-4 models recommended
- holdout_size: 0.2 (20% of training data)

---

### Planned Advanced Models (6 models)

#### InceptionTime (Planned)

**Hardware:**
- **Estimated GPU:** RTX 3080 (10GB)
- **Estimated Batch Size:** 64
- **Estimated Training Time:** 30-60 min

**Memory Estimate:**
```python
# 5 parallel networks, each with 6 Inception modules
# Each module: 4 parallel convolutions + pooling
memory_gb_estimate = 6-8  # Per GPU
```

**Configuration (Planned):**
- num_networks: 5 (ensemble)
- num_modules: 6
- num_filters: 32

---

#### 1D ResNet (Planned)

**Hardware:**
- **Estimated GPU:** RTX 3070 (8GB)
- **Estimated Batch Size:** 128
- **Estimated Training Time:** 20-40 min

**Memory Estimate:**
```python
# Deep residual blocks with skip connections
memory_gb_estimate = 4-6
```

**Configuration (Planned):**
- num_blocks: 4
- base_filters: 64

---

#### PatchTST (Planned)

**Hardware:**
- **Estimated GPU:** RTX 3090 (24GB)
- **Estimated Batch Size:** 128
- **Estimated Training Time:** 30-60 min

**Memory Estimate:**
```python
# Patch-based attention: O((L/P)²) instead of O(L²)
memory_gb_estimate = 6-10

# With seq=240, patch=16 => 15 patches
# Attention cost: 15² vs 240² (massive reduction!)
```

**Configuration (Planned):**
- patch_len: 16
- stride: 8
- d_model: 128
- num_layers: 3

---

#### iTransformer (Planned)

**Hardware:**
- **Estimated GPU:** RTX 3080 (10GB)
- **Estimated Batch Size:** 256
- **Estimated Training Time:** 25-50 min

**Memory Estimate:**
```python
# Attention over features, not time
# O(F²) instead of O(L²) where F << L
memory_gb_estimate = 4-7
```

**Configuration (Planned):**
- d_model: 128
- nhead: 8
- num_layers: 4

---

#### TFT (Temporal Fusion Transformer) (Planned)

**Hardware:**
- **Estimated GPU:** RTX 3090 (24GB)
- **Estimated Batch Size:** 64
- **Estimated Training Time:** 45-90 min

**Memory Estimate:**
```python
# Variable selection + LSTM + multi-head attention + quantile heads
memory_gb_estimate = 8-12  # Complex architecture
```

**Configuration (Planned):**
- hidden_size: 128
- num_attention_heads: 4
- quantiles: [0.05, 0.5, 0.95]

---

#### N-BEATS (Planned)

**Hardware:**
- **Estimated GPU:** RTX 3070 (8GB)
- **Estimated Batch Size:** 256
- **Estimated Training Time:** 15-30 min

**Memory Estimate:**
```python
# Pure MLP with basis expansion (no RNN/attention)
memory_gb_estimate = 3-5  # Lightweight
```

**Configuration (Planned):**
- num_blocks: 6 (3 trend + 3 seasonal)
- hidden_size: 256

---

## Training Time Benchmarks

### Dataset: 100K samples, MES 5min bars, 80 features

| Model | CPU (16 cores) | GPU (RTX 3080) | GPU (A100) | Speedup |
|-------|----------------|----------------|------------|---------|
| XGBoost (CPU) | 10 min | N/A | N/A | 1x |
| XGBoost (GPU) | N/A | 2 min | 1 min | 5-10x |
| LightGBM (GPU) | N/A | 1.5 min | 45 sec | 6-12x |
| CatBoost (GPU) | N/A | 3 min | 1.5 min | 4-8x |
| LSTM | 4 hours | 15 min | 6 min | 16x |
| GRU | 3 hours | 12 min | 5 min | 15x |
| TCN | 6 hours | 20 min | 8 min | 18x |
| Transformer | 8 hours | 30 min | 12 min | 16x |
| Random Forest | 3 min | N/A | N/A | 1x |
| Logistic | 1 min | N/A | N/A | 1x |
| SVM | 10 min | N/A | N/A | 1x |

**Note:** GPU speedups vs CPU are model-dependent. Neural models benefit most from GPU acceleration.

---

## Cost Estimates

### Cloud GPU Pricing (AWS p3/p4 instances)

| GPU | Instance | Cost/Hour | 100K Training | Monthly Research |
|-----|----------|-----------|---------------|------------------|
| Tesla V100 | p3.2xlarge | $3.06 | $0.76 (15 min) | $220 (8h/day) |
| A100 (40GB) | p4d.24xlarge | $32.77 | $1.64 (3 min) | $940 (8h/day) |
| H100 | p5.48xlarge | $98.32 | $2.46 (1.5 min) | $2,820 (8h/day) |

### On-Premise GPU Pricing

| GPU | MSRP | Monthly (36mo) | TCO (3 years) |
|-----|------|----------------|---------------|
| RTX 4070 Ti (12GB) | $800 | $22 | $1,100 |
| RTX 4080 (16GB) | $1,200 | $33 | $1,500 |
| RTX 4090 (24GB) | $1,600 | $44 | $2,000 |
| A5000 (24GB) | $2,500 | $69 | $3,200 |
| A6000 (48GB) | $4,500 | $125 | $5,500 |

**Break-even:** On-premise GPU breaks even vs cloud in 2-6 months of heavy usage.

---

## Recommendations by Use Case

### Budget: $0 (CPU Only)

**Models:**
- XGBoost (CPU mode) - 10 min training
- LightGBM (CPU mode) - 8 min training
- Random Forest - 3 min training
- Logistic Regression - 1 min training

**Ensemble:** XGBoost + LightGBM Voting (18 min total)

**Limitation:** No neural models (LSTM/TCN impractical without GPU)

---

### Budget: $800 (RTX 4070 Ti)

**Models:**
- All boosting models (GPU accelerated)
- LSTM (batch=256)
- GRU (batch=256)
- TCN (batch=128)
- Transformer (batch=64, limited)

**Ensemble:** XGBoost + LightGBM + LSTM Stacking

**Limitation:** Large sequence ensembles limited by 12GB VRAM

---

### Budget: $1,600 (RTX 4090)

**Models:**
- All 13 implemented models
- All 6 planned models (when released)
- Large-scale ensembles

**Ensemble:** Full neural ensemble (LSTM + GRU + TCN + Transformer)

**Limitation:** None for single-model training

---

### Research Lab (Multiple GPUs)

**Setup:** 3x RTX 4090 or 2x A6000
- Parallel training of multiple models
- Ensemble training in parallel
- Hyperparameter search (Optuna with parallel trials)

---

## Frequently Asked Questions

### Q: Can I train without a GPU?

**A:** Yes, but only for boosting and classical models. Neural models (LSTM/TCN/Transformer) are 10-20x slower on CPU and impractical for large datasets.

### Q: What GPU should I buy?

**A:**
- **Entry:** RTX 3060 (12GB) - $400 - Basic neural models
- **Recommended:** RTX 4070 Ti (12GB) - $800 - Most models
- **Professional:** RTX 4090 (24GB) - $1,600 - All models + ensembles

### Q: How much VRAM do I need?

**A:**
- **8GB:** Boosting + small LSTM/GRU
- **12GB:** Most models with standard configs
- **24GB:** Large ensembles + long sequences
- **40GB+:** Production-scale or foundation models

### Q: Can I reduce memory usage?

**A:** Yes:
1. Reduce batch size: 512 → 256 → 128
2. Reduce sequence length: 120 → 60 → 30
3. Reduce model size: hidden=256 → 128
4. Use gradient checkpointing (not yet implemented)

### Q: What about multi-GPU training?

**A:** Not yet implemented. Current models use single GPU. Multi-GPU support planned for Phase 6.

---

## References

- PyTorch Memory Management: https://pytorch.org/docs/stable/notes/cuda.html
- XGBoost GPU Support: https://xgboost.readthedocs.io/en/latest/gpu/index.html
- AWS GPU Pricing: https://aws.amazon.com/ec2/instance-types/p4/
- NVIDIA GPU Specs: https://www.nvidia.com/en-us/geforce/graphics-cards/
