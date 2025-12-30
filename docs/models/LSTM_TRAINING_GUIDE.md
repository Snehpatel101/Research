# LSTM Training Guide

## Model Overview

**Family:** Neural (RNN)
**Type:** Long Short-Term Memory Network
**GPU Support:** Yes (CUDA, highly recommended)
**Input Shape:** 3D `(n_samples, sequence_length, n_features)`
**Output:** 3-class predictions with probabilities

## Hardware Requirements

### Minimum
- **CPU:** 4 cores
- **RAM:** 16GB
- **GPU:** None (CPU training possible but slow)
- **Training Time:** 4-8 hours (CPU)

### Recommended
- **CPU:** 8 cores
- **RAM:** 32GB
- **GPU:** NVIDIA GTX 1080 Ti / RTX 2060 (11GB VRAM)
- **Batch Size:** 256
- **Training Time:** 15-30 minutes

### Optimal
- **CPU:** 16 cores
- **RAM:** 64GB
- **GPU:** RTX 3090 / RTX 4090 (24GB VRAM) or A100 (40GB)
- **Batch Size:** 1024
- **Training Time:** 5-10 minutes

## Hyperparameters

### Default Configuration

```yaml
# config/models/lstm.yaml
# Architecture
hidden_size: 256
num_layers: 2
dropout: 0.3
bidirectional: false

# Input
sequence_length: 60      # 60 bars of 5min data = 5 hours
input_features: null     # Auto-detected (typically 25-80)

# Learning
learning_rate: 0.001
weight_decay: 0.0001
gradient_clip: 1.0

# Optimizer
optimizer: adamw
scheduler: cosine
warmup_epochs: 5

# Training
batch_size: 512
max_epochs: 100
early_stopping_patience: 15
min_delta: 0.0001
```

### Hyperparameter Ranges

```python
{
    "hidden_size": [128, 256, 512],
    "num_layers": [1, 2, 3],
    "dropout": [0.2, 0.3, 0.4, 0.5],
    "sequence_length": [30, 60, 120],
    "learning_rate": [0.0001, 0.0005, 0.001, 0.003],
    "batch_size": [128, 256, 512, 1024],
    "weight_decay": [0.0, 0.0001, 0.001]
}
```

## Training Configuration

### Quick Start (GPU)

```bash
python scripts/train_model.py \
    --model lstm \
    --horizon 20 \
    --seq-len 60 \
    --batch-size 512
```

### High-Performance GPU

```bash
python scripts/train_model.py \
    --model lstm \
    --horizon 20 \
    --seq-len 60 \
    --batch-size 1024 \
    --hidden-size 512 \
    --num-layers 3
```

### Memory-Constrained GPU

```bash
python scripts/train_model.py \
    --model lstm \
    --horizon 20 \
    --seq-len 30 \
    --batch-size 128 \
    --hidden-size 128
```

## GPU Memory Requirements

### Memory Formula

```python
# LSTM memory estimate (in GB)
def estimate_lstm_memory(batch_size, seq_len, n_features, hidden_size, num_layers):
    # Parameters
    params = 4 * hidden_size * (n_features + hidden_size + 1) * num_layers
    model_mem = (params * 4) / (1024**3)  # 4 bytes per float32

    # Activations (4x hidden state for LSTM gates)
    activation_mem = (batch_size * seq_len * hidden_size * 4 * 4) / (1024**3)

    # Optimizer state (AdamW: 2x params for momentum + variance)
    optimizer_mem = model_mem * 2

    # Total with 20% overhead
    total = (model_mem + activation_mem + optimizer_mem) * 1.2
    return total

# Example: batch=512, seq=60, features=25, hidden=256, layers=2
# => ~3.8GB GPU memory
```

### Memory by Configuration

| Batch | Seq Len | Hidden | Layers | Features | GPU Memory | Recommended GPU |
|-------|---------|--------|--------|----------|------------|-----------------|
| 128   | 30      | 128    | 2      | 25       | 1.5 GB     | GTX 1060 (6GB)  |
| 256   | 60      | 128    | 2      | 25       | 2.2 GB     | GTX 1080 (8GB)  |
| 512   | 60      | 256    | 2      | 25       | 3.8 GB     | RTX 2080 (8GB)  |
| 1024  | 60      | 256    | 2      | 25       | 6.5 GB     | RTX 3080 (10GB) |
| 512   | 120     | 256    | 3      | 80       | 8.2 GB     | RTX 3090 (24GB) |
| 1024  | 120     | 512    | 3      | 80       | 18.5 GB    | RTX 4090 (24GB) |

## Batch Size Recommendations

### By GPU VRAM

```python
# GPU VRAM → Recommended Batch Size (seq_len=60, hidden=256, layers=2)
GPU_BATCH_SIZES = {
    "GTX 1060 (6GB)":     128,
    "GTX 1080 Ti (11GB)": 256,
    "RTX 2080 (8GB)":     256,
    "RTX 3070 (8GB)":     256,
    "RTX 3080 (10GB)":    512,
    "RTX 3090 (24GB)":    1024,
    "RTX 4070 Ti (12GB)": 512,
    "RTX 4080 (16GB)":    768,
    "RTX 4090 (24GB)":    1024,
    "Tesla T4 (15GB)":    512,
    "Tesla V100 (16GB)":  512,
    "A100 (40GB)":        2048,
}
```

### Auto-detect Optimal Batch Size

```bash
# Uses device.py to automatically select batch size
python scripts/train_model.py \
    --model lstm \
    --horizon 20 \
    --auto-batch-size
```

## Mixed Precision Training

LSTM supports automatic mixed precision for faster training:

### GPU Generation Support

| GPU Generation | Precision | Speedup | Config |
|----------------|-----------|---------|--------|
| GTX 10xx (Pascal) | FP32 | 1.0x | `mixed_precision: false` |
| RTX 20xx (Turing) | FP16 | 1.5-2x | `mixed_precision: true` (auto) |
| RTX 30xx (Ampere) | BF16 | 1.8-2.5x | `mixed_precision: true` (auto) |
| RTX 40xx (Ada) | BF16 | 2-3x | `mixed_precision: true` (auto) |
| A100 (Ampere) | BF16 | 2-3x | `mixed_precision: true` (auto) |

**Note:** Mixed precision is automatically enabled and dtype is auto-selected based on GPU capabilities.

## Learning Rate Schedule

### Cosine Annealing (Default)

```python
# Starts at learning_rate, decays to 0 following cosine curve
scheduler: cosine
warmup_epochs: 5  # Linear warmup for first 5 epochs

# Effective LR schedule:
# Epoch 0-5:   Linear 0 → 0.001
# Epoch 6-100: Cosine 0.001 → 0
```

### Alternative Schedules

```yaml
# Step decay
scheduler: step
step_size: 30
gamma: 0.1

# Exponential decay
scheduler: exponential
gamma: 0.95

# Reduce on plateau
scheduler: plateau
patience: 10
factor: 0.5
```

## Early Stopping

```yaml
early_stopping_patience: 15  # Stop if no improvement for 15 epochs
min_delta: 0.0001           # Minimum improvement threshold
```

**Logic:**
1. Track validation loss every epoch
2. If `val_loss < best_loss - min_delta`, reset patience counter
3. Otherwise, increment patience counter
4. Stop if patience counter reaches 15
5. Restore best model weights

## Training Time Estimates

| Dataset Size | GPU (RTX 3080) | GPU (A100) | CPU (16 cores) |
|--------------|----------------|------------|----------------|
| 50K samples  | 8 min          | 3 min      | 2 hours        |
| 100K samples | 15 min         | 6 min      | 4 hours        |
| 500K samples | 60 min         | 25 min     | 20 hours       |
| 1M samples   | 120 min        | 50 min     | 40 hours       |

**Note:** Times assume batch_size=512, seq_len=60, hidden=256, layers=2, 100 epochs with early stopping around epoch 50.

## Validation Strategy

```python
# Time-series split (no shuffle to prevent lookahead)
train_set: 70%  # Training data
val_set: 15%    # Validation (early stopping)
test_set: 15%   # Hold-out test set

# Purge & embargo applied to prevent label leakage
purge_bars: 60      # Remove samples near split boundary
embargo_bars: 1440  # ~5 days of 5min bars
```

## Gradient Clipping

```yaml
gradient_clip: 1.0  # Clip gradients to max norm of 1.0
```

**Why:** LSTM can suffer from exploding gradients. Clipping stabilizes training.

## Common Issues

### Issue: GPU Out of Memory

```bash
# Solution 1: Reduce batch size
--batch-size 256

# Solution 2: Reduce sequence length
--seq-len 30

# Solution 3: Reduce hidden size
--hidden-size 128

# Solution 4: Reduce layers
--num-layers 1
```

### Issue: Training too slow

```bash
# Enable mixed precision (auto-enabled on modern GPUs)
# Check GPU utilization
nvidia-smi

# If GPU usage < 80%, increase batch size
--batch-size 1024
```

### Issue: Validation loss not improving

```python
# Reduce learning rate
learning_rate: 0.0005

# Increase dropout
dropout: 0.4

# Add weight decay
weight_decay: 0.001

# Reduce model capacity
hidden_size: 128
num_layers: 1
```

### Issue: Underfitting

```python
# Increase model capacity
hidden_size: 512
num_layers: 3

# Increase sequence length
sequence_length: 120

# Reduce dropout
dropout: 0.2

# Train longer
max_epochs: 200
```

## Example Training Output

```
Initializing LSTM: hidden=256, layers=2, seq_len=60
Device: cuda:0 (NVIDIA GeForce RTX 3080)
Mixed Precision: BF16 enabled
Batch size: 512

Epoch 1/100:  train_loss=1.0856  val_loss=1.0934  val_f1=0.3421  time=12.3s
Epoch 10/100: train_loss=0.8234  val_loss=0.8512  val_f1=0.4823  time=11.8s
Epoch 20/100: train_loss=0.7123  val_loss=0.7834  val_f1=0.5612  time=11.9s
Epoch 30/100: train_loss=0.6456  val_loss=0.7423  val_f1=0.5989  time=11.7s
Epoch 40/100: train_loss=0.5923  val_loss=0.7234  val_f1=0.6145  time=11.8s
Epoch 50/100: train_loss=0.5512  val_loss=0.7189  val_f1=0.6234  time=11.9s
Epoch 60/100: train_loss=0.5234  val_loss=0.7201  val_f1=0.6223  time=11.8s
Early stopping triggered (patience=15)
Best epoch: 45, val_f1=0.6245

Training complete: 15.2 minutes
```

## Cross-Validation

```bash
# 5-fold time-series CV
python scripts/run_cv.py \
    --models lstm \
    --horizons 20 \
    --n-splits 5 \
    --seq-len 60

# With hyperparameter tuning
python scripts/run_cv.py \
    --models lstm \
    --horizons 20 \
    --n-splits 5 \
    --seq-len 60 \
    --tune \
    --n-trials 50
```

## Sequence Model Ensembles

LSTM can be ensembled with other sequence models (NOT tabular models):

```bash
# Valid: LSTM + GRU + TCN
python scripts/train_model.py \
    --model voting \
    --base-models lstm,gru,tcn \
    --horizon 20 \
    --seq-len 60

# INVALID: LSTM + XGBoost (mixed 3D/2D inputs)
# This will raise EnsembleCompatibilityError
```

## Hidden State Extraction

LSTM provides interpretability via hidden state extraction:

```python
from src.models import ModelRegistry

model = ModelRegistry.create("lstm")
model.fit(X_train, y_train, X_val, y_val)

# Extract hidden states for analysis
hidden_states = model.get_hidden_states(X_test)
# Shape: (n_samples, seq_len, hidden_size * num_directions)
```

## Model Files

```
experiments/runs/{run_id}/
├── model.pt                # PyTorch model state dict
├── metadata.pkl            # Config + architecture
├── training_metrics.json   # Loss curves, F1 scores
└── predictions.csv         # Test set predictions
```

## Performance Benchmarks

### Expected Performance (MES, 5min bars)

| Metric | LSTM (hidden=256, layers=2) | Notes |
|--------|------------------------------|-------|
| Validation F1 | 0.60-0.65 | Macro F1 across 3 classes |
| Training Time | 15-30 min | RTX 3080, 100K samples |
| Inference | 2-5 ms/sample | GPU batch inference |
| Sharpe Ratio | 0.8-1.2 | Backtest dependent |

## References

- PyTorch LSTM Docs: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
- Mixed Precision: https://pytorch.org/docs/stable/amp.html
- RNN Tutorial: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
