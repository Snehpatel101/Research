# Models Reference

Comprehensive reference for all models in the ML Model Factory.

**Last Updated:** 2026-01-01

---

## Table of Contents

1. [Overview](#overview)
2. [Implemented Models (13)](#implemented-models)
3. [Planned Models (6)](#planned-models)
4. [Model Registration](#model-registration)
5. [Data Adapter Requirements](#data-adapter-requirements)
6. [Hardware Requirements](#hardware-requirements)
7. [Configuration Files](#configuration-files)

---

## Overview

The ML Model Factory supports **13 implemented models** across 4 families, with **6 additional models planned**.

### Quick Reference

| Family | Models | Input Shape | Status |
|--------|--------|-------------|--------|
| **Boosting** (3) | XGBoost, LightGBM, CatBoost | 2D `(N, F)` | Complete |
| **Neural** (4) | LSTM, GRU, TCN, Transformer | 3D `(N, T, F)` | Complete |
| **Classical** (3) | Random Forest, Logistic, SVM | 2D `(N, F)` | Complete |
| **Ensemble** (3) | Voting, Stacking, Blending | Same as base | ❌ Removed (replaced by meta-learner stacking) |
| **Inference** (4) | Logistic Meta, Ridge Meta, MLP Meta, Calibrated Blender | 2D `(N, n_bases*3)` | Planned |
| **CNN** (2) | InceptionTime, 1D ResNet | 3D `(N, T, F)` | Planned |
| **Advanced** (3) | PatchTST, iTransformer, TFT | 4D `(N, TF, T, 4)` | Planned |
| **MLP** (1) | N-BEATS | 3D `(N, T, F)` | Planned |

**Total:** 10 implemented (13 - 3 removed ensembles) + 10 planned = 20 models

**Note:** Homogeneous ensembles (Voting, Stacking, Blending) removed in favor of heterogeneous meta-learner stacking (Phase 7).

---

## Implemented Models

### Boosting Family (3 Models)

**Input:** 2D arrays `(n_samples, n_features)` - ~180 indicator features

#### XGBoost

| Property | Value |
|----------|-------|
| **Registry Name** | `xgboost` |
| **Implementation** | `src/models/boosting/xgboost_model.py` |
| **Config** | `config/models/xgboost.yaml` |
| **GPU Support** | Yes (`tree_method: hist`) |
| **Training Time** | 2-5 min (GPU), 10-20 min (CPU) |
| **Memory** | 2-4 GB VRAM (fixed) |

**Default Configuration:**
```yaml
n_estimators: 500
max_depth: 6
learning_rate: 0.1
subsample: 0.8
colsample_bytree: 0.8
```

#### LightGBM

| Property | Value |
|----------|-------|
| **Registry Name** | `lightgbm` |
| **Implementation** | `src/models/boosting/lightgbm_model.py` |
| **Config** | `config/models/lightgbm.yaml` |
| **GPU Support** | Yes (`device_type: gpu`) |
| **Training Time** | 1.5-4 min (GPU), 8-15 min (CPU) |
| **Memory** | 2-3 GB VRAM |

**Default Configuration:**
```yaml
num_leaves: 31
max_depth: -1
learning_rate: 0.05
n_estimators: 500
```

#### CatBoost

| Property | Value |
|----------|-------|
| **Registry Name** | `catboost` |
| **Implementation** | `src/models/boosting/catboost_model.py` |
| **Config** | `config/models/catboost.yaml` |
| **GPU Support** | Yes (`task_type: GPU`) |
| **Training Time** | 3-6 min (GPU), 12-25 min (CPU) |
| **Memory** | 2-4 GB VRAM |

**Default Configuration:**
```yaml
iterations: 500
depth: 6
learning_rate: 0.1
```

---

### Neural Family (4 Models)

**Input:** 3D arrays `(n_samples, seq_len, n_features)` - ~180 features per timestep

#### LSTM

| Property | Value |
|----------|-------|
| **Registry Name** | `lstm` |
| **Implementation** | `src/models/neural/lstm_model.py` |
| **Config** | `config/models/lstm.yaml` |
| **GPU Required** | Yes (CPU impractical) |
| **Training Time** | 15-30 min (RTX 3080), 5-10 min (RTX 4090) |
| **Memory Formula** | `4 * hidden * (features + hidden + 1) * layers` |

**Default Configuration:**
```yaml
hidden_size: 256
num_layers: 2
dropout: 0.3
sequence_length: 60
learning_rate: 0.001
batch_size: 512
max_epochs: 100
early_stopping_patience: 15
```

**Batch Size by GPU:**
| GPU | VRAM | Batch Size |
|-----|------|------------|
| GTX 1080 Ti | 11GB | 256 |
| RTX 3080 | 10GB | 512 |
| RTX 3090 | 24GB | 1024 |
| A100 | 40GB | 2048 |

#### GRU

| Property | Value |
|----------|-------|
| **Registry Name** | `gru` |
| **Implementation** | `src/models/neural/gru_model.py` |
| **Config** | `config/models/gru.yaml` |
| **GPU Required** | Yes |
| **Training Time** | 12-25 min (RTX 3080) |
| **Memory** | ~25% less than LSTM (3 gates vs 4) |

**Default Configuration:** Same as LSTM

#### TCN

| Property | Value |
|----------|-------|
| **Registry Name** | `tcn` |
| **Implementation** | `src/models/neural/tcn_model.py` |
| **Config** | `config/models/tcn.yaml` |
| **GPU Required** | Yes |
| **Training Time** | 20-35 min (RTX 3080) |
| **Memory Formula** | `channels * kernel * layers` |

**Default Configuration:**
```yaml
num_channels: [64, 64, 64, 64]
kernel_size: 3
dropout: 0.2
sequence_length: 120
```

#### Transformer

| Property | Value |
|----------|-------|
| **Registry Name** | `transformer` |
| **Implementation** | `src/models/neural/transformer_model.py` |
| **Config** | `config/models/transformer.yaml` |
| **GPU Required** | Yes (high memory) |
| **Training Time** | 30-60 min (RTX 3080) |
| **Memory Formula** | `seq^2 * d_model` (quadratic in sequence length) |

**Default Configuration:**
```yaml
d_model: 256
nhead: 8
num_layers: 4
dropout: 0.1
sequence_length: 60
batch_size: 128
```

---

### Classical Family (3 Models)

**Input:** 2D arrays `(n_samples, n_features)` - ~180 indicator features

#### Random Forest

| Property | Value |
|----------|-------|
| **Registry Name** | `random_forest` |
| **Implementation** | `src/models/classical/random_forest_model.py` |
| **Config** | `config/models/random_forest.yaml` |
| **GPU Support** | No (CPU only) |
| **Training Time** | 3-8 min |
| **Memory** | 1-4 GB RAM |

**Default Configuration:**
```yaml
n_estimators: 300
max_depth: 8
min_samples_split: 5
min_samples_leaf: 2
n_jobs: -1
```

#### Logistic Regression

| Property | Value |
|----------|-------|
| **Registry Name** | `logistic` |
| **Implementation** | `src/models/classical/logistic_model.py` |
| **Config** | `config/models/logistic.yaml` |
| **GPU Support** | No |
| **Training Time** | 1-2 min |
| **Memory** | <1 GB |

**Default Configuration:**
```yaml
C: 1.0
solver: lbfgs
max_iter: 1000
multi_class: multinomial
```

#### SVM

| Property | Value |
|----------|-------|
| **Registry Name** | `svm` |
| **Implementation** | `src/models/classical/svm_model.py` |
| **Config** | `config/models/svm.yaml` |
| **GPU Support** | No |
| **Training Time** | 5-15 min |
| **Memory** | O(n^2) - subsample for large datasets |

**Default Configuration:**
```yaml
C: 1.0
kernel: rbf
probability: true
```

**Warning:** SVM scales poorly with dataset size. Subsample to 10-20K samples for large datasets.

---

### Inference Family (4 Models - Meta-Learners)

**Input:** 2D arrays `(n_samples, n_bases * 3)` - OOF predictions from 3-4 heterogeneous base models

**Use Case:** Train on out-of-fold predictions from heterogeneous base models (e.g., CatBoost + TCN + PatchTST) to create final ensemble

#### Logistic Meta-Learner

| Property | Value |
|----------|-------|
| **Registry Name** | `meta_logistic` |
| **Implementation** | `src/models/inference/logistic_meta.py` |
| **Config** | `config/models/meta_logistic.yaml` |
| **Training Time** | 1-2 min |
| **Memory** | <1 GB |

**Default Configuration:**
```yaml
C: 1.0
solver: lbfgs
multi_class: multinomial
max_iter: 1000
```

#### Ridge Meta-Learner

| Property | Value |
|----------|-------|
| **Registry Name** | `meta_ridge` |
| **Implementation** | `src/models/inference/ridge_meta.py` |
| **Config** | `config/models/meta_ridge.yaml` |
| **Training Time** | 1-2 min |
| **Memory** | <1 GB |

**Default Configuration:**
```yaml
alpha: 1.0
solver: auto
max_iter: 1000
```

#### MLP Meta-Learner

| Property | Value |
|----------|-------|
| **Registry Name** | `meta_mlp` |
| **Implementation** | `src/models/inference/mlp_meta.py` |
| **Config** | `config/models/meta_mlp.yaml` |
| **GPU Support** | Yes (optional) |
| **Training Time** | 3-5 min |
| **Memory** | 1-2 GB |

**Default Configuration:**
```yaml
hidden_layers: [32, 16]
dropout: 0.3
learning_rate: 0.001
max_epochs: 100
batch_size: 512
```

#### Calibrated Blender

| Property | Value |
|----------|-------|
| **Registry Name** | `meta_calibrated` |
| **Implementation** | `src/models/inference/calibrated_blender.py` |
| **Config** | `config/models/meta_calibrated.yaml` |
| **Training Time** | 2-3 min |
| **Memory** | <1 GB |

**Default Configuration:**
```yaml
calibration_method: isotonic  # or 'sigmoid'
voting: soft
ensemble_size: 5
```

---

## Planned Models

### CNN Family (2 Models)

#### InceptionTime

| Property | Value |
|----------|-------|
| **Status** | Planned |
| **Input** | 3D `(N, T, F)` |
| **Strengths** | Multi-scale pattern detection |
| **Est. Training Time** | 30-60 min |
| **Implementation Effort** | 3 days |

**Architecture:**
- 5 parallel Inception networks
- 6 Inception modules per network
- Multi-scale convolutions (kernel sizes 10, 20, 40)

#### 1D ResNet

| Property | Value |
|----------|-------|
| **Status** | Planned |
| **Input** | 3D `(N, T, F)` |
| **Strengths** | Deep residual learning |
| **Est. Training Time** | 20-40 min |
| **Implementation Effort** | 2 days |

**Architecture:**
- Residual blocks with skip connections
- Batch normalization
- 4-8 blocks depth

---

### Advanced Transformer Family (3 Models)

#### PatchTST

| Property | Value |
|----------|-------|
| **Status** | Planned |
| **Input** | 4D `(N, TF, T, 4)` multi-resolution |
| **Strengths** | Patch-based attention, O((L/P)^2) |
| **Est. Training Time** | 30-60 min |
| **Implementation Effort** | 4 days |

**Architecture:**
- Patching: Divide time series into patches
- Channel-independent processing
- Reduced attention complexity

#### iTransformer

| Property | Value |
|----------|-------|
| **Status** | Planned |
| **Input** | 4D `(N, TF, T, 4)` multi-resolution |
| **Strengths** | Attention over features (not time) |
| **Est. Training Time** | 25-50 min |
| **Implementation Effort** | 3 days |

**Architecture:**
- Inverted attention: O(F^2) instead of O(T^2)
- Efficient for high-dimensional features
- Better for multivariate forecasting

#### TFT (Temporal Fusion Transformer)

| Property | Value |
|----------|-------|
| **Status** | Planned |
| **Input** | 4D `(N, TF, T, 4)` multi-resolution |
| **Strengths** | Interpretable, variable selection |
| **Est. Training Time** | 45-90 min |
| **Implementation Effort** | 5 days |

**Architecture:**
- Variable Selection Networks
- LSTM encoder + multi-head attention
- Gated residual connections
- Quantile forecasting

---

### MLP Family (1 Model)

#### N-BEATS

| Property | Value |
|----------|-------|
| **Status** | Planned |
| **Input** | 3D `(N, T, F)` |
| **Strengths** | Interpretable decomposition |
| **Est. Training Time** | 15-30 min |
| **Implementation Effort** | 1 day |

**Architecture:**
- Stacks of fully connected layers
- Trend and seasonality blocks
- Backward and forward residual links
- M4 competition winner

---

## Model Registration

### Plugin Architecture

Models register automatically via the `@register` decorator:

```python
from src.models import register, BaseModel

@register(name="my_model", family="boosting")
class MyModel(BaseModel):
    def fit(self, X_train, y_train, X_val, y_val, sample_weights=None, config=None):
        # Training logic
        return TrainingMetrics(...)

    def predict(self, X):
        # Prediction logic
        return PredictionOutput(...)

    def save(self, path):
        # Persistence logic
        pass

    @classmethod
    def load(cls, path):
        # Loading logic
        pass
```

### BaseModel Interface

```python
class BaseModel(ABC):
    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> TrainingMetrics:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> PredictionOutput:
        """Generate predictions with probabilities and confidence."""
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist trained model."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseModel":
        """Load trained model."""
        pass
```

### Listing Models

```python
from src.models import ModelRegistry

# List all registered models
all_models = ModelRegistry.list_all()
print(all_models)
# ['xgboost', 'lightgbm', 'catboost', 'lstm', 'gru', 'tcn', 'transformer',
#  'random_forest', 'logistic', 'svm', 'voting', 'stacking', 'blending']

# Get model by name
model = ModelRegistry.create("xgboost", config={...})
```

---

## Data Adapter Requirements

### Tabular Adapter (2D)

**Used by:** Boosting (3) + Classical (3) = 6 models

**Input Shape:** `(n_samples, n_features)` - ~180 features

**Adapter Location:** `src/models/data_preparation.py`

```python
# Tabular models receive flat 2D arrays
X_train.shape  # (50000, 180)
X_val.shape    # (10000, 180)
X_test.shape   # (10000, 180)
```

### Sequence Adapter (3D)

**Used by:** Neural (4) + CNN (2, planned) + MLP (1, planned) = 7 models

**Input Shape:** `(n_samples, seq_len, n_features)` - ~180 features per timestep

**Adapter Location:** `src/models/data_preparation.py`

```python
# Sequence models receive windowed 3D arrays
X_train.shape  # (50000, 60, 180)  # seq_len=60
X_val.shape    # (10000, 60, 180)
X_test.shape   # (10000, 60, 180)
```

### Inference Adapter (2D OOF Predictions)

**Used by:** Inference (4 planned models)

**Input Shape:** `(n_samples, n_bases * 3)` - Base model OOF predictions

**Adapter Location:** `src/cross_validation/oof_heterogeneous.py`

```python
# Meta-learners receive OOF predictions from heterogeneous base models
# Example: 3 base models (CatBoost, TCN, PatchTST) → 9 features (3 bases * 3 class probs)
X_meta_train.shape  # (50000, 9)
X_meta_val.shape    # (10000, 9)
X_meta_test.shape   # (10000, 9)
```

### Multi-Resolution Adapter (4D) - Planned

**Used by:** PatchTST, iTransformer, TFT (3 planned models)

**Input Shape:** `(n_samples, n_timeframes, seq_len, 4)` - Raw OHLC from 9 timeframes

**Status:** Not implemented (requires Phase 2 MTF extension)

```python
# Advanced models receive multi-resolution tensors
X_train.shape  # (50000, 9, 60, 4)  # 9 timeframes, 60 bars each, OHLC
```

---

## Hardware Requirements

### By Model Family

| Family | Min GPU | Recommended GPU | CPU Fallback |
|--------|---------|-----------------|--------------|
| **Boosting** | None | RTX 3070 (8GB) | Yes (10-20 min) |
| **Neural** | GTX 1080 Ti | RTX 3080 (10GB) | No (hours) |
| **Classical** | None | None | Yes (native) |
| **Ensemble** | Inherited | Inherited | Depends |
| **CNN** (planned) | RTX 3060 | RTX 3080 | No |
| **Advanced** (planned) | RTX 3070 | RTX 3090 (24GB) | No |
| **MLP** (planned) | RTX 2060 | RTX 3070 | Possible |

### Budget Recommendations

| Budget | GPU | Models Supported |
|--------|-----|------------------|
| $0 | CPU only | Boosting (CPU), Classical |
| $800 | RTX 4070 Ti (12GB) | All 13 implemented |
| $1,600 | RTX 4090 (24GB) | All 19 (when released) |

### Cloud Options

| GPU | Instance | Cost/Hour | 100K Training |
|-----|----------|-----------|---------------|
| Tesla V100 | p3.2xlarge | $3.06 | $0.76 (15 min) |
| A100 (40GB) | p4d.24xlarge | $32.77 | $1.64 (3 min) |

---

## Configuration Files

### Location

```
config/models/
  xgboost.yaml
  lightgbm.yaml
  catboost.yaml
  lstm.yaml
  gru.yaml
  tcn.yaml
  transformer.yaml
  random_forest.yaml
  logistic.yaml
  svm.yaml
  voting.yaml
  stacking.yaml
  blending.yaml
```

### Config Structure

```yaml
model:
  name: xgboost
  family: boosting
  description: Gradient boosted trees

defaults:
  # Model-specific hyperparameters
  n_estimators: 500
  max_depth: 6
  learning_rate: 0.1

training:
  # Training settings
  early_stopping_rounds: 50

device:
  # Hardware settings
  default: auto
  mixed_precision: false
```

### Loading Configs

```python
from src.models.config import load_model_config

config = load_model_config("xgboost")
# Returns dict with defaults and training settings
```

---

## References

- **Architecture:** `docs/ARCHITECTURE.md`
- **Training Guide:** `docs/implementation/PHASE_6_TRAINING.md`
- **Ensemble Guide:** `docs/guides/ENSEMBLE_CONFIGURATION.md`
- **Model Integration:** `docs/guides/MODEL_INTEGRATION.md`
- **Hardware Requirements:** `docs/models/REQUIREMENTS_MATRIX.md`
- **Training Guides:** `docs/models/*_TRAINING_GUIDE.md`
