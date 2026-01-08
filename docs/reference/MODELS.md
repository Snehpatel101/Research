# Model Reference

Comprehensive reference for all models in the ML Model Factory.

**Last Updated:** 2026-01-08

---

## Overview

The ML Model Factory includes **23 models** (22 if CatBoost unavailable) across 4 families:

| Family | Count | Models |
|--------|-------|--------|
| **Tabular** | 6 | XGBoost, LightGBM, CatBoost, Random Forest, Logistic, SVM |
| **Neural** | 10 | LSTM, GRU, TCN, Transformer, PatchTST, iTransformer, TFT, N-BEATS, InceptionTime, ResNet1D |
| **Ensemble** | 3 | Voting, Stacking, Blending |
| **Meta-Learners** | 4 | Ridge Meta, MLP Meta, Calibrated Meta, XGBoost Meta |

---

## Model Families

### Tabular Models (6)

Models that accept 2D input `(n_samples, n_features)`. Do not require feature scaling or sequential input.

| Name | Registry Key | Family | Input Shape | Requires Scaling | GPU Support | Status |
|------|--------------|--------|-------------|------------------|-------------|--------|
| XGBoost | `xgboost` | boosting | 2D | No | Yes | Complete |
| LightGBM | `lightgbm` | boosting | 2D | No | Yes | Complete |
| CatBoost | `catboost` | boosting | 2D | No | Yes | Complete (optional) |
| Random Forest | `random_forest` | classical | 2D | No | No | Complete |
| Logistic Regression | `logistic` | classical | 2D | Yes | No | Complete |
| SVM | `svm` | classical | 2D | Yes | No | Complete |

**Note:** CatBoost has conditional registration - only registers if the `catboost` library is installed. If unavailable, model count is 22 instead of 23.

### Neural Models (10)

Models that accept 3D sequential input `(n_samples, seq_len, n_features)`. Require feature scaling and GPU for practical training times.

| Name | Registry Key | Family | Input Shape | Requires Scaling | GPU Support | Status |
|------|--------------|--------|-------------|------------------|-------------|--------|
| LSTM | `lstm` | neural | 3D | Yes | Yes (required) | Complete |
| GRU | `gru` | neural | 3D | Yes | Yes (required) | Complete |
| TCN | `tcn` | neural | 3D | Yes | Yes (required) | Complete |
| Transformer | `transformer` | neural | 3D | Yes | Yes (required) | Complete |
| PatchTST | `patchtst` | neural | 3D | Yes | Yes (required) | Complete |
| iTransformer | `itransformer` | neural | 3D | Yes | Yes (required) | Complete |
| TFT | `tft` | neural | 3D | Yes | Yes (required) | Complete |
| N-BEATS | `nbeats` | neural | 3D | Yes | Yes (required) | Complete |
| InceptionTime | `inceptiontime` | neural | 3D | Yes | Yes (required) | Complete |
| ResNet1D | `resnet1d` | neural | 3D | Yes | Yes (required) | Complete |

### Ensemble Models (3)

Models that combine predictions from multiple base models.

| Name | Registry Key | Family | Input Shape | Base Model Type | Status |
|------|--------------|--------|-------------|-----------------|--------|
| Voting | `voting` | ensemble | Same as base | Homogeneous only | Complete |
| Stacking | `stacking` | ensemble | Same as base | Homogeneous or Heterogeneous | Complete |
| Blending | `blending` | ensemble | Same as base | Homogeneous only | Complete |

**Homogeneous:** All base models must have same input shape (all tabular or all sequence).
**Heterogeneous:** Stacking supports mixed tabular + sequence base models via dual data loading.

### Meta-Learners (4)

Specialized models for combining out-of-fold (OOF) predictions from base models. Accept 2D input of stacked probabilities.

| Name | Registry Key | Family | Input Shape | Requires Scaling | GPU Support | Status |
|------|--------------|--------|-------------|------------------|-------------|--------|
| Ridge Meta | `ridge_meta` | ensemble | 2D (OOF) | No (internal) | No | Complete |
| MLP Meta | `mlp_meta` | ensemble | 2D (OOF) | No (internal) | Yes (optional) | Complete |
| Calibrated Meta | `calibrated_meta` | ensemble | 2D (OOF) | No | No | Complete |
| XGBoost Meta | `xgboost_meta` | ensemble | 2D (OOF) | No | Yes | Complete |

---

## Model Details

### XGBoost

| Property | Value |
|----------|-------|
| **Registry Key** | `xgboost` |
| **Family** | boosting |
| **Input Shape** | 2D `(n_samples, n_features)` |
| **Requires Scaling** | No |
| **Requires Sequences** | No |
| **GPU Support** | Yes (`tree_method: hist`, `device: cuda`) |
| **Implementation** | `src/models/boosting/xgboost_model.py` |

**Strengths:** Fast training, regularization (L1/L2), handles missing values, feature importance, early stopping.

**Default Configuration:**
```yaml
n_estimators: 500
max_depth: 6
learning_rate: 0.1
subsample: 0.8
colsample_bytree: 0.8
```

---

### LightGBM

| Property | Value |
|----------|-------|
| **Registry Key** | `lightgbm` |
| **Family** | boosting |
| **Input Shape** | 2D `(n_samples, n_features)` |
| **Requires Scaling** | No |
| **Requires Sequences** | No |
| **GPU Support** | Yes (`device_type: gpu`) |
| **Implementation** | `src/models/boosting/lightgbm_model.py` |

**Strengths:** Leaf-wise growth, faster than XGBoost on large datasets, lower memory usage, categorical feature support.

**Default Configuration:**
```yaml
num_leaves: 31
max_depth: -1
learning_rate: 0.05
n_estimators: 500
```

---

### CatBoost

| Property | Value |
|----------|-------|
| **Registry Key** | `catboost` |
| **Family** | boosting |
| **Input Shape** | 2D `(n_samples, n_features)` |
| **Requires Scaling** | No |
| **Requires Sequences** | No |
| **GPU Support** | Yes (`task_type: GPU`) |
| **Implementation** | `src/models/boosting/catboost_model.py` |

**Strengths:** Excellent categorical handling, ordered boosting reduces overfitting, symmetric trees.

**Note:** Optional dependency. Only registered if `catboost` library is installed.

**Default Configuration:**
```yaml
iterations: 500
depth: 6
learning_rate: 0.1
```

---

### Random Forest

| Property | Value |
|----------|-------|
| **Registry Key** | `random_forest` |
| **Family** | classical |
| **Input Shape** | 2D `(n_samples, n_features)` |
| **Requires Scaling** | No |
| **Requires Sequences** | No |
| **GPU Support** | No (CPU only) |
| **Implementation** | `src/models/classical/random_forest.py` |

**Strengths:** Robust to overfitting, interpretable feature importance, handles non-linear relationships, parallelizable.

**Default Configuration:**
```yaml
n_estimators: 300
max_depth: 8
min_samples_split: 5
min_samples_leaf: 2
n_jobs: -1
```

---

### Logistic Regression

| Property | Value |
|----------|-------|
| **Registry Key** | `logistic` |
| **Family** | classical |
| **Input Shape** | 2D `(n_samples, n_features)` |
| **Requires Scaling** | Yes |
| **Requires Sequences** | No |
| **GPU Support** | No |
| **Implementation** | `src/models/classical/logistic.py` |

**Strengths:** Fast training, interpretable coefficients, well-calibrated probabilities, regularization options.

**Default Configuration:**
```yaml
C: 1.0
solver: lbfgs
max_iter: 1000
multi_class: multinomial
```

---

### SVM

| Property | Value |
|----------|-------|
| **Registry Key** | `svm` |
| **Family** | classical |
| **Input Shape** | 2D `(n_samples, n_features)` |
| **Requires Scaling** | Yes |
| **Requires Sequences** | No |
| **GPU Support** | No |
| **Implementation** | `src/models/classical/svm.py` |

**Strengths:** Effective in high-dimensional spaces, kernel trick for non-linear boundaries, margin maximization.

**Warning:** O(n^2) memory complexity. Subsample to 10-20K samples for large datasets.

**Default Configuration:**
```yaml
C: 1.0
kernel: rbf
probability: true
```

---

### LSTM

| Property | Value |
|----------|-------|
| **Registry Key** | `lstm` |
| **Family** | neural |
| **Input Shape** | 3D `(n_samples, seq_len, n_features)` |
| **Requires Scaling** | Yes |
| **Requires Sequences** | Yes |
| **GPU Support** | Yes (required for practical training) |
| **Implementation** | `src/models/neural/lstm_model.py` |

**Strengths:** Long-term dependency modeling, gated memory cells, effective for temporal patterns.

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

---

### GRU

| Property | Value |
|----------|-------|
| **Registry Key** | `gru` |
| **Family** | neural |
| **Input Shape** | 3D `(n_samples, seq_len, n_features)` |
| **Requires Scaling** | Yes |
| **Requires Sequences** | Yes |
| **GPU Support** | Yes (required) |
| **Implementation** | `src/models/neural/gru_model.py` |

**Strengths:** ~25% fewer parameters than LSTM (3 gates vs 4), faster training, similar performance.

**Default Configuration:** Same as LSTM.

---

### TCN

| Property | Value |
|----------|-------|
| **Registry Key** | `tcn` |
| **Family** | neural |
| **Input Shape** | 3D `(n_samples, seq_len, n_features)` |
| **Requires Scaling** | Yes |
| **Requires Sequences** | Yes |
| **GPU Support** | Yes (required) |
| **Implementation** | `src/models/neural/tcn_model.py` |

**Strengths:** Parallelizable (unlike RNNs), dilated causal convolutions, flexible receptive field, stable gradients.

**Default Configuration:**
```yaml
num_channels: [64, 64, 64, 64]
kernel_size: 3
dropout: 0.2
sequence_length: 120
```

---

### Transformer

| Property | Value |
|----------|-------|
| **Registry Key** | `transformer` |
| **Family** | neural |
| **Input Shape** | 3D `(n_samples, seq_len, n_features)` |
| **Requires Scaling** | Yes |
| **Requires Sequences** | Yes |
| **GPU Support** | Yes (high memory required) |
| **Implementation** | `src/models/neural/transformer_model.py` |

**Strengths:** Self-attention captures global dependencies, parallelizable, interpretable attention weights.

**Memory Note:** O(seq^2) attention complexity. Use smaller batch sizes for longer sequences.

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

### PatchTST

| Property | Value |
|----------|-------|
| **Registry Key** | `patchtst` |
| **Family** | neural |
| **Input Shape** | 3D `(n_samples, seq_len, n_features)` |
| **Requires Scaling** | Yes |
| **Requires Sequences** | Yes |
| **GPU Support** | Yes (required) |
| **Implementation** | `src/models/neural/patchtst_model.py` |

**Strengths:** Patch-based attention O((L/P)^2), channel-independence, effective for long sequences, SOTA forecasting.

**Reference:** Nie et al., "A Time Series is Worth 64 Words" (ICLR 2023)

**Default Configuration:**
```yaml
patch_len: 16
stride: 8
d_model: 256
nhead: 8
num_layers: 4
```

---

### iTransformer

| Property | Value |
|----------|-------|
| **Registry Key** | `itransformer` |
| **Family** | neural |
| **Input Shape** | 3D `(n_samples, seq_len, n_features)` |
| **Requires Scaling** | Yes |
| **Requires Sequences** | Yes |
| **GPU Support** | Yes (required) |
| **Implementation** | `src/models/neural/itransformer_model.py` |

**Strengths:** Attention over features (not time), O(F^2) complexity, efficient for high-dimensional features, multivariate focus.

**Reference:** Liu et al., "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting" (ICLR 2024)

**Default Configuration:**
```yaml
d_model: 256
nhead: 8
num_layers: 3
```

---

### TFT (Temporal Fusion Transformer)

| Property | Value |
|----------|-------|
| **Registry Key** | `tft` |
| **Family** | neural |
| **Input Shape** | 3D `(n_samples, seq_len, n_features)` |
| **Requires Scaling** | Yes |
| **Requires Sequences** | Yes |
| **GPU Support** | Yes (required) |
| **Implementation** | `src/models/neural/tft_model.py` |

**Strengths:** Interpretable attention, variable selection networks, gated residual connections, multi-horizon forecasting.

**Reference:** Lim et al., "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting" (2021)

**Default Configuration:**
```yaml
lstm_layers: 2
attention_layers: 1
hidden_size: 256
```

---

### N-BEATS

| Property | Value |
|----------|-------|
| **Registry Key** | `nbeats` |
| **Family** | neural |
| **Input Shape** | 3D `(n_samples, seq_len, n_features)` |
| **Requires Scaling** | Yes |
| **Requires Sequences** | Yes |
| **GPU Support** | Yes (required) |
| **Implementation** | `src/models/neural/nbeats.py` |

**Strengths:** Interpretable decomposition (trend + seasonality), stacked architecture, backward/forward residuals, M4 winner.

**Reference:** Oreshkin et al., "N-BEATS: Neural Basis Expansion Analysis" (ICLR 2020)

**Default Configuration:**
```yaml
n_stacks: 3
n_blocks: 3
hidden_size: 256
theta_dim: 8
```

---

### InceptionTime

| Property | Value |
|----------|-------|
| **Registry Key** | `inceptiontime` |
| **Family** | neural |
| **Input Shape** | 3D `(n_samples, seq_len, n_features)` |
| **Requires Scaling** | Yes |
| **Requires Sequences** | Yes |
| **GPU Support** | Yes (required) |
| **Implementation** | `src/models/neural/cnn.py` |

**Strengths:** Multi-scale convolutions (kernel sizes 10, 20, 40), ensemble of 5 Inception networks, bottleneck layers.

**Reference:** Fawaz et al., "InceptionTime: Finding AlexNet for Time Series Classification" (2020)

**Default Configuration:**
```yaml
n_blocks: 6
n_filters: 32
kernel_sizes: [10, 20, 40]
bottleneck_channels: 32
```

---

### ResNet1D

| Property | Value |
|----------|-------|
| **Registry Key** | `resnet1d` |
| **Family** | neural |
| **Input Shape** | 3D `(n_samples, seq_len, n_features)` |
| **Requires Scaling** | Yes |
| **Requires Sequences** | Yes |
| **GPU Support** | Yes (required) |
| **Implementation** | `src/models/neural/cnn.py` |

**Strengths:** Residual connections prevent vanishing gradients, deep architectures (4-8 blocks), batch normalization.

**Reference:** Wang et al., "Time Series Classification from Scratch with Deep Neural Networks" (2017)

**Default Configuration:**
```yaml
channels: [64, 128, 256, 512]
kernel_size: 3
```

---

### Voting Ensemble

| Property | Value |
|----------|-------|
| **Registry Key** | `voting` |
| **Family** | ensemble |
| **Input Shape** | Same as base models |
| **Requires Scaling** | Inherited from base models |
| **Requires Sequences** | Inherited from base models |
| **Implementation** | `src/models/ensemble/voting.py` |

**Strengths:** Simple averaging of predictions, soft or hard voting, no training required for meta-learner.

**Constraint:** Homogeneous only (all base models must have same input shape).

**Default Configuration:**
```yaml
voting: soft
base_model_names: [xgboost, lightgbm, catboost]
```

---

### Stacking Ensemble

| Property | Value |
|----------|-------|
| **Registry Key** | `stacking` |
| **Family** | ensemble |
| **Input Shape** | Same as base models |
| **Requires Scaling** | Inherited from base models |
| **Requires Sequences** | Inherited from base models |
| **Implementation** | `src/models/ensemble/stacking.py` |

**Strengths:** OOF predictions avoid leakage, meta-learner learns optimal combination, supports heterogeneous bases.

**Heterogeneous Support:** Mix tabular + sequence models by providing both 2D and 3D data.

**Default Configuration:**
```yaml
base_model_names: [xgboost, lightgbm, catboost]
meta_learner_name: logistic
n_folds: 5
```

---

### Blending Ensemble

| Property | Value |
|----------|-------|
| **Registry Key** | `blending` |
| **Family** | ensemble |
| **Input Shape** | Same as base models |
| **Requires Scaling** | Inherited from base models |
| **Requires Sequences** | Inherited from base models |
| **Implementation** | `src/models/ensemble/blending.py` |

**Strengths:** Simpler than stacking (single holdout), faster training, good for quick prototyping.

**Constraint:** Homogeneous only (all base models must have same input shape).

**Default Configuration:**
```yaml
base_model_names: [xgboost, random_forest]
meta_learner_name: logistic
holdout_fraction: 0.2
```

---

### Ridge Meta-Learner

| Property | Value |
|----------|-------|
| **Registry Key** | `ridge_meta` |
| **Family** | ensemble |
| **Input Shape** | 2D `(n_samples, n_base_models * n_classes)` |
| **Requires Scaling** | No (internal scaling) |
| **Requires Sequences** | No |
| **GPU Support** | No |
| **Implementation** | `src/models/ensemble/meta_learners/ridge_meta.py` |

**Strengths:** Fast closed-form solution, L2 regularization, robust to multicollinearity, interpretable weights.

**Aliases:** `ridge_meta_learner`, `ridge_stacking`

**Default Configuration:**
```yaml
alpha: 1.0
fit_intercept: true
class_weight: balanced
```

---

### MLP Meta-Learner

| Property | Value |
|----------|-------|
| **Registry Key** | `mlp_meta` |
| **Family** | ensemble |
| **Input Shape** | 2D `(n_samples, n_base_models * n_classes)` |
| **Requires Scaling** | No (internal scaling) |
| **Requires Sequences** | No |
| **GPU Support** | Yes (optional) |
| **Implementation** | `src/models/ensemble/meta_learners/mlp_meta.py` |

**Strengths:** Non-linear combination of predictions, dropout regularization, learns complex interactions.

**Aliases:** `mlp_meta_learner`

**Default Configuration:**
```yaml
hidden_layers: [32, 16]
dropout: 0.3
learning_rate: 0.001
max_epochs: 100
batch_size: 512
```

---

### Calibrated Meta-Learner

| Property | Value |
|----------|-------|
| **Registry Key** | `calibrated_meta` |
| **Family** | ensemble |
| **Input Shape** | 2D `(n_samples, n_base_models * n_classes)` |
| **Requires Scaling** | No |
| **Requires Sequences** | No |
| **GPU Support** | No |
| **Implementation** | `src/models/ensemble/meta_learners/calibrated_meta.py` |

**Strengths:** Probability calibration (isotonic or Platt), well-calibrated final probabilities.

**Aliases:** `calibrated_meta_learner`, `calibrated_blender`

**Default Configuration:**
```yaml
calibration_method: isotonic
voting: soft
```

---

### XGBoost Meta-Learner

| Property | Value |
|----------|-------|
| **Registry Key** | `xgboost_meta` |
| **Family** | ensemble |
| **Input Shape** | 2D `(n_samples, n_base_models * n_classes)` |
| **Requires Scaling** | No |
| **Requires Sequences** | No |
| **GPU Support** | Yes |
| **Implementation** | `src/models/ensemble/meta_learners/xgboost_meta.py` |

**Strengths:** Non-linear feature interactions, gradient boosting on OOF predictions, handles correlations.

**Aliases:** `xgb_meta`, `xgboost_stacking`

**Default Configuration:**
```yaml
n_estimators: 100
max_depth: 4
learning_rate: 0.1
```

---

## Usage Examples

### Training Individual Models

```bash
# Train boosting models (2D input)
python scripts/train_model.py --model xgboost --horizon 20
python scripts/train_model.py --model lightgbm --horizon 20
python scripts/train_model.py --model catboost --horizon 20

# Train classical models (2D input)
python scripts/train_model.py --model random_forest --horizon 20
python scripts/train_model.py --model logistic --horizon 20

# Train sequence models (3D input)
python scripts/train_model.py --model lstm --horizon 20 --seq-len 60
python scripts/train_model.py --model tcn --horizon 20 --seq-len 120
python scripts/train_model.py --model transformer --horizon 20 --seq-len 60

# Train advanced models
python scripts/train_model.py --model patchtst --horizon 20 --seq-len 64
python scripts/train_model.py --model nbeats --horizon 20 --seq-len 60
python scripts/train_model.py --model inceptiontime --horizon 20 --seq-len 60
```

### Training Ensembles

```bash
# Homogeneous voting (same-family base models)
python scripts/train_model.py --model voting --horizon 20 \
  --base-models xgboost,lightgbm,catboost

# Homogeneous stacking (same-family base models)
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models lstm,gru,tcn --meta-learner ridge_meta --seq-len 60

# Heterogeneous stacking (mixed tabular + sequence)
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models xgboost,lstm,patchtst --meta-learner xgboost_meta
```

### Programmatic Usage

```python
from src.models import ModelRegistry

# List all models
all_models = ModelRegistry.list_all()
print(f"Total models: {len(all_models)}")  # 23 (or 22 without CatBoost)

# List by family
by_family = ModelRegistry.list_models()
print(by_family)
# {'boosting': ['xgboost', 'lightgbm', 'catboost'],
#  'neural': ['lstm', 'gru', 'tcn', 'transformer', 'patchtst', ...],
#  'classical': ['random_forest', 'logistic', 'svm'],
#  'ensemble': ['voting', 'stacking', 'blending', 'ridge_meta', ...]}

# Create model instance
model = ModelRegistry.create("xgboost", config={"max_depth": 8})
print(model.requires_scaling)    # False
print(model.requires_sequences)  # False

# Train
metrics = model.fit(X_train, y_train, X_val, y_val)
print(f"Val F1: {metrics.val_f1:.3f}")

# Predict
output = model.predict(X_test)
print(output.class_predictions.shape)
print(output.class_probabilities.shape)
```

---

## Configuration Reference

### Config File Locations

```
config/models/
  xgboost.yaml
  lightgbm.yaml
  catboost.yaml
  lstm.yaml
  gru.yaml
  tcn.yaml
  transformer.yaml
  patchtst.yaml
  itransformer.yaml
  tft.yaml
  nbeats.yaml
  inceptiontime.yaml
  resnet1d.yaml
  random_forest.yaml
  logistic.yaml
  svm.yaml
  voting.yaml
  stacking.yaml
  blending.yaml
  ridge_meta.yaml
  mlp_meta.yaml
  calibrated_meta.yaml
  xgboost_meta.yaml
```

### Config Structure

```yaml
model:
  name: xgboost
  family: boosting
  description: Gradient boosted trees

defaults:
  n_estimators: 500
  max_depth: 6
  learning_rate: 0.1

training:
  early_stopping_rounds: 50

device:
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

## Hardware Requirements

### By Model Family

| Family | Min GPU | Recommended GPU | CPU Fallback |
|--------|---------|-----------------|--------------|
| **Boosting** | None | RTX 3070 (8GB) | Yes (10-20 min) |
| **Neural** | GTX 1080 Ti | RTX 3080 (10GB) | No (impractical) |
| **Classical** | None | None | Yes (native) |
| **Ensemble** | Inherited | Inherited | Depends on bases |
| **Meta-Learners** | None | None | Yes (most are CPU) |

### GPU Memory by Model

| Model | Batch 256 | Batch 512 | Batch 1024 |
|-------|-----------|-----------|------------|
| LSTM | 3 GB | 5 GB | 8 GB |
| GRU | 2.5 GB | 4 GB | 7 GB |
| TCN | 4 GB | 6 GB | 10 GB |
| Transformer | 6 GB | 10 GB | 16+ GB |
| PatchTST | 5 GB | 8 GB | 14 GB |
| InceptionTime | 4 GB | 7 GB | 12 GB |

---

## Model Registration

### Plugin Architecture

Models register automatically via the `@register` decorator:

```python
from src.models import register, BaseModel

@register(name="my_model", family="boosting", description="Custom boosting model")
class MyModel(BaseModel):
    @property
    def model_family(self) -> str:
        return "boosting"

    @property
    def requires_scaling(self) -> bool:
        return False

    @property
    def requires_sequences(self) -> bool:
        return False

    def get_default_config(self) -> dict:
        return {"param": 1.0}

    def fit(self, X_train, y_train, X_val, y_val, sample_weights=None, config=None):
        # Training logic
        return TrainingMetrics(...)

    def predict(self, X):
        # Prediction logic
        return PredictionOutput(...)

    def save(self, path):
        # Persistence logic
        pass

    def load(self, path):
        # Loading logic
        pass
```

### BaseModel Interface

```python
class BaseModel(ABC):
    @property
    @abstractmethod
    def model_family(self) -> str:
        """Return: 'boosting', 'neural', 'classical', 'ensemble'"""
        pass

    @property
    @abstractmethod
    def requires_scaling(self) -> bool:
        """Whether features should be scaled before training."""
        pass

    @property
    @abstractmethod
    def requires_sequences(self) -> bool:
        """Whether input should be 3D (n_samples, seq_len, n_features)."""
        pass

    @abstractmethod
    def get_default_config(self) -> dict:
        """Return default hyperparameters."""
        pass

    @abstractmethod
    def fit(self, X_train, y_train, X_val, y_val, ...) -> TrainingMetrics:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X) -> PredictionOutput:
        """Generate predictions with probabilities and confidence."""
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist trained model."""
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load trained model."""
        pass
```

---

## References

- **Architecture:** `docs/ARCHITECTURE.md`
- **Training Guide:** `docs/implementation/PHASE_6_TRAINING.md`
- **Meta-Learner Guide:** `docs/guides/META_LEARNER_STACKING.md`
- **Model Integration:** `docs/guides/MODEL_INTEGRATION.md`
- **Infrastructure Requirements:** `docs/reference/INFRASTRUCTURE.md`
