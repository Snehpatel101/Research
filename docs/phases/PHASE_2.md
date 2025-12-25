# Phase 2: Model Factory

## Current Status: IMPLEMENTED

**IMPLEMENTATION STATUS:**
- [x] BaseModel interface - `src/models/base.py`
- [x] ModelRegistry plugin system - `src/models/registry.py` (12 models registered)
- [x] Trainer orchestration - `src/models/trainer.py`
- [x] Boosting models (XGBoost, LightGBM, CatBoost) - `src/models/boosting/`
- [x] Neural models (LSTM, GRU, TCN) - `src/models/neural/`
- [x] Classical models (Random Forest, Logistic, SVM) - `src/models/classical/`
- [x] Ensemble models (Voting, Stacking, Blending) - `src/models/ensemble/`
- [x] Model configuration - `src/models/config.py`
- [x] Device management - `src/models/device.py`
- [x] Notebook utilities - `src/utils/notebook.py`

**DEPENDENCIES:**
- [x] Phase 1 (Data Pipeline) - **COMPLETE** - Datasets ready in `data/splits/datasets/`

**TESTS:**
- 288+ tests in `tests/models/` - All passing (1592 total across all phases)
  - `test_boosting_models.py` - 58 tests
  - `test_neural_models.py` - 60 tests
  - `test_classical_models.py` - 47 tests
  - `test_ensemble_models.py` - 52 tests
  - `test_trainer.py` - 39 tests
  - `test_registry.py` - 32 tests

Phase 2 builds the model training infrastructure. The goal is a plugin-based factory where adding a new model requires only implementing an interface and a config file. This document provides comprehensive specifications for the model registry, training infrastructure, and model-family requirements.

---

## Overview

The Model Factory trains any model family through a unified interface:

```
Phase 1 Datasets  -->  ModelRegistry  -->  Trainer  -->  Trained Models + Metrics
                            |
              [12 Models: Boosting, Neural, Classical, Ensemble]
              XGBoost, LightGBM, CatBoost
              LSTM, GRU, TCN
              Random Forest, Logistic, SVM
              Voting, Stacking, Blending
```

**Key Principles:**
1. **Plugin architecture** - Add models without changing core infrastructure
2. **Model-family awareness** - Different preprocessing per model type
3. **Unified evaluation** - All models produce comparable outputs
4. **Reproducibility** - Deterministic training with seed control

---

## Core Components

### 1. BaseModel Interface

All models must implement this abstract interface:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any
import numpy as np
import pandas as pd


@dataclass
class PredictionOutput:
    """Standardized prediction output for all models."""
    class_predictions: np.ndarray      # Shape: (n_samples,) - predicted class
    class_probabilities: np.ndarray    # Shape: (n_samples, n_classes) - probabilities
    confidence: np.ndarray             # Shape: (n_samples,) - max probability
    metadata: Dict[str, Any]           # Model-specific metadata


@dataclass
class TrainingMetrics:
    """Standardized training metrics for all models."""
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    train_f1: float
    val_f1: float
    epochs_trained: int
    training_time_seconds: float
    early_stopped: bool
    best_epoch: Optional[int]
    history: Dict[str, list]           # Per-epoch metrics


class BaseModel(ABC):
    """Abstract base class for all models in the factory."""

    @property
    @abstractmethod
    def model_family(self) -> str:
        """Return model family: 'boosting', 'neural', 'transformer', 'classical'."""
        pass

    @property
    @abstractmethod
    def requires_scaling(self) -> bool:
        """Whether this model requires feature scaling."""
        pass

    @property
    @abstractmethod
    def requires_sequences(self) -> bool:
        """Whether this model requires sequential input (LSTM, Transformer)."""
        pass

    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """Return default hyperparameters for this model."""
        pass

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        config: Optional[Dict] = None
    ) -> TrainingMetrics:
        """
        Train the model.

        Args:
            X_train: Training features (n_samples, n_features) or (n_samples, seq_len, n_features)
            y_train: Training labels (n_samples,)
            X_val: Validation features
            y_val: Validation labels
            sample_weights: Optional quality-based sample weights
            config: Hyperparameters (uses defaults if None)

        Returns:
            TrainingMetrics with training results
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> PredictionOutput:
        """
        Generate predictions.

        Args:
            X: Features (n_samples, n_features) or (n_samples, seq_len, n_features)

        Returns:
            PredictionOutput with predictions and probabilities
        """
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model to disk."""
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load model from disk."""
        pass

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Return feature importances if available (tree-based models)."""
        return None
```

### 2. ModelRegistry (Plugin System)

The registry enables dynamic model discovery and instantiation:

```python
from typing import Type, Dict, Callable, Optional
from functools import wraps


class ModelRegistry:
    """Plugin registry for model types."""

    _models: Dict[str, Type[BaseModel]] = {}
    _families: Dict[str, list] = {}

    @classmethod
    def register(
        cls,
        name: str,
        family: str,
        description: str = ""
    ) -> Callable:
        """
        Decorator to register a model class.

        Usage:
            @ModelRegistry.register(name="xgboost", family="boosting")
            class XGBoostModel(BaseModel):
                ...
        """
        def decorator(model_class: Type[BaseModel]) -> Type[BaseModel]:
            if name in cls._models:
                raise ValueError(f"Model '{name}' already registered")

            cls._models[name] = model_class

            if family not in cls._families:
                cls._families[family] = []
            cls._families[family].append(name)

            return model_class
        return decorator

    @classmethod
    def create(
        cls,
        name: str,
        config: Optional[Dict] = None,
        **kwargs
    ) -> BaseModel:
        """Instantiate a registered model."""
        if name not in cls._models:
            available = list(cls._models.keys())
            raise ValueError(f"Unknown model '{name}'. Available: {available}")

        model_class = cls._models[name]
        return model_class(config=config, **kwargs)

    @classmethod
    def list_models(cls) -> Dict[str, list]:
        """List all registered models by family."""
        return cls._families.copy()

    @classmethod
    def get_model_info(cls, name: str) -> Dict:
        """Get model metadata."""
        model_class = cls._models[name]
        instance = model_class()
        return {
            "name": name,
            "family": instance.model_family,
            "requires_scaling": instance.requires_scaling,
            "requires_sequences": instance.requires_sequences,
            "default_config": instance.get_default_config()
        }


# Usage example
@ModelRegistry.register(name="xgboost", family="boosting", description="XGBoost classifier")
class XGBoostModel(BaseModel):
    ...

# Instantiate
model = ModelRegistry.create("xgboost", config={"max_depth": 6})
```

### 3. Trainer Orchestration

The trainer handles the complete training workflow:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict
import json
import time


@dataclass
class TrainerConfig:
    """Configuration for model training."""
    model_name: str
    horizon: int
    feature_set: str = "boosting_optimal"      # From Phase 1
    sequence_length: int = 60                   # For sequential models
    batch_size: int = 256
    max_epochs: int = 100
    early_stopping_patience: int = 10
    random_seed: int = 42
    experiment_name: Optional[str] = None
    output_dir: Path = Path("experiments/runs")


class Trainer:
    """Orchestrates model training and evaluation."""

    def __init__(self, config: TrainerConfig):
        self.config = config
        self.model = ModelRegistry.create(config.model_name)
        self.run_id = self._generate_run_id()
        self.output_path = config.output_dir / self.run_id

    def run_full_pipeline(
        self,
        container: "TimeSeriesDataContainer"
    ) -> Dict:
        """
        Execute complete training pipeline.

        1. Load and prepare data
        2. Apply model-specific preprocessing
        3. Train model
        4. Evaluate on validation set
        5. Save artifacts

        Returns:
            Dictionary with metrics and paths
        """
        self._setup_output_dir()

        # Step 1: Get data from container
        if self.model.requires_sequences:
            X_train, y_train, w_train = container.get_pytorch_sequences(
                "train",
                seq_len=self.config.sequence_length
            )
            X_val, y_val, _ = container.get_pytorch_sequences(
                "val",
                seq_len=self.config.sequence_length
            )
        else:
            X_train, y_train, w_train = container.get_sklearn_arrays("train")
            X_val, y_val, _ = container.get_sklearn_arrays("val")

        # Step 2: Apply scaling if needed
        if self.model.requires_scaling:
            X_train, X_val = self._apply_scaling(X_train, X_val)

        # Step 3: Train
        start_time = time.time()
        metrics = self.model.fit(
            X_train, y_train,
            X_val, y_val,
            sample_weights=w_train,
            config=self.config.__dict__
        )

        # Step 4: Evaluate
        val_predictions = self.model.predict(X_val)
        eval_metrics = self._compute_evaluation_metrics(
            y_val, val_predictions
        )

        # Step 5: Save
        self._save_artifacts(metrics, eval_metrics, val_predictions)

        return {
            "run_id": self.run_id,
            "training_metrics": metrics,
            "evaluation_metrics": eval_metrics,
            "output_path": str(self.output_path)
        }
```

---

## Model Families and Requirements

**IMPORTANT NOTE:** All configuration files referenced in this section (e.g., `config/models/*.yaml`) are **PLANNED SPECIFICATIONS** and do not currently exist in the repository. These are design documents to guide implementation, not existing files.

### Model Family Overview

| Family | Models | Scaling | Sequences | Feature Set | Key Strengths |
|--------|--------|---------|-----------|-------------|---------------|
| **Boosting** | XGBoost, LightGBM, CatBoost | No | No | `boosting_optimal` | Fast, interpretable, handles mixed features |
| **Neural-RNN** | LSTM, GRU | Yes (Robust) | Yes (30-60) | `neural_optimal` | Learns temporal patterns, nonlinear |
| **Neural-CNN** | TCN | Yes (Robust) | Yes (60-120) | `neural_optimal` | Parallelizable, longer memory than LSTM |
| **Neural-Interpretable** | N-BEATS | Yes (Standard) | Yes (64-128) | Raw OHLCV | Interpretable decomposition, no features needed |
| **Transformer** | PatchTST, TFT, iTransformer | Yes (Standard) | Yes (64-256) | `transformer_raw` | Long-range dependencies, attention mechanisms |
| **Linear** | DLinear, NLinear | Yes (Standard) | Yes (64-128) | Raw OHLCV | Simple but effective, fast baselines |
| **State-Space** | Mamba, S4 | Yes (Standard) | Yes (128-512) | `neural_optimal` | Linear complexity, handles very long sequences |
| **Foundation** | Chronos-Bolt, TimesFM | Yes (Standard) | Yes (512+) | Raw OHLCV | Zero-shot capable, pre-trained on massive data |
| **Classical** | RandomForest, SVM, Logistic | Yes (Standard) | No | `boosting_optimal` | Simple baselines, fast to train |

---

### Boosting Models (XGBoost, LightGBM, CatBoost)

**Characteristics:**
- Scale-invariant (tree-based splits are order-preserving)
- Handle missing values natively
- Built-in feature importance
- Fast training with GPU support
- Strong regularization options

**Preprocessing Requirements:**
```python
# No scaling needed
requires_scaling = False
requires_sequences = False

# Feature set: Use all useful features
feature_set = "boosting_optimal"  # 80-120 features after correlation pruning
```

**Recommended Configuration:**

**NOTE: This file does not currently exist. This is a planned configuration.**

```yaml
# PLANNED: config/models/xgboost.yaml (does not exist yet)
model_name: xgboost
model_family: boosting

# Tree parameters
n_estimators: 500
max_depth: 6
min_child_weight: 10
subsample: 0.8
colsample_bytree: 0.8

# Learning
learning_rate: 0.05
gamma: 0.1

# Regularization
reg_alpha: 0.1      # L1
reg_lambda: 1.0     # L2

# Training
early_stopping_rounds: 50
eval_metric: "mlogloss"
use_gpu: true

# Class imbalance
scale_pos_weight: null  # Auto-computed from class distribution
```

```yaml
# PLANNED: config/models/lightgbm.yaml (does not exist yet)
model_name: lightgbm
model_family: boosting

n_estimators: 500
max_depth: 6
num_leaves: 31
min_child_samples: 20
subsample: 0.8
colsample_bytree: 0.8

learning_rate: 0.05
reg_alpha: 0.1
reg_lambda: 1.0

early_stopping_rounds: 50
boosting_type: "gbdt"
```

```yaml
# config/models/catboost.yaml
model_name: catboost
model_family: boosting

iterations: 500
depth: 6
learning_rate: 0.05
l2_leaf_reg: 3.0

# CatBoost-specific
random_strength: 1.0
bagging_temperature: 1.0

# Categorical handling (if applicable)
cat_features: []  # List of categorical column indices

early_stopping_rounds: 50
use_best_model: true
```

**Feature Importance Analysis:**

```python
def analyze_boosting_importance(model: BaseModel, feature_names: list) -> pd.DataFrame:
    """Extract and analyze feature importance from boosting models."""
    importance = model.get_feature_importance()

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False)

    # Cumulative importance
    df["cumulative"] = df["importance"].cumsum() / df["importance"].sum()

    # Feature categories
    df["category"] = df["feature"].apply(categorize_feature)

    return df
```

**Expected Performance (Boosting):**

| Horizon | Validation F1 | Validation Sharpe | Notes |
|---------|---------------|-------------------|-------|
| H5 | 0.38 - 0.44 | 0.35 - 0.55 | Most noise, hardest |
| H10 | 0.40 - 0.46 | 0.40 - 0.65 | Good baseline |
| H15 | 0.42 - 0.48 | 0.45 - 0.75 | Sweet spot |
| H20 | 0.44 - 0.50 | 0.50 - 0.85 | More signal, easier |

---

### Neural Networks (LSTM, GRU)

**Characteristics:**
- Learn temporal dependencies from sequences
- Require normalized/scaled inputs
- Sensitive to hyperparameters
- Need early stopping to prevent overfitting
- Wavelets significantly improve accuracy (+18-36%)

**Preprocessing Requirements:**
```python
requires_scaling = True
requires_sequences = True

# Scaling: RobustScaler handles outliers in financial data
scaler = "robust"  # Uses median and IQR

# Feature set: Bounded, normalized features only
feature_set = "neural_optimal"  # 40-60 features

# Sequence configuration
sequence_length = 60    # Short-term patterns
stride = 1              # No overlap skip
```

**Recommended Configuration:**

```yaml
# config/models/lstm.yaml
model_name: lstm
model_family: neural

# Architecture
hidden_size: 128
num_layers: 2
dropout: 0.3
bidirectional: false

# Input
sequence_length: 60
input_features: null  # Auto-detected from data

# Training
batch_size: 256
max_epochs: 100
learning_rate: 0.001
weight_decay: 0.0001
gradient_clip: 1.0

# Early stopping
early_stopping_patience: 15
min_delta: 0.0001

# Optimizer
optimizer: "adamw"
scheduler: "cosine"
warmup_epochs: 5

# Device
device: "cuda"
mixed_precision: true
```

```yaml
# config/models/gru.yaml
model_name: gru
model_family: neural

hidden_size: 128
num_layers: 2
dropout: 0.3

sequence_length: 60
batch_size: 256
max_epochs: 100
learning_rate: 0.001

early_stopping_patience: 15
```

**Neural Network Architecture:**

```python
import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """LSTM model for time series classification."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = False
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        lstm_output_size = hidden_size * (2 if bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_output_size),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last hidden state
        if self.lstm.bidirectional:
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last_hidden = h_n[-1]

        logits = self.classifier(last_hidden)
        return logits
```

**Wavelet Feature Integration:**

Neural networks benefit significantly from wavelet features:

```python
# Phase 1 already provides wavelet features:
# - wavelet_close_approx (trend)
# - wavelet_close_d1, d2, d3 (detail levels)
# - wavelet_close_energy_* (energy distribution)
# - wavelet_close_trend_strength

# These are included in neural_optimal feature set
# Research shows 18-36% accuracy improvement with wavelets
```

**Expected Performance (Neural):**

| Horizon | Validation F1 | Validation Sharpe | Notes |
|---------|---------------|-------------------|-------|
| H5 | 0.36 - 0.42 | 0.30 - 0.50 | Needs more data |
| H10 | 0.38 - 0.45 | 0.35 - 0.60 | Reasonable |
| H15 | 0.40 - 0.48 | 0.40 - 0.70 | Better signal |
| H20 | 0.42 - 0.50 | 0.45 - 0.80 | Best for neural |

---

### Transformers (PatchTST, TimesNet, TimesFM)

**Characteristics:**
- Learn long-range dependencies via attention
- Can work with minimal feature engineering (raw OHLCV)
- Foundation models (TimesFM, Chronos) work zero-shot
- Patching improves efficiency (64-128 token chunks)
- Computationally expensive to train

**Preprocessing Requirements:**
```python
requires_scaling = True
requires_sequences = True

# Scaling: Standard for transformers
scaler = "standard"

# Feature set: Minimal - let transformer learn patterns
feature_set = "transformer_raw"  # 10-15 features (returns + temporal)

# Sequence configuration
sequence_length = 128   # Longer for transformers
patch_size = 16         # For PatchTST
```

**Recommended Configuration:**

```yaml
# config/models/patchtst.yaml
model_name: patchtst
model_family: transformer

# Architecture
d_model: 128
n_heads: 8
n_layers: 3
d_ff: 256
dropout: 0.2

# Patching
patch_size: 16
stride: 8

# Input
sequence_length: 128
input_features: null

# Training
batch_size: 128
max_epochs: 50
learning_rate: 0.0001
weight_decay: 0.01

# Scheduler
scheduler: "cosine"
warmup_ratio: 0.1

# Early stopping
early_stopping_patience: 10
```

**Foundation Model Configuration:**

```yaml
# config/models/timesfm.yaml
model_name: timesfm
model_family: transformer

# Foundation model settings
pretrained_path: "google/timesfm-1.0-200m"
fine_tune: true
freeze_backbone: false
freeze_epochs: 5  # Freeze backbone for first N epochs

# Fine-tuning
learning_rate: 0.00001  # Lower LR for fine-tuning
max_epochs: 20

# Input
context_length: 512
prediction_length: 20  # Match horizon
```

**Zero-Shot vs Fine-Tuned:**

| Approach | Use Case | Performance | Training Time |
|----------|----------|-------------|---------------|
| Zero-shot | Quick baseline, limited data | Lower | None |
| Fine-tuned | Production, sufficient data | Higher | Hours |
| Frozen backbone | Limited data, regularization | Medium | Minutes |

**Expected Performance (Transformer):**

| Horizon | Validation F1 | Validation Sharpe | Notes |
|---------|---------------|-------------------|-------|
| H5 | 0.35 - 0.42 | 0.30 - 0.50 | Limited signal |
| H10 | 0.38 - 0.46 | 0.40 - 0.65 | Good with fine-tuning |
| H15 | 0.40 - 0.50 | 0.45 - 0.75 | Strong |
| H20 | 0.42 - 0.52 | 0.50 - 0.85 | Best for transformers |

---

### Classical Models (RandomForest, SVM, Logistic)

**Characteristics:**
- Simple, interpretable baselines
- Fast to train
- Useful for comparison and sanity checks
- RandomForest provides feature importance
- Logistic regression works well as meta-learner

**Preprocessing Requirements:**
```python
requires_scaling = True  # SVM and Logistic need scaling
requires_sequences = False

scaler = "standard"
feature_set = "boosting_optimal"  # Same as boosting
```

**Recommended Configuration:**

```yaml
# config/models/random_forest.yaml
model_name: random_forest
model_family: classical

n_estimators: 200
max_depth: 10
min_samples_split: 20
min_samples_leaf: 10
max_features: "sqrt"

n_jobs: -1
random_state: 42
```

```yaml
# config/models/logistic.yaml
model_name: logistic
model_family: classical

# Regularization
penalty: "l2"
C: 1.0

# Solver
solver: "lbfgs"
max_iter: 500

# Multi-class
multi_class: "multinomial"
```

---

## Advanced Neural Architectures (2024-2025)

This section documents state-of-the-art models from recent research that have shown strong performance on time series forecasting and classification tasks.

---

### Temporal Convolutional Network (TCN)

**Architecture Overview:**

TCN uses dilated causal convolutions inspired by WaveNet to capture long-range temporal dependencies. Unlike LSTMs, TCNs are fully parallelizable and often achieve better performance with faster training.

```
Input Sequence (seq_len, features)
        |
    [Causal Conv 1D, dilation=1]
        |
    [Causal Conv 1D, dilation=2]
        |
    [Causal Conv 1D, dilation=4]
        |
        ...
    [Causal Conv 1D, dilation=2^n]
        |
    [Global Pooling]
        |
    [Classification Head]
```

**Key Characteristics:**
- **Dilated convolutions** exponentially increase receptive field (2^n growth)
- **Causal padding** ensures no future information leakage
- **Residual connections** enable training of deep networks
- **Parallelizable** - all timesteps processed simultaneously (unlike sequential RNNs)
- **Memory efficient** - O(k*d*n) parameters where k=kernel size, d=dilation, n=channels

**Why TCN Often Beats LSTM:**
1. **Parallel processing** - 5-10x faster training than LSTM
2. **Longer effective memory** - Receptive field = (kernel_size - 1) * sum(dilations) + 1
3. **Stable gradients** - No vanishing gradient problem common in RNNs
4. **Flexible receptive field** - Easily tuned via dilation pattern

**Preprocessing Requirements:**
```python
requires_scaling = True
requires_sequences = True

# Scaling: RobustScaler for financial data outliers
scaler = "robust"

# Feature set: Similar to LSTM
feature_set = "neural_optimal"  # 40-60 features

# Sequence configuration
sequence_length = 120   # Longer than LSTM due to efficient memory
stride = 1
```

**Hyperparameter Ranges:**

| Parameter | Range | Recommended | Notes |
|-----------|-------|-------------|-------|
| `num_channels` | [32, 64, 128] per layer | [64, 64, 64, 64] | 4 layers common |
| `kernel_size` | 2-7 | 3 | Larger = more computation |
| `dropout` | 0.1-0.4 | 0.2 | Applied between layers |
| `dilation_base` | 2 | 2 | Exponential growth base |
| `sequence_length` | 60-240 | 120 | Match receptive field to horizon |

**Recommended Configuration:**

```yaml
# config/models/tcn.yaml
model_name: tcn
model_family: neural

# Architecture
num_channels: [64, 64, 64, 64]  # 4 residual blocks
kernel_size: 3
dropout: 0.2
dilation_base: 2                # Dilations: 1, 2, 4, 8

# Input
sequence_length: 120
input_features: null            # Auto-detected

# Training
batch_size: 256
max_epochs: 100
learning_rate: 0.001
weight_decay: 0.0001

# Optimizer
optimizer: "adamw"
scheduler: "cosine"
warmup_epochs: 5

# Early stopping
early_stopping_patience: 15
min_delta: 0.0001

# Device
device: "cuda"
mixed_precision: true
```

**PyTorch Implementation Reference:**

```python
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class CausalConv1d(nn.Module):
    """Causal convolution with proper padding."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        ))

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.padding] if self.padding > 0 else out


class TemporalBlock(nn.Module):
    """Single TCN block with residual connection."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None

    def forward(self, x):
        out = self.dropout(self.relu(self.conv1(x)))
        out = self.dropout(self.relu(self.conv2(out)))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNClassifier(nn.Module):
    """TCN for time series classification."""

    def __init__(
        self,
        input_size: int,
        num_channels: list = [64, 64, 64, 64],
        kernel_size: int = 3,
        dropout: float = 0.2,
        num_classes: int = 3
    ):
        super().__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = input_size if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))

        self.tcn = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        out = self.tcn(x)
        # Global average pooling
        out = out.mean(dim=2)
        return self.classifier(out)
```

**Expected Performance (TCN):**

| Horizon | Validation F1 | Validation Sharpe | Training Time | Notes |
|---------|---------------|-------------------|---------------|-------|
| H5 | 0.37 - 0.44 | 0.32 - 0.55 | ~15 min | Short patterns |
| H10 | 0.39 - 0.47 | 0.38 - 0.65 | ~15 min | Good baseline |
| H15 | 0.41 - 0.49 | 0.42 - 0.72 | ~15 min | Strong performance |
| H20 | 0.43 - 0.51 | 0.48 - 0.82 | ~15 min | Best horizon |

---

### N-BEATS (Neural Basis Expansion Analysis)

**Architecture Overview:**

N-BEATS is an interpretable deep learning architecture that learns to decompose time series into trend and seasonality components without requiring feature engineering. It uses stacked fully-connected networks with basis expansion.

```
Input Series (backcast window)
        |
    [Stack 1: Trend Block]
        |-- Trend Forecast
        |-- Backcast (residual)
        |
    [Stack 2: Seasonality Block]
        |-- Seasonality Forecast
        |-- Backcast (residual)
        |
    [Stack 3: Generic Block] (optional)
        |-- Generic Forecast
        |
    [Sum of Forecasts]
        |
    Output Prediction
```

**Key Characteristics:**
- **Interpretable decomposition** - Separates trend from seasonality
- **Feature-free** - Works directly on raw price series
- **Double residual stacking** - Learns hierarchical patterns
- **Multi-horizon native** - Predicts multiple future steps simultaneously
- **Strong baseline** - Often matches or beats complex transformers

**Why N-BEATS Works for OHLCV:**
1. **Trend extraction** useful for momentum/mean-reversion signals
2. **Seasonality capture** handles intraday patterns (session effects)
3. **No feature engineering** reduces lookahead bias risk
4. **Interpretability** aids in understanding model decisions

**Preprocessing Requirements:**
```python
requires_scaling = True
requires_sequences = True

# Scaling: Standard works well for N-BEATS
scaler = "standard"

# Feature set: Raw OHLCV - N-BEATS learns its own features
feature_set = "nbeats_raw"  # Close price or returns

# Sequence configuration
sequence_length = 2 * prediction_length  # Standard ratio
prediction_length = 20                    # Match horizon
```

**Hyperparameter Ranges:**

| Parameter | Range | Recommended | Notes |
|-----------|-------|-------------|-------|
| `stack_types` | trend+seasonality or generic | ["trend", "seasonality"] | Interpretable config |
| `num_blocks` | 1-3 per stack | 2 | More blocks = more capacity |
| `num_layers` | 2-4 | 4 | FC layers per block |
| `layer_width` | 128-512 | 256 | Hidden dimension |
| `expansion_coefficient_dim` | 3-8 | 5 | Basis function complexity |
| `backcast_length` | 2x-5x horizon | 2x | Input window |
| `forecast_length` | horizon | 5-20 | Multi-step output |

**Recommended Configuration:**

```yaml
# config/models/nbeats.yaml
model_name: nbeats
model_family: neural_interpretable

# Architecture
stack_types: ["trend", "seasonality"]
num_blocks: [2, 2]                      # Blocks per stack
num_layers: 4                           # FC layers per block
layer_width: 256                        # Hidden dimension

# Trend-specific
trend_polynomial_degree: 3              # Polynomial basis
# Seasonality-specific
seasonality_harmonics: 5                # Fourier harmonics

# Input/Output
backcast_length: 40                     # 2x horizon
forecast_length: 20                     # Match H20

# Training
batch_size: 256
max_epochs: 100
learning_rate: 0.001
weight_decay: 0.0001

# Loss
loss: "mse"                             # For regression
# For classification, add a head and use cross-entropy

# Early stopping
early_stopping_patience: 15
```

**PyTorch Implementation Reference:**

```python
import torch
import torch.nn as nn


class NBEATSBlock(nn.Module):
    """Single N-BEATS block."""

    def __init__(
        self,
        input_size: int,
        theta_size: int,
        num_layers: int = 4,
        layer_width: int = 256,
        basis_function: nn.Module = None
    ):
        super().__init__()

        layers = [nn.Linear(input_size, layer_width), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(layer_width, layer_width), nn.ReLU()])

        self.fc = nn.Sequential(*layers)
        self.theta_layer = nn.Linear(layer_width, theta_size)
        self.basis_function = basis_function

    def forward(self, x):
        out = self.fc(x)
        theta = self.theta_layer(out)
        backcast, forecast = self.basis_function(theta)
        return backcast, forecast


class TrendBasis(nn.Module):
    """Polynomial trend basis."""

    def __init__(self, backcast_length, forecast_length, degree=3):
        super().__init__()
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length

        # Polynomial basis
        backcast_time = torch.arange(backcast_length) / backcast_length
        forecast_time = torch.arange(forecast_length) / forecast_length

        self.register_buffer('backcast_basis',
            torch.stack([backcast_time ** i for i in range(degree + 1)], dim=1))
        self.register_buffer('forecast_basis',
            torch.stack([forecast_time ** i for i in range(degree + 1)], dim=1))

    def forward(self, theta):
        backcast_theta = theta[:, :self.backcast_basis.shape[1]]
        forecast_theta = theta[:, self.backcast_basis.shape[1]:]
        backcast = torch.einsum('bi,ti->bt', backcast_theta, self.backcast_basis)
        forecast = torch.einsum('bi,ti->bt', forecast_theta, self.forecast_basis)
        return backcast, forecast


class NBEATS(nn.Module):
    """N-BEATS for time series forecasting."""

    def __init__(
        self,
        backcast_length: int,
        forecast_length: int,
        num_stacks: int = 2,
        num_blocks: int = 2,
        num_layers: int = 4,
        layer_width: int = 256
    ):
        super().__init__()
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length

        # Create stacks
        self.blocks = nn.ModuleList()
        # Add trend and seasonality blocks here
        # (Full implementation would include basis functions)

    def forward(self, x):
        forecast = torch.zeros(x.shape[0], self.forecast_length, device=x.device)
        backcast = x

        for block in self.blocks:
            block_backcast, block_forecast = block(backcast)
            backcast = backcast - block_backcast
            forecast = forecast + block_forecast

        return forecast
```

**Library Reference:** Use `neuralforecast` library for production implementation:
```python
from neuralforecast.models import NBEATS
from neuralforecast import NeuralForecast

model = NBEATS(
    h=20,                     # Forecast horizon
    input_size=40,            # Lookback window
    stack_types=["trend", "seasonality"],
    n_blocks=[2, 2],
    n_harmonics=5,
    n_polynomials=3,
    learning_rate=1e-3
)
```

**Expected Performance (N-BEATS):**

| Horizon | Validation F1 | Validation Sharpe | Training Time | Notes |
|---------|---------------|-------------------|---------------|-------|
| H5 | 0.35 - 0.42 | 0.28 - 0.48 | ~20 min | Short horizon challenging |
| H10 | 0.38 - 0.45 | 0.35 - 0.58 | ~20 min | Reasonable |
| H15 | 0.40 - 0.48 | 0.40 - 0.68 | ~20 min | Trend component helps |
| H20 | 0.42 - 0.50 | 0.45 - 0.78 | ~20 min | Best for trend capture |

---

### TFT (Temporal Fusion Transformer)

**Architecture Overview:**

TFT is an attention-based architecture specifically designed for multi-horizon forecasting with mixed inputs. It handles static covariates, known future inputs, and time-varying unknown inputs through specialized processing pathways.

```
                    Static Covariates
                          |
                    [Static Encoder]
                          |
    +---------+-----------+-----------+
    |         |                       |
Past Inputs   Known Future    Static Context
    |              |                  |
[LSTM Encoder] [LSTM Decoder]        |
    |              |                  |
    +------+-------+                  |
           |                          |
    [Variable Selection Networks]<----+
           |
    [Interpretable Multi-Head Attention]
           |
    [Gated Residual Networks]
           |
    [Quantile Outputs]
```

**Key Characteristics:**
- **Variable selection** - Learns which features matter at each timestep
- **Multi-horizon native** - Predicts full forecast horizon simultaneously
- **Static + temporal** - Handles both time-invariant and time-varying features
- **Interpretable attention** - Temporal attention weights show important timesteps
- **Quantile forecasts** - Produces prediction intervals, not just point estimates

**Why TFT Works for OHLCV:**
1. **Variable selection** identifies important features across regimes
2. **Known future inputs** - Can incorporate scheduled events, market hours
3. **Attention patterns** reveal which historical bars drive predictions
4. **Quantile outputs** useful for risk management and position sizing

**Preprocessing Requirements:**
```python
requires_scaling = True
requires_sequences = True

# Scaling: Standard scaling
scaler = "standard"

# Feature sets by type
static_features = ["symbol"]                           # Time-invariant
known_future = ["hour_sin", "hour_cos", "session"]    # Known ahead of time
time_varying = ["return_1", "volume", "volatility"]   # Unknown future

# Sequence configuration
encoder_length = 60     # Historical context
decoder_length = 20     # Prediction horizon (matches H20)
```

**Hyperparameter Ranges:**

| Parameter | Range | Recommended | Notes |
|-----------|-------|-------------|-------|
| `hidden_size` | 64-256 | 128 | Model dimension |
| `attention_head_count` | 1-8 | 4 | Multi-head attention |
| `lstm_layers` | 1-3 | 2 | Encoder/decoder depth |
| `dropout` | 0.1-0.3 | 0.1 | Regularization |
| `encoder_length` | 30-120 | 60 | Historical context |
| `decoder_length` | horizon | 5-20 | Forecast steps |
| `quantiles` | [0.1, 0.5, 0.9] | [0.1, 0.5, 0.9] | Prediction intervals |

**Recommended Configuration:**

```yaml
# config/models/tft.yaml
model_name: tft
model_family: transformer

# Architecture
hidden_size: 128
attention_head_count: 4
lstm_layers: 2
dropout: 0.1

# Variable selection
hidden_continuous_size: 64
num_static_categorical_features: 1   # symbol
num_static_real_features: 0
num_known_categorical_features: 1    # session
num_known_real_features: 2           # hour_sin, hour_cos
num_unknown_real_features: 10        # OHLCV features

# Input/Output
encoder_length: 60
max_prediction_length: 20

# Quantiles (for probabilistic forecast)
quantiles: [0.1, 0.25, 0.5, 0.75, 0.9]

# Training
batch_size: 128
max_epochs: 50
learning_rate: 0.0001
weight_decay: 0.01

# Scheduler
scheduler: "reduce_on_plateau"
reduce_on_plateau_patience: 5

# Early stopping
early_stopping_patience: 10
```

**PyTorch Lightning Implementation Reference:**

```python
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.data import TimeSeriesDataSet

# Create dataset
training = TimeSeriesDataSet(
    data=train_df,
    time_idx="time_idx",
    target="target",
    group_ids=["symbol"],
    min_encoder_length=30,
    max_encoder_length=60,
    min_prediction_length=1,
    max_prediction_length=20,
    static_categoricals=["symbol"],
    time_varying_known_reals=["hour_sin", "hour_cos"],
    time_varying_unknown_reals=["return_1", "volume", "rsi"],
    target_normalizer=None,  # Already scaled
)

# Create model
model = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.0001,
    hidden_size=128,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=64,
    output_size=7,  # 7 quantiles
    loss=QuantileLoss(),
    reduce_on_plateau_patience=5,
)
```

**Expected Performance (TFT):**

| Horizon | Validation F1 | Validation Sharpe | Training Time | Notes |
|---------|---------------|-------------------|---------------|-------|
| H5 | 0.36 - 0.43 | 0.30 - 0.52 | ~45 min | Variable selection helps |
| H10 | 0.39 - 0.47 | 0.38 - 0.65 | ~45 min | Good multi-horizon |
| H15 | 0.41 - 0.49 | 0.44 - 0.75 | ~45 min | Strong attention patterns |
| H20 | 0.43 - 0.52 | 0.50 - 0.85 | ~45 min | Best for TFT |

---

### iTransformer (ICLR 2024)

**Architecture Overview:**

iTransformer inverts the standard transformer approach for multivariate time series. Instead of applying attention across time steps, it applies attention across feature channels. Each feature's time series is treated as a token, enabling the model to learn inter-feature dependencies.

```
Input: (batch, seq_len, features)
            |
    [Transpose to (batch, features, seq_len)]
            |
    [Embed each feature's series]
            |
    [Self-Attention across features]  <- Key innovation
            |
    [Feed-Forward per feature]
            |
    [Project to predictions]
            |
Output: (batch, horizon, features) or (batch, horizon)
```

**Key Characteristics:**
- **Inverted attention** - Attention on features, not timesteps
- **Channel-independence** - Each feature maintains its temporal structure
- **Multivariate native** - Designed for many correlated features
- **SOTA on benchmarks** - Best results on ETTh, Weather, Traffic datasets (2024)
- **Efficient** - Linear in sequence length, quadratic in features

**Why iTransformer Works for OHLCV:**
1. **Cross-feature learning** - Learns relationships between indicators
2. **Preserves temporal patterns** - Each feature's time series stays intact
3. **Scales with features** - Handles 50-100 features efficiently
4. **Multivariate signals** - OHLCV naturally has correlated channels

**Preprocessing Requirements:**
```python
requires_scaling = True
requires_sequences = True

# Scaling: Standard or instance normalization
scaler = "standard"

# Feature set: Moderate number of meaningful features
feature_set = "itransformer_optimal"  # 30-60 features

# Sequence configuration
sequence_length = 96      # Standard benchmark length
prediction_length = 20    # Match horizon
```

**Hyperparameter Ranges:**

| Parameter | Range | Recommended | Notes |
|-----------|-------|-------------|-------|
| `d_model` | 128-512 | 256 | Embedding dimension |
| `n_heads` | 4-16 | 8 | Attention heads |
| `e_layers` | 2-4 | 3 | Encoder layers |
| `d_ff` | 256-1024 | 512 | FFN dimension |
| `dropout` | 0.1-0.3 | 0.1 | Regularization |
| `seq_len` | 48-192 | 96 | Input sequence |
| `pred_len` | horizon | 5-20 | Output length |

**Recommended Configuration:**

```yaml
# config/models/itransformer.yaml
model_name: itransformer
model_family: transformer

# Architecture
d_model: 256
n_heads: 8
e_layers: 3
d_ff: 512
dropout: 0.1
activation: "gelu"

# Embedding
embed_type: "timeF"     # Time feature embedding
freq: "t"               # Minute-level

# Input/Output
seq_len: 96
label_len: 48           # Decoder input overlap
pred_len: 20

# Training
batch_size: 32
max_epochs: 50
learning_rate: 0.0001
weight_decay: 0.0001

# Scheduler
scheduler: "cosine"
warmup_epochs: 3

# Early stopping
early_stopping_patience: 10

# Device
device: "cuda"
mixed_precision: true
```

**PyTorch Implementation Reference:**

```python
import torch
import torch.nn as nn


class InvertedAttention(nn.Module):
    """Attention across feature dimension."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, n_features, d_model)
        B, N, D = x.shape

        Q = self.W_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        # Attention across features
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.W_o(out)


class iTransformerBlock(nn.Module):
    """Single iTransformer block."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = InvertedAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class iTransformer(nn.Module):
    """iTransformer for multivariate time series."""

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        pred_len: int,
        d_model: int = 256,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        num_classes: int = 3
    ):
        super().__init__()

        # Embed each feature's sequence
        self.embedding = nn.Linear(seq_len, d_model)

        # Encoder blocks
        self.encoder = nn.ModuleList([
            iTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(e_layers)
        ])

        # Projection to output
        self.projection = nn.Linear(d_model, pred_len)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features * pred_len, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)

        # Embed each feature's sequence
        x = self.embedding(x)  # (batch, features, d_model)

        # Encoder
        for layer in self.encoder:
            x = layer(x)

        # Project to predictions
        x = self.projection(x)  # (batch, features, pred_len)

        # Classify
        logits = self.classifier(x)
        return logits
```

**Expected Performance (iTransformer):**

| Horizon | Validation F1 | Validation Sharpe | Training Time | Notes |
|---------|---------------|-------------------|---------------|-------|
| H5 | 0.38 - 0.45 | 0.35 - 0.58 | ~30 min | Strong short-term |
| H10 | 0.40 - 0.48 | 0.42 - 0.68 | ~30 min | Good multivariate |
| H15 | 0.42 - 0.50 | 0.48 - 0.78 | ~30 min | SOTA potential |
| H20 | 0.44 - 0.53 | 0.52 - 0.88 | ~30 min | Best performance |

---

### DLinear / NLinear (Simple Yet Effective)

**Architecture Overview:**

DLinear and NLinear are embarrassingly simple linear models that often outperform complex deep learning models on time series benchmarks. They challenge the assumption that attention is necessary for time series forecasting.

**DLinear (Decomposition-Linear):**
```
Input Series
      |
  [Series Decomposition]
      |
  +---+---+
  |       |
Trend  Seasonal
  |       |
[Linear] [Linear]
  |       |
  +---+---+
      |
  [Sum]
      |
Prediction
```

**NLinear (Normalization-Linear):**
```
Input Series
      |
[Subtract last value]  <- Instance normalization
      |
   [Linear]
      |
[Add last value back]
      |
Prediction
```

**Key Characteristics:**
- **Minimal complexity** - Just 1-2 linear layers
- **No attention** - Proves attention isn't always necessary
- **Fast training** - Minutes instead of hours
- **Strong baseline** - Beats many transformers on benchmarks
- **Decomposition-based** - DLinear separates trend/seasonal

**Why Linear Models Work:**
1. **Periodicity** - Financial time series have regular patterns (sessions, days)
2. **Trend continuation** - Near-term trends often persist
3. **Overfitting risk** - Complex models overfit noisy financial data
4. **Distribution shift** - Simple models generalize better across regimes

**Preprocessing Requirements:**
```python
requires_scaling = True
requires_sequences = True

# Scaling: Standard for linear models
scaler = "standard"

# Feature set: Raw OHLCV or minimal features
feature_set = "linear_raw"  # Close price or returns

# Sequence configuration
sequence_length = 96      # Match horizon ratio
prediction_length = 20
```

**Hyperparameter Ranges:**

| Parameter | Range | Recommended | Notes |
|-----------|-------|-------------|-------|
| `seq_len` | 48-192 | 96 | Input window |
| `pred_len` | horizon | 5-20 | Output length |
| `individual` | true/false | true | Separate linear per channel |
| `kernel_size` | 13-25 | 25 | For decomposition (DLinear) |

**Recommended Configuration:**

```yaml
# config/models/dlinear.yaml
model_name: dlinear
model_family: linear

# Architecture
individual: true          # Separate linear per feature
kernel_size: 25           # Moving average kernel for decomposition

# Input/Output
seq_len: 96
pred_len: 20

# Training
batch_size: 32
max_epochs: 20            # Converges fast
learning_rate: 0.005      # Higher LR for linear
weight_decay: 0.0

# Early stopping
early_stopping_patience: 5
```

```yaml
# config/models/nlinear.yaml
model_name: nlinear
model_family: linear

# Architecture
individual: true

# Input/Output
seq_len: 96
pred_len: 20

# Training
batch_size: 32
max_epochs: 20
learning_rate: 0.005
weight_decay: 0.0

early_stopping_patience: 5
```

**PyTorch Implementation Reference:**

```python
import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    """Moving average for series decomposition."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        self.avg = nn.AvgPool1d(kernel_size, stride=1, padding=padding)

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        x = self.avg(x)
        x = x.transpose(1, 2)  # (batch, seq_len, features)
        return x


class SeriesDecomposition(nn.Module):
    """Decompose series into trend and residual."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinear(nn.Module):
    """Decomposition-Linear for time series forecasting."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_features: int,
        individual: bool = True,
        kernel_size: int = 25,
        num_classes: int = 3
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features

        self.decomposition = SeriesDecomposition(kernel_size)

        if individual:
            self.linear_seasonal = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(n_features)
            ])
            self.linear_trend = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(n_features)
            ])
        else:
            self.linear_seasonal = nn.Linear(seq_len, pred_len)
            self.linear_trend = nn.Linear(seq_len, pred_len)

        self.individual = individual

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features * pred_len, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        seasonal, trend = self.decomposition(x)

        if self.individual:
            seasonal_out = torch.stack([
                self.linear_seasonal[i](seasonal[:, :, i])
                for i in range(self.n_features)
            ], dim=-1)
            trend_out = torch.stack([
                self.linear_trend[i](trend[:, :, i])
                for i in range(self.n_features)
            ], dim=-1)
        else:
            seasonal_out = self.linear_seasonal(seasonal.transpose(1, 2)).transpose(1, 2)
            trend_out = self.linear_trend(trend.transpose(1, 2)).transpose(1, 2)

        forecast = seasonal_out + trend_out
        logits = self.classifier(forecast)
        return logits


class NLinear(nn.Module):
    """Normalization-Linear for time series forecasting."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_features: int,
        individual: bool = True,
        num_classes: int = 3
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.individual = individual

        if individual:
            self.linear = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(n_features)
            ])
        else:
            self.linear = nn.Linear(seq_len, pred_len)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features * pred_len, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        # Instance normalization: subtract last value
        last_value = x[:, -1:, :]
        x = x - last_value

        if self.individual:
            out = torch.stack([
                self.linear[i](x[:, :, i])
                for i in range(self.n_features)
            ], dim=-1)
        else:
            out = self.linear(x.transpose(1, 2)).transpose(1, 2)

        # Add back last value
        out = out + last_value

        logits = self.classifier(out)
        return logits
```

**Expected Performance (DLinear/NLinear):**

| Horizon | Validation F1 | Validation Sharpe | Training Time | Notes |
|---------|---------------|-------------------|---------------|-------|
| H5 | 0.35 - 0.42 | 0.28 - 0.48 | ~2 min | Simple but effective |
| H10 | 0.37 - 0.44 | 0.32 - 0.55 | ~2 min | Good baseline |
| H15 | 0.39 - 0.46 | 0.36 - 0.62 | ~2 min | Decomposition helps |
| H20 | 0.41 - 0.48 | 0.40 - 0.70 | ~2 min | Best for linear |

---

### Mamba (State Space Model)

**Architecture Overview:**

Mamba is a selective state space model (SSM) that achieves linear complexity O(n) in sequence length while matching or exceeding transformer performance. It combines the modeling power of attention with the efficiency of RNNs.

```
Input Sequence
      |
  [Embedding]
      |
  [Mamba Block 1]
      |-- Conv1D
      |-- Selective SSM
      |-- Multiplicative Gate
      |
  [Mamba Block 2]
      |
      ...
      |
  [Output Projection]
```

**Key Characteristics:**
- **Linear complexity** - O(n) vs O(n^2) for attention
- **Selective mechanism** - Input-dependent state transitions
- **Long context** - Handles 100K+ tokens efficiently
- **Hardware efficient** - Optimized CUDA kernels
- **Recurrent inference** - Fast autoregressive generation

**Why Mamba Works for OHLCV:**
1. **Long sequences** - Can process entire trading days/weeks
2. **Efficiency** - 5x faster than transformers for long sequences
3. **Selectivity** - Learns which past information to remember
4. **State compression** - Efficient memory of long-range patterns

**Preprocessing Requirements:**
```python
requires_scaling = True
requires_sequences = True

# Scaling: Standard
scaler = "standard"

# Feature set: Can handle many features
feature_set = "neural_optimal"  # 40-60 features

# Sequence configuration - Mamba handles long sequences
sequence_length = 256     # Can go much longer
stride = 1
```

**Hyperparameter Ranges:**

| Parameter | Range | Recommended | Notes |
|-----------|-------|-------------|-------|
| `d_model` | 128-512 | 256 | Model dimension |
| `d_state` | 16-64 | 16 | SSM state dimension |
| `d_conv` | 2-8 | 4 | Convolution kernel size |
| `expand` | 2-4 | 2 | Expansion factor |
| `n_layers` | 4-12 | 6 | Number of Mamba blocks |
| `dropout` | 0.0-0.2 | 0.1 | Regularization |
| `seq_len` | 128-1024 | 256 | Input sequence |

**Recommended Configuration:**

```yaml
# config/models/mamba.yaml
model_name: mamba
model_family: state_space

# Architecture
d_model: 256
d_state: 16               # SSM state dimension
d_conv: 4                 # Local convolution
expand: 2                 # Inner dimension expansion
n_layers: 6
dropout: 0.1

# Input/Output
seq_len: 256
pred_len: 20

# Training
batch_size: 64
max_epochs: 50
learning_rate: 0.0001
weight_decay: 0.01

# Scheduler
scheduler: "cosine"
warmup_epochs: 5

# Early stopping
early_stopping_patience: 10

# Device
device: "cuda"
mixed_precision: true     # Requires CUDA kernels
```

**PyTorch Implementation Reference:**

```python
# Official Mamba implementation
# pip install mamba-ssm

from mamba_ssm import Mamba
import torch
import torch.nn as nn


class MambaBlock(nn.Module):
    """Single Mamba block."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x) + residual
        return x


class MambaClassifier(nn.Module):
    """Mamba for time series classification."""

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 6,
        dropout: float = 0.1,
        num_classes: int = 3
    ):
        super().__init__()

        # Input projection
        self.embedding = nn.Linear(n_features, d_model)

        # Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])

        # Output
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        # Use last position for classification
        x = x[:, -1, :]
        logits = self.classifier(x)
        return logits
```

**Expected Performance (Mamba):**

| Horizon | Validation F1 | Validation Sharpe | Training Time | Notes |
|---------|---------------|-------------------|---------------|-------|
| H5 | 0.37 - 0.44 | 0.32 - 0.55 | ~25 min | Good short-term |
| H10 | 0.40 - 0.48 | 0.40 - 0.68 | ~25 min | Strong baseline |
| H15 | 0.42 - 0.50 | 0.46 - 0.78 | ~25 min | Long context helps |
| H20 | 0.44 - 0.52 | 0.52 - 0.85 | ~25 min | Best performance |

---

### Chronos-Bolt (Foundation Model)

**Architecture Overview:**

Chronos-Bolt is Amazon's latest foundation model for time series, offering 600x speedup over the original Chronos. It's a T5-based encoder-decoder architecture pre-trained on massive time series corpora.

```
                 Pre-training
                     |
    [Billions of time series from public datasets]
                     |
              [T5 Architecture]
                     |
    +----------------+----------------+
    |                                 |
Zero-Shot                       Fine-Tuned
    |                                 |
[Direct prediction]         [Task-specific training]
[No training needed]        [Higher accuracy]
```

**Key Characteristics:**
- **Foundation model** - Pre-trained on diverse time series
- **Zero-shot capable** - Works without any training
- **Fine-tunable** - Improves with task-specific data
- **600x faster** - Optimized Bolt variant
- **Probabilistic** - Produces prediction intervals

**Model Variants:**

| Variant | Parameters | Context | Latency | Use Case |
|---------|------------|---------|---------|----------|
| chronos-bolt-tiny | 8M | 512 | 1ms | Edge/Mobile |
| chronos-bolt-mini | 20M | 512 | 2ms | Low-latency |
| chronos-bolt-small | 46M | 512 | 5ms | Balanced |
| chronos-bolt-base | 200M | 512 | 15ms | High accuracy |

**Why Chronos-Bolt Works for OHLCV:**
1. **Zero-shot baseline** - Immediate predictions without training
2. **Transfer learning** - Pre-trained patterns transfer to finance
3. **Uncertainty quantification** - Prediction intervals for risk
4. **Fine-tuning** - Adapts to specific symbols/regimes

**Preprocessing Requirements:**
```python
requires_scaling = False   # Chronos handles normalization internally
requires_sequences = True

# Feature set: Raw price series preferred
feature_set = "chronos_raw"  # Close price only, or OHLCV

# Sequence configuration
context_length = 512      # Maximum context
prediction_length = 20    # Match horizon
```

**Hyperparameter Ranges:**

| Parameter | Range | Recommended | Notes |
|-----------|-------|-------------|-------|
| `model_size` | tiny/mini/small/base | "small" | Size-accuracy tradeoff |
| `context_length` | 64-512 | 512 | More context = better |
| `prediction_length` | 1-64 | 20 | Match horizon |
| `num_samples` | 20-100 | 20 | For probabilistic output |
| `temperature` | 0.5-1.5 | 1.0 | Sampling diversity |

**Zero-Shot Configuration:**

```yaml
# config/models/chronos_bolt_zeroshot.yaml
model_name: chronos_bolt
model_family: foundation

# Model
model_id: "amazon/chronos-bolt-small"
use_zero_shot: true

# Input/Output
context_length: 512
prediction_length: 20

# Inference
num_samples: 20           # Number of probabilistic samples
temperature: 1.0
device: "cuda"
```

**Fine-Tuning Configuration:**

```yaml
# config/models/chronos_bolt_finetune.yaml
model_name: chronos_bolt
model_family: foundation

# Model
model_id: "amazon/chronos-bolt-small"
use_zero_shot: false

# Fine-tuning
learning_rate: 0.00001    # Low LR for fine-tuning
max_epochs: 10
batch_size: 32

# Freeze early layers
freeze_encoder: false
freeze_epochs: 3          # Freeze for first N epochs

# Input/Output
context_length: 512
prediction_length: 20

# Early stopping
early_stopping_patience: 5
```

**Python Implementation Reference:**

```python
from chronos import ChronosBoltPipeline
import torch
import numpy as np


class ChronosBoltModel:
    """Chronos-Bolt wrapper for time series classification."""

    def __init__(
        self,
        model_id: str = "amazon/chronos-bolt-small",
        context_length: int = 512,
        prediction_length: int = 20,
        device: str = "cuda",
        num_classes: int = 3
    ):
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.device = device
        self.num_classes = num_classes

        # Load Chronos-Bolt
        self.pipeline = ChronosBoltPipeline.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        Generate probabilistic forecasts and convert to class probabilities.

        Args:
            x: Input sequences (batch, seq_len) - univariate close prices

        Returns:
            Class probabilities (batch, num_classes)
        """
        # Get probabilistic forecasts
        forecasts = self.pipeline.predict(
            context=torch.tensor(x, dtype=torch.float32),
            prediction_length=self.prediction_length,
            num_samples=20,
        )  # Shape: (batch, num_samples, pred_len)

        # Convert to classification
        # Strategy: Compare median forecast to current price
        median_forecast = np.median(forecasts.numpy(), axis=1)  # (batch, pred_len)
        current_price = x[:, -1]  # (batch,)

        # Compute return prediction
        final_forecast = median_forecast[:, -1]
        predicted_return = (final_forecast - current_price) / current_price

        # Convert to class probabilities using forecast distribution
        lower = np.percentile(forecasts.numpy()[:, :, -1], 25, axis=1)
        upper = np.percentile(forecasts.numpy()[:, :, -1], 75, axis=1)

        # Simple thresholding (can be made more sophisticated)
        proba = np.zeros((len(x), self.num_classes))
        for i in range(len(x)):
            if predicted_return[i] < -0.001:  # Short signal
                proba[i] = [0.6, 0.3, 0.1]
            elif predicted_return[i] > 0.001:  # Long signal
                proba[i] = [0.1, 0.3, 0.6]
            else:  # Neutral
                proba[i] = [0.2, 0.6, 0.2]

        return proba

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return class predictions."""
        proba = self.predict_proba(x)
        return np.argmax(proba, axis=1)


# Usage
model = ChronosBoltModel("amazon/chronos-bolt-small")
predictions = model.predict(close_prices)
```

**Fine-Tuning Example:**

```python
from chronos import ChronosBoltPipeline
from transformers import TrainingArguments, Trainer
import torch


def fine_tune_chronos(
    train_data: np.ndarray,
    train_labels: np.ndarray,
    model_id: str = "amazon/chronos-bolt-small",
    output_dir: str = "chronos_finetuned"
):
    """Fine-tune Chronos-Bolt on custom data."""

    # Load pre-trained model
    pipeline = ChronosBoltPipeline.from_pretrained(model_id)
    model = pipeline.model

    # Add classification head
    model.classifier = torch.nn.Linear(
        model.config.hidden_size,
        3  # num_classes
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=32,
        learning_rate=1e-5,
        warmup_ratio=0.1,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.train()

    return model
```

**Expected Performance (Chronos-Bolt):**

| Horizon | Zero-Shot F1 | Fine-Tuned F1 | Zero-Shot Sharpe | Fine-Tuned Sharpe |
|---------|--------------|---------------|------------------|-------------------|
| H5 | 0.32 - 0.38 | 0.38 - 0.46 | 0.20 - 0.40 | 0.35 - 0.60 |
| H10 | 0.34 - 0.40 | 0.41 - 0.49 | 0.25 - 0.45 | 0.42 - 0.70 |
| H15 | 0.36 - 0.42 | 0.43 - 0.51 | 0.30 - 0.50 | 0.48 - 0.78 |
| H20 | 0.38 - 0.44 | 0.45 - 0.54 | 0.35 - 0.55 | 0.52 - 0.85 |

---

## Model Selection Guide

### Decision Tree

```
Start Here
    |
    v
Do you have enough data (>50K samples)?
    |
    +-- No --> Use Chronos-Bolt (zero-shot)
    |           or DLinear/NLinear (simple)
    |
    +-- Yes
         |
         v
    Is interpretability critical?
         |
         +-- Yes --> Use N-BEATS (trend/seasonal)
         |           or XGBoost (feature importance)
         |
         +-- No
              |
              v
         Need sub-second inference?
              |
              +-- Yes --> Use TCN or DLinear
              |           (parallelizable, fast)
              |
              +-- No
                   |
                   v
              Many correlated features (>30)?
                   |
                   +-- Yes --> Use iTransformer
                   |           (feature attention)
                   |
                   +-- No
                        |
                        v
                   Very long context needed (>256)?
                        |
                        +-- Yes --> Use Mamba
                        |           (linear complexity)
                        |
                        +-- No
                             |
                             v
                        Multi-horizon with covariates?
                             |
                             +-- Yes --> Use TFT
                             |
                             +-- No --> Use TCN or LSTM
```

### Model Comparison Matrix

| Model | Training Speed | Inference Speed | Data Efficiency | Interpretability | Long Context |
|-------|----------------|-----------------|-----------------|------------------|--------------|
| XGBoost | Fast | Fast | High | High | N/A |
| LSTM | Slow | Medium | Medium | Low | Medium |
| TCN | Medium | Fast | Medium | Low | High |
| N-BEATS | Medium | Fast | High | High | Medium |
| TFT | Slow | Medium | Medium | High | Medium |
| iTransformer | Medium | Medium | Medium | Medium | Medium |
| DLinear | Very Fast | Very Fast | High | Medium | Medium |
| Mamba | Medium | Fast | Medium | Low | Very High |
| Chronos-Bolt | N/A (zero-shot) | Fast | Very High | Low | High |

---

## Feature Set Selection

### Phase 1 Feature Sets

Phase 1 provides pre-defined feature sets optimized for each model family:

| Feature Set | Target Models | Features | Scaling |
|-------------|---------------|----------|---------|
| `boosting_optimal` | XGBoost, LightGBM, CatBoost, RF | 80-120 | None |
| `neural_optimal` | LSTM, GRU, MLP | 40-60 | RobustScaler |
| `transformer_raw` | PatchTST, TimesNet, TimesFM | 10-15 | StandardScaler |
| `ensemble_base` | Diverse features for ensembles | 60-80 | RobustScaler |

### Feature Set Contents

**`boosting_optimal`:**
```python
include_prefixes = [
    # Price action
    "return_", "log_return_", "roc_",
    # Momentum oscillators
    "rsi_", "macd_", "stoch_", "williams_", "cci_", "mfi_",
    # Trend indicators
    "adx_", "supertrend",
    # Volatility
    "atr_", "hvol_", "parkinson_", "garman_", "bb_width", "bb_position", "kc_position",
    # Volume
    "volume_", "obv",
    # Temporal
    "hour_", "dayofweek_", "session_",
    # Microstructure
    "micro_",
    # Wavelets (optional, can help)
    "wavelet_",
]
# Excludes: raw prices (sma_, ema_, vwap, open_, high_, low_, close_)
```

**`neural_optimal`:**
```python
include_prefixes = [
    # Normalized returns
    "return_", "log_return_",
    # Bounded oscillators [0-100] or [-100, 100]
    "rsi_", "stoch_", "williams_", "mfi_",
    # Normalized volatility
    "hvol_", "atr_ratio", "bb_position", "kc_position",
    # Volume ratios
    "volume_ratio", "volume_zscore",
    # Wavelets (significant improvement)
    "wavelet_",
]
include_columns = [
    # Cyclical temporal
    "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos",
]
exclude_prefixes = [
    # Raw unbounded values
    "sma_", "ema_", "bb_upper", "bb_lower", "vwap",
    "open_", "high_", "low_", "close_",
]
```

**`transformer_raw`:**
```python
include_prefixes = [
    "return_", "log_return_",
    "volume_ratio",
]
include_columns = [
    "hour_sin", "hour_cos",
    "dayofweek_sin", "dayofweek_cos",
]
# Transformers learn patterns from minimal features
# Add wavelets only if not using raw OHLCV input
```

### Sample-to-Feature Ratio

Maintain proper ratio to prevent overfitting:

| Sample Size | Minimum Ratio | Optimal Ratio | Max Features |
|-------------|---------------|---------------|--------------|
| 50,000 | 10:1 | 20:1 | 100 |
| 100,000 | 10:1 | 20:1 | 200 |
| 200,000 | 10:1 | 20:1 | 400 |
| 500,000 | 10:1 | 20:1 | 500 |

---

## Training Infrastructure

### Experiment Tracking

```python
@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""
    project_name: str = "ohlcv-model-factory"
    experiment_name: str = ""
    run_name: str = ""
    tags: Dict[str, str] = None

    # Tracking backend
    backend: str = "mlflow"  # or "wandb", "tensorboard"
    tracking_uri: str = "mlruns"

    # Logging
    log_params: bool = True
    log_metrics: bool = True
    log_artifacts: bool = True
    log_model: bool = True
```

### Checkpoint Management

```python
class CheckpointManager:
    """Manages model checkpoints during training."""

    def __init__(self, output_dir: Path, keep_top_k: int = 3):
        self.output_dir = output_dir
        self.keep_top_k = keep_top_k
        self.checkpoints = []

    def save_checkpoint(
        self,
        model: BaseModel,
        epoch: int,
        metric_value: float,
        metric_name: str = "val_loss"
    ) -> Path:
        """Save checkpoint if it's in top-k."""
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"

        self.checkpoints.append({
            "path": checkpoint_path,
            "epoch": epoch,
            "metric": metric_value
        })

        # Sort by metric (lower is better for loss)
        self.checkpoints.sort(key=lambda x: x["metric"])

        # Keep only top-k
        if len(self.checkpoints) > self.keep_top_k:
            removed = self.checkpoints.pop()
            removed["path"].unlink(missing_ok=True)

        # Save if in top-k
        if checkpoint_path in [c["path"] for c in self.checkpoints]:
            model.save(checkpoint_path)

        return checkpoint_path
```

### GPU Memory Management

```python
def estimate_memory_requirements(
    model_family: str,
    batch_size: int,
    sequence_length: int,
    n_features: int,
    hidden_size: int = 128
) -> Dict[str, float]:
    """Estimate GPU memory requirements in GB."""

    if model_family == "boosting":
        # CPU-based, minimal GPU memory
        return {"gpu_memory_gb": 0, "cpu_memory_gb": 2}

    elif model_family == "neural":
        # LSTM/GRU memory estimation
        param_count = (
            4 * hidden_size * (n_features + hidden_size + 1) * 2  # 2 layers
            + hidden_size * 3  # classifier
        )
        activation_memory = batch_size * sequence_length * hidden_size * 4  # bytes

        total_gb = (param_count * 4 + activation_memory * 3) / 1e9
        return {"gpu_memory_gb": total_gb * 1.5, "cpu_memory_gb": 4}

    elif model_family == "transformer":
        # Transformer memory estimation
        param_count = (
            12 * hidden_size * hidden_size * 3  # 3 layers
            + sequence_length * hidden_size
        )
        activation_memory = batch_size * sequence_length * hidden_size * 8

        total_gb = (param_count * 4 + activation_memory * 3) / 1e9
        return {"gpu_memory_gb": total_gb * 2, "cpu_memory_gb": 8}
```

---

## Output Structure

### Directory Layout

```
experiments/runs/{model}_{horizon}_{timestamp}/
|
+-- config/
|   +-- training_config.yaml        # Full training configuration
|   +-- model_config.yaml           # Model-specific parameters
|   +-- feature_config.yaml         # Feature set used
|
+-- checkpoints/
|   +-- best_model.pkl              # Best model by validation metric
|   +-- checkpoint_epoch_*.pt       # Training checkpoints
|
+-- predictions/
|   +-- val_predictions.parquet     # Validation set predictions
|   +-- val_probabilities.npy       # Class probabilities
|
+-- metrics/
|   +-- training_metrics.json       # Per-epoch training metrics
|   +-- evaluation_metrics.json     # Final evaluation metrics
|   +-- feature_importance.json     # If available
|
+-- plots/
|   +-- training_curves.png         # Loss/accuracy curves
|   +-- confusion_matrix.png        # Validation confusion matrix
|   +-- feature_importance.png      # If available
|
+-- logs/
    +-- training.log                # Training logs
```

### Metrics Schema

```json
{
  "evaluation_metrics": {
    "horizon": 20,
    "model_name": "xgboost",
    "timestamp": "2025-12-24T10:30:00Z",

    "classification": {
      "accuracy": 0.48,
      "macro_f1": 0.45,
      "weighted_f1": 0.46,
      "per_class_f1": {"short": 0.42, "neutral": 0.48, "long": 0.45},
      "confusion_matrix": [[1200, 400, 200], [300, 1500, 300], [250, 350, 1400]]
    },

    "trading": {
      "sharpe_ratio": 0.65,
      "annualized_return": 0.12,
      "max_drawdown": -0.15,
      "win_rate": 0.52,
      "profit_factor": 1.35,
      "num_trades": 2500
    },

    "training": {
      "epochs_trained": 85,
      "best_epoch": 75,
      "training_time_seconds": 1250,
      "early_stopped": true
    }
  }
}
```

---

## Adding a New Model

### Step-by-Step Guide

1. **Create model file:**
```python
# src/models/boosting/my_model.py

from src.models.base import BaseModel, ModelRegistry, TrainingMetrics, PredictionOutput
from typing import Dict, Optional, Any
import numpy as np


@ModelRegistry.register(name="my_model", family="boosting", description="My custom model")
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

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "param1": 100,
            "param2": 0.1,
        }

    def fit(self, X_train, y_train, X_val, y_val, sample_weights=None, config=None):
        # Training implementation
        ...
        return TrainingMetrics(...)

    def predict(self, X) -> PredictionOutput:
        # Prediction implementation
        ...
        return PredictionOutput(...)

    def save(self, path):
        ...

    def load(self, path):
        ...
```

2. **Add configuration file:**
```yaml
# config/models/my_model.yaml
model_name: my_model
model_family: boosting
param1: 100
param2: 0.1
```

3. **Register in package init:**
```python
# src/models/boosting/__init__.py
from .my_model import MyModel  # Auto-registers via decorator
```

4. **Use the model:**
```bash
python scripts/train_model.py --model my_model --horizon 20
```

---

## Success Criteria

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Registry works | Discover and instantiate any registered model | Unit tests |
| TimeSeriesDataset | Correct windowing, no leakage, symbol isolation | Integration tests |
| Trainer generalized | Works with ANY model via BaseModel interface | All model families pass |
| Multiple families | At least boosting + neural + classical | 3+ families implemented |
| Model thresholds | F1 > 0.35, Sharpe > 0.3 on at least one horizon | Validation metrics |
| Consistent outputs | All models generate same format for ensemble | Schema validation |
| Reproducibility | Same seed = same results | Determinism tests |

---

## Usage Examples

**NOTE: These scripts do not currently exist. The current implementation uses the CLI in `src/cli/run_commands.py`.**

```bash
# PLANNED (not yet implemented):
# python scripts/train_model.py --model xgboost --horizon 20

# CURRENT IMPLEMENTATION:
# Phase 2 model training is not yet implemented.
# Use the Phase 1 data pipeline to prepare datasets:
./pipeline run --symbols MES,MGC

# Then implement Phase 2 model training using the BaseModel interface
# and ModelRegistry defined in this document.
```

---

## Dependencies

```bash
# Core
pip install numpy pandas scikit-learn

# Boosting
pip install xgboost lightgbm catboost

# Neural networks
pip install torch lightning

# Transformers
pip install transformers huggingface_hub

# Time series specific
pip install neuralforecast  # N-HiTS, TFT

# Experiment tracking
pip install mlflow optuna

# Visualization
pip install matplotlib seaborn plotly
```

---

## Next Step

Phase 2 models feed into Phase 3 (Cross-Validation) for unbiased performance estimates and out-of-fold predictions that become training data for the Phase 4 ensemble.
