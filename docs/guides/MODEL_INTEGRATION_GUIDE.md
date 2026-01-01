# Model Integration Guide

**Purpose:** Step-by-step guide for adding new model types to the ML factory infrastructure
**Audience:** ML engineers adding models to the pipeline
**Last Updated:** 2026-01-01

---

## IMPORTANT: Current vs Intended Architecture

> **WARNING:** This guide describes model integration for the current universal pipeline.
> The INTENDED architecture uses model-specific data strategies. See:
> - `docs/INTENDED_ARCHITECTURE.md` - Target state (not yet implemented)
> - `docs/CURRENT_LIMITATIONS.md` - What's wrong with current approach
> - `docs/MIGRATION_ROADMAP.md` - 6-8 week implementation plan

### Current State (Temporary)

All models currently receive the same ~180 indicator-derived features:
- **Tabular models (2D):** Get indicator features - **appropriate**
- **Sequence models (3D):** Get indicator features (windowed) - **suboptimal, should get raw OHLCV bars**

### Intended State (Goal)

Three MTF strategies based on model family:
| Strategy | Models | Data Type | Status |
|----------|--------|-----------|--------|
| **Strategy 1: Single-TF** | All (baselines) | One timeframe, no MTF | Not implemented |
| **Strategy 2: MTF Indicators** | Tabular (6) | Indicators from 9 TFs | Partial (5 TFs) |
| **Strategy 3: MTF Ingestion** | Sequence (13) | Raw OHLCV from 9 TFs | Not implemented |

When adding a new model, document its **intended data requirements** even if the current pipeline cannot fulfill them.

---

## Table of Contents

1. [Overview](#overview)
2. [Model Factory Architecture](#model-factory-architecture)
3. [Adding a New Model: Quick Start](#adding-a-new-model-quick-start)
4. [BaseModel Interface](#basemodel-interface)
5. [Model Family Integration](#model-family-integration)
6. [Input Shape Requirements](#input-shape-requirements)
7. [Configuration Schema](#configuration-schema)
8. [Testing Requirements](#testing-requirements)
9. [Common Pitfalls](#common-pitfalls)
10. [Examples by Model Family](#examples-by-model-family)

---

## Overview

The ML factory uses a **plugin architecture** where models register themselves and implement a common interface. This allows:

- **Unified training pipeline:** All models train with shared infrastructure
- **Model-specific data selection:** (INTENDED) Different model families receive optimized data
- **Fair comparisons:** Identical backtest assumptions across models
- **Easy extensibility:** Add new models without rewriting core infrastructure
- **Ensemble support:** Mix any models that share input shape requirements

**Current model count:** 13 models across 4 families (boosting, neural, classical, ensemble)

**Note:** Current pipeline serves all models the same data. This is temporary - see `docs/INTENDED_ARCHITECTURE.md` for the goal state where tabular models get MTF indicators and sequence models get raw multi-resolution OHLCV bars.

---

## Model Factory Architecture

```
Raw OHLCV (1min)
    ↓
[ Phase 1: Data Pipeline ]
    ├── Clean & resample → training_timeframe (5min, 15min, etc.)
    ├── Features → base + MTF + wavelet + microstructure
    ├── Labels → triple-barrier with GA-optimized params
    ├── Splits → train/val/test with purge/embargo
    ├── Scaling → robust scaler (train-only fit)
    └── Datasets → TimeSeriesDataContainer
    ↓
[ Model Registry ]
    ├── get_tabular_data() → Boosting, Classical (2D)
    ├── get_sequence_data() → LSTM, GRU, TCN, Transformer (3D)
    └── get_multi_resolution_tensors() → PatchTST, TFT (4D)
    ↓
[ Model Training ]
    ├── fit(X_train, y_train, X_val, y_val, sample_weights)
    ├── predict(X_test) → PredictionOutput
    └── save(path) → model artifacts
    ↓
[ Unified Evaluation ]
    ├── Classification metrics (accuracy, F1, precision, recall)
    ├── Financial metrics (Sharpe, win rate, max drawdown)
    └── Regime-aware performance
```

---

## Adding a New Model: Quick Start

**Time required:** 2-6 hours for simple models, 1-2 days for complex models

### Step 1: Choose Model Family

Determine which family your model belongs to:

| Family | Input Shape | Examples | Interface |
|--------|-------------|----------|-----------|
| **Boosting** | 2D (n_samples, n_features) | XGBoost, LightGBM, CatBoost | Direct array |
| **Neural Sequence** | 3D (n_samples, seq_len, n_features) | LSTM, GRU, TCN, Transformer | Sequences |
| **Classical** | 2D (n_samples, n_features) | Random Forest, Logistic, SVM | Direct array |
| **Ensemble** | Varies | Voting, Stacking, Blending | Combines base models |
| **CNN** | 3D (n_samples, seq_len, n_features) | InceptionTime, ResNet | Sequences |
| **Advanced Transformer** | 3D/4D multi-resolution | PatchTST, iTransformer, TFT | Multi-scale |
| **Foundation** | 2D (n_samples, context_len) | Chronos, TimesFM, Moirai | Zero-shot |

### Step 2: Create Model File

```bash
# Boosting/Classical/Ensemble
touch src/models/{family}/{model_name}.py

# Neural
touch src/models/neural/{model_name}.py

# Foundation
touch src/models/foundation/{model_name}.py
```

### Step 3: Implement BaseModel Interface

```python
# src/models/{family}/{model_name}.py
from src.models.base import BaseModel, TrainingMetrics, PredictionOutput
from src.models.registry import register
from typing import Dict, Any, Optional
import numpy as np

@register(name="my_model", family="boosting")  # or "neural", "classical", etc.
class MyModel(BaseModel):
    """
    Brief description of the model.

    Key characteristics:
    - Input: 2D array (n_samples, n_features)
    - Strengths: Fast training, handles missing values, interpretable
    - Weaknesses: May overfit, no temporal modeling
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model = None
        # Initialize model-specific attributes

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
        # Validation
        self._validate_input_shape(X_train, expected_ndim=2)

        # Extract hyperparameters
        params = self._get_params(config)

        # Initialize model
        self.model = SomeModelClass(**params)

        # Train with validation set
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            sample_weight=sample_weights,
            early_stopping_rounds=50,
            verbose=False
        )

        # Return training metrics
        return TrainingMetrics(
            train_loss=self.model.best_score['train'],
            val_loss=self.model.best_score['val'],
            best_iteration=self.model.best_iteration,
            training_time=time.time() - start_time
        )

    def predict(self, X: np.ndarray) -> PredictionOutput:
        """Generate predictions."""
        self._validate_input_shape(X, expected_ndim=2)

        # Get probabilities (for classification)
        proba = self.model.predict_proba(X)

        # Get predicted class
        y_pred = np.argmax(proba, axis=1)

        # Confidence: max probability
        confidence = np.max(proba, axis=1)

        return PredictionOutput(
            predictions=y_pred,
            probabilities=proba,
            confidence=confidence
        )

    def save(self, path: Path) -> None:
        """Save model to disk."""
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "model.pkl")
        # Save config
        with open(path / "config.json", "w") as f:
            json.dump(self.config, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "MyModel":
        """Load model from disk."""
        model = joblib.load(path / "model.pkl")
        with open(path / "config.json") as f:
            config = json.load(f)
        instance = cls(config)
        instance.model = model
        return instance

    def _get_params(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract hyperparameters from config."""
        merged_config = {**self.config, **(config or {})}
        return {
            'param1': merged_config.get('param1', default_value),
            'param2': merged_config.get('param2', default_value),
            # ... all hyperparameters
        }
```

### Step 4: Register Model

The `@register(name="my_model", family="boosting")` decorator automatically adds your model to the registry.

Verify registration:
```bash
python -c "from src.models import ModelRegistry; print(ModelRegistry.list_all())"
```

### Step 5: Add Configuration File

```yaml
# config/models/my_model.yaml
model_name: my_model
family: boosting

# Hyperparameters
param1: value1
param2: value2

# Training settings
early_stopping_rounds: 50
verbose: false

# MTF strategy (optional)
mtf_strategy: single_tf  # or mtf_indicators, mtf_ingestion
training_timeframe: 15min
```

### Step 6: Add Tests

```python
# tests/phase_2_tests/test_my_model.py
import pytest
import numpy as np
from src.models.boosting.my_model import MyModel

def test_my_model_fit_predict():
    """Test basic fit/predict cycle."""
    # Create synthetic data
    X_train = np.random.randn(1000, 50)
    y_train = np.random.randint(0, 3, 1000)
    X_val = np.random.randn(200, 50)
    y_val = np.random.randint(0, 3, 200)

    # Initialize and train
    model = MyModel()
    metrics = model.fit(X_train, y_train, X_val, y_val)

    # Validate metrics
    assert metrics.train_loss is not None
    assert metrics.val_loss is not None
    assert metrics.best_iteration > 0

    # Predict
    output = model.predict(X_val)

    # Validate output
    assert output.predictions.shape == (200,)
    assert output.probabilities.shape == (200, 3)
    assert output.confidence.shape == (200,)
    assert np.all(output.confidence >= 0) and np.all(output.confidence <= 1)

def test_my_model_save_load(tmp_path):
    """Test save/load cycle."""
    # Train model
    model = MyModel()
    X_train = np.random.randn(1000, 50)
    y_train = np.random.randint(0, 3, 1000)
    X_val = np.random.randn(200, 50)
    y_val = np.random.randint(0, 3, 200)
    model.fit(X_train, y_train, X_val, y_val)

    # Save
    save_path = tmp_path / "my_model"
    model.save(save_path)

    # Load
    loaded_model = MyModel.load(save_path)

    # Compare predictions
    X_test = np.random.randn(100, 50)
    pred1 = model.predict(X_test)
    pred2 = loaded_model.predict(X_test)

    np.testing.assert_array_equal(pred1.predictions, pred2.predictions)
    np.testing.assert_array_almost_equal(pred1.probabilities, pred2.probabilities)
```

### Step 7: Train Model

```bash
# Phase 1: Generate datasets (if not already done)
./pipeline run --symbols MES

# Phase 2: Train your model
python scripts/train_model.py --model my_model --horizon 20
```

### Step 8: Evaluate Performance

```bash
# Cross-validation
python scripts/run_cv.py --models my_model --horizons 20 --n-splits 5

# Walk-forward validation
python scripts/run_walk_forward.py --models my_model --horizons 20

# Compare to baseline
python scripts/run_cv.py --models my_model,xgboost,lstm --horizons 20
```

---

## BaseModel Interface

All models must implement this interface from `src/models/base.py`:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
from pathlib import Path

@dataclass
class TrainingMetrics:
    """Metrics returned after training."""
    train_loss: float
    val_loss: float
    best_iteration: int
    training_time: float
    additional_metrics: Optional[Dict[str, Any]] = None

@dataclass
class PredictionOutput:
    """Predictions with probabilities and confidence."""
    predictions: np.ndarray  # Shape: (n_samples,) - predicted class
    probabilities: np.ndarray  # Shape: (n_samples, n_classes) - class probabilities
    confidence: np.ndarray  # Shape: (n_samples,) - confidence scores [0, 1]

class BaseModel(ABC):
    """Base class for all models in the factory."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model = None

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
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels (0, 1, 2 for long, neutral, short)
            X_val: Validation features
            y_val: Validation labels
            sample_weights: Quality-based sample weights (optional)
            config: Runtime hyperparameters (overrides self.config)

        Returns:
            TrainingMetrics with train/val loss, best iteration, time
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> PredictionOutput:
        """
        Generate predictions.

        Args:
            X: Input features

        Returns:
            PredictionOutput with predictions, probabilities, confidence
        """
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """
        Save model to disk.

        Args:
            path: Directory to save model artifacts
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseModel":
        """
        Load model from disk.

        Args:
            path: Directory containing model artifacts

        Returns:
            Loaded model instance
        """
        pass

    def _validate_input_shape(self, X: np.ndarray, expected_ndim: int):
        """Validate input array shape."""
        if X.ndim != expected_ndim:
            raise ValueError(
                f"{self.__class__.__name__} expects {expected_ndim}D input, "
                f"got {X.ndim}D with shape {X.shape}"
            )
```

### Key Design Principles

1. **Validation set is required:** All models must use validation set for early stopping or hyperparameter selection
2. **Sample weights are optional:** Tabular models should use quality-weighted samples; neural models may ignore
3. **Probabilities are required:** Even for regression-style models, output class probabilities
4. **Confidence scores:** Typically `max(probabilities)`, but can be custom (e.g., model uncertainty)
5. **Config merging:** Runtime config overrides instance config
6. **Deterministic:** Same inputs + same seed → same outputs

---

## Model Family Integration

### Boosting Models (XGBoost, LightGBM, CatBoost)

**Input:** 2D array `(n_samples, n_features)`
**Recommended features:** 150-200 (dense MTF indicators)
**Sample weights:** Strongly recommended
**Early stopping:** Required (use validation set)

**Example: XGBoost Integration**

```python
# src/models/boosting/xgboost_model.py
import xgboost as xgb
from src.models.base import BaseModel, TrainingMetrics, PredictionOutput
from src.models.registry import register

@register(name="xgboost", family="boosting")
class XGBoostModel(BaseModel):
    def fit(self, X_train, y_train, X_val, y_val, sample_weights=None, config=None):
        params = self._get_params(config)

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
        dval = xgb.DMatrix(X_val, label=y_val)

        evals = [(dtrain, 'train'), (dval, 'val')]
        evals_result = {}

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=params.get('num_boost_round', 1000),
            evals=evals,
            early_stopping_rounds=params.get('early_stopping_rounds', 50),
            evals_result=evals_result,
            verbose_eval=False
        )

        return TrainingMetrics(
            train_loss=evals_result['train']['mlogloss'][-1],
            val_loss=evals_result['val']['mlogloss'][-1],
            best_iteration=self.model.best_iteration,
            training_time=time.time() - start_time
        )
```

**Key considerations:**
- Use `DMatrix` for efficient data handling
- Enable early stopping (50 rounds typical)
- Log loss ('mlogloss') for 3-class classification
- Sample weights directly supported

---

### Neural Sequence Models (LSTM, GRU, TCN, Transformer)

**Input:** 3D array `(n_samples, seq_len, n_features)`
**Recommended features:** 25-30 (raw OHLCV + wavelets)
**Sample weights:** Optional (can use weighted loss)
**Early stopping:** Required (monitor validation loss)
**Device:** GPU recommended (10-50x speedup)

**Example: LSTM Integration**

```python
# src/models/neural/lstm_model.py
import torch
import torch.nn as nn
from src.models.base import BaseModel, TrainingMetrics, PredictionOutput
from src.models.registry import register
from src.models.device import get_device

@register(name="lstm", family="neural")
class LSTMModel(BaseModel):
    def __init__(self, config=None):
        super().__init__(config)
        self.device = get_device()
        self.model = None

    def fit(self, X_train, y_train, X_val, y_val, sample_weights=None, config=None):
        self._validate_input_shape(X_train, expected_ndim=3)

        params = self._get_params(config)

        # Build model
        n_features = X_train.shape[2]
        self.model = nn.LSTM(
            input_size=n_features,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params['dropout'],
            batch_first=True
        ).to(self.device)

        # Add classifier head
        self.classifier = nn.Linear(params['hidden_size'], 3).to(self.device)

        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=params['learning_rate'])
        criterion = nn.CrossEntropyLoss()

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(params['max_epochs']):
            # Train
            train_loss = self._train_epoch(X_train, y_train, optimizer, criterion)

            # Validate
            val_loss = self._validate_epoch(X_val, y_val, criterion)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= params['patience']:
                    break

        return TrainingMetrics(
            train_loss=train_loss,
            val_loss=best_val_loss,
            best_iteration=epoch,
            training_time=time.time() - start_time
        )

    def predict(self, X):
        self._validate_input_shape(X, expected_ndim=3)

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)

            # LSTM forward
            lstm_out, _ = self.model(X_tensor)

            # Take last timestep
            last_hidden = lstm_out[:, -1, :]

            # Classifier
            logits = self.classifier(last_hidden)
            proba = torch.softmax(logits, dim=1).cpu().numpy()

        predictions = np.argmax(proba, axis=1)
        confidence = np.max(proba, axis=1)

        return PredictionOutput(predictions, proba, confidence)
```

**Key considerations:**
- Always use `batch_first=True` for consistency
- Gradient clipping recommended (`torch.nn.utils.clip_grad_norm_`)
- Monitor GPU memory usage
- Save model state dict, not entire model

---

### Classical Models (Random Forest, Logistic, SVM)

**Input:** 2D array `(n_samples, n_features)`
**Recommended features:** 150-200 (dense MTF indicators)
**Sample weights:** Supported by scikit-learn
**Early stopping:** Not applicable (RF has implicit stopping)

**Example: Random Forest Integration**

```python
# src/models/classical/random_forest.py
from sklearn.ensemble import RandomForestClassifier
from src.models.base import BaseModel, TrainingMetrics, PredictionOutput
from src.models.registry import register

@register(name="random_forest", family="classical")
class RandomForestModel(BaseModel):
    def fit(self, X_train, y_train, X_val, y_val, sample_weights=None, config=None):
        self._validate_input_shape(X_train, expected_ndim=2)

        params = self._get_params(config)

        self.model = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            max_features=params['max_features'],
            random_state=params.get('random_state', 42),
            n_jobs=-1
        )

        self.model.fit(X_train, y_train, sample_weight=sample_weights)

        # Compute losses
        train_proba = self.model.predict_proba(X_train)
        val_proba = self.model.predict_proba(X_val)

        from sklearn.metrics import log_loss
        train_loss = log_loss(y_train, train_proba)
        val_loss = log_loss(y_val, val_proba)

        return TrainingMetrics(
            train_loss=train_loss,
            val_loss=val_loss,
            best_iteration=params['n_estimators'],
            training_time=time.time() - start_time
        )
```

---

### Ensemble Models (Voting, Stacking, Blending)

**Input:** Varies (depends on base models)
**Base models:** Must have same input shape (all 2D or all 3D)
**OOF predictions:** Required for stacking/blending

**Example: Voting Ensemble**

```python
# src/models/ensemble/voting.py
from src.models.base import BaseModel, TrainingMetrics, PredictionOutput
from src.models.registry import register, ModelRegistry

@register(name="voting", family="ensemble")
class VotingEnsemble(BaseModel):
    def __init__(self, config=None):
        super().__init__(config)
        self.base_models = []

    def fit(self, X_train, y_train, X_val, y_val, sample_weights=None, config=None):
        params = self._get_params(config)

        # Initialize base models
        base_model_names = params['base_models']  # e.g., ['xgboost', 'lightgbm', 'catboost']

        for name in base_model_names:
            model_class = ModelRegistry.get(name)
            model = model_class()

            # Train base model
            model.fit(X_train, y_train, X_val, y_val, sample_weights)
            self.base_models.append(model)

        # Compute ensemble validation loss
        val_proba = self._vote(X_val)
        val_loss = log_loss(y_val, val_proba)

        return TrainingMetrics(
            train_loss=0.0,  # Not computed for voting
            val_loss=val_loss,
            best_iteration=len(self.base_models),
            training_time=time.time() - start_time
        )

    def _vote(self, X):
        """Average probabilities from all base models."""
        probas = []
        for model in self.base_models:
            output = model.predict(X)
            probas.append(output.probabilities)

        # Simple average (can also use weighted)
        avg_proba = np.mean(probas, axis=0)
        return avg_proba

    def predict(self, X):
        avg_proba = self._vote(X)
        predictions = np.argmax(avg_proba, axis=1)
        confidence = np.max(avg_proba, axis=1)

        return PredictionOutput(predictions, avg_proba, confidence)
```

---

## Input Shape Requirements

> **Note:** This section describes both CURRENT and INTENDED data access patterns.
> The intended patterns for Strategy 3 (multi-resolution) are NOT YET IMPLEMENTED.

### 2D Input: Tabular Models (Strategy 2 - Appropriate)

**Shape:** `(n_samples, n_features)`
**Models:** XGBoost, LightGBM, CatBoost, Random Forest, Logistic, SVM (6 models)
**Intended Data:** MTF indicator features from 9 timeframes
**Current Data:** MTF indicator features from 5 timeframes (~180 features)
**Status:** Appropriate - tabular models work well with indicator features

**Data source:** `container.get_sklearn_arrays(split='train')`

```python
# Get 2D data (CURRENT - WORKS)
X_train, y_train, weights_train = container.get_sklearn_arrays(split='train')
X_val, y_val, weights_val = container.get_sklearn_arrays(split='val')

# Shape: (n_samples, n_features)
# Example: (15000, 180) - 15k samples, 180 indicator features
```

### 3D Input: Sequence Models (Strategy 3 - Not Implemented)

**Shape:** `(n_samples, seq_len, n_features)`
**Models:** LSTM, GRU, TCN, Transformer (4 implemented) + InceptionTime, ResNet, PatchTST, iTransformer, TFT, N-BEATS (6 planned) = 13 models
**Intended Data:** Raw multi-resolution OHLCV bars from 9 timeframes
**Current Data:** Same ~180 indicator features (windowed) - **SUBOPTIMAL**
**Status:** Current approach works but is suboptimal for temporal learning

**Current data source:** `container.get_pytorch_sequences(split='train', seq_len=60)`

```python
# CURRENT IMPLEMENTATION (suboptimal for sequence models)
X_train, y_train, weights_train = container.get_pytorch_sequences(
    split='train',
    seq_len=60  # Use last 60 bars for each prediction
)

# Shape: (n_samples, seq_len, n_features)
# Example: (14940, 60, 180) - 14940 samples, 60 timesteps, 180 INDICATOR features
# Problem: Models receive pre-computed indicators, not raw OHLCV bars
```

```python
# INTENDED IMPLEMENTATION (Strategy 3 - NOT YET IMPLEMENTED)
# This is what sequence models SHOULD receive:
X_multi_train = container.get_multi_resolution_bars(
    split='train',
    input_timeframes=['1min', '5min', '15min', '1h']
)

# Shape: dict of (n_samples, seq_len, 5) for OHLCV+Volume
# Example:
# X_multi_train = {
#     '1min': (14940, 60, 5),   # Last 60 1min bars (raw OHLCV)
#     '5min': (14940, 12, 5),   # Last 60 minutes at 5min resolution
#     '15min': (14940, 4, 5),   # Last 60 minutes at 15min resolution
#     '1h': (14940, 1, 5),      # Current 1h bar
# }
# Models learn temporal patterns from RAW price movements
```

**Important:** When adding a new sequence model, document that it should ideally receive raw OHLCV bars (Strategy 3) but currently receives indicator features.

### 4D Input: Multi-Resolution Models (Strategy 3 - Not Implemented)

**Shape:** `(n_samples, n_timeframes, seq_len, n_features)`
**Models:** PatchTST, iTransformer, TFT (Strategy 3: MTF ingestion)
**Status:** NOT IMPLEMENTED - these advanced models are planned but Strategy 3 infrastructure doesn't exist yet

```python
# INTENDED IMPLEMENTATION (NOT YET AVAILABLE)
# This interface doesn't exist yet - planned for Strategy 3 implementation
X_multi_train, y_train, weights_train = container.get_multi_resolution_tensors(
    split='train',
    input_timeframes=['1min', '5min', '15min']
)

# Would return dict of tensors:
# X_multi_train = {
#     '1min': (14940, 15, 5),   # Last 15 minutes at 1min resolution
#     '5min': (14940, 3, 5),    # Last 15 minutes at 5min resolution
#     '15min': (14940, 1, 5),   # Current 15min bar
# }

# Stack into 4D tensor for models like PatchTST
X_train = stack_multi_resolution(X_multi_train, ['1min', '5min', '15min'])
# Shape: (14940, 3, max_seq_len, 5)
```

**Note:** When implementing advanced sequence models (PatchTST, TFT, etc.), they will initially use the current 3D indicator approach until Strategy 3 is implemented. Document the intended data requirements in the model's docstring.

---

## Configuration Schema

Each model should have a YAML config file in `config/models/{model_name}.yaml`:

```yaml
# config/models/xgboost.yaml
model_name: xgboost
family: boosting

# Hyperparameters
learning_rate: 0.1
max_depth: 6
num_leaves: 31
min_child_samples: 20
subsample: 0.8
colsample_bytree: 0.8
reg_alpha: 0.1
reg_lambda: 1.0

# Training
num_boost_round: 1000
early_stopping_rounds: 50
verbose_eval: false

# MTF Strategy
mtf_strategy: mtf_indicators
training_timeframe: 15min
mtf_source_timeframes:
  - 1min
  - 5min
  - 30min
  - 1h

# Device (for neural models)
# device: cuda  # or cpu, auto
```

Load config in training script:

```python
# scripts/train_model.py
from src.models.config import load_model_config

config = load_model_config(model_name='xgboost')
model = XGBoostModel(config=config)
```

---

## Testing Requirements

Every model must have:

1. **Unit tests** - Test fit/predict/save/load in isolation
2. **Integration tests** - Test with real Phase 1 datasets
3. **Smoke tests** - Quick end-to-end training

### Unit Tests

```python
# tests/phase_2_tests/test_my_model.py
import pytest
import numpy as np
from src.models.boosting.my_model import MyModel

def test_my_model_fit():
    """Test model training."""
    X_train = np.random.randn(1000, 50)
    y_train = np.random.randint(0, 3, 1000)
    X_val = np.random.randn(200, 50)
    y_val = np.random.randint(0, 3, 200)

    model = MyModel()
    metrics = model.fit(X_train, y_train, X_val, y_val)

    assert metrics.train_loss is not None
    assert metrics.val_loss is not None
    assert metrics.best_iteration > 0

def test_my_model_predict():
    """Test prediction output."""
    model = MyModel()
    X_train = np.random.randn(1000, 50)
    y_train = np.random.randint(0, 3, 1000)
    X_val = np.random.randn(200, 50)
    y_val = np.random.randint(0, 3, 200)
    model.fit(X_train, y_train, X_val, y_val)

    X_test = np.random.randn(100, 50)
    output = model.predict(X_test)

    assert output.predictions.shape == (100,)
    assert output.probabilities.shape == (100, 3)
    assert output.confidence.shape == (100,)
    assert np.all((output.predictions >= 0) & (output.predictions < 3))
    assert np.allclose(output.probabilities.sum(axis=1), 1.0)

def test_my_model_save_load(tmp_path):
    """Test persistence."""
    model = MyModel()
    X_train = np.random.randn(1000, 50)
    y_train = np.random.randint(0, 3, 1000)
    X_val = np.random.randn(200, 50)
    y_val = np.random.randint(0, 3, 200)
    model.fit(X_train, y_train, X_val, y_val)

    # Save
    save_path = tmp_path / "my_model"
    model.save(save_path)
    assert (save_path / "model.pkl").exists()

    # Load
    loaded_model = MyModel.load(save_path)

    # Verify predictions match
    X_test = np.random.randn(100, 50)
    pred1 = model.predict(X_test)
    pred2 = loaded_model.predict(X_test)

    np.testing.assert_array_equal(pred1.predictions, pred2.predictions)

def test_my_model_input_validation():
    """Test input shape validation."""
    model = MyModel()
    X_wrong_shape = np.random.randn(100, 50, 10)  # 3D instead of 2D
    y = np.random.randint(0, 3, 100)

    with pytest.raises(ValueError, match="expects 2D input"):
        model.fit(X_wrong_shape, y, X_wrong_shape, y)
```

### Integration Tests

```python
# tests/integration/test_my_model_integration.py
import pytest
from pathlib import Path
from src.phase1.stages.datasets.container import TimeSeriesDataContainer
from src.models.boosting.my_model import MyModel

def test_my_model_with_real_data():
    """Test with real Phase 1 datasets."""
    # Load container (assumes Phase 1 has run)
    container_path = Path("data/splits/scaled/test_run_20250101_120000")
    if not container_path.exists():
        pytest.skip("Phase 1 data not available")

    container = TimeSeriesDataContainer.load(container_path)

    # Get data
    X_train, y_train, weights_train = container.get_tabular_data('train')
    X_val, y_val, weights_val = container.get_tabular_data('val')

    # Train
    model = MyModel()
    metrics = model.fit(X_train, y_train, X_val, y_val, weights_train)

    # Validate
    assert metrics.val_loss < 1.5  # Reasonable performance

    # Predict
    X_test, y_test, weights_test = container.get_tabular_data('test')
    output = model.predict(X_test)

    # Check predictions
    assert output.predictions.shape == y_test.shape
    assert np.all((output.predictions >= 0) & (output.predictions < 3))
```

### Smoke Tests

```bash
# Quick end-to-end test
pytest tests/integration/test_my_model_integration.py -v

# Full test suite
pytest tests/phase_2_tests/test_my_model.py -v
```

---

## Common Pitfalls

### 1. Input Shape Mismatches

**Problem:** Feeding 3D data to tabular model or 2D data to sequence model

```python
# BAD: Feeding 3D data to XGBoost
X_train = container.get_sequence_data('train', seq_len=60)  # (n, 60, 25)
model = XGBoostModel()
model.fit(X_train, ...)  # ERROR: XGBoost expects 2D!

# GOOD: Use correct data retrieval
X_train = container.get_tabular_data('train')  # (n, 150)
model = XGBoostModel()
model.fit(X_train, ...)  # ✓
```

**Solution:** Always validate input shape in `fit()` and `predict()`:
```python
self._validate_input_shape(X_train, expected_ndim=2)
```

### 2. Forgetting Validation Set

**Problem:** Training without early stopping leads to overfitting

```python
# BAD: No validation set
model.fit(X_train, y_train)  # Trains to completion, overfits!

# GOOD: Use validation for early stopping
model.fit(X_train, y_train, X_val, y_val)  # Stops when val loss stops improving
```

### 3. Ignoring Sample Weights

**Problem:** Not using quality-weighted samples loses performance

```python
# BAD: Ignoring sample weights
model.fit(X_train, y_train, X_val, y_val)

# GOOD: Use sample weights
model.fit(X_train, y_train, X_val, y_val, sample_weights=weights_train)
```

**Note:** Tabular models strongly benefit from sample weights. Neural models may benefit less.

### 4. Leakage in Feature Engineering

**Problem:** Scaling using test set statistics

```python
# BAD: Fitting scaler on full dataset
scaler.fit(np.concatenate([X_train, X_val, X_test]))  # LEAKAGE!

# GOOD: Phase 1 fits scaler on train only
# You don't need to scale - Phase 1 already did it
```

### 5. Non-Deterministic Predictions

**Problem:** Same inputs producing different outputs

```python
# BAD: No random seed
model = RandomForestClassifier()  # Different results each run

# GOOD: Set seed
model = RandomForestClassifier(random_state=42)

# For PyTorch:
torch.manual_seed(42)
np.random.seed(42)
```

### 6. GPU Memory Overflow

**Problem:** Sequence models with large batch size run out of memory

```python
# BAD: Batch size too large
params = {'batch_size': 1024, 'seq_len': 120, 'hidden_size': 512}
# OOM on 8GB GPU!

# GOOD: Estimate memory first
from src.models.device import estimate_gpu_memory
mem_required = estimate_gpu_memory('lstm', seq_len=120, batch_size=512, hidden_size=512)
if mem_required > gpu_memory:
    batch_size = batch_size // 2  # Reduce batch size
```

### 7. Ensemble Input Shape Compatibility

**Problem:** Mixing tabular and sequence models in ensemble

```python
# BAD: Mixing 2D and 3D models
VotingEnsemble(base_models=['xgboost', 'lstm'])  # ERROR: Different input shapes!

# GOOD: Same input shape
VotingEnsemble(base_models=['xgboost', 'lightgbm', 'catboost'])  # All 2D ✓
VotingEnsemble(base_models=['lstm', 'gru', 'tcn'])  # All 3D ✓
```

**Solution:** Validate ensemble compatibility:
```python
from src.models.ensemble.validation import validate_ensemble_compatibility
validate_ensemble_compatibility(['xgboost', 'lightgbm'])  # ✓
validate_ensemble_compatibility(['xgboost', 'lstm'])  # Raises EnsembleCompatibilityError
```

---

## Examples by Model Family

### Boosting: CatBoost

```python
# src/models/boosting/catboost_model.py
from catboost import CatBoostClassifier, Pool
from src.models.base import BaseModel, TrainingMetrics, PredictionOutput
from src.models.registry import register

@register(name="catboost", family="boosting")
class CatBoostModel(BaseModel):
    def fit(self, X_train, y_train, X_val, y_val, sample_weights=None, config=None):
        self._validate_input_shape(X_train, expected_ndim=2)

        params = self._get_params(config)

        train_pool = Pool(X_train, y_train, weight=sample_weights)
        val_pool = Pool(X_val, y_val)

        self.model = CatBoostClassifier(
            iterations=params.get('iterations', 1000),
            learning_rate=params.get('learning_rate', 0.1),
            depth=params.get('depth', 6),
            l2_leaf_reg=params.get('l2_leaf_reg', 3.0),
            early_stopping_rounds=params.get('early_stopping_rounds', 50),
            verbose=False
        )

        self.model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        return TrainingMetrics(
            train_loss=self.model.best_score_['learn']['MultiClass'],
            val_loss=self.model.best_score_['validation']['MultiClass'],
            best_iteration=self.model.best_iteration_,
            training_time=time.time() - start_time
        )
```

### Neural: GRU

```python
# src/models/neural/gru_model.py
import torch
import torch.nn as nn
from src.models.neural.base_rnn import BaseRNNModel
from src.models.registry import register

@register(name="gru", family="neural")
class GRUModel(BaseRNNModel):
    def _build_model(self, n_features, params):
        """Build GRU architecture."""
        self.rnn = nn.GRU(
            input_size=n_features,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params.get('dropout', 0.2),
            batch_first=True
        ).to(self.device)

        self.classifier = nn.Sequential(
            nn.Linear(params['hidden_size'], params.get('fc_hidden', 64)),
            nn.ReLU(),
            nn.Dropout(params.get('dropout', 0.2)),
            nn.Linear(params.get('fc_hidden', 64), 3)
        ).to(self.device)

    def forward(self, X):
        # GRU forward
        rnn_out, _ = self.rnn(X)

        # Take last timestep
        last_hidden = rnn_out[:, -1, :]

        # Classifier
        logits = self.classifier(last_hidden)

        return logits
```

### Classical: Logistic Regression

```python
# src/models/classical/logistic.py
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from src.models.base import BaseModel, TrainingMetrics, PredictionOutput
from src.models.registry import register

@register(name="logistic", family="classical")
class LogisticModel(BaseModel):
    def fit(self, X_train, y_train, X_val, y_val, sample_weights=None, config=None):
        self._validate_input_shape(X_train, expected_ndim=2)

        params = self._get_params(config)

        self.model = LogisticRegression(
            C=params.get('C', 1.0),
            penalty=params.get('penalty', 'l2'),
            solver=params.get('solver', 'lbfgs'),
            max_iter=params.get('max_iter', 1000),
            random_state=params.get('random_state', 42),
            n_jobs=-1
        )

        self.model.fit(X_train, y_train, sample_weight=sample_weights)

        # Compute losses
        train_proba = self.model.predict_proba(X_train)
        val_proba = self.model.predict_proba(X_val)

        train_loss = log_loss(y_train, train_proba)
        val_loss = log_loss(y_val, val_proba)

        return TrainingMetrics(
            train_loss=train_loss,
            val_loss=val_loss,
            best_iteration=0,  # No iterations for logistic
            training_time=time.time() - start_time
        )

    def predict(self, X):
        self._validate_input_shape(X, expected_ndim=2)

        proba = self.model.predict_proba(X)
        predictions = np.argmax(proba, axis=1)
        confidence = np.max(proba, axis=1)

        return PredictionOutput(predictions, proba, confidence)
```

---

## Next Steps

After integrating a new model:

1. **Phase 3: Cross-Validation**
   ```bash
   python scripts/run_cv.py --models my_model --horizons 5,10,15,20 --n-splits 5
   ```

2. **Phase 3: Hyperparameter Tuning**
   ```bash
   python scripts/run_cv.py --models my_model --tune --n-trials 100
   ```

3. **Phase 4: Ensemble Integration**
   ```bash
   # Add to ensemble base models
   python scripts/train_model.py --model voting --base-models xgboost,my_model --horizon 20
   ```

4. **Documentation:**
   - Update `CLAUDE.md` model count
   - Update `ALIGNMENT_PLAN.md` with model-specific recommendations
   - Add model to MTF strategy table

---

## Summary

Adding a new model requires:

1. ✅ Choose model family
2. ✅ Implement `BaseModel` interface (fit, predict, save, load)
3. ✅ Register with `@register(name, family)` decorator
4. ✅ Add configuration YAML
5. ✅ Write unit tests + integration tests
6. ✅ Train and evaluate
7. ✅ Update documentation

**Estimated time:** 2-6 hours for simple models, 1-2 days for complex models

**Files touched:**
- `src/models/{family}/{model_name}.py` (new)
- `config/models/{model_name}.yaml` (new)
- `tests/phase_2_tests/test_{model_name}.py` (new)
- `CLAUDE.md` (update model count)
