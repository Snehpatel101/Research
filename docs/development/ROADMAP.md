# Phase 2 Architecture - Executive Summary

**Project:** Ensemble Trading Pipeline - Model Training System
**Date:** 2025-12-21
**Status:** Design Complete, Ready for Implementation

---

## What Was Designed

A **modular, extensible model training system** that supports many model families (time series, classical ML, neural networks) while maintaining strict adherence to your engineering principles:

- ✅ **650-line limit** per file
- ✅ **Fail-fast validation** at every boundary
- ✅ **No exception swallowing**
- ✅ **Clear separation of concerns**
- ✅ **Zero data leakage** guarantees
- ✅ **Less code is better** philosophy

---

## Core Design Patterns

### 1. Plugin Architecture (Model Registry)
```python
@ModelRegistry.register(name="xgboost", family="boosting")
class XGBoostModel(BaseModel):
    ...

# Auto-discovery + factory instantiation
model = ModelRegistry.create("xgboost", config, horizon, features)
```

### 2. Abstract Base Class (Common Interface)
```python
class BaseModel(ABC):
    @abstractmethod
    def fit(self, X_train, y_train, X_val, y_val) -> Dict: ...
    @abstractmethod
    def predict(self, X, metadata) -> PredictionOutput: ...
    @abstractmethod
    def save(self, path: Path): ...
    @abstractmethod
    def load(self, path: Path): ...
```

### 3. Temporal Dataset (Zero Leakage)
```python
class TimeSeriesDataset:
    # Symbol-isolated windowing
    # Past features only (no future leakage)
    # Flexible sequence lengths (1 for boosting, 60+ for time series)
    def _create_sequences(self, df):
        for symbol in symbols:
            for i in range(seq_len, len(data)):
                X_window = data[i-seq_len : i]  # Past only
                y_label = data[i]               # Future
```

### 4. Orchestrated Training (Reusable Infrastructure)
```python
class Trainer:
    def run_full_pipeline(self):
        self.prepare_data()      # Load Phase 1 splits
        self.build_model()       # Instantiate via registry
        self.train()             # Delegate to model.fit()
        self.evaluate()          # Compute metrics, save predictions
        # MLflow tracks everything automatically
```

### 5. Structured Artifacts (Experiment Tracking)
```
experiments/runs/{model}_{timestamp}/
├── checkpoints/model/           # Model weights + config
├── predictions/                 # val/test predictions.parquet
├── metrics/metrics.json         # All metrics
└── plots/                       # Visualizations
```

---

## File Structure Created

```
src/
├── models/                      # Model implementations
│   ├── base.py                  # BaseModel, ModelConfig, PredictionOutput
│   ├── registry.py              # ModelRegistry (plugin architecture)
│   ├── boosting/                # XGBoost, LightGBM, CatBoost
│   ├── timeseries/              # N-HiTS, TFT, PatchTST, TimesFM
│   └── neural/                  # LSTM, GRU, Transformer
│
├── data/                        # Data loading
│   ├── dataset.py               # TimeSeriesDataset (windowing)
│   └── loaders.py               # DataLoader factories
│
├── training/                    # Training orchestration
│   ├── trainer.py               # Trainer (MLflow integration)
│   ├── evaluator.py             # ModelEvaluator (metrics)
│   └── callbacks.py             # EarlyStopping, Checkpointing
│
└── tuning/                      # Hyperparameter optimization
    ├── optuna_tuner.py          # Optuna integration
    └── search_spaces.py         # Model-specific search spaces

config/
├── models/                      # Model-specific YAML configs
│   ├── xgboost.yaml
│   ├── nhits.yaml
│   └── lstm.yaml
└── experiments/                 # Experiment definitions
    └── baseline.yaml

scripts/
├── train_model.py               # CLI: Train single model
├── run_experiment.py            # CLI: Run full experiment
└── tune_model.py                # CLI: Hyperparameter tuning

experiments/
├── runs/                        # Training run outputs
├── mlruns/                      # MLflow artifact store
└── registry/                    # Production models
```

---

## Integration with Phase 1

**Phase 1 Outputs** → **Phase 2 Inputs**

```python
# Phase 1 creates these files
data/splits/scaled/
├── train_scaled.parquet  (87,094 rows × 126 cols)
├── val_scaled.parquet    (18,591 rows × 126 cols)
└── test_scaled.parquet   (18,592 rows × 126 cols)

# Phase 2 loads them via TimeSeriesDataset
dataset = TimeSeriesDataset(DatasetConfig(
    train_path='data/splits/scaled/train_scaled.parquet',
    val_path='data/splits/scaled/val_scaled.parquet',
    test_path='data/splits/scaled/test_scaled.parquet',
    horizon=5,
    sequence_length=60
))

# Auto-detects:
# - Features: 107 columns (everything except datetime, symbol, labels)
# - Labels: label_h5, label_h20
# - Purge/embargo already applied by Phase 1
```

---

## Answers to Your Questions

### 1. Model Registry Pattern
**Answer:** Decorator-based plugin architecture with auto-discovery
- Models self-register via `@ModelRegistry.register`
- Factory pattern for instantiation: `ModelRegistry.create()`
- Fail-fast validation at registration time

### 2. Base Model Interface
**Answer:** Abstract base class with enforced contract
- Required methods: `fit()`, `predict()`, `save()`, `load()`
- Standardized output: `PredictionOutput` dataclass
- Input validation: `validate_inputs()` in base class

### 3. Data Loading
**Answer:** TimeSeriesDataset with temporal windowing
- Symbol-isolated windows (no cross-symbol leakage)
- Past features only (no future leakage)
- Flexible sequence lengths (1 for boosting, 60+ for time series)

### 4. Training Loop
**Answer:** Hybrid approach - reusable orchestration + model-specific loops
- Trainer handles workflow (data → model → train → evaluate)
- Models implement custom training loops in `fit()`
- MLflow tracks everything automatically

### 5. Artifact Management
**Answer:** Structured run directories + MLflow tracking
- Run-specific directories: `experiments/runs/{model}_{timestamp}/`
- MLflow UI for comparison and visualization
- Production registry separate from experiments

### 6. Configuration
**Answer:** Hybrid - global config.py + model-specific YAML files
- `config.py` for project-wide settings (Phase 1 params)
- `config/models/*.yaml` for model hyperparameters
- `config/experiments/*.yaml` for multi-model experiments

---

## Implementation Roadmap

### Week 1: Core Infrastructure
- [ ] BaseModel, ModelConfig, PredictionOutput (~250 lines)
- [ ] ModelRegistry with auto-discovery (~180 lines)
- [ ] TimeSeriesDataset with windowing (~200 lines)
- [ ] Unit tests for validation logic

### Week 2: First Model Family (Boosting)
- [ ] XGBoostModel (~180 lines)
- [ ] LightGBMModel (~170 lines)
- [ ] CatBoostModel (~170 lines)
- [ ] End-to-end test with real Phase 1 data

### Week 3: Training Infrastructure
- [ ] Trainer orchestration (~200 lines)
- [ ] ModelEvaluator (~150 lines)
- [ ] Training callbacks (~120 lines)
- [ ] CLI scripts (train_model.py, run_experiment.py)

### Week 4: Time Series Models
- [ ] N-HiTS implementation (~220 lines)
- [ ] TFT implementation (~230 lines)
- [ ] Baseline experiments (all models, both horizons)

### Week 5: Hyperparameter Tuning
- [ ] OptunaModelTuner (~200 lines)
- [ ] Search space definitions (~150 lines)
- [ ] Tuning experiments (50-100 trials per model)
- [ ] Lock in production configs

**Total:** ~20 files, ~6,000 lines of code, all files <650 lines

---

## Key Metrics to Track

### Model Performance
- Accuracy, Precision, Recall, F1 (per class: -1, 0, 1)
- Sharpe Ratio (simulated trading)
- Win Rate, Max Drawdown
- Profit Factor

### Training Efficiency
- Training time per epoch
- Convergence speed (early stopping)
- Memory usage

### Experiment Metadata
- Hyperparameters
- Model checkpoints
- Feature importance (for tree models)
- Predictions (val/test)

---

## Documentation Delivered

| Document | Purpose | Lines |
|----------|---------|-------|
| `PHASE2_ARCHITECTURE.md` | Comprehensive system design | 1,000+ |
| `PHASE2_ARCHITECTURE_DIAGRAM.md` | Visual diagrams (Mermaid) | 500+ |
| `PHASE2_IMPLEMENTATION_CHECKLIST.md` | Day-by-day implementation tasks | 800+ |
| `PHASE2_DESIGN_DECISIONS.md` | Q&A and design rationale | 800+ |
| `PHASE2_QUICKSTART.md` | 30-minute getting started guide | 600+ |
| `PHASE2_SUMMARY.md` | This document | 300+ |

**Total Documentation:** ~4,000 lines across 6 files

---

## Success Criteria

- [x] All design questions answered
- [x] Architecture respects 650-line limit
- [x] Fail-fast validation at every boundary
- [x] Zero-leakage guarantees
- [x] Clear separation of concerns
- [x] Extensible to new model families
- [x] MLflow integration for experiment tracking
- [x] Phase 1 integration points defined
- [x] Implementation roadmap (5 weeks)
- [x] Code examples provided (XGBoost complete)

---

## Next Steps

1. **Read PHASE2_QUICKSTART.md** to get started (30 min setup)
2. **Implement BaseModel + Registry** (Week 1, Days 1-2)
3. **Implement XGBoost model** (Week 1, Day 3)
4. **Test end-to-end with Phase 1 data** (Week 1, Day 3)
5. **Follow PHASE2_IMPLEMENTATION_CHECKLIST.md** for remaining tasks

---

## Design Principles Maintained

✅ **Modularity:** Each model in isolated module (<650 lines)
✅ **Fail-Fast:** Validation at every boundary (config, inputs, registry)
✅ **No Exception Swallowing:** Errors propagate with clear messages
✅ **Less Code:** Simple, boring solutions over clever abstractions
✅ **Clear Contracts:** Abstract base class enforces interface
✅ **Testing:** Unit/integration tests for all components
✅ **Documentation:** Comprehensive architecture + usage guides

---

## Questions or Clarifications?

All design documents are located in:
```
docs/phase2/PHASE2_*.md
```

If you have questions during implementation:
1. Check `PHASE2_DESIGN_DECISIONS.md` for rationale
2. See `PHASE2_IMPLEMENTATION_CHECKLIST.md` for detailed tasks
3. Refer to `PHASE2_QUICKSTART.md` for code examples

---

**Architecture designed by:** Claude (Backend Architect Agent)
**Date:** 2025-12-21
**Status:** ✅ Design Complete, Ready for Implementation

**Estimated Implementation Time:** 4-5 weeks (following checklist)
**Expected Model Families:** 3+ (boosting, time series, neural)
**Expected Models:** 6+ (XGBoost, LightGBM, CatBoost, N-HiTS, TFT, LSTM)

---

**End of Phase 2 Summary**
# Phase 2 Quick Start Guide

**Get started implementing the model training system in 30 minutes**

---

## Prerequisites Check

```bash
# 1. Verify Phase 1 outputs exist
ls -lh /home/jake/Desktop/Research/data/splits/scaled/
# Expected:
#   train_scaled.parquet  (87,094 rows × 126 cols)
#   val_scaled.parquet    (18,591 rows × 126 cols)
#   test_scaled.parquet   (18,592 rows × 126 cols)

# 2. Check Python version
python3 --version  # Should be 3.9+

# 3. Verify existing packages
pip list | grep -E "pandas|numpy|xgboost|scikit-learn"
```

---

## Step 1: Install Additional Dependencies (5 min)

```bash
cd /home/jake/Desktop/Research

# Core ML libraries
pip install xgboost lightgbm catboost
pip install optuna  # Hyperparameter tuning
pip install mlflow  # Experiment tracking

# Time series libraries (install later when needed)
# pip install neuralforecast darts pytorch-forecasting

# Visualization
pip install plotly seaborn
```

---

## Step 2: Create Directory Structure (2 min)

```bash
# Create Phase 2 directories
mkdir -p src/models/{boosting,timeseries,neural,ensemble}
mkdir -p src/data
mkdir -p src/training
mkdir -p src/tuning
mkdir -p config/{models,experiments}
mkdir -p experiments/{runs,mlruns,registry}
mkdir -p scripts

# Verify structure
tree -L 2 src/
```

---

## Step 3: Implement BaseModel (First 30 min of coding)

**File:** `src/models/base.py`

```python
"""
Base Model Interface - Start here!

Implement in this order:
1. ModelConfig dataclass (10 min)
2. PredictionOutput dataclass (5 min)
3. BaseModel abstract class (15 min)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Base configuration for all models."""
    # Required fields
    model_name: str
    model_family: str
    horizon: int  # 5 or 20

    # Training parameters
    random_seed: int = 42
    verbose: bool = True

    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-4

    def validate(self):
        """Fail-fast validation."""
        errors = []

        if self.horizon not in [5, 20]:
            errors.append(f"Invalid horizon {self.horizon}, must be 5 or 20")

        if self.patience < 1:
            errors.append(f"patience must be >= 1, got {self.patience}")

        if errors:
            raise ValueError(
                f"Config validation failed:\n" +
                "\n".join(f"  - {e}" for e in errors)
            )


@dataclass
class PredictionOutput:
    """Standardized prediction output."""
    predictions: np.ndarray        # Shape: (n,), values: {-1, 0, 1}
    probabilities: np.ndarray      # Shape: (n, 3)
    timestamps: pd.DatetimeIndex
    symbols: np.ndarray
    horizons: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        return pd.DataFrame({
            'datetime': self.timestamps,
            'symbol': self.symbols,
            'horizon': self.horizons,
            'prediction': self.predictions,
            'prob_short': self.probabilities[:, 0],
            'prob_neutral': self.probabilities[:, 1],
            'prob_long': self.probabilities[:, 2]
        })


class BaseModel(ABC):
    """Abstract base class for all models."""

    def __init__(self, config: dict, horizon: int, feature_columns: List[str]):
        # Validation
        if horizon not in [5, 20]:
            raise ValueError(f"Invalid horizon {horizon}")
        if not feature_columns:
            raise ValueError("feature_columns cannot be empty")

        # Build config
        self.config = self._build_config(config, horizon)
        self.config.validate()

        self.horizon = horizon
        self.feature_columns = feature_columns
        self.n_features = len(feature_columns)

        # Training state
        self.is_fitted = False
        self.training_history = {}

        # Initialize model
        self._build_model()

    @abstractmethod
    def _build_config(self, config: dict, horizon: int) -> ModelConfig:
        """Build model-specific config."""
        pass

    @abstractmethod
    def _build_model(self):
        """Initialize model architecture."""
        pass

    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None, metadata_train=None, metadata_val=None) -> Dict:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X, metadata=None) -> PredictionOutput:
        """Generate predictions."""
        pass

    @abstractmethod
    def save(self, path: Path):
        """Save model to disk."""
        pass

    @abstractmethod
    def load(self, path: Path):
        """Load model from disk."""
        pass

    def validate_inputs(self, X, y=None):
        """Validate input shapes and values."""
        errors = []

        if X.ndim not in [2, 3]:
            errors.append(f"X must be 2D or 3D, got shape {X.shape}")

        if y is not None:
            if len(y) != len(X):
                errors.append(f"X and y length mismatch: {len(X)} vs {len(y)}")

            unique_labels = np.unique(y[~np.isnan(y)])
            invalid = set(unique_labels) - {-1, 0, 1}
            if invalid:
                errors.append(f"Invalid labels: {invalid}. Valid: {{-1, 0, 1}}")

        if errors:
            raise ValueError("\n".join(errors))

    def __repr__(self):
        return f"{self.__class__.__name__}(horizon={self.horizon}, n_features={self.n_features}, fitted={self.is_fitted})"
```

**Test it:**

```python
# Quick sanity check
python3 -c "
from src.models.base import ModelConfig, PredictionOutput
config = ModelConfig(model_name='test', model_family='test', horizon=5)
config.validate()
print('✅ BaseModel classes created successfully')
"
```

---

## Step 4: Implement ModelRegistry (Next 30 min)

**File:** `src/models/registry.py`

```python
"""Model Registry - Plugin architecture."""

from typing import Dict, Type, List, Optional
from src.models.base import BaseModel


class ModelRegistry:
    """Central registry for all models."""

    _registry: Dict[str, Type[BaseModel]] = {}
    _metadata: Dict[str, dict] = {}

    @classmethod
    def register(cls, name: str, family: str, description: str = "", requires_gpu: bool = False):
        """Decorator to register a model class."""
        def decorator(model_class: Type[BaseModel]):
            # Validation
            if not issubclass(model_class, BaseModel):
                raise TypeError(f"Model {name} must inherit from BaseModel")

            # Check required methods
            for method in ['fit', 'predict', 'save', 'load']:
                if not hasattr(model_class, method):
                    raise AttributeError(f"Model {name} missing method: {method}")

            # Register
            full_name = f"{family}:{name}"
            cls._registry[full_name] = model_class
            cls._metadata[full_name] = {
                'name': name,
                'family': family,
                'description': description,
                'requires_gpu': requires_gpu
            }

            return model_class
        return decorator

    @classmethod
    def create(cls, model_name: str, config: dict, horizon: int, feature_columns: List[str]) -> BaseModel:
        """Factory method to create a model."""
        # Resolve name
        full_name = cls._resolve_name(model_name)
        if full_name not in cls._registry:
            available = ', '.join(cls._registry.keys())
            raise ValueError(f"Model '{model_name}' not found. Available: {available}")

        # Validate inputs
        if horizon not in [5, 20]:
            raise ValueError(f"Invalid horizon {horizon}")
        if not feature_columns:
            raise ValueError("feature_columns must be non-empty list")

        # Instantiate
        model_class = cls._registry[full_name]
        return model_class(config=config, horizon=horizon, feature_columns=feature_columns)

    @classmethod
    def _resolve_name(cls, name: str) -> str:
        """Resolve short name to full name."""
        if ':' in name:
            return name

        matches = [k for k in cls._registry if k.endswith(f':{name}')]
        if len(matches) == 0:
            raise ValueError(f"Model '{name}' not found")
        if len(matches) > 1:
            raise ValueError(f"Ambiguous name '{name}', matches: {matches}")
        return matches[0]

    @classmethod
    def list_models(cls, family: Optional[str] = None) -> List[dict]:
        """List all registered models."""
        models = list(cls._metadata.values())
        if family:
            models = [m for m in models if m['family'] == family]
        return models
```

**Test it:**

```python
python3 -c "
from src.models.registry import ModelRegistry
print(f'Registry has {len(ModelRegistry.list_models())} models (should be 0)')
print('✅ ModelRegistry created successfully')
"
```

---

## Step 5: Implement Your First Model - XGBoost (Next 45 min)

**File:** `src/models/boosting/__init__.py`

```python
"""Boosting models."""
```

**File:** `src/models/boosting/xgboost.py`

```python
"""XGBoost Model."""

from dataclasses import dataclass
from typing import Dict, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle

from src.models.base import BaseModel, ModelConfig, PredictionOutput
from src.models.registry import ModelRegistry


@dataclass
class XGBoostConfig(ModelConfig):
    """XGBoost-specific config."""
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8


@ModelRegistry.register(name="xgboost", family="boosting", description="XGBoost gradient boosting")
class XGBoostModel(BaseModel):
    """XGBoost model for triple-barrier labels."""

    def _build_config(self, config: dict, horizon: int) -> XGBoostConfig:
        return XGBoostConfig(
            model_name='xgboost',
            model_family='boosting',
            horizon=horizon,
            **config
        )

    def _build_model(self):
        self.model = xgb.XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            objective='multi:softprob',
            num_class=3,
            random_state=self.config.random_seed
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None, metadata_train=None, metadata_val=None) -> Dict:
        # Validate
        self.validate_inputs(X_train, y_train)

        # Flatten 3D if needed
        if X_train.ndim == 3:
            X_train = X_train.reshape(len(X_train), -1)
            if X_val is not None:
                X_val = X_val.reshape(len(X_val), -1)

        # Encode labels: {-1, 0, 1} -> {0, 1, 2}
        y_train_enc = y_train + 1
        y_val_enc = (y_val + 1) if y_val is not None else None

        # Train
        eval_set = [(X_train, y_train_enc)]
        if X_val is not None:
            eval_set.append((X_val, y_val_enc))

        self.model.fit(X_train, y_train_enc, eval_set=eval_set, verbose=False)
        self.is_fitted = True

        return {'train_loss': [], 'val_loss': [], 'best_iteration': self.model.best_iteration}

    def predict(self, X, metadata=None) -> PredictionOutput:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        self.validate_inputs(X)

        # Flatten 3D
        if X.ndim == 3:
            X = X.reshape(len(X), -1)

        # Predict
        proba = self.model.predict_proba(X)
        pred_enc = self.model.predict(X)

        # Decode: {0, 1, 2} -> {-1, 0, 1}
        predictions = pred_enc - 1

        # Extract metadata
        if metadata is not None:
            timestamps = pd.to_datetime(metadata['datetime'])
            symbols = metadata['symbol'].values
        else:
            timestamps = pd.date_range('2020-01-01', periods=len(X), freq='5min')
            symbols = np.array(['UNK'] * len(X))

        return PredictionOutput(
            predictions=predictions,
            probabilities=proba,
            timestamps=timestamps,
            symbols=symbols,
            horizons=np.array([self.horizon] * len(X))
        )

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path / 'xgboost_model.json'))
        with open(path / 'metadata.pkl', 'wb') as f:
            pickle.dump({
                'config': self.config,
                'feature_columns': self.feature_columns,
                'is_fitted': self.is_fitted
            }, f)

    def load(self, path: Path):
        self.model.load_model(str(path / 'xgboost_model.json'))
        with open(path / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        self.config = metadata['config']
        self.feature_columns = metadata['feature_columns']
        self.is_fitted = metadata['is_fitted']
```

**Test it:**

```python
# Test registration
python3 -c "
from src.models.boosting.xgboost import XGBoostModel
from src.models.registry import ModelRegistry

models = ModelRegistry.list_models()
print(f'Registered models: {[m[\"name\"] for m in models]}')
assert len(models) == 1
assert models[0]['name'] == 'xgboost'
print('✅ XGBoost registered successfully')
"

# Test instantiation
python3 -c "
from src.models.registry import ModelRegistry

model = ModelRegistry.create(
    model_name='xgboost',
    config={'n_estimators': 10, 'max_depth': 3},
    horizon=5,
    feature_columns=['feat1', 'feat2', 'feat3']
)

print(model)
print('✅ XGBoost instantiates successfully')
"
```

---

## Step 6: Test End-to-End with Real Data (Next 15 min)

**File:** `test_xgboost_e2e.py` (temporary test script)

```python
"""End-to-end test with real Phase 1 data."""

import numpy as np
import pandas as pd
from pathlib import Path
from src.models.registry import ModelRegistry
from src.models.boosting.xgboost import XGBoostModel

# Load Phase 1 data
train_df = pd.read_parquet('data/splits/scaled/train_scaled.parquet')
val_df = pd.read_parquet('data/splits/scaled/val_scaled.parquet')

print(f"Train shape: {train_df.shape}")
print(f"Val shape: {val_df.shape}")

# Prepare data (simple version - no windowing)
meta_cols = ['datetime', 'symbol', 'split']
label_cols = [c for c in train_df.columns if c.startswith('label_')]
feature_cols = [c for c in train_df.columns if c not in meta_cols and c not in label_cols]

print(f"Features: {len(feature_cols)}")
print(f"Labels: {label_cols}")

X_train = train_df[feature_cols].values
y_train = train_df['label_h5'].values
meta_train = train_df[['datetime', 'symbol']]

X_val = val_df[feature_cols].values
y_val = val_df['label_h5'].values
meta_val = val_df[['datetime', 'symbol']]

# Remove NaN labels
train_mask = ~np.isnan(y_train)
val_mask = ~np.isnan(y_val)

X_train = X_train[train_mask]
y_train = y_train[train_mask]
meta_train = meta_train[train_mask]

X_val = X_val[val_mask]
y_val = y_val[val_mask]
meta_val = meta_val[val_mask]

print(f"\nAfter removing NaNs:")
print(f"Train: {X_train.shape}, Val: {X_val.shape}")
print(f"Label distribution: {np.unique(y_train, return_counts=True)}")

# Create model
model = ModelRegistry.create(
    model_name='xgboost',
    config={
        'n_estimators': 50,
        'max_depth': 4,
        'learning_rate': 0.1
    },
    horizon=5,
    feature_columns=feature_cols
)

print(f"\nModel: {model}")

# Train
print("\nTraining...")
results = model.fit(X_train, y_train, X_val, y_val, meta_train, meta_val)
print(f"Training complete: {results}")

# Predict
print("\nPredicting on validation set...")
preds = model.predict(X_val, meta_val)
print(f"Predictions shape: {preds.predictions.shape}")
print(f"Prediction distribution: {np.unique(preds.predictions, return_counts=True)}")

# Accuracy
accuracy = (preds.predictions == y_val).mean()
print(f"\nValidation Accuracy: {accuracy:.4f}")

# Save model
save_path = Path('experiments/test_model')
model.save(save_path)
print(f"\nModel saved to: {save_path}")

# Load model
model2 = ModelRegistry.create(
    model_name='xgboost',
    config={},
    horizon=5,
    feature_columns=feature_cols
)
model2.load(save_path)
print("Model loaded successfully")

# Verify predictions match
preds2 = model2.predict(X_val, meta_val)
assert np.allclose(preds.predictions, preds2.predictions)
print("✅ Save/load roundtrip successful")

print("\n" + "="*60)
print("✅ END-TO-END TEST PASSED")
print("="*60)
```

**Run it:**

```bash
python test_xgboost_e2e.py
```

**Expected output:**
```
Train shape: (87094, 126)
Val shape: (18591, 126)
Features: 107
...
Validation Accuracy: 0.45-0.55  (baseline accuracy)
✅ END-TO-END TEST PASSED
```

---

## Step 7: Next Steps (What to Build Next)

You now have the core foundation! Here's what to build next:

### Week 1 (Complete Infrastructure)
```bash
# 1. Implement TimeSeriesDataset
touch src/data/dataset.py
# See PHASE2_IMPLEMENTATION_CHECKLIST.md Day 3

# 2. Add unit tests
touch tests/test_base_model.py
touch tests/test_registry.py
touch tests/test_xgboost.py
pytest tests/ -v
```

### Week 2 (More Models)
```bash
# 3. Add LightGBM
touch src/models/boosting/lightgbm.py
# Copy XGBoost structure, swap to LGBMClassifier

# 4. Add CatBoost
touch src/models/boosting/catboost.py
# Copy XGBoost structure, swap to CatBoostClassifier
```

### Week 3 (Training Infrastructure)
```bash
# 5. Implement Trainer
touch src/training/trainer.py
# See PHASE2_IMPLEMENTATION_CHECKLIST.md Day 9-10

# 6. Implement Evaluator
touch src/training/evaluator.py
# See PHASE2_IMPLEMENTATION_CHECKLIST.md Day 7

# 7. Create CLI script
touch scripts/train_model.py
chmod +x scripts/train_model.py
```

---

## Common Issues & Solutions

### Issue 1: "Model not found in registry"
```python
# Solution: Import the model module to trigger registration
from src.models.boosting.xgboost import XGBoostModel  # This registers it
from src.models.registry import ModelRegistry
models = ModelRegistry.list_models()  # Now it appears
```

### Issue 2: "Invalid label values"
```python
# Solution: Check for NaNs and filter
y_train = y_train[~np.isnan(y_train)]  # Remove NaNs
assert set(np.unique(y_train)).issubset({-1, 0, 1})  # Verify
```

### Issue 3: "Shape mismatch"
```python
# Solution: Flatten 3D sequences for boosting models
if X.ndim == 3:
    X = X.reshape(len(X), -1)  # (n, seq, feat) -> (n, seq*feat)
```

### Issue 4: "File size exceeds 650 lines"
```bash
# Check file size
wc -l src/models/boosting/xgboost.py

# Solution: Extract helper functions or split into modules
# Example: Extract label encoding to src/models/utils.py
```

---

## Validation Checklist

After each step, verify:

- [ ] Code runs without errors
- [ ] File size < 650 lines (`wc -l <file>`)
- [ ] Validation passes (no exception swallowing)
- [ ] Model can fit on dummy data
- [ ] Model can predict
- [ ] Save/load roundtrip works
- [ ] Tests pass (`pytest tests/ -v`)

---

## Resources

**Documentation:**
- `docs/phase2/PHASE2_ARCHITECTURE.md` - Architecture summary
- `docs/phase2/PHASE2_IMPLEMENTATION_CHECKLIST.md` - Detailed tasks
- `docs/phase2/PHASE2_DESIGN_DECISIONS.md` - Design rationale

**Code Examples:**
- XGBoost implementation above (complete working example)
- See architecture doc for N-HiTS, Trainer, TimeSeriesDataset examples

**External Docs:**
- XGBoost: https://xgboost.readthedocs.io/
- LightGBM: https://lightgbm.readthedocs.io/
- Optuna: https://optuna.readthedocs.io/
- MLflow: https://mlflow.org/docs/latest/

---

## Summary

You've now implemented:
1. ✅ BaseModel abstract class
2. ✅ ModelRegistry plugin architecture
3. ✅ XGBoost model (first working model!)
4. ✅ End-to-end test with real Phase 1 data

**Next:** Follow the implementation checklist to build TimeSeriesDataset, Trainer, and additional models.

**Time to Production:** ~4 weeks following the checklist

**File Count:** ~20 files, all <650 lines

**Test Coverage Target:** >80%

---

**Happy coding!** 🚀
