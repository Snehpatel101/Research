# Phase 2 Architecture: Model Training & Evaluation System

**Date:** 2025-12-21
**Status:** Design Document
**Author:** Claude (Backend Architect Agent)

---

## Executive Summary

Phase 2 builds a **modular, extensible model training system** that supports many model families (time series, classical ML, neural nets) while maintaining strict adherence to the 650-line limit, fail-fast validation, and clear separation of concerns.

**Key Design Principles:**
1. **Plugin Architecture**: Models as self-contained plugins with registry pattern
2. **Common Interface**: Abstract base class enforces consistent model contracts
3. **Isolated Training**: Each model family in separate module (<650 lines)
4. **Experiment Tracking**: MLflow integration for all hyperparameters/metrics/artifacts
5. **Zero Leakage**: TimeSeriesDataset with strict temporal ordering
6. **Fail-Fast**: Comprehensive validation at every boundary

---

## 1. Proposed Directory Structure

```
/home/jake/Desktop/Research/
│
├── src/
│   ├── config.py                          # Existing (extended with model configs)
│   │
│   ├── models/                            # NEW: Model implementations
│   │   ├── __init__.py
│   │   ├── base.py                        # Abstract base class + interfaces
│   │   ├── registry.py                    # Model factory/registry
│   │   │
│   │   ├── timeseries/                    # Time series models
│   │   │   ├── __init__.py
│   │   │   ├── nhits.py                   # N-HiTS (<650 lines)
│   │   │   ├── tft.py                     # Temporal Fusion Transformer
│   │   │   ├── patchtst.py                # PatchTST
│   │   │   └── timesfm.py                 # TimesFM (Google's foundation model)
│   │   │
│   │   ├── boosting/                      # Gradient boosting models
│   │   │   ├── __init__.py
│   │   │   ├── xgboost.py                 # XGBoost (<650 lines)
│   │   │   ├── lightgbm.py                # LightGBM
│   │   │   └── catboost.py                # CatBoost
│   │   │
│   │   ├── neural/                        # Neural network models
│   │   │   ├── __init__.py
│   │   │   ├── lstm.py                    # LSTM variants
│   │   │   ├── gru.py                     # GRU variants
│   │   │   └── transformer.py             # Transformer variants
│   │   │
│   │   └── ensemble/                      # Ensemble methods (Phase 3)
│   │       ├── __init__.py
│   │       ├── stacking.py
│   │       └── voting.py
│   │
│   ├── data/                              # NEW: Data loading infrastructure
│   │   ├── __init__.py
│   │   ├── dataset.py                     # TimeSeriesDataset class
│   │   ├── loaders.py                     # DataLoader factories
│   │   └── transforms.py                  # Data transformations
│   │
│   ├── training/                          # NEW: Training orchestration
│   │   ├── __init__.py
│   │   ├── trainer.py                     # Base trainer class
│   │   ├── experiment.py                  # Experiment manager (MLflow)
│   │   ├── callbacks.py                   # Training callbacks
│   │   └── evaluator.py                   # Model evaluation
│   │
│   ├── tuning/                            # NEW: Hyperparameter optimization
│   │   ├── __init__.py
│   │   ├── optuna_tuner.py                # Optuna integration
│   │   └── search_spaces.py               # Model-specific search spaces
│   │
│   └── pipeline/                          # Existing (Phase 1)
│       ├── runner.py
│       └── stage_registry.py
│
├── config/                                # Configuration files
│   ├── models/                            # NEW: Model-specific configs
│   │   ├── nhits.yaml
│   │   ├── xgboost.yaml
│   │   ├── lstm.yaml
│   │   └── default.yaml                   # Default training params
│   │
│   └── experiments/                       # NEW: Experiment configs
│       ├── baseline.yaml
│       └── production.yaml
│
├── experiments/                           # NEW: MLflow tracking
│   ├── mlruns/                            # MLflow artifact store
│   ├── runs/                              # Run-specific outputs
│   │   └── {run_id}/
│   │       ├── checkpoints/               # Model checkpoints
│   │       ├── predictions/               # Out-of-sample predictions
│   │       ├── metrics/                   # Detailed metrics
│   │       └── artifacts/                 # Plots, reports, etc.
│   │
│   └── registry/                          # Model registry (production models)
│       ├── models/                        # Registered model versions
│       └── metadata.json                  # Model lineage
│
├── data/                                  # Existing (Phase 1 outputs)
│   └── splits/
│       ├── scaled/
│       │   ├── train_scaled.parquet       # Phase 1 output
│       │   ├── val_scaled.parquet         # Phase 1 output
│       │   └── test_scaled.parquet        # Phase 1 output
│       │
│       └── sequences/                     # NEW: Prepared sequences
│           ├── train_sequences.pkl        # Windowed sequences for TS models
│           ├── val_sequences.pkl
│           └── test_sequences.pkl
│
└── scripts/                               # CLI entry points
    ├── train_model.py                     # NEW: Train single model
    ├── run_experiment.py                  # NEW: Run full experiment
    └── evaluate_model.py                  # NEW: Evaluate saved model
```

---

## 2. Model Registry & Factory Pattern

### 2.1 Registry Design (`src/models/registry.py`)

**Purpose**: Central registry for all model families with plugin architecture.

```python
"""
Model Registry - Plugin Architecture for Dynamic Model Loading

Responsibilities:
1. Register model classes by family and name
2. Instantiate models with validated configs
3. List available models and their metadata
4. Fail-fast validation of model implementations
"""
from typing import Dict, Type, Optional, List
from pathlib import Path
import importlib
import inspect
from src.models.base import BaseModel

class ModelRegistry:
    """
    Central registry for all model implementations.

    Models self-register via @ModelRegistry.register decorator.
    Factory pattern for instantiation with config validation.
    """

    _registry: Dict[str, Type[BaseModel]] = {}
    _metadata: Dict[str, dict] = {}

    @classmethod
    def register(
        cls,
        name: str,
        family: str,
        description: str = "",
        requires_gpu: bool = False,
        supports_multivariate: bool = True,
        supports_horizon_multi: bool = False
    ):
        """
        Decorator to register a model class.

        Usage:
            @ModelRegistry.register(
                name="nhits",
                family="timeseries",
                description="N-HiTS neural hierarchical interpolation",
                requires_gpu=True
            )
            class NHiTSModel(BaseModel):
                ...
        """
        def decorator(model_class: Type[BaseModel]):
            # Validation: must inherit from BaseModel
            if not issubclass(model_class, BaseModel):
                raise TypeError(
                    f"Model {name} must inherit from BaseModel, "
                    f"got {model_class.__bases__}"
                )

            # Validation: must implement required methods
            required_methods = ['fit', 'predict', 'save', 'load']
            for method in required_methods:
                if not hasattr(model_class, method):
                    raise AttributeError(
                        f"Model {name} missing required method: {method}"
                    )

            # Register
            full_name = f"{family}:{name}"
            cls._registry[full_name] = model_class
            cls._metadata[full_name] = {
                'name': name,
                'family': family,
                'description': description,
                'requires_gpu': requires_gpu,
                'supports_multivariate': supports_multivariate,
                'supports_horizon_multi': supports_horizon_multi,
                'class': model_class.__name__,
                'module': model_class.__module__
            }

            return model_class
        return decorator

    @classmethod
    def create(
        cls,
        model_name: str,
        config: dict,
        horizon: int,
        feature_columns: List[str]
    ) -> BaseModel:
        """
        Factory method to instantiate a model.

        Parameters:
        -----------
        model_name : str
            Full name (family:name) or short name (tries all families)
        config : dict
            Model-specific configuration
        horizon : int
            Prediction horizon (5 or 20)
        feature_columns : List[str]
            List of feature column names

        Returns:
        --------
        BaseModel instance

        Raises:
        -------
        ValueError: If model not found or config invalid
        """
        # Resolve full name
        full_name = cls._resolve_name(model_name)
        if full_name not in cls._registry:
            available = ', '.join(cls._registry.keys())
            raise ValueError(
                f"Model '{model_name}' not found. Available: {available}"
            )

        # Validate config
        if not isinstance(config, dict):
            raise TypeError(f"Config must be dict, got {type(config)}")

        # Validate horizon
        if horizon not in [5, 20]:
            raise ValueError(f"Invalid horizon {horizon}, must be 5 or 20")

        # Validate feature columns
        if not feature_columns or not isinstance(feature_columns, list):
            raise ValueError("feature_columns must be non-empty list")

        # Instantiate
        model_class = cls._registry[full_name]
        try:
            model = model_class(
                config=config,
                horizon=horizon,
                feature_columns=feature_columns
            )
            return model
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate {full_name}: {e}"
            ) from e

    @classmethod
    def _resolve_name(cls, name: str) -> str:
        """Resolve short name to full name (family:name)."""
        if ':' in name:
            return name

        # Try to find in any family
        matches = [k for k in cls._registry if k.endswith(f':{name}')]
        if len(matches) == 0:
            raise ValueError(f"Model '{name}' not found")
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous name '{name}', matches: {matches}. "
                f"Use full name (family:name)"
            )
        return matches[0]

    @classmethod
    def list_models(cls, family: Optional[str] = None) -> List[dict]:
        """List all registered models with metadata."""
        models = list(cls._metadata.values())
        if family:
            models = [m for m in models if m['family'] == family]
        return models

    @classmethod
    def get_metadata(cls, model_name: str) -> dict:
        """Get metadata for a specific model."""
        full_name = cls._resolve_name(model_name)
        return cls._metadata.get(full_name, {})


# Auto-discovery: import all model modules to trigger registration
def auto_register_models():
    """
    Auto-discover and register all models in src/models/.

    Scans timeseries/, boosting/, neural/ subdirectories.
    """
    models_dir = Path(__file__).parent
    families = ['timeseries', 'boosting', 'neural']

    for family in families:
        family_dir = models_dir / family
        if not family_dir.exists():
            continue

        for module_path in family_dir.glob('*.py'):
            if module_path.name.startswith('_'):
                continue

            module_name = f"src.models.{family}.{module_path.stem}"
            try:
                importlib.import_module(module_name)
            except Exception as e:
                # Log but don't fail - allows partial model loading
                print(f"Warning: Failed to import {module_name}: {e}")
```

**Key Features:**
- **Decorator-based registration**: Models self-register with `@ModelRegistry.register`
- **Auto-discovery**: Scans `models/{family}/` directories on import
- **Validation**: Enforces BaseModel inheritance and required methods
- **Factory pattern**: `create()` instantiates with validated config
- **Metadata tracking**: GPU requirements, multivariate support, etc.

---

## 3. Base Model Interface (`src/models/base.py`)

### 3.1 Abstract Base Class

```python
"""
Base Model Interface - Contract for All Model Implementations

All models must inherit from BaseModel and implement:
- fit(): Train the model
- predict(): Generate predictions
- save(): Persist model to disk
- load(): Restore model from disk
- validate_config(): Validate model-specific configuration
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    """
    Base configuration for all models.

    Model-specific configs should subclass this and add fields.
    """
    # Model metadata
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

    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_dir: Optional[Path] = None

    # Additional config (model-specific)
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def validate(self):
        """Validate configuration values."""
        errors = []

        if self.horizon not in [5, 20]:
            errors.append(f"Invalid horizon {self.horizon}, must be 5 or 20")

        if self.patience < 1:
            errors.append(f"patience must be >= 1, got {self.patience}")

        if self.min_delta < 0:
            errors.append(f"min_delta must be >= 0, got {self.min_delta}")

        if errors:
            raise ValueError(
                f"Config validation failed:\n" +
                "\n".join(f"  - {e}" for e in errors)
            )


@dataclass
class PredictionOutput:
    """
    Standardized prediction output format.

    All models return this structure for consistency.
    """
    # Predictions
    predictions: np.ndarray  # Shape: (n_samples,) - class labels {-1, 0, 1}
    probabilities: np.ndarray  # Shape: (n_samples, 3) - class probs

    # Metadata
    timestamps: pd.DatetimeIndex
    symbols: np.ndarray
    horizons: np.ndarray  # All same value (5 or 20)

    # Optional: uncertainty/confidence
    uncertainty: Optional[np.ndarray] = None  # Model-specific

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for easy analysis."""
        return pd.DataFrame({
            'datetime': self.timestamps,
            'symbol': self.symbols,
            'horizon': self.horizons,
            'prediction': self.predictions,
            'prob_short': self.probabilities[:, 0],
            'prob_neutral': self.probabilities[:, 1],
            'prob_long': self.probabilities[:, 2],
            'uncertainty': self.uncertainty if self.uncertainty is not None else np.nan
        })


class BaseModel(ABC):
    """
    Abstract base class for all model implementations.

    Enforces consistent interface across time series, boosting, neural models.
    """

    def __init__(
        self,
        config: dict,
        horizon: int,
        feature_columns: List[str]
    ):
        """
        Initialize model.

        Parameters:
        -----------
        config : dict
            Model-specific configuration
        horizon : int
            Prediction horizon (5 or 20)
        feature_columns : List[str]
            List of feature column names
        """
        # Validate inputs
        if horizon not in [5, 20]:
            raise ValueError(f"Invalid horizon {horizon}, must be 5 or 20")

        if not feature_columns:
            raise ValueError("feature_columns cannot be empty")

        self.config = self._build_config(config, horizon)
        self.config.validate()

        self.horizon = horizon
        self.feature_columns = feature_columns
        self.n_features = len(feature_columns)

        # Training state
        self.is_fitted = False
        self.training_history: Dict[str, List[float]] = {}
        self.best_metric: Optional[float] = None

        # Model-specific initialization
        self._build_model()

    @abstractmethod
    def _build_config(self, config: dict, horizon: int) -> ModelConfig:
        """
        Build model-specific config from dict.

        Subclasses override to create their specific config dataclass.
        """
        pass

    @abstractmethod
    def _build_model(self):
        """
        Build the underlying model architecture.

        Called in __init__ after config validation.
        """
        pass

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        metadata_train: Optional[pd.DataFrame] = None,
        metadata_val: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Train the model.

        Parameters:
        -----------
        X_train : np.ndarray
            Training features, shape (n_samples, n_features) or (n_samples, seq_len, n_features)
        y_train : np.ndarray
            Training labels, shape (n_samples,), values in {-1, 0, 1}
        X_val : np.ndarray, optional
            Validation features (for early stopping)
        y_val : np.ndarray, optional
            Validation labels
        metadata_train : pd.DataFrame, optional
            Metadata (datetime, symbol, etc.) for train set
        metadata_val : pd.DataFrame, optional
            Metadata for validation set

        Returns:
        --------
        Dict with training metrics:
            {
                'train_loss': [...],
                'val_loss': [...],
                'best_epoch': int,
                'final_train_acc': float,
                'final_val_acc': float
            }
        """
        pass

    @abstractmethod
    def predict(
        self,
        X: np.ndarray,
        metadata: Optional[pd.DataFrame] = None
    ) -> PredictionOutput:
        """
        Generate predictions.

        Parameters:
        -----------
        X : np.ndarray
            Features, shape (n_samples, n_features) or (n_samples, seq_len, n_features)
        metadata : pd.DataFrame, optional
            Metadata (datetime, symbol) for output

        Returns:
        --------
        PredictionOutput with predictions, probabilities, timestamps
        """
        pass

    @abstractmethod
    def save(self, path: Path):
        """
        Save model to disk.

        Must save:
        - Model weights/parameters
        - Configuration
        - Training history
        - Feature columns (for validation)
        """
        pass

    @abstractmethod
    def load(self, path: Path):
        """
        Load model from disk.

        Must restore all state saved by save().
        Sets self.is_fitted = True if successful.
        """
        pass

    def validate_inputs(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ):
        """
        Validate input shapes and values.

        Called at start of fit() and predict().
        """
        errors = []

        # Check X shape
        if X.ndim not in [2, 3]:
            errors.append(
                f"X must be 2D or 3D, got shape {X.shape}"
            )

        if X.ndim == 2 and X.shape[1] != self.n_features:
            errors.append(
                f"X has {X.shape[1]} features, expected {self.n_features}"
            )

        if X.ndim == 3 and X.shape[2] != self.n_features:
            errors.append(
                f"X has {X.shape[2]} features, expected {self.n_features}"
            )

        # Check y if provided
        if y is not None:
            if y.ndim != 1:
                errors.append(f"y must be 1D, got shape {y.shape}")

            if len(y) != len(X):
                errors.append(
                    f"X and y length mismatch: {len(X)} vs {len(y)}"
                )

            # Check label values
            unique_labels = np.unique(y[~np.isnan(y)])
            valid_labels = {-1, 0, 1}
            invalid = set(unique_labels) - valid_labels
            if invalid:
                errors.append(
                    f"y contains invalid labels: {invalid}. "
                    f"Valid labels: {valid_labels}"
                )

        if errors:
            raise ValueError(
                f"Input validation failed:\n" +
                "\n".join(f"  - {e}" for e in errors)
            )

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance scores (if supported).

        Returns None for models without intrinsic importance.
        Subclasses override for tree-based models.
        """
        return None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"horizon={self.horizon}, "
            f"n_features={self.n_features}, "
            f"fitted={self.is_fitted})"
        )
```

**Key Features:**
- **Enforced contract**: All models implement fit/predict/save/load
- **Standardized outputs**: `PredictionOutput` dataclass
- **Input validation**: `validate_inputs()` catches shape/value errors early
- **Metadata propagation**: Timestamps/symbols flow through predictions
- **Training history**: Automatic tracking of metrics
- **Configuration validation**: Fail-fast on invalid configs

---

## 4. TimeSeriesDataset Design (`src/data/dataset.py`)

### 4.1 Zero-Leakage Temporal Dataset

```python
"""
TimeSeriesDataset - Temporal Data Loading with Leakage Prevention

Responsibilities:
1. Load train/val/test splits from Phase 1
2. Create windowed sequences for time series models
3. Enforce strict temporal ordering (no future data in past windows)
4. Provide DataLoader interface for batch iteration
"""
from typing import Optional, Tuple, List
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class DatasetConfig:
    """Configuration for TimeSeriesDataset."""
    # Paths
    train_path: Path
    val_path: Path
    test_path: Path

    # Sequence parameters
    sequence_length: int = 60  # Lookback window (60 bars = 5 hours)
    horizon: int = 5  # Prediction horizon (5 or 20)

    # Feature/label columns
    feature_columns: List[str] = None  # Auto-detected if None
    label_column: str = None  # e.g., 'label_h5'

    # Filtering
    include_symbols: Optional[List[str]] = None  # None = all symbols
    exclude_neutrals: bool = False  # Filter out label=0 samples

    # Memory optimization
    use_memmap: bool = False  # For very large datasets

    def validate(self):
        """Validate configuration."""
        errors = []

        if not self.train_path.exists():
            errors.append(f"train_path does not exist: {self.train_path}")
        if not self.val_path.exists():
            errors.append(f"val_path does not exist: {self.val_path}")
        if not self.test_path.exists():
            errors.append(f"test_path does not exist: {self.test_path}")

        if self.sequence_length < 1:
            errors.append(f"sequence_length must be >= 1, got {self.sequence_length}")

        if self.horizon not in [5, 20]:
            errors.append(f"Invalid horizon {self.horizon}, must be 5 or 20")

        if errors:
            raise ValueError(
                f"DatasetConfig validation failed:\n" +
                "\n".join(f"  - {e}" for e in errors)
            )


class TimeSeriesDataset:
    """
    Temporal dataset with windowing and strict ordering.

    Prevents leakage by:
    1. Never looking forward in time for features
    2. Respecting purge/embargo boundaries from Phase 1
    3. Maintaining symbol isolation (no cross-symbol windows)
    """

    def __init__(self, config: DatasetConfig):
        """Initialize dataset."""
        config.validate()
        self.config = config

        # Load data
        self.train_df = pd.read_parquet(config.train_path)
        self.val_df = pd.read_parquet(config.val_path)
        self.test_df = pd.read_parquet(config.test_path)

        # Detect columns
        self._detect_columns()

        # Create sequences
        self.train_sequences = self._create_sequences(self.train_df, 'train')
        self.val_sequences = self._create_sequences(self.val_df, 'val')
        self.test_sequences = self._create_sequences(self.test_df, 'test')

    def _detect_columns(self):
        """Auto-detect feature and label columns."""
        all_cols = self.train_df.columns.tolist()

        # Metadata columns (not features)
        meta_cols = ['datetime', 'symbol', 'split']

        # Label columns
        label_cols = [c for c in all_cols if c.startswith('label_')]

        # Feature columns = everything except metadata and labels
        feature_cols = [
            c for c in all_cols
            if c not in meta_cols and c not in label_cols
        ]

        # Use config if provided, otherwise auto-detect
        self.feature_columns = (
            self.config.feature_columns
            if self.config.feature_columns
            else feature_cols
        )

        # Determine label column
        if self.config.label_column:
            self.label_column = self.config.label_column
        else:
            # Infer from horizon
            self.label_column = f'label_h{self.config.horizon}'

        # Validation
        if self.label_column not in label_cols:
            raise ValueError(
                f"Label column '{self.label_column}' not found. "
                f"Available: {label_cols}"
            )

        for fc in self.feature_columns:
            if fc not in all_cols:
                raise ValueError(f"Feature column '{fc}' not found")

    def _create_sequences(
        self,
        df: pd.DataFrame,
        split_name: str
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Create windowed sequences from dataframe.

        Returns:
        --------
        X : np.ndarray, shape (n_sequences, seq_len, n_features)
        y : np.ndarray, shape (n_sequences,)
        metadata : pd.DataFrame with (datetime, symbol, index)
        """
        sequences_X = []
        sequences_y = []
        sequences_meta = []

        # Process each symbol separately (prevents cross-symbol leakage)
        for symbol in df['symbol'].unique():
            if self.config.include_symbols and symbol not in self.config.include_symbols:
                continue

            symbol_df = df[df['symbol'] == symbol].sort_values('datetime')

            # Extract arrays
            X_full = symbol_df[self.feature_columns].values
            y_full = symbol_df[self.label_column].values
            times = symbol_df['datetime'].values

            # Create windows
            seq_len = self.config.sequence_length
            for i in range(seq_len, len(X_full)):
                # Feature window: [i-seq_len : i] (past only)
                X_window = X_full[i - seq_len : i]

                # Label: at time i (future relative to window)
                y_label = y_full[i]

                # Skip if label is NaN or neutral (if configured)
                if np.isnan(y_label):
                    continue
                if self.config.exclude_neutrals and y_label == 0:
                    continue

                # Store
                sequences_X.append(X_window)
                sequences_y.append(y_label)
                sequences_meta.append({
                    'datetime': times[i],
                    'symbol': symbol,
                    'index': i
                })

        # Convert to arrays
        X = np.array(sequences_X, dtype=np.float32)
        y = np.array(sequences_y, dtype=np.int8)
        metadata = pd.DataFrame(sequences_meta)

        print(f"{split_name}: Created {len(X)} sequences from {len(df)} rows")

        return X, y, metadata

    def get_split(
        self,
        split: str
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Get a specific split.

        Parameters:
        -----------
        split : str
            'train', 'val', or 'test'

        Returns:
        --------
        X, y, metadata
        """
        if split == 'train':
            return self.train_sequences
        elif split == 'val':
            return self.val_sequences
        elif split == 'test':
            return self.test_sequences
        else:
            raise ValueError(f"Invalid split '{split}', must be train/val/test")

    def get_feature_columns(self) -> List[str]:
        """Get list of feature column names."""
        return self.feature_columns

    def __repr__(self) -> str:
        train_X, _, _ = self.train_sequences
        val_X, _, _ = self.val_sequences
        test_X, _, _ = self.test_sequences

        return (
            f"TimeSeriesDataset(\n"
            f"  horizon={self.config.horizon},\n"
            f"  sequence_length={self.config.sequence_length},\n"
            f"  n_features={len(self.feature_columns)},\n"
            f"  train_sequences={len(train_X)},\n"
            f"  val_sequences={len(val_X)},\n"
            f"  test_sequences={len(test_X)}\n"
            f")"
        )
```

**Key Features:**
- **Symbol isolation**: Windows never span multiple symbols
- **Temporal ordering**: Past features only, no future leakage
- **Flexible filtering**: Include/exclude symbols, neutrals
- **Metadata tracking**: Timestamps preserved for predictions
- **Memory efficient**: Option for memmap on large datasets

---

## 5. Training Orchestration (`src/training/trainer.py`)

### 5.1 Reusable Training Loop

```python
"""
Model Trainer - Orchestration for Training Workflow

Responsibilities:
1. Load data via TimeSeriesDataset
2. Instantiate model via ModelRegistry
3. Execute training with callbacks
4. Track experiments via MLflow
5. Save checkpoints and artifacts
"""
from pathlib import Path
from typing import Optional, Dict, Any, List
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from datetime import datetime

from src.data.dataset import TimeSeriesDataset, DatasetConfig
from src.models.registry import ModelRegistry
from src.models.base import BaseModel, PredictionOutput
from src.training.callbacks import CallbackList, EarlyStoppingCallback, CheckpointCallback
from src.training.evaluator import ModelEvaluator

class Trainer:
    """
    Training orchestrator for all model families.

    Handles:
    - Data loading
    - Model instantiation
    - Training loop
    - Experiment tracking (MLflow)
    - Checkpoint management
    """

    def __init__(
        self,
        model_name: str,
        model_config: dict,
        dataset_config: DatasetConfig,
        experiment_name: str = "default",
        output_dir: Path = Path("experiments/runs"),
        use_mlflow: bool = True
    ):
        """
        Initialize trainer.

        Parameters:
        -----------
        model_name : str
            Model to train (e.g., 'xgboost', 'timeseries:nhits')
        model_config : dict
            Model-specific configuration
        dataset_config : DatasetConfig
            Configuration for data loading
        experiment_name : str
            MLflow experiment name
        output_dir : Path
            Directory for checkpoints/artifacts
        use_mlflow : bool
            Enable MLflow tracking
        """
        self.model_name = model_name
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.use_mlflow = use_mlflow

        # Create output directory
        self.run_id = f"{model_name.replace(':', '_')}_{datetime.now():%Y%m%d_%H%M%S}"
        self.run_dir = output_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components (lazy)
        self.dataset: Optional[TimeSeriesDataset] = None
        self.model: Optional[BaseModel] = None
        self.evaluator: Optional[ModelEvaluator] = None

    def prepare_data(self):
        """Load and prepare dataset."""
        print(f"Loading data with horizon={self.dataset_config.horizon}")
        self.dataset = TimeSeriesDataset(self.dataset_config)
        print(self.dataset)

    def build_model(self):
        """Instantiate model from registry."""
        if self.dataset is None:
            raise RuntimeError("Call prepare_data() before build_model()")

        print(f"Building model: {self.model_name}")
        self.model = ModelRegistry.create(
            model_name=self.model_name,
            config=self.model_config,
            horizon=self.dataset_config.horizon,
            feature_columns=self.dataset.get_feature_columns()
        )
        print(self.model)

    def train(
        self,
        callbacks: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Execute training.

        Parameters:
        -----------
        callbacks : List, optional
            Training callbacks (early stopping, checkpointing, etc.)

        Returns:
        --------
        Dict with training results
        """
        if self.model is None:
            raise RuntimeError("Call build_model() before train()")

        # Get data
        X_train, y_train, meta_train = self.dataset.get_split('train')
        X_val, y_val, meta_val = self.dataset.get_split('val')

        # Setup MLflow
        if self.use_mlflow:
            mlflow.set_experiment(self.experiment_name)
            mlflow.start_run(run_name=self.run_id)

            # Log parameters
            mlflow.log_params({
                'model_name': self.model_name,
                'horizon': self.dataset_config.horizon,
                'sequence_length': self.dataset_config.sequence_length,
                'n_features': len(self.dataset.get_feature_columns()),
                'train_samples': len(X_train),
                'val_samples': len(X_val)
            })
            mlflow.log_params(self.model_config)

        # Train model
        print(f"\nTraining {self.model_name} for horizon H{self.dataset_config.horizon}")
        print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

        training_results = self.model.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            metadata_train=meta_train,
            metadata_val=meta_val
        )

        # Log metrics
        if self.use_mlflow:
            for metric_name, values in training_results.items():
                if isinstance(values, list):
                    for epoch, value in enumerate(values):
                        mlflow.log_metric(metric_name, value, step=epoch)
                else:
                    mlflow.log_metric(metric_name, values)

        # Save model
        model_path = self.run_dir / "model"
        self.model.save(model_path)

        if self.use_mlflow:
            mlflow.log_artifacts(str(model_path))

        return training_results

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate trained model on validation and test sets."""
        if not self.model.is_fitted:
            raise RuntimeError("Model not trained. Call train() first.")

        self.evaluator = ModelEvaluator(self.model)

        # Evaluate on validation set
        X_val, y_val, meta_val = self.dataset.get_split('val')
        val_metrics = self.evaluator.evaluate(X_val, y_val, meta_val, split_name='val')

        # Evaluate on test set
        X_test, y_test, meta_test = self.dataset.get_split('test')
        test_metrics = self.evaluator.evaluate(X_test, y_test, meta_test, split_name='test')

        # Save predictions
        val_preds = self.model.predict(X_val, meta_val)
        test_preds = self.model.predict(X_test, meta_test)

        val_preds.to_dataframe().to_parquet(self.run_dir / "val_predictions.parquet")
        test_preds.to_dataframe().to_parquet(self.run_dir / "test_predictions.parquet")

        # Log to MLflow
        if self.use_mlflow:
            mlflow.log_metrics({f'val_{k}': v for k, v in val_metrics.items()})
            mlflow.log_metrics({f'test_{k}': v for k, v in test_metrics.items()})
            mlflow.log_artifact(str(self.run_dir / "val_predictions.parquet"))
            mlflow.log_artifact(str(self.run_dir / "test_predictions.parquet"))

        results = {
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'run_dir': str(self.run_dir)
        }

        if self.use_mlflow:
            mlflow.end_run()

        return results

    def run_full_pipeline(self) -> Dict[str, Any]:
        """Execute full training pipeline: prepare -> build -> train -> evaluate."""
        self.prepare_data()
        self.build_model()
        train_results = self.train()
        eval_results = self.evaluate()

        return {
            'training': train_results,
            'evaluation': eval_results,
            'run_id': self.run_id,
            'run_dir': str(self.run_dir)
        }
```

**Key Features:**
- **MLflow integration**: Automatic experiment tracking
- **Unified interface**: Same trainer for all model families
- **Artifact management**: Checkpoints, predictions, metrics
- **Fail-fast validation**: Data/model validation before training
- **Modular callbacks**: Easy extension for custom behaviors

---

## 6. Example Model Implementation

### 6.1 XGBoost Model (`src/models/boosting/xgboost.py`)

```python
"""
XGBoost Model - Gradient Boosting Classifier

Self-registers with ModelRegistry via decorator.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle

from src.models.base import BaseModel, ModelConfig, PredictionOutput
from src.models.registry import ModelRegistry

@dataclass
class XGBoostConfig(ModelConfig):
    """XGBoost-specific configuration."""
    # XGBoost hyperparameters
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 1
    gamma: float = 0.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0

    # Multi-class strategy
    objective: str = 'multi:softprob'  # {-1, 0, 1} -> 3 classes
    num_class: int = 3
    eval_metric: str = 'mlogloss'

    # Training
    early_stopping_rounds: int = 10
    n_jobs: int = -1


@ModelRegistry.register(
    name="xgboost",
    family="boosting",
    description="XGBoost gradient boosting with early stopping",
    requires_gpu=False,
    supports_multivariate=True
)
class XGBoostModel(BaseModel):
    """
    XGBoost model for triple-barrier label prediction.

    Converts labels {-1, 0, 1} to {0, 1, 2} for XGBoost.
    """

    def _build_config(self, config: dict, horizon: int) -> XGBoostConfig:
        """Build XGBoost config from dict."""
        return XGBoostConfig(
            model_name='xgboost',
            model_family='boosting',
            horizon=horizon,
            **config
        )

    def _build_model(self):
        """Initialize XGBoost model."""
        params = {
            'objective': self.config.objective,
            'num_class': self.config.num_class,
            'max_depth': self.config.max_depth,
            'learning_rate': self.config.learning_rate,
            'subsample': self.config.subsample,
            'colsample_bytree': self.config.colsample_bytree,
            'min_child_weight': self.config.min_child_weight,
            'gamma': self.config.gamma,
            'reg_alpha': self.config.reg_alpha,
            'reg_lambda': self.config.reg_lambda,
            'random_state': self.config.random_seed,
            'n_jobs': self.config.n_jobs,
            'verbosity': 1 if self.config.verbose else 0
        }

        self.model = xgb.XGBClassifier(
            n_estimators=self.config.n_estimators,
            **params
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        metadata_train: Optional[pd.DataFrame] = None,
        metadata_val: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Train XGBoost model."""
        # Validate inputs
        self.validate_inputs(X_train, y_train)
        if X_val is not None:
            self.validate_inputs(X_val, y_val)

        # XGBoost expects 2D input - flatten if 3D sequences
        if X_train.ndim == 3:
            # Flatten: (n_samples, seq_len, n_features) -> (n_samples, seq_len * n_features)
            n_samples, seq_len, n_features = X_train.shape
            X_train = X_train.reshape(n_samples, seq_len * n_features)
            if X_val is not None:
                X_val = X_val.reshape(len(X_val), seq_len * n_features)

        # Convert labels {-1, 0, 1} -> {0, 1, 2} for XGBoost
        y_train_xgb = self._encode_labels(y_train)
        y_val_xgb = self._encode_labels(y_val) if y_val is not None else None

        # Prepare evaluation set
        eval_set = [(X_train, y_train_xgb)]
        if X_val is not None:
            eval_set.append((X_val, y_val_xgb))

        # Train
        self.model.fit(
            X_train,
            y_train_xgb,
            eval_set=eval_set,
            verbose=self.config.verbose
        )

        self.is_fitted = True

        # Extract training history
        results = self.model.evals_result()
        self.training_history = {
            'train_mlogloss': results['validation_0']['mlogloss'],
        }
        if 'validation_1' in results:
            self.training_history['val_mlogloss'] = results['validation_1']['mlogloss']

        return {
            'train_loss': self.training_history.get('train_mlogloss', []),
            'val_loss': self.training_history.get('val_mlogloss', []),
            'best_iteration': self.model.best_iteration,
            'n_estimators': self.model.n_estimators
        }

    def predict(
        self,
        X: np.ndarray,
        metadata: Optional[pd.DataFrame] = None
    ) -> PredictionOutput:
        """Generate predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.validate_inputs(X)

        # Flatten if 3D
        if X.ndim == 3:
            n_samples, seq_len, n_features = X.shape
            X = X.reshape(n_samples, seq_len * n_features)

        # Predict probabilities
        proba_xgb = self.model.predict_proba(X)  # Shape: (n, 3)

        # Predict class
        pred_xgb = self.model.predict(X)

        # Convert back: {0, 1, 2} -> {-1, 0, 1}
        predictions = self._decode_labels(pred_xgb)

        # Extract metadata
        if metadata is not None:
            timestamps = pd.to_datetime(metadata['datetime'])
            symbols = metadata['symbol'].values
        else:
            timestamps = pd.date_range('2020-01-01', periods=len(X), freq='5min')
            symbols = np.array(['UNK'] * len(X))

        return PredictionOutput(
            predictions=predictions,
            probabilities=proba_xgb,
            timestamps=timestamps,
            symbols=symbols,
            horizons=np.array([self.horizon] * len(X))
        )

    def save(self, path: Path):
        """Save model to disk."""
        path.mkdir(parents=True, exist_ok=True)

        # Save XGBoost model
        self.model.save_model(str(path / 'xgboost_model.json'))

        # Save metadata
        metadata = {
            'config': self.config,
            'feature_columns': self.feature_columns,
            'training_history': self.training_history,
            'is_fitted': self.is_fitted
        }
        with open(path / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)

    def load(self, path: Path):
        """Load model from disk."""
        # Load XGBoost model
        self.model.load_model(str(path / 'xgboost_model.json'))

        # Load metadata
        with open(path / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)

        self.config = metadata['config']
        self.feature_columns = metadata['feature_columns']
        self.training_history = metadata['training_history']
        self.is_fitted = metadata['is_fitted']

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)

    def _encode_labels(self, y: np.ndarray) -> np.ndarray:
        """Convert {-1, 0, 1} -> {0, 1, 2}."""
        return y + 1  # -1->0, 0->1, 1->2

    def _decode_labels(self, y: np.ndarray) -> np.ndarray:
        """Convert {0, 1, 2} -> {-1, 0, 1}."""
        return y - 1  # 0->-1, 1->0, 2->1
```

**Key Features:**
- **Auto-registration**: `@ModelRegistry.register` decorator
- **Label encoding**: Handles {-1, 0, 1} conversion for XGBoost
- **3D->2D flattening**: Supports sequence inputs by flattening
- **Feature importance**: Tree-based intrinsic importance
- **Early stopping**: Built-in validation-based stopping

---

## 7. Configuration Extension

### 7.1 Model Configs (`config/models/xgboost.yaml`)

```yaml
# XGBoost Model Configuration
model:
  name: "xgboost"
  family: "boosting"

hyperparameters:
  n_estimators: 500
  max_depth: 8
  learning_rate: 0.05
  subsample: 0.8
  colsample_bytree: 0.8
  min_child_weight: 3
  gamma: 0.1
  reg_alpha: 0.1
  reg_lambda: 1.0

training:
  early_stopping_rounds: 20
  random_seed: 42
  verbose: true
  n_jobs: -1

dataset:
  sequence_length: 1  # No windowing for XGBoost (use current bar features only)
  exclude_neutrals: false
  include_symbols: null  # All symbols
```

### 7.2 Experiment Config (`config/experiments/baseline.yaml`)

```yaml
# Baseline Experiment Configuration
experiment:
  name: "baseline_h5_h20"
  description: "Baseline models for H5 and H20 horizons"

horizons:
  - 5
  - 20

models:
  - name: "xgboost"
    config_path: "config/models/xgboost.yaml"

  - name: "lightgbm"
    config_path: "config/models/lightgbm.yaml"

  - name: "timeseries:nhits"
    config_path: "config/models/nhits.yaml"

data:
  train_path: "data/splits/scaled/train_scaled.parquet"
  val_path: "data/splits/scaled/val_scaled.parquet"
  test_path: "data/splits/scaled/test_scaled.parquet"

mlflow:
  tracking_uri: "experiments/mlruns"
  experiment_name: "baseline"

output:
  base_dir: "experiments/runs"
```

---

## 8. Integration with Phase 1 Pipeline

### 8.1 Integration Points

```
Phase 1 Output                    Phase 2 Input
----------------                  ----------------
data/splits/scaled/               TimeSeriesDataset loads:
├── train_scaled.parquet    --->    train_path
├── val_scaled.parquet      --->    val_path
└── test_scaled.parquet     --->    test_path

Features: 107 columns             feature_columns auto-detected
Labels: label_h5, label_h20       label_column = f'label_h{horizon}'
Purge/Embargo: Already applied    No additional filtering needed
```

### 8.2 CLI Entry Point (`scripts/train_model.py`)

```python
"""
CLI for training individual models.

Usage:
    python scripts/train_model.py \
        --model xgboost \
        --horizon 5 \
        --config config/models/xgboost.yaml \
        --output experiments/runs
"""
import argparse
from pathlib import Path
import yaml

from src.config import SPLITS_DIR, set_global_seeds
from src.training.trainer import Trainer
from src.data.dataset import DatasetConfig

def main():
    parser = argparse.ArgumentParser(description="Train a single model")
    parser.add_argument('--model', required=True, help='Model name (e.g., xgboost, timeseries:nhits)')
    parser.add_argument('--horizon', type=int, required=True, choices=[5, 20], help='Prediction horizon')
    parser.add_argument('--config', type=Path, required=True, help='Path to model config YAML')
    parser.add_argument('--output', type=Path, default=Path('experiments/runs'), help='Output directory')
    parser.add_argument('--no-mlflow', action='store_true', help='Disable MLflow tracking')

    args = parser.parse_args()

    # Set seeds
    set_global_seeds()

    # Load model config
    with open(args.config) as f:
        config_yaml = yaml.safe_load(f)

    model_config = config_yaml['hyperparameters']
    dataset_params = config_yaml.get('dataset', {})

    # Build dataset config
    dataset_config = DatasetConfig(
        train_path=SPLITS_DIR / 'scaled' / 'train_scaled.parquet',
        val_path=SPLITS_DIR / 'scaled' / 'val_scaled.parquet',
        test_path=SPLITS_DIR / 'scaled' / 'test_scaled.parquet',
        horizon=args.horizon,
        sequence_length=dataset_params.get('sequence_length', 60),
        exclude_neutrals=dataset_params.get('exclude_neutrals', False)
    )

    # Create trainer
    trainer = Trainer(
        model_name=args.model,
        model_config=model_config,
        dataset_config=dataset_config,
        experiment_name=f"{args.model}_h{args.horizon}",
        output_dir=args.output,
        use_mlflow=not args.no_mlflow
    )

    # Run full pipeline
    results = trainer.run_full_pipeline()

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Run ID: {results['run_id']}")
    print(f"Run Dir: {results['run_dir']}")
    print(f"\nValidation Metrics:")
    for k, v in results['evaluation']['val_metrics'].items():
        print(f"  {k}: {v:.4f}")
    print(f"\nTest Metrics:")
    for k, v in results['evaluation']['test_metrics'].items():
        print(f"  {k}: {v:.4f}")

if __name__ == '__main__':
    main()
```

---

## 9. Hyperparameter Tuning Integration

### 9.1 Optuna Tuner (`src/tuning/optuna_tuner.py`)

```python
"""
Optuna Hyperparameter Tuner

Responsibilities:
1. Define search spaces for each model family
2. Run Optuna trials with cross-validation
3. Track best hyperparameters
4. Integration with MLflow
"""
from typing import Dict, Any, Callable
import optuna
from optuna.integration.mlflow import MLflowCallback

from src.training.trainer import Trainer
from src.data.dataset import DatasetConfig

class OptunaModelTuner:
    """
    Hyperparameter tuning with Optuna.

    Integrates with ModelRegistry and Trainer for seamless tuning.
    """

    def __init__(
        self,
        model_name: str,
        dataset_config: DatasetConfig,
        search_space_fn: Callable,
        n_trials: int = 50,
        direction: str = 'maximize',
        metric_name: str = 'val_f1'
    ):
        """
        Initialize tuner.

        Parameters:
        -----------
        model_name : str
            Model to tune
        dataset_config : DatasetConfig
            Data configuration
        search_space_fn : Callable
            Function that returns search space dict given trial
        n_trials : int
            Number of Optuna trials
        direction : str
            'maximize' or 'minimize'
        metric_name : str
            Metric to optimize
        """
        self.model_name = model_name
        self.dataset_config = dataset_config
        self.search_space_fn = search_space_fn
        self.n_trials = n_trials
        self.direction = direction
        self.metric_name = metric_name

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function."""
        # Sample hyperparameters
        model_config = self.search_space_fn(trial)

        # Create trainer
        trainer = Trainer(
            model_name=self.model_name,
            model_config=model_config,
            dataset_config=self.dataset_config,
            experiment_name=f"optuna_{self.model_name}",
            use_mlflow=True
        )

        # Train and evaluate
        trainer.prepare_data()
        trainer.build_model()
        trainer.train()
        results = trainer.evaluate()

        # Extract metric
        metric_value = results['evaluation']['val_metrics'][self.metric_name]

        return metric_value

    def tune(self) -> Dict[str, Any]:
        """Run hyperparameter tuning."""
        study = optuna.create_study(
            direction=self.direction,
            study_name=f"{self.model_name}_tuning"
        )

        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            callbacks=[MLflowCallback(tracking_uri="experiments/mlruns")]
        )

        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'best_trial': study.best_trial.number
        }
```

---

## 10. Summary & Next Steps

### 10.1 What This Architecture Provides

✅ **Modular Design**: Each model in isolated module (<650 lines)
✅ **Common Interface**: BaseModel enforces consistent contracts
✅ **Plugin Architecture**: Models self-register via decorator
✅ **Zero Leakage**: TimeSeriesDataset enforces temporal ordering
✅ **Fail-Fast**: Comprehensive validation at every boundary
✅ **Experiment Tracking**: MLflow integration for all runs
✅ **Extensibility**: Easy to add new model families
✅ **Phase 1 Integration**: Seamless loading of Phase 1 outputs

### 10.2 Implementation Roadmap

**Week 1: Core Infrastructure**
1. Implement `src/models/base.py` (BaseModel, ModelConfig, PredictionOutput)
2. Implement `src/models/registry.py` (ModelRegistry with auto-discovery)
3. Implement `src/data/dataset.py` (TimeSeriesDataset)
4. Write unit tests for core components

**Week 2: First Model Family (Boosting)**
5. Implement `src/models/boosting/xgboost.py`
6. Implement `src/models/boosting/lightgbm.py`
7. Implement `src/models/boosting/catboost.py`
8. Test boosting models end-to-end

**Week 3: Training Infrastructure**
9. Implement `src/training/trainer.py`
10. Implement `src/training/evaluator.py`
11. Implement `src/training/callbacks.py`
12. Setup MLflow tracking

**Week 4: Time Series Models**
13. Implement `src/models/timeseries/nhits.py`
14. Implement `src/models/timeseries/tft.py`
15. Run baseline experiments
16. Generate comparison report

**Week 5: Hyperparameter Tuning**
17. Implement `src/tuning/optuna_tuner.py`
18. Define search spaces for each model
19. Run tuning experiments
20. Lock in production hyperparameters

### 10.3 Key Metrics to Track

**Model Performance:**
- Accuracy, Precision, Recall, F1 (per class: -1, 0, 1)
- Sharpe Ratio (simulated trading)
- Win Rate, Max Drawdown
- Profit Factor

**Training Efficiency:**
- Training time per epoch
- Convergence speed (early stopping)
- Memory usage

**Experiment Metadata:**
- Hyperparameters
- Model checkpoints
- Feature importance (for tree models)
- Predictions (val/test)

### 10.4 File Size Compliance

All modules respect 650-line limit:
- `base.py`: ~250 lines (BaseModel, ModelConfig, PredictionOutput)
- `registry.py`: ~180 lines (ModelRegistry + auto-discovery)
- `dataset.py`: ~200 lines (TimeSeriesDataset)
- `trainer.py`: ~200 lines (Trainer orchestration)
- `xgboost.py`: ~180 lines (XGBoostModel implementation)
- Each additional model: ~150-200 lines

Total new code: ~6,000 lines across 20+ modules, all <650 lines each.

---

## Appendix A: Decision Log

**Decision 1: Why Plugin Architecture over Monolithic Trainer?**
- **Rationale**: Easier to add new model families without modifying core
- **Trade-off**: Slightly more boilerplate (decorators, registration)
- **Benefit**: Models are self-contained, testable in isolation

**Decision 2: Why MLflow over Custom Tracking?**
- **Rationale**: Industry-standard, mature, excellent UI
- **Trade-off**: Additional dependency
- **Benefit**: Model registry, artifact management, comparison UI

**Decision 3: Why Abstract Base Class over Duck Typing?**
- **Rationale**: Fail-fast at instantiation, not at runtime
- **Trade-off**: More rigid interface
- **Benefit**: Catches contract violations early in development

**Decision 4: Why TimeSeriesDataset over Direct Parquet Loading?**
- **Rationale**: Centralize windowing logic, enforce temporal ordering
- **Trade-off**: Memory overhead for sequence storage
- **Benefit**: Guaranteed leakage prevention, reusable across models

**Decision 5: Why YAML Configs over Python Dicts?**
- **Rationale**: Easier for non-developers to modify, version control friendly
- **Trade-off**: Requires parsing, less type-safe
- **Benefit**: Experiment reproducibility, shareable configs

---

## Appendix B: Example End-to-End Workflow

```bash
# Step 1: Train baseline XGBoost for H5
python scripts/train_model.py \
    --model xgboost \
    --horizon 5 \
    --config config/models/xgboost.yaml \
    --output experiments/runs

# Step 2: Train N-HiTS time series model for H20
python scripts/train_model.py \
    --model timeseries:nhits \
    --horizon 20 \
    --config config/models/nhits.yaml \
    --output experiments/runs

# Step 3: Tune LightGBM hyperparameters
python scripts/tune_model.py \
    --model lightgbm \
    --horizon 5 \
    --n-trials 100 \
    --output experiments/tuning

# Step 4: Run full experiment (all models, all horizons)
python scripts/run_experiment.py \
    --config config/experiments/baseline.yaml

# Step 5: View results in MLflow UI
mlflow ui --backend-store-uri experiments/mlruns
```

---

**End of Phase 2 Architecture Document**
