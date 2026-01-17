# PHASE 0: FOUNDATION - Clean ML Factory Core Package

**Status:** PLANNING
**Created:** 2026-01-16
**Purpose:** Define foundational interfaces, types, and constants for clean ML Factory implementation

---

## Overview

Phase 0 establishes the core infrastructure for a clean ML Factory. This is a **clean slate** implementation - no backward compatibility constraints. All core abstractions are defined in a single `src/core/` package.

**Design Principles:**
1. **Single Source of Truth** - All interfaces, types, and constants in one package
2. **Explicit Contracts** - Every interface has clear input/output specifications
3. **Fail Fast** - Validate at boundaries, crash early on invalid state
4. **Minimal Coupling** - Each module does one thing well

---

## Task 0.1: Create `src/core/` Package Structure

Create the following file structure:

```
src/core/
├── __init__.py           # Package exports
├── interfaces.py         # All abstract base classes
├── types.py              # All type definitions and enums
├── constants.py          # All constants (timeframes, horizons, etc.)
└── validation.py         # Input validation utilities
```

### File: `src/core/__init__.py`

```python
"""
Core package - Foundational interfaces, types, and constants.

Clean ML Factory Phase 0 Foundation.

This package is the SINGLE SOURCE OF TRUTH for:
- Abstract interfaces (ModelContract, AdapterContract, DataContract)
- Type definitions (DataRank, ModelFamily, FeatureFamily, etc.)
- Constants (CANONICAL_TIMEFRAMES, MODEL_FAMILIES, etc.)
- Validation utilities
"""

from src.core.interfaces import (
    DataContract,
    ModelContract,
    AdapterContract,
    AdapterResult,
    TrainingResult,
    OOFResult,
)
from src.core.types import (
    DataRank,
    ModelFamily,
    FeatureFamily,
    TrainingMode,
    CVMethod,
    Features,
    Labels,
    ModelType,
)
from src.core.constants import (
    CANONICAL_TIMEFRAMES,
    DEFAULT_HORIZONS,
    DEFAULT_SPLIT_RATIOS,
    DEFAULT_PURGE_BARS,
    DEFAULT_EMBARGO_BARS,
    MODEL_FAMILIES,
    MODEL_TO_FAMILY,
    MODEL_DATA_RANKS,
)
from src.core.validation import (
    validate_input_shape,
    validate_features,
    validate_labels,
    validate_dataframe,
    ValidationError,
)

__all__ = [
    # Interfaces
    "DataContract",
    "ModelContract",
    "AdapterContract",
    # Result types
    "AdapterResult",
    "TrainingResult",
    "OOFResult",
    # Enums
    "DataRank",
    "ModelFamily",
    "FeatureFamily",
    "TrainingMode",
    "CVMethod",
    # Type aliases
    "Features",
    "Labels",
    "ModelType",
    # Constants
    "CANONICAL_TIMEFRAMES",
    "DEFAULT_HORIZONS",
    "DEFAULT_SPLIT_RATIOS",
    "DEFAULT_PURGE_BARS",
    "DEFAULT_EMBARGO_BARS",
    "MODEL_FAMILIES",
    "MODEL_TO_FAMILY",
    "MODEL_DATA_RANKS",
    # Validation
    "validate_input_shape",
    "validate_features",
    "validate_labels",
    "validate_dataframe",
    "ValidationError",
]
```

---

## Task 0.2: Define Core Interfaces in `interfaces.py`

### File: `src/core/interfaces.py`

```python
"""
Core interfaces - Abstract base classes for the ML Factory.

Every model, adapter, and data handler implements these contracts.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar, List, Dict, Optional
import numpy as np
import pandas as pd


@dataclass
class AdapterResult:
    """Output from any adapter transformation."""
    data: np.ndarray  # Transformed features (2D, 3D, or 4D)
    labels: np.ndarray  # Target labels (1D)
    feature_names: List[str]
    original_indices: np.ndarray  # For OOF alignment
    weights: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return self.data.shape[0]

    @property
    def n_features(self) -> int:
        return self.data.shape[-1]

    @property
    def rank(self) -> int:
        return self.data.ndim


@dataclass
class TrainingResult:
    """Output from model training."""
    model: Any
    metrics: Dict[str, float]
    oof_predictions: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None
    training_time_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OOFResult:
    """Out-of-fold predictions with alignment info."""
    predictions: np.ndarray
    probabilities: np.ndarray
    indices: np.ndarray
    fold_ids: np.ndarray
    model_name: str = ""

    @property
    def n_samples(self) -> int:
        return len(self.predictions)

    @property
    def n_classes(self) -> int:
        return self.probabilities.shape[1]


class DataContract(ABC):
    """Contract for model data requirements."""

    @property
    @abstractmethod
    def rank(self) -> int:
        """Data dimensionality: 2=tabular, 3=sequence, 4=multi-stream."""
        pass

    @property
    @abstractmethod
    def required_features(self) -> List[str]:
        """Minimum required feature families."""
        pass

    @property
    @abstractmethod
    def feature_bounds(self) -> tuple:
        """(min_features, max_features) tuple."""
        pass


class ModelContract(ABC):
    """Contract all models must implement."""

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> TrainingResult:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "ModelContract":
        pass

    @property
    @abstractmethod
    def data_contract(self) -> DataContract:
        pass


class AdapterContract(ABC):
    """Contract all adapters must implement."""

    @abstractmethod
    def transform(self, df: pd.DataFrame, contract: DataContract) -> AdapterResult:
        pass

    @property
    @abstractmethod
    def output_rank(self) -> int:
        pass


ModelType = TypeVar("ModelType", bound=ModelContract)
```

---

## Task 0.3: Define Types in `types.py`

### File: `src/core/types.py`

```python
"""Core type definitions - Enums and type aliases."""

from enum import Enum
from typing import TypeVar, Union
import numpy as np
import pandas as pd


class DataRank(int, Enum):
    TABULAR_2D = 2
    SEQUENCE_3D = 3
    MULTI_TF_4D = 4


class ModelFamily(str, Enum):
    BOOSTING = "boosting"
    CLASSICAL = "classical"
    NEURAL = "neural"
    ENSEMBLE = "ensemble"
    META_LEARNER = "meta_learner"


class FeatureFamily(str, Enum):
    RAW = "raw"
    MOMENTUM = "momentum"
    MOVING_AVERAGE = "moving_average"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    TREND = "trend"
    PRICE = "price"
    MICROSTRUCTURE = "microstructure"
    ENTROPY = "entropy"
    WAVELETS = "wavelets"
    TEMPORAL = "temporal"
    REGIME = "regime"
    MTF = "mtf"


class TrainingMode(str, Enum):
    STANDARD = "standard"
    WALK_FORWARD = "walk_forward"
    REGIME_AWARE = "regime_aware"
    META_LABELING = "meta_labeling"


class CVMethod(str, Enum):
    PURGED_KFOLD = "purged_kfold"
    WALK_FORWARD = "walk_forward"
    CPCV = "cpcv"
    PBO = "pbo"


# Type aliases
Features = Union[np.ndarray, pd.DataFrame]
Labels = np.ndarray
ModelType = TypeVar("ModelType", bound="ModelContract")
```

---

## Task 0.4: Define Constants in `constants.py`

### File: `src/core/constants.py`

```python
"""Core constants - Canonical values for the ML Factory."""

# 9-TF intraday ladder
CANONICAL_TIMEFRAMES = [
    "1min", "5min", "10min", "15min", "20min", "25min", "30min", "45min", "60min"
]

DEFAULT_HORIZONS = [5, 10, 15, 20]

DEFAULT_SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

DEFAULT_PURGE_BARS = 60
DEFAULT_EMBARGO_BARS = 1440

MODEL_FAMILIES = {
    "boosting": ["xgboost", "lightgbm", "catboost"],
    "classical": ["random_forest", "logistic", "svm"],
    "neural": ["lstm", "gru", "tcn", "transformer", "patchtst", "itransformer",
               "tft", "nbeats", "inceptiontime", "resnet1d"],
    "ensemble": ["voting", "stacking", "blending"],
    "meta_learner": ["ridge_meta", "mlp_meta", "xgboost_meta", "calibrated_meta"],
}

MODEL_TO_FAMILY = {
    model: family for family, models in MODEL_FAMILIES.items() for model in models
}

MODEL_DATA_RANKS = {
    # Boosting - 2D
    "xgboost": 2, "lightgbm": 2, "catboost": 2,
    # Classical - 2D
    "random_forest": 2, "logistic": 2, "svm": 2,
    # Neural - 3D
    "lstm": 3, "gru": 3, "tcn": 3, "transformer": 3,
    "nbeats": 3, "inceptiontime": 3, "resnet1d": 3,
    # Advanced Neural - 4D
    "patchtst": 4, "itransformer": 4, "tft": 4,
    # Ensemble/Meta - 2D
    "voting": 2, "stacking": 2, "blending": 2,
    "ridge_meta": 2, "mlp_meta": 2, "xgboost_meta": 2, "calibrated_meta": 2,
}
```

---

## Task 0.5: Define Validation in `validation.py`

### File: `src/core/validation.py`

```python
"""Input validation utilities - Fail fast at boundaries."""

import numpy as np
import pandas as pd
from typing import List

from src.core.constants import MODEL_TO_FAMILY, CANONICAL_TIMEFRAMES


class ValidationError(ValueError):
    """Validation error with actionable message."""
    def __init__(self, message: str, field: str = None):
        self.field = field
        super().__init__(message)


def validate_input_shape(X: np.ndarray, expected_rank: int, context: str = "input"):
    if X.ndim != expected_rank:
        raise ValidationError(
            f"{context} must be {expected_rank}D, got {X.ndim}D with shape {X.shape}",
            field="X",
        )
    if X.size == 0:
        raise ValidationError(f"{context} is empty", field="X")
    if np.isnan(X).any():
        raise ValidationError(f"{context} contains NaN values", field="X")


def validate_labels(y: np.ndarray, n_samples: int, context: str = "labels"):
    if y.ndim != 1:
        raise ValidationError(f"{context} must be 1D", field="y")
    if len(y) != n_samples:
        raise ValidationError(f"{context} length != feature samples", field="y")


def validate_features(feature_columns: List[str], min_f: int, max_f: int, context: str = "features"):
    n = len(feature_columns)
    if n < min_f:
        raise ValidationError(f"{context}: too few features ({n})", field="features")
    if n > max_f:
        raise ValidationError(f"{context}: too many features ({n})", field="features")


def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None, context: str = "DataFrame"):
    if df.empty:
        raise ValidationError(f"{context} is empty", field="df")
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValidationError(f"{context} missing columns: {sorted(missing)[:10]}", field="df")
```

---

## Task 0.6: Data Flow Diagram

```
Raw 1-min OHLCV
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  PHASE 1: Unified Features                          │
│  FeatureRegistry.compute_all() → 150+ indicators    │
│  Output: DataFrame with ALL features                │
└─────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  PHASE 2: Adapters (per model)                      │
│  TabularAdapter  → 2D (n, ~100)    [boosting]       │
│  SequenceAdapter → 3D (n, 60, ~80) [neural]         │
│  MultiStreamAdapter → 4D (n, 3, 60, 5) [transformer]│
└─────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  PHASE 3: Training Orchestrator                     │
│  Single entry: orchestrator.train(models=[...])     │
│  Handles: CV, OOF generation, metrics               │
└─────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  PHASE 4: Meta-Learners                             │
│  Stack OOF from heterogeneous bases                 │
│  Train: ridge_meta, mlp_meta, xgboost_meta          │
└─────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  PHASE 5: Inference Bundle                          │
│  Auto-create bundle with feature lineage            │
│  Deploy: ModelBundle.infer(raw_ohlcv)               │
└─────────────────────────────────────────────────────┘
```

---

## Implementation Checklist

### Task 0.1: Create Package Structure
- [ ] Create `src/core/__init__.py`
- [ ] Create `src/core/interfaces.py`
- [ ] Create `src/core/types.py`
- [ ] Create `src/core/constants.py`
- [ ] Create `src/core/validation.py`

### Task 0.2: Implement Interfaces
- [ ] `AdapterResult` dataclass
- [ ] `TrainingResult` dataclass
- [ ] `OOFResult` dataclass
- [ ] `DataContract` ABC
- [ ] `ModelContract` ABC
- [ ] `AdapterContract` ABC

### Task 0.3: Implement Types
- [ ] `DataRank` enum (2, 3, 4)
- [ ] `ModelFamily` enum
- [ ] `FeatureFamily` enum
- [ ] `TrainingMode` enum
- [ ] `CVMethod` enum
- [ ] Type aliases

### Task 0.4: Implement Constants
- [ ] `CANONICAL_TIMEFRAMES` (9 TFs)
- [ ] `DEFAULT_HORIZONS`
- [ ] `DEFAULT_SPLIT_RATIOS`
- [ ] `MODEL_FAMILIES` (22 models)
- [ ] `MODEL_TO_FAMILY` mapping
- [ ] `MODEL_DATA_RANKS` mapping

### Task 0.5: Implement Validation
- [ ] `ValidationError` exception
- [ ] `validate_input_shape()`
- [ ] `validate_labels()`
- [ ] `validate_features()`
- [ ] `validate_dataframe()`

### Verification
- [ ] All imports resolve
- [ ] No circular dependencies
- [ ] All `__all__` exports work
