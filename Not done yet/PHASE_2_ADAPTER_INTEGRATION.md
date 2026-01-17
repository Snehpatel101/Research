# PHASE 2: ADAPTER INTEGRATION - No Bypass

**Status:** PLANNING
**Created:** 2026-01-16
**Purpose:** Integrate adapters as sole pathway for data preparation - ALL data goes through adapters

---

## Overview

Phase 2 ensures that ALL data preparation flows through the adapter system. The Trainer must use adapters exclusively - no direct numpy conversions or bypass paths.

**Key Principle:** One model → One adapter → One data format

---

## Task 2.1: Create Adapter Registry

### File: `src/adapters/registry.py`

```python
from typing import Dict, Type
from src.core.interfaces import AdapterContract

class AdapterRegistry:
    _adapters: Dict[str, Type[AdapterContract]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register an adapter."""
        def decorator(adapter_cls):
            cls._adapters[name] = adapter_cls
            return adapter_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[AdapterContract]:
        if name not in cls._adapters:
            raise ValueError(f"Unknown adapter: {name}")
        return cls._adapters[name]

    @classmethod
    def list_all(cls) -> list:
        return list(cls._adapters.keys())
```

---

## Task 2.2: Define Model-to-Adapter Mapping

```python
MODEL_ADAPTER_MAP = {
    # Boosting → Tabular (2D)
    "xgboost": "tabular",
    "lightgbm": "tabular",
    "catboost": "tabular",

    # Classical → Tabular (2D)
    "random_forest": "tabular",
    "logistic": "tabular",
    "svm": "tabular",

    # Neural → Sequence (3D)
    "lstm": "sequence",
    "gru": "sequence",
    "tcn": "sequence",
    "transformer": "sequence",
    "nbeats": "sequence",
    "inceptiontime": "sequence",
    "resnet1d": "sequence",

    # Advanced Neural → Multi-Stream (4D)
    "patchtst": "multi_stream",
    "itransformer": "multi_stream",
    "tft": "multi_stream",

    # Ensemble/Meta → Tabular (2D on OOF)
    "voting": "tabular",
    "stacking": "tabular",
    "blending": "tabular",
    "ridge_meta": "tabular",
    "mlp_meta": "tabular",
    "xgboost_meta": "tabular",
    "calibrated_meta": "tabular",
}
```

---

## Task 2.3: TabularAdapter Implementation

### File: `src/adapters/tabular.py`

```python
import numpy as np
import pandas as pd
from typing import List, Optional

from src.core.interfaces import AdapterContract, AdapterResult, DataContract
from .registry import AdapterRegistry


@AdapterRegistry.register("tabular")
class TabularAdapter(AdapterContract):
    """2D adapter for boosting and classical models."""

    def __init__(self, feature_columns: List[str], label_column: str = "label"):
        self.feature_columns = feature_columns
        self.label_column = label_column

    def transform(self, df: pd.DataFrame, contract: DataContract) -> AdapterResult:
        """Transform DataFrame to 2D arrays."""
        # Select features
        available = [c for c in self.feature_columns if c in df.columns]

        # Validate bounds
        min_f, max_f = contract.feature_bounds
        if len(available) < min_f:
            raise ValueError(f"Too few features: {len(available)} < {min_f}")
        if len(available) > max_f:
            available = available[:max_f]

        X = df[available].values.astype(np.float32)
        y = df[self.label_column].values if self.label_column in df.columns else np.zeros(len(df))

        return AdapterResult(
            data=X,  # Shape: (n_samples, n_features)
            labels=y,
            feature_names=available,
            original_indices=df.index.values,
            metadata={"rank": 2, "shape": X.shape}
        )

    @property
    def output_rank(self) -> int:
        return 2
```

---

## Task 2.4: SequenceAdapter Implementation

### File: `src/adapters/sequence.py`

```python
import numpy as np
import pandas as pd
from typing import List

from src.core.interfaces import AdapterContract, AdapterResult, DataContract
from .registry import AdapterRegistry


@AdapterRegistry.register("sequence")
class SequenceAdapter(AdapterContract):
    """3D adapter for neural sequence models."""

    def __init__(
        self,
        feature_columns: List[str],
        label_column: str = "label",
        sequence_length: int = 60
    ):
        self.feature_columns = feature_columns
        self.label_column = label_column
        self.sequence_length = sequence_length

    def transform(self, df: pd.DataFrame, contract: DataContract) -> AdapterResult:
        """Transform DataFrame to 3D sequences."""
        available = [c for c in self.feature_columns if c in df.columns]

        data = df[available].values.astype(np.float32)
        labels = df[self.label_column].values if self.label_column in df.columns else np.zeros(len(df))

        # Build sequences
        sequences, seq_labels, indices = self._build_sequences(data, labels, df.index.values)

        return AdapterResult(
            data=sequences,  # Shape: (n_samples, seq_len, n_features)
            labels=seq_labels,
            feature_names=available,
            original_indices=indices,
            metadata={
                "rank": 3,
                "sequence_length": self.sequence_length,
                "shape": sequences.shape
            }
        )

    def _build_sequences(self, data: np.ndarray, labels: np.ndarray, indices: np.ndarray):
        """Build sliding window sequences."""
        n_total = len(data)
        n_samples = n_total - self.sequence_length + 1
        n_features = data.shape[1]

        sequences = np.zeros((n_samples, self.sequence_length, n_features), dtype=np.float32)
        seq_labels = np.zeros(n_samples, dtype=labels.dtype)
        seq_indices = np.zeros(n_samples, dtype=indices.dtype)

        for i in range(n_samples):
            sequences[i] = data[i:i + self.sequence_length]
            seq_labels[i] = labels[i + self.sequence_length - 1]
            seq_indices[i] = indices[i + self.sequence_length - 1]

        return sequences, seq_labels, seq_indices

    @property
    def output_rank(self) -> int:
        return 3
```

---

## Task 2.5: MultiStreamAdapter Implementation

### File: `src/adapters/multi_stream.py`

```python
import numpy as np
import pandas as pd
from typing import Dict, List

from src.core.interfaces import AdapterContract, AdapterResult, DataContract
from .registry import AdapterRegistry


@AdapterRegistry.register("multi_stream")
class MultiStreamAdapter(AdapterContract):
    """4D adapter for multi-timeframe models."""

    def __init__(
        self,
        feature_columns: List[str] = None,  # Default: raw OHLCV
        label_column: str = "label",
        sequence_length: int = 60,
        timeframes: List[str] = None
    ):
        self.feature_columns = feature_columns or ["open", "high", "low", "close", "volume"]
        self.label_column = label_column
        self.sequence_length = sequence_length
        self.timeframes = timeframes or ["1min", "5min", "15min"]

    def transform(
        self,
        df_dict: Dict[str, pd.DataFrame],
        contract: DataContract
    ) -> AdapterResult:
        """
        Transform multi-TF DataFrames to 4D tensor.

        Args:
            df_dict: Dict mapping timeframe -> DataFrame

        Returns:
            AdapterResult with shape (n_samples, n_timeframes, seq_len, n_features)
        """
        streams = []
        common_indices = None

        for tf in self.timeframes:
            if tf not in df_dict:
                raise ValueError(f"Missing timeframe: {tf}")

            df = df_dict[tf]
            available = [c for c in self.feature_columns if c in df.columns]
            data = df[available].values.astype(np.float32)

            # Build sequences for this TF
            seq, _, indices = self._build_sequences(data, df.index.values)
            streams.append(seq)

            # Track common indices
            if common_indices is None:
                common_indices = set(indices)
            else:
                common_indices &= set(indices)

        # Stack timeframes: (n_samples, n_timeframes, seq_len, n_features)
        tensor = np.stack(streams, axis=1)

        # Get labels from primary TF
        primary_df = df_dict[self.timeframes[0]]
        labels = primary_df[self.label_column].values if self.label_column in primary_df.columns else np.zeros(len(tensor))

        return AdapterResult(
            data=tensor,
            labels=labels[-len(tensor):],
            feature_names=self.feature_columns,
            original_indices=np.array(sorted(common_indices)),
            metadata={
                "rank": 4,
                "timeframes": self.timeframes,
                "sequence_length": self.sequence_length,
                "shape": tensor.shape
            }
        )

    def _build_sequences(self, data: np.ndarray, indices: np.ndarray):
        n_total = len(data)
        n_samples = n_total - self.sequence_length + 1
        n_features = data.shape[1]

        sequences = np.zeros((n_samples, self.sequence_length, n_features), dtype=np.float32)
        seq_indices = indices[self.sequence_length - 1:]

        for i in range(n_samples):
            sequences[i] = data[i:i + self.sequence_length]

        return sequences, None, seq_indices

    @property
    def output_rank(self) -> int:
        return 4
```

---

## Task 2.6: Integrate Adapters into Trainer

### File: `src/models/training/trainer.py` (modifications)

```python
from src.adapters import AdapterRegistry, MODEL_ADAPTER_MAP
from src.models.registry import ModelRegistry
from src.features.strategies import get_strategy_for_model


class Trainer:
    """Unified trainer - ALL data prep goes through adapters."""

    def _get_adapter_for_model(self, model_name: str):
        """Get appropriate adapter for model."""
        adapter_name = MODEL_ADAPTER_MAP.get(model_name)
        if not adapter_name:
            raise ValueError(f"No adapter mapping for model: {model_name}")
        return adapter_name

    def _prepare_data(self, model_name: str, df: pd.DataFrame) -> AdapterResult:
        """
        ALL data prep goes through adapters - NO BYPASS.

        Args:
            model_name: Name of the model
            df: Raw feature DataFrame

        Returns:
            AdapterResult with properly shaped data
        """
        # Get model's data contract
        model_cls = ModelRegistry.get(model_name)
        contract = model_cls.data_contract

        # Get feature strategy
        strategy = get_strategy_for_model(model_name)
        feature_cols = FeatureRegistry.get_by_families(strategy.baseline_families)

        # Get adapter
        adapter_name = self._get_adapter_for_model(model_name)
        AdapterClass = AdapterRegistry.get(adapter_name)

        # Configure adapter
        adapter_kwargs = {"feature_columns": feature_cols}
        if adapter_name == "sequence":
            adapter_kwargs["sequence_length"] = getattr(self.config, "sequence_length", 60)
        elif adapter_name == "multi_stream":
            adapter_kwargs["sequence_length"] = getattr(self.config, "sequence_length", 60)
            adapter_kwargs["timeframes"] = getattr(self.config, "mtf_timeframes", ["1min", "5min", "15min"])

        adapter = AdapterClass(**adapter_kwargs)

        return adapter.transform(df, contract)

    def train(self, model_name: str, df: pd.DataFrame, labels: np.ndarray):
        """Train model using adapter-prepared data."""
        # Prepare data via adapter
        result = self._prepare_data(model_name, df)

        # Split using original indices for proper alignment
        X_train, X_val = self._split(result.data, result.original_indices)
        y_train, y_val = self._split(labels, result.original_indices)

        # Train model
        model = ModelRegistry.get(model_name)()
        return model.fit(X_train, y_train, X_val, y_val)
```

---

## Task 2.7: Heterogeneous Data Preparation

For ensembles with mixed model families:

```python
def _prepare_heterogeneous_data(
    self,
    base_models: List[str],
    df: pd.DataFrame
) -> Dict[str, AdapterResult]:
    """
    Prepare different data formats for heterogeneous ensemble.

    Each base model gets its own adapter and data format.
    """
    results = {}
    for model_name in base_models:
        results[model_name] = self._prepare_data(model_name, df)
    return results


def train_heterogeneous_ensemble(
    self,
    base_models: List[str],
    meta_learner: str,
    df: pd.DataFrame,
    labels: np.ndarray
):
    """Train ensemble with different adapters per base model."""
    # Prepare data for each base
    data_results = self._prepare_heterogeneous_data(base_models, df)

    # Generate OOF for each base
    oof_results = {}
    for model_name, adapter_result in data_results.items():
        oof_results[model_name] = self._generate_oof(
            model_name, adapter_result.data, labels[adapter_result.original_indices]
        )

    # Align OOF predictions
    aligned_oof, common_idx = self._align_oof(oof_results)

    # Train meta-learner on aligned OOF
    meta = ModelRegistry.get(meta_learner)()
    return meta.fit(aligned_oof, labels[common_idx])
```

---

## Task 2.8: Delete Legacy Code

After adapter integration is complete, delete:

```bash
# Delete legacy data preparation
rm src/models/data_preparation.py

# Remove any direct numpy conversions in trainer.py
# that bypass the adapter system
```

---

## Implementation Checklist

### Task 2.1: Adapter Registry
- [ ] Create `src/adapters/registry.py`
- [ ] `@register` decorator
- [ ] `get()` and `list_all()` methods

### Task 2.2: Model-Adapter Mapping
- [ ] Define `MODEL_ADAPTER_MAP` constant
- [ ] Map all 22 models to adapters

### Task 2.3: TabularAdapter
- [ ] Create `src/adapters/tabular.py`
- [ ] `transform()` method
- [ ] Feature bounds validation

### Task 2.4: SequenceAdapter
- [ ] Create `src/adapters/sequence.py`
- [ ] `_build_sequences()` method
- [ ] Index tracking for OOF alignment

### Task 2.5: MultiStreamAdapter
- [ ] Create `src/adapters/multi_stream.py`
- [ ] Multi-TF DataFrame input
- [ ] 4D tensor output

### Task 2.6: Trainer Integration
- [ ] `_get_adapter_for_model()`
- [ ] `_prepare_data()` - NO BYPASS
- [ ] Update `train()` to use adapters

### Task 2.7: Heterogeneous Support
- [ ] `_prepare_heterogeneous_data()`
- [ ] `train_heterogeneous_ensemble()`

### Task 2.8: Cleanup
- [ ] Delete `src/models/data_preparation.py`
- [ ] Remove bypass paths in trainer

---

## Data Flow Summary

```
DataFrame (features)
       │
       ▼
┌─────────────────────────────────────┐
│  MODEL_ADAPTER_MAP[model_name]      │
│  "xgboost" → "tabular"              │
│  "lstm" → "sequence"                │
│  "patchtst" → "multi_stream"        │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  AdapterRegistry.get(adapter_name)  │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  adapter.transform(df, contract)    │
│  Returns: AdapterResult             │
│    - data: 2D/3D/4D array           │
│    - original_indices: for OOF      │
│    - feature_names: columns used    │
└─────────────────────────────────────┘
       │
       ▼
     Model.fit(X, y)
```
