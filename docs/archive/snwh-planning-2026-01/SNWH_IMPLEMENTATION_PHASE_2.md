# SNwH Implementation: Phase 2 - Adapter Architecture

## Overview

Phase 2 implements the adapter system that automatically converts canonical data into model-specific formats. The adapters handle:
1. **TabularAdapter** - 2D arrays for boosting/classical models
2. **SequenceAdapter** - 3D sequences for LSTM/GRU/TCN
3. **MultiStreamAdapter** - 4D multi-timeframe tensors for transformers

---

## 2.1 Adapter Base Class and Registry

### New File: `src/adapters/__init__.py`

```python
"""
Adapters package - Convert canonical data to model-specific formats.
"""

from .registry import AdapterRegistry, get_adapter
from .base import BaseAdapter, AdapterResult
from .tabular import TabularAdapter
from .sequence import SequenceAdapter
from .multi_stream import MultiStreamAdapter

__all__ = [
    "AdapterRegistry",
    "get_adapter",
    "BaseAdapter",
    "AdapterResult",
    "TabularAdapter",
    "SequenceAdapter",
    "MultiStreamAdapter",
]
```

### New File: `src/adapters/base.py`

```python
"""
Base adapter class and result dataclass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.contracts import DataContract, ModelContract, DataRank


@dataclass
class AdapterResult:
    """
    Result from adapter transformation.

    Contains the transformed data arrays plus metadata
    for validation and debugging.
    """
    # Data arrays
    X: np.ndarray
    y: np.ndarray
    weights: np.ndarray | None = None

    # Shape info
    n_samples: int = 0
    n_features: int = 0
    data_rank: DataRank = DataRank.TABULAR_2D

    # For sequences
    sequence_length: int | None = None
    original_indices: np.ndarray | None = None  # Maps back to source DataFrame

    # For multi-stream
    n_timeframes: int | None = None
    timeframe_names: list[str] = field(default_factory=list)

    # Metadata
    feature_columns: list[str] = field(default_factory=list)
    data_contract: DataContract | None = None
    adapter_name: str = ""

    def __post_init__(self):
        """Compute derived fields."""
        if self.n_samples == 0:
            self.n_samples = self.X.shape[0]
        if self.n_features == 0:
            if self.X.ndim == 2:
                self.n_features = self.X.shape[1]
            elif self.X.ndim == 3:
                self.n_features = self.X.shape[2]
            elif self.X.ndim == 4:
                self.n_features = self.X.shape[3]

    def validate(self) -> tuple[bool, list[str]]:
        """Validate adapter result."""
        issues = []

        # Check X shape matches declared rank
        if self.X.ndim != self.data_rank.value:
            issues.append(
                f"X rank mismatch: expected {self.data_rank.value}D, got {self.X.ndim}D"
            )

        # Check y length
        if self.y.shape[0] != self.X.shape[0]:
            issues.append(
                f"y length mismatch: X has {self.X.shape[0]} samples, "
                f"y has {self.y.shape[0]}"
            )

        # Check weights if present
        if self.weights is not None and self.weights.shape[0] != self.X.shape[0]:
            issues.append(
                f"weights length mismatch: X has {self.X.shape[0]} samples, "
                f"weights has {self.weights.shape[0]}"
            )

        # Check for NaN/Inf
        if np.isnan(self.X).any():
            n_nan = int(np.isnan(self.X).sum())
            issues.append(f"X contains {n_nan} NaN values")
        if np.isinf(self.X).any():
            n_inf = int(np.isinf(self.X).sum())
            issues.append(f"X contains {n_inf} Inf values")

        return len(issues) == 0, issues


class BaseAdapter(ABC):
    """
    Base class for data adapters.

    Adapters transform canonical DataFrames into model-specific
    numpy array formats (2D, 3D, or 4D).
    """

    # Adapter identity
    adapter_id: str = "base"
    output_rank: DataRank = DataRank.TABULAR_2D

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        label_column: str = "label_h20",
        weight_column: str | None = "sample_weight_h20",
    ):
        """
        Initialize adapter.

        Args:
            feature_columns: Feature columns to use (None = auto-detect)
            label_column: Label column name
            weight_column: Weight column name (None = uniform weights)
        """
        self.feature_columns = feature_columns
        self.label_column = label_column
        self.weight_column = weight_column

    @abstractmethod
    def transform(
        self,
        df: pd.DataFrame,
        model_contract: ModelContract | None = None,
    ) -> AdapterResult:
        """
        Transform DataFrame to model-specific format.

        Args:
            df: Source DataFrame with features, labels, weights
            model_contract: Optional contract for validation

        Returns:
            AdapterResult with transformed arrays
        """
        pass

    def validate_input(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        """
        Validate input DataFrame.

        Args:
            df: DataFrame to validate

        Returns:
            (is_valid, list_of_issues)
        """
        issues = []

        # Check label column
        if self.label_column not in df.columns:
            issues.append(f"Missing label column: {self.label_column}")

        # Check feature columns if specified
        if self.feature_columns:
            missing = set(self.feature_columns) - set(df.columns)
            if missing:
                issues.append(f"Missing feature columns: {sorted(missing)[:10]}")

        return len(issues) == 0, issues

    def _get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Get feature columns (explicit or auto-detected)."""
        if self.feature_columns:
            return self.feature_columns

        # Auto-detect: exclude metadata and labels
        from src.phase1.utils.constants import METADATA_COLUMNS
        from src.phase1.utils.feature_sets import _is_label_column

        return [
            col for col in df.columns
            if col not in METADATA_COLUMNS and not _is_label_column(col)
        ]

    def _get_weights(self, df: pd.DataFrame) -> np.ndarray | None:
        """Get sample weights from DataFrame."""
        if self.weight_column and self.weight_column in df.columns:
            return df[self.weight_column].values
        return None
```

### New File: `src/adapters/registry.py`

```python
"""
Adapter Registry - Maps model requirements to adapters.
"""

from __future__ import annotations

import logging
from typing import Any

from src.contracts import ModelContract, get_model_contract, DataRank

from .base import BaseAdapter

logger = logging.getLogger(__name__)


class AdapterRegistry:
    """
    Registry for data adapters.

    Maps adapter IDs to adapter classes and provides
    automatic routing based on model contracts.
    """

    _adapters: dict[str, type[BaseAdapter]] = {}

    @classmethod
    def register(cls, adapter_id: str):
        """
        Decorator to register an adapter class.

        Args:
            adapter_id: Unique adapter identifier

        Example:
            @AdapterRegistry.register("tabular")
            class TabularAdapter(BaseAdapter):
                pass
        """
        def decorator(adapter_class: type[BaseAdapter]) -> type[BaseAdapter]:
            if adapter_id in cls._adapters:
                raise ValueError(f"Adapter '{adapter_id}' already registered")
            cls._adapters[adapter_id] = adapter_class
            adapter_class.adapter_id = adapter_id
            logger.debug(f"Registered adapter: {adapter_id}")
            return adapter_class
        return decorator

    @classmethod
    def get(cls, adapter_id: str) -> type[BaseAdapter]:
        """
        Get adapter class by ID.

        Args:
            adapter_id: Adapter identifier

        Returns:
            Adapter class

        Raises:
            ValueError: If adapter not found
        """
        if adapter_id not in cls._adapters:
            raise ValueError(
                f"Unknown adapter '{adapter_id}'. "
                f"Available: {sorted(cls._adapters.keys())}"
            )
        return cls._adapters[adapter_id]

    @classmethod
    def create(
        cls,
        adapter_id: str,
        **kwargs: Any,
    ) -> BaseAdapter:
        """
        Create adapter instance by ID.

        Args:
            adapter_id: Adapter identifier
            **kwargs: Arguments to pass to adapter constructor

        Returns:
            Adapter instance
        """
        adapter_class = cls.get(adapter_id)
        return adapter_class(**kwargs)

    @classmethod
    def get_for_model(
        cls,
        model_name: str,
        **kwargs: Any,
    ) -> BaseAdapter:
        """
        Get adapter for a model based on its contract.

        Args:
            model_name: Name of the model
            **kwargs: Arguments to pass to adapter constructor

        Returns:
            Adapter instance appropriate for the model
        """
        contract = get_model_contract(model_name)
        return cls.create(contract.adapter_id, **kwargs)

    @classmethod
    def get_for_contract(
        cls,
        contract: ModelContract,
        **kwargs: Any,
    ) -> BaseAdapter:
        """
        Get adapter for a model contract.

        Args:
            contract: ModelContract instance
            **kwargs: Arguments to pass to adapter constructor

        Returns:
            Adapter instance appropriate for the contract
        """
        return cls.create(contract.adapter_id, **kwargs)

    @classmethod
    def list_adapters(cls) -> list[str]:
        """List all registered adapter IDs."""
        return sorted(cls._adapters.keys())


def get_adapter(
    model_name: str | None = None,
    adapter_id: str | None = None,
    **kwargs: Any,
) -> BaseAdapter:
    """
    Convenience function to get an adapter.

    Either model_name or adapter_id must be provided.

    Args:
        model_name: Model name (uses contract to determine adapter)
        adapter_id: Direct adapter ID
        **kwargs: Arguments to pass to adapter constructor

    Returns:
        Adapter instance
    """
    if adapter_id:
        return AdapterRegistry.create(adapter_id, **kwargs)
    elif model_name:
        return AdapterRegistry.get_for_model(model_name, **kwargs)
    else:
        raise ValueError("Either model_name or adapter_id must be provided")


__all__ = [
    "AdapterRegistry",
    "get_adapter",
]
```

---

## 2.2 TabularAdapter (2D)

### New File: `src/adapters/tabular.py`

```python
"""
Tabular Adapter - Converts data to 2D arrays for boosting/classical models.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.contracts import ModelContract, DataContract, DataRank

from .base import BaseAdapter, AdapterResult
from .registry import AdapterRegistry

logger = logging.getLogger(__name__)


@AdapterRegistry.register("tabular")
class TabularAdapter(BaseAdapter):
    """
    Adapter for tabular (2D) models.

    Converts DataFrame to (n_samples, n_features) arrays.
    Used by: XGBoost, LightGBM, CatBoost, RandomForest, Logistic, SVM
    """

    adapter_id = "tabular"
    output_rank = DataRank.TABULAR_2D

    def transform(
        self,
        df: pd.DataFrame,
        model_contract: ModelContract | None = None,
    ) -> AdapterResult:
        """
        Transform DataFrame to 2D arrays.

        Args:
            df: Source DataFrame
            model_contract: Optional contract for validation

        Returns:
            AdapterResult with 2D arrays
        """
        # Validate input
        is_valid, issues = self.validate_input(df)
        if not is_valid:
            raise ValueError(f"Invalid input: {issues}")

        # Get feature columns
        feature_cols = self._get_feature_columns(df)

        # Validate feature count against contract
        if model_contract:
            if len(feature_cols) < model_contract.min_features:
                raise ValueError(
                    f"Too few features: {len(feature_cols)} < "
                    f"{model_contract.min_features} (min for {model_contract.model_name})"
                )
            if len(feature_cols) > model_contract.max_features:
                logger.warning(
                    f"Many features: {len(feature_cols)} > "
                    f"{model_contract.max_features} (max for {model_contract.model_name}). "
                    f"Consider feature selection."
                )

        # Extract arrays
        X = df[feature_cols].values.astype(np.float32)
        y = df[self.label_column].values.astype(np.int64)
        weights = self._get_weights(df)

        # Create data contract
        data_contract = DataContract(
            symbol=df.get("symbol", pd.Series(["unknown"])).iloc[0] if "symbol" in df.columns else "unknown",
            timeframe=df.get("timeframe", pd.Series(["5min"])).iloc[0] if "timeframe" in df.columns else "5min",
            horizon=int(self.label_column.split("_h")[-1]) if "_h" in self.label_column else 20,
            split="unknown",
            n_samples=len(df),
            n_features=len(feature_cols),
            data_rank=DataRank.TABULAR_2D,
            feature_columns=feature_cols,
        )

        result = AdapterResult(
            X=X,
            y=y,
            weights=weights,
            n_samples=len(df),
            n_features=len(feature_cols),
            data_rank=DataRank.TABULAR_2D,
            feature_columns=feature_cols,
            data_contract=data_contract,
            adapter_name=self.adapter_id,
        )

        # Validate result
        is_valid, issues = result.validate()
        if not is_valid:
            raise ValueError(f"Adapter result validation failed: {issues}")

        logger.debug(
            f"TabularAdapter: transformed {len(df)} rows -> "
            f"X{X.shape}, y{y.shape}"
        )

        return result


__all__ = ["TabularAdapter"]
```

---

## 2.3 SequenceAdapter (3D)

### New File: `src/adapters/sequence.py`

```python
"""
Sequence Adapter - Converts data to 3D arrays for sequence models.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.contracts import ModelContract, DataContract, DataRank

from .base import BaseAdapter, AdapterResult
from .registry import AdapterRegistry

logger = logging.getLogger(__name__)


@AdapterRegistry.register("sequence")
class SequenceAdapter(BaseAdapter):
    """
    Adapter for sequence (3D) models.

    Converts DataFrame to (n_samples, seq_len, n_features) arrays.
    Used by: LSTM, GRU, TCN, Transformer, PatchTST, etc.
    """

    adapter_id = "sequence"
    output_rank = DataRank.SEQUENCE_3D

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        label_column: str = "label_h20",
        weight_column: str | None = "sample_weight_h20",
        sequence_length: int = 60,
        stride: int = 1,
        symbol_column: str | None = "symbol",
    ):
        """
        Initialize sequence adapter.

        Args:
            feature_columns: Feature columns to use
            label_column: Label column name
            weight_column: Weight column name
            sequence_length: Length of each sequence
            stride: Step between sequences
            symbol_column: Column for symbol isolation (no cross-symbol sequences)
        """
        super().__init__(feature_columns, label_column, weight_column)
        self.sequence_length = sequence_length
        self.stride = stride
        self.symbol_column = symbol_column

    def transform(
        self,
        df: pd.DataFrame,
        model_contract: ModelContract | None = None,
    ) -> AdapterResult:
        """
        Transform DataFrame to 3D sequences.

        Sequences are created with sliding windows. The label for each
        sequence is the label at the LAST timestep of the window.

        Args:
            df: Source DataFrame (must be sorted by time)
            model_contract: Optional contract for validation

        Returns:
            AdapterResult with 3D arrays (n_sequences, seq_len, n_features)
        """
        # Validate input
        is_valid, issues = self.validate_input(df)
        if not is_valid:
            raise ValueError(f"Invalid input: {issues}")

        # Get feature columns
        feature_cols = self._get_feature_columns(df)

        # Override sequence length from contract if provided
        seq_len = self.sequence_length
        if model_contract and model_contract.sequence_length:
            seq_len = model_contract.sequence_length

        # Build sequences with symbol isolation
        if self.symbol_column and self.symbol_column in df.columns:
            # Build sequences per symbol, then concatenate
            all_X = []
            all_y = []
            all_weights = []
            all_indices = []

            for symbol in df[self.symbol_column].unique():
                symbol_df = df[df[self.symbol_column] == symbol].reset_index(drop=True)
                X_seq, y_seq, w_seq, indices = self._build_sequences(
                    symbol_df, feature_cols, seq_len
                )
                if len(X_seq) > 0:
                    all_X.append(X_seq)
                    all_y.append(y_seq)
                    if w_seq is not None:
                        all_weights.append(w_seq)
                    all_indices.append(indices)

            if not all_X:
                raise ValueError(
                    f"No valid sequences created. "
                    f"DataFrame has {len(df)} rows, seq_len={seq_len}"
                )

            X = np.concatenate(all_X, axis=0)
            y = np.concatenate(all_y, axis=0)
            weights = np.concatenate(all_weights, axis=0) if all_weights else None
            original_indices = np.concatenate(all_indices, axis=0)
        else:
            # No symbol isolation
            X, y, weights, original_indices = self._build_sequences(
                df, feature_cols, seq_len
            )

        # Validate feature count
        if model_contract:
            if len(feature_cols) < model_contract.min_features:
                raise ValueError(
                    f"Too few features: {len(feature_cols)} < "
                    f"{model_contract.min_features}"
                )

        # Create data contract
        symbol = "unknown"
        if self.symbol_column and self.symbol_column in df.columns:
            symbol = str(df[self.symbol_column].iloc[0])

        data_contract = DataContract(
            symbol=symbol,
            timeframe=df.get("timeframe", pd.Series(["5min"])).iloc[0] if "timeframe" in df.columns else "5min",
            horizon=int(self.label_column.split("_h")[-1]) if "_h" in self.label_column else 20,
            split="unknown",
            n_samples=X.shape[0],
            n_features=len(feature_cols),
            data_rank=DataRank.SEQUENCE_3D,
            sequence_length=seq_len,
            feature_columns=feature_cols,
        )

        result = AdapterResult(
            X=X,
            y=y,
            weights=weights,
            n_samples=X.shape[0],
            n_features=len(feature_cols),
            data_rank=DataRank.SEQUENCE_3D,
            sequence_length=seq_len,
            original_indices=original_indices,
            feature_columns=feature_cols,
            data_contract=data_contract,
            adapter_name=self.adapter_id,
        )

        # Validate result
        is_valid, issues = result.validate()
        if not is_valid:
            raise ValueError(f"Adapter result validation failed: {issues}")

        logger.debug(
            f"SequenceAdapter: transformed {len(df)} rows -> "
            f"X{X.shape} (seq_len={seq_len})"
        )

        return result

    def _build_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        seq_len: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
        """
        Build sequences from a DataFrame.

        Args:
            df: Source DataFrame
            feature_cols: Feature column names
            seq_len: Sequence length

        Returns:
            (X, y, weights, original_indices)
        """
        n_rows = len(df)

        if n_rows < seq_len:
            return (
                np.empty((0, seq_len, len(feature_cols)), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                None,
                np.empty((0,), dtype=np.int64),
            )

        # Number of sequences
        n_sequences = (n_rows - seq_len) // self.stride + 1

        # Pre-allocate arrays
        X = np.empty((n_sequences, seq_len, len(feature_cols)), dtype=np.float32)
        y = np.empty(n_sequences, dtype=np.int64)
        original_indices = np.empty(n_sequences, dtype=np.int64)

        # Get feature and label data
        features = df[feature_cols].values.astype(np.float32)
        labels = df[self.label_column].values.astype(np.int64)

        # Build sequences
        for i in range(n_sequences):
            start_idx = i * self.stride
            end_idx = start_idx + seq_len
            X[i] = features[start_idx:end_idx]
            y[i] = labels[end_idx - 1]  # Label at last timestep
            original_indices[i] = end_idx - 1

        # Get weights if available
        weights = None
        if self.weight_column and self.weight_column in df.columns:
            all_weights = df[self.weight_column].values
            weights = all_weights[original_indices].astype(np.float32)

        return X, y, weights, original_indices


__all__ = ["SequenceAdapter"]
```

---

## 2.4 MultiStreamAdapter (4D)

### New File: `src/adapters/multi_stream.py`

```python
"""
Multi-Stream Adapter - Converts data to 4D arrays for multi-timeframe models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.contracts import ModelContract, DataContract, DataRank
from src.common.timeframes import normalize_timeframe

from .base import BaseAdapter, AdapterResult
from .registry import AdapterRegistry

logger = logging.getLogger(__name__)


@AdapterRegistry.register("multi_stream")
class MultiStreamAdapter(BaseAdapter):
    """
    Adapter for multi-timeframe (4D) models.

    Converts multiple timeframe DataFrames to (n_samples, n_tfs, seq_len, n_features).
    Used by: Transformer, PatchTST, iTransformer with multi-stream MTF mode.

    Each timeframe stream has the same features (raw OHLCV typically).
    Streams are aligned to the anchor (smallest) timeframe's timestamps.
    """

    adapter_id = "multi_stream"
    output_rank = DataRank.MULTI_TF_4D

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        label_column: str = "label_h20",
        weight_column: str | None = "sample_weight_h20",
        sequence_length: int = 60,
        stride: int = 1,
        timeframes: list[str] | None = None,
        data_dir: Path | str | None = None,
    ):
        """
        Initialize multi-stream adapter.

        Args:
            feature_columns: Feature columns per timeframe (default: OHLCV)
            label_column: Label column name
            weight_column: Weight column name
            sequence_length: Length of each sequence per timeframe
            stride: Step between sequences
            timeframes: List of timeframes to include (e.g., ["1min", "5min", "15min"])
            data_dir: Directory containing timeframe data files
        """
        # Default to raw OHLCV for multi-stream
        if feature_columns is None:
            feature_columns = ["open", "high", "low", "close", "volume"]

        super().__init__(feature_columns, label_column, weight_column)
        self.sequence_length = sequence_length
        self.stride = stride
        self.timeframes = [normalize_timeframe(tf) for tf in (timeframes or ["1min", "5min", "15min"])]
        self.data_dir = Path(data_dir) if data_dir else None

    def transform(
        self,
        df: pd.DataFrame,
        model_contract: ModelContract | None = None,
        additional_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> AdapterResult:
        """
        Transform multiple timeframe DataFrames to 4D array.

        Args:
            df: Primary timeframe DataFrame
            model_contract: Optional contract for validation
            additional_dfs: Dict mapping timeframe -> DataFrame for other TFs

        Returns:
            AdapterResult with 4D array (n_samples, n_tfs, seq_len, n_features)
        """
        # Override timeframes from contract if provided
        timeframes = self.timeframes
        if model_contract and model_contract.mtf_timeframes:
            timeframes = [model_contract.primary_timeframe] + list(model_contract.mtf_timeframes)
            timeframes = [normalize_timeframe(tf) for tf in timeframes]

        # Override sequence length from contract
        seq_len = self.sequence_length
        if model_contract and model_contract.sequence_length:
            seq_len = model_contract.sequence_length

        # Collect DataFrames for each timeframe
        tf_dfs: dict[str, pd.DataFrame] = {}
        anchor_tf = timeframes[0]  # Smallest timeframe is anchor

        # Primary DataFrame is for anchor timeframe
        tf_dfs[anchor_tf] = df

        # Load or use provided DataFrames for other timeframes
        additional_dfs = additional_dfs or {}
        for tf in timeframes[1:]:
            if tf in additional_dfs:
                tf_dfs[tf] = additional_dfs[tf]
            elif self.data_dir:
                # Try to load from data directory
                tf_path = self.data_dir / f"features_{tf}.parquet"
                if tf_path.exists():
                    tf_dfs[tf] = pd.read_parquet(tf_path)
                else:
                    raise FileNotFoundError(
                        f"Timeframe data not found: {tf_path}. "
                        f"Provide via additional_dfs or ensure file exists."
                    )
            else:
                raise ValueError(
                    f"No data for timeframe '{tf}'. "
                    f"Provide via additional_dfs or set data_dir."
                )

        # Validate all timeframes have required columns
        feature_cols = self._get_feature_columns(df)
        for tf, tf_df in tf_dfs.items():
            missing = set(feature_cols) - set(tf_df.columns)
            if missing:
                raise ValueError(
                    f"Timeframe '{tf}' missing features: {sorted(missing)[:5]}"
                )

        # Build 4D array
        X, y, weights, original_indices = self._build_multi_stream(
            tf_dfs, timeframes, feature_cols, seq_len
        )

        # Create data contract
        data_contract = DataContract(
            symbol=df.get("symbol", pd.Series(["unknown"])).iloc[0] if "symbol" in df.columns else "unknown",
            timeframe=anchor_tf,
            horizon=int(self.label_column.split("_h")[-1]) if "_h" in self.label_column else 20,
            split="unknown",
            n_samples=X.shape[0],
            n_features=len(feature_cols),
            data_rank=DataRank.MULTI_TF_4D,
            sequence_length=seq_len,
            n_timeframes=len(timeframes),
            feature_columns=feature_cols,
        )

        result = AdapterResult(
            X=X,
            y=y,
            weights=weights,
            n_samples=X.shape[0],
            n_features=len(feature_cols),
            data_rank=DataRank.MULTI_TF_4D,
            sequence_length=seq_len,
            n_timeframes=len(timeframes),
            timeframe_names=timeframes,
            original_indices=original_indices,
            feature_columns=feature_cols,
            data_contract=data_contract,
            adapter_name=self.adapter_id,
        )

        # Validate
        is_valid, issues = result.validate()
        if not is_valid:
            raise ValueError(f"Adapter result validation failed: {issues}")

        logger.debug(
            f"MultiStreamAdapter: transformed to X{X.shape} "
            f"(tfs={timeframes}, seq_len={seq_len})"
        )

        return result

    def _build_multi_stream(
        self,
        tf_dfs: dict[str, pd.DataFrame],
        timeframes: list[str],
        feature_cols: list[str],
        seq_len: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
        """
        Build 4D array from multiple timeframe DataFrames.

        The anchor (smallest) timeframe determines sample count and alignment.
        Higher timeframes are resampled/aligned to anchor timestamps.

        Args:
            tf_dfs: Dict mapping timeframe -> DataFrame
            timeframes: Ordered list of timeframes
            feature_cols: Feature columns
            seq_len: Sequence length

        Returns:
            (X, y, weights, original_indices) where X is 4D
        """
        anchor_tf = timeframes[0]
        anchor_df = tf_dfs[anchor_tf]
        n_rows = len(anchor_df)
        n_tfs = len(timeframes)
        n_features = len(feature_cols)

        if n_rows < seq_len:
            return (
                np.empty((0, n_tfs, seq_len, n_features), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                None,
                np.empty((0,), dtype=np.int64),
            )

        n_sequences = (n_rows - seq_len) // self.stride + 1

        # Pre-allocate 4D array
        X = np.empty((n_sequences, n_tfs, seq_len, n_features), dtype=np.float32)
        y = np.empty(n_sequences, dtype=np.int64)
        original_indices = np.empty(n_sequences, dtype=np.int64)

        # Get anchor features and labels
        anchor_features = anchor_df[feature_cols].values.astype(np.float32)
        anchor_labels = anchor_df[self.label_column].values.astype(np.int64)

        # Build sequences for anchor timeframe
        for i in range(n_sequences):
            start_idx = i * self.stride
            end_idx = start_idx + seq_len
            X[i, 0] = anchor_features[start_idx:end_idx]
            y[i] = anchor_labels[end_idx - 1]
            original_indices[i] = end_idx - 1

        # Build sequences for other timeframes
        # For higher timeframes, we need to align to anchor timestamps
        # This is a simplified version - proper alignment requires timestamp matching
        for tf_idx, tf in enumerate(timeframes[1:], start=1):
            tf_df = tf_dfs[tf]
            tf_features = tf_df[feature_cols].values.astype(np.float32)

            # Calculate downsampling ratio (approximate)
            from src.common.timeframes import get_timeframe_minutes
            anchor_minutes = get_timeframe_minutes(anchor_tf)
            tf_minutes = get_timeframe_minutes(tf)
            ratio = tf_minutes // anchor_minutes

            for i in range(n_sequences):
                start_idx = i * self.stride
                end_idx = start_idx + seq_len

                # Map to higher timeframe indices
                tf_start = start_idx // ratio
                tf_end = tf_start + seq_len

                # Ensure we don't go out of bounds
                if tf_end <= len(tf_features):
                    X[i, tf_idx] = tf_features[tf_start:tf_start + seq_len]
                else:
                    # Pad with last available values
                    available = len(tf_features) - tf_start
                    if available > 0:
                        X[i, tf_idx, :available] = tf_features[tf_start:]
                        X[i, tf_idx, available:] = tf_features[-1]
                    else:
                        X[i, tf_idx] = tf_features[-seq_len:]

        # Get weights
        weights = None
        if self.weight_column and self.weight_column in anchor_df.columns:
            all_weights = anchor_df[self.weight_column].values
            weights = all_weights[original_indices].astype(np.float32)

        return X, y, weights, original_indices


__all__ = ["MultiStreamAdapter"]
```

---

## Summary: Phase 2 Changes

| File | Type | Purpose |
|------|------|---------|
| `src/adapters/__init__.py` | NEW | Package init with exports |
| `src/adapters/base.py` | NEW | BaseAdapter, AdapterResult |
| `src/adapters/registry.py` | NEW | AdapterRegistry, get_adapter() |
| `src/adapters/tabular.py` | NEW | TabularAdapter (2D) |
| `src/adapters/sequence.py` | NEW | SequenceAdapter (3D) |
| `src/adapters/multi_stream.py` | NEW | MultiStreamAdapter (4D) |

## Dependencies

- Phase 0 (contracts) for DataContract, ModelContract, DataRank
- Phase 1 (config) for model contract lookups

## Usage Example

```python
from src.adapters import get_adapter, AdapterRegistry

# Get adapter for a model (uses contract)
adapter = get_adapter(model_name="xgboost")
result = adapter.transform(df)
# result.X.shape = (n_samples, n_features)

# Get adapter for sequence model
adapter = get_adapter(model_name="lstm", sequence_length=60)
result = adapter.transform(df)
# result.X.shape = (n_sequences, 60, n_features)

# Get adapter for multi-stream model
adapter = get_adapter(
    model_name="patchtst",
    timeframes=["1min", "5min", "15min"],
    data_dir="data/splits/scaled",
)
result = adapter.transform(df, additional_dfs={"5min": df_5min, "15min": df_15min})
# result.X.shape = (n_sequences, 3, seq_len, n_features)
```

## Next Steps

After Phase 2 is implemented, proceed to Phase 3 (Timeframe Coordination) which will:
1. Create TimeframeCoordinator for managing multi-TF data loading
2. Ensure proper alignment between timeframes
3. Wire into trainer.py for automatic data routing
