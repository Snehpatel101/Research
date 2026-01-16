# SNwH Implementation: Phase 0 - Canonical Contracts

## Overview

Phase 0 establishes the foundational contracts that all subsequent phases depend on. These contracts define:
1. **DataContract** - Schema for canonical data flowing through the pipeline
2. **ModelContract** - Input requirements each model declares at registration
3. **ArtifactManifest** - Safety and reproducibility metadata for saved artifacts

---

## 0.1 DataContract (Canonical Data Schema)

### New File: `src/contracts/data_contract.py`

```python
"""
Canonical Data Contract - Schema for all data flowing through the pipeline.

Every adapter, trainer, and validator uses this contract to ensure
data consistency and traceability.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class DataRank(int, Enum):
    """Data dimensionality rank."""
    TABULAR_2D = 2   # (n_samples, n_features)
    SEQUENCE_3D = 3  # (n_samples, seq_len, n_features)
    MULTI_TF_4D = 4  # (n_samples, n_timeframes, seq_len, n_features)


class FeatureMode(str, Enum):
    """Feature generation mode."""
    ENGINEERED = "engineered"  # Pre-computed indicators (~180 features)
    RAW = "raw"                # Raw OHLCV only (4-5 features)
    HYBRID = "hybrid"          # Mix of raw + selected indicators


class MTFMode(str, Enum):
    """Multi-timeframe mode."""
    NONE = "none"              # Single timeframe, no MTF
    INDICATORS = "indicators"  # MTF indicator features added to primary TF
    MULTI_STREAM = "multi_stream"  # Multiple TF streams (4D input)


@dataclass(frozen=True)
class DataContractSchema:
    """
    Schema definition for canonical data.

    This defines the expected columns and their types for data
    at various stages of the pipeline.
    """
    # Required OHLCV columns
    REQUIRED_OHLCV: tuple[str, ...] = (
        "timestamp", "open", "high", "low", "close", "volume"
    )

    # Label column pattern
    LABEL_PATTERN: str = "label_h{horizon}"
    WEIGHT_PATTERN: str = "sample_weight_h{horizon}"
    LABEL_END_TIME_PATTERN: str = "label_end_time_h{horizon}"

    # Metadata columns (not features)
    METADATA_COLUMNS: tuple[str, ...] = (
        "timestamp", "datetime", "symbol", "timeframe",
        "feature_version", "label_version", "split"
    )

    def get_label_column(self, horizon: int) -> str:
        return self.LABEL_PATTERN.format(horizon=horizon)

    def get_weight_column(self, horizon: int) -> str:
        return self.WEIGHT_PATTERN.format(horizon=horizon)

    def get_label_end_time_column(self, horizon: int) -> str:
        return self.LABEL_END_TIME_PATTERN.format(horizon=horizon)


# Singleton schema instance
DATA_SCHEMA = DataContractSchema()


@dataclass
class DataContract:
    """
    Canonical data contract for pipeline data.

    This contract captures:
    - Data shape and type information
    - Source lineage (symbol, timeframe, pipeline run)
    - Feature and label metadata
    - Schema hash for validation

    Every data artifact stores this contract alongside the data
    to ensure traceability and compatibility.
    """
    # Identity
    symbol: str
    timeframe: str
    horizon: int
    split: str  # "train", "val", "test"

    # Shape
    n_samples: int
    n_features: int
    data_rank: DataRank
    sequence_length: int | None = None  # For 3D/4D data
    n_timeframes: int | None = None     # For 4D data

    # Feature metadata
    feature_mode: FeatureMode = FeatureMode.ENGINEERED
    feature_columns: list[str] = field(default_factory=list)
    feature_version: str = "v1"

    # Label metadata
    label_column: str = ""
    weight_column: str = ""
    label_version: str = "v1"
    n_classes: int = 3  # short, neutral, long

    # Lineage
    pipeline_run_id: str = ""
    source_file: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Schema hash (computed)
    schema_hash: str = ""

    def __post_init__(self):
        """Compute schema hash and set defaults."""
        if not self.label_column:
            self.label_column = DATA_SCHEMA.get_label_column(self.horizon)
        if not self.weight_column:
            self.weight_column = DATA_SCHEMA.get_weight_column(self.horizon)
        if not self.schema_hash:
            self.schema_hash = self._compute_schema_hash()

    def _compute_schema_hash(self) -> str:
        """Compute deterministic hash of schema-defining fields."""
        schema_dict = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "horizon": self.horizon,
            "data_rank": self.data_rank.value,
            "n_features": self.n_features,
            "sequence_length": self.sequence_length,
            "n_timeframes": self.n_timeframes,
            "feature_mode": self.feature_mode.value,
            "feature_columns": sorted(self.feature_columns),
            "feature_version": self.feature_version,
            "label_version": self.label_version,
            "n_classes": self.n_classes,
        }
        schema_str = json.dumps(schema_dict, sort_keys=True)
        return hashlib.sha256(schema_str.encode()).hexdigest()[:16]

    def validate_dataframe(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        """
        Validate a DataFrame against this contract.

        Returns:
            (is_valid, list_of_issues)
        """
        issues = []

        # Check sample count
        if len(df) != self.n_samples:
            issues.append(
                f"Sample count mismatch: expected {self.n_samples}, got {len(df)}"
            )

        # Check feature columns exist
        missing_features = set(self.feature_columns) - set(df.columns)
        if missing_features:
            issues.append(
                f"Missing feature columns: {sorted(missing_features)[:10]}"
            )

        # Check label column
        if self.label_column not in df.columns:
            issues.append(f"Missing label column: {self.label_column}")

        # Check weight column (warning only)
        if self.weight_column and self.weight_column not in df.columns:
            issues.append(f"Missing weight column: {self.weight_column}")

        return len(issues) == 0, issues

    def validate_array(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None
    ) -> tuple[bool, list[str]]:
        """
        Validate numpy arrays against this contract.

        Returns:
            (is_valid, list_of_issues)
        """
        issues = []

        # Check rank
        if X.ndim != self.data_rank.value:
            issues.append(
                f"Rank mismatch: expected {self.data_rank.value}D, got {X.ndim}D"
            )
            return False, issues

        # Check shape based on rank
        if self.data_rank == DataRank.TABULAR_2D:
            if X.shape[0] != self.n_samples:
                issues.append(
                    f"Sample count mismatch: expected {self.n_samples}, got {X.shape[0]}"
                )
            if X.shape[1] != self.n_features:
                issues.append(
                    f"Feature count mismatch: expected {self.n_features}, got {X.shape[1]}"
                )

        elif self.data_rank == DataRank.SEQUENCE_3D:
            # (n_samples, seq_len, n_features)
            if X.shape[1] != self.sequence_length:
                issues.append(
                    f"Sequence length mismatch: expected {self.sequence_length}, got {X.shape[1]}"
                )
            if X.shape[2] != self.n_features:
                issues.append(
                    f"Feature count mismatch: expected {self.n_features}, got {X.shape[2]}"
                )

        elif self.data_rank == DataRank.MULTI_TF_4D:
            # (n_samples, n_timeframes, seq_len, n_features)
            if X.shape[1] != self.n_timeframes:
                issues.append(
                    f"Timeframe count mismatch: expected {self.n_timeframes}, got {X.shape[1]}"
                )
            if X.shape[2] != self.sequence_length:
                issues.append(
                    f"Sequence length mismatch: expected {self.sequence_length}, got {X.shape[2]}"
                )
            if X.shape[3] != self.n_features:
                issues.append(
                    f"Feature count mismatch: expected {self.n_features}, got {X.shape[3]}"
                )

        # Validate y if provided
        if y is not None:
            if y.shape[0] != X.shape[0]:
                issues.append(
                    f"Label count mismatch: X has {X.shape[0]} samples, y has {y.shape[0]}"
                )

        return len(issues) == 0, issues

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "horizon": self.horizon,
            "split": self.split,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "data_rank": self.data_rank.value,
            "sequence_length": self.sequence_length,
            "n_timeframes": self.n_timeframes,
            "feature_mode": self.feature_mode.value,
            "feature_columns": self.feature_columns,
            "feature_version": self.feature_version,
            "label_column": self.label_column,
            "weight_column": self.weight_column,
            "label_version": self.label_version,
            "n_classes": self.n_classes,
            "pipeline_run_id": self.pipeline_run_id,
            "source_file": self.source_file,
            "created_at": self.created_at,
            "schema_hash": self.schema_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataContract:
        """Deserialize from dictionary."""
        return cls(
            symbol=data["symbol"],
            timeframe=data["timeframe"],
            horizon=data["horizon"],
            split=data["split"],
            n_samples=data["n_samples"],
            n_features=data["n_features"],
            data_rank=DataRank(data["data_rank"]),
            sequence_length=data.get("sequence_length"),
            n_timeframes=data.get("n_timeframes"),
            feature_mode=FeatureMode(data.get("feature_mode", "engineered")),
            feature_columns=data.get("feature_columns", []),
            feature_version=data.get("feature_version", "v1"),
            label_column=data.get("label_column", ""),
            weight_column=data.get("weight_column", ""),
            label_version=data.get("label_version", "v1"),
            n_classes=data.get("n_classes", 3),
            pipeline_run_id=data.get("pipeline_run_id", ""),
            source_file=data.get("source_file", ""),
            created_at=data.get("created_at", ""),
            schema_hash=data.get("schema_hash", ""),
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        horizon: int,
        split: str,
        feature_columns: list[str],
        pipeline_run_id: str = "",
    ) -> DataContract:
        """Create contract from a DataFrame."""
        return cls(
            symbol=symbol,
            timeframe=timeframe,
            horizon=horizon,
            split=split,
            n_samples=len(df),
            n_features=len(feature_columns),
            data_rank=DataRank.TABULAR_2D,
            feature_columns=feature_columns,
            pipeline_run_id=pipeline_run_id,
        )

    @classmethod
    def from_array(
        cls,
        X: np.ndarray,
        symbol: str,
        timeframe: str,
        horizon: int,
        split: str,
        feature_columns: list[str] | None = None,
        pipeline_run_id: str = "",
    ) -> DataContract:
        """Create contract from numpy array."""
        if X.ndim == 2:
            return cls(
                symbol=symbol,
                timeframe=timeframe,
                horizon=horizon,
                split=split,
                n_samples=X.shape[0],
                n_features=X.shape[1],
                data_rank=DataRank.TABULAR_2D,
                feature_columns=feature_columns or [],
                pipeline_run_id=pipeline_run_id,
            )
        elif X.ndim == 3:
            return cls(
                symbol=symbol,
                timeframe=timeframe,
                horizon=horizon,
                split=split,
                n_samples=X.shape[0],
                n_features=X.shape[2],
                data_rank=DataRank.SEQUENCE_3D,
                sequence_length=X.shape[1],
                feature_columns=feature_columns or [],
                pipeline_run_id=pipeline_run_id,
            )
        elif X.ndim == 4:
            return cls(
                symbol=symbol,
                timeframe=timeframe,
                horizon=horizon,
                split=split,
                n_samples=X.shape[0],
                n_features=X.shape[3],
                data_rank=DataRank.MULTI_TF_4D,
                n_timeframes=X.shape[1],
                sequence_length=X.shape[2],
                feature_columns=feature_columns or [],
                pipeline_run_id=pipeline_run_id,
            )
        else:
            raise ValueError(f"Unsupported array rank: {X.ndim}")


__all__ = [
    "DataRank",
    "FeatureMode",
    "MTFMode",
    "DataContractSchema",
    "DATA_SCHEMA",
    "DataContract",
]
```

---

## 0.2 ModelContract (Model Input Requirements)

### New File: `src/contracts/model_contract.py`

```python
"""
Model Contract - Declares input requirements for each model type.

Every registered model must declare its requirements so the
adapter system can route data correctly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .data_contract import DataRank, FeatureMode, MTFMode


@dataclass(frozen=True)
class ModelContract:
    """
    Input requirements contract for a model.

    Every model declares:
    - What data rank it expects (2D/3D/4D)
    - What feature mode it prefers (engineered/raw/hybrid)
    - What MTF mode it uses (none/indicators/multi_stream)
    - Its primary timeframe preference
    - Sequence length requirements (for 3D/4D)

    This contract is used by:
    1. AdapterRegistry to route data correctly
    2. Trainer to load appropriate timeframe data
    3. Validator to check compatibility before training
    """
    # Model identity
    model_name: str
    model_family: str  # boosting, neural, transformer, classical, ensemble, meta_learner

    # Input requirements
    input_rank: DataRank = DataRank.TABULAR_2D
    feature_mode: FeatureMode = FeatureMode.ENGINEERED
    mtf_mode: MTFMode = MTFMode.NONE

    # Timeframe configuration
    primary_timeframe: str = "5min"  # Default primary TF
    mtf_timeframes: tuple[str, ...] = ()  # Additional TFs for multi_stream

    # Sequence requirements (for 3D/4D models)
    sequence_length: int = 60
    patch_length: int | None = None  # For patch-based transformers

    # Scaling requirements
    requires_scaling: bool = True
    scaler_type: str = "robust"  # robust, standard, minmax, none

    # Feature bounds
    min_features: int = 4
    max_features: int = 200

    # Description
    description: str = ""

    @property
    def requires_sequences(self) -> bool:
        """Whether model requires sequential (3D+) input."""
        return self.input_rank.value >= 3

    @property
    def requires_multi_timeframe(self) -> bool:
        """Whether model requires multi-timeframe (4D) input."""
        return self.input_rank == DataRank.MULTI_TF_4D

    @property
    def adapter_id(self) -> str:
        """
        Determine the adapter ID based on requirements.

        Returns:
            Adapter identifier: "tabular", "sequence", or "multi_stream"
        """
        if self.input_rank == DataRank.TABULAR_2D:
            return "tabular"
        elif self.input_rank == DataRank.SEQUENCE_3D:
            return "sequence"
        elif self.input_rank == DataRank.MULTI_TF_4D:
            return "multi_stream"
        else:
            raise ValueError(f"Unknown input rank: {self.input_rank}")

    def validate_data_contract(
        self,
        data_contract: "DataContract"
    ) -> tuple[bool, list[str]]:
        """
        Validate that a data contract satisfies this model's requirements.

        Args:
            data_contract: DataContract to validate

        Returns:
            (is_valid, list_of_issues)
        """
        from .data_contract import DataContract

        issues = []

        # Check rank compatibility
        if data_contract.data_rank != self.input_rank:
            issues.append(
                f"Data rank mismatch: model expects {self.input_rank.value}D, "
                f"data is {data_contract.data_rank.value}D"
            )

        # Check sequence length for 3D/4D
        if self.requires_sequences:
            if data_contract.sequence_length != self.sequence_length:
                issues.append(
                    f"Sequence length mismatch: model expects {self.sequence_length}, "
                    f"data has {data_contract.sequence_length}"
                )

        # Check timeframe count for 4D
        if self.requires_multi_timeframe:
            expected_n_tf = len(self.mtf_timeframes) + 1  # primary + mtf
            if data_contract.n_timeframes != expected_n_tf:
                issues.append(
                    f"Timeframe count mismatch: model expects {expected_n_tf}, "
                    f"data has {data_contract.n_timeframes}"
                )

        # Check feature count bounds
        if data_contract.n_features < self.min_features:
            issues.append(
                f"Too few features: model needs >= {self.min_features}, "
                f"data has {data_contract.n_features}"
            )
        if data_contract.n_features > self.max_features:
            issues.append(
                f"Too many features: model max is {self.max_features}, "
                f"data has {data_contract.n_features}"
            )

        return len(issues) == 0, issues

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "model_name": self.model_name,
            "model_family": self.model_family,
            "input_rank": self.input_rank.value,
            "feature_mode": self.feature_mode.value,
            "mtf_mode": self.mtf_mode.value,
            "primary_timeframe": self.primary_timeframe,
            "mtf_timeframes": list(self.mtf_timeframes),
            "sequence_length": self.sequence_length,
            "patch_length": self.patch_length,
            "requires_scaling": self.requires_scaling,
            "scaler_type": self.scaler_type,
            "min_features": self.min_features,
            "max_features": self.max_features,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelContract:
        """Deserialize from dictionary."""
        return cls(
            model_name=data["model_name"],
            model_family=data["model_family"],
            input_rank=DataRank(data.get("input_rank", 2)),
            feature_mode=FeatureMode(data.get("feature_mode", "engineered")),
            mtf_mode=MTFMode(data.get("mtf_mode", "none")),
            primary_timeframe=data.get("primary_timeframe", "5min"),
            mtf_timeframes=tuple(data.get("mtf_timeframes", [])),
            sequence_length=data.get("sequence_length", 60),
            patch_length=data.get("patch_length"),
            requires_scaling=data.get("requires_scaling", True),
            scaler_type=data.get("scaler_type", "robust"),
            min_features=data.get("min_features", 4),
            max_features=data.get("max_features", 200),
            description=data.get("description", ""),
        )


# =============================================================================
# MODEL CONTRACT REGISTRY
# =============================================================================

# Pre-defined contracts for all 23 models
MODEL_CONTRACTS: dict[str, ModelContract] = {
    # Boosting models (2D tabular)
    "xgboost": ModelContract(
        model_name="xgboost",
        model_family="boosting",
        input_rank=DataRank.TABULAR_2D,
        feature_mode=FeatureMode.ENGINEERED,
        mtf_mode=MTFMode.INDICATORS,
        primary_timeframe="15min",
        requires_scaling=False,
        scaler_type="none",
        min_features=40,
        max_features=120,
        description="XGBoost gradient boosting",
    ),
    "lightgbm": ModelContract(
        model_name="lightgbm",
        model_family="boosting",
        input_rank=DataRank.TABULAR_2D,
        feature_mode=FeatureMode.ENGINEERED,
        mtf_mode=MTFMode.INDICATORS,
        primary_timeframe="15min",
        requires_scaling=False,
        scaler_type="none",
        min_features=40,
        max_features=120,
        description="LightGBM gradient boosting",
    ),
    "catboost": ModelContract(
        model_name="catboost",
        model_family="boosting",
        input_rank=DataRank.TABULAR_2D,
        feature_mode=FeatureMode.ENGINEERED,
        mtf_mode=MTFMode.INDICATORS,
        primary_timeframe="15min",
        requires_scaling=False,
        scaler_type="none",
        min_features=40,
        max_features=120,
        description="CatBoost gradient boosting",
    ),

    # Classical models (2D tabular)
    "random_forest": ModelContract(
        model_name="random_forest",
        model_family="classical",
        input_rank=DataRank.TABULAR_2D,
        feature_mode=FeatureMode.ENGINEERED,
        mtf_mode=MTFMode.INDICATORS,
        primary_timeframe="15min",
        requires_scaling=False,
        scaler_type="none",
        min_features=30,
        max_features=100,
        description="Random Forest ensemble",
    ),
    "logistic": ModelContract(
        model_name="logistic",
        model_family="classical",
        input_rank=DataRank.TABULAR_2D,
        feature_mode=FeatureMode.ENGINEERED,
        mtf_mode=MTFMode.NONE,
        primary_timeframe="15min",
        requires_scaling=True,
        scaler_type="standard",
        min_features=15,
        max_features=50,
        description="Logistic Regression",
    ),
    "svm": ModelContract(
        model_name="svm",
        model_family="classical",
        input_rank=DataRank.TABULAR_2D,
        feature_mode=FeatureMode.ENGINEERED,
        mtf_mode=MTFMode.NONE,
        primary_timeframe="15min",
        requires_scaling=True,
        scaler_type="standard",
        min_features=15,
        max_features=50,
        description="Support Vector Machine",
    ),

    # Neural models (3D sequence)
    "lstm": ModelContract(
        model_name="lstm",
        model_family="neural",
        input_rank=DataRank.SEQUENCE_3D,
        feature_mode=FeatureMode.ENGINEERED,
        mtf_mode=MTFMode.INDICATORS,
        primary_timeframe="5min",
        sequence_length=60,
        requires_scaling=True,
        scaler_type="robust",
        min_features=50,
        max_features=150,
        description="LSTM recurrent network",
    ),
    "gru": ModelContract(
        model_name="gru",
        model_family="neural",
        input_rank=DataRank.SEQUENCE_3D,
        feature_mode=FeatureMode.ENGINEERED,
        mtf_mode=MTFMode.INDICATORS,
        primary_timeframe="5min",
        sequence_length=60,
        requires_scaling=True,
        scaler_type="robust",
        min_features=50,
        max_features=150,
        description="GRU recurrent network",
    ),
    "tcn": ModelContract(
        model_name="tcn",
        model_family="neural",
        input_rank=DataRank.SEQUENCE_3D,
        feature_mode=FeatureMode.ENGINEERED,
        mtf_mode=MTFMode.NONE,
        primary_timeframe="5min",
        sequence_length=120,
        requires_scaling=True,
        scaler_type="robust",
        min_features=50,
        max_features=120,
        description="Temporal Convolutional Network",
    ),

    # Transformer models (3D sequence or 4D multi-stream)
    "transformer": ModelContract(
        model_name="transformer",
        model_family="neural",
        input_rank=DataRank.SEQUENCE_3D,
        feature_mode=FeatureMode.RAW,
        mtf_mode=MTFMode.MULTI_STREAM,
        primary_timeframe="1min",
        mtf_timeframes=("5min", "15min"),
        sequence_length=128,
        requires_scaling=True,
        scaler_type="standard",
        min_features=4,
        max_features=20,
        description="Vanilla Transformer encoder",
    ),
    "patchtst": ModelContract(
        model_name="patchtst",
        model_family="neural",
        input_rank=DataRank.SEQUENCE_3D,
        feature_mode=FeatureMode.RAW,
        mtf_mode=MTFMode.MULTI_STREAM,
        primary_timeframe="1min",
        mtf_timeframes=("5min", "15min"),
        sequence_length=60,
        patch_length=16,
        requires_scaling=True,
        scaler_type="standard",
        min_features=4,
        max_features=10,
        description="PatchTST Transformer",
    ),
    "itransformer": ModelContract(
        model_name="itransformer",
        model_family="neural",
        input_rank=DataRank.SEQUENCE_3D,
        feature_mode=FeatureMode.RAW,
        mtf_mode=MTFMode.MULTI_STREAM,
        primary_timeframe="1min",
        mtf_timeframes=("5min", "15min"),
        sequence_length=60,
        requires_scaling=True,
        scaler_type="robust",
        min_features=4,
        max_features=10,
        description="iTransformer channel attention",
    ),
    "tft": ModelContract(
        model_name="tft",
        model_family="neural",
        input_rank=DataRank.SEQUENCE_3D,
        feature_mode=FeatureMode.HYBRID,
        mtf_mode=MTFMode.INDICATORS,
        primary_timeframe="5min",
        sequence_length=60,
        requires_scaling=True,
        scaler_type="robust",
        min_features=10,
        max_features=40,
        description="Temporal Fusion Transformer",
    ),
    "nbeats": ModelContract(
        model_name="nbeats",
        model_family="neural",
        input_rank=DataRank.SEQUENCE_3D,
        feature_mode=FeatureMode.RAW,
        mtf_mode=MTFMode.NONE,
        primary_timeframe="5min",
        sequence_length=60,
        requires_scaling=True,
        scaler_type="robust",
        min_features=2,
        max_features=10,
        description="N-BEATS decomposition",
    ),
    "inceptiontime": ModelContract(
        model_name="inceptiontime",
        model_family="neural",
        input_rank=DataRank.SEQUENCE_3D,
        feature_mode=FeatureMode.ENGINEERED,
        mtf_mode=MTFMode.NONE,
        primary_timeframe="5min",
        sequence_length=60,
        requires_scaling=True,
        scaler_type="robust",
        min_features=30,
        max_features=100,
        description="InceptionTime CNN",
    ),
    "resnet1d": ModelContract(
        model_name="resnet1d",
        model_family="neural",
        input_rank=DataRank.SEQUENCE_3D,
        feature_mode=FeatureMode.ENGINEERED,
        mtf_mode=MTFMode.NONE,
        primary_timeframe="5min",
        sequence_length=60,
        requires_scaling=True,
        scaler_type="robust",
        min_features=30,
        max_features=100,
        description="ResNet1D",
    ),

    # Ensemble models (variable - depends on base models)
    "voting": ModelContract(
        model_name="voting",
        model_family="ensemble",
        input_rank=DataRank.TABULAR_2D,  # Default, varies
        feature_mode=FeatureMode.ENGINEERED,
        mtf_mode=MTFMode.NONE,
        requires_scaling=False,
        min_features=10,
        max_features=200,
        description="Voting ensemble",
    ),
    "stacking": ModelContract(
        model_name="stacking",
        model_family="ensemble",
        input_rank=DataRank.TABULAR_2D,  # Meta-learner sees 2D OOF
        feature_mode=FeatureMode.ENGINEERED,
        mtf_mode=MTFMode.NONE,
        requires_scaling=False,
        min_features=10,
        max_features=200,
        description="Stacking ensemble",
    ),
    "blending": ModelContract(
        model_name="blending",
        model_family="ensemble",
        input_rank=DataRank.TABULAR_2D,  # Meta-learner sees 2D holdout
        feature_mode=FeatureMode.ENGINEERED,
        mtf_mode=MTFMode.NONE,
        requires_scaling=False,
        min_features=10,
        max_features=200,
        description="Blending ensemble",
    ),

    # Meta-learners (always 2D - OOF predictions)
    "ridge_meta": ModelContract(
        model_name="ridge_meta",
        model_family="meta_learner",
        input_rank=DataRank.TABULAR_2D,
        feature_mode=FeatureMode.ENGINEERED,  # OOF features
        mtf_mode=MTFMode.NONE,
        requires_scaling=False,
        min_features=2,
        max_features=20,
        description="Ridge meta-learner",
    ),
    "mlp_meta": ModelContract(
        model_name="mlp_meta",
        model_family="meta_learner",
        input_rank=DataRank.TABULAR_2D,
        feature_mode=FeatureMode.ENGINEERED,
        mtf_mode=MTFMode.NONE,
        requires_scaling=True,
        scaler_type="standard",
        min_features=2,
        max_features=20,
        description="MLP meta-learner",
    ),
    "calibrated_meta": ModelContract(
        model_name="calibrated_meta",
        model_family="meta_learner",
        input_rank=DataRank.TABULAR_2D,
        feature_mode=FeatureMode.ENGINEERED,
        mtf_mode=MTFMode.NONE,
        requires_scaling=False,
        min_features=2,
        max_features=20,
        description="Calibrated meta-learner",
    ),
    "xgboost_meta": ModelContract(
        model_name="xgboost_meta",
        model_family="meta_learner",
        input_rank=DataRank.TABULAR_2D,
        feature_mode=FeatureMode.ENGINEERED,
        mtf_mode=MTFMode.NONE,
        requires_scaling=False,
        min_features=2,
        max_features=20,
        description="XGBoost meta-learner",
    ),
}


def get_model_contract(model_name: str) -> ModelContract:
    """
    Get the contract for a model.

    Args:
        model_name: Model name (e.g., "xgboost", "lstm")

    Returns:
        ModelContract for the model

    Raises:
        ValueError: If model not found
    """
    name_lower = model_name.lower().strip()
    if name_lower not in MODEL_CONTRACTS:
        raise ValueError(
            f"No contract for model '{model_name}'. "
            f"Available: {sorted(MODEL_CONTRACTS.keys())}"
        )
    return MODEL_CONTRACTS[name_lower]


__all__ = [
    "ModelContract",
    "MODEL_CONTRACTS",
    "get_model_contract",
]
```

---

## 0.3 ArtifactManifest (Safety and Reproducibility)

### New File: `src/contracts/artifact_manifest.py`

```python
"""
Artifact Manifest - Safety and reproducibility metadata for saved artifacts.

Every saved artifact (model, predictions, metrics) includes a manifest
that enables verification and reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .data_contract import DataContract
from .model_contract import ModelContract


@dataclass
class ArtifactManifest:
    """
    Manifest for saved artifacts (models, predictions, etc.).

    Enables:
    - Verification that loaded artifacts match training environment
    - Reproducibility through config and code version tracking
    - Safe loading with hash validation
    """
    # Identity
    artifact_type: str  # "model", "predictions", "metrics", "oof"
    artifact_name: str
    artifact_path: str

    # Hashes
    config_hash: str = ""
    data_hash: str = ""
    code_version: str = ""
    artifact_hash: str = ""  # Hash of the artifact file itself

    # Contracts (serialized)
    data_contract: dict[str, Any] = field(default_factory=dict)
    model_contract: dict[str, Any] = field(default_factory=dict)

    # Training context
    model_name: str = ""
    model_family: str = ""
    horizon: int = 0
    timeframe: str = ""

    # Runtime info
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    python_version: str = ""
    package_versions: dict[str, str] = field(default_factory=dict)

    # Pipeline lineage
    pipeline_run_id: str = ""
    training_run_id: str = ""

    def __post_init__(self):
        """Capture runtime info if not provided."""
        import sys
        if not self.python_version:
            self.python_version = sys.version
        if not self.code_version:
            self.code_version = self._get_git_commit()

    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()[:8]
        except Exception:
            pass
        return "unknown"

    def compute_artifact_hash(self, artifact_path: Path | str) -> str:
        """Compute SHA256 hash of artifact file."""
        path = Path(artifact_path)
        if not path.exists():
            return ""

        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()[:16]

    def validate_artifact(self, artifact_path: Path | str) -> tuple[bool, list[str]]:
        """
        Validate artifact against manifest.

        Args:
            artifact_path: Path to artifact file

        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        path = Path(artifact_path)

        # Check file exists
        if not path.exists():
            issues.append(f"Artifact file not found: {path}")
            return False, issues

        # Check artifact hash if stored
        if self.artifact_hash:
            current_hash = self.compute_artifact_hash(path)
            if current_hash != self.artifact_hash:
                issues.append(
                    f"Artifact hash mismatch: expected {self.artifact_hash}, "
                    f"got {current_hash}"
                )

        return len(issues) == 0, issues

    def validate_environment(self) -> tuple[bool, list[str]]:
        """
        Validate current environment matches manifest.

        Returns:
            (is_valid, list_of_warnings)
        """
        import sys
        warnings = []

        # Check Python version (major.minor)
        current_py = ".".join(sys.version.split(".")[:2])
        manifest_py = ".".join(self.python_version.split(".")[:2]) if self.python_version else ""
        if manifest_py and current_py != manifest_py:
            warnings.append(
                f"Python version mismatch: trained with {manifest_py}, "
                f"running with {current_py}"
            )

        # Check key packages
        key_packages = ["torch", "numpy", "pandas", "sklearn"]
        for pkg in key_packages:
            if pkg in self.package_versions:
                try:
                    import importlib
                    mod = importlib.import_module(pkg)
                    current_version = getattr(mod, "__version__", "unknown")
                    saved_version = self.package_versions[pkg]
                    if current_version != saved_version:
                        warnings.append(
                            f"{pkg} version mismatch: trained with {saved_version}, "
                            f"running with {current_version}"
                        )
                except ImportError:
                    warnings.append(f"Package {pkg} not installed")

        return len(warnings) == 0, warnings

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "artifact_type": self.artifact_type,
            "artifact_name": self.artifact_name,
            "artifact_path": self.artifact_path,
            "config_hash": self.config_hash,
            "data_hash": self.data_hash,
            "code_version": self.code_version,
            "artifact_hash": self.artifact_hash,
            "data_contract": self.data_contract,
            "model_contract": self.model_contract,
            "model_name": self.model_name,
            "model_family": self.model_family,
            "horizon": self.horizon,
            "timeframe": self.timeframe,
            "created_at": self.created_at,
            "python_version": self.python_version,
            "package_versions": self.package_versions,
            "pipeline_run_id": self.pipeline_run_id,
            "training_run_id": self.training_run_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactManifest:
        """Deserialize from dictionary."""
        return cls(
            artifact_type=data.get("artifact_type", "unknown"),
            artifact_name=data.get("artifact_name", ""),
            artifact_path=data.get("artifact_path", ""),
            config_hash=data.get("config_hash", ""),
            data_hash=data.get("data_hash", ""),
            code_version=data.get("code_version", ""),
            artifact_hash=data.get("artifact_hash", ""),
            data_contract=data.get("data_contract", {}),
            model_contract=data.get("model_contract", {}),
            model_name=data.get("model_name", ""),
            model_family=data.get("model_family", ""),
            horizon=data.get("horizon", 0),
            timeframe=data.get("timeframe", ""),
            created_at=data.get("created_at", ""),
            python_version=data.get("python_version", ""),
            package_versions=data.get("package_versions", {}),
            pipeline_run_id=data.get("pipeline_run_id", ""),
            training_run_id=data.get("training_run_id", ""),
        )

    def save(self, path: Path | str) -> None:
        """Save manifest to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> ArtifactManifest:
        """Load manifest from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def create_for_model(
        cls,
        model_path: Path | str,
        model_name: str,
        data_contract: DataContract,
        model_contract: ModelContract,
        config_hash: str = "",
        pipeline_run_id: str = "",
        training_run_id: str = "",
    ) -> ArtifactManifest:
        """Create manifest for a saved model."""
        path = Path(model_path)

        # Capture key package versions
        package_versions = {}
        for pkg in ["torch", "numpy", "pandas", "sklearn", "xgboost", "lightgbm"]:
            try:
                import importlib
                mod = importlib.import_module(pkg)
                package_versions[pkg] = getattr(mod, "__version__", "unknown")
            except ImportError:
                pass

        manifest = cls(
            artifact_type="model",
            artifact_name=model_name,
            artifact_path=str(path),
            config_hash=config_hash,
            data_hash=data_contract.schema_hash,
            data_contract=data_contract.to_dict(),
            model_contract=model_contract.to_dict(),
            model_name=model_name,
            model_family=model_contract.model_family,
            horizon=data_contract.horizon,
            timeframe=data_contract.timeframe,
            package_versions=package_versions,
            pipeline_run_id=pipeline_run_id,
            training_run_id=training_run_id,
        )

        # Compute artifact hash if file exists
        if path.exists():
            manifest.artifact_hash = manifest.compute_artifact_hash(path)

        return manifest


__all__ = [
    "ArtifactManifest",
]
```

---

## 0.4 Package Init

### New File: `src/contracts/__init__.py`

```python
"""
Contracts package - Canonical contracts for data, models, and artifacts.

This package defines the foundational contracts that ensure
compatibility and reproducibility across the ML pipeline.
"""

from .data_contract import (
    DataRank,
    FeatureMode,
    MTFMode,
    DataContractSchema,
    DATA_SCHEMA,
    DataContract,
)
from .model_contract import (
    ModelContract,
    MODEL_CONTRACTS,
    get_model_contract,
)
from .artifact_manifest import (
    ArtifactManifest,
)

__all__ = [
    # Data contract
    "DataRank",
    "FeatureMode",
    "MTFMode",
    "DataContractSchema",
    "DATA_SCHEMA",
    "DataContract",
    # Model contract
    "ModelContract",
    "MODEL_CONTRACTS",
    "get_model_contract",
    # Artifact manifest
    "ArtifactManifest",
]
```

---

## Summary: Phase 0 Changes

| File | Type | Purpose |
|------|------|---------|
| `src/contracts/__init__.py` | NEW | Package init with exports |
| `src/contracts/data_contract.py` | NEW | DataContract, DataRank, FeatureMode, MTFMode |
| `src/contracts/model_contract.py` | NEW | ModelContract, MODEL_CONTRACTS registry |
| `src/contracts/artifact_manifest.py` | NEW | ArtifactManifest for reproducibility |

## Dependencies

Phase 0 has no dependencies on other phases and introduces no breaking changes.

## Migration Notes

- No existing code needs modification for Phase 0
- Phase 1+ will import and use these contracts
- Existing code continues to work unchanged

## Next Steps

After Phase 0 is implemented, proceed to Phase 1 (Configuration Layer) which will:
1. Extend TrainerConfig with fields from ModelContract
2. Add per-model configuration to UnifiedConfig
3. Wire contracts into the existing configuration system
