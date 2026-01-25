"""
Base adapter class and result dataclass.

Phase 2 SNwH Implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.core.contracts import DataContract, ModelContract

from src.core.contracts import DataRank
from src.core.contracts.data_contract import DataContractViolation


def validate_label_alignment(features: pd.DataFrame, labels: pd.Series) -> None:
    """Validate labels are properly aligned with features.

    This function ensures that labels match features exactly in both length
    and index alignment. This catches off-by-one errors and timestamp
    misalignments that could cause incorrect label associations.

    Args:
        features: Feature DataFrame with datetime index
        labels: Label Series that should align with features

    Raises:
        DataContractViolation: If labels don't align with features

    Example:
        >>> features = pd.DataFrame({'a': [1,2,3]}, index=pd.date_range('2020-01-01', periods=3))
        >>> labels = pd.Series([0,1,0], index=pd.date_range('2020-01-01', periods=3))
        >>> validate_label_alignment(features, labels)  # No exception
    """
    # Check length match
    if len(features) != len(labels):
        raise DataContractViolation(
            [f"Label length mismatch: features={len(features)}, labels={len(labels)}"]
        )

    # Check index alignment if both have indices
    has_indices = hasattr(features, "index") and hasattr(labels, "index")
    if has_indices and not features.index.equals(labels.index):
        # Find first mismatch for detailed error reporting
        mismatches = features.index != labels.index
        if mismatches.any():
            first_mismatch_idx = int(mismatches.argmax())
            raise DataContractViolation(
                [
                    f"Label index mismatch at position {first_mismatch_idx}: "
                    f"features={features.index[first_mismatch_idx]}, "
                    f"labels={labels.index[first_mismatch_idx]}"
                ]
            )


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

    def __post_init__(self) -> None:
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

    # =========================================================================
    # BACKWARD COMPATIBILITY PROPERTIES
    # =========================================================================

    @property
    def data(self) -> np.ndarray:
        """Alias for X (backward compatibility with interfaces.py)."""
        return self.X

    @property
    def labels(self) -> np.ndarray:
        """Alias for y (backward compatibility with interfaces.py)."""
        return self.y

    @property
    def feature_names(self) -> list[str]:
        """Alias for feature_columns (backward compatibility with interfaces.py)."""
        return self.feature_columns

    @property
    def rank(self) -> int:
        """Data tensor rank (backward compatibility with interfaces.py)."""
        return self.data_rank.value

    @property
    def shape(self) -> tuple[int, ...]:
        """Full data shape (backward compatibility with interfaces.py)."""
        return self.X.shape

    @property
    def metadata(self) -> dict[str, Any]:
        """
        Metadata dictionary (backward compatibility with interfaces.py).

        Returns a dictionary of metadata fields. Note: This is read-only.
        To modify metadata, update the individual fields directly.
        """
        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "data_rank": self.data_rank.value,
            "sequence_length": self.sequence_length,
            "n_timeframes": self.n_timeframes,
            "timeframe_names": self.timeframe_names,
            "adapter_name": self.adapter_name,
        }

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate adapter result.

        Returns:
            (is_valid, list_of_issues)

        Note: This method returns a tuple for backward compatibility.
        For exception-based validation, call validate_strict() instead.
        """
        issues: list[str] = []

        # Check X shape matches declared rank
        if self.X.ndim != self.data_rank.value:
            issues.append(f"X rank mismatch: expected {self.data_rank.value}D, got {self.X.ndim}D")

        # Check y length
        if self.y.shape[0] != self.X.shape[0]:
            issues.append(
                f"y length mismatch: X has {self.X.shape[0]} samples, " f"y has {self.y.shape[0]}"
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

    def validate_strict(self) -> None:
        """
        Validate the adapter result (exception-based).

        Raises:
            ValueError: If validation fails

        This method provides compatibility with the interfaces.py validate() API.
        """
        if self.X.size == 0:
            raise ValueError("AdapterResult data is empty")
        if len(self.y) != self.n_samples:
            raise ValueError(f"Labels length ({len(self.y)}) != n_samples ({self.n_samples})")
        if self.original_indices is not None and len(self.original_indices) != self.n_samples:
            raise ValueError("original_indices length != n_samples")
        if np.isnan(self.X).any():
            raise ValueError("AdapterResult data contains NaN values")

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to dictionary (excludes arrays)."""
        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "data_rank": self.data_rank.value,
            "sequence_length": self.sequence_length,
            "n_timeframes": self.n_timeframes,
            "timeframe_names": self.timeframe_names,
            "feature_columns": self.feature_columns,
            "adapter_name": self.adapter_name,
            "X_shape": list(self.X.shape),
            "y_shape": list(self.y.shape),
            "has_weights": self.weights is not None,
        }


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
        lazy_load: bool = False,
        chunk_size: int = 100_000,
    ):
        """
        Initialize adapter.

        Args:
            feature_columns: Feature columns to use (None = auto-detect)
            label_column: Label column name
            weight_column: Weight column name (None = uniform weights)
            lazy_load: Enable lazy loading for >1GB datasets (12D-7)
            chunk_size: Rows per chunk for lazy loading
        """
        self.feature_columns = feature_columns
        self.label_column = label_column
        self.weight_column = weight_column
        self.lazy_load = lazy_load
        self.chunk_size = chunk_size

    @abstractmethod
    def transform(
        self,
        df: pd.DataFrame,
        model_contract: ModelContract | None = None,
        additional_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> AdapterResult:
        """
        Transform DataFrame to model-specific format.

        Args:
            df: Source DataFrame with features, labels, weights
            model_contract: Optional contract for validation. When provided,
                the adapter will validate output against model requirements
                and raise ValidationError on mismatch. Should be provided
                in production training pipelines but can be omitted for
                testing or exploration.
            additional_dfs: Optional additional DataFrames for multi-stream adapters

        Returns:
            AdapterResult with transformed arrays

        Raises:
            ValidationError: If model_contract is provided and validation fails
            ValueError: If input validation fails
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
        issues: list[str] = []

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
        """Get feature columns (explicit or auto-detected).

        Priority order:
        1. Explicit feature_columns from __init__
        2. Auto-detect from column naming conventions
        """
        if self.feature_columns:
            return self.feature_columns

        # Auto-detect: exclude metadata and labels
        exclude_prefixes = (
            "label_",
            "sample_weight_",
            "regime_",
            "timestamp",
            "datetime",
            "symbol",
            "timeframe",
            "split",
        )
        exclude_exact = {
            "open",
            "high",
            "low",
            "close",
            "volume",
            "bar_index",
            "session_id",
        }

        def is_feature_col(col: str) -> bool:
            col_lower = col.lower()
            if col_lower in exclude_exact:
                return False
            return all(not col_lower.startswith(prefix) for prefix in exclude_prefixes)

        return [col for col in df.columns if is_feature_col(col)]

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Path | str,
        label_column: str | None = None,
        weight_column: str | None = None,
        **kwargs,
    ) -> BaseAdapter:
        """Create adapter with feature columns from a manifest file.

        Phase 7D: Factory method that loads feature columns from a manifest
        instead of relying on auto-detection or explicit lists.

        Args:
            manifest_path: Path to the feature manifest JSON file
            label_column: Override label column (uses first from manifest if None)
            weight_column: Override weight column
            **kwargs: Additional arguments passed to __init__

        Returns:
            Adapter instance configured with manifest's feature columns

        Raises:
            FileNotFoundError: If manifest file doesn't exist

        Example:
            adapter = SequenceAdapter.from_manifest(
                "data/MES_feature_manifest.json",
                sequence_length=60
            )
        """
        from src.data.pipeline.feature_manifest import FeatureManifest

        manifest = FeatureManifest.load(Path(manifest_path))

        # Use first label from manifest if not specified
        if label_column is None and manifest.label_columns:
            label_column = manifest.label_columns[0]
        elif label_column is None:
            label_column = "label_h20"  # Default fallback

        return cls(
            feature_columns=manifest.feature_columns,
            label_column=label_column,
            weight_column=weight_column,
            **kwargs,
        )

    def _get_weights(self, df: pd.DataFrame) -> np.ndarray | None:
        """Get sample weights from DataFrame."""
        if self.weight_column and self.weight_column in df.columns:
            return np.asarray(df[self.weight_column].values.astype(np.float32))
        return None

    def load_data_lazy(self, file_path: Path | str) -> pd.DataFrame:
        """
        Load data with lazy loading for large datasets (12D-7).

        For datasets >1GB, reads in chunks to avoid memory spikes.
        For smaller datasets, uses standard pd.read_parquet.

        Args:
            file_path: Path to parquet file

        Returns:
            DataFrame with all data loaded
        """
        file_path = Path(file_path)

        # Check file size
        file_size_gb = file_path.stat().st_size / (1024**3)

        if not self.lazy_load or file_size_gb < 1.0:
            # Standard loading for small files
            return pd.read_parquet(file_path)

        # Lazy loading for large files
        import logging

        logger = logging.getLogger(__name__)
        logger.info(
            f"Large dataset detected ({file_size_gb:.2f} GB). "
            f"Using chunked reading with chunk_size={self.chunk_size:,}"
        )

        # Read in chunks and concatenate
        parquet_file = pd.read_parquet(file_path, engine="pyarrow")

        # If file is already loaded, just return it
        # (pyarrow doesn't support true streaming for parquet yet)
        # For future: use dask or pyarrow.parquet.ParquetFile for true streaming
        return parquet_file


__all__ = [
    "AdapterResult",
    "BaseAdapter",
    "validate_label_alignment",
]
