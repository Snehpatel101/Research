"""
Stage 7.6: Dataset Build.

Creates split-ready datasets and feature-set manifests for downstream models.

This module provides:
- run_build_datasets: Pipeline stage for creating dataset splits
- TimeSeriesDataContainer: Unified container for Phase 2 model training
- SequenceDataset: PyTorch Dataset for sequence models (3D)
- MultiResolution4DDataset: PyTorch Dataset for multi-timeframe models (4D)
- validate_model_ready: Comprehensive validation for model training readiness

Usage:
------
    # Pipeline stage
    from src.phase1.stages.datasets import run_build_datasets
    result = run_build_datasets(config, manifest)

    # Direct data loading for Phase 2
    from src.phase1.stages.datasets import TimeSeriesDataContainer

    container = TimeSeriesDataContainer.from_parquet_dir(
        path="data/splits/scaled",
        horizon=20
    )

    # Get sklearn arrays (2D)
    X, y, w = container.get_sklearn_arrays("train")

    # Get PyTorch sequences (3D)
    dataset = container.get_pytorch_sequences("train", seq_len=60)

    # Get Multi-Resolution 4D sequences
    dataset_4d = container.get_multi_resolution_4d("train", seq_len=60)
    # X_4d shape: (batch, n_timeframes, seq_len, n_features)

    # Get NeuralForecast format
    nf_df = container.get_neuralforecast_df("train")

    # Validate model readiness
    from src.phase1.stages.datasets import validate_model_ready, ValidationResult
    result = validate_model_ready(container)
    if not result.is_valid:
        raise ValueError(f"Validation failed: {result.errors}")
"""

from src.phase1.stages.datasets.container import (
    INVALID_LABEL,
    METADATA_COLUMNS,
    DataContainerConfig,
    SplitData,
    TimeSeriesDataContainer,
)
from src.phase1.utils.feature_sets import LABEL_PREFIXES
from src.phase1.stages.datasets.run import run_build_datasets
from src.phase1.stages.datasets.sequences import (
    SequenceConfig,
    SequenceDataset,
    build_sequence_indices,
    create_sequence_dataset,
    find_symbol_boundaries,
)
from src.phase1.stages.datasets.validators import (
    ValidationResult,
    validate_model_ready,
)
from src.phase1.stages.datasets.adapters import (
    MultiResolution4DAdapter,
    MultiResolution4DConfig,
    MultiResolution4DDataset,
    create_multi_resolution_dataset,
)

__all__ = [
    # Pipeline stage
    "run_build_datasets",
    # Container classes
    "TimeSeriesDataContainer",
    "DataContainerConfig",
    "SplitData",
    # Sequence classes (3D)
    "SequenceDataset",
    "SequenceConfig",
    "create_sequence_dataset",
    # Multi-Resolution classes (4D)
    "MultiResolution4DAdapter",
    "MultiResolution4DConfig",
    "MultiResolution4DDataset",
    "create_multi_resolution_dataset",
    # Validation
    "validate_model_ready",
    "ValidationResult",
    # Utility functions
    "build_sequence_indices",
    "find_symbol_boundaries",
    # Constants
    "METADATA_COLUMNS",
    "LABEL_PREFIXES",
    "INVALID_LABEL",
]
