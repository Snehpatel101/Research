"""
Data preparation utilities for model training.

This module handles loading and converting data from TimeSeriesDataContainer
into formats suitable for different model types (tabular vs sequential).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

if TYPE_CHECKING:
    from src.core.container import TimeSeriesDataContainer


class SizedDataset(Protocol):
    """Protocol for datasets that have __len__ and __getitem__."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> tuple[Any, Any, Any]: ...


logger = logging.getLogger(__name__)


def prepare_training_data(
    container: TimeSeriesDataContainer,
    requires_sequences: bool,
    sequence_length: int = 60,
    feature_columns: list[str] | None = None,
    requires_4d: bool = False,
    timeframes: list[str] | None = None,
    features_per_timeframe: list[str] | None = None,
    include_base_features: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for training based on model requirements.

    Args:
        container: TimeSeriesDataContainer with data
        requires_sequences: Whether model needs sequential data (LSTM, etc.)
        sequence_length: Sequence length for sequential models
        feature_columns: Optional list of feature columns to use for sequences.
            If None, uses all available features. Use this to filter to
            model-specific optimal feature sets (e.g., tcn_optimal, neural_optimal).
        requires_4d: Whether model needs 4D multi-resolution data
            (batch, n_timeframes, seq_len, features). When True, uses
            MultiResolution4DAdapter via container.get_multi_resolution_4d().
        timeframes: Optional list of timeframes for 4D data (base first).
            Only used when requires_4d=True.
        features_per_timeframe: Optional list of base feature names (without suffix)
            to extract per timeframe for 4D data (e.g., ["open", "high", "low", "close", "volume"]).
            Only used when requires_4d=True.
        include_base_features: Include non-suffixed base features for the base timeframe.
            Only used when requires_4d=True.

    Returns:
        Tuple of (X_train, y_train, w_train, X_val, y_val)
    """
    if requires_4d and requires_sequences:
        raise ValueError("requires_4d and requires_sequences cannot both be True")

    if requires_4d:
        # Get 4D multi-resolution data for transformer models
        train_dataset = container.get_multi_resolution_4d(
            "train",
            seq_len=sequence_length,
            timeframes=timeframes,
            features_per_timeframe=features_per_timeframe,
            include_base_features=include_base_features,
        )
        val_dataset = container.get_multi_resolution_4d(
            "val",
            seq_len=sequence_length,
            timeframes=timeframes,
            features_per_timeframe=features_per_timeframe,
            include_base_features=include_base_features,
        )

        logger.info(
            f"4D multi-resolution data prepared: "
            f"train={len(train_dataset)} sequences, val={len(val_dataset)} sequences"
        )

        X_train, y_train, w_train = dataset_to_arrays(cast(SizedDataset, train_dataset))
        X_val, y_val, _ = dataset_to_arrays(cast(SizedDataset, val_dataset))

    elif requires_sequences:
        # Get sequence data for sequential models
        train_dataset = container.get_pytorch_sequences(
            "train",
            seq_len=sequence_length,
            symbol_isolated=True,
            feature_columns=feature_columns,
        )
        val_dataset = container.get_pytorch_sequences(
            "val",
            seq_len=sequence_length,
            symbol_isolated=True,
            feature_columns=feature_columns,
        )

        # Log feature count if filtering was applied
        if feature_columns is not None:
            logger.info(
                f"Sequence data prepared with {len(feature_columns)} filtered features "
                f"(from feature set)"
            )

        # Convert to numpy arrays
        X_train, y_train, w_train = dataset_to_arrays(cast(SizedDataset, train_dataset))
        X_val, y_val, _ = dataset_to_arrays(cast(SizedDataset, val_dataset))

    else:
        # Get tabular data for non-sequential models
        X_train, y_train, w_train = container.get_sklearn_arrays("train")
        X_val, y_val, _ = container.get_sklearn_arrays("val")

    return X_train, y_train, w_train, X_val, y_val


def prepare_test_data(
    container: TimeSeriesDataContainer,
    requires_sequences: bool,
    sequence_length: int = 60,
    feature_columns: list[str] | None = None,
    requires_4d: bool = False,
    timeframes: list[str] | None = None,
    features_per_timeframe: list[str] | None = None,
    include_base_features: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare test data for final evaluation.

    Args:
        container: TimeSeriesDataContainer with test split
        requires_sequences: Whether model needs sequential data
        sequence_length: Sequence length for sequential models
        feature_columns: Optional list of feature columns to use for sequences.
            If None, uses all available features.
        requires_4d: Whether model needs 4D multi-resolution data.
        timeframes: Optional list of timeframes for 4D data (base first).
            Only used when requires_4d=True.
        features_per_timeframe: Optional list of base feature names (without suffix)
            to extract per timeframe for 4D data.
            Only used when requires_4d=True.
        include_base_features: Include non-suffixed base features for the base timeframe.
            Only used when requires_4d=True.

    Returns:
        Tuple of (X_test, y_test, w_test)
    """
    if requires_4d and requires_sequences:
        raise ValueError("requires_4d and requires_sequences cannot both be True")

    if requires_4d:
        test_dataset = container.get_multi_resolution_4d(
            "test",
            seq_len=sequence_length,
            timeframes=timeframes,
            features_per_timeframe=features_per_timeframe,
            include_base_features=include_base_features,
        )
        X_test, y_test, w_test = dataset_to_arrays(cast(SizedDataset, test_dataset))
    elif requires_sequences:
        # Get sequence data for sequential models
        test_dataset = container.get_pytorch_sequences(
            "test",
            seq_len=sequence_length,
            symbol_isolated=True,
            feature_columns=feature_columns,
        )
        # Convert to numpy arrays
        X_test, y_test, w_test = dataset_to_arrays(cast(SizedDataset, test_dataset))
    else:
        # Get tabular data for non-sequential models
        X_test, y_test, w_test = container.get_sklearn_arrays("test")

    return X_test, y_test, w_test


def dataset_to_arrays(
    dataset: SizedDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert PyTorch dataset to numpy arrays.

    Memory-efficient implementation that pre-allocates arrays to avoid
    accumulating tensors in lists (which can cause 200GB+ memory usage).

    Args:
        dataset: SequenceDataset from container

    Returns:
        Tuple of (X, y, weights) numpy arrays
    """
    import gc

    import torch

    n_samples = len(dataset)
    if n_samples == 0:
        return np.array([]), np.array([]), np.array([])

    # Get first sample to determine shapes
    X_0, y_0, w_0 = dataset[0]

    # Convert to numpy immediately to avoid holding tensors
    if isinstance(X_0, torch.Tensor):  # noqa: SIM108
        X_0_np = X_0.detach().cpu().numpy()
    else:
        X_0_np = np.asarray(X_0)

    # Pre-allocate arrays (avoids list accumulation memory leak)
    X = np.empty((n_samples, *X_0_np.shape), dtype=np.float32)
    y = np.empty(n_samples, dtype=np.float32)
    w = np.empty(n_samples, dtype=np.float32)

    # Fill first element
    X[0] = X_0_np
    y[0] = float(y_0.item() if isinstance(y_0, torch.Tensor) else y_0)
    w[0] = float(w_0.item() if isinstance(w_0, torch.Tensor) else w_0)

    # Clean up first sample references
    del X_0, y_0, w_0, X_0_np

    # Fill remaining elements in-place
    for i in range(1, n_samples):
        X_i, y_i, w_i = dataset[i]

        # Convert tensor to numpy immediately and store in pre-allocated array
        if isinstance(X_i, torch.Tensor):
            X[i] = X_i.detach().cpu().numpy()
        else:
            X[i] = np.asarray(X_i)

        y[i] = float(y_i.item() if isinstance(y_i, torch.Tensor) else y_i)
        w[i] = float(w_i.item() if isinstance(w_i, torch.Tensor) else w_i)

        # Explicitly delete tensor references to allow garbage collection
        del X_i, y_i, w_i

        # Periodic garbage collection for very large datasets
        if i % 10000 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    # Final cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return X, y, w


__all__ = [
    "prepare_training_data",
    "prepare_test_data",
    "dataset_to_arrays",
]
