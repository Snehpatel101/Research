"""
Data preparation utilities for model training.

This module handles loading and converting data from TimeSeriesDataContainer
into formats suitable for different model types (tabular vs sequential).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from torch.utils.data import Dataset

    from src.phase1.stages.datasets.container import TimeSeriesDataContainer


def prepare_training_data(
    container: TimeSeriesDataContainer,
    requires_sequences: bool,
    sequence_length: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for training based on model requirements.

    Args:
        container: TimeSeriesDataContainer with data
        requires_sequences: Whether model needs sequential data (LSTM, etc.)
        sequence_length: Sequence length for sequential models

    Returns:
        Tuple of (X_train, y_train, w_train, X_val, y_val)
    """
    if requires_sequences:
        # Get sequence data for sequential models
        train_dataset = container.get_pytorch_sequences(
            "train",
            seq_len=sequence_length,
            symbol_isolated=True,
        )
        val_dataset = container.get_pytorch_sequences(
            "val",
            seq_len=sequence_length,
            symbol_isolated=True,
        )

        # Convert to numpy arrays
        X_train, y_train, w_train = dataset_to_arrays(train_dataset)
        X_val, y_val, _ = dataset_to_arrays(val_dataset)

    else:
        # Get tabular data for non-sequential models
        X_train, y_train, w_train = container.get_sklearn_arrays("train")
        X_val, y_val, _ = container.get_sklearn_arrays("val")

    return X_train, y_train, w_train, X_val, y_val


def prepare_test_data(
    container: TimeSeriesDataContainer,
    requires_sequences: bool,
    sequence_length: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare test data for final evaluation.

    Args:
        container: TimeSeriesDataContainer with test split
        requires_sequences: Whether model needs sequential data
        sequence_length: Sequence length for sequential models

    Returns:
        Tuple of (X_test, y_test, w_test)
    """
    if requires_sequences:
        # Get sequence data for sequential models
        test_dataset = container.get_pytorch_sequences(
            "test",
            seq_len=sequence_length,
            symbol_isolated=True,
        )
        # Convert to numpy arrays
        X_test, y_test, w_test = dataset_to_arrays(test_dataset)
    else:
        # Get tabular data for non-sequential models
        X_test, y_test, w_test = container.get_sklearn_arrays("test")

    return X_test, y_test, w_test


def dataset_to_arrays(
    dataset: Dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert PyTorch dataset to numpy arrays.

    Args:
        dataset: SequenceDataset from container

    Returns:
        Tuple of (X, y, weights) numpy arrays
    """
    # Get all data from dataset
    X_list = []
    y_list = []
    w_list = []

    for i in range(len(dataset)):
        X_i, y_i, w_i = dataset[i]
        X_list.append(X_i)
        y_list.append(y_i)
        w_list.append(w_i)

    X = np.stack(X_list)
    y = np.array(y_list)
    w = np.array(w_list)

    return X, y, w


__all__ = [
    "prepare_training_data",
    "prepare_test_data",
    "dataset_to_arrays",
]
