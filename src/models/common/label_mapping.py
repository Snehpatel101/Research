"""
Label mapping utilities for converting between trading signals and ML class indices.

This module provides shared constants and functions for converting between:
- Trading labels: -1 (short), 0 (neutral), 1 (long)
- ML class indices: 0, 1, 2 (required by classification models)

Used across all model implementations (boosting, neural, classical) to ensure
consistent label encoding/decoding.
"""


import numpy as np
import pandas as pd

# Label mapping: trading signals -> ML class indices
LABEL_TO_CLASS = {-1: 0, 0: 1, 1: 2}

# Reverse mapping: ML class indices -> trading signals
CLASS_TO_LABEL = {0: -1, 1: 0, 2: 1}


def map_labels_to_classes(
    y: np.ndarray | pd.Series
) -> np.ndarray:
    """
    Convert trading labels (-1, 0, 1) to ML class indices (0, 1, 2).

    Parameters
    ----------
    y : np.ndarray or pd.Series
        Array of trading labels with values in {-1, 0, 1}

    Returns
    -------
    np.ndarray
        Array of class indices with values in {0, 1, 2}

    Raises
    ------
    ValueError
        If any label is not in {-1, 0, 1}

    Examples
    --------
    >>> labels = np.array([-1, 0, 1, -1])
    >>> map_labels_to_classes(labels)
    array([0, 1, 2, 0])
    """
    # Vectorized implementation for 100x+ speedup on large arrays
    arr = np.asarray(y, dtype=np.int32)

    # Validate all values are in {-1, 0, 1}
    valid_labels = np.array([-1, 0, 1])
    invalid_mask = ~np.isin(arr, valid_labels)
    if invalid_mask.any():
        invalid_vals = np.unique(arr[invalid_mask])
        raise ValueError(
            f"Invalid labels: {invalid_vals.tolist()}. "
            f"Expected one of {list(LABEL_TO_CLASS.keys())}"
        )

    # Vectorized mapping: -1 -> 0, 0 -> 1, 1 -> 2
    return (arr + 1).astype(np.int32)


def map_classes_to_labels(
    y: np.ndarray | pd.Series
) -> np.ndarray:
    """
    Convert ML class indices (0, 1, 2) to trading labels (-1, 0, 1).

    Parameters
    ----------
    y : np.ndarray or pd.Series
        Array of class indices with values in {0, 1, 2}

    Returns
    -------
    np.ndarray
        Array of trading labels with values in {-1, 0, 1}

    Raises
    ------
    ValueError
        If any class index is not in {0, 1, 2}

    Examples
    --------
    >>> classes = np.array([0, 1, 2, 0])
    >>> map_classes_to_labels(classes)
    array([-1, 0, 1, -1])
    """
    # Vectorized implementation for 100x+ speedup on large arrays
    arr = np.asarray(y, dtype=np.int32)

    # Validate all values are in {0, 1, 2}
    valid_classes = np.array([0, 1, 2])
    invalid_mask = ~np.isin(arr, valid_classes)
    if invalid_mask.any():
        invalid_vals = np.unique(arr[invalid_mask])
        raise ValueError(
            f"Invalid class indices: {invalid_vals.tolist()}. "
            f"Expected one of {list(CLASS_TO_LABEL.keys())}"
        )

    # Vectorized mapping: 0 -> -1, 1 -> 0, 2 -> 1
    return (arr - 1).astype(np.int32)
