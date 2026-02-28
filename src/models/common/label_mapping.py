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

# 3-class label mapping: trading signals -> ML class indices
LABEL_TO_CLASS = {-1: 0, 0: 1, 1: 2}

# 3-class reverse mapping: ML class indices -> trading signals
CLASS_TO_LABEL = {0: -1, 1: 0, 2: 1}

# Binary label mapping: binary labels -> ML class indices (identity)
BINARY_LABEL_TO_CLASS = {0: 0, 1: 1}

# Binary reverse mapping: ML class indices -> binary labels (identity)
BINARY_CLASS_TO_LABEL = {0: 0, 1: 1}


def map_labels_to_classes(y: np.ndarray | pd.Series, n_classes: int = 3) -> np.ndarray:
    """
    Convert trading labels to ML class indices.

    For 3-class: (-1, 0, 1) -> (0, 1, 2)
    For 2-class (binary): (0, 1) -> (0, 1)

    Parameters
    ----------
    y : np.ndarray or pd.Series
        Array of trading labels.
        3-class: values in {-1, 0, 1}
        2-class: values in {0, 1}
    n_classes : int, default 3
        Number of classes (2 for binary, 3 for standard).

    Returns
    -------
    np.ndarray
        Array of class indices.

    Raises
    ------
    ValueError
        If any label is not in the expected set.

    Examples
    --------
    >>> labels = np.array([-1, 0, 1, -1])
    >>> map_labels_to_classes(labels)
    array([0, 1, 2, 0])
    >>> binary_labels = np.array([0, 1, 1, 0])
    >>> map_labels_to_classes(binary_labels, n_classes=2)
    array([0, 1, 1, 0])
    """
    arr = np.asarray(y, dtype=np.int32)

    if n_classes == 2:
        valid_labels = np.array([0, 1])
        invalid_mask = ~np.isin(arr, valid_labels)
        if invalid_mask.any():
            invalid_vals = np.unique(arr[invalid_mask])
            raise ValueError(
                f"Invalid binary labels: {invalid_vals.tolist()}. "
                f"Expected one of {list(BINARY_LABEL_TO_CLASS.keys())}"
            )
        return arr.copy()

    # 3-class path (original)
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


def map_classes_to_labels(y: np.ndarray | pd.Series, n_classes: int = 3) -> np.ndarray:
    """
    Convert ML class indices to trading labels.

    For 3-class: (0, 1, 2) -> (-1, 0, 1)
    For 2-class (binary): (0, 1) -> (0, 1)

    Parameters
    ----------
    y : np.ndarray or pd.Series
        Array of class indices.
        3-class: values in {0, 1, 2}
        2-class: values in {0, 1}
    n_classes : int, default 3
        Number of classes (2 for binary, 3 for standard).

    Returns
    -------
    np.ndarray
        Array of trading labels.

    Raises
    ------
    ValueError
        If any class index is not in the expected set.

    Examples
    --------
    >>> classes = np.array([0, 1, 2, 0])
    >>> map_classes_to_labels(classes)
    array([-1, 0, 1, -1])
    >>> binary_classes = np.array([0, 1, 1, 0])
    >>> map_classes_to_labels(binary_classes, n_classes=2)
    array([0, 1, 1, 0])
    """
    arr = np.asarray(y, dtype=np.int32)

    if n_classes == 2:
        valid_classes = np.array([0, 1])
        invalid_mask = ~np.isin(arr, valid_classes)
        if invalid_mask.any():
            invalid_vals = np.unique(arr[invalid_mask])
            raise ValueError(
                f"Invalid binary class indices: {invalid_vals.tolist()}. "
                f"Expected one of {list(BINARY_CLASS_TO_LABEL.keys())}"
            )
        return arr.copy()

    # 3-class path (original)
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
