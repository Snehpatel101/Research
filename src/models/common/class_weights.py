"""
Class weight computation for imbalanced classification.

This module provides utilities for computing class weights that help
address class imbalance in ML training. Weights are computed inversely
proportional to class frequency, giving more importance to minority classes.
"""

from collections import Counter
from typing import Any

import numpy as np


def compute_balanced_weights(y: np.ndarray | list) -> dict[Any, float]:
    """
    Compute balanced class weights inversely proportional to frequency.

    Uses the formula: weight[class] = n_samples / (n_classes * n_samples_for_class)

    This is equivalent to sklearn's compute_class_weight with class_weight='balanced'.

    Parameters
    ----------
    y : np.ndarray | list
        Array of class labels

    Returns
    -------
    dict[Any, float]
        Dictionary mapping class labels to their weights

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([0, 0, 0, 0, 1, 1, 2])  # Imbalanced: 4 class-0, 2 class-1, 1 class-2
    >>> weights = compute_balanced_weights(y)
    >>> weights[0]  # Most common class gets lowest weight
    0.5833333333333334
    >>> weights[2]  # Rarest class gets highest weight
    2.3333333333333335
    """
    counts = Counter(y)
    n_samples = len(y)
    n_classes = len(counts)

    weights = {}
    for cls, count in counts.items():
        weights[cls] = n_samples / (n_classes * count)

    return weights


def compute_sample_weights(y: np.ndarray | list) -> np.ndarray:
    """
    Compute per-sample weights based on class frequency.

    Each sample gets the weight of its class. This is useful for models
    that accept sample_weight parameters (e.g., XGBoost, LightGBM).

    Parameters
    ----------
    y : np.ndarray | list
        Array of class labels

    Returns
    -------
    np.ndarray
        Array of sample weights, same length as y

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([0, 0, 1, 1, 1])
    >>> weights = compute_sample_weights(y)
    >>> weights
    array([1.25, 1.25, 0.83333333, 0.83333333, 0.83333333])
    """
    class_weights = compute_balanced_weights(y)
    return np.array([class_weights[label] for label in y])


def compute_class_weight_dict(y: np.ndarray | list, as_list: bool = False) -> dict | list:
    """
    Compute class weights in formats suitable for various ML frameworks.

    Parameters
    ----------
    y : np.ndarray | list
        Array of class labels
    as_list : bool, default False
        If True, return as list ordered by class index (for frameworks like Keras).
        If False, return as dict (for frameworks like XGBoost, sklearn).

    Returns
    -------
    dict | list
        Class weights in requested format

    Examples
    --------
    >>> y = np.array([0, 0, 0, 1, 1, 2])
    >>> compute_class_weight_dict(y)
    {0: 0.6666666666666666, 1: 1.0, 2: 2.0}
    >>> compute_class_weight_dict(y, as_list=True)
    [0.6666666666666666, 1.0, 2.0]
    """
    weights = compute_balanced_weights(y)

    if as_list:
        # Return as list ordered by class index
        classes = sorted(weights.keys())
        return [weights[c] for c in classes]

    return weights


__all__ = [
    "compute_balanced_weights",
    "compute_sample_weights",
    "compute_class_weight_dict",
]
