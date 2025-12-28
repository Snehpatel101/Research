"""
Label mapping utilities for converting between trading signals and ML class indices.

This module provides shared constants and functions for converting between:
- Trading labels: -1 (short), 0 (neutral), 1 (long)
- ML class indices: 0, 1, 2 (required by classification models)

Used across all model implementations (boosting, neural, classical) to ensure
consistent label encoding/decoding.
"""

from typing import Union
import numpy as np
import pandas as pd


# Label mapping: trading signals -> ML class indices
LABEL_TO_CLASS = {-1: 0, 0: 1, 1: 2}

# Reverse mapping: ML class indices -> trading signals
CLASS_TO_LABEL = {0: -1, 1: 0, 2: 1}


def map_labels_to_classes(
    y: Union[np.ndarray, pd.Series]
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
    result = []
    for val in y:
        v = int(val)
        if v not in LABEL_TO_CLASS:
            raise ValueError(
                f"Invalid label: {v}. Expected one of {list(LABEL_TO_CLASS.keys())}"
            )
        result.append(LABEL_TO_CLASS[v])
    return np.array(result)


def map_classes_to_labels(
    y: Union[np.ndarray, pd.Series]
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
    result = []
    for val in y:
        v = int(val)
        if v not in CLASS_TO_LABEL:
            raise ValueError(
                f"Invalid class index: {v}. Expected one of {list(CLASS_TO_LABEL.keys())}"
            )
        result.append(CLASS_TO_LABEL[v])
    return np.array(result)
