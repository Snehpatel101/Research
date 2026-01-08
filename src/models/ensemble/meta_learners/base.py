"""
Base utilities for meta-learner implementations.

Shared functionality used across all meta-learner models.
"""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute softmax probabilities.

    Args:
        x: Decision values of shape (n_samples, n_classes)

    Returns:
        Probability distribution over classes
    """
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


__all__ = ["softmax"]
