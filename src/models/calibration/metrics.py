"""
Calibration Metrics for Probability Outputs.

Provides metrics to evaluate how well predicted probabilities
match actual frequencies:
- Brier Score: MSE between predicted probabilities and one-hot labels
- ECE: Expected Calibration Error (gap between confidence and accuracy)
- Reliability Bins: Data for reliability diagrams
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReliabilityBins:
    """Data for reliability diagram visualization."""

    bin_centers: np.ndarray
    bin_accuracies: np.ndarray
    bin_confidences: np.ndarray
    bin_counts: np.ndarray
    n_bins: int

    def to_dict(self) -> dict[str, list]:
        """Convert to serializable dict."""
        return {
            "bin_centers": self.bin_centers.tolist(),
            "bin_accuracies": self.bin_accuracies.tolist(),
            "bin_confidences": self.bin_confidences.tolist(),
            "bin_counts": self.bin_counts.tolist(),
            "n_bins": self.n_bins,
        }


def _normalize_labels(y_true: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Normalize labels to 0-indexed format.

    Handles labels in {-1, 0, 1} format (trading signals) by mapping to {0, 1, 2}.

    Args:
        y_true: True class labels (may be -1, 0, 1 or 0, 1, 2)
        n_classes: Number of classes

    Returns:
        Labels normalized to {0, 1, ..., n_classes-1}
    """
    y_int = y_true.astype(int)

    # Check if labels contain negative values (trading signal format)
    if y_int.min() < 0:
        # Map -1 -> 0, 0 -> 1, 1 -> 2
        y_int = y_int + 1

    return y_int


def compute_brier_score(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """
    Compute multi-class Brier score.

    Brier score measures the mean squared error between predicted
    probabilities and one-hot encoded true labels. Lower is better.

    Args:
        y_true: True class labels, shape (n_samples,). Can be {-1,0,1} or {0,1,2}.
        probabilities: Predicted probabilities, shape (n_samples, n_classes)

    Returns:
        Brier score in [0, 2] for 3-class. 0 = perfect, 2 = worst.
    """
    if len(y_true) == 0:
        return 0.0

    n_classes = probabilities.shape[1]
    y_int = _normalize_labels(y_true, n_classes)

    # One-hot encode
    y_onehot = np.zeros_like(probabilities)
    y_onehot[np.arange(len(y_int)), y_int] = 1.0

    # MSE between probabilities and one-hot
    brier = np.mean(np.sum((probabilities - y_onehot) ** 2, axis=1))
    return float(brier)


def compute_ece(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error.

    ECE measures the weighted average gap between confidence and
    accuracy across confidence bins. Lower is better.

    Args:
        y_true: True class labels, shape (n_samples,). Can be {-1,0,1} or {0,1,2}.
        probabilities: Predicted probabilities, shape (n_samples, n_classes)
        n_bins: Number of confidence bins

    Returns:
        ECE in [0, 1]. 0 = perfectly calibrated.
    """
    if len(y_true) == 0:
        return 0.0

    n_classes = probabilities.shape[1]
    y_int = _normalize_labels(y_true, n_classes)

    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    accuracies = (predictions == y_int).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_size = mask.sum()

        if bin_size > 0:
            bin_accuracy = accuracies[mask].mean()
            bin_confidence = confidences[mask].mean()
            ece += bin_size * abs(bin_accuracy - bin_confidence)

    return float(ece / total_samples) if total_samples > 0 else 0.0


def compute_reliability_bins(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> ReliabilityBins:
    """
    Compute reliability diagram bins.

    For each confidence bin, computes the actual accuracy and mean
    confidence. A well-calibrated model has accuracy = confidence.

    Args:
        y_true: True class labels, shape (n_samples,). Can be {-1,0,1} or {0,1,2}.
        probabilities: Predicted probabilities, shape (n_samples, n_classes)
        n_bins: Number of confidence bins

    Returns:
        ReliabilityBins with bin data for plotting
    """
    n_classes = probabilities.shape[1]
    y_int = _normalize_labels(y_true, n_classes)

    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    accuracies = (predictions == y_int).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_counts[i] = mask.sum()

        if bin_counts[i] > 0:
            bin_accuracies[i] = accuracies[mask].mean()
            bin_confidences[i] = confidences[mask].mean()

    return ReliabilityBins(
        bin_centers=bin_centers,
        bin_accuracies=bin_accuracies,
        bin_confidences=bin_confidences,
        bin_counts=bin_counts,
        n_bins=n_bins,
    )


__all__ = [
    "ReliabilityBins",
    "compute_brier_score",
    "compute_ece",
    "compute_reliability_bins",
]
