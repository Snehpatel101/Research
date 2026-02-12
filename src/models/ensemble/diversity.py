"""
Ensemble Diversity Metrics - Measure diversity between base model predictions.

Provides comprehensive diversity analysis for ensemble models following
FinRL research recommendations. High diversity among base models typically
leads to better ensemble performance due to uncorrelated errors.

Key Metrics:
- Pairwise correlation: Average correlation between base predictions
- Q-statistic (Yule's Q): Agreement measure between classifier pairs
- Disagreement: Fraction of samples where classifiers disagree
- Double fault: Fraction where both classifiers are wrong
- Entropy: Entropy of voting distribution
- KL divergence: Distribution divergence between probability outputs

Reference:
- Kuncheva & Whitaker (2003) "Measures of Diversity in Classifier Ensembles"
- FinRL research on ensemble diversity penalties
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import kl_div
from sklearn.metrics import matthews_corrcoef

logger = logging.getLogger(__name__)

# Constants for numerical stability
_EPS = 1e-10
_MIN_PROB = 1e-7


@dataclass
class DiversityAnalysisResult:
    """Result from diversity analysis for model selection."""

    selected_models: list[str]
    diversity_matrix: np.ndarray
    model_names: list[str]
    avg_diversity: float


@dataclass
class DiversityMetrics:
    """
    Container for ensemble diversity measurements.

    Attributes:
        pairwise_correlation: Average Pearson correlation between base predictions.
            Range [-1, 1]. Lower (closer to 0 or negative) = more diverse.
        q_statistic: Average Yule's Q statistic across all model pairs.
            Range [-1, 1]. Q=1: identical, Q=0: independent, Q=-1: opposite.
        disagreement: Average disagreement rate between model pairs.
            Range [0, 1]. Higher = more diverse.
        double_fault: Average double fault rate (both wrong) across pairs.
            Range [0, 1]. Lower = better (less correlated errors).
        entropy: Average entropy of voting distribution per sample.
            Higher entropy = more disagreement = more diverse.
        kl_divergence: Average KL divergence between probability distributions.
            Higher = more diverse predictions.
        diversity_score: Composite diversity score combining all metrics.
            Range [0, 1]. Higher = more diverse ensemble.
        model_pair_correlations: Pairwise correlations between each model pair.
        recommendations: Suggestions for improving diversity.
    """

    pairwise_correlation: float = 0.0
    q_statistic: float = 0.0
    disagreement: float = 0.0
    double_fault: float = 0.0
    entropy: float = 0.0
    kl_divergence: float = 0.0
    diversity_score: float = 0.0
    model_pair_correlations: dict[tuple[str, str], float] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, float | dict[str, float] | list[str]]:
        """Convert metrics to dictionary for logging/serialization."""
        return {
            "pairwise_correlation": self.pairwise_correlation,
            "q_statistic": self.q_statistic,
            "disagreement": self.disagreement,
            "double_fault": self.double_fault,
            "entropy": self.entropy,
            "kl_divergence": self.kl_divergence,
            "diversity_score": self.diversity_score,
            "model_pair_correlations": {
                f"{m1}_vs_{m2}": corr for (m1, m2), corr in self.model_pair_correlations.items()
            },
            "recommendations": self.recommendations,
        }


def compute_pairwise_correlation(predictions: np.ndarray) -> float:
    """
    Compute average pairwise Pearson correlation between base model predictions.

    Args:
        predictions: Shape (n_samples, n_models) - class predictions from each base model

    Returns:
        Average pairwise correlation. Lower values indicate more diverse predictions.
        Returns 0.0 if fewer than 2 models.
    """
    if predictions.ndim != 2:
        raise ValueError(
            f"predictions must be 2D (n_samples, n_models), got shape {predictions.shape}"
        )

    n_models = predictions.shape[1]
    if n_models < 2:
        return 0.0

    # Compute correlation matrix between model predictions
    # corrcoef expects (n_features, n_samples), so transpose
    corr_matrix = np.corrcoef(predictions.T)

    # Handle NaN values (can occur if a model has constant predictions)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # Extract upper triangle (excluding diagonal) for average
    n_pairs = n_models * (n_models - 1) // 2
    if n_pairs == 0:
        return 0.0

    upper_indices = np.triu_indices(n_models, k=1)
    pairwise_correlations = corr_matrix[upper_indices]

    return float(np.mean(pairwise_correlations))


def compute_pairwise_correlation_matrix(
    predictions: np.ndarray,
    model_names: list[str] | None = None,
) -> dict[tuple[str, str], float]:
    """
    Compute pairwise correlation matrix between all model pairs.

    Args:
        predictions: Shape (n_samples, n_models) - predictions from each model
        model_names: Optional list of model names for labeling

    Returns:
        Dictionary mapping (model_i, model_j) -> correlation coefficient
    """
    if predictions.ndim != 2:
        raise ValueError(f"predictions must be 2D, got shape {predictions.shape}")

    n_models = predictions.shape[1]
    if model_names is None:
        model_names = [f"model_{i}" for i in range(n_models)]

    if len(model_names) != n_models:
        raise ValueError(f"model_names length {len(model_names)} != n_models {n_models}")

    corr_matrix = np.corrcoef(predictions.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    result: dict[tuple[str, str], float] = {}
    for i in range(n_models):
        for j in range(i + 1, n_models):
            result[(model_names[i], model_names[j])] = float(corr_matrix[i, j])

    return result


def compute_q_statistic(
    y_true: np.ndarray,
    preds_i: np.ndarray,
    preds_j: np.ndarray,
) -> float:
    """
    Compute Yule's Q statistic for a pair of classifiers.

    Q = (N11*N00 - N01*N10) / (N11*N00 + N01*N10)

    where:
    - N11: both correct
    - N00: both wrong
    - N01: i correct, j wrong
    - N10: i wrong, j correct

    Args:
        y_true: Ground truth labels
        preds_i: Predictions from classifier i
        preds_j: Predictions from classifier j

    Returns:
        Q statistic in range [-1, 1]:
        - Q = 1: identical predictions
        - Q = 0: independent predictions (ideal diversity)
        - Q = -1: opposite predictions (maximum diversity)
    """
    if len(y_true) != len(preds_i) or len(y_true) != len(preds_j):
        raise ValueError("All arrays must have same length")

    # Compute correctness
    correct_i = preds_i == y_true
    correct_j = preds_j == y_true

    # Count joint outcomes
    n11 = np.sum(correct_i & correct_j)  # Both correct
    n00 = np.sum(~correct_i & ~correct_j)  # Both wrong
    n01 = np.sum(correct_i & ~correct_j)  # i correct, j wrong
    n10 = np.sum(~correct_i & correct_j)  # i wrong, j correct

    numerator = n11 * n00 - n01 * n10
    denominator = n11 * n00 + n01 * n10

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def compute_average_q_statistic(
    y_true: np.ndarray,
    predictions: np.ndarray,
) -> float:
    """
    Compute average Q statistic across all model pairs.

    Args:
        y_true: Ground truth labels
        predictions: Shape (n_samples, n_models) - predictions from each model

    Returns:
        Average Q statistic. Lower (closer to 0 or negative) = more diverse.
    """
    if predictions.ndim != 2:
        raise ValueError(f"predictions must be 2D, got shape {predictions.shape}")

    n_models = predictions.shape[1]
    if n_models < 2:
        return 0.0

    q_values = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            q = compute_q_statistic(y_true, predictions[:, i], predictions[:, j])
            q_values.append(q)

    return float(np.mean(q_values)) if q_values else 0.0


def compute_disagreement(preds_i: np.ndarray, preds_j: np.ndarray) -> float:
    """
    Compute disagreement measure between two classifiers.

    Disagreement = fraction of samples where classifiers make different predictions.

    Args:
        preds_i: Predictions from classifier i
        preds_j: Predictions from classifier j

    Returns:
        Disagreement rate in range [0, 1]. Higher = more diverse.
    """
    if len(preds_i) != len(preds_j):
        raise ValueError("Both prediction arrays must have same length")

    if len(preds_i) == 0:
        return 0.0

    disagreements = preds_i != preds_j
    return float(np.mean(disagreements))


def compute_average_disagreement(predictions: np.ndarray) -> float:
    """
    Compute average disagreement across all model pairs.

    Args:
        predictions: Shape (n_samples, n_models) - predictions from each model

    Returns:
        Average disagreement rate. Higher = more diverse.
    """
    if predictions.ndim != 2:
        raise ValueError(f"predictions must be 2D, got shape {predictions.shape}")

    n_models = predictions.shape[1]
    if n_models < 2:
        return 0.0

    disagreements = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            d = compute_disagreement(predictions[:, i], predictions[:, j])
            disagreements.append(d)

    return float(np.mean(disagreements)) if disagreements else 0.0


def compute_double_fault(
    y_true: np.ndarray,
    preds_i: np.ndarray,
    preds_j: np.ndarray,
) -> float:
    """
    Compute double fault measure for a pair of classifiers.

    Double fault = fraction of samples where both classifiers are wrong.
    Lower double fault indicates less correlated errors, which is desirable.

    Args:
        y_true: Ground truth labels
        preds_i: Predictions from classifier i
        preds_j: Predictions from classifier j

    Returns:
        Double fault rate in range [0, 1]. Lower = better (less correlated errors).
    """
    if len(y_true) != len(preds_i) or len(y_true) != len(preds_j):
        raise ValueError("All arrays must have same length")

    if len(y_true) == 0:
        return 0.0

    wrong_i = preds_i != y_true
    wrong_j = preds_j != y_true
    both_wrong = wrong_i & wrong_j

    return float(np.mean(both_wrong))


def compute_average_double_fault(
    y_true: np.ndarray,
    predictions: np.ndarray,
) -> float:
    """
    Compute average double fault across all model pairs.

    Args:
        y_true: Ground truth labels
        predictions: Shape (n_samples, n_models) - predictions from each model

    Returns:
        Average double fault rate. Lower = better.
    """
    if predictions.ndim != 2:
        raise ValueError(f"predictions must be 2D, got shape {predictions.shape}")

    n_models = predictions.shape[1]
    if n_models < 2:
        return 0.0

    double_faults = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            df = compute_double_fault(y_true, predictions[:, i], predictions[:, j])
            double_faults.append(df)

    return float(np.mean(double_faults)) if double_faults else 0.0


def compute_entropy_of_votes(predictions: np.ndarray, n_classes: int = 3) -> float:
    """
    Compute entropy of the voting distribution per sample.

    For each sample, compute the entropy of the vote distribution across models.
    Higher entropy indicates more disagreement among models.

    Args:
        predictions: Shape (n_samples, n_models) - class predictions from each model
        n_classes: Number of classes in the classification problem

    Returns:
        Average entropy across all samples. Higher = more diverse.
    """
    if predictions.ndim != 2:
        raise ValueError(f"predictions must be 2D, got shape {predictions.shape}")

    n_samples, n_models = predictions.shape
    if n_models < 2 or n_samples == 0:
        return 0.0

    # Ensure predictions are valid class indices (0 to n_classes-1)
    # Clip negative values and values >= n_classes
    predictions_clipped = np.clip(predictions.astype(int), 0, n_classes - 1)

    entropies = []
    for i in range(n_samples):
        # Count votes for each class
        vote_counts = np.bincount(predictions_clipped[i], minlength=n_classes)
        vote_probs = vote_counts / n_models

        # Compute entropy (with small epsilon to avoid log(0))
        vote_probs = np.clip(vote_probs, _MIN_PROB, 1.0)
        entropy = -np.sum(vote_probs * np.log(vote_probs))
        entropies.append(entropy)

    return float(np.mean(entropies))


def compute_kl_divergence(probs_i: np.ndarray, probs_j: np.ndarray) -> float:
    """
    Compute KL divergence between two probability distributions.

    KL(P || Q) measures how much P differs from Q. Higher values indicate
    more different predictions between the two models.

    Args:
        probs_i: Probability distribution from model i, shape (n_samples, n_classes)
        probs_j: Probability distribution from model j, shape (n_samples, n_classes)

    Returns:
        Average KL divergence. Higher = more diverse predictions.
    """
    if probs_i.shape != probs_j.shape:
        raise ValueError(f"Shape mismatch: {probs_i.shape} vs {probs_j.shape}")

    if probs_i.ndim != 2:
        raise ValueError(f"probs must be 2D (n_samples, n_classes), got shape {probs_i.shape}")

    # Clip probabilities to avoid numerical issues
    probs_i = np.clip(probs_i, _MIN_PROB, 1.0 - _MIN_PROB)
    probs_j = np.clip(probs_j, _MIN_PROB, 1.0 - _MIN_PROB)

    # Renormalize after clipping
    probs_i = probs_i / probs_i.sum(axis=1, keepdims=True)
    probs_j = probs_j / probs_j.sum(axis=1, keepdims=True)

    # Compute KL divergence per sample (summed over classes)
    # scipy's kl_div returns element-wise divergence, sum over classes
    kl_per_sample = np.sum(kl_div(probs_i, probs_j), axis=1)

    return float(np.mean(kl_per_sample))


def compute_average_kl_divergence(probabilities: np.ndarray) -> float:
    """
    Compute average KL divergence across all model pairs.

    Args:
        probabilities: Shape (n_samples, n_models, n_classes) or dict of arrays

    Returns:
        Average KL divergence. Higher = more diverse.
    """
    if probabilities.ndim != 3:
        raise ValueError(
            f"probabilities must be 3D (n_samples, n_models, n_classes), got {probabilities.shape}"
        )

    n_samples, n_models, n_classes = probabilities.shape
    if n_models < 2:
        return 0.0

    kl_values = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            kl = compute_kl_divergence(probabilities[:, i, :], probabilities[:, j, :])
            kl_values.append(kl)

    return float(np.mean(kl_values)) if kl_values else 0.0


def compute_diversity_score(
    pairwise_correlation: float,
    q_statistic: float,
    disagreement: float,
    double_fault: float,
    entropy: float,
    kl_divergence: float,
    weights: dict[str, float] | None = None,
) -> float:
    """
    Compute composite diversity score from individual metrics.

    The score normalizes and combines multiple diversity measures into
    a single value in range [0, 1], where higher = more diverse.

    Args:
        pairwise_correlation: Average pairwise correlation [-1, 1]
        q_statistic: Average Q statistic [-1, 1]
        disagreement: Average disagreement [0, 1]
        double_fault: Average double fault [0, 1]
        entropy: Average entropy [0, inf)
        kl_divergence: Average KL divergence [0, inf)
        weights: Optional custom weights for each metric

    Returns:
        Composite diversity score in range [0, 1].
    """
    default_weights = {
        "correlation": 0.20,
        "q_statistic": 0.20,
        "disagreement": 0.20,
        "double_fault": 0.15,
        "entropy": 0.15,
        "kl_divergence": 0.10,
    }
    weights = weights or default_weights

    # Normalize each metric to [0, 1] range where higher = more diverse
    # Correlation: [-1, 1] -> [0, 1] (inverted: lower correlation = higher diversity)
    norm_correlation = (1 - pairwise_correlation) / 2

    # Q-statistic: [-1, 1] -> [0, 1] (inverted: lower Q = higher diversity)
    norm_q = (1 - q_statistic) / 2

    # Disagreement: [0, 1] -> [0, 1] (already normalized, higher = more diverse)
    norm_disagreement = disagreement

    # Double fault: [0, 1] -> [0, 1] (inverted: lower = better)
    norm_double_fault = 1 - double_fault

    # Entropy: [0, inf) -> [0, 1] using tanh normalization
    # For 3 classes, max entropy is ln(3) ~ 1.1
    norm_entropy = np.tanh(entropy / 1.1)

    # KL divergence: [0, inf) -> [0, 1] using tanh normalization
    # Typical values are 0.01-0.5, scale by 2 for better range
    norm_kl = np.tanh(kl_divergence * 2)

    # Weighted combination
    score = (
        weights["correlation"] * norm_correlation
        + weights["q_statistic"] * norm_q
        + weights["disagreement"] * norm_disagreement
        + weights["double_fault"] * norm_double_fault
        + weights["entropy"] * norm_entropy
        + weights["kl_divergence"] * norm_kl
    )

    return float(np.clip(score, 0.0, 1.0))


class DiversityAnalyzer:
    """
    Analyze diversity of ensemble base models.

    Provides comprehensive diversity analysis and recommendations for
    improving ensemble performance through model selection.

    Example:
        analyzer = DiversityAnalyzer(min_diversity_threshold=0.3)
        metrics = analyzer.analyze(
            base_predictions={"xgboost": preds_xgb, "lstm": preds_lstm},
            base_probabilities={"xgboost": probs_xgb, "lstm": probs_lstm},
            y_true=y_val,
        )
        print(f"Diversity score: {metrics.diversity_score:.3f}")
        if metrics.recommendations:
            print("Recommendations:", metrics.recommendations)
    """

    def __init__(
        self,
        min_diversity_threshold: float = 0.3,
        correlation_threshold: float = 0.8,
        n_classes: int = 3,
    ) -> None:
        """
        Initialize DiversityAnalyzer.

        Args:
            min_diversity_threshold: Minimum acceptable diversity score.
                Ensembles below this threshold trigger warnings.
            correlation_threshold: Maximum acceptable pairwise correlation.
                Model pairs above this threshold are flagged as redundant.
            n_classes: Number of classes in the classification problem.
        """
        self.min_diversity_threshold = min_diversity_threshold
        self.correlation_threshold = correlation_threshold
        self.n_classes = n_classes

    def analyze(
        self,
        base_predictions: dict[str, np.ndarray],
        base_probabilities: dict[str, np.ndarray] | None = None,
        y_true: np.ndarray | None = None,
    ) -> DiversityMetrics:
        """
        Perform comprehensive diversity analysis on base model predictions.

        Args:
            base_predictions: Dict mapping model_name -> predictions (n_samples,)
            base_probabilities: Dict mapping model_name -> probabilities (n_samples, n_classes).
                If None, KL divergence will be 0.
            y_true: Ground truth labels. Required for Q-statistic and double fault.

        Returns:
            DiversityMetrics containing all computed metrics and recommendations.
        """
        model_names = list(base_predictions.keys())
        n_models = len(model_names)

        if n_models < 2:
            logger.warning("Need at least 2 models for diversity analysis")
            return DiversityMetrics(recommendations=["Add more base models (minimum 2 required)"])

        # Stack predictions into matrix (n_samples, n_models)
        first_preds = base_predictions[model_names[0]]
        n_samples = len(first_preds)

        predictions_matrix = np.zeros((n_samples, n_models), dtype=first_preds.dtype)
        for i, name in enumerate(model_names):
            preds = base_predictions[name]
            if len(preds) != n_samples:
                raise ValueError(f"Model {name} has {len(preds)} predictions, expected {n_samples}")
            predictions_matrix[:, i] = preds

        # Compute pairwise correlations
        pairwise_corr = compute_pairwise_correlation(predictions_matrix)
        corr_matrix = compute_pairwise_correlation_matrix(predictions_matrix, model_names)

        # Compute disagreement
        disagreement = compute_average_disagreement(predictions_matrix)

        # Compute entropy
        entropy = compute_entropy_of_votes(predictions_matrix, self.n_classes)

        # Compute metrics requiring ground truth
        q_statistic = 0.0
        double_fault = 0.0
        if y_true is not None:
            if len(y_true) != n_samples:
                raise ValueError(f"y_true has {len(y_true)} samples, expected {n_samples}")
            q_statistic = compute_average_q_statistic(y_true, predictions_matrix)
            double_fault = compute_average_double_fault(y_true, predictions_matrix)

        # Compute KL divergence if probabilities provided
        kl_divergence = 0.0
        if base_probabilities is not None:
            # Stack probabilities into 3D array (n_samples, n_models, n_classes)
            first_probs = base_probabilities[model_names[0]]
            n_classes = first_probs.shape[1] if first_probs.ndim > 1 else self.n_classes

            probs_matrix = np.zeros((n_samples, n_models, n_classes))
            for i, name in enumerate(model_names):
                probs = base_probabilities.get(name)
                if probs is not None and probs.shape == (n_samples, n_classes):
                    probs_matrix[:, i, :] = probs

            kl_divergence = compute_average_kl_divergence(probs_matrix)

        # Compute composite diversity score
        diversity_score = compute_diversity_score(
            pairwise_correlation=pairwise_corr,
            q_statistic=q_statistic,
            disagreement=disagreement,
            double_fault=double_fault,
            entropy=entropy,
            kl_divergence=kl_divergence,
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            pairwise_corr=pairwise_corr,
            q_statistic=q_statistic,
            disagreement=disagreement,
            diversity_score=diversity_score,
            corr_matrix=corr_matrix,
        )

        logger.info(
            f"Diversity analysis complete: score={diversity_score:.3f}, "
            f"correlation={pairwise_corr:.3f}, disagreement={disagreement:.3f}"
        )

        return DiversityMetrics(
            pairwise_correlation=pairwise_corr,
            q_statistic=q_statistic,
            disagreement=disagreement,
            double_fault=double_fault,
            entropy=entropy,
            kl_divergence=kl_divergence,
            diversity_score=diversity_score,
            model_pair_correlations=corr_matrix,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        pairwise_corr: float,
        q_statistic: float,
        disagreement: float,
        diversity_score: float,
        corr_matrix: dict[tuple[str, str], float],
    ) -> list[str]:
        """Generate actionable recommendations based on diversity metrics."""
        recommendations: list[str] = []

        # Check overall diversity
        if diversity_score < self.min_diversity_threshold:
            recommendations.append(
                f"Low diversity score ({diversity_score:.3f} < {self.min_diversity_threshold}). "
                "Consider adding models from different families (e.g., mix boosting with neural)."
            )

        # Check pairwise correlation
        if pairwise_corr > self.correlation_threshold:
            recommendations.append(
                f"High average correlation ({pairwise_corr:.3f}). "
                "Base models may be too similar. Consider using diverse architectures."
            )

        # Find highly correlated pairs
        high_corr_pairs = [
            (m1, m2, corr)
            for (m1, m2), corr in corr_matrix.items()
            if corr > self.correlation_threshold
        ]
        if high_corr_pairs:
            for m1, m2, corr in high_corr_pairs[:3]:  # Limit to top 3
                recommendations.append(
                    f"Models '{m1}' and '{m2}' are highly correlated ({corr:.3f}). "
                    "Consider removing one or replacing with a different model type."
                )

        # Check Q-statistic
        if q_statistic > 0.8:
            recommendations.append(
                f"High Q-statistic ({q_statistic:.3f}) indicates near-identical predictions. "
                "Ensemble may not benefit from combining these models."
            )

        # Check disagreement
        if disagreement < 0.1:
            recommendations.append(
                f"Low disagreement ({disagreement:.3f}). Models rarely disagree. "
                "Consider adding models with different inductive biases."
            )

        return recommendations

    def suggest_model_removal(
        self,
        base_predictions: dict[str, np.ndarray],
        y_true: np.ndarray | None = None,
    ) -> list[str]:
        """
        Suggest models to remove due to redundancy (low diversity).

        Uses pairwise correlations to identify models that provide
        similar predictions and could be removed without losing diversity.

        Args:
            base_predictions: Dict mapping model_name -> predictions
            y_true: Optional ground truth for accuracy-based filtering

        Returns:
            List of model names that are redundant and could be removed.
        """
        model_names = list(base_predictions.keys())
        n_models = len(model_names)

        if n_models < 3:
            # Need at least 3 models to suggest removal
            return []

        # Stack predictions
        n_samples = len(base_predictions[model_names[0]])
        predictions_matrix = np.zeros((n_samples, n_models))
        for i, name in enumerate(model_names):
            predictions_matrix[:, i] = base_predictions[name]

        # Compute correlation matrix
        corr_matrix = compute_pairwise_correlation_matrix(predictions_matrix, model_names)

        # Find highly correlated pairs
        high_corr_pairs: list[tuple[str, str, float]] = []
        for (m1, m2), corr in corr_matrix.items():
            if corr > self.correlation_threshold:
                high_corr_pairs.append((m1, m2, corr))

        if not high_corr_pairs:
            return []

        # Sort by correlation (highest first)
        high_corr_pairs.sort(key=lambda x: x[2], reverse=True)

        # If ground truth available, prefer removing the less accurate model
        removals: list[str] = []
        if y_true is not None:
            accuracies = {}
            for name in model_names:
                preds = base_predictions[name]
                accuracies[name] = float(np.mean(preds == y_true))

            for m1, m2, _ in high_corr_pairs:
                if m1 in removals or m2 in removals:
                    continue
                # Remove the less accurate one
                if accuracies.get(m1, 0) < accuracies.get(m2, 0):
                    removals.append(m1)
                else:
                    removals.append(m2)
        else:
            # Without accuracy, just suggest one from each pair
            for m1, m2, _ in high_corr_pairs:
                if m1 not in removals and m2 not in removals:
                    removals.append(m2)  # Arbitrary choice

        # Ensure we don't remove too many models
        max_removals = n_models - 2
        return removals[:max_removals]


def compute_mcc_diversity_matrix(predictions: list[np.ndarray]) -> np.ndarray:
    """
    Compute pairwise diversity (disagreement) between models using MCC.

    Uses Matthews Correlation Coefficient (MCC) between predictions.
    Higher values = more diverse (less correlated).

    Args:
        predictions: List of prediction arrays (n_models arrays, each n_samples)

    Returns:
        Diversity matrix (n_models, n_models)
    """
    n_models = len(predictions)
    diversity_matrix = np.zeros((n_models, n_models))

    for i in range(n_models):
        for j in range(i + 1, n_models):
            # MCC ranges from -1 to 1
            # Convert to diversity: diversity = 1 - abs(mcc)
            try:
                mcc = matthews_corrcoef(predictions[i], predictions[j])
                diversity = 1.0 - abs(mcc)
            except Exception as e:
                logger.warning(
                    f"MCC diversity calculation failed for models {i},{j}: {e}. Using default 0.5."
                )
                diversity = 0.5  # Default if calculation fails

            diversity_matrix[i, j] = diversity
            diversity_matrix[j, i] = diversity

    # Diagonal is 0 (perfect correlation with self)
    np.fill_diagonal(diversity_matrix, 0.0)

    return diversity_matrix


def select_diverse_models(
    oof_predictions: dict[str, Any],  # model_name -> OOFPrediction
    min_diversity: float = 0.3,
    max_models: int = 10,
) -> DiversityAnalysisResult:
    """
    Select diverse subset of models for ensembling.

    Strategy:
    1. Start with best-performing model
    2. Iteratively add models that maximize diversity
    3. Stop when diversity threshold not met or max_models reached

    Args:
        oof_predictions: Dict of model_name -> OOFPrediction (must have .predictions and .metrics)
        min_diversity: Minimum required diversity (0.0 to 1.0)
        max_models: Maximum models to select

    Returns:
        DiversityAnalysisResult with selected models and analysis
    """
    if len(oof_predictions) == 0:
        return DiversityAnalysisResult([], np.array([]), [], 0.0)

    model_names = list(oof_predictions.keys())

    # Handle case where we have fewer models than max
    if len(oof_predictions) <= max_models:
        logger.info(f"Only {len(model_names)} models, selecting all")
        return DiversityAnalysisResult(
            selected_models=model_names,
            diversity_matrix=np.zeros((len(model_names), len(model_names))),
            model_names=model_names,
            avg_diversity=1.0,
        )

    # Get predictions as arrays
    predictions = []
    for name in model_names:
        oof = oof_predictions[name]
        if hasattr(oof, "predictions"):
            predictions.append(np.asarray(oof.predictions))
        else:
            predictions.append(np.asarray(oof))

    # Compute diversity matrix using MCC
    diversity_matrix = compute_mcc_diversity_matrix(predictions)

    # Get accuracies/scores for each model (prefer better models)
    scores: list[float] = []
    for name in model_names:
        oof = oof_predictions[name]
        if hasattr(oof, "metrics") and isinstance(oof.metrics, dict):
            score = oof.metrics.get("val_f1", oof.metrics.get("accuracy", 0.5))
        else:
            score = 0.5
        scores.append(float(score) if score is not None else 0.5)
    scores_array = np.array(scores)

    # Start with best model
    selected_indices = [int(np.argmax(scores_array))]
    logger.info(f"Starting with best model: {model_names[selected_indices[0]]}")

    # Greedily add diverse models
    while len(selected_indices) < max_models:
        # Compute average diversity of each candidate to selected models
        avg_diversity = np.mean(diversity_matrix[selected_indices, :], axis=0)

        # Mask already selected
        avg_diversity[selected_indices] = -1.0

        # Find most diverse candidate
        next_idx = int(np.argmax(avg_diversity))

        # Check if meets diversity threshold
        if avg_diversity[next_idx] < min_diversity:
            logger.info(
                f"Stopping: next best diversity {avg_diversity[next_idx]:.3f} < threshold {min_diversity}"
            )
            break

        selected_indices.append(next_idx)
        logger.info(f"Added {model_names[next_idx]} (diversity={avg_diversity[next_idx]:.3f})")

    selected_names = [model_names[i] for i in selected_indices]
    overall_diversity = np.mean(diversity_matrix[np.ix_(selected_indices, selected_indices)])

    logger.info(f"Selected {len(selected_names)}/{len(model_names)} diverse models")
    logger.info(f"Average pairwise diversity: {overall_diversity:.3f}")

    return DiversityAnalysisResult(
        selected_models=selected_names,
        diversity_matrix=diversity_matrix,
        model_names=model_names,
        avg_diversity=overall_diversity,
    )


def filter_correlated_models(
    oof_predictions: dict[str, Any],
    correlation_threshold: float = 0.9,
) -> dict[str, Any]:
    """
    Remove models that are too correlated with better-performing models.

    Args:
        oof_predictions: Dict of model_name -> OOFPrediction
        correlation_threshold: Remove if correlation > this (default 0.9)

    Returns:
        Filtered dict with correlated models removed
    """
    result = select_diverse_models(
        oof_predictions,
        min_diversity=1.0 - correlation_threshold,
        max_models=len(oof_predictions),
    )

    return {k: v for k, v in oof_predictions.items() if k in result.selected_models}


def compute_kl_diversity_penalty(
    probabilities: np.ndarray,
    target_kl: float = 0.1,
) -> float:
    """
    Compute KL divergence penalty for training loss (per FinRL research).

    This penalty encourages the meta-learner to maintain diversity
    by penalizing predictions that are too similar to the uniform
    weighted average of base model predictions.

    Args:
        probabilities: Shape (n_samples, n_models, n_classes) - probability outputs
        target_kl: Target minimum KL divergence between model pairs

    Returns:
        Penalty term to add to training loss. Higher when diversity is low.
    """
    if probabilities.ndim != 3:
        raise ValueError(f"probabilities must be 3D, got shape {probabilities.shape}")

    n_samples, n_models, n_classes = probabilities.shape
    if n_models < 2:
        return 0.0

    # Compute average KL divergence
    avg_kl = compute_average_kl_divergence(probabilities)

    # Penalty is high when avg_kl is below target
    # Uses hinge loss: max(0, target_kl - avg_kl)
    penalty = max(0.0, target_kl - avg_kl)

    return float(penalty)


__all__ = [
    "DiversityAnalysisResult",
    "DiversityMetrics",
    "DiversityAnalyzer",
    "compute_pairwise_correlation",
    "compute_pairwise_correlation_matrix",
    "compute_q_statistic",
    "compute_average_q_statistic",
    "compute_disagreement",
    "compute_average_disagreement",
    "compute_double_fault",
    "compute_average_double_fault",
    "compute_entropy_of_votes",
    "compute_kl_divergence",
    "compute_average_kl_divergence",
    "compute_diversity_score",
    "compute_kl_diversity_penalty",
    "compute_mcc_diversity_matrix",
    "select_diverse_models",
    "filter_correlated_models",
]
