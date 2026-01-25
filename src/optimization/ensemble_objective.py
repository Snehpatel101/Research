"""
Diversity-aware Optuna objective for ensemble optimization.

Phase 16A-B: Combines Sharpe ratio with ensemble diversity to select models that
are both profitable AND diverse from existing ensemble members.

Key Functions:
- diversity_aware_objective: Combined objective = 0.7*sharpe + 0.3*diversity
- check_feature_diversity: Constrain feature overlap <30% between models

Reference:
- Kuncheva & Whitaker (2003) "Measures of Diversity in Classifier Ensembles"
- FinRL research on ensemble diversity penalties
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna

logger = logging.getLogger(__name__)

# Default weights for combined objective
DEFAULT_SHARPE_WEIGHT = 0.7
DEFAULT_DIVERSITY_WEIGHT = 0.3

# Default maximum feature overlap threshold
DEFAULT_MAX_OVERLAP = 0.3


# =============================================================================
# 16A: DIVERSITY-AWARE OBJECTIVE
# =============================================================================


def compute_prediction_diversity(
    existing_predictions: np.ndarray,
    new_predictions: np.ndarray,
) -> float:
    """
    Compute diversity between new model predictions and existing ensemble predictions.

    Diversity is calculated as 1 - mean(absolute correlation with existing models).
    This measures how different the new model's predictions are from existing ones.

    Args:
        existing_predictions: Shape (n_samples, n_existing_models) - OOF predictions
            from existing ensemble members.
        new_predictions: Shape (n_samples,) - OOF predictions from the current trial's model.

    Returns:
        Diversity score in range [0, 1]. Higher = more diverse.
        Returns 1.0 if no existing predictions (first model is maximally diverse).
    """
    if existing_predictions is None or existing_predictions.size == 0:
        # No existing models, first model is maximally diverse
        return 1.0

    if new_predictions is None or new_predictions.size == 0:
        return 0.0

    # Ensure correct shapes
    if existing_predictions.ndim == 1:
        existing_predictions = existing_predictions.reshape(-1, 1)

    n_samples = len(new_predictions)

    if existing_predictions.shape[0] != n_samples:
        logger.warning(
            f"Shape mismatch: existing_predictions has {existing_predictions.shape[0]} samples, "
            f"new_predictions has {n_samples}. Returning 0.0 diversity."
        )
        return 0.0

    # Check for constant predictions (new_predictions)
    new_std = np.std(new_predictions)
    if new_std < 1e-10:
        # Constant predictions are considered maximally correlated (not diverse)
        return 0.0

    # Vectorized correlation computation
    # Stack new_predictions with existing_predictions and compute full correlation matrix
    all_preds = np.column_stack([new_predictions, existing_predictions])
    corr_matrix = np.corrcoef(all_preds.T)
    # Extract correlations between new_predictions (index 0) and all existing (indices 1:)
    correlations = np.abs(corr_matrix[0, 1:])

    # Handle NaN values (can occur with constant predictions in existing models)
    correlations = np.where(np.isnan(correlations), 1.0, correlations)

    if not correlations:
        return 1.0

    # Diversity = 1 - mean(absolute correlation)
    mean_abs_corr = float(np.mean(correlations))
    diversity = 1.0 - mean_abs_corr

    return max(0.0, min(1.0, diversity))


def diversity_aware_objective(
    trial: optuna.Trial,
    base_objective_value: float,
    existing_predictions: np.ndarray | None = None,
    new_predictions: np.ndarray | None = None,
    sharpe_weight: float = DEFAULT_SHARPE_WEIGHT,
    diversity_weight: float = DEFAULT_DIVERSITY_WEIGHT,
) -> float:
    """
    Combined objective: sharpe_weight * sharpe + diversity_weight * diversity.

    This objective function rewards models that are both profitable (high Sharpe)
    AND diverse from existing ensemble members (low correlation).

    Phase 16A implementation: Optuna objective = 0.7*sharpe + 0.3*diversity

    Args:
        trial: Optuna trial (used for logging/user attributes).
        base_objective_value: Sharpe ratio (or other base metric) from base objective.
            Expected range depends on the metric, typically -inf to +inf for Sharpe.
        existing_predictions: Shape (n_samples, n_existing_models) - OOF predictions
            from existing ensemble members. If None, assumes this is the first model.
        new_predictions: Shape (n_samples,) - OOF predictions from the current trial.
            If None, only base_objective_value is used.
        sharpe_weight: Weight for the Sharpe component (default: 0.7).
        diversity_weight: Weight for the diversity component (default: 0.3).

    Returns:
        Combined objective value. Higher is better.

    Example:
        >>> # In Optuna objective function:
        >>> def objective(trial):
        ...     # ... train model and get predictions ...
        ...     sharpe = compute_sharpe(val_returns)
        ...     combined = diversity_aware_objective(
        ...         trial=trial,
        ...         base_objective_value=sharpe,
        ...         existing_predictions=ensemble_oof_preds,
        ...         new_predictions=model_oof_preds,
        ...     )
        ...     return combined
    """
    # Validate weights
    if abs(sharpe_weight + diversity_weight - 1.0) > 1e-6:
        logger.warning(
            f"Weights sum to {sharpe_weight + diversity_weight:.3f}, not 1.0. "
            "Consider normalizing weights."
        )

    # If no existing predictions or new predictions, just return base objective
    if existing_predictions is None or new_predictions is None:
        # Store metrics for analysis
        trial.set_user_attr("diversity_score", 1.0)
        trial.set_user_attr("base_objective", float(base_objective_value))
        trial.set_user_attr("combined_objective", float(base_objective_value))
        return float(base_objective_value)

    # Compute diversity score
    diversity = compute_prediction_diversity(existing_predictions, new_predictions)

    # Normalize Sharpe for combination (optional, but helps with interpretability)
    # Clip Sharpe to reasonable range for combination with diversity [0, 1]
    # Typical Sharpe range: -2 to +3, we normalize to [0, 1] using tanh-like scaling
    normalized_sharpe = 0.5 * (1.0 + np.tanh(base_objective_value / 2.0))

    # Combined objective
    combined = sharpe_weight * normalized_sharpe + diversity_weight * diversity

    # Store metrics in trial for analysis
    trial.set_user_attr("diversity_score", float(diversity))
    trial.set_user_attr("base_objective", float(base_objective_value))
    trial.set_user_attr("normalized_sharpe", float(normalized_sharpe))
    trial.set_user_attr("combined_objective", float(combined))

    logger.debug(
        f"Trial {trial.number}: sharpe={base_objective_value:.3f}, "
        f"diversity={diversity:.3f}, combined={combined:.3f}"
    )

    return float(combined)


# =============================================================================
# 16B: FEATURE OVERLAP CONSTRAINT
# =============================================================================


def calculate_feature_overlap(
    features_a: list[str],
    features_b: list[str],
) -> float:
    """
    Calculate Jaccard overlap between two feature sets.

    Jaccard overlap = |intersection| / |union|

    Args:
        features_a: List of feature names for model A.
        features_b: List of feature names for model B.

    Returns:
        Overlap coefficient in range [0, 1].
        0.0 = no overlap (completely different features)
        1.0 = complete overlap (identical feature sets)
    """
    if not features_a or not features_b:
        return 0.0

    set_a = set(features_a)
    set_b = set(features_b)

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    if union == 0:
        return 0.0

    return intersection / union


def calculate_feature_overlap_ratio(
    features_a: list[str],
    features_b: list[str],
) -> float:
    """
    Calculate overlap as ratio of shared features to minimum set size.

    This is an alternative overlap metric that measures how much of the
    smaller feature set is shared with the larger one.

    Args:
        features_a: List of feature names for model A.
        features_b: List of feature names for model B.

    Returns:
        Overlap ratio in range [0, 1].
    """
    if not features_a or not features_b:
        return 0.0

    set_a = set(features_a)
    set_b = set(features_b)

    intersection = len(set_a & set_b)
    min_size = min(len(set_a), len(set_b))

    if min_size == 0:
        return 0.0

    return intersection / min_size


def check_feature_diversity(
    trial_features: list[str],
    existing_model_features: list[list[str]],
    max_overlap: float = DEFAULT_MAX_OVERLAP,
) -> bool:
    """
    Check if feature overlap is below threshold for all existing models.

    Phase 16B implementation: Constrain feature overlap <30% between models.

    This function is used during Optuna optimization to prune trials that
    would result in models with too-similar feature sets.

    Args:
        trial_features: List of feature names for the current trial's model.
        existing_model_features: List of feature lists for each existing model.
        max_overlap: Maximum allowed Jaccard overlap (default: 0.3 = 30%).

    Returns:
        True if feature overlap is below threshold for ALL existing models.
        False if any existing model has overlap >= max_overlap.

    Example:
        >>> # In Optuna objective function:
        >>> def objective(trial):
        ...     # Sample features
        ...     selected_features = sample_features(trial)
        ...
        ...     # Check diversity constraint
        ...     if not check_feature_diversity(
        ...         selected_features,
        ...         existing_model_features,
        ...         max_overlap=0.3,
        ...     ):
        ...         raise optuna.TrialPruned("Feature overlap too high")
        ...
        ...     # Continue with training...
    """
    if not existing_model_features:
        # No existing models, constraint is satisfied
        return True

    if not trial_features:
        # No features selected, constraint is satisfied
        return True

    for i, model_features in enumerate(existing_model_features):
        overlap = calculate_feature_overlap(trial_features, model_features)

        if overlap > max_overlap:
            logger.debug(f"Feature overlap {overlap:.3f} > {max_overlap:.3f} with model {i}")
            return False

    return True


def compute_feature_diversity_score(
    trial_features: list[str],
    existing_model_features: list[list[str]],
) -> float:
    """
    Compute aggregate feature diversity score against all existing models.

    This is a softer version of check_feature_diversity that returns a
    continuous score instead of a boolean, suitable for use in the objective.

    Args:
        trial_features: List of feature names for the current trial's model.
        existing_model_features: List of feature lists for each existing model.

    Returns:
        Feature diversity score in range [0, 1].
        1.0 = no overlap with any existing model (maximum diversity)
        0.0 = complete overlap with all existing models
    """
    if not existing_model_features:
        return 1.0

    if not trial_features:
        return 1.0

    overlaps = []
    for model_features in existing_model_features:
        overlap = calculate_feature_overlap(trial_features, model_features)
        overlaps.append(overlap)

    # Diversity = 1 - mean overlap
    mean_overlap = float(np.mean(overlaps))
    diversity = 1.0 - mean_overlap

    return max(0.0, min(1.0, diversity))


class EnsembleAwareObjective:
    """
    Wrapper class for creating ensemble-aware Optuna objectives.

    This class maintains state about existing ensemble members and provides
    methods for computing diversity-aware objectives and checking constraints.

    Example:
        >>> # Create ensemble-aware objective
        >>> ensemble_obj = EnsembleAwareObjective(
        ...     sharpe_weight=0.7,
        ...     diversity_weight=0.3,
        ...     max_feature_overlap=0.3,
        ... )
        >>>
        >>> # Add existing models as they are selected
        >>> ensemble_obj.add_model("xgboost", xgb_predictions, xgb_features)
        >>> ensemble_obj.add_model("lstm", lstm_predictions, lstm_features)
        >>>
        >>> # Use in Optuna
        >>> def objective(trial):
        ...     # ... train model ...
        ...     return ensemble_obj.compute_objective(
        ...         trial, sharpe, predictions, features
        ...     )
    """

    def __init__(
        self,
        sharpe_weight: float = DEFAULT_SHARPE_WEIGHT,
        diversity_weight: float = DEFAULT_DIVERSITY_WEIGHT,
        max_feature_overlap: float = DEFAULT_MAX_OVERLAP,
        prune_on_overlap: bool = True,
    ) -> None:
        """
        Initialize EnsembleAwareObjective.

        Args:
            sharpe_weight: Weight for Sharpe ratio in combined objective.
            diversity_weight: Weight for diversity in combined objective.
            max_feature_overlap: Maximum allowed feature overlap (Jaccard).
            prune_on_overlap: If True, raise TrialPruned when overlap exceeds threshold.
        """
        self.sharpe_weight = sharpe_weight
        self.diversity_weight = diversity_weight
        self.max_feature_overlap = max_feature_overlap
        self.prune_on_overlap = prune_on_overlap

        # State: existing ensemble members
        self._model_names: list[str] = []
        self._model_predictions: list[np.ndarray] = []
        self._model_features: list[list[str]] = []

    @property
    def n_models(self) -> int:
        """Number of models currently in the ensemble."""
        return len(self._model_names)

    @property
    def model_names(self) -> list[str]:
        """Names of models currently in the ensemble."""
        return self._model_names.copy()

    def add_model(
        self,
        name: str,
        predictions: np.ndarray,
        features: list[str],
    ) -> None:
        """
        Add a model to the existing ensemble state.

        Args:
            name: Model name/identifier.
            predictions: OOF predictions from this model.
            features: List of feature names used by this model.
        """
        self._model_names.append(name)
        self._model_predictions.append(predictions.copy())
        self._model_features.append(features.copy())
        logger.info(f"Added model '{name}' to ensemble state ({self.n_models} total)")

    def clear(self) -> None:
        """Clear all existing ensemble state."""
        self._model_names.clear()
        self._model_predictions.clear()
        self._model_features.clear()
        logger.info("Cleared ensemble state")

    def get_existing_predictions(self) -> np.ndarray | None:
        """
        Get stacked predictions from all existing models.

        Returns:
            Shape (n_samples, n_models) or None if no models.
        """
        if not self._model_predictions:
            return None

        return np.column_stack(self._model_predictions)

    def compute_objective(
        self,
        trial: optuna.Trial,
        base_objective_value: float,
        new_predictions: np.ndarray | None = None,
        new_features: list[str] | None = None,
    ) -> float:
        """
        Compute the diversity-aware objective for a trial.

        Args:
            trial: Optuna trial.
            base_objective_value: Sharpe ratio or base metric.
            new_predictions: OOF predictions from current trial's model.
            new_features: Features used by current trial's model.

        Returns:
            Combined objective value.

        Raises:
            optuna.TrialPruned: If prune_on_overlap=True and overlap exceeds threshold.
        """
        # Check feature diversity constraint
        has_features_to_check = new_features is not None and self._model_features
        if has_features_to_check and not check_feature_diversity(
            new_features,
            self._model_features,
            max_overlap=self.max_feature_overlap,
        ):
            if self.prune_on_overlap:
                raise optuna.TrialPruned(
                    f"Feature overlap exceeds {self.max_feature_overlap:.0%} threshold"
                )
            # Log warning but continue
            logger.warning(f"Trial {trial.number}: Feature overlap exceeds threshold")

        # Compute diversity-aware objective
        existing_preds = self.get_existing_predictions()

        return diversity_aware_objective(
            trial=trial,
            base_objective_value=base_objective_value,
            existing_predictions=existing_preds,
            new_predictions=new_predictions,
            sharpe_weight=self.sharpe_weight,
            diversity_weight=self.diversity_weight,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize ensemble state to dictionary."""
        return {
            "sharpe_weight": self.sharpe_weight,
            "diversity_weight": self.diversity_weight,
            "max_feature_overlap": self.max_feature_overlap,
            "prune_on_overlap": self.prune_on_overlap,
            "n_models": self.n_models,
            "model_names": self._model_names.copy(),
            "model_features": [f.copy() for f in self._model_features],
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # 16A: Diversity-aware objective
    "diversity_aware_objective",
    "compute_prediction_diversity",
    "DEFAULT_SHARPE_WEIGHT",
    "DEFAULT_DIVERSITY_WEIGHT",
    # 16B: Feature overlap constraint
    "calculate_feature_overlap",
    "calculate_feature_overlap_ratio",
    "check_feature_diversity",
    "compute_feature_diversity_score",
    "DEFAULT_MAX_OVERLAP",
    # Combined wrapper
    "EnsembleAwareObjective",
]
