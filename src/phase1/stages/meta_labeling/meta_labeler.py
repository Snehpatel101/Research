"""
Meta-Labeling Core Logic.

This module implements the core meta-labeling algorithm as described by Lopez de Prado
in "Advances in Financial Machine Learning" (AFML Chapter 3).

Meta-labeling transforms the trading problem:
- Original: Predict direction {-1, 0, 1}
- Meta-labeling: Given a primary signal, predict if it's correct {0, 1}

The meta-label is 1 if the primary prediction was correct, 0 otherwise.
This binary classification problem is easier to model and allows for
confidence-based position sizing.

Engineering Rules Applied:
- Clear contracts with type hints
- Input validation at boundaries
- Fail fast with explicit errors
- Comprehensive quality metrics
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# =============================================================================
# CONSTANTS
# =============================================================================

# Invalid label sentinel value (samples to exclude)
INVALID_LABEL = -99

# Meta-label values
META_LABEL_CORRECT = 1  # Primary prediction was correct
META_LABEL_INCORRECT = 0  # Primary prediction was incorrect
META_LABEL_INVALID = -99  # Invalid/excluded sample


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class MetaLabelingConfig:
    """Configuration for meta-labeling.

    Attributes:
        horizon: Label horizon for forward returns
        primary_signal_column: Column name containing primary predictions
        return_column: Column name containing forward returns (optional)
        include_neutrals: Whether to include neutral signals in meta-labeling
        min_confidence_threshold: Minimum confidence for valid predictions
        correctness_margin_type: How to measure correctness ('binary', 'magnitude')
    """

    horizon: int
    primary_signal_column: str | None = None
    return_column: str | None = None
    include_neutrals: bool = False
    min_confidence_threshold: float = 0.0
    correctness_margin_type: str = "binary"

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {self.horizon}")

        if self.correctness_margin_type not in {"binary", "magnitude"}:
            raise ValueError(
                f"correctness_margin_type must be 'binary' or 'magnitude', "
                f"got {self.correctness_margin_type}"
            )

        # Set default column names based on horizon
        if self.primary_signal_column is None:
            self.primary_signal_column = f"primary_pred_h{self.horizon}"

        if self.return_column is None:
            self.return_column = f"fwd_return_h{self.horizon}"


@dataclass
class MetaLabelResult:
    """Results from meta-labeling.

    Attributes:
        meta_labels: Array of meta-labels {0, 1, -99}
        primary_signals: Array of primary predictions
        correctness_margin: Array of signed margins (return * sign)
        quality_metrics: Dictionary of quality metrics
        metadata: Additional metadata
    """

    meta_labels: np.ndarray
    primary_signals: np.ndarray
    correctness_margin: np.ndarray
    quality_metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        """Total number of samples."""
        return len(self.meta_labels)

    @property
    def n_valid(self) -> int:
        """Number of valid (non-excluded) samples."""
        return (self.meta_labels != META_LABEL_INVALID).sum()

    @property
    def n_correct(self) -> int:
        """Number of correct predictions."""
        return (self.meta_labels == META_LABEL_CORRECT).sum()

    @property
    def n_incorrect(self) -> int:
        """Number of incorrect predictions."""
        return (self.meta_labels == META_LABEL_INCORRECT).sum()

    @property
    def accuracy(self) -> float:
        """Primary model accuracy on valid samples."""
        if self.n_valid == 0:
            return 0.0
        return self.n_correct / self.n_valid


# =============================================================================
# META-LABEL GENERATOR
# =============================================================================


class MetaLabelGenerator:
    """
    Generate meta-labels for bet sizing.

    This class creates binary meta-labels indicating whether a primary
    prediction was correct. The meta-labels can then be used to train
    a secondary model for position sizing.

    The workflow:
    1. Primary model predicts direction {-1, 1}
    2. MetaLabelGenerator checks if prediction matches actual return direction
    3. Meta-label = 1 if correct, 0 if incorrect, -99 if invalid

    Attributes:
        config: MetaLabelingConfig with settings
        last_result: Most recent MetaLabelResult (after generate())

    Example:
        >>> generator = MetaLabelGenerator(horizon=20)
        >>> result = generator.generate(
        ...     y_true=direction_labels,
        ...     y_primary=primary_predictions,
        ...     returns=forward_returns
        ... )
        >>> print(f"Primary accuracy: {result.accuracy:.2%}")
    """

    def __init__(
        self,
        horizon: int = 20,
        include_neutrals: bool = False,
        correctness_margin_type: str = "binary",
        **kwargs: Any,
    ):
        """
        Initialize MetaLabelGenerator.

        Args:
            horizon: Label horizon for returns (default 20)
            include_neutrals: Include neutral predictions (default False)
            correctness_margin_type: 'binary' or 'magnitude' (default 'binary')
            **kwargs: Additional config parameters
        """
        self.config = MetaLabelingConfig(
            horizon=horizon,
            include_neutrals=include_neutrals,
            correctness_margin_type=correctness_margin_type,
            **kwargs,
        )
        self.last_result: MetaLabelResult | None = None

    def generate(
        self,
        y_true: np.ndarray,
        y_primary: np.ndarray,
        returns: np.ndarray | None = None,
    ) -> MetaLabelResult:
        """
        Generate meta-labels from primary predictions and true labels.

        Args:
            y_true: True direction labels {-1, 0, 1}
            y_primary: Primary model predictions {-1, 1} (or {-1, 0, 1})
            returns: Optional forward returns for magnitude-based correctness

        Returns:
            MetaLabelResult with meta-labels and metrics

        Raises:
            ValueError: If inputs are invalid or incompatible
        """
        # Validate inputs
        y_true = np.asarray(y_true).ravel()
        y_primary = np.asarray(y_primary).ravel()

        if len(y_true) != len(y_primary):
            raise ValueError(
                f"y_true and y_primary must have same length, "
                f"got {len(y_true)} and {len(y_primary)}"
            )

        if len(y_true) == 0:
            raise ValueError("Input arrays are empty")

        n = len(y_true)

        # Validate returns if provided
        if returns is not None:
            returns = np.asarray(returns).ravel()
            if len(returns) != n:
                raise ValueError(
                    f"returns must have same length as labels, "
                    f"got {len(returns)} vs {n}"
                )

        logger.info(f"Generating meta-labels for {n} samples, horizon={self.config.horizon}")

        # Initialize output arrays
        meta_labels = np.full(n, META_LABEL_INVALID, dtype=np.int8)
        correctness_margin = np.zeros(n, dtype=np.float32)

        # Generate meta-labels
        for i in range(n):
            true_dir = y_true[i]
            pred_dir = y_primary[i]

            # Skip invalid true labels
            if true_dir == INVALID_LABEL:
                continue

            # Handle neutral predictions
            if pred_dir == 0:
                if self.config.include_neutrals:
                    # Neutral prediction is "correct" if true is also neutral
                    meta_labels[i] = META_LABEL_CORRECT if true_dir == 0 else META_LABEL_INCORRECT
                # else: leave as invalid
                continue

            # Handle neutral true labels (no direction)
            if true_dir == 0:
                # Primary predicted a direction, but true is neutral
                # This is incorrect (predicted signal when there was none)
                meta_labels[i] = META_LABEL_INCORRECT
                continue

            # Both have direction: check if they match
            # Correct if signs match (both positive or both negative)
            is_correct = np.sign(pred_dir) == np.sign(true_dir)
            meta_labels[i] = META_LABEL_CORRECT if is_correct else META_LABEL_INCORRECT

            # Compute correctness margin
            if returns is not None and not np.isnan(returns[i]):
                # Margin = return * prediction_sign
                # Positive margin = correct direction, negative = incorrect
                correctness_margin[i] = returns[i] * np.sign(pred_dir)
            else:
                # Binary margin: 1 if correct, -1 if incorrect
                correctness_margin[i] = 1.0 if is_correct else -1.0

        # Compute quality metrics
        quality_metrics = self._compute_quality_metrics(
            meta_labels, y_primary, correctness_margin
        )

        # Create result
        result = MetaLabelResult(
            meta_labels=meta_labels,
            primary_signals=y_primary.copy(),
            correctness_margin=correctness_margin,
            quality_metrics=quality_metrics,
            metadata={
                "horizon": self.config.horizon,
                "include_neutrals": self.config.include_neutrals,
                "correctness_margin_type": self.config.correctness_margin_type,
            },
        )

        self.last_result = result
        self._log_summary(result)

        return result

    def generate_from_dataframe(
        self,
        df: pd.DataFrame,
        primary_signal_column: str | None = None,
        true_label_column: str | None = None,
        return_column: str | None = None,
    ) -> tuple[MetaLabelResult, pd.DataFrame]:
        """
        Generate meta-labels from a DataFrame.

        Convenience method that extracts columns from DataFrame and adds
        meta-label columns back to it.

        Args:
            df: DataFrame with labels and predictions
            primary_signal_column: Column with primary predictions
            true_label_column: Column with true direction labels
            return_column: Column with forward returns (optional)

        Returns:
            Tuple of (MetaLabelResult, DataFrame with meta-label columns)

        Raises:
            KeyError: If required columns are missing
            ValueError: If DataFrame is empty
        """
        if df.empty:
            raise ValueError("DataFrame is empty")

        horizon = self.config.horizon

        # Determine column names
        primary_col = primary_signal_column or self.config.primary_signal_column
        true_col = true_label_column or f"label_h{horizon}"
        return_col = return_column or self.config.return_column

        # Validate columns exist
        if primary_col not in df.columns:
            raise KeyError(
                f"Primary signal column '{primary_col}' not found. "
                f"Available: {list(df.columns)[:10]}..."
            )

        if true_col not in df.columns:
            raise KeyError(
                f"True label column '{true_col}' not found. "
                f"Available: {list(df.columns)[:10]}..."
            )

        # Extract arrays
        y_primary = df[primary_col].values
        y_true = df[true_col].values

        returns = None
        if return_col and return_col in df.columns:
            returns = df[return_col].values

        # Generate meta-labels
        result = self.generate(y_true=y_true, y_primary=y_primary, returns=returns)

        # Add to DataFrame
        df_out = df.copy()
        df_out[f"meta_label_h{horizon}"] = result.meta_labels
        df_out[f"correctness_margin_h{horizon}"] = result.correctness_margin

        return result, df_out

    def _compute_quality_metrics(
        self,
        meta_labels: np.ndarray,
        primary_signals: np.ndarray,
        correctness_margin: np.ndarray,
    ) -> dict[str, float]:
        """Compute quality metrics for the meta-labeling result."""
        n_total = len(meta_labels)
        valid_mask = meta_labels != META_LABEL_INVALID
        n_valid = valid_mask.sum()

        metrics: dict[str, float] = {
            "n_total": float(n_total),
            "n_valid": float(n_valid),
            "n_invalid": float(n_total - n_valid),
            "valid_rate": n_valid / n_total if n_total > 0 else 0.0,
        }

        if n_valid == 0:
            return metrics

        valid_labels = meta_labels[valid_mask]
        n_correct = (valid_labels == META_LABEL_CORRECT).sum()
        n_incorrect = (valid_labels == META_LABEL_INCORRECT).sum()

        # Basic counts
        metrics["n_correct"] = float(n_correct)
        metrics["n_incorrect"] = float(n_incorrect)

        # Accuracy (primary model correctness rate)
        metrics["primary_accuracy"] = n_correct / n_valid

        # Direction breakdown for valid samples
        valid_signals = primary_signals[valid_mask]
        n_long = (valid_signals > 0).sum()
        n_short = (valid_signals < 0).sum()
        metrics["n_long_signals"] = float(n_long)
        metrics["n_short_signals"] = float(n_short)

        # Accuracy by direction
        if n_long > 0:
            long_mask = (valid_signals > 0)
            long_correct = (valid_labels[long_mask] == META_LABEL_CORRECT).sum()
            metrics["long_accuracy"] = long_correct / n_long
        else:
            metrics["long_accuracy"] = 0.0

        if n_short > 0:
            short_mask = (valid_signals < 0)
            short_correct = (valid_labels[short_mask] == META_LABEL_CORRECT).sum()
            metrics["short_accuracy"] = short_correct / n_short
        else:
            metrics["short_accuracy"] = 0.0

        # Correctness margin statistics
        valid_margins = correctness_margin[valid_mask]
        metrics["avg_margin"] = float(np.mean(valid_margins))
        metrics["std_margin"] = float(np.std(valid_margins))

        # Margin for correct vs incorrect predictions
        correct_mask = valid_labels == META_LABEL_CORRECT
        if correct_mask.sum() > 0:
            metrics["avg_margin_correct"] = float(np.mean(valid_margins[correct_mask]))
        else:
            metrics["avg_margin_correct"] = 0.0

        incorrect_mask = valid_labels == META_LABEL_INCORRECT
        if incorrect_mask.sum() > 0:
            metrics["avg_margin_incorrect"] = float(np.mean(valid_margins[incorrect_mask]))
        else:
            metrics["avg_margin_incorrect"] = 0.0

        return metrics

    def _log_summary(self, result: MetaLabelResult) -> None:
        """Log summary of meta-labeling results."""
        metrics = result.quality_metrics

        logger.info("Meta-labeling summary:")
        logger.info(f"  Total samples:   {metrics['n_total']:.0f}")
        logger.info(f"  Valid samples:   {metrics['n_valid']:.0f} ({metrics['valid_rate']:.1%})")
        logger.info(f"  Correct:         {metrics['n_correct']:.0f}")
        logger.info(f"  Incorrect:       {metrics['n_incorrect']:.0f}")
        logger.info(f"  Primary accuracy: {metrics['primary_accuracy']:.2%}")

        if metrics.get("n_long_signals", 0) > 0:
            logger.info(f"  Long accuracy:   {metrics['long_accuracy']:.2%}")
        if metrics.get("n_short_signals", 0) > 0:
            logger.info(f"  Short accuracy:  {metrics['short_accuracy']:.2%}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_meta_labels(
    y_true: np.ndarray,
    y_primary: np.ndarray,
    returns: np.ndarray | None = None,
    horizon: int = 20,
) -> np.ndarray:
    """
    Convenience function to create meta-labels.

    Args:
        y_true: True direction labels {-1, 0, 1}
        y_primary: Primary model predictions {-1, 1}
        returns: Optional forward returns
        horizon: Label horizon (default 20)

    Returns:
        Meta-labels array {0, 1, -99}
    """
    generator = MetaLabelGenerator(horizon=horizon)
    result = generator.generate(y_true=y_true, y_primary=y_primary, returns=returns)
    return result.meta_labels


def add_meta_labels_to_dataframe(
    df: pd.DataFrame,
    primary_signal_column: str,
    horizon: int = 20,
) -> pd.DataFrame:
    """
    Add meta-label columns to a DataFrame.

    Args:
        df: DataFrame with labels and primary predictions
        primary_signal_column: Column with primary predictions
        horizon: Label horizon (default 20)

    Returns:
        DataFrame with meta_label_h{horizon} column added
    """
    generator = MetaLabelGenerator(horizon=horizon)
    _, df_out = generator.generate_from_dataframe(
        df=df,
        primary_signal_column=primary_signal_column,
    )
    return df_out
