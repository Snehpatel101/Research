"""
Meta-Labeling Strategy.

Implements meta-labeling as described by Lopez de Prado in "Advances in Financial
Machine Learning". Meta-labeling generates confidence labels (0/1) for a primary
signal, enabling a two-model approach:
  1. Primary model: Predicts direction (side)
  2. Meta model: Predicts whether to act on the primary signal (size)

This approach separates the decision of what to trade (side) from whether to trade
(size), allowing each model to be optimized independently.

Engineering Rules Applied:
- Clear contracts with type hints
- Input validation at boundaries
- Fail fast with explicit errors
- Comprehensive quality metrics
"""

import logging
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from .base import LabelingResult, LabelingStrategy, LabelingType

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class BetSizeMethod(Enum):
    """Methods for computing bet size in meta-labeling."""

    PROBABILITY = 'probability'  # Use model probability as bet size
    FIXED = 'fixed'  # Use fixed bet size (1.0)


class MetaLabeler(LabelingStrategy):
    """
    Meta-labeling strategy for generating confidence labels on primary signals.

    Meta-labeling answers the question: "Given a primary signal, should we act on it?"
    The output is binary (0/1) indicating whether the primary signal was correct.

    This enables a two-model approach where:
    - Primary model predicts direction (-1, 0, 1)
    - Meta model predicts confidence in acting on that direction

    Parameters
    ----------
    primary_signal_column : str
        Column name containing primary labels (-1, 0, 1).
        These are typically from triple-barrier or directional labeling.
    return_column : str
        Column name containing forward returns (or the column to check correctness).
        If not provided, computed from close prices.
    bet_size_method : str or BetSizeMethod
        Method for computing bet sizes:
        - 'probability': Use model probability as bet size (for weighted positions)
        - 'fixed': Use fixed bet size of 1.0 (for binary positions)
    horizon : int, optional
        Horizon for computing forward returns if return_column not provided.
        Required when return_column is None.

    Examples
    --------
    >>> meta_labeler = MetaLabeler(
    ...     primary_signal_column='label_h5',
    ...     return_column='fwd_return_h5',
    ...     bet_size_method='probability'
    ... )
    >>> result = meta_labeler.compute_labels(df, horizon=5)
    >>> # result.labels contains: 1 (correct prediction), 0 (incorrect), -99 (neutral/invalid)
    """

    def __init__(
        self,
        primary_signal_column: str,
        return_column: str | None = None,
        bet_size_method: str | BetSizeMethod = BetSizeMethod.PROBABILITY,
        horizon: int | None = None
    ):
        # Validate primary signal column
        if not primary_signal_column or not isinstance(primary_signal_column, str):
            raise ValueError(
                f"primary_signal_column must be a non-empty string, "
                f"got {primary_signal_column!r}"
            )

        self._primary_signal_column = primary_signal_column
        self._return_column = return_column
        self._horizon = horizon

        # Convert bet_size_method to enum if string
        if isinstance(bet_size_method, str):
            try:
                bet_size_method = BetSizeMethod(bet_size_method)
            except ValueError:
                valid_values = [m.value for m in BetSizeMethod]
                raise ValueError(
                    f"Invalid bet_size_method: '{bet_size_method}'. "
                    f"Valid values are: {valid_values}"
                )
        self._bet_size_method = bet_size_method

        # Validate that we can compute returns
        if return_column is None and horizon is None:
            raise ValueError(
                "Either return_column or horizon must be provided. "
                "horizon is used to compute forward returns from close prices."
            )

    @property
    def labeling_type(self) -> LabelingType:
        """Return the type of this labeling strategy."""
        return LabelingType.META

    @property
    def required_columns(self) -> list[str]:
        """Return list of required DataFrame columns."""
        columns = [self._primary_signal_column]
        if self._return_column:
            columns.append(self._return_column)
        else:
            columns.append('close')  # Needed to compute forward returns
        return columns

    @property
    def primary_signal_column(self) -> str:
        """Return the primary signal column name."""
        return self._primary_signal_column

    @property
    def bet_size_method(self) -> BetSizeMethod:
        """Return the bet size method."""
        return self._bet_size_method

    def _compute_forward_returns(
        self,
        df: pd.DataFrame,
        horizon: int
    ) -> np.ndarray:
        """
        Compute forward returns from close prices.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'close' column
        horizon : int
            Number of bars to look ahead

        Returns
        -------
        np.ndarray
            Forward returns (next_close / current_close - 1)
        """
        close = df['close'].values
        n = len(close)
        returns = np.full(n, np.nan)

        for i in range(n - horizon):
            returns[i] = (close[i + horizon] / close[i]) - 1.0

        return returns

    def compute_labels(
        self,
        df: pd.DataFrame,
        horizon: int,
        **kwargs: Any
    ) -> LabelingResult:
        """
        Compute meta-labels indicating whether primary signals were correct.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with primary signal column and return data
        horizon : int
            Horizon identifier (used for output column naming)
        **kwargs : Any
            Additional parameters (ignored)

        Returns
        -------
        LabelingResult
            Container with meta-labels:
            - 1: Primary signal was correct (profitable direction)
            - 0: Primary signal was incorrect (unprofitable direction)
            - -99: Neutral in primary or invalid (excluded from training)

        Raises
        ------
        ValueError
            If DataFrame is empty or parameters are invalid
        KeyError
            If required columns are missing
        """
        # Validate inputs
        self.validate_inputs(df)

        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError(f"horizon must be a positive integer, got {horizon}")

        logger.info(f"Computing meta-labels for horizon {horizon}")
        logger.info(f"  Primary signal column: {self._primary_signal_column}")
        logger.info(f"  Bet size method: {self._bet_size_method.value}")

        n = len(df)

        # Get primary signals
        primary_signals = df[self._primary_signal_column].values.copy()

        # Get or compute forward returns
        if self._return_column and self._return_column in df.columns:
            forward_returns = df[self._return_column].values
        else:
            # Use horizon from constructor if set, otherwise use provided horizon
            ret_horizon = self._horizon if self._horizon is not None else horizon
            forward_returns = self._compute_forward_returns(df, ret_horizon)
            logger.debug(f"  Computed forward returns with horizon={ret_horizon}")

        # Initialize output arrays
        meta_labels = np.full(n, -99, dtype=np.int8)  # Default: invalid
        bet_sizes = np.zeros(n, dtype=np.float32)
        correctness_margin = np.zeros(n, dtype=np.float32)

        # Compute meta-labels
        for i in range(n):
            signal = primary_signals[i]
            ret = forward_returns[i]

            # Skip neutral signals (0) - they remain as -99
            if signal == 0 or signal == -99:
                continue

            # Skip if return is NaN
            if np.isnan(ret):
                continue

            # Check if primary signal was correct
            # signal = 1 (long): correct if return > 0
            # signal = -1 (short): correct if return < 0
            if signal == 1:
                is_correct = ret > 0
                correctness_margin[i] = ret
            elif signal == -1:
                is_correct = ret < 0
                correctness_margin[i] = -ret  # Flip sign for short
            else:
                # Invalid signal value
                continue

            # Set meta-label
            meta_labels[i] = 1 if is_correct else 0

            # Set bet size
            if self._bet_size_method == BetSizeMethod.FIXED:
                bet_sizes[i] = 1.0 if is_correct else 0.0
            else:
                # For probability method, bet_size equals correctness margin magnitude
                # This will be scaled by model probability at prediction time
                bet_sizes[i] = abs(correctness_margin[i])

        # Mark last horizon bars as invalid (no forward return data)
        ret_horizon = self._horizon if self._horizon is not None else horizon
        for i in range(max(0, n - ret_horizon), n):
            meta_labels[i] = -99
            bet_sizes[i] = 0.0
            correctness_margin[i] = 0.0

        # Build result
        result = LabelingResult(
            labels=meta_labels,
            horizon=horizon,
            metadata={
                'bet_size': bet_sizes,
                'correctness_margin': correctness_margin,
                'primary_signal': primary_signals.astype(np.int8)
            }
        )

        # Compute quality metrics
        result.quality_metrics = self.get_quality_metrics(result)

        # Log statistics
        self._log_label_statistics(result, horizon)

        return result

    def _log_label_statistics(self, result: LabelingResult, horizon: int) -> None:
        """Log meta-label distribution statistics."""
        labels = result.labels
        valid_mask = labels != -99
        valid_labels = labels[valid_mask]

        if len(valid_labels) == 0:
            logger.warning(f"No valid meta-labels for horizon {horizon}")
            return

        total = len(valid_labels)
        correct_count = (valid_labels == 1).sum()
        incorrect_count = (valid_labels == 0).sum()

        logger.info(f"Meta-label distribution for horizon {horizon}:")
        logger.info(f"  Correct (1):   {correct_count:6d} ({correct_count/total*100:5.1f}%)")
        logger.info(f"  Incorrect (0): {incorrect_count:6d} ({incorrect_count/total*100:5.1f}%)")

        # Primary signal accuracy
        accuracy = correct_count / total * 100 if total > 0 else 0.0
        logger.info(f"  Primary signal accuracy: {accuracy:.1f}%")

        # Log invalid sample count
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            logger.info(f"  Invalid/Neutral samples: {invalid_count} (excluded from training)")

    def get_quality_metrics(self, result: LabelingResult) -> dict[str, float]:
        """
        Compute quality metrics specific to meta-labeling.

        Includes accuracy, precision, recall for primary signal correctness.

        Parameters
        ----------
        result : LabelingResult
            The labeling result to analyze

        Returns
        -------
        dict[str, float]
            Dictionary of quality metrics including:
            - total_samples: Total number of samples
            - valid_samples: Number of samples with valid meta-labels
            - invalid_samples: Number of invalid/neutral samples
            - correct_count: Number of correct primary predictions
            - incorrect_count: Number of incorrect primary predictions
            - accuracy: Primary signal accuracy (correct / total valid)
            - precision: Precision of correct predictions
            - recall: Recall of correct predictions
            - avg_correctness_margin: Average magnitude of returns when correct
        """
        labels = result.labels
        valid_mask = labels != -99
        valid_labels = labels[valid_mask]

        metrics: dict[str, float] = {
            'total_samples': float(len(labels)),
            'valid_samples': float(len(valid_labels)),
            'invalid_samples': float((~valid_mask).sum()),
        }

        if len(valid_labels) == 0:
            return metrics

        # Correct/incorrect counts
        correct_count = (valid_labels == 1).sum()
        incorrect_count = (valid_labels == 0).sum()

        metrics['correct_count'] = float(correct_count)
        metrics['incorrect_count'] = float(incorrect_count)

        # Accuracy: overall correctness rate
        total_valid = len(valid_labels)
        metrics['accuracy'] = correct_count / total_valid if total_valid > 0 else 0.0

        # For binary classification metrics, treat correct(1) as positive class
        # Precision: TP / (TP + FP) - but in meta context, all "1" labels are true positives
        # For meta-labeling, precision = recall = accuracy since labels are ground truth
        # However, we can compute these in the context of predicting "tradeable" signals

        # In meta-labeling context:
        # - True Positive (TP): Signal was 1 and return was positive, or signal was -1 and return was negative
        # - All "correct" labels are "tradeable" predictions
        # Since we're labeling correctness, precision and recall equal accuracy

        metrics['precision'] = metrics['accuracy']  # Precision = accuracy for ground truth
        metrics['recall'] = metrics['accuracy']  # Recall = accuracy for ground truth

        # F1 score
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = (
                2 * metrics['precision'] * metrics['recall']
                / (metrics['precision'] + metrics['recall'])
            )
        else:
            metrics['f1_score'] = 0.0

        # Correctness margin analysis
        correctness_margin = result.metadata.get('correctness_margin', np.array([]))
        if len(correctness_margin) > 0:
            valid_margins = correctness_margin[valid_mask]

            # Average margin when correct
            correct_mask = valid_labels == 1
            if correct_mask.sum() > 0:
                correct_margins = valid_margins[correct_mask]
                metrics['avg_correct_margin'] = float(np.mean(correct_margins))
                metrics['std_correct_margin'] = float(np.std(correct_margins))

            # Average margin when incorrect (this is the loss magnitude)
            incorrect_mask = valid_labels == 0
            if incorrect_mask.sum() > 0:
                incorrect_margins = valid_margins[incorrect_mask]
                # Note: These will be negative since direction was wrong
                metrics['avg_incorrect_margin'] = float(np.mean(incorrect_margins))

        # Bet size analysis
        bet_sizes = result.metadata.get('bet_size', np.array([]))
        if len(bet_sizes) > 0:
            valid_bet_sizes = bet_sizes[valid_mask]
            if len(valid_bet_sizes) > 0:
                metrics['avg_bet_size'] = float(np.mean(valid_bet_sizes))
                metrics['max_bet_size'] = float(np.max(valid_bet_sizes))

        # Class balance
        if incorrect_count > 0:
            metrics['correct_incorrect_ratio'] = correct_count / incorrect_count
        else:
            metrics['correct_incorrect_ratio'] = (
                float('inf') if correct_count > 0 else 0.0
            )

        return metrics

    def add_labels_to_dataframe(
        self,
        df: pd.DataFrame,
        result: LabelingResult,
        prefix: str = 'meta_label'
    ) -> pd.DataFrame:
        """
        Add meta-labeling results to the DataFrame as new columns.

        Parameters
        ----------
        df : pd.DataFrame
            Original DataFrame
        result : LabelingResult
            Meta-labeling result to add
        prefix : str
            Column name prefix (default: 'meta_label')

        Returns
        -------
        pd.DataFrame
            DataFrame with new meta-label columns added
        """
        return super().add_labels_to_dataframe(df, result, prefix=prefix)
