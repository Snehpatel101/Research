"""
TrainerEvaluationMixin - Test set evaluation functionality.

Contains methods for:
- Evaluating model on test set with proper warnings
- Applying feature selection to test data
- Handling heterogeneous ensemble evaluation
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..base import PredictionOutput
from ..data_preparation import prepare_test_data
from ..metrics import compute_classification_metrics, compute_trading_metrics
from ..registry import ModelRegistry

# Invalid label sentinel used by the pipeline
INVALID_LABEL_SENTINEL = -99

if TYPE_CHECKING:
    from src.core.container import TimeSeriesDataContainer

logger = logging.getLogger(__name__)


def _validate_labels(
    y: np.ndarray | pd.Series,
    context: str = "labels",
) -> None:
    """
    Validate that labels do not contain invalid sentinel values.

    LEAKAGE PREVENTION: Labels marked as -99 indicate edge cases where
    the triple-barrier outcome could not be computed (e.g., end of data).
    Training on these corrupted labels inflates metrics artificially.

    The TimeSeriesDataContainer filters these by default (exclude_invalid_labels=True),
    but this provides a defensive safety check in case filtering was bypassed.

    Args:
        y: Labels array or series
        context: Description for error message

    Raises:
        ValueError: If invalid labels are found
    """
    y_array = y.values if hasattr(y, "values") else y
    n_invalid = (y_array == INVALID_LABEL_SENTINEL).sum()

    if n_invalid > 0:
        raise ValueError(
            f"LEAKAGE DETECTED: {context} contains {n_invalid} invalid labels "
            f"(sentinel value {INVALID_LABEL_SENTINEL}). These indicate edge cases "
            "where the triple-barrier outcome could not be computed. Training on "
            "these corrupted labels inflates metrics artificially.\n\n"
            "Fix: Ensure TimeSeriesDataContainer was loaded with "
            "exclude_invalid_labels=True (the default), or filter manually:\n"
            f"  mask = y != {INVALID_LABEL_SENTINEL}\n"
            "  X, y, weights = X[mask], y[mask], weights[mask]"
        )


class TrainerEvaluationMixin:
    """
    Mixin providing test set evaluation capabilities.

    This mixin assumes the following attributes exist on the class:
    - self.config: TrainerConfig with training settings
    - self.model: Instantiated model from registry
    - self.feature_selector: FeatureSelectionManager instance
    - self._feature_set_columns: List of feature columns or None
    - self._is_feature_selection_enabled(): Method to check if feature selection is enabled
    - self._is_heterogeneous_ensemble(): Method to check for heterogeneous ensembles
    - self._apply_feature_set_filter(): Method to apply feature set filtering
    - self._get_sequence_model_feature_columns(): Method to get sequence model features
    """

    # Type hints for mixin attributes (provided by the class this is mixed into)
    config: Any  # TrainerConfig
    model: Any  # Model instance
    feature_selector: Any | None
    _feature_set_columns: list[str] | None

    def _is_feature_selection_enabled(self) -> bool:
        """Check if feature selection is enabled. Must be implemented by subclass."""
        raise NotImplementedError

    def _is_heterogeneous_ensemble(self) -> bool:
        """Check for heterogeneous ensembles. Must be implemented by subclass."""
        raise NotImplementedError

    def _apply_feature_set_filter(
        self, X_df: pd.DataFrame, feature_columns: list[str]
    ) -> pd.DataFrame:
        """Apply feature set filtering. Must be implemented by subclass."""
        raise NotImplementedError

    def _get_sequence_model_feature_columns(
        self, model_name: str, container: TimeSeriesDataContainer
    ) -> list[str] | None:
        """Get sequence model features. Must be implemented by subclass."""
        raise NotImplementedError

    def _evaluate_test_set(
        self,
        container: TimeSeriesDataContainer,
    ) -> tuple[dict[str, Any] | None, PredictionOutput | None]:
        """
        Evaluate model on test set with warnings about one-shot evaluation.

        Applies the same feature selection used during training to test data.

        Args:
            container: TimeSeriesDataContainer with test split

        Returns:
            Tuple of (test_metrics, test_predictions)
        """
        logger.warning("=" * 70)
        logger.warning("TEST SET EVALUATION - ONE-SHOT GENERALIZATION ESTIMATE")
        logger.warning("=" * 70)
        logger.warning(
            "You are evaluating on the TEST SET. This is your final, "
            "one-shot generalization estimate."
        )
        logger.warning("DO NOT iterate on these results. If you do, you're overfitting to test.")
        logger.warning(
            "Iterate on VALIDATION metrics during development, "
            "then evaluate test ONCE when ready."
        )
        logger.warning("=" * 70)

        # Load test data
        # Track sequence data for heterogeneous stacking ensembles
        X_test_seq = None

        if self._is_heterogeneous_ensemble():
            # Heterogeneous stacking: load BOTH tabular and sequence test data
            logger.info("Heterogeneous stacking: preparing both tabular and sequence test data")

            # Tabular test data
            X_test_result, y_test_series, _ = container.get_sklearn_arrays("test", return_df=True)
            # Ensure we have a DataFrame for feature operations
            X_test_df = (
                X_test_result
                if isinstance(X_test_result, pd.DataFrame)
                else pd.DataFrame(X_test_result)
            )

            # MOD-006 FIX: Validate and apply feature set filtering (must happen before feature selection)
            if self._feature_set_columns is not None:
                # Validate that required feature columns exist in test data
                missing_features = set(self._feature_set_columns) - set(X_test_df.columns)
                if missing_features:
                    available_features = set(X_test_df.columns) & set(self._feature_set_columns)
                    raise ValueError(
                        f"MOD-006: Test data missing {len(missing_features)} required feature columns. "
                        f"Missing: {sorted(missing_features)[:10]}{'...' if len(missing_features) > 10 else ''}. "
                        f"Available: {len(available_features)}/{len(self._feature_set_columns)} features. "
                        f"Ensure test data was processed with the same pipeline as training data."
                    )
                n_features_before = X_test_df.shape[1]
                X_test_df = self._apply_feature_set_filter(X_test_df, self._feature_set_columns)

                # MOD-004: Post-filter shape validation for heterogeneous test data
                len([c for c in self._feature_set_columns if c in X_test_df.columns])
                if X_test_df.shape[1] == 0:
                    raise ValueError(
                        f"MOD-004: No features remain after filtering test data for heterogeneous ensemble. "
                        f"Had {n_features_before} features before, 0 after. "
                        f"Check that feature_set_columns matches test data columns."
                    )

                logger.debug(
                    f"Applied feature set filter to test set: {X_test_df.shape[1]} features"
                )

            # Apply feature selection if enabled
            if (
                self._is_feature_selection_enabled()
                and self.feature_selector is not None
                and self.feature_selector.is_fitted
            ):
                X_test_df = self.feature_selector.apply_selection_df(X_test_df)
                logger.debug(
                    f"Applied feature selection to test set: {X_test_df.shape[1]} features"
                )

            X_test = np.asarray(X_test_df)
            y_test = np.asarray(y_test_series)

            # Get feature columns for sequence models (same as training)
            seq_feature_columns = None
            base_model_names = self.config.model_config.get("base_model_names", [])
            for model_name in base_model_names:
                model_info = ModelRegistry.get_model_info(model_name)
                if model_info and model_info.get("requires_sequences", False):
                    seq_feature_columns = self._get_sequence_model_feature_columns(
                        model_name, container
                    )
                    break

            # Sequence test data for sequence-based base models (with feature filtering)
            X_test_seq, _, _ = prepare_test_data(
                container,
                requires_sequences=True,
                sequence_length=self.config.sequence_length,
                feature_columns=seq_feature_columns,
            )

            logger.info(
                f"Heterogeneous test data prepared: "
                f"tabular={X_test.shape}, sequence={X_test_seq.shape}"
            )

        elif self.model.requires_4d:
            # 4D models: load multi-resolution 4D tensors from container
            from src.core.contracts import FeatureMode, get_model_contract

            contract = get_model_contract(self.config.model_name)
            timeframes = [contract.primary_timeframe, *list(contract.mtf_timeframes)]

            features_per_timeframe = None
            if contract.feature_mode == FeatureMode.RAW:
                features_per_timeframe = ["open", "high", "low", "close", "volume"]

            X_test, y_test, _ = prepare_test_data(
                container,
                requires_sequences=False,
                requires_4d=True,
                sequence_length=self.config.sequence_length,
                timeframes=timeframes,
                features_per_timeframe=features_per_timeframe,
                include_base_features=True,
            )
        elif self.model.requires_sequences:
            # Sequence models: load sequences directly with model-specific features
            seq_feature_columns = self._get_sequence_model_feature_columns(
                self.config.model_name, container
            )
            X_test, y_test, _ = prepare_test_data(
                container,
                requires_sequences=True,
                sequence_length=self.config.sequence_length,
                feature_columns=seq_feature_columns,
            )
        else:
            # Tabular models: load DataFrame and apply feature selection
            X_test_result, y_test_series, _ = container.get_sklearn_arrays("test", return_df=True)
            # Ensure we have a DataFrame for feature operations
            X_test_df = (
                X_test_result
                if isinstance(X_test_result, pd.DataFrame)
                else pd.DataFrame(X_test_result)
            )

            # MOD-006 FIX: Validate and apply feature set filtering (must happen before feature selection)
            if self._feature_set_columns is not None:
                # Validate that required feature columns exist in test data
                missing_features = set(self._feature_set_columns) - set(X_test_df.columns)
                if missing_features:
                    available_features = set(X_test_df.columns) & set(self._feature_set_columns)
                    raise ValueError(
                        f"MOD-006: Test data missing {len(missing_features)} required feature columns. "
                        f"Missing: {sorted(missing_features)[:10]}{'...' if len(missing_features) > 10 else ''}. "
                        f"Available: {len(available_features)}/{len(self._feature_set_columns)} features. "
                        f"Ensure test data was processed with the same pipeline as training data."
                    )
                n_features_before = X_test_df.shape[1]
                X_test_df = self._apply_feature_set_filter(X_test_df, self._feature_set_columns)

                # MOD-004: Post-filter shape validation for tabular test data
                if X_test_df.shape[1] == 0:
                    raise ValueError(
                        f"MOD-004: No features remain after filtering test data for tabular model. "
                        f"Had {n_features_before} features before, 0 after. "
                        f"Check that feature_set_columns matches test data columns."
                    )

                logger.debug(
                    f"Applied feature set filter to test set: {X_test_df.shape[1]} features"
                )

            # Apply feature selection if enabled
            if (
                self._is_feature_selection_enabled()
                and self.feature_selector is not None
                and self.feature_selector.is_fitted
            ):
                X_test_df = self.feature_selector.apply_selection_df(X_test_df)
                logger.debug(
                    f"Applied feature selection to test set: {X_test_df.shape[1]} features"
                )

            X_test = np.asarray(X_test_df)
            y_test = np.asarray(y_test_series)

        # LEAKAGE PREVENTION: Validate no invalid labels (-99) in test data
        _validate_labels(y_test, "test labels")

        logger.info(f"Test set size: {X_test.shape}")

        # Evaluate on test set
        # For heterogeneous stacking, pass both tabular and sequence data
        if self._is_heterogeneous_ensemble() and X_test_seq is not None:
            # Heterogeneous stacking models accept X_seq as keyword argument
            test_predictions = self.model.predict(X_test, X_seq=X_test_seq)
        else:
            test_predictions = self.model.predict(X_test)

        # Align y_test with predictions (may be trimmed for heterogeneous ensembles)
        y_test_aligned = y_test
        if len(test_predictions.class_predictions) < len(y_test):
            offset = len(y_test) - len(test_predictions.class_predictions)
            y_test_aligned = y_test[offset:]

        test_metrics = compute_classification_metrics(
            y_true=y_test_aligned,
            y_pred=test_predictions.class_predictions,
            y_proba=test_predictions.class_probabilities,
        )
        test_metrics["trading"] = compute_trading_metrics(
            y_true=y_test_aligned,
            y_pred=test_predictions.class_predictions,
        )

        logger.warning("=" * 70)
        logger.warning("TEST SET RESULTS (DO NOT ITERATE ON THESE)")
        logger.warning("=" * 70)
        logger.warning(
            f"Test Accuracy: {test_metrics['accuracy']:.4f}, "
            f"Test F1: {test_metrics['macro_f1']:.4f}"
        )
        logger.warning(
            "If test results are disappointing: DO NOT tune and re-evaluate. "
            "Move on to the next experiment."
        )
        logger.warning("=" * 70)

        return test_metrics, test_predictions
