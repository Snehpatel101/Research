"""
TrainerFeaturesMixin - Feature set resolution.

Contains methods for:
- Validating feature sets against model recommendations
- Resolving feature set columns from configuration
- Applying feature set filters to DataFrames
- Getting sequence model feature columns

Feature SELECTION is not done here: MLFactory's orchestrator selects each
model's features on train-only data and hands the Trainer those columns.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

# Import MODEL_DATA_REQUIREMENTS for feature_set validation (MOD-005)
# Canonical location: src.models.config.data_requirements
# NOTE: This module validates against MODEL_DATA_REQUIREMENTS. MLFactory uses
# MODEL_CONTRACTS (src/core/contracts/model_contract.py) which may define different
# max_features limits. Both registries should be kept in sync.
_MODEL_DATA_REQUIREMENTS: dict[str, Any] | None
try:
    from src.models.config import MODEL_DATA_REQUIREMENTS as _MODEL_DATA_REQUIREMENTS
except ImportError:
    _MODEL_DATA_REQUIREMENTS = None

if TYPE_CHECKING:
    from src.core.container import TimeSeriesDataContainer

logger = logging.getLogger(__name__)


class TrainerFeaturesMixin:
    """
    Mixin providing feature set resolution capabilities.

    This mixin assumes the following attributes exist on the class:
    - self.config: TrainerConfig with training settings
    - self.model: Instantiated model from registry
    """

    # Type stubs for mixin - actual values provided by composing class
    config: Any
    model: Any

    def _validate_feature_set(self) -> None:
        """
        Validate feature_set against model's recommended feature set (MOD-005).

        Logs an INFO message if the selected feature_set differs from the
        recommended set for this model type. This is advisory only - the
        trainer will still proceed with the configured feature_set.
        """
        if _MODEL_DATA_REQUIREMENTS is None:
            return  # Skip if module not available

        model_name = self.config.model_name.lower()
        if model_name not in _MODEL_DATA_REQUIREMENTS:
            logger.debug(
                f"Model '{model_name}' not in MODEL_DATA_REQUIREMENTS, "
                "skipping feature_set validation"
            )
            return

        requirements = _MODEL_DATA_REQUIREMENTS[model_name]
        recommended_feature_set = requirements.feature_set
        configured_feature_set = self.config.feature_set

        # If no feature_set is configured, it will be auto-resolved later
        if not configured_feature_set:
            return

        if configured_feature_set != recommended_feature_set:
            logger.info(
                f"Using feature_set='{configured_feature_set}' "
                f"(recommended for {model_name}: '{recommended_feature_set}')"
            )

    def _resolve_feature_set_columns(self, df: pd.DataFrame) -> list[str] | None:
        """
        Resolve feature set columns based on config.feature_set.

        Uses FEATURE_SET_ALIASES to map model names to feature sets.
        Returns None if no feature set filtering should be applied.

        Args:
            df: DataFrame with all available columns

        Returns:
            List of column names to use, or None to use all features
        """
        # Import using importlib to avoid circular imports through __init__.py chain
        import importlib

        feature_sets_config = importlib.import_module("src.data.pipeline.config.feature_sets")
        FEATURE_SET_ALIASES = feature_sets_config.FEATURE_SET_ALIASES
        FEATURE_SET_DEFINITIONS = feature_sets_config.FEATURE_SET_DEFINITIONS

        feature_sets_utils = importlib.import_module("src.data.pipeline.utils.feature_sets")
        resolve_feature_set = feature_sets_utils.resolve_feature_set

        feature_set_name = self.config.feature_set
        if feature_set_name is None:
            # Caller already chose the columns (e.g. orchestrator per-model selection)
            return None

        # If not specified, try to get from model family alias
        if not feature_set_name:
            # Check if model name has an alias
            model_name = self.config.model_name.lower()
            feature_set_name = FEATURE_SET_ALIASES.get(model_name)

            if not feature_set_name:
                # Try model family
                model_family = self.model.model_family.lower()
                feature_set_name = FEATURE_SET_ALIASES.get(model_family)

        if not feature_set_name:
            logger.debug("No feature set specified, using all features")
            return None

        # Resolve alias to canonical name
        canonical_name = FEATURE_SET_ALIASES.get(feature_set_name, feature_set_name)

        if canonical_name not in FEATURE_SET_DEFINITIONS:
            logger.warning(
                f"Unknown feature set '{feature_set_name}' (resolved to '{canonical_name}'). "
                f"Available: {list(FEATURE_SET_DEFINITIONS.keys())}. Using all features."
            )
            return None

        # Resolve feature set
        definition = FEATURE_SET_DEFINITIONS[canonical_name]
        feature_columns = resolve_feature_set(df, definition)

        # If no features matched, fall back to using all features
        # This happens with mock/test data that doesn't have realistic feature names
        if len(feature_columns) == 0:
            logger.warning(
                f"Feature set '{canonical_name}' resolved to 0 features. "
                f"Falling back to using all {len(df.columns)} columns. "
                "This may indicate mock/test data without realistic feature names."
            )
            return None

        logger.info(
            f"Feature set '{canonical_name}' resolved: {len(feature_columns)} features "
            f"(from {len(df.columns)} total columns)"
        )

        return list(feature_columns)

    def _apply_feature_set_filter(
        self,
        X_df: pd.DataFrame,
        feature_columns: list[str] | None,
    ) -> pd.DataFrame:
        """
        Apply feature set filtering to a DataFrame.

        Args:
            X_df: DataFrame with features
            feature_columns: List of columns to keep, or None to keep all

        Returns:
            Filtered DataFrame

        Raises:
            ValueError: If no features remain after filtering (MOD-004)
        """
        if feature_columns is None:
            return X_df

        # Filter to only columns that exist in both
        available_cols = [c for c in feature_columns if c in X_df.columns]
        missing_cols = set(feature_columns) - set(available_cols)

        if missing_cols:
            logger.warning(
                f"Feature set requested {len(feature_columns)} features, "
                f"but {len(missing_cols)} are missing from data: {list(missing_cols)[:5]}..."
            )

        # MOD-004: Validate that we have features after filtering
        if len(available_cols) == 0:
            raise ValueError(
                f"MOD-004: No features remain after filtering. "
                f"Requested {len(feature_columns)} features but none exist in data. "
                f"Available columns: {list(X_df.columns)[:10]}{'...' if len(X_df.columns) > 10 else ''}"
            )

        X_filtered = X_df[available_cols]

        # MOD-004: Post-filter shape validation
        expected_n_features = len(available_cols)
        actual_n_features = X_filtered.shape[1]
        if actual_n_features != expected_n_features:
            raise ValueError(
                f"MOD-004: Shape mismatch after feature filtering - "
                f"expected {expected_n_features} features but got {actual_n_features}. "
                f"This indicates a bug in the filtering logic."
            )

        return X_filtered

    def _get_sequence_model_feature_columns(
        self,
        model_name: str,
        container: TimeSeriesDataContainer,
    ) -> list[str] | None:
        """
        Get feature columns for a specific sequence model from the feature set manifest.

        Uses FEATURE_SET_ALIASES to map model names to their optimal feature sets:
        - tcn -> tcn_optimal (50 features)
        - lstm/gru -> neural_optimal (43 features)
        - patchtst -> patchtst_optimal (23 features)
        - transformer -> transformer_raw (23 features)

        Args:
            model_name: Name of the sequence model (e.g., 'tcn', 'lstm')
            container: TimeSeriesDataContainer to get available features

        Returns:
            List of feature column names, or None to use all features
        """
        import importlib

        feature_sets_config = importlib.import_module("src.data.pipeline.config.feature_sets")
        FEATURE_SET_ALIASES = feature_sets_config.FEATURE_SET_ALIASES

        # Get the optimal feature set for this model
        model_lower = model_name.lower()
        feature_set_name = FEATURE_SET_ALIASES.get(model_lower)

        if not feature_set_name:
            logger.debug(f"No feature set alias for model '{model_name}', using all features")
            return None

        # Try to load feature list from manifest (produced by Phase 1 pipeline)
        # The manifest contains pre-computed feature lists for each feature set
        manifest_path = None

        # Try to find manifest from container's source path or common locations
        possible_paths = []

        # Check if container has source_path attribute
        if hasattr(container, "source_path") and container.source_path:
            base = Path(container.source_path)
            possible_paths.extend(
                [
                    base / "artifacts" / "feature_set_manifest.json",
                    base.parent / "artifacts" / "feature_set_manifest.json",
                    base.parent.parent / "artifacts" / "feature_set_manifest.json",
                ]
            )

        # Also check common project locations
        project_root = Path(__file__).parent.parent.parent.parent
        possible_paths.extend(
            [
                project_root / "runs" / "latest" / "artifacts" / "feature_set_manifest.json",
            ]
        )

        for path in possible_paths:
            if path.exists():
                manifest_path = path
                break

        if manifest_path and manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)

                if feature_set_name in manifest:
                    feature_info = manifest[feature_set_name]
                    features = feature_info.get("features", [])
                    if features:
                        logger.info(
                            f"Sequence model '{model_name}' using feature set "
                            f"'{feature_set_name}': {len(features)} features"
                        )
                        return list(features)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load feature set manifest: {e}")

        # Fallback: resolve feature set from definition (slower but works without manifest)
        logger.debug(
            "Manifest not found or feature set missing, falling back to definition-based resolution"
        )

        # Get sample DataFrame to resolve features
        split_data = container.get_split("train")
        split_data.df[split_data.feature_columns[:1]]  # Just need columns

        # Use the existing resolve method with a DataFrame that has all feature columns
        feature_cols_df = pd.DataFrame(columns=split_data.feature_columns)

        # Temporarily set config to use this feature set
        original_feature_set = self.config.feature_set
        self.config.feature_set = feature_set_name
        try:
            result = self._resolve_feature_set_columns(feature_cols_df)
        finally:
            self.config.feature_set = original_feature_set

        if result:
            logger.info(
                f"Sequence model '{model_name}' using feature set "
                f"'{feature_set_name}': {len(result)} features (from definition)"
            )

        return result
