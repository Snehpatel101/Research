"""
TrainerFeaturesMixin - Feature selection and feature set resolution.

Contains methods for:
- Validating feature sets against model recommendations
- Setting up feature selection based on model family
- Resolving feature set columns from configuration
- Applying feature set filters to DataFrames
- Getting sequence model feature columns
- Running walk-forward feature selection
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

# Import MODEL_DATA_REQUIREMENTS for feature_set validation (MOD-005)
try:
    from src.phase1.config.model_config import MODEL_DATA_REQUIREMENTS
except ImportError:
    MODEL_DATA_REQUIREMENTS = None

if TYPE_CHECKING:
    from src.phase1.stages.datasets.container import TimeSeriesDataContainer

logger = logging.getLogger(__name__)


class TrainerFeaturesMixin:
    """
    Mixin providing feature selection and feature set resolution capabilities.

    This mixin assumes the following attributes exist on the class:
    - self.config: TrainerConfig with training settings
    - self.model: Instantiated model from registry
    - self.feature_selector: FeatureSelectionManager instance
    """

    def _validate_feature_set(self) -> None:
        """
        Validate feature_set against model's recommended feature set (MOD-005).

        Logs an INFO message if the selected feature_set differs from the
        recommended set for this model type. This is advisory only - the
        trainer will still proceed with the configured feature_set.
        """
        if MODEL_DATA_REQUIREMENTS is None:
            return  # Skip if module not available

        model_name = self.config.model_name.lower()
        if model_name not in MODEL_DATA_REQUIREMENTS:
            logger.debug(
                f"Model '{model_name}' not in MODEL_DATA_REQUIREMENTS, "
                "skipping feature_set validation"
            )
            return

        requirements = MODEL_DATA_REQUIREMENTS[model_name]
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

    def _setup_feature_selection(self) -> None:
        """Initialize feature selection manager based on model family and config."""
        from ..feature_selection import FeatureSelectionConfig, FeatureSelectionManager

        if not self.config.use_feature_selection:
            self.feature_selector = FeatureSelectionManager.disabled()
            return

        # Sequence models don't use external feature selection (MOD-006: improved logging)
        # WHY: Sequence models (LSTM, GRU, TCN, Transformers) learn their own temporal
        # feature representations through recurrent/convolutional/attention layers.
        # External MDA-based feature selection would:
        # 1. Lose temporal dependencies (MDA uses permutation which breaks sequences)
        # 2. Be redundant (attention/gating mechanisms already perform selection)
        # 3. Conflict with the model's inductive bias for sequential patterns
        if self.model.requires_sequences:
            logger.info(
                f"Feature selection disabled for {self.config.model_name}: "
                "sequence models learn feature representations internally via "
                "recurrent/convolutional/attention layers (MDA permutation would break "
                "temporal dependencies)"
            )
            self.feature_selector = FeatureSelectionManager.disabled()
            return

        # Create feature selection config based on model family
        fs_config = FeatureSelectionConfig.from_model_family(
            model_family=self.model.model_family,
            override={
                "n_features": self.config.feature_selection_n_features,
                "method": self.config.feature_selection_method,
                "random_state": self.config.random_seed,
            },
        )

        # Override n_features if explicitly set to 0 (use family default)
        if self.config.feature_selection_n_features == 0:
            from ..feature_selection.config import ModelFamilyDefaults

            defaults = ModelFamilyDefaults.get_defaults(self.model.model_family)
            fs_config.n_features = defaults.get("n_features", 50)

        self.feature_selector = FeatureSelectionManager(config=fs_config)

    def _is_feature_selection_enabled(self) -> bool:
        """Check if feature selection is enabled for this trainer."""
        return self.feature_selector is not None and self.feature_selector.is_enabled

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

        feature_sets_config = importlib.import_module("src.phase1.config.feature_sets")
        FEATURE_SET_ALIASES = feature_sets_config.FEATURE_SET_ALIASES
        FEATURE_SET_DEFINITIONS = feature_sets_config.FEATURE_SET_DEFINITIONS

        feature_sets_utils = importlib.import_module("src.phase1.utils.feature_sets")
        resolve_feature_set = feature_sets_utils.resolve_feature_set

        feature_set_name = self.config.feature_set

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

        return feature_columns

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

        return X_df[available_cols]

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

        feature_sets_config = importlib.import_module("src.phase1.config.feature_sets")
        FEATURE_SET_ALIASES = feature_sets_config.FEATURE_SET_ALIASES

        # Get the optimal feature set for this model
        model_lower = model_name.lower()
        feature_set_name = FEATURE_SET_ALIASES.get(model_lower)

        if not feature_set_name:
            logger.debug(
                f"No feature set alias for model '{model_name}', using all features"
            )
            return None

        # Try to load feature list from manifest (produced by Phase 1 pipeline)
        # The manifest contains pre-computed feature lists for each feature set
        manifest_path = None

        # Try to find manifest from container's source path or common locations
        possible_paths = []

        # Check if container has source_path attribute
        if hasattr(container, "source_path") and container.source_path:
            base = Path(container.source_path)
            possible_paths.extend([
                base / "artifacts" / "feature_set_manifest.json",
                base.parent / "artifacts" / "feature_set_manifest.json",
                base.parent.parent / "artifacts" / "feature_set_manifest.json",
            ])

        # Also check common project locations
        project_root = Path(__file__).parent.parent.parent.parent
        possible_paths.extend([
            project_root / "runs" / "latest" / "artifacts" / "feature_set_manifest.json",
        ])

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
                        return features
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load feature set manifest: {e}")

        # Fallback: resolve feature set from definition (slower but works without manifest)
        logger.debug(
            f"Manifest not found or feature set missing, falling back to definition-based resolution"
        )

        # Get sample DataFrame to resolve features
        split_data = container.get_split("train")
        sample_df = split_data.df[split_data.feature_columns[:1]]  # Just need columns

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

    def _run_feature_selection(
        self,
        X_train_df: pd.DataFrame,
        y_train: pd.Series,
        w_train: pd.Series | None,
        label_end_times: pd.Series | None,
    ) -> Any:
        """
        Run feature selection on training data.

        Uses walk-forward selection to prevent lookahead bias.

        Args:
            X_train_df: Training features DataFrame
            y_train: Training labels
            w_train: Sample weights (optional)
            label_end_times: Label end times for purging (optional)

        Returns:
            FeatureSelectionResult with selected features
        """
        logger.info(
            f"Running feature selection: method={self.config.feature_selection_method}, "
            f"n_features={self.config.feature_selection_n_features}"
        )

        # Run walk-forward feature selection
        if self.feature_selector is None:
            raise RuntimeError("Feature selector not initialized")
        result = self.feature_selector.select_features(
            X=X_train_df,
            y=y_train,
            sample_weights=w_train,
            n_splits=self.config.feature_selection_cv_splits,
            purge_bars=self.config.horizon * 3,  # Purge based on horizon
            embargo_bars=1440,  # ~5 days at 5-min resolution
            label_end_times=label_end_times,
        )

        return result
