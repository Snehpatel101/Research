"""
Multi-model pipeline configuration for Phase 1.

This module provides utilities for preparing data for multiple model types
simultaneously. When a user specifies multiple models (e.g., --models xgboost,lstm),
Phase 1 should prepare data that satisfies ALL model requirements.

Key Principles:
- UNION of features: Include all features needed by any model
- STRICTEST scaling: Apply scaling if any model requires it
- LONGEST sequences: Create sequences with max length across models
- SAVE multiple scalers: Persist scalers for each model family's preferences
"""
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
import logging

from .model_config import (
    ModelDataRequirements,
    MODEL_DATA_REQUIREMENTS,
    ENSEMBLE_CONFIGS,
    get_model_requirements,
    get_combined_requirements,
    ScalerType,
    ModelFamily,
)
from .feature_sets import (
    FeatureSetDefinition,
    FEATURE_SET_DEFINITIONS,
    resolve_feature_set_name,
)

logger = logging.getLogger(__name__)


@dataclass
class MultiModelPipelineConfig:
    """
    Configuration for preparing data for multiple models.

    This config is derived from the union of requirements for all
    specified models. Phase 1 uses this to prepare comprehensive datasets.

    Attributes:
        model_names: List of target model names
        feature_sets: Set of feature sets to include (union)
        requires_scaling: True if any model needs scaling
        scaler_types_needed: Set of scaler types to save
        requires_sequences: True if any model needs sequences
        sequence_length: Maximum sequence length needed
        max_features: Minimum of all max_features constraints (conservative)
        save_unscaled: Whether to also save unscaled data (for boosting models)
    """
    model_names: List[str]
    feature_sets: Set[str] = field(default_factory=set)
    requires_scaling: bool = False
    scaler_types_needed: Set[ScalerType] = field(default_factory=set)
    requires_sequences: bool = False
    sequence_length: int = 60
    max_features: Optional[int] = None
    save_unscaled: bool = False

    def __post_init__(self):
        # Validate model names
        for name in self.model_names:
            if name.lower() not in MODEL_DATA_REQUIREMENTS:
                logger.warning(f"Unknown model: {name}")

    def get_all_feature_prefixes(self) -> List[str]:
        """Get union of all feature prefixes from all feature sets."""
        all_prefixes = set()
        for fs_name in self.feature_sets:
            if fs_name in FEATURE_SET_DEFINITIONS:
                fs = FEATURE_SET_DEFINITIONS[fs_name]
                all_prefixes.update(fs.include_prefixes)
        return sorted(all_prefixes)

    def get_all_include_columns(self) -> List[str]:
        """Get union of all explicitly included columns."""
        all_columns = set()
        for fs_name in self.feature_sets:
            if fs_name in FEATURE_SET_DEFINITIONS:
                fs = FEATURE_SET_DEFINITIONS[fs_name]
                all_columns.update(fs.include_columns)
        return sorted(all_columns)

    def requires_mtf(self) -> bool:
        """Check if any feature set requires MTF features."""
        for fs_name in self.feature_sets:
            if fs_name in FEATURE_SET_DEFINITIONS:
                if FEATURE_SET_DEFINITIONS[fs_name].include_mtf:
                    return True
        return False


def build_multi_model_config(model_names: List[str]) -> MultiModelPipelineConfig:
    """
    Build pipeline configuration for multiple models.

    This analyzes the requirements of all specified models and creates
    a unified configuration that satisfies all of them.

    Parameters
    ----------
    model_names : List[str]
        List of model names (e.g., ['xgboost', 'lstm', 'transformer'])

    Returns
    -------
    MultiModelPipelineConfig
        Configuration for preparing data

    Example
    -------
    >>> config = build_multi_model_config(['xgboost', 'lstm'])
    >>> config.requires_scaling
    True  # LSTM needs scaling
    >>> config.save_unscaled
    True  # XGBoost prefers unscaled
    >>> config.sequence_length
    60    # LSTM default
    """
    if not model_names:
        raise ValueError("At least one model name required")

    # Normalize model names
    model_names = [m.lower().strip() for m in model_names]

    # Collect all requirements
    feature_sets: Set[str] = set()
    scaler_types: Set[ScalerType] = set()
    sequence_lengths: List[int] = []
    max_features_list: List[int] = []
    requires_scaling = False
    requires_sequences = False
    has_boosting = False

    for name in model_names:
        if name not in MODEL_DATA_REQUIREMENTS:
            logger.warning(f"Unknown model '{name}', skipping")
            continue

        req = MODEL_DATA_REQUIREMENTS[name]

        # Collect feature set
        feature_sets.add(req.feature_set)

        # Track scaling requirements
        if req.requires_scaling:
            requires_scaling = True
            scaler_types.add(req.scaler_type)

        # Track sequence requirements
        if req.requires_sequences:
            requires_sequences = True
            sequence_lengths.append(req.sequence_length)

        # Track max features
        if req.max_features:
            max_features_list.append(req.max_features)

        # Track if any boosting model (prefers unscaled)
        if req.family == ModelFamily.BOOSTING:
            has_boosting = True

    # Compute derived values
    max_seq_len = max(sequence_lengths) if sequence_lengths else 60
    min_max_features = min(max_features_list) if max_features_list else None

    # Save unscaled if any boosting model is included
    save_unscaled = has_boosting and requires_scaling

    return MultiModelPipelineConfig(
        model_names=model_names,
        feature_sets=feature_sets,
        requires_scaling=requires_scaling,
        scaler_types_needed=scaler_types,
        requires_sequences=requires_sequences,
        sequence_length=max_seq_len,
        max_features=min_max_features,
        save_unscaled=save_unscaled,
    )


def expand_ensemble_models(model_names: List[str]) -> List[str]:
    """
    Expand ensemble names to their constituent base models.

    If a user specifies 'hybrid_ensemble', this expands it to
    ['xgboost', 'lstm', 'transformer'].

    Parameters
    ----------
    model_names : List[str]
        List of model or ensemble names

    Returns
    -------
    List[str]
        Expanded list with ensembles replaced by base models
    """
    expanded = []
    for name in model_names:
        name_lower = name.lower().strip()
        if name_lower in ENSEMBLE_CONFIGS:
            # Expand ensemble to base models
            config = ENSEMBLE_CONFIGS[name_lower]
            expanded.extend(config.base_models)
            # Also include meta-learner if not already in base models
            if config.meta_learner not in config.base_models:
                expanded.append(config.meta_learner)
        else:
            expanded.append(name_lower)

    # De-duplicate while preserving order
    seen = set()
    unique = []
    for m in expanded:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    return unique


def get_recommended_feature_set(model_names: List[str]) -> str:
    """
    Get the most appropriate single feature set for a list of models.

    When only one feature set can be used, this selects the most
    comprehensive one that works for all specified models.

    Parameters
    ----------
    model_names : List[str]
        List of model names

    Returns
    -------
    str
        Recommended feature set name

    Notes
    -----
    Priority order:
    1. If any model uses ensemble_base -> ensemble_base (most features)
    2. If mix of boosting and neural -> core_full (comprehensive)
    3. If all boosting -> boosting_optimal
    4. If all neural -> neural_optimal
    5. Default -> core_full
    """
    model_names = [m.lower().strip() for m in model_names]

    families = set()
    feature_sets = set()

    for name in model_names:
        if name in MODEL_DATA_REQUIREMENTS:
            req = MODEL_DATA_REQUIREMENTS[name]
            families.add(req.family)
            feature_sets.add(req.feature_set)

    # Check for ensemble requirement
    if "ensemble_base" in feature_sets:
        return "ensemble_base"

    # Check for mixed families
    if len(families) > 1:
        return "core_full"

    # Single family - return its preferred set
    if ModelFamily.BOOSTING in families:
        return "boosting_optimal"
    if ModelFamily.NEURAL in families:
        return "neural_optimal"
    if ModelFamily.TRANSFORMER in families:
        return "transformer_raw"

    return "core_full"


def validate_multi_model_setup(model_names: List[str]) -> List[str]:
    """
    Validate a multi-model setup and return any warnings.

    Parameters
    ----------
    model_names : List[str]
        List of model names

    Returns
    -------
    List[str]
        List of warning messages (empty if setup is optimal)
    """
    warnings = []
    model_names = [m.lower().strip() for m in model_names]

    # Check for unknown models
    unknown = [m for m in model_names if m not in MODEL_DATA_REQUIREMENTS]
    if unknown:
        warnings.append(f"Unknown models will be ignored: {unknown}")

    # Check for conflicting scaler requirements
    scaler_types = set()
    for name in model_names:
        if name in MODEL_DATA_REQUIREMENTS:
            req = MODEL_DATA_REQUIREMENTS[name]
            if req.requires_scaling:
                scaler_types.add(req.scaler_type)

    if len(scaler_types) > 1:
        warnings.append(
            f"Models have different scaling preferences: {[s.value for s in scaler_types]}. "
            f"Will save multiple scaled versions."
        )

    # Check for wide sequence length variation
    seq_lengths = []
    for name in model_names:
        if name in MODEL_DATA_REQUIREMENTS:
            req = MODEL_DATA_REQUIREMENTS[name]
            if req.requires_sequences:
                seq_lengths.append(req.sequence_length)

    if seq_lengths:
        if max(seq_lengths) > 2 * min(seq_lengths):
            warnings.append(
                f"Wide range of sequence lengths: {min(seq_lengths)} to {max(seq_lengths)}. "
                f"Using max ({max(seq_lengths)}) may include unnecessary padding for shorter models."
            )

    return warnings


__all__ = [
    "MultiModelPipelineConfig",
    "build_multi_model_config",
    "expand_ensemble_models",
    "get_recommended_feature_set",
    "validate_multi_model_setup",
]
