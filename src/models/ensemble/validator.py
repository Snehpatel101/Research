"""
Ensemble Configuration Validator - Checks compatibility of base models.

Validates ensemble configurations based on ensemble type:
- Voting/Blending: Require same-family models (all tabular or all sequence)
  because they need same input shapes during inference
- Stacking: Allow heterogeneous models (mixed tabular + sequence) because
  the meta-learner always receives 2D OOF predictions regardless of base model type

The key insight for stacking: Meta-learners always receive (n_samples, n_base_models * n_classes)
shaped OOF predictions, regardless of whether base models were tabular (2D input) or
sequence (3D input). This means heterogeneous stacking is mathematically valid.
"""

from __future__ import annotations

import logging
from typing import Literal

from ..registry import ModelRegistry

logger = logging.getLogger(__name__)

# Ensemble types that support heterogeneous base models
HETEROGENEOUS_ENSEMBLE_TYPES: set[str] = {"stacking"}

# Ensemble types that require homogeneous base models (same input shape)
HOMOGENEOUS_ENSEMBLE_TYPES: set[str] = {"voting", "blending"}


class EnsembleCompatibilityError(ValueError):
    """Raised when incompatible models are combined in an ensemble."""

    pass


def validate_ensemble_config(
    base_model_names: list[str],
    ensemble_type: Literal["voting", "blending", "stacking"] | None = None,
) -> tuple[bool, str]:
    """
    Validate that base models are compatible for ensemble creation.

    Validation rules depend on ensemble type:
    - Voting/Blending: All base models must have same input shape (2D or 3D)
      because these ensembles pass the same input X to all base models
    - Stacking: Allows mixed tabular + sequence models (heterogeneous)
      because the meta-learner receives OOF predictions (always 2D)

    Args:
        base_model_names: List of model names to validate
        ensemble_type: Type of ensemble ("voting", "blending", "stacking").
            If None, applies strict validation (no mixed models).

    Returns:
        Tuple of (is_valid, error_message)
        - If valid: (True, "")
        - If invalid: (False, detailed error message with suggestions)

    Example:
        >>> # Homogeneous - always valid
        >>> is_valid, error = validate_ensemble_config(["xgboost", "lightgbm"])
        >>> assert is_valid

        >>> # Heterogeneous - invalid for voting
        >>> is_valid, error = validate_ensemble_config(
        ...     ["xgboost", "lstm"], ensemble_type="voting"
        ... )
        >>> assert not is_valid

        >>> # Heterogeneous - VALID for stacking (meta-learner gets 2D OOF)
        >>> is_valid, error = validate_ensemble_config(
        ...     ["xgboost", "lstm"], ensemble_type="stacking"
        ... )
        >>> assert is_valid  # Heterogeneous stacking is allowed!
    """
    if not base_model_names:
        return False, "No base models specified"

    if len(base_model_names) < 2:
        return False, f"Need at least 2 base models for ensemble, got {len(base_model_names)}"

    # Check all models are registered
    for model_name in base_model_names:
        if not ModelRegistry.is_registered(model_name):
            available = ModelRegistry.list_all()
            # Provide helpful message for commonly expected but unavailable models
            hint = ""
            if model_name.lower() in ("catboost", "cat"):
                hint = " (CatBoost is optional - install with: pip install catboost)"
            return False, (
                f"Model '{model_name}' is not registered{hint}. " f"Available models: {available}"
            )

    # Get model info for all base models
    model_infos = []
    for model_name in base_model_names:
        try:
            info = ModelRegistry.get_model_info(model_name)
            model_infos.append((model_name, info))
        except Exception as e:
            return False, f"Failed to get info for model '{model_name}': {e}"

    # Check sequence requirements compatibility
    requires_sequences = [info["requires_sequences"] for _, info in model_infos]

    # Check if this is a heterogeneous configuration (mixed tabular + sequence)
    is_heterogeneous = not all(requires_sequences) and any(requires_sequences)

    if is_heterogeneous:
        tabular_models = [name for name, info in model_infos if not info["requires_sequences"]]
        sequence_models = [name for name, info in model_infos if info["requires_sequences"]]

        # Stacking allows heterogeneous models
        if ensemble_type in HETEROGENEOUS_ENSEMBLE_TYPES:
            logger.info(
                f"Heterogeneous stacking ensemble validated: "
                f"tabular={tabular_models}, sequence={sequence_models}. "
                f"Meta-learner will receive 2D OOF predictions from all models."
            )
            return True, ""

        # Voting/Blending do not allow heterogeneous models
        error_msg = _build_compatibility_error_message(
            tabular_models, sequence_models, ensemble_type
        )
        return False, error_msg

    # All compatible (homogeneous)
    return True, ""


def _build_compatibility_error_message(
    tabular_models: list[str],
    sequence_models: list[str],
    ensemble_type: str | None = None,
) -> str:
    """
    Build a detailed error message for incompatible ensemble configurations.

    Args:
        tabular_models: List of tabular model names (2D input)
        sequence_models: List of sequence model names (3D input)
        ensemble_type: The ensemble type being validated (for context)

    Returns:
        Detailed error message with suggestions
    """
    ensemble_name = ensemble_type or "voting/blending"

    msg = [
        f"Ensemble Compatibility Error: Cannot mix tabular and sequence models "
        f"in {ensemble_name} ensemble.",
        "",
        "REASON:",
        "  - Tabular models expect 2D input: (n_samples, n_features)",
        "  - Sequence models expect 3D input: (n_samples, seq_len, n_features)",
        f"  - {ensemble_name.title()} ensembles pass the same input X to all base models,",
        "    causing shape mismatches during training/prediction",
        "",
        "YOUR CONFIGURATION:",
        f"  Tabular models (2D): {tabular_models}",
        f"  Sequence models (3D): {sequence_models}",
        "",
        "SUPPORTED ENSEMBLE CONFIGURATIONS:",
        "",
        "Option 1: Use STACKING for heterogeneous ensembles (RECOMMENDED)",
        "  - Stacking ensembles support mixed tabular + sequence models",
        "  - The meta-learner receives 2D OOF predictions from all models",
        "  - Example:",
        "    ModelRegistry.create('stacking', config={",
        f"        'base_model_names': {tabular_models + sequence_models},",
        "        'meta_learner_name': 'logistic',",
        "    })",
        "",
        "Option 2: All Tabular Models:",
        "  - Boosting: xgboost, lightgbm, catboost",
        "  - Classical: random_forest, logistic, svm",
        "  - Example: base_model_names=['xgboost', 'lightgbm', 'random_forest']",
        "",
        "Option 3: All Sequence Models:",
        "  - Neural: lstm, gru, tcn, transformer",
        "  - Example: base_model_names=['lstm', 'gru', 'tcn']",
        "",
        "RECOMMENDATIONS:",
    ]

    # Provide specific recommendations based on the models
    if len(tabular_models) >= 2:
        msg.append(f"  - For {ensemble_name}: Use only tabular models: {tabular_models}")
    if len(sequence_models) >= 2:
        msg.append(f"  - For {ensemble_name}: Use only sequence models: {sequence_models}")
    msg.append(f"  - For mixed models: Use stacking ensemble instead")

    msg.extend(["", "For more information, see docs/phases/PHASE_4.md"])

    return "\n".join(msg)


def validate_base_model_compatibility(
    base_model_names: list[str],
    ensemble_type: Literal["voting", "blending", "stacking"] | None = None,
) -> None:
    """
    Validate base model compatibility and raise exception if incompatible.

    Args:
        base_model_names: List of model names to validate
        ensemble_type: Type of ensemble ("voting", "blending", "stacking").
            If None, applies strict validation (no mixed models).
            Stacking allows heterogeneous (mixed) base models.

    Raises:
        EnsembleCompatibilityError: If models are incompatible

    Example:
        >>> validate_base_model_compatibility(["xgboost", "lightgbm"])  # OK
        >>> validate_base_model_compatibility(["xgboost", "lstm"])  # Raises error (strict)
        >>> validate_base_model_compatibility(
        ...     ["xgboost", "lstm"], ensemble_type="stacking"
        ... )  # OK - stacking allows mixed
    """
    is_valid, error_msg = validate_ensemble_config(base_model_names, ensemble_type)
    if not is_valid:
        raise EnsembleCompatibilityError(error_msg)


def get_compatible_models(reference_model: str) -> list[str]:
    """
    Get list of models compatible with a reference model for ensembles.

    Returns all registered models that have the same sequence requirements
    as the reference model.

    Args:
        reference_model: Model name to use as reference

    Returns:
        List of compatible model names

    Raises:
        ValueError: If reference model is not registered

    Example:
        >>> compatible = get_compatible_models("xgboost")
        >>> print(compatible)
        ['xgboost', 'lightgbm', 'catboost', 'random_forest', 'logistic', 'svm']

        >>> compatible = get_compatible_models("lstm")
        >>> print(compatible)
        ['lstm', 'gru', 'tcn', 'transformer']
    """
    if not ModelRegistry.is_registered(reference_model):
        available = ModelRegistry.list_all()
        raise ValueError(
            f"Reference model '{reference_model}' is not registered. " f"Available: {available}"
        )

    ref_info = ModelRegistry.get_model_info(reference_model)
    ref_requires_sequences = ref_info["requires_sequences"]

    # Get all models with same sequence requirement
    compatible = []
    for model_name in ModelRegistry.list_all():
        info = ModelRegistry.get_model_info(model_name)
        if info["requires_sequences"] == ref_requires_sequences:
            compatible.append(model_name)

    return sorted(compatible)


def is_heterogeneous_ensemble(base_model_names: list[str]) -> bool:
    """
    Check if a base model configuration is heterogeneous (mixed tabular + sequence).

    Args:
        base_model_names: List of model names to check

    Returns:
        True if the configuration contains both tabular and sequence models

    Example:
        >>> is_heterogeneous_ensemble(["xgboost", "lightgbm"])
        False  # All tabular
        >>> is_heterogeneous_ensemble(["lstm", "gru"])
        False  # All sequence
        >>> is_heterogeneous_ensemble(["xgboost", "lstm"])
        True  # Mixed
    """
    if len(base_model_names) < 2:
        return False

    requires_sequences = []
    for model_name in base_model_names:
        if ModelRegistry.is_registered(model_name):
            info = ModelRegistry.get_model_info(model_name)
            requires_sequences.append(info["requires_sequences"])

    if not requires_sequences:
        return False

    return not all(requires_sequences) and any(requires_sequences)


def classify_base_models(
    base_model_names: list[str],
) -> tuple[list[str], list[str]]:
    """
    Classify base models into tabular and sequence categories.

    Args:
        base_model_names: List of model names to classify

    Returns:
        Tuple of (tabular_models, sequence_models)

    Example:
        >>> tabular, sequence = classify_base_models(["xgboost", "lstm", "catboost", "tcn"])
        >>> print(tabular)
        ['xgboost', 'catboost']
        >>> print(sequence)
        ['lstm', 'tcn']
    """
    tabular_models = []
    sequence_models = []

    for model_name in base_model_names:
        if ModelRegistry.is_registered(model_name):
            info = ModelRegistry.get_model_info(model_name)
            if info["requires_sequences"]:
                sequence_models.append(model_name)
            else:
                tabular_models.append(model_name)

    return tabular_models, sequence_models


__all__ = [
    "validate_ensemble_config",
    "validate_base_model_compatibility",
    "get_compatible_models",
    "is_heterogeneous_ensemble",
    "classify_base_models",
    "EnsembleCompatibilityError",
    "HETEROGENEOUS_ENSEMBLE_TYPES",
    "HOMOGENEOUS_ENSEMBLE_TYPES",
]
