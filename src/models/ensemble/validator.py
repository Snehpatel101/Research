"""
Ensemble Configuration Validator - Checks compatibility of base models.

Prevents mixing tabular and sequence models in ensembles, which would cause
shape mismatches (2D vs 3D inputs).
"""
from __future__ import annotations

import logging

from ..registry import ModelRegistry

logger = logging.getLogger(__name__)


class EnsembleCompatibilityError(ValueError):
    """Raised when incompatible models are combined in an ensemble."""
    pass


def validate_ensemble_config(base_model_names: list[str]) -> tuple[bool, str]:
    """
    Validate that base models are compatible for ensemble creation.

    Ensembles require all base models to have the same input shape requirements:
    - Tabular models expect 2D input: (n_samples, n_features)
    - Sequence models expect 3D input: (n_samples, seq_len, n_features)

    Mixing these families will cause shape mismatches during training/prediction.

    Args:
        base_model_names: List of model names to validate

    Returns:
        Tuple of (is_valid, error_message)
        - If valid: (True, "")
        - If invalid: (False, detailed error message with suggestions)

    Example:
        >>> is_valid, error = validate_ensemble_config(["xgboost", "lightgbm"])
        >>> assert is_valid  # Both are tabular models

        >>> is_valid, error = validate_ensemble_config(["xgboost", "lstm"])
        >>> assert not is_valid  # Mixing tabular + sequence
        >>> print(error)  # Shows helpful error message
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
                f"Model '{model_name}' is not registered{hint}. "
                f"Available models: {available}"
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

    # All models must have the same sequence requirement
    if not all(requires_sequences) and any(requires_sequences):
        # Mixed tabular and sequence models
        tabular_models = [
            name for name, info in model_infos
            if not info["requires_sequences"]
        ]
        sequence_models = [
            name for name, info in model_infos
            if info["requires_sequences"]
        ]

        error_msg = _build_compatibility_error_message(
            tabular_models, sequence_models
        )
        return False, error_msg

    # All compatible
    return True, ""


def _build_compatibility_error_message(
    tabular_models: list[str],
    sequence_models: list[str],
) -> str:
    """
    Build a detailed error message for incompatible ensemble configurations.

    Args:
        tabular_models: List of tabular model names (2D input)
        sequence_models: List of sequence model names (3D input)

    Returns:
        Detailed error message with suggestions
    """
    msg = [
        "Ensemble Compatibility Error: Cannot mix tabular and sequence models.",
        "",
        "REASON:",
        "  - Tabular models expect 2D input: (n_samples, n_features)",
        "  - Sequence models expect 3D input: (n_samples, seq_len, n_features)",
        "  - Mixed ensembles would cause shape mismatches during training/prediction",
        "",
        "YOUR CONFIGURATION:",
        f"  Tabular models (2D): {tabular_models}",
        f"  Sequence models (3D): {sequence_models}",
        "",
        "SUPPORTED ENSEMBLE CONFIGURATIONS:",
        "",
        "✅ All Tabular Models:",
        "  - Boosting: xgboost, lightgbm, catboost",
        "  - Classical: random_forest, logistic, svm",
        "  - Example: base_model_names=['xgboost', 'lightgbm', 'random_forest']",
        "",
        "✅ All Sequence Models:",
        "  - Neural: lstm, gru, tcn, transformer",
        "  - Example: base_model_names=['lstm', 'gru', 'tcn']",
        "",
        "❌ Mixed Models (NOT SUPPORTED):",
        "  - Example: base_model_names=['xgboost', 'lstm']  # WILL FAIL",
        "",
        "RECOMMENDATIONS:",
    ]

    # Provide specific recommendations based on the models
    if len(tabular_models) >= 2:
        msg.append(f"  - Use only tabular models: {tabular_models}")
    if len(sequence_models) >= 2:
        msg.append(f"  - Use only sequence models: {sequence_models}")

    msg.extend([
        "",
        "For more information, see docs/phases/PHASE_4.md"
    ])

    return "\n".join(msg)


def validate_base_model_compatibility(base_model_names: list[str]) -> None:
    """
    Validate base model compatibility and raise exception if incompatible.

    Args:
        base_model_names: List of model names to validate

    Raises:
        EnsembleCompatibilityError: If models are incompatible

    Example:
        >>> validate_base_model_compatibility(["xgboost", "lightgbm"])  # OK
        >>> validate_base_model_compatibility(["xgboost", "lstm"])  # Raises error
    """
    is_valid, error_msg = validate_ensemble_config(base_model_names)
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
            f"Reference model '{reference_model}' is not registered. "
            f"Available: {available}"
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


__all__ = [
    "validate_ensemble_config",
    "validate_base_model_compatibility",
    "get_compatible_models",
    "EnsembleCompatibilityError",
]
