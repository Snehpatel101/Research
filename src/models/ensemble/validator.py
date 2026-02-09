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

from src.core.exceptions import EnsembleCompatibilityError

from ..registry import ModelRegistry

logger = logging.getLogger(__name__)

# Ensemble types that support heterogeneous base models
HETEROGENEOUS_ENSEMBLE_TYPES: set[str] = {"stacking"}

# Ensemble types that require homogeneous base models (same input shape)
HOMOGENEOUS_ENSEMBLE_TYPES: set[str] = {"voting", "blending"}


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

    def _input_rank(info: dict[str, object]) -> int:
        """
        Compute required input rank from ModelRegistry info.

        Rank mapping:
        - 2: Tabular (n_samples, n_features)
        - 3: Sequence (n_samples, seq_len, n_features)
        - 4: Multi-timeframe sequence (n_samples, n_timeframes, seq_len, n_features)
        """
        if bool(info.get("requires_4d", False)):
            return 4
        if bool(info.get("requires_sequences", False)):
            return 3
        return 2

    # Group models by required input rank (2D/3D/4D)
    rank_to_models: dict[int, list[str]] = {2: [], 3: [], 4: []}
    ranks: list[int] = []
    for name, info in model_infos:
        rank = _input_rank(info)
        ranks.append(rank)
        rank_to_models[rank].append(name)

    unique_ranks = set(ranks)

    # Homogeneous (all same rank) is always valid
    if len(unique_ranks) == 1:
        return True, ""

    # Heterogeneous: multiple ranks present
    if ensemble_type in HETEROGENEOUS_ENSEMBLE_TYPES:
        # Stacking supports mixed 2D + 3D only (meta-learner receives 2D OOF)
        if unique_ranks == {2, 3}:
            logger.info(
                "Heterogeneous stacking ensemble validated: "
                "tabular=%s, sequence_3d=%s. Meta-learner will receive 2D OOF predictions.",
                rank_to_models[2],
                rank_to_models[3],
            )
            return True, ""

        # Any involvement of 4D requires explicit 4D support in the stacking implementation
        return False, _build_rank_compatibility_error_message(rank_to_models, ensemble_type)

    # Voting/Blending (and strict validation) require identical input rank
    return False, _build_rank_compatibility_error_message(rank_to_models, ensemble_type)


def _build_rank_compatibility_error_message(
    rank_to_models: dict[int, list[str]],
    ensemble_type: str | None = None,
) -> str:
    """
    Build a detailed error message for incompatible ensemble configurations.

    Args:
        rank_to_models: Dict mapping input rank -> list of model names
        ensemble_type: The ensemble type being validated (for context)

    Returns:
        Detailed error message with suggestions
    """
    ensemble_name = ensemble_type or "voting/blending/strict"

    tabular_models = rank_to_models.get(2, [])
    sequence_3d_models = rank_to_models.get(3, [])
    sequence_4d_models = rank_to_models.get(4, [])

    msg = [
        f"Ensemble Compatibility Error: Cannot mix models with different input ranks "
        f"in {ensemble_name} ensemble.",
        "",
        "REASON:",
        "  - Tabular models expect 2D input: (n_samples, n_features)",
        "  - Sequence models expect 3D input: (n_samples, seq_len, n_features)",
        "  - Multi-timeframe sequence models expect 4D input: (n_samples, n_timeframes, seq_len, n_features)",
        f"  - {ensemble_name.title()} ensembles pass the same input X to all base models,",
        "    causing shape mismatches during training/prediction when ranks differ",
        "",
        "YOUR CONFIGURATION:",
        f"  Tabular models (2D): {tabular_models}",
        f"  Sequence models (3D): {sequence_3d_models}",
        f"  Multi-timeframe models (4D): {sequence_4d_models}",
        "",
        "SUPPORTED ENSEMBLE CONFIGURATIONS:",
        "",
        "Option 1: Homogeneous ranks (RECOMMENDED):",
        "  - All base models are 2D, OR all are 3D, OR all are 4D.",
        "",
        "Option 2: Heterogeneous stacking (2D + 3D only):",
        "  - Stacking supports mixed tabular + 3D sequence models",
        "  - The meta-learner receives 2D OOF predictions from all models",
        "  - NOTE: 4D base models are not supported in heterogeneous stacking (yet).",
        "",
        "Option 3: All Tabular Models (2D):",
        "  - Boosting: xgboost, lightgbm, catboost",
        "  - Classical: random_forest, logistic, svm",
        "  - Example: base_model_names=['xgboost', 'lightgbm', 'random_forest']",
        "",
        "Option 4: All Sequence Models (3D):",
        "  - lstm, gru, tcn, transformer",
        "  - Example: base_model_names=['lstm', 'gru', 'tcn']",
        "",
        "Option 5: All Multi-Timeframe Models (4D):",
        "  - patchtst, itransformer",
        "  - Example: base_model_names=['patchtst', 'itransformer']",
        "",
        "RECOMMENDATIONS:",
    ]

    # Provide specific recommendations based on the models
    if len(tabular_models) >= 2:
        msg.append(f"  - For {ensemble_name}: Use only tabular models: {tabular_models}")
    if len(sequence_3d_models) >= 2:
        msg.append(f"  - For {ensemble_name}: Use only 3D sequence models: {sequence_3d_models}")
    if len(sequence_4d_models) >= 2:
        msg.append(f"  - For {ensemble_name}: Use only 4D models: {sequence_4d_models}")
    if tabular_models and sequence_3d_models and not sequence_4d_models:
        msg.append("  - For mixed 2D+3D: Use stacking ensemble")
    if sequence_4d_models and (tabular_models or sequence_3d_models):
        msg.append("  - Do not mix 4D models with 2D/3D models in an ensemble")

    msg.extend(["", "For more information, see docs/implementation/PHASE_5_ADAPTERS.md"])

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

    def _input_rank(info: dict[str, object]) -> int:
        if bool(info.get("requires_4d", False)):
            return 4
        if bool(info.get("requires_sequences", False)):
            return 3
        return 2

    ref_rank = _input_rank(ref_info)

    # Get all models with same sequence requirement
    compatible = []
    for model_name in ModelRegistry.list_all():
        info = ModelRegistry.get_model_info(model_name)
        if _input_rank(info) == ref_rank:
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
