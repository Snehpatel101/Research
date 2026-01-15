"""
DEPRECATED: This module has moved to src.models.config.data_requirements.

All imports should use:
    from src.models.config import MODEL_DATA_REQUIREMENTS, ModelFamily
    from src.models.config.data_requirements import get_model_requirements

This shim provides backward compatibility with deprecation warnings.
"""

import warnings


def __getattr__(name):
    """Lazy import with deprecation warning for backward compatibility."""
    warnings.warn(
        f"Importing {name} from src.phase1.config.model_config is deprecated. "
        "Use src.models.config instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from src.models.config import data_requirements

    return getattr(data_requirements, name)


# Define __all__ to match the original module for introspection
__all__ = [
    # Enums
    "ModelFamily",
    "ScalerType",
    # Dataclasses
    "ModelDataRequirements",
    "EnsembleConfig",
    # Data
    "MODEL_DATA_REQUIREMENTS",
    "ENSEMBLE_CONFIGS",
    # Functions
    "get_model_requirements",
    "get_ensemble_config",
    "get_models_by_family",
    "get_combined_requirements",
    "validate_model_config",
    "get_all_model_names",
    "get_all_ensemble_names",
]
