"""
Model configuration - LAZY facade over src.models.config.

All config modules remain in their original locations; this is a facade.

Why lazy (Stage 3): `src.models.config` is a subpackage of `src.models`, so
importing it executes `src/models/__init__.py`, which eagerly imports every
model family for registration side effects -- pulling in torch, xgboost,
lightgbm and catboost. That made `import src.config` cost ~3.7s and require a
full GPU-capable ML stack just to read configuration.

PEP 562 module-level __getattr__ defers that until a name is actually used.
Measured: `import src.config` 3.74s -> see docs/program/handoffs/03_*.md.

If you need the model registry populated, call
`src.models.registry.ModelRegistry.ensure_registered()` explicitly rather
than relying on an import side effect. Implicit registration-by-import is
exactly what made this facade eager.

Usage (unchanged for callers):
    from src.config.models import TrainerConfig, detect_environment
    from src.config.models import load_model_config, build_config
"""

from typing import Any

__all__ = [
    # Paths
    "CONFIG_ROOT",
    "CONFIG_DIR",
    "TRAINING_CONFIG_PATH",
    "CV_CONFIG_PATH",
    # Exceptions
    "ConfigError",
    "ConfigValidationError",
    # Environment
    "Environment",
    "detect_environment",
    "is_colab",
    "resolve_device",
    # TrainerConfig
    "TrainerConfig",
    # Validation
    "validate_model_config_structure",
    "validate_config",
    "validate_config_strict",
    # Loaders
    "load_yaml_config",
    "load_model_config",
    "flatten_model_config",
    "find_model_config",
    "load_training_config",
    "load_cv_config",
    "get_environment_overrides",
    # Merging
    "merge_configs",
    "build_config",
    "create_trainer_config",
    "get_applied_overrides",
    "AppliedOverrides",
    "ConfigBuildResult",
    # Serialization
    "save_config",
    "save_config_json",
    # Utils
    "list_available_models",
    "get_model_info",
]


def __getattr__(name: str) -> Any:
    """Resolve a re-exported name on first use (PEP 562).

    Anything in __all__ is fetched from src.models.config on demand; the
    heavy import happens here, not at `import src.config` time.
    """
    if name in __all__:
        import src.models.config as _mc

        value = getattr(_mc, name)
        globals()[name] = value  # cache: subsequent lookups skip __getattr__
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
