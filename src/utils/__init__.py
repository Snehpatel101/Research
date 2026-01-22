"""
DEPRECATED: Import from 'src.core.utils' instead.

This module will be removed in v2.0.0.

Migration:
    # Old (deprecated)
    from src.utils import CacheManager, CheckpointManager

    # New (preferred)
    from src.core.utils import CacheManager, CheckpointManager
"""

import warnings

_UTILS_EXPORTS = [
    # Colab & Environment setup
    "is_colab",
    "setup_environment",
    "setup_colab_environment",
    "get_trainer_for_colab",
    "train_ensemble_colab",
    "get_colab_dataloader_kwargs",
    "ensure_data_in_workspace",
    # Checkpoint management
    "CheckpointManager",
    # Memory management
    "MemoryInfo",
    "CacheEntry",
    "CacheStats",
    "CacheConfig",
    "CacheManager",
    "estimate_array_size",
    "estimate_object_size",
    "get_memory_info",
    "check_available_memory",
    "check_memory_sufficient",
    "log_memory_usage",
    "memory_logged",
    "get_global_cache",
    "PSUTIL_AVAILABLE",
    # Data caching
    "CacheMetadata",
    "DataCacheConfig",
    "DataCache",
    "cached_result",
    "get_global_data_cache",
    # Notebook utilities
    "setup_notebook",
    "install_dependencies",
    "mount_drive",
    "download_sample_data",
    "display_metrics",
    "plot_confusion_matrix",
    "plot_training_history",
    "plot_model_comparison",
    "get_sample_config",
    "create_progress_callback",
    # Configuration validation
    "ValidationResult",
    "validate_pipeline_config",
    "validate_trainer_config",
    "validate_cv_config",
    "validate_ensemble_config",
    "run_all_validations",
    "generate_validation_report",
    "quick_validate",
]


def __getattr__(name: str):
    """Lazy import to avoid circular imports."""
    if name in _UTILS_EXPORTS:
        warnings.warn(
            f"Importing '{name}' from 'src.utils' is deprecated. "
            f"Use 'from src.core.utils import {name}' instead. "
            "This will be removed in v2.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        from src.core import utils as utils_module

        return getattr(utils_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return _UTILS_EXPORTS
