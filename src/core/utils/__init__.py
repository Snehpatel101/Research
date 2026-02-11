"""
Utility modules for the ML Model Factory.

This package provides helper functions for notebooks, visualization,
configuration validation, memory management, caching, and common operations.
"""

from .colab_setup import (
    ensure_data_in_workspace,
    get_colab_dataloader_kwargs,
    get_trainer_for_colab,
    is_colab,
    setup_colab_environment,
    setup_environment,
    train_ensemble_colab,
)
from .config_validator import (
    ValidationResult,
    generate_validation_report,
    quick_validate,
    run_all_validations,
    validate_cv_config,
    validate_ensemble_config,
    validate_pipeline_config,
    validate_trainer_config,
)

# Device utilities (Phase 8A)
from .device_utils import (
    check_cuda_available,
    get_device_string,
    get_torch_device,
)

# Math utilities (Phase 8A)
from .math_utils import (
    ema,
    normalize_series,
    safe_divide,
    sma,
)
from .memory import (
    PSUTIL_AVAILABLE,
    CacheConfig,
    CacheEntry,
    CacheManager,
    CacheStats,
    MemoryInfo,
    check_available_memory,
    check_memory_sufficient,
    estimate_array_size,
    estimate_object_size,
    get_global_cache,
    get_memory_info,
    log_memory_usage,
    memory_logged,
)
from .notebook import (
    create_progress_callback,
    display_metrics,
    download_sample_data,
    get_sample_config,
    install_dependencies,
    mount_drive,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_training_history,
    setup_notebook,
)

__all__ = [
    # Colab & Environment setup
    "is_colab",
    "setup_environment",
    "setup_colab_environment",
    "get_trainer_for_colab",
    "train_ensemble_colab",
    "get_colab_dataloader_kwargs",
    "ensure_data_in_workspace",
    # Memory management (MOD-007)
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
    # Math utilities (Phase 8A)
    "safe_divide",
    "sma",
    "ema",
    "normalize_series",
    # Device utilities (Phase 8A)
    "check_cuda_available",
    "get_device_string",
    "get_torch_device",
]
