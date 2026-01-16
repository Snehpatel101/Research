"""
Utility modules for the ML Model Factory.

This package provides helper functions for notebooks, visualization,
configuration validation, memory management, caching, and common operations.
"""

from src.utils.cache import (
    CacheMetadata,
    DataCache,
    DataCacheConfig,
    cached_result,
    get_global_data_cache,
)
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.colab_setup import (
    ensure_data_in_workspace,
    get_colab_dataloader_kwargs,
    get_trainer_for_colab,
    is_colab,
    setup_colab_environment,
    setup_environment,
    train_ensemble_colab,
)
from src.utils.config_validator import (
    ValidationResult,
    generate_validation_report,
    quick_validate,
    run_all_validations,
    validate_cv_config,
    validate_ensemble_config,
    validate_pipeline_config,
    validate_trainer_config,
)
from src.utils.memory import (
    CacheConfig,
    CacheEntry,
    CacheManager,
    CacheStats,
    MemoryInfo,
    PSUTIL_AVAILABLE,
    check_available_memory,
    check_memory_sufficient,
    estimate_array_size,
    estimate_object_size,
    get_global_cache,
    get_memory_info,
    log_memory_usage,
    memory_logged,
)
from src.utils.notebook import (
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
    # Checkpoint management
    "CheckpointManager",
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
    # Data caching (MOD-007)
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
