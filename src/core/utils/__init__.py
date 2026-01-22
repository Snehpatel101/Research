"""
Core Utils - Re-export from src.utils and src.coordination.

This subpackage consolidates utility functions from:
- src.utils (memory, cache, notebook, colab setup)
- src.coordination (temporal alignment, MTF coordination)

New import path:
    from src.core.utils import CacheManager, TimeframeCoordinator

Legacy import paths (still work):
    from src.utils import CacheManager
    from src.coordination import TimeframeCoordinator
"""

# Re-export from utils
from src.utils import (
    # Cache
    CacheMetadata,
    DataCache,
    DataCacheConfig,
    cached_result,
    get_global_data_cache,
    # Checkpoint
    CheckpointManager,
    # Colab setup
    ensure_data_in_workspace,
    get_colab_dataloader_kwargs,
    get_trainer_for_colab,
    is_colab,
    setup_colab_environment,
    setup_environment,
    train_ensemble_colab,
    # Config validation
    ValidationResult,
    generate_validation_report,
    quick_validate,
    run_all_validations,
    validate_cv_config,
    validate_ensemble_config,
    validate_pipeline_config,
    validate_trainer_config,
    # Memory
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
    # Notebook
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

# Re-export from coordination
from src.coordination import (
    align_to_anchor,
    apply_mtf_lag,
    compute_sequence_offset,
    validate_timestamp_alignment,
    TimeframeCoordinator,
    TimeframeData,
)

__all__ = [
    # Cache
    "CacheMetadata",
    "DataCache",
    "DataCacheConfig",
    "cached_result",
    "get_global_data_cache",
    # Checkpoint
    "CheckpointManager",
    # Colab setup
    "ensure_data_in_workspace",
    "get_colab_dataloader_kwargs",
    "get_trainer_for_colab",
    "is_colab",
    "setup_colab_environment",
    "setup_environment",
    "train_ensemble_colab",
    # Config validation
    "ValidationResult",
    "generate_validation_report",
    "quick_validate",
    "run_all_validations",
    "validate_cv_config",
    "validate_ensemble_config",
    "validate_pipeline_config",
    "validate_trainer_config",
    # Memory
    "CacheConfig",
    "CacheEntry",
    "CacheManager",
    "CacheStats",
    "MemoryInfo",
    "PSUTIL_AVAILABLE",
    "check_available_memory",
    "check_memory_sufficient",
    "estimate_array_size",
    "estimate_object_size",
    "get_global_cache",
    "get_memory_info",
    "log_memory_usage",
    "memory_logged",
    # Notebook
    "create_progress_callback",
    "display_metrics",
    "download_sample_data",
    "get_sample_config",
    "install_dependencies",
    "mount_drive",
    "plot_confusion_matrix",
    "plot_model_comparison",
    "plot_training_history",
    "setup_notebook",
    # Coordination
    "align_to_anchor",
    "apply_mtf_lag",
    "compute_sequence_offset",
    "validate_timestamp_alignment",
    "TimeframeCoordinator",
    "TimeframeData",
]
