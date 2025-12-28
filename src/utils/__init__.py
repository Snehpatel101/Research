"""
Utility modules for the ML Model Factory.

This package provides helper functions for notebooks, visualization,
configuration validation, and common operations.
"""
from src.utils.notebook import (
    setup_notebook,
    install_dependencies,
    mount_drive,
    download_sample_data,
    display_metrics,
    plot_confusion_matrix,
    plot_training_history,
    plot_model_comparison,
    get_sample_config,
    create_progress_callback,
)

from src.utils.config_validator import (
    ValidationResult,
    validate_pipeline_config,
    validate_trainer_config,
    validate_cv_config,
    validate_ensemble_config,
    run_all_validations,
    generate_validation_report,
    quick_validate,
)

__all__ = [
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
