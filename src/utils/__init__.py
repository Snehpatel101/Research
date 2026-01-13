"""
Utility modules for the ML Model Factory.

This package provides helper functions for notebooks, visualization,
configuration validation, and common operations.
"""

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
