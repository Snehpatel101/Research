"""
Utility modules for the ML Model Factory.

This package provides helper functions for notebooks, visualization,
and common operations.
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

__all__ = [
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
]
