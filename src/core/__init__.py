"""
Core package - unified utilities, paths, and defaults.

REORG-001: This package consolidates:
- src/utils/ (1,973 lines - 5 files)
- src/common/ (1,602 lines - 3 files)
- src/phase1/utils/ (~25KB - 3 files)

Import patterns:
    from src.core import DEFAULTS
    from src.core.paths import PROJECT_ROOT, DATA_DIR
    from src.core.defaults import DEFAULTS, get_default
"""

from src.core.defaults import DEFAULTS, GlobalDefaults, as_dict, get_default
from src.core.paths import (
    CONFIG_DIR,
    CONFIG_MODELS_DIR,
    CONFIG_PIPELINE_DIR,
    CONFIG_ROOT,
    CV_CONFIG_PATH,
    DATA_DIR,
    EXPERIMENTS_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    RESULTS_DIR,
    RUNS_DIR,
    TRAINING_CONFIG_PATH,
)
from src.core.reproducibility import (
    ReproducibilityConfig,
    ReproducibilityInfo,
    ensure_reproducibility,
    get_reproducibility_info,
    get_worker_init_fn,
    set_all_seeds,
)

__all__ = [
    # Paths
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "RESULTS_DIR",
    "RUNS_DIR",
    "EXPERIMENTS_DIR",
    "CONFIG_ROOT",
    "CONFIG_MODELS_DIR",
    "CONFIG_PIPELINE_DIR",
    "CONFIG_DIR",
    "TRAINING_CONFIG_PATH",
    "CV_CONFIG_PATH",
    # Defaults
    "DEFAULTS",
    "GlobalDefaults",
    "get_default",
    "as_dict",
    # Reproducibility
    "ReproducibilityConfig",
    "ReproducibilityInfo",
    "set_all_seeds",
    "get_reproducibility_info",
    "ensure_reproducibility",
    "get_worker_init_fn",
]
