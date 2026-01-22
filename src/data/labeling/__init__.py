"""
Labeling Package for PHASE_1B - Triple-Barrier Labeling and Optimization.

This package provides the triple-barrier labeling method from Lopez de Prado's
"Advances in Financial Machine Learning" along with Optuna-based optimization
for finding optimal labeling parameters.

Components:
    - TripleBarrierConfig: Configuration dataclass for labeling parameters
    - TripleBarrierLabeler: Triple-barrier label generation
    - LabelOptimizer: Optuna-based parameter optimization
    - LabelOptimizationResult: Optimization result container

Labels:
    -1 = Short (lower barrier hit first)
     0 = Neutral/Timeout (time barrier hit)
    +1 = Long (upper barrier hit first)

Usage:
    Basic Labeling:
        >>> from src.data.labeling import TripleBarrierConfig, TripleBarrierLabeler
        >>> config = TripleBarrierConfig(upper_mult=2.0, lower_mult=2.0, horizon=20)
        >>> labeler = TripleBarrierLabeler(config)
        >>> labels = labeler.create_labels(ohlcv_df)

    With Optimization:
        >>> from src.data.labeling import LabelOptimizer, optimize_labels
        >>> result = optimize_labels(ohlcv_df, feature_df, n_trials=100)
        >>> print(f"Best config: {result.best_config}")
        >>> labels = TripleBarrierLabeler(result.best_config).create_labels(ohlcv_df)

Integration with PipelineConfig:
    >>> from src.core import PipelineConfig
    >>> config = PipelineConfig(
    ...     symbol="MES",
    ...     data_path="./data/mes.parquet",
    ...     output_dir="./experiments/exp_001",
    ...     labeling_method="triple_barrier",
    ...     optimize_labels=True,
    ...     label_optimization_trials=100,
    ... )
"""

from .base import (
    LabelingResult,
    LabelingStrategy,
    LabelingType,
)
from .optimization import (
    LabelOptimizationResult,
    LabelOptimizer,
    optimize_labels,
)
from .triple_barrier import (
    TripleBarrierConfig,
    TripleBarrierLabeler,
    triple_barrier_numba,
    triple_barrier_numba_with_costs,
)

__all__ = [
    # Base classes
    "LabelingResult",
    "LabelingStrategy",
    "LabelingType",
    # Configuration
    "TripleBarrierConfig",
    # Labeler
    "TripleBarrierLabeler",
    # Numba functions
    "triple_barrier_numba",
    "triple_barrier_numba_with_costs",
    # Optimization
    "LabelOptimizationResult",
    "LabelOptimizer",
    "optimize_labels",
]
