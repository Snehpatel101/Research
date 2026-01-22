"""
Data Labeling - Re-export from src.labeling.

New import path:
    from src.data.labeling import TripleBarrierLabeler, TripleBarrierConfig

Legacy import path (still works):
    from src.labeling import TripleBarrierLabeler, TripleBarrierConfig
"""

from src.labeling import (
    LabelingResult,
    LabelingStrategy,
    LabelingType,
    TripleBarrierConfig,
    TripleBarrierLabeler,
    triple_barrier_numba,
    triple_barrier_numba_with_costs,
    LabelOptimizationResult,
    LabelOptimizer,
    optimize_labels,
)

__all__ = [
    "LabelingResult",
    "LabelingStrategy",
    "LabelingType",
    "TripleBarrierConfig",
    "TripleBarrierLabeler",
    "triple_barrier_numba",
    "triple_barrier_numba_with_costs",
    "LabelOptimizationResult",
    "LabelOptimizer",
    "optimize_labels",
]
