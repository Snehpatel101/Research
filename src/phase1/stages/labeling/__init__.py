"""
Labeling Strategies Module.

This module provides a unified interface for generating labels from price data
using various labeling strategies. All strategies implement the LabelingStrategy
interface, making them interchangeable in the pipeline.

Available Strategies
--------------------
- TripleBarrierLabeler: Lopez de Prado triple-barrier with ATR-based barriers
- AdaptiveTripleBarrierLabeler: Regime-adaptive triple-barrier labeling
- DirectionalLabeler: Simple direction of return labeling
- ThresholdLabeler: Percentage threshold-based labeling
- RegressionLabeler: Continuous return targets for regression models

Factory Functions
-----------------
- get_labeler: Create a labeling strategy by type
- get_available_strategies: List available strategy types
- create_multi_labeler: Create multiple labelers from config

Usage Examples
--------------
>>> from stages.labeling import get_labeler, LabelingType
>>>
>>> # Create a triple-barrier labeler
>>> labeler = get_labeler(LabelingType.TRIPLE_BARRIER, k_up=1.5, k_down=1.0)
>>> result = labeler.compute_labels(df, horizon=5)
>>>
>>> # Add labels to dataframe
>>> df = labeler.add_labels_to_dataframe(df, result)
>>>
>>> # Create by string name
>>> labeler = get_labeler('directional', threshold=0.001)
"""

# Base classes and types
from .base import LabelingResult, LabelingStrategy, LabelingType

# Concrete implementations
from .adaptive_barriers import AdaptiveTripleBarrierLabeler
from .directional import DirectionalLabeler
from .meta import BetSizeMethod, MetaLabeler
from .regression import RegressionLabeler
from .threshold import ThresholdLabeler
from .triple_barrier import TripleBarrierLabeler, triple_barrier_numba

# Factory functions
from .factory import (
    create_multi_labeler,
    get_available_strategies,
    get_labeler,
    register_strategy,
)

__all__ = [
    # Types and base classes
    'LabelingType',
    'LabelingStrategy',
    'LabelingResult',
    # Strategy implementations
    'TripleBarrierLabeler',
    'AdaptiveTripleBarrierLabeler',
    'DirectionalLabeler',
    'ThresholdLabeler',
    'RegressionLabeler',
    'MetaLabeler',
    'BetSizeMethod',
    # Numba functions (for backward compatibility)
    'triple_barrier_numba',
    # Factory functions
    'get_labeler',
    'get_available_strategies',
    'register_strategy',
    'create_multi_labeler',
]
