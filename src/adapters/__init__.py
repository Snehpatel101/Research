"""
Adapters package - Convert canonical data to model-specific formats.

Phase 2 SNwH Implementation.

This package provides adapters that transform canonical DataFrames
into model-specific numpy array formats:
- TabularAdapter: 2D arrays (n_samples, n_features) for boosting/classical
- SequenceAdapter: 3D arrays (n_samples, seq_len, n_features) for RNN/TCN
- MultiStreamAdapter: 4D arrays (n_samples, n_tfs, seq_len, n_features) for transformers

Usage:
    from src.adapters import get_adapter, AdapterRegistry

    # Get adapter for a model (uses contract)
    adapter = get_adapter(model_name="xgboost")
    result = adapter.transform(df)
    # result.X.shape = (n_samples, n_features)

    # Get adapter for sequence model
    adapter = get_adapter(model_name="lstm", sequence_length=60)
    result = adapter.transform(df)
    # result.X.shape = (n_sequences, 60, n_features)

    # Get adapter by ID directly
    adapter = get_adapter(adapter_id="tabular")
    result = adapter.transform(df)
"""

from .base import AdapterResult, BaseAdapter
from .registry import AdapterRegistry, get_adapter

# Import adapters to register them
from .tabular import TabularAdapter
from .sequence import SequenceAdapter
from .multi_stream import MultiStreamAdapter

__all__ = [
    # Registry
    "AdapterRegistry",
    "get_adapter",
    # Base classes
    "BaseAdapter",
    "AdapterResult",
    # Adapters
    "TabularAdapter",
    "SequenceAdapter",
    "MultiStreamAdapter",
]
