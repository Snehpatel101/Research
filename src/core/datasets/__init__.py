"""
Core Datasets - PyTorch datasets and data utilities.

This module provides canonical implementations of:
- SequenceDataset: Sliding window sequences for LSTM/Transformer models
- Sequence utilities: Index building, symbol boundary detection

Usage:
    from src.core.datasets import SequenceDataset, SequenceConfig
    from src.core.datasets import build_sequence_indices, find_symbol_boundaries
"""

from src.core.datasets.sequences import (
    SequenceConfig,
    SequenceDataset,
    build_sequence_indices,
    create_sequence_dataset,
    find_symbol_boundaries,
)

__all__ = [
    "SequenceConfig",
    "SequenceDataset",
    "build_sequence_indices",
    "create_sequence_dataset",
    "find_symbol_boundaries",
]
