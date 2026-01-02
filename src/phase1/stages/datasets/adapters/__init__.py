"""
Data Adapters for Multi-Resolution Time Series.

This module provides adapters that transform time series data into
different tensor formats for various model architectures:

- MultiResolution4DAdapter: 4D tensors (batch, timeframes, seq_len, features)
  for multi-resolution models that process multiple timeframes simultaneously

Usage:
------
    from src.phase1.stages.datasets.adapters import (
        MultiResolution4DAdapter,
        MultiResolution4DDataset,
        MultiResolution4DConfig,
    )

    # Create adapter
    adapter = MultiResolution4DAdapter(
        timeframes=['1min', '5min', '15min', '30min', '1h'],
        seq_len=60,
        features_per_timeframe=50
    )

    # Get PyTorch dataset
    dataset = adapter.create_dataset(df, feature_columns, label_column)

    # Use with DataLoader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=32)

    for X_4d, y, weights in loader:
        # X_4d: (batch, n_timeframes, seq_len, features)
        pass
"""

from src.phase1.stages.datasets.adapters.multi_resolution import (
    MultiResolution4DAdapter,
    MultiResolution4DConfig,
    MultiResolution4DDataset,
    create_multi_resolution_dataset,
)

__all__ = [
    "MultiResolution4DAdapter",
    "MultiResolution4DConfig",
    "MultiResolution4DDataset",
    "create_multi_resolution_dataset",
]
