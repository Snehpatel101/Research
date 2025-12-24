# Phase 1 Status & Phase 2 Readiness

## Overview

Phase 1 is **COMPLETE** as a modular data preparation pipeline. This document tracks the single blocker for Phase 2.

**Overall Score:** 8.5/10 - Production-ready for data prep
**Phase 2 Blocker:** TimeSeriesDataContainer (3-5 days effort)

---

## Phase 1 Completion Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Data Ingestion | COMPLETE | Validates OHLCV schema, handles CSV/Parquet |
| Data Cleaning | COMPLETE | 1m → 5m resampling, gap filling, outlier removal |
| Feature Engineering | COMPLETE | 107 features, configurable periods |
| Triple-Barrier Labeling | COMPLETE | Symbol-specific asymmetric barriers (MES 1.5:1.0) |
| GA Optimization | COMPLETE | DEAP-based barrier tuning with Sharpe fitness |
| Train/Val/Test Splits | COMPLETE | 70/15/15 with purge (60) + embargo (288) |
| Feature Scaling | COMPLETE | Train-only fit, RobustScaler with clipping |
| Validation | COMPLETE | Schema, leakage, distribution checks |

### Phase 1 Outputs

```
data/splits/scaled/
├── train_scaled.parquet    (87,094 × 126)
├── val_scaled.parquet      (18,591 × 126)
├── test_scaled.parquet     (18,592 × 126)
├── feature_scaler.pkl      (for production inference)
└── scaling_metadata.json   (column info)
```

---

## Critical Path for Phase 2

### TimeSeriesDataContainer - REQUIRED (NOT BUILT)

Phase 2 models need proper sequence windowing. Current Phase 1 outputs flat parquet files. Neural models need:

- Sliding window generation (seq_len=128)
- Symbol-isolated sequences (no MES→MGC bleeding)
- Encoder/decoder length separation (for TFT)
- Multiple output formats (sklearn arrays, PyTorch datasets, NeuralForecast df)

**Specification:**

```python
from dataclasses import dataclass
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

@dataclass
class TimeSeriesDataContainer:
    """
    Universal container for Phase 1 outputs that provides model-specific formats.

    This is the ONLY missing piece for Phase 2. All Phase 1 data prep is complete,
    but models need proper windowing/sequencing interfaces.
    """
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_columns: List[str]
    label_column: str  # e.g., 'label_h5'
    horizon: int       # e.g., 5
    freq: str = "5min"

    # Computed on init
    _feature_count: int = None
    _train_size: int = None

    def __post_init__(self):
        self._feature_count = len(self.feature_columns)
        self._train_size = len(self.train_df[self.train_df[self.label_column] != -99])
        self._validate()

    def _validate(self):
        """Validate data consistency."""
        # Check all feature columns exist
        for col in self.feature_columns:
            assert col in self.train_df.columns, f"Missing feature: {col}"

        # Check label column exists
        assert self.label_column in self.train_df.columns

        # Check no NaN in features (after filtering invalid labels)
        valid_train = self.train_df[self.train_df[self.label_column] != -99]
        nan_count = valid_train[self.feature_columns].isnull().sum().sum()
        assert nan_count == 0, f"Found {nan_count} NaN values in features"

    # ========== OUTPUT FORMATS ==========

    def get_sklearn_arrays(self, split: str = 'train') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        For boosting models (XGBoost, LightGBM, CatBoost).
        Returns flat X, y, sample_weight arrays.

        Args:
            split: 'train', 'val', or 'test'

        Returns:
            X: (n_samples, n_features) array
            y: (n_samples,) array of labels
            weights: (n_samples,) array of sample weights
        """
        df = self._get_split(split)
        df = df[df[self.label_column] != -99]  # Filter invalid

        X = df[self.feature_columns].values
        y = df[self.label_column].values
        weight_col = f'sample_weight_h{self.horizon}'
        weights = df[weight_col].values if weight_col in df.columns else np.ones(len(df))

        return X, y, weights

    def get_pytorch_sequences(
        self,
        split: str = 'train',
        seq_len: int = 128,
        stride: int = 1
    ) -> Dataset:
        """
        For neural models (LSTM, TCN, Transformers).
        Returns PyTorch Dataset with windowed sequences.

        CRITICAL: Sequences are symbol-isolated (no MES→MGC bleeding).

        Args:
            split: 'train', 'val', or 'test'
            seq_len: Lookback window length
            stride: Step between windows (1 = fully overlapping)

        Returns:
            PyTorch Dataset yielding (X_seq, y, weight) tuples
            X_seq: (seq_len, n_features) tensor
            y: scalar label
            weight: scalar sample weight
        """
        df = self._get_split(split)
        return _SequenceDataset(
            df=df,
            feature_columns=self.feature_columns,
            label_column=self.label_column,
            horizon=self.horizon,
            seq_len=seq_len,
            stride=stride
        )

    def get_neuralforecast_df(self, split: str = 'train') -> pd.DataFrame:
        """
        For NeuralForecast library (N-HiTS, TFT, PatchTST, TimesNet).
        Returns DataFrame in [unique_id, ds, y, features...] format.

        Args:
            split: 'train', 'val', or 'test'

        Returns:
            DataFrame with columns: unique_id, ds, y, feature_1, ..., feature_n
        """
        df = self._get_split(split)
        df = df[df[self.label_column] != -99]

        result = pd.DataFrame({
            'unique_id': df['symbol'],
            'ds': df['datetime'],
            'y': df[self.label_column],
        })

        # Add features as exogenous variables
        for col in self.feature_columns:
            result[col] = df[col].values

        return result

    def get_darts_series(self, split: str = 'train'):
        """
        For Darts library (alternative to NeuralForecast).
        Returns list of TimeSeries objects per symbol.
        """
        from darts import TimeSeries

        df = self._get_split(split)
        series_list = []

        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].sort_values('datetime')
            symbol_df = symbol_df[symbol_df[self.label_column] != -99]

            ts = TimeSeries.from_dataframe(
                symbol_df,
                time_col='datetime',
                value_cols=self.feature_columns + [self.label_column],
                fill_missing_dates=True,
                freq=self.freq
            )
            series_list.append(ts)

        return series_list

    # ========== HELPER METHODS ==========

    def _get_split(self, split: str) -> pd.DataFrame:
        """Get DataFrame for specified split."""
        if split == 'train':
            return self.train_df
        elif split == 'val':
            return self.val_df
        elif split == 'test':
            return self.test_df
        else:
            raise ValueError(f"Unknown split: {split}")

    @classmethod
    def from_parquet_dir(cls, path: str, horizon: int = 5):
        """
        Load from Phase 1 output directory.

        Args:
            path: Path to scaled/ directory
            horizon: Which horizon to use for labels (5, 10, 15, or 20)
        """
        import json
        from pathlib import Path

        base = Path(path)

        train_df = pd.read_parquet(base / 'train_scaled.parquet')
        val_df = pd.read_parquet(base / 'val_scaled.parquet')
        test_df = pd.read_parquet(base / 'test_scaled.parquet')

        # Load feature list from metadata
        with open(base / 'scaling_metadata.json') as f:
            metadata = json.load(f)

        feature_columns = metadata.get('scaled_columns', [])

        return cls(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            feature_columns=feature_columns,
            label_column=f'label_h{horizon}',
            horizon=horizon
        )


class _SequenceDataset(Dataset):
    """
    Internal PyTorch Dataset for windowed sequences.

    CRITICAL INVARIANT: Sequences never cross symbol boundaries.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        label_column: str,
        horizon: int,
        seq_len: int = 128,
        stride: int = 1
    ):
        self.feature_columns = feature_columns
        self.label_column = label_column
        self.horizon = horizon
        self.seq_len = seq_len

        # Build index of valid sequences (symbol-isolated)
        self.sequences = []

        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].sort_values('datetime')
            valid_mask = symbol_df[label_column] != -99

            # Get indices where we can form complete sequences
            valid_indices = symbol_df[valid_mask].index.tolist()

            for i in range(seq_len - 1, len(valid_indices), stride):
                # Check all seq_len elements are valid and contiguous
                end_idx = valid_indices[i]
                start_idx = valid_indices[i - seq_len + 1]

                # Verify contiguity (indices should be sequential)
                if end_idx - start_idx == seq_len - 1:
                    self.sequences.append((symbol_df, start_idx, end_idx))

        # Pre-extract data for efficiency
        self._precompute(df)

    def _precompute(self, df: pd.DataFrame):
        """Pre-extract features and labels for all sequences."""
        self.X_data = []
        self.y_data = []
        self.w_data = []

        weight_col = f'sample_weight_h{self.horizon}'

        for symbol_df, start_idx, end_idx in self.sequences:
            seq_df = symbol_df.loc[start_idx:end_idx]

            X = seq_df[self.feature_columns].values  # (seq_len, n_features)
            y = seq_df[self.label_column].iloc[-1]   # Label at sequence end
            w = seq_df[weight_col].iloc[-1] if weight_col in seq_df.columns else 1.0

            self.X_data.append(X)
            self.y_data.append(y)
            self.w_data.append(w)

        # Convert to tensors
        self.X_data = torch.tensor(np.array(self.X_data), dtype=torch.float32)
        self.y_data = torch.tensor(np.array(self.y_data), dtype=torch.long)
        self.w_data = torch.tensor(np.array(self.w_data), dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.X_data[idx], self.y_data[idx], self.w_data[idx]
```

**Location:** `src/datasets/container.py` (new)
**Effort:** 3-5 days
**Tests:** `tests/test_timeseries_container.py` (new)

### PURGE_BARS Alignment - CHECK NEEDED

If using transformer models with `seq_len=128`, verify:

```python
PURGE_BARS = max(60, 128)  # Must be >= max sequence length
```

Current: `PURGE_BARS = 60` (correct for boosting, may need increase for transformers)

---

## Architecture Review Summary

**Score: 8/10** - Solid foundation with addressable gaps

### What's Good

| Aspect | Score | Notes |
|--------|-------|-------|
| Plugin Architecture | 9/10 | Industry-standard decorator registration |
| Leakage Prevention | 9/10 | 4-layer defense (purge, embargo, train-only scaling) |
| Data Flow | 8/10 | Clean parquet outputs, proper split ratios |
| Triple-Barrier Labeling | 9/10 | GA-optimized, symbol-specific barriers |
| Modular Design | 8/10 | 650-line limit enforced, clear stage separation |

### Gaps to Address (Phase 2-4)

| Gap | Priority | When |
|-----|----------|------|
| Probability Calibration | Medium | Phase 2 (after base model training) |
| Regime-Aware Meta-Learner | Medium | Phase 3 (ensemble) |
| Walk-Forward CV | Medium | Phase 3 (validation) |
| Prediction Confidence | Low | Phase 4 (trading) |
| Model Lifecycle Management | Low | Phase 5 (production) |

---

## Quick Start for Phase 2

```python
# 1. Load Phase 1 data
from src.datasets.container import TimeSeriesDataContainer

container = TimeSeriesDataContainer.from_parquet_dir(
    'data/splits/scaled/',
    horizon=5
)

# 2. For XGBoost
X_train, y_train, weights = container.get_sklearn_arrays('train')

# 3. For LSTM/TCN
train_dataset = container.get_pytorch_sequences('train', seq_len=128)

# 4. For NeuralForecast
train_df = container.get_neuralforecast_df('train')
```

---

## Known Limitations (By Design)

1. **No Cross-Asset Features** - Disabled by default (single-symbol trading)
2. **Batch Processing Only** - No streaming support (Phase 5)
3. **Framework-Agnostic** - No PyTorch/TF in Phase 1 (intentional separation)
4. **Parquet Output** - Universal format, any framework can consume

---

**Last Updated:** 2025-12-24
**Next Step:** Build TimeSeriesDataContainer, then start Phase 2
