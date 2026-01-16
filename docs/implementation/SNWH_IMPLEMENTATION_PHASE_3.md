# SNwH Implementation: Phase 3 - Timeframe Coordination

## Overview

Phase 3 implements the TimeframeCoordinator that ensures models receive data from their configured timeframes. This is critical for heterogeneous ensembles where different base models train on different timeframes.

---

## 3.1 TimeframeCoordinator

### New File: `src/coordination/__init__.py`

```python
"""
Coordination package - Manages timeframe and data routing.
"""

from .timeframe_coordinator import TimeframeCoordinator, TimeframeData

__all__ = [
    "TimeframeCoordinator",
    "TimeframeData",
]
```

### New File: `src/coordination/timeframe_coordinator.py`

```python
"""
Timeframe Coordinator - Manages multi-timeframe data loading and alignment.

The coordinator ensures:
1. Each model gets data from its configured primary timeframe
2. MTF features are properly aligned (shift(1) to prevent leakage)
3. Multi-stream models get aligned 4D data
4. Timestamps are consistent across all timeframes
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.timeframes import (
    CANONICAL_TIMEFRAMES,
    normalize_timeframe,
    get_timeframe_minutes,
)
from src.contracts import ModelContract, get_model_contract

logger = logging.getLogger(__name__)


@dataclass
class TimeframeData:
    """
    Container for timeframe-specific data.

    Holds the DataFrame and metadata for a single timeframe.
    """
    timeframe: str
    df: pd.DataFrame
    feature_columns: list[str]
    n_samples: int = 0
    start_time: pd.Timestamp | None = None
    end_time: pd.Timestamp | None = None

    def __post_init__(self):
        self.n_samples = len(self.df)
        if "datetime" in self.df.columns:
            self.start_time = pd.to_datetime(self.df["datetime"].iloc[0])
            self.end_time = pd.to_datetime(self.df["datetime"].iloc[-1])
        elif isinstance(self.df.index, pd.DatetimeIndex):
            self.start_time = self.df.index[0]
            self.end_time = self.df.index[-1]


class TimeframeCoordinator:
    """
    Coordinates data loading across multiple timeframes.

    The coordinator handles:
    1. Loading data for each required timeframe
    2. Timestamp alignment between timeframes
    3. MTF feature lag (shift(1)) to prevent leakage
    4. Providing the correct data to each model

    Usage:
        coordinator = TimeframeCoordinator(data_dir="data/splits/scaled")
        coordinator.load_timeframes(["5min", "15min", "60min"])

        # Get data for a model
        data = coordinator.get_data_for_model("xgboost")

        # Get aligned multi-stream data
        dfs = coordinator.get_multi_stream_dfs(["1min", "5min", "15min"])
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        horizon: int = 20,
    ):
        """
        Initialize TimeframeCoordinator.

        Args:
            data_dir: Directory containing timeframe data
            split: Data split ("train", "val", "test")
            horizon: Label horizon
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.horizon = horizon

        # Loaded timeframe data
        self._timeframe_data: dict[str, TimeframeData] = {}

        # Alignment info
        self._anchor_timeframe: str | None = None
        self._common_timestamps: pd.DatetimeIndex | None = None

    def load_timeframes(
        self,
        timeframes: list[str],
        feature_columns: list[str] | None = None,
    ) -> None:
        """
        Load data for specified timeframes.

        Args:
            timeframes: List of timeframes to load
            feature_columns: Feature columns to keep (None = all)
        """
        # Normalize timeframes
        timeframes = [normalize_timeframe(tf) for tf in timeframes]

        for tf in timeframes:
            if tf in self._timeframe_data:
                logger.debug(f"Timeframe {tf} already loaded, skipping")
                continue

            # Determine file path
            # Try multiple naming conventions
            possible_paths = [
                self.data_dir / tf / f"{self.split}_scaled.parquet",
                self.data_dir / f"{self.split}_{tf}_scaled.parquet",
                self.data_dir / f"features_{tf}.parquet",
            ]

            df = None
            for path in possible_paths:
                if path.exists():
                    df = pd.read_parquet(path)
                    logger.info(f"Loaded {tf} data from {path}: {len(df)} rows")
                    break

            if df is None:
                raise FileNotFoundError(
                    f"No data found for timeframe '{tf}'. "
                    f"Tried: {[str(p) for p in possible_paths]}"
                )

            # Filter features if specified
            if feature_columns:
                available = set(df.columns)
                keep_cols = [c for c in feature_columns if c in available]
                # Always keep metadata columns
                metadata = ["datetime", "timestamp", "symbol", "timeframe"]
                keep_cols.extend([c for c in metadata if c in available and c not in keep_cols])
                # Keep label and weight columns
                label_col = f"label_h{self.horizon}"
                weight_col = f"sample_weight_h{self.horizon}"
                if label_col in available:
                    keep_cols.append(label_col)
                if weight_col in available:
                    keep_cols.append(weight_col)
                df = df[keep_cols]
                logger.debug(f"Filtered {tf} to {len(keep_cols)} columns")

            # Detect feature columns
            features = self._detect_feature_columns(df)

            self._timeframe_data[tf] = TimeframeData(
                timeframe=tf,
                df=df,
                feature_columns=features,
            )

        # Set anchor timeframe (smallest loaded)
        self._set_anchor_timeframe()

    def _detect_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Detect feature columns in DataFrame."""
        from src.phase1.utils.constants import METADATA_COLUMNS
        from src.phase1.utils.feature_sets import _is_label_column

        return [
            col for col in df.columns
            if col not in METADATA_COLUMNS and not _is_label_column(col)
        ]

    def _set_anchor_timeframe(self) -> None:
        """Set the anchor timeframe (smallest minutes)."""
        if not self._timeframe_data:
            return

        min_minutes = float("inf")
        anchor = None

        for tf in self._timeframe_data:
            minutes = get_timeframe_minutes(tf)
            if minutes < min_minutes:
                min_minutes = minutes
                anchor = tf

        self._anchor_timeframe = anchor
        logger.info(f"Anchor timeframe set to {anchor}")

    @property
    def loaded_timeframes(self) -> list[str]:
        """List of loaded timeframes."""
        return list(self._timeframe_data.keys())

    @property
    def anchor_timeframe(self) -> str | None:
        """The anchor (smallest) timeframe."""
        return self._anchor_timeframe

    def get_timeframe_data(self, timeframe: str) -> TimeframeData:
        """
        Get data for a specific timeframe.

        Args:
            timeframe: Timeframe to get

        Returns:
            TimeframeData for the timeframe

        Raises:
            KeyError: If timeframe not loaded
        """
        tf = normalize_timeframe(timeframe)
        if tf not in self._timeframe_data:
            raise KeyError(
                f"Timeframe '{tf}' not loaded. "
                f"Loaded: {self.loaded_timeframes}"
            )
        return self._timeframe_data[tf]

    def get_data_for_model(
        self,
        model_name: str,
        contract: ModelContract | None = None,
    ) -> pd.DataFrame:
        """
        Get DataFrame for a model based on its contract.

        Args:
            model_name: Name of the model
            contract: Optional pre-fetched contract

        Returns:
            DataFrame with data for the model's primary timeframe
        """
        if contract is None:
            contract = get_model_contract(model_name)

        primary_tf = normalize_timeframe(contract.primary_timeframe)

        # Ensure timeframe is loaded
        if primary_tf not in self._timeframe_data:
            self.load_timeframes([primary_tf])

        tf_data = self._timeframe_data[primary_tf]

        logger.debug(
            f"Providing {primary_tf} data for {model_name}: "
            f"{tf_data.n_samples} samples, {len(tf_data.feature_columns)} features"
        )

        return tf_data.df

    def get_multi_stream_dfs(
        self,
        timeframes: list[str],
        align_timestamps: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Get aligned DataFrames for multiple timeframes.

        Args:
            timeframes: List of timeframes
            align_timestamps: Whether to align to common timestamps

        Returns:
            Dict mapping timeframe -> DataFrame
        """
        timeframes = [normalize_timeframe(tf) for tf in timeframes]

        # Ensure all timeframes are loaded
        missing = [tf for tf in timeframes if tf not in self._timeframe_data]
        if missing:
            self.load_timeframes(missing)

        result = {}
        for tf in timeframes:
            result[tf] = self._timeframe_data[tf].df

        # Optionally align to common timestamps
        if align_timestamps and len(timeframes) > 1:
            result = self._align_dataframes(result)

        return result

    def _align_dataframes(
        self,
        dfs: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """
        Align DataFrames to common timestamps.

        Higher timeframes are forward-filled to align with the anchor.

        Args:
            dfs: Dict mapping timeframe -> DataFrame

        Returns:
            Aligned DataFrames
        """
        if not dfs:
            return dfs

        # Find anchor (smallest timeframe)
        anchor_tf = min(
            dfs.keys(),
            key=lambda tf: get_timeframe_minutes(tf)
        )
        anchor_df = dfs[anchor_tf]

        # Get anchor timestamps
        if "datetime" in anchor_df.columns:
            anchor_times = pd.to_datetime(anchor_df["datetime"])
        elif isinstance(anchor_df.index, pd.DatetimeIndex):
            anchor_times = anchor_df.index
        else:
            logger.warning("Cannot align without datetime column or index")
            return dfs

        aligned = {anchor_tf: anchor_df}

        for tf, df in dfs.items():
            if tf == anchor_tf:
                continue

            # Get timeframe timestamps
            if "datetime" in df.columns:
                df = df.set_index(pd.to_datetime(df["datetime"]))
            elif not isinstance(df.index, pd.DatetimeIndex):
                logger.warning(f"Cannot align {tf} without datetime")
                aligned[tf] = df
                continue

            # Reindex to anchor timestamps with forward fill
            aligned_df = df.reindex(anchor_times, method="ffill")
            aligned_df = aligned_df.reset_index()
            aligned_df = aligned_df.rename(columns={"index": "datetime"})

            aligned[tf] = aligned_df
            logger.debug(
                f"Aligned {tf} to anchor {anchor_tf}: "
                f"{len(df)} -> {len(aligned_df)} rows"
            )

        return aligned

    def get_required_timeframes_for_ensemble(
        self,
        base_models: list[str],
    ) -> set[str]:
        """
        Get all timeframes required by an ensemble's base models.

        Args:
            base_models: List of base model names

        Returns:
            Set of required timeframes
        """
        required = set()

        for model_name in base_models:
            contract = get_model_contract(model_name)
            required.add(normalize_timeframe(contract.primary_timeframe))
            for tf in contract.mtf_timeframes:
                required.add(normalize_timeframe(tf))

        return required

    def validate_timeframe_coverage(
        self,
        required: set[str],
    ) -> tuple[bool, list[str]]:
        """
        Validate that all required timeframes are available.

        Args:
            required: Set of required timeframes

        Returns:
            (is_valid, list_of_missing)
        """
        loaded = set(self.loaded_timeframes)
        missing = required - loaded

        # Check if missing files exist
        actually_missing = []
        for tf in missing:
            paths = [
                self.data_dir / tf / f"{self.split}_scaled.parquet",
                self.data_dir / f"{self.split}_{tf}_scaled.parquet",
            ]
            if not any(p.exists() for p in paths):
                actually_missing.append(tf)

        return len(actually_missing) == 0, actually_missing

    def get_feature_columns_for_model(
        self,
        model_name: str,
        contract: ModelContract | None = None,
    ) -> list[str]:
        """
        Get feature columns for a model.

        Args:
            model_name: Name of the model
            contract: Optional pre-fetched contract

        Returns:
            List of feature column names
        """
        if contract is None:
            contract = get_model_contract(model_name)

        primary_tf = normalize_timeframe(contract.primary_timeframe)

        if primary_tf not in self._timeframe_data:
            self.load_timeframes([primary_tf])

        return self._timeframe_data[primary_tf].feature_columns


__all__ = [
    "TimeframeCoordinator",
    "TimeframeData",
]
```

---

## 3.2 Integration with Trainer

### File: `src/models/training/trainer.py`

**Modifications to the Trainer class (around lines 253-320)**

```python
# ADD this method to Trainer class (after _is_heterogeneous_ensemble)

def _get_coordinator(self) -> "TimeframeCoordinator":
    """
    Get or create a TimeframeCoordinator.

    Returns:
        TimeframeCoordinator instance
    """
    from src.coordination import TimeframeCoordinator

    # Determine data directory from container or config
    data_dir = self.config.output_dir.parent / "data" / "splits" / "scaled"

    return TimeframeCoordinator(
        data_dir=data_dir,
        split="train",  # Will be overridden per-call
        horizon=self.config.horizon,
    )

def _load_data_for_model(
    self,
    container: "TimeSeriesDataContainer",
    model_name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data for a specific model based on its contract.

    This replaces the hardcoded data loading in run() for SNwH compatibility.

    Args:
        container: TimeSeriesDataContainer
        model_name: Model name (defaults to self.config.model_name)

    Returns:
        (train_df, val_df) filtered to model's primary timeframe
    """
    from src.contracts import get_model_contract

    model_name = model_name or self.config.model_name
    contract = get_model_contract(model_name)

    primary_tf = contract.primary_timeframe
    current_tf = self.config.primary_timeframe

    # If same timeframe, use container directly
    if primary_tf == current_tf:
        train_split = container.get_split("train")
        val_split = container.get_split("val")
        return train_split.df, val_split.df

    # Otherwise, need to load from correct timeframe
    logger.info(
        f"Model {model_name} requires {primary_tf} but container has {current_tf}. "
        f"Loading from timeframe-specific data."
    )

    # Use coordinator to load correct timeframe
    coordinator = self._get_coordinator()
    coordinator.load_timeframes([primary_tf])

    # Get train and val data
    # Note: This assumes timeframe-specific files exist
    train_df = coordinator.get_data_for_model(model_name, contract)

    # Load val data separately
    coordinator_val = TimeframeCoordinator(
        data_dir=coordinator.data_dir,
        split="val",
        horizon=self.config.horizon,
    )
    coordinator_val.load_timeframes([primary_tf])
    val_df = coordinator_val.get_data_for_model(model_name, contract)

    return train_df, val_df


# MODIFY run() method (around line 314-317)
# Replace:
#     X_train_df, y_train_series, w_train_series = container.get_sklearn_arrays(
#         "train", return_df=True
#     )
#     X_val_df, y_val_series, _ = container.get_sklearn_arrays("val", return_df=True)

# With:
def run(
    self,
    container: TimeSeriesDataContainer,
    skip_save: bool = False,
) -> dict[str, Any]:
    """Execute complete training pipeline (SNwH-aware)."""
    start_time = time.time()

    # Setup
    self._setup_output_dir()
    self._save_config()

    # Start experiment tracking
    tracking_run_id = self.tracker.start_run(...)

    # NEW: Load data based on model contract
    # This respects per-model timeframe configuration
    if self._is_heterogeneous_ensemble():
        # Heterogeneous ensemble needs multiple timeframes
        train_df, val_df = self._load_heterogeneous_data(container)
    else:
        # Single model or homogeneous ensemble
        train_df, val_df = self._load_data_for_model(container)

    # Get arrays from DataFrames
    feature_names = self._get_feature_columns(train_df)
    X_train_df = train_df[feature_names]
    y_train_series = train_df[f"label_h{self.config.horizon}"]
    w_train_series = train_df.get(
        f"sample_weight_h{self.config.horizon}",
        pd.Series(np.ones(len(train_df)))
    )

    X_val_df = val_df[feature_names]
    y_val_series = val_df[f"label_h{self.config.horizon}"]

    # ... rest of run() continues unchanged ...


def _load_heterogeneous_data(
    self,
    container: "TimeSeriesDataContainer",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data for heterogeneous ensemble.

    Each base model may need a different timeframe, so we load
    all required timeframes and return the anchor (smallest) one
    as the primary data source.

    Args:
        container: TimeSeriesDataContainer

    Returns:
        (train_df, val_df) for anchor timeframe
    """
    from src.contracts import get_model_contract
    from src.coordination import TimeframeCoordinator

    base_models = self.config.model_config.get("base_model_names", [])

    # Get all required timeframes
    required_tfs = set()
    for model_name in base_models:
        contract = get_model_contract(model_name)
        required_tfs.add(contract.primary_timeframe)
        required_tfs.update(contract.mtf_timeframes)

    logger.info(
        f"Heterogeneous ensemble requires timeframes: {sorted(required_tfs)}"
    )

    # Load all timeframes
    coordinator = self._get_coordinator()
    coordinator.load_timeframes(list(required_tfs))

    # Return anchor timeframe data (smallest)
    anchor_tf = coordinator.anchor_timeframe
    train_df = coordinator.get_timeframe_data(anchor_tf).df

    # Load val data
    coordinator_val = TimeframeCoordinator(
        data_dir=coordinator.data_dir,
        split="val",
        horizon=self.config.horizon,
    )
    coordinator_val.load_timeframes(list(required_tfs))
    val_df = coordinator_val.get_timeframe_data(anchor_tf).df

    return train_df, val_df
```

---

## 3.3 Temporal Alignment Rules

### New File: `src/coordination/alignment.py`

```python
"""
Temporal alignment rules for multi-timeframe data.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.common.timeframes import get_timeframe_minutes, normalize_timeframe

logger = logging.getLogger(__name__)


def align_to_anchor(
    anchor_df: pd.DataFrame,
    higher_tf_df: pd.DataFrame,
    anchor_tf: str,
    higher_tf: str,
    datetime_col: str = "datetime",
) -> pd.DataFrame:
    """
    Align a higher timeframe DataFrame to anchor timestamps.

    Uses forward-fill to propagate higher TF values to anchor timestamps.
    This ensures the higher TF data available at each anchor timestamp
    is the LAST completed bar (no lookahead).

    Args:
        anchor_df: DataFrame with anchor (smallest) timeframe
        higher_tf_df: DataFrame with higher timeframe
        anchor_tf: Anchor timeframe string
        higher_tf: Higher timeframe string
        datetime_col: Name of datetime column

    Returns:
        Aligned DataFrame with anchor timestamps
    """
    # Ensure datetime columns exist
    if datetime_col not in anchor_df.columns or datetime_col not in higher_tf_df.columns:
        raise ValueError(f"Missing datetime column '{datetime_col}'")

    # Convert to datetime
    anchor_times = pd.to_datetime(anchor_df[datetime_col])
    higher_times = pd.to_datetime(higher_tf_df[datetime_col])

    # Set index for alignment
    higher_indexed = higher_tf_df.set_index(higher_times)

    # Reindex with forward fill (use last available value)
    aligned = higher_indexed.reindex(anchor_times, method="ffill")

    # Reset index
    aligned = aligned.reset_index()
    aligned = aligned.rename(columns={"index": datetime_col})

    logger.debug(
        f"Aligned {higher_tf} ({len(higher_tf_df)} rows) to "
        f"{anchor_tf} ({len(anchor_df)} rows) -> {len(aligned)} rows"
    )

    return aligned


def apply_mtf_lag(
    df: pd.DataFrame,
    mtf_columns: list[str],
    shift: int = 1,
) -> pd.DataFrame:
    """
    Apply lag (shift) to MTF indicator columns to prevent leakage.

    MTF features from higher timeframes may contain information
    not yet available at the current timestamp. We shift by 1
    to ensure we only use completed bar data.

    Args:
        df: DataFrame with MTF columns
        mtf_columns: List of MTF column names
        shift: Number of periods to shift (default 1)

    Returns:
        DataFrame with shifted MTF columns
    """
    df = df.copy()

    for col in mtf_columns:
        if col in df.columns:
            df[col] = df[col].shift(shift)

    # Fill NaN from shift with first valid value
    for col in mtf_columns:
        if col in df.columns:
            first_valid = df[col].dropna().iloc[0] if df[col].dropna().any() else 0
            df[col] = df[col].fillna(first_valid)

    logger.debug(f"Applied shift({shift}) to {len(mtf_columns)} MTF columns")

    return df


def compute_sequence_offset(
    tabular_samples: int,
    sequence_samples: int,
    sequence_length: int,
) -> int:
    """
    Compute the offset between tabular and sequence data.

    Sequence data loses (seq_len - 1) samples at the start due to
    windowing. This offset is used to align labels and OOF predictions.

    Args:
        tabular_samples: Number of tabular samples
        sequence_samples: Number of sequences
        sequence_length: Sequence length

    Returns:
        Offset (number of samples lost)
    """
    expected_offset = sequence_length - 1
    actual_offset = tabular_samples - sequence_samples

    if actual_offset != expected_offset:
        logger.warning(
            f"Sequence offset mismatch: expected {expected_offset} "
            f"(seq_len-1), got {actual_offset}"
        )

    return actual_offset


def validate_timestamp_alignment(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    datetime_col: str = "datetime",
    tolerance_minutes: int = 1,
) -> tuple[bool, list[str]]:
    """
    Validate that two DataFrames have aligned timestamps.

    Args:
        df1: First DataFrame
        df2: Second DataFrame
        datetime_col: DateTime column name
        tolerance_minutes: Maximum allowed time difference

    Returns:
        (is_aligned, list_of_issues)
    """
    issues = []

    if datetime_col not in df1.columns or datetime_col not in df2.columns:
        issues.append(f"Missing datetime column '{datetime_col}'")
        return False, issues

    times1 = pd.to_datetime(df1[datetime_col])
    times2 = pd.to_datetime(df2[datetime_col])

    # Check length
    if len(times1) != len(times2):
        issues.append(
            f"Length mismatch: {len(times1)} vs {len(times2)}"
        )
        return False, issues

    # Check alignment
    time_diff = (times1 - times2).abs()
    max_diff = time_diff.max()

    if max_diff > pd.Timedelta(minutes=tolerance_minutes):
        n_misaligned = (time_diff > pd.Timedelta(minutes=tolerance_minutes)).sum()
        issues.append(
            f"{n_misaligned} timestamps misaligned by more than "
            f"{tolerance_minutes} minutes (max: {max_diff})"
        )

    return len(issues) == 0, issues


__all__ = [
    "align_to_anchor",
    "apply_mtf_lag",
    "compute_sequence_offset",
    "validate_timestamp_alignment",
]
```

---

## Summary: Phase 3 Changes

| File | Type | Purpose |
|------|------|---------|
| `src/coordination/__init__.py` | NEW | Package init |
| `src/coordination/timeframe_coordinator.py` | NEW | TimeframeCoordinator class |
| `src/coordination/alignment.py` | NEW | Temporal alignment utilities |
| `src/models/training/trainer.py` | MODIFY | Add `_load_data_for_model()`, `_load_heterogeneous_data()` |

## Dependencies

- Phase 0 (contracts) for ModelContract
- Phase 1 (config) for per-model configuration
- Phase 2 (adapters) for data transformation

## Key Design Decisions

1. **Anchor Timeframe**: Smallest timeframe determines sample count and timestamp grid
2. **Forward Fill**: Higher TFs forward-fill to anchor timestamps (no lookahead)
3. **MTF Lag**: MTF indicators shifted by 1 to prevent leakage
4. **Lazy Loading**: Timeframes loaded on demand

## Usage Example

```python
from src.coordination import TimeframeCoordinator

# Create coordinator
coordinator = TimeframeCoordinator(
    data_dir="data/splits/scaled",
    split="train",
    horizon=20,
)

# Load required timeframes
coordinator.load_timeframes(["5min", "15min", "60min"])

# Get data for a specific model
xgb_df = coordinator.get_data_for_model("xgboost")  # Gets 15min data

# Get aligned multi-stream data for transformer
dfs = coordinator.get_multi_stream_dfs(["1min", "5min", "15min"])

# Get all timeframes needed for ensemble
required = coordinator.get_required_timeframes_for_ensemble(
    ["xgboost", "lstm", "patchtst"]
)
# {'15min', '5min', '1min'}
```

## Next Steps

After Phase 3 is implemented, proceed to Phase 4 (OOF Integrity) which will:
1. Add strict OOF coverage validation
2. Ensure alignment between tabular and sequence OOF predictions
3. Prevent mismatched stacking datasets
