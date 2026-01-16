"""
Shared fixtures for Phase 3 Timeframe Coordination tests.

Provides:
- Sample OHLCV data at multiple timeframes
- Temporary directories with parquet files
- Mock coordinators for unit testing
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock


# =============================================================================
# TIMEFRAME GENERATION UTILITIES
# =============================================================================


def generate_ohlcv_df(
    n_rows: int,
    freq: str = "1min",
    base_time: datetime = None,
    base_price: float = 100.0,
    n_features: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data with features and labels.

    Args:
        n_rows: Number of rows to generate
        freq: Pandas frequency string (1min, 5min, 15min, etc.)
        base_time: Starting timestamp (default: market open)
        base_price: Starting price
        n_features: Number of feature columns to add
        seed: Random seed for reproducibility

    Returns:
        DataFrame with datetime index, OHLCV, features, labels, weights
    """
    np.random.seed(seed)

    if base_time is None:
        base_time = datetime(2024, 1, 15, 9, 30)

    # Generate timestamps
    timestamps = pd.date_range(start=base_time, periods=n_rows, freq=freq)

    # Generate prices with random walk
    returns = np.random.randn(n_rows) * 0.001  # 0.1% per bar
    prices = base_price * np.cumprod(1 + returns)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": prices * (1 + np.random.randn(n_rows) * 0.0005),
        "high": prices * (1 + np.abs(np.random.randn(n_rows) * 0.001)),
        "low": prices * (1 - np.abs(np.random.randn(n_rows) * 0.001)),
        "close": prices,
        "volume": np.random.randint(100, 10000, n_rows).astype(float),
    })

    # Ensure high >= open, close, low and low <= open, close, high
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)

    # Add feature columns
    for i in range(n_features):
        df[f"feature_{i}"] = np.random.randn(n_rows).astype(np.float32)

    # Add labels and weights
    df["label_h20"] = np.random.choice([-1, 0, 1], size=n_rows)
    df["sample_weight_h20"] = np.random.uniform(0.5, 1.5, size=n_rows).astype(np.float32)

    df.set_index("timestamp", inplace=True)
    return df


def resample_ohlcv(df: pd.DataFrame, target_freq: str) -> pd.DataFrame:
    """
    Resample OHLCV DataFrame to a higher timeframe.

    Args:
        df: Source DataFrame with datetime index
        target_freq: Target frequency (e.g., "5min", "15min")

    Returns:
        Resampled DataFrame
    """
    # OHLCV aggregation rules
    ohlcv_agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    # Resample OHLCV
    df_resampled = df[["open", "high", "low", "close", "volume"]].resample(
        target_freq
    ).agg(ohlcv_agg)

    # Resample features (use last value - matches point-in-time logic)
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    for col in feature_cols:
        df_resampled[col] = df[col].resample(target_freq).last()

    # Resample labels and weights
    if "label_h20" in df.columns:
        df_resampled["label_h20"] = df["label_h20"].resample(target_freq).last()
    if "sample_weight_h20" in df.columns:
        df_resampled["sample_weight_h20"] = df["sample_weight_h20"].resample(
            target_freq
        ).last()

    # Drop incomplete bars
    df_resampled = df_resampled.dropna(subset=["close"])

    return df_resampled


# =============================================================================
# FIXTURES - Basic Data
# =============================================================================


@pytest.fixture
def base_1min_df() -> pd.DataFrame:
    """Generate base 1-min data (200 rows)."""
    return generate_ohlcv_df(n_rows=200, freq="1min", seed=42)


@pytest.fixture
def base_5min_df(base_1min_df: pd.DataFrame) -> pd.DataFrame:
    """Generate 5-min data resampled from 1-min."""
    return resample_ohlcv(base_1min_df, "5min")


@pytest.fixture
def base_15min_df(base_1min_df: pd.DataFrame) -> pd.DataFrame:
    """Generate 15-min data resampled from 1-min."""
    return resample_ohlcv(base_1min_df, "15min")


@pytest.fixture
def base_60min_df(base_1min_df: pd.DataFrame) -> pd.DataFrame:
    """Generate 60-min data resampled from 1-min."""
    return resample_ohlcv(base_1min_df, "60min")


# =============================================================================
# FIXTURES - Larger Data for Integration Tests
# =============================================================================


@pytest.fixture
def large_1min_df() -> pd.DataFrame:
    """Generate larger 1-min data (1000 rows ~ 16 hours)."""
    return generate_ohlcv_df(n_rows=1000, freq="1min", seed=123)


@pytest.fixture
def large_5min_df(large_1min_df: pd.DataFrame) -> pd.DataFrame:
    """Generate 5-min data from large dataset."""
    return resample_ohlcv(large_1min_df, "5min")


@pytest.fixture
def large_15min_df(large_1min_df: pd.DataFrame) -> pd.DataFrame:
    """Generate 15-min data from large dataset."""
    return resample_ohlcv(large_1min_df, "15min")


# =============================================================================
# FIXTURES - Temporary Directories
# =============================================================================


@pytest.fixture
def tmp_data_dir_basic(
    tmp_path: Path,
    base_1min_df: pd.DataFrame,
    base_5min_df: pd.DataFrame,
    base_15min_df: pd.DataFrame,
) -> Path:
    """
    Create temporary directory with basic parquet files.

    Contains: 1min, 5min, 15min data
    """
    data_dir = tmp_path / "data" / "splits" / "scaled"
    data_dir.mkdir(parents=True, exist_ok=True)

    base_1min_df.to_parquet(data_dir / "features_1min.parquet")
    base_5min_df.to_parquet(data_dir / "features_5min.parquet")
    base_15min_df.to_parquet(data_dir / "features_15min.parquet")

    return data_dir


@pytest.fixture
def tmp_data_dir_large(
    tmp_path: Path,
    large_1min_df: pd.DataFrame,
    large_5min_df: pd.DataFrame,
    large_15min_df: pd.DataFrame,
) -> Path:
    """
    Create temporary directory with larger parquet files.

    Contains: 1min, 5min, 15min data (larger dataset)
    """
    data_dir = tmp_path / "data" / "splits" / "scaled"
    data_dir.mkdir(parents=True, exist_ok=True)

    large_1min_df.to_parquet(data_dir / "features_1min.parquet")
    large_5min_df.to_parquet(data_dir / "features_5min.parquet")
    large_15min_df.to_parquet(data_dir / "features_15min.parquet")

    return data_dir


@pytest.fixture
def tmp_data_dir_all_tfs(
    tmp_path: Path,
    base_1min_df: pd.DataFrame,
) -> Path:
    """
    Create temporary directory with all 9 canonical timeframes.

    Contains: 1min, 5min, 10min, 15min, 20min, 25min, 30min, 45min, 60min
    """
    data_dir = tmp_path / "data" / "splits" / "scaled"
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1min (canonical source)
    base_1min_df.to_parquet(data_dir / "features_1min.parquet")

    # Generate all other TFs from 1min
    timeframes = ["5min", "10min", "15min", "20min", "25min", "30min", "45min", "60min"]
    for tf in timeframes:
        df_tf = resample_ohlcv(base_1min_df, tf)
        df_tf.to_parquet(data_dir / f"features_{tf}.parquet")

    return data_dir


# =============================================================================
# FIXTURES - Mock Objects
# =============================================================================


@pytest.fixture
def mock_timeframe_data():
    """Create a mock TimeframeData object."""
    def _factory(
        timeframe: str = "5min",
        n_samples: int = 100,
        n_features: int = 10,
    ):
        mock = MagicMock()
        mock.timeframe = timeframe
        mock.n_samples = n_samples
        mock.feature_columns = [f"feature_{i}" for i in range(n_features)]
        mock.start_timestamp = datetime(2024, 1, 15, 9, 30)
        mock.end_timestamp = datetime(2024, 1, 15, 11, 30)
        mock.minutes = int(timeframe.replace("min", ""))

        # Create actual DataFrame
        df = generate_ohlcv_df(n_rows=n_samples, freq=timeframe, n_features=n_features)
        mock.df = df

        return mock

    return _factory


@pytest.fixture
def mock_coordinator(mock_timeframe_data):
    """Create a mock TimeframeCoordinator."""
    def _factory(loaded_tfs: list = None):
        if loaded_tfs is None:
            loaded_tfs = ["5min", "15min"]

        mock = MagicMock()
        mock._loaded_timeframes = {}

        for tf in loaded_tfs:
            mock._loaded_timeframes[tf] = mock_timeframe_data(timeframe=tf)

        def get_timeframe_data(tf):
            if tf not in mock._loaded_timeframes:
                raise KeyError(f"Timeframe {tf} not loaded")
            return mock._loaded_timeframes[tf]

        mock.get_timeframe_data = get_timeframe_data
        mock.anchor_timeframe = min(loaded_tfs, key=lambda x: int(x.replace("min", "")))

        return mock

    return _factory


# =============================================================================
# FIXTURES - Contract Helpers
# =============================================================================


@pytest.fixture
def model_contracts_by_tf() -> Dict[str, list]:
    """Map of primary timeframe to models that use it."""
    from src.contracts import MODEL_CONTRACTS

    tf_to_models: Dict[str, list] = {}
    for name, contract in MODEL_CONTRACTS.items():
        tf = contract.primary_timeframe
        if tf not in tf_to_models:
            tf_to_models[tf] = []
        tf_to_models[tf].append(name)

    return tf_to_models


@pytest.fixture
def multi_stream_models() -> list:
    """List of models that use multi-stream MTF mode."""
    from src.contracts import MODEL_CONTRACTS, MTFMode

    return [
        name for name, contract in MODEL_CONTRACTS.items()
        if contract.mtf_mode == MTFMode.MULTI_STREAM
    ]


# =============================================================================
# SKIP MARKERS
# =============================================================================


def _coordination_available() -> bool:
    """Check if coordination module is implemented."""
    try:
        from src.coordination import TimeframeCoordinator
        return True
    except ImportError:
        return False


requires_coordination = pytest.mark.skipif(
    not _coordination_available(),
    reason="src.coordination module not implemented"
)
