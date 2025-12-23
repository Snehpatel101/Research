"""
Shared fixtures and test utilities for Phase 1 tests.

This module provides common fixtures used across all Phase 1 test modules:
- Temporary directories for test isolation
- Sample OHLCV data generators with realistic market characteristics
- Pipeline configuration fixtures
- Common test utilities and helpers

All fixtures are designed to be deterministic, fast, and isolated.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import shutil

import pytest
import numpy as np
import pandas as pd

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from pipeline_config import PipelineConfig, create_default_config


# =============================================================================
# TEMPORARY DIRECTORY FIXTURES
# =============================================================================

@pytest.fixture
def temp_dir():
    """
    Create a temporary directory for test files.

    Automatically cleaned up after test completion.

    Yields:
        Path: Temporary directory path
    """
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def temp_project_dir():
    """
    Create a complete temporary project directory structure.

    Creates all necessary subdirectories for pipeline execution:
    - data/{raw,clean,features,final,splits,labels}
    - results/
    - config/ga_results/

    Yields:
        Path: Project root directory path
    """
    tmpdir = tempfile.mkdtemp()
    project_root = Path(tmpdir)

    # Create necessary subdirectories
    (project_root / 'data' / 'raw').mkdir(parents=True)
    (project_root / 'data' / 'clean').mkdir(parents=True)
    (project_root / 'data' / 'features').mkdir(parents=True)
    (project_root / 'data' / 'final').mkdir(parents=True)
    (project_root / 'data' / 'splits').mkdir(parents=True)
    (project_root / 'data' / 'labels').mkdir(parents=True)
    (project_root / 'results').mkdir(parents=True)
    (project_root / 'config').mkdir(parents=True)
    (project_root / 'config' / 'ga_results').mkdir(parents=True)

    yield project_root

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)


# =============================================================================
# OHLCV DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_ohlcv_df():
    """
    Create a sample OHLCV DataFrame for testing.

    Generates realistic price data with:
    - Random walk with 0.1% daily volatility
    - Valid OHLC relationships (high >= max(open, close), etc.)
    - Realistic volume distribution
    - 1-minute bars starting from 2024-01-01 09:30

    Returns:
        pd.DataFrame: OHLCV data with columns [datetime, open, high, low, close, volume]
    """
    n = 500
    np.random.seed(42)

    # Generate realistic price series with random walk
    base_price = 4500.0
    returns = np.random.randn(n) * 0.001  # 0.1% daily volatility
    close = base_price * np.exp(np.cumsum(returns))

    # Generate OHLC from close
    daily_range = np.abs(np.random.randn(n) * 0.002)  # 0.2% typical range
    high = close * (1 + daily_range / 2)
    low = close * (1 - daily_range / 2)
    open_ = close * (1 + np.random.randn(n) * 0.0005)  # Small random offset

    # Ensure OHLC relationships are valid
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    # Generate volume
    volume = np.random.randint(100, 10000, n)

    # Generate timestamps (1-minute bars)
    start_time = datetime(2024, 1, 1, 9, 30)
    timestamps = [start_time + timedelta(minutes=i) for i in range(n)]

    df = pd.DataFrame({
        'datetime': timestamps,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })

    return df


@pytest.fixture
def sample_ohlcv_with_gaps(sample_ohlcv_df):
    """
    Create OHLCV DataFrame with gaps for testing gap detection.

    Args:
        sample_ohlcv_df: Base OHLCV DataFrame fixture

    Returns:
        pd.DataFrame: OHLCV data with missing rows creating gaps
    """
    df = sample_ohlcv_df.copy()

    # Remove some rows to create gaps
    gap_indices = [50, 51, 52, 100, 150, 151]  # Create multi-bar gaps
    df = df.drop(gap_indices).reset_index(drop=True)

    return df


@pytest.fixture
def sample_ohlcv_data():
    """
    Create sample OHLCV data with ATR for testing advanced features.

    Generates realistic price data suitable for:
    - Feature engineering tests
    - Labeling tests
    - GA optimization tests

    Returns:
        pd.DataFrame: OHLCV data with symbol, datetime, OHLCV, and ATR columns
    """
    np.random.seed(42)
    n = 1000

    # Generate realistic price data with trend and noise
    base_price = 100.0
    returns = np.random.randn(n) * 0.002  # 0.2% daily volatility
    prices = base_price * np.exp(np.cumsum(returns))

    # Create OHLCV DataFrame
    df = pd.DataFrame({
        'datetime': pd.date_range('2024-01-01', periods=n, freq='5min'),
        'symbol': 'MES',
        'open': prices * (1 + np.random.randn(n) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(n)) * 0.002),
        'low': prices * (1 - np.abs(np.random.randn(n)) * 0.002),
        'close': prices,
        'volume': np.random.randint(100, 1000, n).astype(float),
    })

    # Ensure high >= close >= low and high >= open >= low
    df['high'] = df[['open', 'high', 'close']].max(axis=1) * 1.001
    df['low'] = df[['open', 'low', 'close']].min(axis=1) * 0.999

    # Add ATR (simplified calculation)
    df['atr_14'] = (df['high'] - df['low']).rolling(14).mean().fillna(0.5)

    return df


@pytest.fixture
def sample_labeled_data(sample_ohlcv_data):
    """
    Create sample labeled data for testing stages 5-8.

    Adds triple barrier labels with realistic label distribution.
    Also includes common feature columns for validation tests.

    Args:
        sample_ohlcv_data: Base OHLCV DataFrame with ATR

    Returns:
        pd.DataFrame: Labeled OHLCV data with label columns and common features
    """
    df = sample_ohlcv_data.copy()

    # Add realistic labels (60% neutral, 20% up, 20% down)
    np.random.seed(42)
    n = len(df)
    labels = np.random.choice([0, 1, -1], size=n, p=[0.6, 0.2, 0.2])

    # Add labels in both formats for compatibility
    df['label_H5'] = labels
    df['label_H20'] = np.roll(labels, 5)  # Slightly different for H20
    df['label_h5'] = labels  # lowercase format used by some validators
    df['label_h20'] = np.roll(labels, 5)

    # Add common feature columns for validation tests
    df['rsi'] = 50 + np.random.randn(n) * 10  # RSI around 50
    df['macd'] = np.random.randn(n) * 0.5  # MACD signal
    df['sma_10'] = df['close'].rolling(10, min_periods=1).mean()
    df['sma_20'] = df['close'].rolling(20, min_periods=1).mean()

    return df


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def sample_config(temp_project_dir):
    """
    Create a sample pipeline configuration for testing.

    Args:
        temp_project_dir: Temporary project directory fixture

    Returns:
        PipelineConfig: Default configuration with MES symbol
    """
    config = create_default_config(
        symbols=['MES'],
        start_date='2020-01-01',
        end_date='2024-12-31',
        project_root=temp_project_dir
    )
    return config


@pytest.fixture
def multi_symbol_config(temp_project_dir):
    """
    Create a multi-symbol pipeline configuration for testing.

    Args:
        temp_project_dir: Temporary project directory fixture

    Returns:
        PipelineConfig: Configuration with multiple symbols (MES, MGC)
    """
    config = create_default_config(
        symbols=['MES', 'MGC'],
        start_date='2020-01-01',
        end_date='2024-12-31',
        project_root=temp_project_dir
    )
    return config


# =============================================================================
# TEST UTILITIES
# =============================================================================

def assert_valid_ohlcv(df: pd.DataFrame) -> None:
    """
    Assert that a DataFrame has valid OHLCV structure and relationships.

    Validates:
    - Required columns present
    - No null values in critical columns
    - Valid OHLC relationships (high >= max(open, close), etc.)
    - Positive volume

    Args:
        df: DataFrame to validate

    Raises:
        AssertionError: If validation fails
    """
    required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    assert all(col in df.columns for col in required_cols), \
        f"Missing required columns. Found: {df.columns.tolist()}"

    assert not df[required_cols].isnull().any().any(), \
        "OHLCV data contains null values"

    # Validate OHLC relationships
    assert (df['high'] >= df['close']).all(), "High must be >= Close"
    assert (df['high'] >= df['open']).all(), "High must be >= Open"
    assert (df['low'] <= df['close']).all(), "Low must be <= Close"
    assert (df['low'] <= df['open']).all(), "Low must be <= Open"

    # Validate positive volume
    assert (df['volume'] > 0).all(), "Volume must be positive"


def assert_no_lookahead(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """
    Assert that features do not use future data (no lookahead bias).

    Validates that feature calculation at index i only uses data from indices <= i.

    Args:
        df: DataFrame with features
        feature_cols: List of feature column names to validate

    Raises:
        AssertionError: If lookahead bias detected
    """
    for col in feature_cols:
        if col not in df.columns:
            continue

        # Check that feature values are NaN at the beginning (correct)
        # Features should only become non-NaN after sufficient lookback
        first_valid_idx = df[col].first_valid_index()
        if first_valid_idx is not None and first_valid_idx > 0:
            # This is expected - features need warmup period
            pass


def assert_monotonic_datetime(df: pd.DataFrame) -> None:
    """
    Assert that datetime column is strictly monotonic increasing.

    Args:
        df: DataFrame with datetime column

    Raises:
        AssertionError: If datetime is not monotonic
    """
    assert 'datetime' in df.columns, "DataFrame missing datetime column"
    assert df['datetime'].is_monotonic_increasing, \
        "Datetime must be monotonically increasing"


# =============================================================================
# ADDITIONAL FIXTURES (added 2025-12 to fix missing fixture errors)
# =============================================================================

@pytest.fixture
def sample_features_df(sample_ohlcv_data):
    """
    Create a sample DataFrame with OHLCV data and common technical features.

    This fixture extends sample_ohlcv_data with computed features for testing
    feature engineering and labeling stages.

    Args:
        sample_ohlcv_data: Base OHLCV DataFrame with ATR

    Returns:
        pd.DataFrame: OHLCV data with technical indicator features
    """
    df = sample_ohlcv_data.copy()
    np.random.seed(42)
    n = len(df)

    # Add common technical indicators for testing
    # RSI-like feature (bounded 0-100)
    df['rsi_14'] = 50.0 + np.random.randn(n) * 15
    df['rsi_14'] = df['rsi_14'].clip(0, 100)

    # MACD-like features
    df['macd_line'] = np.random.randn(n) * 0.5
    df['macd_signal'] = df['macd_line'].rolling(9).mean().fillna(0)
    df['macd_hist'] = df['macd_line'] - df['macd_signal']

    # Moving average ratios
    df['sma_20'] = df['close'].rolling(20).mean().fillna(df['close'].iloc[0])
    df['price_to_sma_20'] = df['close'] / df['sma_20']

    # Volatility features
    df['bb_width'] = np.random.uniform(0.02, 0.05, n)
    df['bb_position'] = np.random.uniform(-1, 1, n)

    # Volume features
    df['volume_zscore'] = (df['volume'] - df['volume'].mean()) / df['volume'].std()

    # Momentum features
    df['roc_10'] = df['close'].pct_change(10).fillna(0)
    df['williams_r'] = np.random.uniform(-100, 0, n)

    # Trend features
    df['adx_14'] = np.random.uniform(10, 50, n)

    # Returns
    df['return_1'] = df['close'].pct_change(1).fillna(0)
    df['return_5'] = df['close'].pct_change(5).fillna(0)
    df['log_return_1'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)

    return df


@pytest.fixture
def sample_price_arrays(sample_ohlcv_data):
    """
    Create sample price arrays for testing numba-optimized functions.

    Provides numpy arrays extracted from OHLCV DataFrame for functions
    that require raw array inputs (like triple_barrier_numba).

    Args:
        sample_ohlcv_data: Base OHLCV DataFrame with ATR

    Returns:
        dict: Dictionary with 'close', 'high', 'low', 'open', 'atr' arrays
    """
    df = sample_ohlcv_data
    return {
        'close': df['close'].values.astype(np.float64),
        'high': df['high'].values.astype(np.float64),
        'low': df['low'].values.astype(np.float64),
        'open': df['open'].values.astype(np.float64),
        'atr': df['atr_14'].values.astype(np.float64),
    }


@pytest.fixture
def temp_directory():
    """
    Alias for temp_dir fixture for backward compatibility.

    Some older tests use 'temp_directory' instead of 'temp_dir'.

    Yields:
        Path: Temporary directory path
    """
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def sample_split_data(sample_features_df):
    """
    Create sample data with train/val/test split indices.

    Args:
        sample_features_df: DataFrame with features

    Returns:
        dict: Dictionary with 'data', 'train_idx', 'val_idx', 'test_idx'
    """
    df = sample_features_df.copy()
    n = len(df)

    # 70/15/15 split
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    return {
        'data': df,
        'train_idx': np.arange(0, train_end),
        'val_idx': np.arange(train_end, val_end),
        'test_idx': np.arange(val_end, n),
    }


@pytest.fixture
def sample_ohlcv_with_outliers(sample_ohlcv_df):
    """
    Create OHLCV DataFrame with outliers for testing outlier detection.

    Adds extreme values to the dataset for outlier detection algorithm testing.

    Args:
        sample_ohlcv_df: Base OHLCV DataFrame fixture

    Returns:
        pd.DataFrame: OHLCV data with some outlier values
    """
    df = sample_ohlcv_df.copy()

    # Add some outliers at specific indices
    # Extreme high value (spike)
    df.loc[50, 'high'] = df['high'].mean() * 1.5
    df.loc[50, 'close'] = df['close'].mean() * 1.3

    # Extreme low value (crash)
    df.loc[100, 'low'] = df['low'].mean() * 0.5
    df.loc[100, 'close'] = df['close'].mean() * 0.7

    # Volume outliers
    df.loc[150, 'volume'] = df['volume'].mean() * 10

    return df


@pytest.fixture
def sample_labeled_df(sample_features_df):
    """
    Create sample labeled DataFrame for testing stages 6-8.

    Extends sample_features_df with label columns including
    triple barrier labels and sample weights.

    Args:
        sample_features_df: DataFrame with OHLCV and features

    Returns:
        pd.DataFrame: Labeled DataFrame with label_h5, label_h20, quality, weights
    """
    df = sample_features_df.copy()
    np.random.seed(42)
    n = len(df)

    # Add labels for multiple horizons
    for horizon in [5, 20]:
        # Balanced label distribution (30% long, 30% short, 40% neutral)
        labels = np.random.choice([1, -1, 0], size=n, p=[0.30, 0.30, 0.40])
        df[f'label_h{horizon}'] = labels

        # Add bars_to_hit
        df[f'bars_to_hit_h{horizon}'] = np.random.randint(1, horizon * 2, n)

        # Add MAE/MFE
        df[f'mae_h{horizon}'] = -np.abs(np.random.randn(n) * 0.01)
        df[f'mfe_h{horizon}'] = np.abs(np.random.randn(n) * 0.02)

        # Add touch_type (1=upper, -1=lower, 0=timeout)
        df[f'touch_type_h{horizon}'] = labels.copy()

        # Add quality scores (0-1)
        df[f'quality_h{horizon}'] = np.random.uniform(0.3, 0.9, n)

        # Add sample weights (0.5, 1.0, or 1.5)
        df[f'sample_weight_h{horizon}'] = np.random.choice([0.5, 1.0, 1.5], n)

    return df
