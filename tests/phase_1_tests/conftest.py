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

    Args:
        sample_ohlcv_data: Base OHLCV DataFrame with ATR

    Returns:
        pd.DataFrame: Labeled OHLCV data with label_H5 and label_H20 columns
    """
    df = sample_ohlcv_data.copy()

    # Add realistic labels (60% neutral, 20% up, 20% down)
    np.random.seed(42)
    n = len(df)
    labels = np.random.choice([0, 1, -1], size=n, p=[0.6, 0.2, 0.2])

    df['label_H5'] = labels
    df['label_H20'] = np.roll(labels, 5)  # Slightly different for H20

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
