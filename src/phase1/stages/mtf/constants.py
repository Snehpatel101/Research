"""
Constants for Multi-Timeframe (MTF) Feature Integration.

Supports timeframes from 5min (base) up to daily.
All timeframes must be integer multiples of the base timeframe.
"""

from enum import Enum


class MTFMode(str, Enum):
    """Mode for MTF feature generation."""
    BARS = 'bars'           # Only generate MTF OHLCV bars
    INDICATORS = 'indicators'  # Only generate MTF indicators
    BOTH = 'both'           # Generate both bars and indicators


# Supported MTF timeframes with their minute equivalents
# All values must be integer multiples of the base timeframe (1min)
MTF_TIMEFRAMES = {
    # Base timeframe
    '1min': 1,
    '5min': 5,
    # Short-term MTF (9-timeframe ladder)
    '10min': 10,
    '15min': 15,
    '20min': 20,    # Added for 9-timeframe ladder
    '25min': 25,    # Added for 9-timeframe ladder
    '30min': 30,
    '45min': 45,
    # Hourly
    '60min': 60,
    '1h': 60,       # Alias for 60min
    # Multi-hour
    '4h': 240,      # 4 hours = 240 minutes
    '240min': 240,  # Alias for 4h
    # Daily
    'daily': 1440,  # 24 hours = 1440 minutes
    '1d': 1440,     # Alias for daily
    'D': 1440,      # Pandas convention alias
}

# Required OHLCV columns
REQUIRED_OHLCV_COLS = ['open', 'high', 'low', 'close', 'volume']

# Default MTF configuration
# Includes all commonly useful timeframes for multi-timeframe analysis
DEFAULT_MTF_TIMEFRAMES = ['15min', '30min', '1h', '4h', 'daily']

# Default mode is to generate both bars and indicators
DEFAULT_MTF_MODE = MTFMode.BOTH

# Base timeframe for resampling
DEFAULT_BASE_TIMEFRAME = '1min'

# Minimum data requirements
MIN_BASE_BARS = 100     # Minimum base TF bars required
MIN_MTF_BARS = 30       # Minimum MTF bars required per timeframe

# Pandas frequency aliases for resampling
# Maps our timeframe strings to pandas frequency strings
PANDAS_FREQ_MAP = {
    '1min': '1min',
    '5min': '5min',
    '10min': '10min',
    '15min': '15min',
    '30min': '30min',
    '45min': '45min',
    '60min': '60min',
    '1h': '1h',
    '4h': '4h',
    '240min': '4h',
    'daily': 'D',
    '1d': 'D',
    'D': 'D',
}
