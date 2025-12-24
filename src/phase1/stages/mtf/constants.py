"""
Constants for Multi-Timeframe (MTF) Feature Integration.
"""

# Supported MTF timeframes with their minute equivalents
MTF_TIMEFRAMES = {
    '5min': 5,
    '15min': 15,
    '30min': 30,
    '60min': 60,
    '1h': 60,  # Alias for 60min
}

# Required OHLCV columns
REQUIRED_OHLCV_COLS = ['open', 'high', 'low', 'close', 'volume']

# Default MTF configuration
DEFAULT_MTF_TIMEFRAMES = ['15min', '60min']
DEFAULT_BASE_TIMEFRAME = '5min'
MIN_BASE_BARS = 100
MIN_MTF_BARS = 30
