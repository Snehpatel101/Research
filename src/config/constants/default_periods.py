"""
Default period constants for technical indicators.

These constants define sensible defaults for technical analysis calculations.
Import from here to avoid magic numbers throughout the codebase.

Usage:
    from src.config.constants.default_periods import SMA_PERIOD, RSI_PERIOD

    sma = df['close'].rolling(SMA_PERIOD).mean()
    rsi = compute_rsi(df['close'], period=RSI_PERIOD)
"""

# =============================================================================
# MOVING AVERAGE PERIODS
# =============================================================================
SMA_PERIOD = 20  # Simple Moving Average
EMA_PERIOD = 12  # Exponential Moving Average (fast)
EMA_SLOW_PERIOD = 26  # Exponential Moving Average (slow, for MACD)

# =============================================================================
# OSCILLATOR PERIODS
# =============================================================================
RSI_PERIOD = 14  # Relative Strength Index
STOCH_PERIOD = 14  # Stochastic Oscillator
STOCH_SMOOTH = 3  # Stochastic smoothing (K and D)
MACD_SIGNAL_PERIOD = 9  # MACD signal line

# =============================================================================
# VOLATILITY PERIODS
# =============================================================================
ATR_PERIOD = 14  # Average True Range
BB_PERIOD = 20  # Bollinger Bands
BB_STD = 2.0  # Bollinger Bands standard deviation multiplier

# =============================================================================
# VOLUME PERIODS
# =============================================================================
VWAP_PERIOD = 20  # Volume Weighted Average Price
OBV_PERIOD = 20  # On-Balance Volume smoothing

# =============================================================================
# LOOKBACK DEFAULTS
# =============================================================================
DEFAULT_LOOKBACK = 100  # Default lookback window
MIN_LOOKBACK = 10  # Minimum lookback (prevents too small windows)
MAX_LOOKBACK = 500  # Maximum lookback (prevents memory issues)

# =============================================================================
# SEQUENCE DEFAULTS (for neural models)
# =============================================================================
DEFAULT_SEQUENCE_LENGTH = 60  # Default sequence length for RNN/Transformer
MIN_SEQUENCE_LENGTH = 10  # Minimum sequence length
MAX_SEQUENCE_LENGTH = 200  # Maximum sequence length

__all__ = [
    # Moving averages
    "SMA_PERIOD",
    "EMA_PERIOD",
    "EMA_SLOW_PERIOD",
    # Oscillators
    "RSI_PERIOD",
    "STOCH_PERIOD",
    "STOCH_SMOOTH",
    "MACD_SIGNAL_PERIOD",
    # Volatility
    "ATR_PERIOD",
    "BB_PERIOD",
    "BB_STD",
    # Volume
    "VWAP_PERIOD",
    "OBV_PERIOD",
    # Lookback
    "DEFAULT_LOOKBACK",
    "MIN_LOOKBACK",
    "MAX_LOOKBACK",
    # Sequences
    "DEFAULT_SEQUENCE_LENGTH",
    "MIN_SEQUENCE_LENGTH",
    "MAX_SEQUENCE_LENGTH",
]
