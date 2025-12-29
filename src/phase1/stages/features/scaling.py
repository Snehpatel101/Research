"""
Timeframe-aware indicator period scaling.

Scales indicator periods to maintain consistent lookback duration
when changing timeframes. This ensures that indicators have the same
temporal meaning regardless of the bar frequency.

Example:
    RSI-14 on 5-min bars = 70 min lookback
    When using 15-min bars, we scale to RSI-5 to maintain ~70 min lookback
"""
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Base timeframe in minutes (all periods are defined relative to this)
BASE_TIMEFRAME_MINUTES = 5

# Timeframe string to minutes mapping
TIMEFRAME_MINUTES: dict[str, int] = {
    '1min': 1,
    '5min': 5,
    '10min': 10,
    '15min': 15,
    '20min': 20,
    '30min': 30,
    '45min': 45,
    '60min': 60,
    '1h': 60,
    '2h': 120,
    '4h': 240,
    '1d': 1440,
}


def get_timeframe_minutes(timeframe: str) -> int:
    """
    Get the number of minutes for a timeframe string.

    Parameters
    ----------
    timeframe : str
        Timeframe string (e.g., '5min', '1h', '15min')

    Returns
    -------
    int
        Number of minutes

    Raises
    ------
    ValueError
        If timeframe is not recognized
    """
    minutes = TIMEFRAME_MINUTES.get(timeframe)
    if minutes is None:
        valid_tfs = ', '.join(sorted(TIMEFRAME_MINUTES.keys()))
        raise ValueError(
            f"Unknown timeframe: '{timeframe}'. "
            f"Valid timeframes: {valid_tfs}"
        )
    return minutes


def scale_period(period: int, source_tf: str, target_tf: str) -> int:
    """
    Scale indicator period to maintain consistent lookback duration.

    Converts a period defined for a source timeframe to an equivalent
    period for a target timeframe, preserving the lookback duration in
    real time (minutes).

    Parameters
    ----------
    period : int
        Original indicator period (e.g., 14 for RSI-14)
    source_tf : str
        Source timeframe (e.g., '5min')
    target_tf : str
        Target timeframe (e.g., '15min')

    Returns
    -------
    int
        Scaled period (minimum 2 to ensure valid calculation)

    Examples
    --------
    >>> scale_period(14, '5min', '15min')  # 70min lookback
    5  # 5 bars * 15min = 75min (closest to 70min)

    >>> scale_period(20, '5min', '1min')
    100  # 100 bars * 1min = 100min (same as 20 * 5min)

    >>> scale_period(14, '5min', '5min')  # same timeframe
    14  # unchanged
    """
    source_minutes = get_timeframe_minutes(source_tf)
    target_minutes = get_timeframe_minutes(target_tf)

    # Calculate lookback duration in minutes
    lookback_minutes = period * source_minutes

    # Scale to target timeframe (round to nearest integer)
    scaled = round(lookback_minutes / target_minutes)

    # Enforce minimum period of 2 (required for valid indicator calculations)
    result = max(2, scaled)

    if result != period:
        logger.debug(
            f"Scaled period {period} from {source_tf} to {target_tf}: "
            f"{period} -> {result} ({lookback_minutes}min lookback)"
        )

    return result


def get_scaled_periods(
    periods: list[int],
    target_tf: str,
    base_tf: str = '5min'
) -> list[int]:
    """
    Scale a list of periods to target timeframe.

    Parameters
    ----------
    periods : List[int]
        List of periods to scale
    target_tf : str
        Target timeframe
    base_tf : str, default '5min'
        Base timeframe periods are defined for

    Returns
    -------
    List[int]
        List of scaled periods (duplicates may occur)
    """
    return [scale_period(p, base_tf, target_tf) for p in periods]


def get_unique_scaled_periods(
    periods: list[int],
    target_tf: str,
    base_tf: str = '5min'
) -> list[int]:
    """
    Scale periods and remove duplicates, preserving order.

    Parameters
    ----------
    periods : List[int]
        List of periods to scale
    target_tf : str
        Target timeframe
    base_tf : str, default '5min'
        Base timeframe periods are defined for

    Returns
    -------
    List[int]
        List of unique scaled periods in original order
    """
    scaled = get_scaled_periods(periods, target_tf, base_tf)
    # Preserve order while removing duplicates
    seen = set()
    unique = []
    for p in scaled:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


# Default base periods (defined for 5-min bars)
# These represent the "standard" indicator periods used in technical analysis
DEFAULT_BASE_PERIODS: dict[str, list[int]] = {
    # Moving averages
    'sma': [10, 20, 50, 100, 200],
    'ema': [9, 12, 21, 26, 50],

    # Momentum
    'rsi': [14],
    'stochastic_k': [14],
    'stochastic_d': [3],
    'macd_fast': [12],
    'macd_slow': [26],
    'macd_signal': [9],
    'williams_r': [14],
    'roc': [5, 10, 20],
    'cci': [20],
    'mfi': [14],

    # Volatility
    'atr': [7, 14, 21],
    'bollinger': [20],
    'keltner': [20],
    'hvol': [10, 20, 60],
    'parkinson': [20],
    'garman_klass': [20],

    # Trend
    'adx': [14],
    'supertrend_period': [10],

    # Volume
    'volume_sma': [20],
    'obv_sma': [20],
}


def create_period_config(
    target_tf: str,
    base_tf: str = '5min',
    base_periods: dict[str, list[int]] | None = None
) -> dict[str, list[int]]:
    """
    Create scaled period configuration for all indicators.

    Scales all indicator periods from the base timeframe to the target
    timeframe to maintain consistent lookback durations.

    Parameters
    ----------
    target_tf : str
        Target timeframe (e.g., '15min', '1h')
    base_tf : str, default '5min'
        Base timeframe periods are defined for
    base_periods : Dict[str, List[int]], optional
        Custom base periods. If None, uses DEFAULT_BASE_PERIODS.

    Returns
    -------
    Dict[str, List[int]]
        Dictionary mapping indicator names to scaled periods

    Examples
    --------
    >>> config = create_period_config('5min')  # no scaling
    >>> config['rsi']
    [14]

    >>> config = create_period_config('15min')  # 3x compression
    >>> config['rsi']
    [5]

    >>> config = create_period_config('1min')  # 5x expansion
    >>> config['rsi']
    [70]
    """
    if base_periods is None:
        base_periods = DEFAULT_BASE_PERIODS

    # If same timeframe, return base periods unchanged
    if target_tf == base_tf:
        logger.info(f"Using base periods for {target_tf} (no scaling needed)")
        return {k: list(v) for k, v in base_periods.items()}

    scaled = {}
    for indicator, periods in base_periods.items():
        scaled[indicator] = get_scaled_periods(periods, target_tf, base_tf)

    source_mins = get_timeframe_minutes(base_tf)
    target_mins = get_timeframe_minutes(target_tf)
    scale_factor = source_mins / target_mins

    logger.info(
        f"Scaled indicator periods from {base_tf} to {target_tf} "
        f"(scale factor: {scale_factor:.2f}x)"
    )

    return scaled


class PeriodScaler:
    """
    Utility class for managing timeframe-aware indicator periods.

    This class provides a convenient interface for scaling indicator
    periods based on the target timeframe. It caches the scaled
    configuration for efficient access.

    Parameters
    ----------
    target_tf : str
        Target timeframe for indicators
    base_tf : str, default '5min'
        Base timeframe periods are defined for

    Examples
    --------
    >>> scaler = PeriodScaler('15min')
    >>> scaler.get_period('rsi')
    5
    >>> scaler.get_periods('sma')
    [3, 7, 17, 33, 67]
    """

    def __init__(self, target_tf: str, base_tf: str = '5min'):
        self.target_tf = target_tf
        self.base_tf = base_tf
        self._config = create_period_config(target_tf, base_tf)

    def get_periods(self, indicator: str) -> list[int]:
        """Get scaled periods for an indicator."""
        if indicator not in self._config:
            raise KeyError(
                f"Unknown indicator: '{indicator}'. "
                f"Available: {list(self._config.keys())}"
            )
        return self._config[indicator]

    def get_period(self, indicator: str, index: int = 0) -> int:
        """Get a specific scaled period for an indicator."""
        periods = self.get_periods(indicator)
        if index >= len(periods):
            raise IndexError(
                f"Period index {index} out of range for '{indicator}' "
                f"(has {len(periods)} periods)"
            )
        return periods[index]

    def scale_custom_period(self, period: int) -> int:
        """Scale a custom period from base_tf to target_tf."""
        return scale_period(period, self.base_tf, self.target_tf)

    @property
    def config(self) -> dict[str, list[int]]:
        """Return the full scaled configuration."""
        return self._config

    def __repr__(self) -> str:
        return f"PeriodScaler(target_tf='{self.target_tf}', base_tf='{self.base_tf}')"


__all__ = [
    'scale_period',
    'get_scaled_periods',
    'get_unique_scaled_periods',
    'create_period_config',
    'get_timeframe_minutes',
    'PeriodScaler',
    'DEFAULT_BASE_PERIODS',
    'TIMEFRAME_MINUTES',
    'BASE_TIMEFRAME_MINUTES',
]
