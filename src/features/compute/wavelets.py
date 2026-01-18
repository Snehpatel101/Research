"""
Wavelet feature computation - DWT coefficients, energy, trend analysis.

PHASE_1 Unified Features: 15 WAVELETS features.

Uses PyWavelets (pywt) if available, otherwise returns NaN stubs.
"""

from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

# Try to import pywt, set flag if not available
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _wavelet_decompose(
    series: pd.Series,
    wavelet: str = "db4",
    level: int = 3,
    window: int = 64,
) -> tuple[Optional[np.ndarray], Optional[list[np.ndarray]]]:
    """
    Perform wavelet decomposition on a rolling basis.

    Returns approximation coefficients and list of detail coefficients.
    """
    if not PYWT_AVAILABLE:
        return None, None

    # Ensure power of 2 for proper decomposition
    if len(series) < window:
        return None, None

    data = series.values[-window:]

    # Pad to power of 2 if needed
    n = len(data)
    pad_len = 2 ** int(np.ceil(np.log2(n))) - n
    if pad_len > 0:
        data = np.pad(data, (pad_len, 0), mode="edge")

    try:
        coeffs = pywt.wavedec(data, wavelet, level=level)
        # coeffs[0] is approximation, coeffs[1:] are details
        return coeffs[0], coeffs[1:]
    except Exception:
        return None, None


def _rolling_wavelet_feature(
    series: pd.Series,
    extract_fn: Callable,
    window: int = 64,
) -> pd.Series:
    """
    Apply wavelet feature extraction on a rolling basis.

    Args:
        series: Input time series
        extract_fn: Function to extract feature from wavelet coefficients
        window: Window size for decomposition
    """
    if not PYWT_AVAILABLE:
        return pd.Series(np.nan, index=series.index)

    result = pd.Series(np.nan, index=series.index)

    for i in range(window, len(series) + 1):
        window_data = series.iloc[i - window:i]
        try:
            value = extract_fn(window_data)
            result.iloc[i - 1] = value
        except Exception:
            result.iloc[i - 1] = np.nan

    return result


def _get_wavelet_coeffs(series: pd.Series, wavelet: str = "db4", level: int = 3):
    """Get wavelet coefficients for a window of data."""
    if not PYWT_AVAILABLE or len(series) < 16:
        return None

    data = series.values
    try:
        coeffs = pywt.wavedec(data, wavelet, level=level)
        return coeffs
    except Exception:
        return None


def _wavelet_energy(coeffs: list) -> list:
    """Calculate energy of each wavelet coefficient level."""
    if coeffs is None:
        return []
    return [np.sum(c ** 2) for c in coeffs]


# =============================================================================
# CLOSE PRICE WAVELET FEATURES
# =============================================================================


def compute_wavelet_close_approx(df: pd.DataFrame) -> pd.Series:
    """
    Wavelet approximation coefficient (trend component) of close price.

    The approximation captures the low-frequency trend.
    """
    def extract_approx(window_data):
        coeffs = _get_wavelet_coeffs(window_data)
        if coeffs is None:
            return np.nan
        # Return mean of approximation coefficients
        return np.mean(coeffs[0])

    return _rolling_wavelet_feature(df["close"], extract_approx, window=64)


def compute_wavelet_close_d1(df: pd.DataFrame) -> pd.Series:
    """
    Wavelet detail level 1 (highest frequency) of close price.

    Captures short-term noise/fluctuations.
    """
    def extract_d1(window_data):
        coeffs = _get_wavelet_coeffs(window_data)
        if coeffs is None or len(coeffs) < 2:
            return np.nan
        return np.mean(np.abs(coeffs[1]))

    return _rolling_wavelet_feature(df["close"], extract_d1, window=64)


def compute_wavelet_close_d2(df: pd.DataFrame) -> pd.Series:
    """
    Wavelet detail level 2 of close price.

    Captures medium-frequency fluctuations.
    """
    def extract_d2(window_data):
        coeffs = _get_wavelet_coeffs(window_data)
        if coeffs is None or len(coeffs) < 3:
            return np.nan
        return np.mean(np.abs(coeffs[2]))

    return _rolling_wavelet_feature(df["close"], extract_d2, window=64)


def compute_wavelet_close_d3(df: pd.DataFrame) -> pd.Series:
    """
    Wavelet detail level 3 of close price.

    Captures lower-frequency fluctuations (more trend-like).
    """
    def extract_d3(window_data):
        coeffs = _get_wavelet_coeffs(window_data)
        if coeffs is None or len(coeffs) < 4:
            return np.nan
        return np.mean(np.abs(coeffs[3]))

    return _rolling_wavelet_feature(df["close"], extract_d3, window=64)


# =============================================================================
# VOLUME WAVELET FEATURES
# =============================================================================


def compute_wavelet_volume_approx(df: pd.DataFrame) -> pd.Series:
    """Wavelet approximation of volume (trend component)."""
    def extract_approx(window_data):
        coeffs = _get_wavelet_coeffs(window_data)
        if coeffs is None:
            return np.nan
        return np.mean(coeffs[0])

    return _rolling_wavelet_feature(df["volume"], extract_approx, window=64)


def compute_wavelet_volume_d1(df: pd.DataFrame) -> pd.Series:
    """Wavelet detail level 1 of volume (high-frequency component)."""
    def extract_d1(window_data):
        coeffs = _get_wavelet_coeffs(window_data)
        if coeffs is None or len(coeffs) < 2:
            return np.nan
        return np.mean(np.abs(coeffs[1]))

    return _rolling_wavelet_feature(df["volume"], extract_d1, window=64)


# =============================================================================
# WAVELET ENERGY FEATURES
# =============================================================================


def compute_wavelet_close_energy_approx(df: pd.DataFrame) -> pd.Series:
    """Energy of approximation coefficients (trend energy)."""
    def extract_energy(window_data):
        coeffs = _get_wavelet_coeffs(window_data)
        if coeffs is None:
            return np.nan
        return np.sum(coeffs[0] ** 2)

    return _rolling_wavelet_feature(df["close"], extract_energy, window=64)


def compute_wavelet_close_energy_d1(df: pd.DataFrame) -> pd.Series:
    """Energy of detail level 1 (high-frequency energy)."""
    def extract_energy(window_data):
        coeffs = _get_wavelet_coeffs(window_data)
        if coeffs is None or len(coeffs) < 2:
            return np.nan
        return np.sum(coeffs[1] ** 2)

    return _rolling_wavelet_feature(df["close"], extract_energy, window=64)


def compute_wavelet_close_energy_d2(df: pd.DataFrame) -> pd.Series:
    """Energy of detail level 2."""
    def extract_energy(window_data):
        coeffs = _get_wavelet_coeffs(window_data)
        if coeffs is None or len(coeffs) < 3:
            return np.nan
        return np.sum(coeffs[2] ** 2)

    return _rolling_wavelet_feature(df["close"], extract_energy, window=64)


def compute_wavelet_close_energy_d3(df: pd.DataFrame) -> pd.Series:
    """Energy of detail level 3."""
    def extract_energy(window_data):
        coeffs = _get_wavelet_coeffs(window_data)
        if coeffs is None or len(coeffs) < 4:
            return np.nan
        return np.sum(coeffs[3] ** 2)

    return _rolling_wavelet_feature(df["close"], extract_energy, window=64)


def compute_wavelet_close_energy_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Ratio of trend energy to detail energy.

    High ratio indicates strong trend, low ratio indicates noisy market.
    """
    def extract_ratio(window_data):
        coeffs = _get_wavelet_coeffs(window_data)
        if coeffs is None or len(coeffs) < 2:
            return np.nan

        trend_energy = np.sum(coeffs[0] ** 2)
        detail_energy = sum(np.sum(c ** 2) for c in coeffs[1:])

        if detail_energy == 0:
            return np.nan

        return trend_energy / detail_energy

    return _rolling_wavelet_feature(df["close"], extract_ratio, window=64)


# =============================================================================
# WAVELET DERIVED FEATURES
# =============================================================================


def compute_wavelet_close_volatility(df: pd.DataFrame) -> pd.Series:
    """
    Wavelet-based volatility estimate.

    Uses detail coefficients to estimate volatility.
    """
    def extract_volatility(window_data):
        coeffs = _get_wavelet_coeffs(window_data)
        if coeffs is None or len(coeffs) < 2:
            return np.nan

        # MAD estimator from finest detail coefficients
        d1 = coeffs[1]
        median_d1 = np.median(np.abs(d1))
        sigma = median_d1 / 0.6745  # Robust volatility estimate

        return sigma

    return _rolling_wavelet_feature(df["close"], extract_volatility, window=64)


def compute_wavelet_close_trend_strength(df: pd.DataFrame) -> pd.Series:
    """
    Wavelet-based trend strength indicator.

    Measures how much of the signal is captured by the trend component.
    """
    def extract_trend_strength(window_data):
        coeffs = _get_wavelet_coeffs(window_data)
        if coeffs is None or len(coeffs) < 2:
            return np.nan

        total_energy = sum(np.sum(c ** 2) for c in coeffs)
        if total_energy == 0:
            return np.nan

        trend_energy = np.sum(coeffs[0] ** 2)
        return trend_energy / total_energy

    return _rolling_wavelet_feature(df["close"], extract_trend_strength, window=64)


def compute_wavelet_close_trend_direction(df: pd.DataFrame) -> pd.Series:
    """
    Wavelet-based trend direction.

    Uses slope of approximation coefficients.
    Returns 1 for uptrend, -1 for downtrend, 0 for flat.
    """
    def extract_direction(window_data):
        coeffs = _get_wavelet_coeffs(window_data)
        if coeffs is None:
            return np.nan

        approx = coeffs[0]
        if len(approx) < 2:
            return 0.0

        # Simple slope estimation
        slope = np.polyfit(range(len(approx)), approx, 1)[0]

        if slope > 0.01:
            return 1.0
        elif slope < -0.01:
            return -1.0
        else:
            return 0.0

    return _rolling_wavelet_feature(df["close"], extract_direction, window=64)


def compute_wavelet_volume_energy_ratio(df: pd.DataFrame) -> pd.Series:
    """Volume wavelet energy ratio (trend vs detail)."""
    def extract_ratio(window_data):
        coeffs = _get_wavelet_coeffs(window_data)
        if coeffs is None or len(coeffs) < 2:
            return np.nan

        trend_energy = np.sum(coeffs[0] ** 2)
        detail_energy = sum(np.sum(c ** 2) for c in coeffs[1:])

        if detail_energy == 0:
            return np.nan

        return trend_energy / detail_energy

    return _rolling_wavelet_feature(df["volume"], extract_ratio, window=64)


# =============================================================================
# FEATURE MAP
# =============================================================================

WAVELETS_FEATURES: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    # Close price wavelets
    "wavelet_close_approx": compute_wavelet_close_approx,
    "wavelet_close_d1": compute_wavelet_close_d1,
    "wavelet_close_d2": compute_wavelet_close_d2,
    "wavelet_close_d3": compute_wavelet_close_d3,
    # Volume wavelets
    "wavelet_volume_approx": compute_wavelet_volume_approx,
    "wavelet_volume_d1": compute_wavelet_volume_d1,
    # Energy features
    "wavelet_close_energy_approx": compute_wavelet_close_energy_approx,
    "wavelet_close_energy_d1": compute_wavelet_close_energy_d1,
    "wavelet_close_energy_d2": compute_wavelet_close_energy_d2,
    "wavelet_close_energy_d3": compute_wavelet_close_energy_d3,
    "wavelet_close_energy_ratio": compute_wavelet_close_energy_ratio,
    # Derived features
    "wavelet_close_volatility": compute_wavelet_close_volatility,
    "wavelet_close_trend_strength": compute_wavelet_close_trend_strength,
    "wavelet_close_trend_direction": compute_wavelet_close_trend_direction,
    "wavelet_volume_energy_ratio": compute_wavelet_volume_energy_ratio,
}

# Feature family metadata
FEATURE_FAMILY = "wavelets"
FEATURE_COUNT = 15

__all__ = [
    "WAVELETS_FEATURES",
    "FEATURE_FAMILY",
    "FEATURE_COUNT",
    "PYWT_AVAILABLE",
    # Close wavelets
    "compute_wavelet_close_approx",
    "compute_wavelet_close_d1",
    "compute_wavelet_close_d2",
    "compute_wavelet_close_d3",
    # Volume wavelets
    "compute_wavelet_volume_approx",
    "compute_wavelet_volume_d1",
    # Energy
    "compute_wavelet_close_energy_approx",
    "compute_wavelet_close_energy_d1",
    "compute_wavelet_close_energy_d2",
    "compute_wavelet_close_energy_d3",
    "compute_wavelet_close_energy_ratio",
    # Derived
    "compute_wavelet_close_volatility",
    "compute_wavelet_close_trend_strength",
    "compute_wavelet_close_trend_direction",
    "compute_wavelet_volume_energy_ratio",
]
