"""
Wavelet decomposition features for multi-scale signal analysis.

Provides DWT features that decompose price/volume signals into frequency components:
- Approximation coefficients (low-frequency trend)
- Detail coefficients at multiple levels (high-frequency noise/patterns)
- Energy features at each decomposition level
- Wavelet-based volatility and trend strength

Wavelets capture non-stationary patterns that pure MTF resampling misses.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    logger.warning("PyWavelets not installed. Wavelet features will be skipped.")

# Supported wavelet families for financial time series
SUPPORTED_WAVELETS = {
    'db4': 'Daubechies 4 - good general-purpose choice',
    'db8': 'Daubechies 8 - smoother approximation',
    'sym5': 'Symlet 5 - nearly symmetric, less phase distortion',
    'coif3': 'Coiflet 3 - symmetric with vanishing moments',
    'haar': 'Haar - simplest wavelet, good for abrupt changes',
}

DEFAULT_WAVELET = 'db4'
DEFAULT_LEVEL = 3
DEFAULT_WINDOW = 64


def _compute_dwt_rolling(
    signal: np.ndarray, wavelet: str, level: int, window_size: int
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Compute DWT on rolling windows to ensure causality (no lookahead)."""
    n = len(signal)
    approx = np.full(n, np.nan)
    details = [np.full(n, np.nan) for _ in range(level)]

    min_window = 2 ** level
    if window_size < min_window:
        window_size = min_window

    for i in range(window_size - 1, n):
        window_data = signal[i - window_size + 1:i + 1]
        if np.any(np.isnan(window_data)):
            continue
        try:
            coeffs = pywt.wavedec(window_data, wavelet, level=level)
            approx[i] = coeffs[0][-1] if len(coeffs[0]) > 0 else np.nan
            for lev in range(level):
                detail_idx = level - lev
                if len(coeffs[detail_idx]) > 0:
                    details[lev][i] = coeffs[detail_idx][-1]
        except Exception:
            continue
    return approx, details


def _compute_energy_rolling(
    signal: np.ndarray, wavelet: str, level: int, window_size: int
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Compute wavelet energy at each decomposition level."""
    n = len(signal)
    approx_energy = np.full(n, np.nan)
    detail_energies = [np.full(n, np.nan) for _ in range(level)]

    window_size = max(window_size, 2 ** level)

    for i in range(window_size - 1, n):
        window_data = signal[i - window_size + 1:i + 1]
        if np.any(np.isnan(window_data)):
            continue
        try:
            coeffs = pywt.wavedec(window_data, wavelet, level=level)
            approx_energy[i] = np.sum(coeffs[0] ** 2)
            for lev in range(level):
                detail_energies[lev][i] = np.sum(coeffs[level - lev] ** 2)
        except Exception:
            continue
    return approx_energy, detail_energies


def _compute_energy_ratio(
    approx_energy: np.ndarray, detail_energies: List[np.ndarray]
) -> np.ndarray:
    """Compute ratio of approximation energy to total energy."""
    total_energy = approx_energy.copy()
    for de in detail_energies:
        total_energy = total_energy + de
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(total_energy > 0, approx_energy / total_energy, np.nan)


def _normalize_coefficients(coeffs: np.ndarray) -> np.ndarray:
    """Normalize wavelet coefficients to z-scores using expanding window."""
    result = np.full_like(coeffs, np.nan)
    for i in range(1, len(coeffs)):
        valid_data = coeffs[:i+1]
        valid_data = valid_data[~np.isnan(valid_data)]
        if len(valid_data) < 20:
            continue
        mean, std = np.mean(valid_data), np.std(valid_data)
        result[i] = (coeffs[i] - mean) / std if std > 1e-10 else 0.0
    return result


def _get_freq_label(lev: int) -> str:
    """Get frequency label for detail level."""
    return 'high' if lev == 0 else 'mid' if lev == 1 else 'low'


def add_wavelet_coefficients(
    df: pd.DataFrame,
    feature_metadata: Dict[str, str],
    price_col: str = 'close',
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
    window: int = DEFAULT_WINDOW,
    feature_prefix: str = 'wavelet',
    normalize: bool = True
) -> pd.DataFrame:
    """
    Add wavelet decomposition coefficient features.

    Creates approximation (low-freq trend) and detail (high-freq noise)
    coefficients at multiple scales.
    """
    if not PYWT_AVAILABLE:
        return df

    if wavelet not in pywt.wavelist():
        wavelet = DEFAULT_WAVELET

    logger.info(f"Adding wavelet coefficients ({wavelet}, level={level})...")
    signal = df[price_col].values
    approx, details = _compute_dwt_rolling(signal, wavelet, level, window)

    # ANTI-LOOKAHEAD: shift(1) ensures features at bar[t] use data up to bar[t-1]
    approx_col = f'{feature_prefix}_{price_col}_approx'
    if normalize:
        df[approx_col] = pd.Series(_normalize_coefficients(approx)).shift(1).values
        feature_metadata[approx_col] = f"Wavelet approx {wavelet} L{level} normalized (lagged)"
    else:
        df[approx_col] = pd.Series(approx).shift(1).values
        feature_metadata[approx_col] = f"Wavelet approx {wavelet} L{level} (lagged)"

    for lev in range(level):
        detail_col = f'{feature_prefix}_{price_col}_d{lev + 1}'
        freq = _get_freq_label(lev)
        if normalize:
            df[detail_col] = pd.Series(_normalize_coefficients(details[lev])).shift(1).values
            feature_metadata[detail_col] = f"Wavelet detail {wavelet} L{lev+1} norm ({freq} freq, lagged)"
        else:
            df[detail_col] = pd.Series(details[lev]).shift(1).values
            feature_metadata[detail_col] = f"Wavelet detail {wavelet} L{lev+1} ({freq} freq, lagged)"
    return df


def add_wavelet_energy(
    df: pd.DataFrame,
    feature_metadata: Dict[str, str],
    price_col: str = 'close',
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
    window: int = DEFAULT_WINDOW,
    feature_prefix: str = 'wavelet'
) -> pd.DataFrame:
    """
    Add wavelet energy features at each decomposition level.

    Energy = sum of squared coefficients, indicating signal power in frequency bands.
    """
    if not PYWT_AVAILABLE:
        return df

    logger.info(f"Adding wavelet energy features ({wavelet}, level={level})...")
    signal = df[price_col].values
    approx_energy, detail_energies = _compute_energy_rolling(signal, wavelet, level, window)

    # ANTI-LOOKAHEAD: shift(1)
    approx_energy_col = f'{feature_prefix}_{price_col}_energy_approx'
    with np.errstate(divide='ignore', invalid='ignore'):
        df[approx_energy_col] = pd.Series(np.log1p(approx_energy)).shift(1).values
    feature_metadata[approx_energy_col] = f"Wavelet approx energy log1p {wavelet} L{level} (lagged)"

    for lev in range(level):
        energy_col = f'{feature_prefix}_{price_col}_energy_d{lev + 1}'
        freq = _get_freq_label(lev)
        with np.errstate(divide='ignore', invalid='ignore'):
            df[energy_col] = pd.Series(np.log1p(detail_energies[lev])).shift(1).values
        feature_metadata[energy_col] = f"Wavelet energy {wavelet} L{lev+1} ({freq} freq, lagged)"

    ratio_col = f'{feature_prefix}_{price_col}_energy_ratio'
    energy_ratio = _compute_energy_ratio(approx_energy, detail_energies)
    df[ratio_col] = pd.Series(energy_ratio).shift(1).values
    feature_metadata[ratio_col] = f"Wavelet energy ratio {wavelet} L{level} (lagged)"
    return df


def add_wavelet_volatility(
    df: pd.DataFrame,
    feature_metadata: Dict[str, str],
    price_col: str = 'close',
    wavelet: str = DEFAULT_WAVELET,
    window: int = DEFAULT_WINDOW,
    feature_prefix: str = 'wavelet'
) -> pd.DataFrame:
    """
    Add wavelet-based volatility estimate using MAD of detail coefficients.

    More robust to trends than standard deviation.
    """
    if not PYWT_AVAILABLE:
        return df

    logger.info(f"Adding wavelet volatility ({wavelet}, window={window})...")
    signal = df[price_col].values
    n = len(signal)
    wavelet_vol = np.full(n, np.nan)
    actual_window = max(window, 8)

    for i in range(actual_window - 1, n):
        window_data = signal[i - actual_window + 1:i + 1]
        if np.any(np.isnan(window_data)):
            continue
        try:
            _, detail = pywt.dwt(window_data, wavelet)
            mad = np.median(np.abs(detail - np.median(detail)))
            wavelet_vol[i] = mad / 0.6745  # MAD to sigma for Gaussian
        except Exception:
            continue

    # ANTI-LOOKAHEAD: shift(1)
    vol_col = f'{feature_prefix}_{price_col}_volatility'
    df[vol_col] = pd.Series(wavelet_vol).shift(1).values
    feature_metadata[vol_col] = f"Wavelet volatility MAD {wavelet} (lagged)"
    return df


def add_wavelet_trend_strength(
    df: pd.DataFrame,
    feature_metadata: Dict[str, str],
    price_col: str = 'close',
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
    window: int = DEFAULT_WINDOW,
    feature_prefix: str = 'wavelet'
) -> pd.DataFrame:
    """
    Add wavelet-based trend strength using slope of approximation coefficients.
    """
    if not PYWT_AVAILABLE:
        return df

    logger.info(f"Adding wavelet trend strength ({wavelet}, level={level})...")
    signal = df[price_col].values
    n = len(signal)
    trend_strength = np.full(n, np.nan)
    trend_direction = np.full(n, np.nan)
    actual_window = max(window, 2 ** level)

    for i in range(actual_window - 1, n):
        window_data = signal[i - actual_window + 1:i + 1]
        if np.any(np.isnan(window_data)):
            continue
        try:
            coeffs = pywt.wavedec(window_data, wavelet, level=level)
            approx = coeffs[0]
            if len(approx) >= 2:
                slope = (approx[-1] - approx[0]) / len(approx)
                std = np.std(window_data)
                if std > 1e-10:
                    trend_strength[i] = np.abs(slope) / std
                    trend_direction[i] = np.sign(slope)
        except Exception:
            continue

    # ANTI-LOOKAHEAD: shift(1)
    strength_col = f'{feature_prefix}_{price_col}_trend_strength'
    direction_col = f'{feature_prefix}_{price_col}_trend_direction'
    df[strength_col] = pd.Series(trend_strength).shift(1).values
    df[direction_col] = pd.Series(trend_direction).shift(1).values
    feature_metadata[strength_col] = f"Wavelet trend strength {wavelet} L{level} (lagged)"
    feature_metadata[direction_col] = f"Wavelet trend direction {wavelet} L{level} (lagged)"
    return df


def add_wavelet_features(
    df: pd.DataFrame,
    feature_metadata: Dict[str, str],
    price_col: str = 'close',
    volume_col: str = 'volume',
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
    window: int = DEFAULT_WINDOW,
    feature_prefix: str = 'wavelet',
    include_volume: bool = True,
    include_energy: bool = True,
    include_volatility: bool = True,
    include_trend: bool = True
) -> pd.DataFrame:
    """
    Add all wavelet decomposition features for multi-scale signal analysis.

    Features include:
    1. Coefficient features (approx + details at each level)
    2. Energy features (power at each frequency band)
    3. Volatility estimate (robust to trends)
    4. Trend strength and direction

    For level=3 with all options enabled, adds ~24 features:
    - 4 price coefficients (1 approx + 3 details)
    - 4 volume coefficients (if include_volume=True)
    - 5 price energy features (4 levels + ratio)
    - 5 volume energy features (if include_volume and include_energy)
    - 1 wavelet volatility
    - 2 trend features (strength + direction)
    """
    if not PYWT_AVAILABLE:
        logger.warning("PyWavelets not installed. pip install PyWavelets>=1.4.0")
        return df

    logger.info(f"Adding wavelet features (wavelet={wavelet}, level={level})")
    initial_cols = len(df.columns)

    # Price coefficients
    df = add_wavelet_coefficients(
        df, feature_metadata, price_col=price_col, wavelet=wavelet,
        level=level, window=window, feature_prefix=feature_prefix, normalize=True
    )

    # Volume coefficients
    if include_volume and volume_col in df.columns and df[volume_col].sum() > 0:
        df = add_wavelet_coefficients(
            df, feature_metadata, price_col=volume_col, wavelet=wavelet,
            level=level, window=window, feature_prefix=feature_prefix, normalize=True
        )

    # Energy features
    if include_energy:
        df = add_wavelet_energy(
            df, feature_metadata, price_col=price_col, wavelet=wavelet,
            level=level, window=window, feature_prefix=feature_prefix
        )
        if include_volume and volume_col in df.columns and df[volume_col].sum() > 0:
            df = add_wavelet_energy(
                df, feature_metadata, price_col=volume_col, wavelet=wavelet,
                level=level, window=window, feature_prefix=feature_prefix
            )

    # Volatility
    if include_volatility:
        df = add_wavelet_volatility(
            df, feature_metadata, price_col=price_col, wavelet=wavelet,
            window=window, feature_prefix=feature_prefix
        )

    # Trend strength
    if include_trend:
        df = add_wavelet_trend_strength(
            df, feature_metadata, price_col=price_col, wavelet=wavelet,
            level=level, window=window, feature_prefix=feature_prefix
        )

    logger.info(f"Added {len(df.columns) - initial_cols} wavelet features")
    return df


__all__ = [
    'add_wavelet_features',
    'add_wavelet_coefficients',
    'add_wavelet_energy',
    'add_wavelet_volatility',
    'add_wavelet_trend_strength',
    'SUPPORTED_WAVELETS',
    'DEFAULT_WAVELET',
    'DEFAULT_LEVEL',
    'DEFAULT_WINDOW',
    'PYWT_AVAILABLE',
]
