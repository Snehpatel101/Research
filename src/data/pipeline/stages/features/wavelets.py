"""
Wavelet decomposition features for multi-scale signal analysis.

Provides DWT features that decompose price/volume signals into frequency components:
- Approximation coefficients (low-frequency trend)
- Detail coefficients at multiple levels (high-frequency noise/patterns)
- Energy features at each decomposition level
- Wavelet-based volatility and trend strength

Wavelets capture non-stationary patterns that pure MTF resampling misses.
"""

import logging

import numpy as np
import pandas as pd
from numba import njit

from ._helpers import np_shift1 as _np_shift1

logger = logging.getLogger(__name__)


try:
    import pywt

    PYWT_AVAILABLE = True
except ImportError:
    pywt = None  # bound so guarded uses are well-defined
    PYWT_AVAILABLE = False
    logger.warning("PyWavelets not installed. Wavelet features will be skipped.")

# Supported wavelet families for financial time series
SUPPORTED_WAVELETS = {
    "db4": "Daubechies 4 - good general-purpose choice",
    "db8": "Daubechies 8 - smoother approximation",
    "sym5": "Symlet 5 - nearly symmetric, less phase distortion",
    "coif3": "Coiflet 3 - symmetric with vanishing moments",
    "haar": "Haar - simplest wavelet, good for abrupt changes",
}

DEFAULT_WAVELET = "db4"
DEFAULT_LEVEL = 3
DEFAULT_WINDOW = 64


def _compute_dwt_all(signal: np.ndarray, wavelet: str, level: int, window_size: int) -> dict:
    """
    Compute all DWT-derived features in a single rolling pass.

    Consolidates 4 separate DWT calls (coefficients, energy, volatility, trend)
    into one pywt.wavedec per window position.
    """
    n = len(signal)
    window_size = max(window_size, 2**level)

    # Coefficient outputs
    approx = np.full(n, np.nan)
    details = [np.full(n, np.nan) for _ in range(level)]

    # Energy outputs
    approx_energy = np.full(n, np.nan)
    detail_energies = [np.full(n, np.nan) for _ in range(level)]

    # Volatility output (from highest-freq detail)
    wavelet_vol = np.full(n, np.nan)

    # Trend outputs
    trend_strength = np.full(n, np.nan)
    trend_direction = np.full(n, np.nan)

    for i in range(window_size - 1, n):
        window_data = signal[i - window_size + 1 : i + 1]
        if np.any(np.isnan(window_data)):
            continue
        try:
            coeffs = pywt.wavedec(window_data, wavelet, level=level)

            # Coefficients (last value per level)
            if len(coeffs[0]) > 0:
                approx[i] = coeffs[0][-1]
            for lev in range(level):
                detail_idx = level - lev
                if len(coeffs[detail_idx]) > 0:
                    details[lev][i] = coeffs[detail_idx][-1]

            # Energy (sum of squares per level)
            approx_energy[i] = np.sum(coeffs[0] ** 2)
            for lev in range(level):
                detail_energies[lev][i] = np.sum(coeffs[level - lev] ** 2)

            # Volatility from highest-freq detail (equivalent to pywt.dwt detail)
            highest_detail = coeffs[-1]  # Last element = highest frequency detail
            mad = np.median(np.abs(highest_detail - np.median(highest_detail)))
            wavelet_vol[i] = mad / 0.6745  # MAD to sigma for Gaussian

            # Trend strength from approx slope
            approx_coeffs = coeffs[0]
            if len(approx_coeffs) >= 2:
                slope = (approx_coeffs[-1] - approx_coeffs[0]) / len(approx_coeffs)
                std = np.std(window_data)
                if std > 1e-10:
                    trend_strength[i] = np.abs(slope) / std
                    trend_direction[i] = np.sign(slope)
        except Exception as e:
            logger.debug(f"DWT failed for window ending at index {i}: {e}")
            continue

    return {
        "approx": approx,
        "details": details,
        "approx_energy": approx_energy,
        "detail_energies": detail_energies,
        "wavelet_vol": wavelet_vol,
        "trend_strength": trend_strength,
        "trend_direction": trend_direction,
    }


def _compute_dwt_rolling(
    signal: np.ndarray, wavelet: str, level: int, window_size: int, precomputed: dict | None = None
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Compute DWT coefficients on rolling windows. Uses precomputed results if available."""
    if precomputed is not None:
        return precomputed["approx"], precomputed["details"]

    # Fallback: compute standalone (when called without precomputation)
    result = _compute_dwt_all(signal, wavelet, level, window_size)
    return result["approx"], result["details"]


def _compute_energy_rolling(
    signal: np.ndarray, wavelet: str, level: int, window_size: int, precomputed: dict | None = None
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Compute wavelet energy at each decomposition level. Uses precomputed results if available."""
    if precomputed is not None:
        return precomputed["approx_energy"], precomputed["detail_energies"]

    # Fallback: compute standalone (when called without precomputation)
    result = _compute_dwt_all(signal, wavelet, level, window_size)
    return result["approx_energy"], result["detail_energies"]


def _compute_energy_ratio(
    approx_energy: np.ndarray, detail_energies: list[np.ndarray]
) -> np.ndarray:
    """Compute ratio of approximation energy to total energy."""
    total_energy = approx_energy.copy()
    for de in detail_energies:
        total_energy = total_energy + de
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total_energy > 0, approx_energy / total_energy, np.nan)


@njit(cache=True)
def _normalize_coefficients_numba(coeffs: np.ndarray) -> np.ndarray:
    """
    Normalize wavelet coefficients to z-scores using Welford's online algorithm.

    O(n) single-pass algorithm instead of O(n²) expanding window approach.
    Uses Welford's numerically stable online variance computation.
    """
    n = len(coeffs)
    result = np.full(n, np.nan)

    # Welford's online algorithm state
    count = 0
    mean = 0.0
    M2 = 0.0  # Sum of squared deviations

    for i in range(n):
        val = coeffs[i]
        if np.isnan(val):
            continue

        # Update Welford's accumulators
        count += 1
        delta = val - mean
        mean += delta / count
        delta2 = val - mean
        M2 += delta * delta2

        # Only output after minimum warmup period
        if count >= 20:
            # Compute std from M2 (population std)
            variance = M2 / count
            std = np.sqrt(variance) if variance > 0 else 0.0

            if std > 1e-10:
                result[i] = (val - mean) / std
            else:
                result[i] = 0.0

    return result


def _normalize_coefficients(coeffs: np.ndarray) -> np.ndarray:
    """Normalize wavelet coefficients to z-scores using expanding window."""
    return _normalize_coefficients_numba(coeffs.astype(np.float64))


def _get_freq_label(lev: int) -> str:
    """Get frequency label for detail level."""
    return "high" if lev == 0 else "mid" if lev == 1 else "low"


def add_wavelet_coefficients(
    df: pd.DataFrame,
    feature_metadata: dict[str, str],
    price_col: str = "close",
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
    window: int = DEFAULT_WINDOW,
    feature_prefix: str = "wavelet",
    normalize: bool = True,
    precomputed: dict | None = None,
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
    approx, details = _compute_dwt_rolling(signal, wavelet, level, window, precomputed)

    # ANTI-LOOKAHEAD: shift(1) ensures features at bar[t] use data up to bar[t-1]
    approx_col = f"{feature_prefix}_{price_col}_approx"
    if normalize:
        df[approx_col] = _np_shift1(_normalize_coefficients(approx))
        feature_metadata[approx_col] = f"Wavelet approx {wavelet} L{level} normalized (lagged)"
    else:
        df[approx_col] = _np_shift1(approx)
        feature_metadata[approx_col] = f"Wavelet approx {wavelet} L{level} (lagged)"

    for lev in range(level):
        detail_col = f"{feature_prefix}_{price_col}_d{lev + 1}"
        freq = _get_freq_label(lev)
        if normalize:
            df[detail_col] = _np_shift1(_normalize_coefficients(details[lev]))
            feature_metadata[detail_col] = (
                f"Wavelet detail {wavelet} L{lev+1} norm ({freq} freq, lagged)"
            )
        else:
            df[detail_col] = _np_shift1(details[lev])
            feature_metadata[detail_col] = (
                f"Wavelet detail {wavelet} L{lev+1} ({freq} freq, lagged)"
            )
    return df


def add_wavelet_energy(
    df: pd.DataFrame,
    feature_metadata: dict[str, str],
    price_col: str = "close",
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
    window: int = DEFAULT_WINDOW,
    feature_prefix: str = "wavelet",
    precomputed: dict | None = None,
) -> pd.DataFrame:
    """
    Add wavelet energy features at each decomposition level.

    Energy = sum of squared coefficients, indicating signal power in frequency bands.
    """
    if not PYWT_AVAILABLE:
        return df

    logger.info(f"Adding wavelet energy features ({wavelet}, level={level})...")
    signal = df[price_col].values
    approx_energy, detail_energies = _compute_energy_rolling(
        signal, wavelet, level, window, precomputed
    )

    # ANTI-LOOKAHEAD: shift(1)
    approx_energy_col = f"{feature_prefix}_{price_col}_energy_approx"
    with np.errstate(divide="ignore", invalid="ignore"):
        df[approx_energy_col] = _np_shift1(np.log1p(approx_energy))
    feature_metadata[approx_energy_col] = f"Wavelet approx energy log1p {wavelet} L{level} (lagged)"

    for lev in range(level):
        energy_col = f"{feature_prefix}_{price_col}_energy_d{lev + 1}"
        freq = _get_freq_label(lev)
        with np.errstate(divide="ignore", invalid="ignore"):
            df[energy_col] = _np_shift1(np.log1p(detail_energies[lev]))
        feature_metadata[energy_col] = f"Wavelet energy {wavelet} L{lev+1} ({freq} freq, lagged)"

    ratio_col = f"{feature_prefix}_{price_col}_energy_ratio"
    energy_ratio = _compute_energy_ratio(approx_energy, detail_energies)
    df[ratio_col] = _np_shift1(energy_ratio)
    feature_metadata[ratio_col] = f"Wavelet energy ratio {wavelet} L{level} (lagged)"
    return df


def add_wavelet_volatility(
    df: pd.DataFrame,
    feature_metadata: dict[str, str],
    price_col: str = "close",
    wavelet: str = DEFAULT_WAVELET,
    window: int = DEFAULT_WINDOW,
    feature_prefix: str = "wavelet",
    precomputed: dict | None = None,
) -> pd.DataFrame:
    """
    Add wavelet-based volatility estimate using MAD of detail coefficients.

    More robust to trends than standard deviation.
    """
    if not PYWT_AVAILABLE:
        return df

    logger.info(f"Adding wavelet volatility ({wavelet}, window={window})...")

    if precomputed is not None:
        wavelet_vol = precomputed["wavelet_vol"]
    else:
        signal = df[price_col].values
        n = len(signal)
        wavelet_vol = np.full(n, np.nan)
        actual_window = max(window, 8)

        for i in range(actual_window - 1, n):
            window_data = signal[i - actual_window + 1 : i + 1]
            if np.any(np.isnan(window_data)):
                continue
            try:
                _, detail = pywt.dwt(window_data, wavelet)
                mad = np.median(np.abs(detail - np.median(detail)))
                wavelet_vol[i] = mad / 0.6745  # MAD to sigma for Gaussian
            except Exception as e:
                logger.debug(f"Wavelet volatility failed for window ending at index {i}: {e}")
                continue

    # ANTI-LOOKAHEAD: shift(1)
    vol_col = f"{feature_prefix}_{price_col}_volatility"
    df[vol_col] = _np_shift1(wavelet_vol)
    feature_metadata[vol_col] = f"Wavelet volatility MAD {wavelet} (lagged)"
    return df


def add_wavelet_trend_strength(
    df: pd.DataFrame,
    feature_metadata: dict[str, str],
    price_col: str = "close",
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
    window: int = DEFAULT_WINDOW,
    feature_prefix: str = "wavelet",
    precomputed: dict | None = None,
) -> pd.DataFrame:
    """
    Add wavelet-based trend strength using slope of approximation coefficients.
    """
    if not PYWT_AVAILABLE:
        return df

    logger.info(f"Adding wavelet trend strength ({wavelet}, level={level})...")

    if precomputed is not None:
        trend_strength = precomputed["trend_strength"]
        trend_direction = precomputed["trend_direction"]
    else:
        signal = df[price_col].values
        n = len(signal)
        trend_strength = np.full(n, np.nan)
        trend_direction = np.full(n, np.nan)
        actual_window = max(window, 2**level)

        for i in range(actual_window - 1, n):
            window_data = signal[i - actual_window + 1 : i + 1]
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
            except Exception as e:
                logger.debug(f"Wavelet trend failed for window ending at index {i}: {e}")
                continue

    # ANTI-LOOKAHEAD: shift(1)
    strength_col = f"{feature_prefix}_{price_col}_trend_strength"
    direction_col = f"{feature_prefix}_{price_col}_trend_direction"
    df[strength_col] = _np_shift1(trend_strength)
    df[direction_col] = _np_shift1(trend_direction)
    feature_metadata[strength_col] = f"Wavelet trend strength {wavelet} L{level} (lagged)"
    feature_metadata[direction_col] = f"Wavelet trend direction {wavelet} L{level} (lagged)"
    return df


def add_wavelet_features(
    df: pd.DataFrame,
    feature_metadata: dict[str, str],
    price_col: str = "close",
    volume_col: str = "volume",
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
    window: int = DEFAULT_WINDOW,
    feature_prefix: str = "wavelet",
    include_volume: bool = True,
    include_energy: bool = True,
    include_volatility: bool = True,
    include_trend: bool = True,
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

    # Precompute DWT once for price signal (consolidates 4 redundant DWT calls)
    price_precomputed = _compute_dwt_all(df[price_col].values, wavelet, level, window)

    # Price coefficients (uses precomputed)
    df = add_wavelet_coefficients(
        df,
        feature_metadata,
        price_col=price_col,
        wavelet=wavelet,
        level=level,
        window=window,
        feature_prefix=feature_prefix,
        normalize=True,
        precomputed=price_precomputed,
    )

    # Volume coefficients (separate signal, computed independently)
    if include_volume and volume_col in df.columns and df[volume_col].sum() > 0:
        df = add_wavelet_coefficients(
            df,
            feature_metadata,
            price_col=volume_col,
            wavelet=wavelet,
            level=level,
            window=window,
            feature_prefix=feature_prefix,
            normalize=True,
        )

    # Energy features (uses precomputed for price)
    if include_energy:
        df = add_wavelet_energy(
            df,
            feature_metadata,
            price_col=price_col,
            wavelet=wavelet,
            level=level,
            window=window,
            feature_prefix=feature_prefix,
            precomputed=price_precomputed,
        )
        if include_volume and volume_col in df.columns and df[volume_col].sum() > 0:
            df = add_wavelet_energy(
                df,
                feature_metadata,
                price_col=volume_col,
                wavelet=wavelet,
                level=level,
                window=window,
                feature_prefix=feature_prefix,
            )

    # Volatility (uses precomputed)
    if include_volatility:
        df = add_wavelet_volatility(
            df,
            feature_metadata,
            price_col=price_col,
            wavelet=wavelet,
            window=window,
            feature_prefix=feature_prefix,
            precomputed=price_precomputed,
        )

    # Trend strength (uses precomputed)
    if include_trend:
        df = add_wavelet_trend_strength(
            df,
            feature_metadata,
            price_col=price_col,
            wavelet=wavelet,
            level=level,
            window=window,
            feature_prefix=feature_prefix,
            precomputed=price_precomputed,
        )

    logger.info(f"Added {len(df.columns) - initial_cols} wavelet features")
    return df


__all__ = [
    "add_wavelet_features",
    "add_wavelet_coefficients",
    "add_wavelet_energy",
    "add_wavelet_volatility",
    "add_wavelet_trend_strength",
    "SUPPORTED_WAVELETS",
    "DEFAULT_WAVELET",
    "DEFAULT_LEVEL",
    "DEFAULT_WINDOW",
    "PYWT_AVAILABLE",
]
