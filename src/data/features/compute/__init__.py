"""
Feature Computation Package - PHASE_1 Unified Features.

This package provides computation functions for all 196 base features
across 15 feature families.

Feature Families (196 total):
- raw: 5 features (OHLCV passthrough)
- momentum: 23 features (RSI, MACD, Stochastic, etc.)
- moving_average: 16 features (SMA, EMA, crossovers)
- volatility: 25 features (ATR, BB, KC, GARCH)
- volume: 15 features (OBV, VWAP, TWAP)
- trend: 6 features (ADX, Supertrend)
- price: 12 features (returns, ratios, autocorr)
- microstructure: 15 features (Amihud, Roll, Kyle)
- entropy: 12 features (Shannon, LZ, ApEn, Hurst)
- wavelets: 15 features (DWT coefficients)
- temporal: 9 features (hour/day encoding, sessions)
- regime: 9 features (vol/trend regimes)
- order_flow: 12 features (buy/sell pressure, volume delta)
- liquidity: 12 features (spread estimation, liquidity regime)
- mean_reversion: 10 features (z-scores, Hurst, variance ratios)

Usage:
    from src.data.features.compute import (
        compute_all_features,
        compute_all_features_parallel,
        compute_features_by_family,
        compute_single_feature,
        FEATURE_COMPUTE_MAP,
        get_feature_families,
        get_features_in_family,
    )

    # Compute all features (sequential)
    features_df = compute_all_features(ohlcv_df)

    # Compute all features (parallel - 2-4x speedup)
    features_df = compute_all_features_parallel(ohlcv_df, n_jobs=4)

    # Compute specific families
    mom_features = compute_features_by_family(ohlcv_df, ["momentum", "volatility"])

    # Compute single feature
    rsi_14 = compute_single_feature(ohlcv_df, "rsi_14")
"""

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from src.data.features.compute.entropy import ENTROPY_FEATURES
from src.data.features.compute.liquidity import LIQUIDITY_FEATURES
from src.data.features.compute.mean_reversion import MEAN_REVERSION_FEATURES
from src.data.features.compute.microstructure import MICROSTRUCTURE_FEATURES
from src.data.features.compute.momentum import MOMENTUM_FEATURES
from src.data.features.compute.moving_average import MOVING_AVERAGE_FEATURES

# MTF (Multi-Timeframe) features - PHASE_1
from src.data.features.compute.mtf import (
    DEFAULT_MTF_FEATURES,
    MTFConfig,
    MTFFeatureComputer,
    compute_mtf_features,
    get_mtf_feature_names,
    resample_ohlcv,
    validate_mtf_config,
)
from src.data.features.compute.order_flow import ORDER_FLOW_FEATURES
from src.data.features.compute.price import PRICE_FEATURES

# Import all feature maps from submodules
from src.data.features.compute.raw import RAW_FEATURES
from src.data.features.compute.regime import REGIME_FEATURES
from src.data.features.compute.temporal import TEMPORAL_FEATURES
from src.data.features.compute.trend import TREND_FEATURES
from src.data.features.compute.volatility import VOLATILITY_FEATURES
from src.data.features.compute.volume import VOLUME_FEATURES
from src.data.features.compute.wavelets import PYWT_AVAILABLE, WAVELETS_FEATURES

# =============================================================================
# FEATURE FAMILY MAPS
# =============================================================================

FAMILY_FEATURE_MAPS: dict[str, dict[str, Callable[[pd.DataFrame], pd.Series]]] = {
    "raw": RAW_FEATURES,
    "momentum": MOMENTUM_FEATURES,
    "moving_average": MOVING_AVERAGE_FEATURES,
    "volatility": VOLATILITY_FEATURES,
    "volume": VOLUME_FEATURES,
    "trend": TREND_FEATURES,
    "price": PRICE_FEATURES,
    "microstructure": MICROSTRUCTURE_FEATURES,
    "entropy": ENTROPY_FEATURES,
    "wavelets": WAVELETS_FEATURES,
    "temporal": TEMPORAL_FEATURES,
    "regime": REGIME_FEATURES,
    # Phase 19A - ML Pipeline Enhancements
    "order_flow": ORDER_FLOW_FEATURES,
    "liquidity": LIQUIDITY_FEATURES,
    "mean_reversion": MEAN_REVERSION_FEATURES,
}

# =============================================================================
# UNIFIED FEATURE MAP - All 162 features
# =============================================================================

FEATURE_COMPUTE_MAP: dict[str, Callable[[pd.DataFrame], pd.Series]] = {}

# Build unified map from all families
for _family_name, family_map in FAMILY_FEATURE_MAPS.items():
    FEATURE_COMPUTE_MAP.update(family_map)

# Verify feature count
TOTAL_FEATURES = len(FEATURE_COMPUTE_MAP)
EXPECTED_FEATURES = 196  # 162 original + 12 order_flow + 12 liquidity + 10 mean_reversion

# Log warning if count mismatch (don't raise error to allow flexibility)
if TOTAL_FEATURES != EXPECTED_FEATURES:
    import warnings

    warnings.warn(
        f"Feature count mismatch: expected {EXPECTED_FEATURES}, got {TOTAL_FEATURES}. "
        "This may indicate missing or extra features.",
        stacklevel=2,
    )


# =============================================================================
# FEATURE FAMILY METADATA
# =============================================================================

FEATURE_FAMILY_COUNTS: dict[str, int] = {
    family: len(features) for family, features in FAMILY_FEATURE_MAPS.items()
}

FEATURE_TO_FAMILY: dict[str, str] = {
    feature_name: family
    for family, features in FAMILY_FEATURE_MAPS.items()
    for feature_name in features
}


# =============================================================================
# COMPUTATION FUNCTIONS
# =============================================================================


def get_feature_families() -> list[str]:
    """Get list of all feature family names."""
    return list(FAMILY_FEATURE_MAPS.keys())


def get_features_in_family(family: str) -> list[str]:
    """Get list of feature names in a specific family."""
    if family not in FAMILY_FEATURE_MAPS:
        raise ValueError(
            f"Unknown family '{family}'. Valid families: {list(FAMILY_FEATURE_MAPS.keys())}"
        )
    return list(FAMILY_FEATURE_MAPS[family].keys())


def get_all_feature_names() -> list[str]:
    """Get list of all feature names."""
    return list(FEATURE_COMPUTE_MAP.keys())


def get_feature_family(feature_name: str) -> str:
    """Get the family name for a specific feature."""
    if feature_name not in FEATURE_TO_FAMILY:
        raise ValueError(f"Unknown feature '{feature_name}'")
    return FEATURE_TO_FAMILY[feature_name]


def compute_single_feature(
    df: pd.DataFrame,
    feature_name: str,
) -> pd.Series:
    """
    Compute a single feature.

    Args:
        df: DataFrame with OHLCV columns
        feature_name: Name of the feature to compute

    Returns:
        pd.Series with computed feature values

    Raises:
        ValueError: If feature_name is unknown
    """
    if feature_name not in FEATURE_COMPUTE_MAP:
        raise ValueError(
            f"Unknown feature '{feature_name}'. " f"Available features: {len(FEATURE_COMPUTE_MAP)}"
        )

    compute_fn = FEATURE_COMPUTE_MAP[feature_name]
    return compute_fn(df)


def compute_features_by_family(
    df: pd.DataFrame,
    families: str | list[str],
    return_dataframe: bool = True,
) -> pd.DataFrame | dict[str, pd.Series]:
    """
    Compute all features for specified families.

    Args:
        df: DataFrame with OHLCV columns
        families: Single family name or list of family names
        return_dataframe: If True, return DataFrame; else return dict

    Returns:
        DataFrame or dict with computed features
    """
    if isinstance(families, str):
        families = [families]

    # Validate families
    for family in families:
        if family not in FAMILY_FEATURE_MAPS:
            raise ValueError(
                f"Unknown family '{family}'. Valid: {list(FAMILY_FEATURE_MAPS.keys())}"
            )

    results = {}

    for family in families:
        family_map = FAMILY_FEATURE_MAPS[family]
        for feature_name, compute_fn in family_map.items():
            try:
                results[feature_name] = compute_fn(df)
            except Exception as e:
                # Log error and continue with NaN
                import warnings

                warnings.warn(f"Error computing {feature_name}: {e}", stacklevel=2)
                results[feature_name] = pd.Series(float("nan"), index=df.index)

    if return_dataframe:
        return pd.DataFrame(results, index=df.index)
    return results


def compute_features_by_names(
    df: pd.DataFrame,
    feature_names: list[str],
    return_dataframe: bool = True,
) -> pd.DataFrame | dict[str, pd.Series]:
    """
    Compute specific features by name.

    Args:
        df: DataFrame with OHLCV columns
        feature_names: List of feature names to compute
        return_dataframe: If True, return DataFrame; else return dict

    Returns:
        DataFrame or dict with computed features
    """
    results = {}

    for feature_name in feature_names:
        if feature_name not in FEATURE_COMPUTE_MAP:
            import warnings

            warnings.warn(f"Unknown feature '{feature_name}', skipping", stacklevel=2)
            continue

        compute_fn = FEATURE_COMPUTE_MAP[feature_name]
        try:
            results[feature_name] = compute_fn(df)
        except Exception as e:
            import warnings

            warnings.warn(f"Error computing {feature_name}: {e}", stacklevel=2)
            results[feature_name] = pd.Series(float("nan"), index=df.index)

    if return_dataframe:
        return pd.DataFrame(results, index=df.index)
    return results


def compute_all_features(
    df: pd.DataFrame,
    exclude_families: list[str] | None = None,
    exclude_features: list[str] | None = None,
    include_original: bool = False,
) -> pd.DataFrame:
    """
    Compute all 162 base features.

    Args:
        df: DataFrame with OHLCV columns (open, high, low, close, volume)
        exclude_families: Optional list of families to skip
        exclude_features: Optional list of specific features to skip
        include_original: If True, include original OHLCV columns in output

    Returns:
        DataFrame with all computed features

    Example:
        >>> df = pd.read_parquet("data.parquet")
        >>> features = compute_all_features(df)
        >>> print(features.shape)  # (n_rows, 162)
    """
    exclude_families = exclude_families or []
    exclude_features = exclude_features or []

    results = {}

    # Optionally include original columns
    if include_original:
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                results[col] = df[col].copy()

    # Compute features family by family
    for family, family_map in FAMILY_FEATURE_MAPS.items():
        if family in exclude_families:
            continue

        for feature_name, compute_fn in family_map.items():
            if feature_name in exclude_features:
                continue

            # Skip if already included from original
            if feature_name in results:
                continue

            try:
                results[feature_name] = compute_fn(df)
            except Exception as e:
                import warnings

                warnings.warn(f"Error computing {feature_name}: {e}", stacklevel=2)
                results[feature_name] = pd.Series(float("nan"), index=df.index)

    return pd.DataFrame(results, index=df.index)


def _compute_family_features(
    df_values: dict,
    df_index: pd.Index,
    family: str,
    exclude_features: list[str],
) -> dict[str, pd.Series]:
    """
    Worker function to compute features for a single family.

    This function is designed to be called in parallel processes.
    It reconstructs the DataFrame from values to avoid pickling issues.

    Args:
        df_values: Dict of column name to values (numpy arrays)
        df_index: DataFrame index
        family: Feature family name
        exclude_features: Features to skip

    Returns:
        Dict mapping feature names to computed Series
    """
    # Reconstruct DataFrame in worker process
    df = pd.DataFrame(df_values, index=df_index)

    family_map = FAMILY_FEATURE_MAPS[family]
    results = {}

    for feature_name, compute_fn in family_map.items():
        if feature_name in exclude_features:
            continue

        try:
            results[feature_name] = compute_fn(df)
        except Exception as e:
            import warnings

            warnings.warn(f"Error computing {feature_name}: {e}", stacklevel=2)
            results[feature_name] = pd.Series(float("nan"), index=df.index)

    return results


def compute_all_features_parallel(
    df: pd.DataFrame,
    exclude_families: list[str] | None = None,
    exclude_features: list[str] | None = None,
    include_original: bool = False,
    n_jobs: int = 4,
) -> pd.DataFrame:
    """
    Compute all features with parallel execution by family.

    Uses ProcessPoolExecutor to compute feature families in parallel.
    This provides 2-4x speedup on multi-core systems for large DataFrames.

    Args:
        df: DataFrame with OHLCV columns (open, high, low, close, volume)
        exclude_families: Optional list of families to skip
        exclude_features: Optional list of specific features to skip
        include_original: If True, include original OHLCV columns in output
        n_jobs: Number of parallel workers (default: 4)

    Returns:
        DataFrame with all computed features

    Note:
        For small DataFrames (<1000 rows), sequential execution may be faster
        due to process spawning overhead. Use compute_all_features() instead.

    Example:
        >>> df = pd.read_parquet("data.parquet")
        >>> features = compute_all_features_parallel(df, n_jobs=4)
        >>> print(features.shape)  # (n_rows, 162)
    """
    exclude_families = exclude_families or []
    exclude_features = exclude_features or []

    results = {}

    # Optionally include original columns
    if include_original:
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                results[col] = df[col].copy()

    # Get families to compute
    families_to_compute = [
        family for family in FAMILY_FEATURE_MAPS if family not in exclude_families
    ]

    # For small datasets or few families, use sequential execution
    if len(df) < 1000 or len(families_to_compute) <= 2:
        return compute_all_features(
            df,
            exclude_families=exclude_families,
            exclude_features=exclude_features,
            include_original=include_original,
        )

    # Prepare data for workers (avoid pickling DataFrame directly)
    df_values = {col: df[col].values for col in df.columns}
    df_index = df.index

    # Submit family computations in parallel
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {}
        for family in families_to_compute:
            future = executor.submit(
                _compute_family_features,
                df_values,
                df_index,
                family,
                exclude_features,
            )
            futures[future] = family

        # Collect results as they complete
        for future in as_completed(futures):
            family = futures[future]
            try:
                family_results = future.result()
                for feature_name, series in family_results.items():
                    if feature_name not in results:
                        results[feature_name] = series
            except Exception as e:
                import warnings

                warnings.warn(f"Error computing family {family}: {e}", stacklevel=2)

    return pd.DataFrame(results, index=df.index)


def get_feature_info() -> dict[str, dict]:
    """
    Get metadata about all features.

    Returns:
        Dict mapping feature names to info dict with:
        - family: Feature family name
        - index: Index within family
    """
    info = {}

    for family, features in FAMILY_FEATURE_MAPS.items():
        for idx, feature_name in enumerate(features.keys()):
            info[feature_name] = {
                "family": family,
                "index": idx,
            }

    return info


def validate_features_computed(
    df: pd.DataFrame,
    expected_features: list[str] | None = None,
) -> dict[str, bool]:
    """
    Validate that expected features exist and have valid values.

    Args:
        df: DataFrame with computed features
        expected_features: List of feature names to check (default: all)

    Returns:
        Dict mapping feature names to validity (True if column exists and has data)
    """
    if expected_features is None:
        expected_features = list(FEATURE_COMPUTE_MAP.keys())

    results = {}
    for feature in expected_features:
        if feature in df.columns:
            # Check if column has any non-NaN values
            results[feature] = df[feature].notna().any()
        else:
            results[feature] = False

    return results


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main computation functions
    "compute_all_features",
    "compute_all_features_parallel",
    "compute_features_by_family",
    "compute_features_by_names",
    "compute_single_feature",
    # Query functions
    "get_feature_families",
    "get_features_in_family",
    "get_all_feature_names",
    "get_feature_family",
    "get_feature_info",
    "validate_features_computed",
    # Feature maps
    "FEATURE_COMPUTE_MAP",
    "FAMILY_FEATURE_MAPS",
    "FEATURE_FAMILY_COUNTS",
    "FEATURE_TO_FAMILY",
    # Individual family maps
    "RAW_FEATURES",
    "MOMENTUM_FEATURES",
    "MOVING_AVERAGE_FEATURES",
    "VOLATILITY_FEATURES",
    "VOLUME_FEATURES",
    "TREND_FEATURES",
    "PRICE_FEATURES",
    "MICROSTRUCTURE_FEATURES",
    "ENTROPY_FEATURES",
    "WAVELETS_FEATURES",
    "TEMPORAL_FEATURES",
    "REGIME_FEATURES",
    # Phase 19A - ML Pipeline Enhancements
    "ORDER_FLOW_FEATURES",
    "LIQUIDITY_FEATURES",
    "MEAN_REVERSION_FEATURES",
    # Constants
    "TOTAL_FEATURES",
    "PYWT_AVAILABLE",
    # MTF (Multi-Timeframe) - PHASE_1
    "MTFConfig",
    "MTFFeatureComputer",
    "compute_mtf_features",
    "get_mtf_feature_names",
    "validate_mtf_config",
    "resample_ohlcv",
    "DEFAULT_MTF_FEATURES",
]
