"""
Stage 3: Feature Engineering.

Orchestration logic for Stage 3 of the pipeline.
Generates technical indicators and derived features from cleaned OHLCV data.

Uses the modular feature engineering implementation from stages.features.
"""

import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from joblib import Parallel, delayed

from src.core.exceptions import FeatureSchemaError
from src.data.pipeline.constants import EXCLUDED_COLUMNS
from src.data.pipeline.utils import StageResult, create_failed_result, create_stage_result
from src.data.store.store import FeatureStore

if TYPE_CHECKING:
    from src.core.common.manifest import ArtifactManifest
    from src.data.pipeline.data_config import DataConfig as PipelineConfig

logger = logging.getLogger(__name__)


# =============================================================================
# NaN Monitoring Configuration
# =============================================================================
DEFAULT_NAN_THRESHOLD = 0.10  # Max 10% NaN allowed after warmup
DEFAULT_WARMUP_BARS = 200  # Skip first 200 bars (expected NaN from indicators)
NAN_WARNING_THRESHOLD = 0.05  # Log warning if NaN > 5% (even if below max)


def validate_feature_nan_ratio(
    df: pd.DataFrame,
    ohlcv_cols: set[str],
    warmup_bars: int = DEFAULT_WARMUP_BARS,
    max_nan_ratio: float = DEFAULT_NAN_THRESHOLD,
    symbol: str = "",
    timeframe: str = "",
) -> dict[str, float]:
    """
    Validate that features don't have excessive NaN values after warmup period.

    This function provides fail-fast behavior when features have too many NaN values,
    which could indicate calculation errors, missing data, or other issues that
    would propagate through the pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with feature columns to validate
    ohlcv_cols : set[str]
        Set of OHLCV column names to exclude from feature validation
    warmup_bars : int
        Number of initial bars to skip (expected NaN from indicator warmup)
    max_nan_ratio : float
        Maximum allowed NaN ratio (0.10 = 10%)
    symbol : str
        Symbol name for error messages
    timeframe : str
        Timeframe for error messages

    Returns
    -------
    dict[str, float]
        Dictionary mapping feature names to their NaN ratios (for features with any NaN)

    Raises
    ------
    FeatureSchemaError
        If any feature has NaN ratio > max_nan_ratio after warmup period
    """
    # Skip if DataFrame is too small for validation
    if len(df) <= warmup_bars:
        logger.warning(
            f"DataFrame has {len(df)} rows, less than warmup_bars={warmup_bars}. "
            "Skipping NaN validation."
        )
        return {}

    # Get feature columns (exclude OHLCV and metadata columns)
    feature_cols = [c for c in df.columns if c not in ohlcv_cols]

    if not feature_cols:
        logger.warning("No feature columns found for NaN validation")
        return {}

    # Slice to post-warmup period
    post_warmup_df = df.iloc[warmup_bars:]
    post_warmup_rows = len(post_warmup_df)

    # Calculate NaN ratios for each feature
    nan_ratios: dict[str, float] = {}
    excessive_nan_features: list[tuple[str, float, int]] = []
    elevated_nan_features: list[tuple[str, float]] = []

    for col in feature_cols:
        if col not in post_warmup_df.columns:
            continue

        nan_count = post_warmup_df[col].isna().sum()
        nan_ratio = nan_count / post_warmup_rows

        if nan_ratio > 0:
            nan_ratios[col] = nan_ratio

        if nan_ratio > max_nan_ratio:
            excessive_nan_features.append((col, nan_ratio, nan_count))
        elif nan_ratio > NAN_WARNING_THRESHOLD:
            elevated_nan_features.append((col, nan_ratio))

    # Log warnings for elevated NaN (below threshold but notable)
    if elevated_nan_features:
        context = f"{symbol}@{timeframe}" if symbol else "features"
        for col, ratio in elevated_nan_features:
            logger.warning(
                f"[{context}] Feature '{col}' has elevated NaN ratio: "
                f"{ratio:.1%} (warning threshold: {NAN_WARNING_THRESHOLD:.0%})"
            )

    # Fail-fast if any feature exceeds threshold
    if excessive_nan_features:
        context = f"{symbol}@{timeframe}" if symbol else "features"
        # Sort by NaN ratio descending
        excessive_nan_features.sort(key=lambda x: x[1], reverse=True)

        # Build detailed error message
        feature_details = "\n".join(
            f"  - {col}: {ratio:.1%} NaN ({nan_count:,} of {post_warmup_rows:,} rows)"
            for col, ratio, nan_count in excessive_nan_features[:10]  # Show top 10
        )

        more_count = len(excessive_nan_features) - 10
        if more_count > 0:
            feature_details += f"\n  ... and {more_count} more features with excessive NaN"

        error_msg = (
            f"[{context}] {len(excessive_nan_features)} feature(s) exceed NaN threshold "
            f"({max_nan_ratio:.0%}) after warmup period ({warmup_bars} bars):\n"
            f"{feature_details}\n\n"
            f"This indicates potential issues with:\n"
            f"  - Feature calculation errors\n"
            f"  - Missing source data\n"
            f"  - Indicator configuration problems\n\n"
            f"Fix the underlying data issues or adjust nan_threshold/warmup_bars in config."
        )

        raise FeatureSchemaError(error_msg)

    # Log summary if any NaN found
    if nan_ratios:
        context = f"{symbol}@{timeframe}" if symbol else "features"
        logger.info(
            f"[{context}] NaN validation passed: {len(nan_ratios)} features have some NaN "
            f"(all below {max_nan_ratio:.0%} threshold after {warmup_bars} warmup bars)"
        )

    return nan_ratios


# Local import to avoid circular dependency
from .engineer import FeatureEngineer  # noqa: E402


def _process_symbol_timeframe(
    symbol: str,
    tf: str,
    config: "PipelineConfig",
    feature_store: FeatureStore,
    ohlcv_cols: set[str],
    mtf_mode: str,
    mtf_timeframes: list[str],
    mtf_include_ohlcv: bool,
    mtf_include_indicators: bool,
    enable_wavelets: bool,
    enable_microstructure: bool,
    enable_volume_features: bool,
    enable_volatility_features: bool,
    nan_threshold: float = DEFAULT_NAN_THRESHOLD,
    warmup_bars: int = DEFAULT_WARMUP_BARS,
) -> tuple[str, str, Path, dict[str, Any]] | None:
    """
    Process features for a single symbol/timeframe combination.

    Returns tuple of (symbol, tf, output_file, metadata) or None on failure.
    """
    try:
        input_file = config.clean_data_dir / f"{symbol}_{tf}_clean.parquet"
        if not input_file.exists():
            logger.warning(f"No cleaned data found for {symbol} @ {tf}")
            return None

        df = pd.read_parquet(input_file)
        logger.info(f"Loaded {symbol} @ {tf}: {len(df):,} rows")

        # Filter MTF timeframes
        from src.core.common.timeframes import timeframe_to_minutes

        current_minutes = timeframe_to_minutes(tf)
        valid_mtf = [mtf for mtf in mtf_timeframes if timeframe_to_minutes(mtf) > current_minutes]

        # Build cache key
        feature_set = f"{tf}_features"
        cache_key_config = {
            "mtf_mode": mtf_mode,
            "mtf_timeframes": valid_mtf,
            "wavelets": enable_wavelets,
            "microstructure": enable_microstructure,
            "volume": enable_volume_features,
            "volatility": enable_volatility_features,
        }

        # Check cache
        df_features = None
        if feature_store.has_features(symbol=symbol, feature_set=feature_set):
            try:
                df_cached, cache_metadata = feature_store.get_features_with_metadata(
                    symbol=symbol, feature_set=feature_set
                )
                cached_config = cache_metadata.metadata.get("feature_config", {})
                if cached_config == cache_key_config:
                    logger.info(
                        f"  ✓ Using cached features for {symbol} @ {tf} "
                        f"({len(df_cached):,} rows, {len(df_cached.columns)} cols)"
                    )
                    df_features = df_cached
                else:
                    logger.info(f"  Cache miss: Config mismatch for {symbol} @ {tf}")
            except Exception as e:
                logger.warning(f"  Cache retrieval failed for {symbol} @ {tf}: {e}")

        # Compute if not cached
        needs_caching = False
        if df_features is None:
            engineer = FeatureEngineer(
                input_dir=config.clean_data_dir,
                output_dir=config.features_dir,
                timeframe=tf,
                enable_mtf=bool(valid_mtf),
                mtf_timeframes=valid_mtf,
                mtf_include_ohlcv=mtf_include_ohlcv,
                mtf_include_indicators=mtf_include_indicators,
                base_timeframe=tf,
                enable_wavelets=enable_wavelets,
                enable_microstructure=enable_microstructure,
                enable_volume_features=enable_volume_features,
                enable_volatility_features=enable_volatility_features,
            )

            logger.info(f"Computing features for {symbol} @ {tf}...")
            df_features, _ = engineer.engineer_features(df, symbol)
            needs_caching = True

        # Validate NaN ratios after warmup period (applies to both cached and fresh)
        # This is fail-fast: raises FeatureSchemaError if any feature exceeds threshold
        validate_feature_nan_ratio(
            df=df_features,
            ohlcv_cols=ohlcv_cols,
            warmup_bars=warmup_bars,
            max_nan_ratio=nan_threshold,
            symbol=symbol,
            timeframe=tf,
        )

        # Cache the results (only for freshly computed features)
        if needs_caching:
            try:
                feature_store.put_features(
                    df=df_features,
                    symbol=symbol,
                    feature_set=feature_set,
                    lineage={
                        "raw_path": str(input_file),
                        "timeframe": tf,
                        "config": cache_key_config,
                        "metadata": {"symbol": symbol, "timeframe": tf},
                    },
                    description=f"Features for {symbol} @ {tf}",
                    metadata={"feature_config": cache_key_config},
                )
                logger.info(f"  ✓ Cached features for {symbol} @ {tf}")
            except Exception as e:
                logger.warning(f"  Failed to cache features for {symbol} @ {tf}: {e}")

        # Save to file
        output_file = config.features_dir / f"{symbol}_{tf}_features.parquet"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df_features.to_parquet(output_file, index=False)

        # Build metadata
        feature_cols = [c for c in df_features.columns if c not in ohlcv_cols]
        metadata = {
            "total_rows": len(df_features),
            "feature_count": len(feature_cols),
            "feature_columns": feature_cols[:20],
        }

        logger.info(f"  {symbol} @ {tf}: {len(df_features):,} rows, {len(feature_cols)} features")

        return (symbol, tf, output_file, metadata)
    except Exception as e:
        logger.error(f"Failed processing {symbol} @ {tf}: {e}")
        return None


def run_feature_engineering(
    config: "PipelineConfig", manifest: "ArtifactManifest"
) -> "StageResult":
    """
    Stage 3: Feature Engineering.

    Generates 150+ technical features including:
    - Moving averages (SMA, EMA)
    - Momentum indicators (RSI, MACD, Stochastic)
    - Volatility indicators (ATR, Bollinger Bands)
    - Volume-based indicators
    - Price patterns and derived features
    - Temporal and regime indicators
    - Wavelet decomposition features
    - Microstructure proxy features
    - Multi-timeframe (MTF) indicator features

    NOTE: Symbol Isolation Policy - Each symbol is processed in complete
    isolation. There are NO cross-symbol or cross-asset features (no correlation,
    beta, spread, or relative strength features between symbols). See
    src/phase1/config/features.py SYMBOL ISOLATION POLICY for details.

    Args:
        config: Pipeline configuration
        manifest: Artifact manifest for tracking outputs

    Returns:
        StageResult with status and artifacts
    """
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("STAGE 3: Feature Engineering")
    logger.info("=" * 70)

    try:
        # Get effective output timeframes (supports multi-TF mode)
        output_timeframes = config.effective_output_timeframes
        is_multi_tf = len(output_timeframes) > 1

        logger.info(f"Output timeframes: {output_timeframes}")
        logger.info(f"Multi-TF mode: {is_multi_tf}")

        # Determine MTF include flags based on config.mtf_mode
        # mtf_mode: 'bars' -> only OHLCV, 'indicators' -> only indicators, 'both' -> both
        mtf_mode = getattr(config, "mtf_mode", "both")
        mtf_include_ohlcv = mtf_mode in ("bars", "both")
        mtf_include_indicators = mtf_mode in ("indicators", "both")

        # Use MTF timeframes from PipelineConfig (respects CLI overrides)
        mtf_timeframes = getattr(config, "mtf_timeframes", ["15min", "60min"])

        # Get feature toggles from config (if specified)
        feature_toggles = getattr(config, "feature_toggles", None) or {}
        enable_wavelets = feature_toggles.get("wavelets", True)
        enable_microstructure = feature_toggles.get("microstructure", True)
        enable_volume_features = feature_toggles.get("volume", True)
        enable_volatility_features = feature_toggles.get("volatility", True)

        # Get NaN monitoring thresholds from config (with strict defaults)
        nan_threshold = getattr(config, "nan_threshold", DEFAULT_NAN_THRESHOLD)
        warmup_bars = getattr(config, "warmup_bars", DEFAULT_WARMUP_BARS)
        logger.info(f"NaN monitoring: threshold={nan_threshold:.0%}, warmup_bars={warmup_bars}")

        # Build effective toggles dict for saving (captures actual runtime values)
        effective_toggles = {
            "wavelets_enabled": enable_wavelets,
            "microstructure_enabled": enable_microstructure,
            "volume_enabled": enable_volume_features,
            "volatility_enabled": enable_volatility_features,
            "mtf_enabled": bool(mtf_timeframes),
            "mtf_mode": mtf_mode,
            "mtf_timeframes": mtf_timeframes,
            "mtf_include_ohlcv": mtf_include_ohlcv,
            "mtf_include_indicators": mtf_include_indicators,
            "nan_threshold": nan_threshold,
            "warmup_bars": warmup_bars,
        }

        # Initialize FeatureStore for caching (30-120s speedup on repeated runs)
        feature_cache_dir = config.features_dir.parent / "feature_cache"
        feature_store = FeatureStore(cache_dir=feature_cache_dir)
        logger.info(f"FeatureStore initialized at: {feature_cache_dir}")

        # OHLCV columns to exclude from feature count (use canonical constant)
        ohlcv_cols = EXCLUDED_COLUMNS

        # Build list of all symbol/timeframe combinations
        tasks = [(symbol, tf) for symbol in config.symbols for tf in output_timeframes]

        # Process in parallel (n_jobs=-1 uses all cores)
        n_jobs = getattr(config, "n_jobs", -1)
        logger.info(
            f"Processing {len(tasks)} symbol/timeframe combinations in parallel (n_jobs={n_jobs})"
        )

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_process_symbol_timeframe)(
                symbol=symbol,
                tf=tf,
                config=config,
                feature_store=feature_store,
                ohlcv_cols=ohlcv_cols,
                mtf_mode=mtf_mode,
                mtf_timeframes=mtf_timeframes,
                mtf_include_ohlcv=mtf_include_ohlcv,
                mtf_include_indicators=mtf_include_indicators,
                enable_wavelets=enable_wavelets,
                enable_microstructure=enable_microstructure,
                enable_volume_features=enable_volume_features,
                enable_volatility_features=enable_volatility_features,
                nan_threshold=nan_threshold,
                warmup_bars=warmup_bars,
            )
            for symbol, tf in tasks
        )

        # Collect results and track failures
        artifacts: list[Path] = []
        feature_metadata: dict[str, dict[str, Any]] = {}

        # Track failed tasks (where result is None)
        failed_tasks = [
            (task, result) for task, result in zip(tasks, results, strict=True) if result is None
        ]

        if failed_tasks:
            failed_symbols = [f"{symbol}_{tf}" for (symbol, tf), _ in failed_tasks]
            logger.error(
                f"Feature engineering failed for {len(failed_tasks)} tasks: {failed_symbols}"
            )
            for (symbol, tf), _ in failed_tasks:
                logger.error(f"  - FAILED: {symbol}@{tf}")

            # Phase 43: Configurable fail-fast on partial failures
            total_tasks = len(tasks)
            success_count = total_tasks - len(failed_tasks)
            success_rate = success_count / total_tasks if total_tasks > 0 else 0.0

            fail_on_partial = getattr(config, "stage3_fail_on_partial", True)
            min_success_rate = getattr(config, "stage3_min_success_rate", 0.95)

            if fail_on_partial and success_rate < min_success_rate:
                error_msg = (
                    f"Feature engineering failed: only {success_rate:.1%} tasks succeeded "
                    f"(minimum required: {min_success_rate:.0%}). "
                    f"Failed: {len(failed_tasks)}/{total_tasks} tasks. "
                    f"Set stage3_fail_on_partial=False to allow partial failures."
                )
                logger.error(error_msg)
                return create_failed_result(
                    stage_name="feature_engineering",
                    start_time=start_time,
                    error=error_msg,
                )
            else:
                logger.warning(
                    f"Partial failure allowed: {success_rate:.1%} succeeded "
                    f"({success_count}/{total_tasks}). Continuing with available results."
                )

        for result in results:
            if result is None:
                continue

            symbol, tf, output_file, metadata = result
            artifacts.append(output_file)

            if symbol not in feature_metadata:
                feature_metadata[symbol] = {}
            feature_metadata[symbol][tf] = metadata

            # Add to manifest
            manifest.add_artifact(
                name=f"features_{symbol}_{tf}",
                file_path=output_file,
                stage="feature_engineering",
                metadata={
                    "symbol": symbol,
                    "timeframe": tf,
                    "feature_count": metadata["feature_count"],
                    "row_count": metadata["total_rows"],
                },
            )

        # Save feature toggles to JSON for reproducibility and downstream stages
        toggles_file = config.features_dir / "feature_toggles.json"
        config.features_dir.mkdir(parents=True, exist_ok=True)
        with open(toggles_file, "w") as f:
            json.dump(effective_toggles, f, indent=2)
        artifacts.append(toggles_file)
        logger.info(f"Saved feature toggles to: {toggles_file}")

        # Add feature toggles to manifest
        manifest.add_artifact(
            name="feature_toggles",
            file_path=toggles_file,
            stage="feature_engineering",
            metadata=effective_toggles,
        )

        return create_stage_result(
            stage_name="feature_engineering",
            start_time=start_time,
            artifacts=artifacts,
            metadata={
                "feature_results": feature_metadata,
                "output_timeframes": output_timeframes,
                "multi_tf_mode": is_multi_tf,
                "feature_toggles": effective_toggles,
            },
        )

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        logger.error(traceback.format_exc())
        return create_failed_result(
            stage_name="feature_engineering", start_time=start_time, error=str(e)
        )
