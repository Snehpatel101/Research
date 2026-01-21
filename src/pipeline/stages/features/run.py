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
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from manifest import ArtifactManifest
    from pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)

# Import from local modules
from .engineer import FeatureEngineer

# StageResult imports - adjust path based on pipeline structure
try:
    from src.pipeline.utils import StageResult, create_failed_result, create_stage_result
except ImportError:
    # Fallback for different import paths
    from pipeline.utils import StageResult, create_failed_result, create_stage_result


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
        }

        # Process each symbol independently (no cross-symbol correlation)
        artifacts = []
        feature_metadata = {}

        # OHLCV columns to exclude from feature count
        ohlcv_cols = {
            "datetime",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "timeframe",
            "session_id",
            "missing_bar",
            "roll_event",
            "roll_window",
            "filled",
        }

        for symbol in config.symbols:
            feature_metadata[symbol] = {}

            # Process each output timeframe
            for tf in output_timeframes:
                input_file = config.clean_data_dir / f"{symbol}_{tf}_clean.parquet"
                if not input_file.exists():
                    logger.warning(f"No cleaned data found for {symbol} @ {tf}")
                    continue

                df = pd.read_parquet(input_file)
                logger.info(f"Loaded {symbol} @ {tf}: {len(df):,} rows")

                # Filter MTF timeframes to only those > current TF (for enrichment)
                from src.common.timeframes import timeframe_to_minutes
                current_minutes = timeframe_to_minutes(tf)
                valid_mtf = [
                    mtf for mtf in mtf_timeframes
                    if timeframe_to_minutes(mtf) > current_minutes
                ]

                # Initialize FeatureEngineer for this timeframe
                engineer = FeatureEngineer(
                    input_dir=config.clean_data_dir,
                    output_dir=config.features_dir,
                    timeframe=tf,
                    enable_mtf=bool(valid_mtf),  # Enable if any valid MTF timeframes
                    mtf_timeframes=valid_mtf,
                    mtf_include_ohlcv=mtf_include_ohlcv,
                    mtf_include_indicators=mtf_include_indicators,
                    base_timeframe=tf,  # Use current timeframe as base
                    enable_wavelets=enable_wavelets,
                    enable_microstructure=enable_microstructure,
                    enable_volume_features=enable_volume_features,
                    enable_volatility_features=enable_volatility_features,
                )

                output_file = config.features_dir / f"{symbol}_{tf}_features.parquet"
                logger.info(f"Processing {symbol} @ {tf} (symbol-isolated, no cross-correlation)...")

                # Engineer features (each symbol processed independently)
                df_features, feature_info = engineer.engineer_features(df, symbol)

                # Save features
                output_file.parent.mkdir(parents=True, exist_ok=True)
                df_features.to_parquet(output_file, index=False)

                artifacts.append(output_file)

                # Count feature columns
                feature_cols = [c for c in df_features.columns if c not in ohlcv_cols]

                feature_metadata[symbol][tf] = {
                    "total_rows": len(df_features),
                    "feature_count": len(feature_cols),
                    "feature_columns": feature_cols[:20],  # First 20 for reference
                }

                manifest.add_artifact(
                    name=f"features_{symbol}_{tf}",
                    file_path=output_file,
                    stage="feature_engineering",
                    metadata={
                        "symbol": symbol,
                        "timeframe": tf,
                        "feature_count": len(feature_cols),
                        "row_count": len(df_features),
                    },
                )

                logger.info(f"  {symbol} @ {tf}: {len(df_features):,} rows, {len(feature_cols)} features")

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
