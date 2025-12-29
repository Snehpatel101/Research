"""
Stage 3: Feature Engineering.

Orchestration logic for Stage 3 of the pipeline.
Generates technical indicators and derived features from cleaned OHLCV data.

Uses the modular feature engineering implementation from stages.features.
"""
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
    config: 'PipelineConfig',
    manifest: 'ArtifactManifest'
) -> 'StageResult':
    """
    Stage 3: Feature Engineering.

    Generates 50+ technical features including:
    - Moving averages (SMA, EMA)
    - Momentum indicators (RSI, MACD, Stochastic)
    - Volatility indicators (ATR, Bollinger Bands)
    - Volume-based indicators
    - Price patterns and derived features
    - Temporal and regime indicators
    - Cross-asset features (MES-MGC correlation, beta, etc.)

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
        target_timeframe = config.target_timeframe

        # Determine MTF include flags based on config.mtf_mode
        # mtf_mode: 'bars' -> only OHLCV, 'indicators' -> only indicators, 'both' -> both
        mtf_mode = getattr(config, 'mtf_mode', 'both')
        mtf_include_ohlcv = mtf_mode in ('bars', 'both')
        mtf_include_indicators = mtf_mode in ('indicators', 'both')

        # Use MTF timeframes from PipelineConfig (respects CLI overrides)
        mtf_timeframes = getattr(config, 'mtf_timeframes', ['15min', '60min'])

        # Initialize FeatureEngineer from modular implementation
        # MTF settings come from PipelineConfig, not global MTF_CONFIG
        engineer = FeatureEngineer(
            input_dir=config.clean_data_dir,
            output_dir=config.features_dir,
            timeframe=target_timeframe,
            enable_mtf=bool(mtf_timeframes),  # Enable if any MTF timeframes specified
            mtf_timeframes=mtf_timeframes,
            mtf_include_ohlcv=mtf_include_ohlcv,
            mtf_include_indicators=mtf_include_indicators,
            base_timeframe=target_timeframe,  # Use run's target timeframe, not hardcoded '5min'
        )

        # Process each symbol independently (no cross-symbol correlation)
        artifacts = []
        feature_metadata = {}

        for symbol in config.symbols:
            input_file = config.clean_data_dir / f"{symbol}_{target_timeframe}_clean.parquet"
            if not input_file.exists():
                logger.warning(f"No cleaned data found for {symbol}")
                continue

            df = pd.read_parquet(input_file)
            logger.info(f"Loaded {symbol}: {len(df):,} rows")

            output_file = config.features_dir / f"{symbol}_{target_timeframe}_features.parquet"
            logger.info(f"Processing {symbol} (symbol-isolated, no cross-correlation)...")

            # Engineer features (each symbol processed independently)
            df_features, feature_info = engineer.engineer_features(df, symbol)

            # Save features
            output_file.parent.mkdir(parents=True, exist_ok=True)
            df_features.to_parquet(output_file, index=False)

            artifacts.append(output_file)

            # Count feature columns
            ohlcv_cols = {
                'datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume',
                'timeframe', 'session_id', 'missing_bar', 'roll_event', 'roll_window', 'filled'
            }
            feature_cols = [c for c in df_features.columns if c not in ohlcv_cols]

            feature_metadata[symbol] = {
                'total_rows': len(df_features),
                'feature_count': len(feature_cols),
                'feature_columns': feature_cols[:20]  # First 20 for reference
            }

            manifest.add_artifact(
                name=f"features_{symbol}",
                file_path=output_file,
                stage="feature_engineering",
                metadata={
                    'symbol': symbol,
                    'feature_count': len(feature_cols),
                    'row_count': len(df_features)
                }
            )

            logger.info(
                f"  {symbol}: {len(df_features):,} rows, {len(feature_cols)} features"
            )

        return create_stage_result(
            stage_name="feature_engineering",
            start_time=start_time,
            artifacts=artifacts,
            metadata={'feature_results': feature_metadata}
        )

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        logger.error(traceback.format_exc())
        return create_failed_result(
            stage_name="feature_engineering",
            start_time=start_time,
            error=str(e)
        )
