"""
Stage 3: Feature Engineering.

Orchestration logic for Stage 3 of the pipeline.
Generates technical indicators and derived features from cleaned OHLCV data.

Uses the modular feature engineering implementation from stages.features.
"""
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from pipeline_config import PipelineConfig
    from manifest import ArtifactManifest

logger = logging.getLogger(__name__)

# Import from local modules
from .engineer import FeatureEngineer

# StageResult imports - adjust path based on pipeline structure
try:
    from src.pipeline.utils import StageResult, StageStatus, create_stage_result, create_failed_result
except ImportError:
    # Fallback for different import paths
    from pipeline.utils import StageResult, StageStatus, create_stage_result, create_failed_result


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
        from src.phase1.config.features import get_mtf_config, CROSS_ASSET_FEATURES

        target_timeframe = config.target_timeframe
        mtf_config = get_mtf_config()

        # Initialize FeatureEngineer from modular implementation
        engineer = FeatureEngineer(
            input_dir=config.clean_data_dir,
            output_dir=config.features_dir,
            timeframe=target_timeframe,
            enable_mtf=mtf_config.get('enabled', True),
            mtf_timeframes=mtf_config.get('mtf_timeframes'),
            mtf_include_ohlcv=mtf_config.get('include_ohlcv', True),
            mtf_include_indicators=mtf_config.get('include_indicators', True),
            base_timeframe=mtf_config.get('base_timeframe', target_timeframe),
        )

        # Load all symbol data first
        symbol_data = {}
        for symbol in config.symbols:
            input_file = config.clean_data_dir / f"{symbol}_{target_timeframe}_clean.parquet"
            if input_file.exists():
                symbol_data[symbol] = pd.read_parquet(input_file)
                logger.info(f"Loaded {symbol}: {len(symbol_data[symbol]):,} rows")
            else:
                logger.warning(f"No cleaned data found for {symbol}")

        # Prepare cross-asset data if both MES and MGC are available
        cross_asset_data = None
        has_mes = 'MES' in symbol_data
        has_mgc = 'MGC' in symbol_data

        if has_mes and has_mgc and CROSS_ASSET_FEATURES.get('enabled', True):
            logger.info("Aligning MES and MGC data for cross-asset features...")
            mes_df = symbol_data['MES'].copy().set_index('datetime')
            mgc_df = symbol_data['MGC'].copy().set_index('datetime')

            # Get common timestamps
            common_idx = mes_df.index.intersection(mgc_df.index)
            logger.info(f"Common timestamps: {len(common_idx):,}")

            if len(common_idx) > 0:
                # Align data to common timestamps
                mes_aligned = mes_df.loc[common_idx].reset_index()
                mgc_aligned = mgc_df.loc[common_idx].reset_index()

                # Update symbol_data with aligned data
                symbol_data['MES'] = mes_aligned
                symbol_data['MGC'] = mgc_aligned

                # Prepare cross-asset data arrays
                cross_asset_data = {
                    'mes_close': mes_aligned['close'].values,
                    'mgc_close': mgc_aligned['close'].values
                }
                logger.info("Cross-asset data prepared for feature computation")
        elif has_mes and has_mgc:
            logger.info("Skipping cross-asset alignment (disabled in config)")

        # Process each symbol
        artifacts = []
        feature_metadata = {}

        for symbol in symbol_data.keys():
            df = symbol_data[symbol]
            output_file = config.features_dir / f"{symbol}_{target_timeframe}_features.parquet"

            logger.info(f"Processing {symbol}...")

            # Engineer features with cross-asset data
            df_features, feature_info = engineer.engineer_features(
                df, symbol, cross_asset_data=cross_asset_data
            )

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
