"""
Stage 3: Feature Engineering.

Generates technical indicators and derived features from cleaned OHLCV data.

Uses the modular feature engineering implementation from stages.features.
"""
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from ..utils import StageResult, StageStatus, create_stage_result, create_failed_result
from src.stages.stage3_features import FeatureEngineer

if TYPE_CHECKING:
    from pipeline_config import PipelineConfig
    from manifest import ArtifactManifest

logger = logging.getLogger(__name__)


def run_feature_engineering(
    config: 'PipelineConfig',
    manifest: 'ArtifactManifest'
) -> StageResult:
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
        # Initialize FeatureEngineer from modular implementation
        engineer = FeatureEngineer(
            input_dir=config.clean_data_dir,
            output_dir=config.features_dir,
            timeframe='5min'
        )

        # Process each symbol
        artifacts = []
        feature_metadata = {}

        for symbol in config.symbols:
            input_file = config.clean_data_dir / f"{symbol}_5m_clean.parquet"
            output_file = config.features_dir / f"{symbol}_5m_features.parquet"

            if not input_file.exists():
                logger.warning(f"No cleaned data found for {symbol}")
                continue

            logger.info(f"Processing {symbol}...")

            # Read cleaned data
            df = pd.read_parquet(input_file)

            # Engineer features
            df_features, feature_info = engineer.engineer_features(df, symbol)

            # Save features
            output_file.parent.mkdir(parents=True, exist_ok=True)
            df_features.to_parquet(output_file, index=False)

            artifacts.append(output_file)

            # Count feature columns
            ohlcv_cols = {'datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume'}
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
