"""
Stage 3: Feature Engineering.

Generates technical indicators and derived features from cleaned OHLCV data.
"""
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ..utils import StageResult, StageStatus, create_stage_result, create_failed_result

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
        from feature_engineering import main as generate_features
        generate_features()

        artifacts = []
        feature_metadata = {}

        for symbol in config.symbols:
            file_path = config.features_dir / f"{symbol}_5m_features.parquet"
            if file_path.exists():
                artifacts.append(file_path)

                # Read file to get feature count
                import pandas as pd
                df = pd.read_parquet(file_path)

                # Count feature columns
                ohlcv_cols = {'datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume'}
                feature_cols = [c for c in df.columns if c not in ohlcv_cols]

                feature_metadata[symbol] = {
                    'total_rows': len(df),
                    'feature_count': len(feature_cols),
                    'feature_columns': feature_cols[:20]  # First 20 for reference
                }

                manifest.add_artifact(
                    name=f"features_{symbol}",
                    file_path=file_path,
                    stage="feature_engineering",
                    metadata={
                        'symbol': symbol,
                        'feature_count': len(feature_cols),
                        'row_count': len(df)
                    }
                )

                logger.info(
                    f"  {symbol}: {len(df):,} rows, {len(feature_cols)} features"
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
