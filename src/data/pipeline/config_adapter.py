"""Adapter for converting core.PipelineConfig to pipeline.DataConfig."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.config import PipelineConfig
    from src.data.pipeline.data_config import DataConfig


def to_data_config(config: PipelineConfig) -> DataConfig:
    """Convert core PipelineConfig to pipeline DataConfig for PipelineRunner."""
    import secrets
    from datetime import datetime

    from src.data.pipeline.data_config import DataConfig

    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{secrets.token_hex(2)}"

    feature_toggles = {}
    if isinstance(config.feature_families, list):
        for family in ["wavelets", "microstructure", "volume", "volatility"]:
            feature_toggles[family] = family in config.feature_families

    barrier_overrides = None
    if config.upper_mult != 2.0 or config.lower_mult != 2.0:
        barrier_overrides = {
            "k_up": config.upper_mult,
            "k_down": config.lower_mult,
        }

    return DataConfig(
        run_id=run_id,
        description=f"MLFactory run for {config.symbol}",
        symbols=[config.symbol],
        target_timeframe="5min",
        mtf_timeframes=config.mtf_timeframes,
        mtf_mode="aligned",
        label_horizons=config.horizons,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        purge_bars=config.purge_bars,
        embargo_bars=config.embargo_bars,
        scaler_type="standard",
        random_seed=config.random_state,
        feature_toggles=feature_toggles if feature_toggles else None,
        barrier_overrides=barrier_overrides,
        project_root=Path(config.output_dir).parent,
    )
