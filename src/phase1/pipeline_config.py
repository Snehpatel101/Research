"""
Pipeline Configuration Management System
Handles all configuration for Phase 1 pipeline with validation and persistence.
"""
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Import HorizonConfig and active horizons from the dedicated horizon module
# Re-exported here for backward compatibility
from src.common.horizon_config import ACTIVE_HORIZONS, HorizonConfig
from src.phase1.config.pipeline_defaults import create_default_config

# Import extracted modules
from src.phase1.config.pipeline_paths import PipelinePathMixin
from src.phase1.config.pipeline_persistence import PipelinePersistenceMixin
from src.phase1.config.pipeline_summary import generate_pipeline_summary
from src.phase1.config.pipeline_validation import validate_pipeline_config

# Import MTF configuration
from src.phase1.stages.mtf.constants import (
    DEFAULT_MTF_MODE,
    DEFAULT_MTF_TIMEFRAMES,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class PipelineConfig(PipelinePathMixin, PipelinePersistenceMixin):
    """Complete configuration for Phase 1 pipeline."""

    # Run identification
    # Format: {timestamp_with_ms}_{random_suffix} for collision prevention
    # Example: 20251228_143025_789456_a3f9
    run_id: str = field(
        default_factory=lambda: (
            lambda: f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{__import__('secrets').token_hex(2)}"
        )()
    )
    description: str = "Phase 1 pipeline run"

    # Data parameters
    # Symbols to process. Each symbol is processed in complete isolation.
    symbols: list[str] = field(default_factory=list)
    start_date: str | None = None  # YYYY-MM-DD format
    end_date: str | None = None    # YYYY-MM-DD format

    # Timeframe configuration
    target_timeframe: str = '5min'
    bar_resolution: str = field(default=None)  # Legacy alias

    # Feature engineering
    feature_set: str = 'full'  # 'full', 'minimal', 'custom'
    sma_periods: list[int] = field(default_factory=lambda: [10, 20, 50, 100, 200])
    ema_periods: list[int] = field(default_factory=lambda: [9, 21, 50])
    atr_periods: list[int] = field(default_factory=lambda: [7, 14, 21])
    rsi_period: int = 14
    macd_params: dict[str, int] = field(default_factory=lambda: {'fast': 12, 'slow': 26, 'signal': 9})
    bb_period: int = 20
    bb_std: float = 2.0

    # Multi-Timeframe (MTF) configuration
    mtf_timeframes: list[str] = field(default_factory=lambda: DEFAULT_MTF_TIMEFRAMES.copy())
    mtf_mode: str = field(default_factory=lambda: DEFAULT_MTF_MODE.value)

    # Labeling parameters - Dynamic Horizon Configuration
    horizon_config: HorizonConfig | None = None
    label_horizons: list[int] = field(default_factory=lambda: list(ACTIVE_HORIZONS))
    max_bars_ahead: int = 50
    auto_scale_purge_embargo: bool = True

    # Split parameters
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    purge_bars: int = 60
    embargo_bars: int = 1440

    # Genetic Algorithm / Optuna settings (for Phase 2)
    ga_population_size: int = 50
    ga_generations: int = 100
    ga_crossover_rate: float = 0.8
    ga_mutation_rate: float = 0.1
    ga_elite_size: int = 5
    # CRITICAL: Safe mode prevents test data leakage in barrier optimization
    # When True, optimization only uses training data (first train_ratio% of data)
    # Set to False ONLY for research purposes when you understand the implications
    ga_safe_mode: bool = True

    # Processing options
    n_jobs: int = -1  # -1 for all cores
    random_seed: int = 42
    allow_batch_symbols: bool = False

    # Optional configurations
    feature_toggles: dict[str, bool] | None = None
    barrier_overrides: dict[str, float] | None = None
    scaler_type: str = 'robust'
    model_config: dict[str, Any] | None = None

    # Paths (auto-generated from run_id)
    project_root: Path = field(default=None)

    def __post_init__(self):
        """Validate configuration after initialization."""
        from src.phase1.config import (
            SUPPORTED_HORIZONS,
            auto_scale_purge_embargo,
            validate_feature_set_config,
            validate_timeframe,
        )
        from src.phase1.stages.mtf.constants import MTF_TIMEFRAMES

        # Set project_root if not provided
        if self.project_root is None:
            # __file__ is src/phase1/pipeline_config.py, so .parent.parent.parent = project root
            self.project_root = Path(__file__).parent.parent.parent.resolve()
        if isinstance(self.project_root, str):
            self.project_root = Path(self.project_root)

        # Handle bar_resolution backward compatibility
        if self.bar_resolution is not None and self.bar_resolution != self.target_timeframe:
            self.target_timeframe = self.bar_resolution
        elif self.bar_resolution is None:
            self.bar_resolution = self.target_timeframe

        # Validate target_timeframe
        validate_timeframe(self.target_timeframe)

        # Validate feature set
        feature_set_issues = validate_feature_set_config(self.feature_set)
        if feature_set_issues:
            raise ValueError(f"Feature set validation failed: {feature_set_issues}")

        # Validate MTF configuration
        valid_mtf_modes = ['bars', 'indicators', 'both']
        if self.mtf_mode not in valid_mtf_modes:
            raise ValueError(f"mtf_mode must be one of {valid_mtf_modes}, got '{self.mtf_mode}'")
        for tf in self.mtf_timeframes:
            if tf not in MTF_TIMEFRAMES:
                raise ValueError(f"Unsupported MTF timeframe: '{tf}'. Supported: {list(MTF_TIMEFRAMES.keys())}")

        # Handle horizon configuration
        if self.horizon_config is not None:
            self.label_horizons = self.horizon_config.horizons
            horizon_issues = self.horizon_config.validate()
            if horizon_issues:
                raise ValueError(f"HorizonConfig validation failed: {horizon_issues}")
        else:
            if not self.label_horizons:
                raise ValueError("At least one label horizon must be specified")
            for h in self.label_horizons:
                if h not in SUPPORTED_HORIZONS:
                    logger.warning(f"Horizon {h} not in SUPPORTED_HORIZONS {SUPPORTED_HORIZONS}.")

        # Auto-scale purge and embargo bars based on horizons
        # IMPORTANT: Pass target_timeframe to ensure embargo scales correctly
        # with bar resolution (e.g., 15min bars need fewer bars than 5min for same time buffer)
        if self.auto_scale_purge_embargo:
            self.purge_bars, self.embargo_bars = auto_scale_purge_embargo(
                self.label_horizons,
                timeframe=self.target_timeframe,  # Timeframe-aware embargo calculation
            )
            logger.debug(
                f"Auto-scaled purge={self.purge_bars}, embargo={self.embargo_bars} "
                f"(timeframe={self.target_timeframe})"
            )

        # Validate ratios sum to 1.0
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not (0.99 <= total_ratio <= 1.01):
            raise ValueError(f"Train/val/test ratios must sum to 1.0, got {total_ratio}")

        # Validate date format if provided
        for name, date_val in [("start_date", self.start_date), ("end_date", self.end_date)]:
            if date_val:
                try:
                    datetime.strptime(date_val, "%Y-%m-%d")
                except ValueError:
                    raise ValueError(f"{name} must be in YYYY-MM-DD format, got {date_val}")

        # Validate symbols
        if not self.symbols:
            raise ValueError("At least one symbol must be specified. Use --symbols MES or symbols=['MES'].")

        # Enforce single-symbol runs by default
        if len(self.symbols) > 1 and not self.allow_batch_symbols:
            raise ValueError(
                f"Batch processing of multiple symbols requires explicit opt-in. "
                f"Got {len(self.symbols)} symbols: {self.symbols}. "
                f"Use --batch-symbols flag or set allow_batch_symbols=True."
            )

    def validate(self) -> list[str]:
        """Validate configuration and return list of issues."""
        return validate_pipeline_config(self)

    def summary(self) -> str:
        """Generate a human-readable summary of the configuration."""
        return generate_pipeline_summary(self)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = asdict(self)
        config_dict['project_root'] = str(self.project_root)
        return config_dict


# Re-export create_default_config for backward compatibility
__all__ = ['PipelineConfig', 'create_default_config', 'HorizonConfig']


if __name__ == "__main__":
    # Example usage - configure logging for standalone execution
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    # Create config for single symbol (default behavior)
    config = create_default_config(
        symbols=['MES'],
        start_date='2020-01-01',
        end_date='2024-12-31',
        description='Single symbol run'
    )

    print(config.summary())

    # Validate
    issues = config.validate()
    if issues:
        print("\nValidation Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nConfiguration is valid!")

    # Save config
    config.create_directories()
    config_path = config.save_config()
    print(f"\nSaved to: {config_path}")

    # Load config back
    loaded_config = PipelineConfig.load_config(config_path)
    print(f"\nLoaded run_id: {loaded_config.run_id}")
