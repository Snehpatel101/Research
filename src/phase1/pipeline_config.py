"""
Pipeline Configuration Management System
Handles all configuration for Phase 1 pipeline with validation and persistence
"""
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import json
import logging

# Import HorizonConfig from the dedicated horizon module
# Re-exported here for backward compatibility
from src.common.horizon_config import HorizonConfig

# Import MTF configuration
from src.phase1.stages.mtf.constants import (
    MTFMode,
    DEFAULT_MTF_TIMEFRAMES,
    DEFAULT_MTF_MODE,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class PipelineConfig:
    """Complete configuration for Phase 1 pipeline."""

    # Run identification
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    description: str = "Phase 1 pipeline run"

    # Data parameters
    symbols: List[str] = field(default_factory=lambda: ['MES', 'MGC'])
    start_date: Optional[str] = None  # YYYY-MM-DD format
    end_date: Optional[str] = None    # YYYY-MM-DD format

    # Multi-Timeframe (MTF) configuration
    # target_timeframe: Target resolution for resampling (e.g., '5min', '15min', '30min')
    # Input data is assumed to be 1-minute bars which get resampled to this resolution.
    # Supported: '1min', '5min', '10min', '15min', '20min', '30min', '45min', '60min'
    target_timeframe: str = '5min'

    # Legacy alias - kept for backward compatibility with existing code
    bar_resolution: str = field(default=None)

    # Feature engineering
    feature_set: str = 'full'  # 'full', 'minimal', 'custom'
    sma_periods: List[int] = field(default_factory=lambda: [10, 20, 50, 100, 200])
    ema_periods: List[int] = field(default_factory=lambda: [9, 21, 50])
    atr_periods: List[int] = field(default_factory=lambda: [7, 14, 21])
    rsi_period: int = 14
    macd_params: Dict[str, int] = field(default_factory=lambda: {'fast': 12, 'slow': 26, 'signal': 9})
    bb_period: int = 20
    bb_std: float = 2.0

    # Multi-Timeframe (MTF) configuration
    # mtf_timeframes: List of higher timeframes to compute features for
    # Supported: '5min', '15min', '30min', '1h', '4h', 'daily'
    mtf_timeframes: List[str] = field(default_factory=lambda: DEFAULT_MTF_TIMEFRAMES.copy())
    # mtf_mode: What to generate - 'bars', 'indicators', or 'both'
    # - 'bars': Only OHLCV data at higher timeframes (open_4h, high_4h, etc.)
    # - 'indicators': Only technical indicators at higher timeframes
    # - 'both': Generate both bars and indicators (default)
    mtf_mode: str = field(default_factory=lambda: DEFAULT_MTF_MODE.value)

    # Labeling parameters - Dynamic Horizon Configuration
    # Option 1: Use horizon_config for full control (HorizonConfig instance)
    # Option 2: Use label_horizons for simple usage (legacy compatibility)
    horizon_config: Optional[HorizonConfig] = None
    label_horizons: List[int] = field(default_factory=lambda: [5, 10, 15, 20])
    # Note: Barrier parameters moved to config.py as BARRIER_PARAMS.
    # Use config.get_barrier_params(symbol, horizon) for symbol-specific values.
    max_bars_ahead: int = 50

    # Auto-scale purge/embargo with horizon
    # When True, purge_bars and embargo_bars are computed from max horizon
    auto_scale_purge_embargo: bool = True

    # Split parameters
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    # PURGE_BARS: Must equal max(max_bars) across horizons to prevent leakage.
    # H20 uses max_bars=60, so purge_bars must be at least 60.
    purge_bars: int = 60  # Default, overridden if auto_scale_purge_embargo=True
    # EMBARGO_BARS: Buffer for serial correlation in features.
    # 1440 bars = 5 days for 5-min data (288 bars/day * 5 days).
    # Must match src/config/splits.py EMBARGO_BARS for consistency.
    embargo_bars: int = 1440  # Default, overridden if auto_scale_purge_embargo=True

    # Genetic Algorithm settings (for future Phase 2)
    ga_population_size: int = 50
    ga_generations: int = 100
    ga_crossover_rate: float = 0.8
    ga_mutation_rate: float = 0.1
    ga_elite_size: int = 5

    # Processing options
    use_synthetic_data: bool = False
    n_jobs: int = -1  # -1 for all cores
    random_seed: int = 42

    # Paths (auto-generated from run_id)
    project_root: Path = field(default=None)

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Import here to avoid circular imports
        from src.phase1.config import (
            SUPPORTED_TIMEFRAMES,
            validate_timeframe,
            auto_scale_purge_embargo,
            validate_horizons,
            SUPPORTED_HORIZONS,
            validate_feature_set_config,
        )

        # Set project_root if not provided
        if self.project_root is None:
            self.project_root = Path(__file__).parent.parent.resolve()

        # Convert string paths to Path objects
        if isinstance(self.project_root, str):
            self.project_root = Path(self.project_root)

        # Handle bar_resolution backward compatibility
        # If bar_resolution is set but target_timeframe uses default, sync them
        if self.bar_resolution is not None and self.bar_resolution != self.target_timeframe:
            # bar_resolution was explicitly set, use it as the source of truth
            self.target_timeframe = self.bar_resolution
        elif self.bar_resolution is None:
            # bar_resolution not set, sync from target_timeframe
            self.bar_resolution = self.target_timeframe

        # Validate target_timeframe
        validate_timeframe(self.target_timeframe)

        feature_set_issues = validate_feature_set_config(self.feature_set)
        if feature_set_issues:
            raise ValueError(f"Feature set validation failed: {feature_set_issues}")

        # Validate MTF configuration
        from src.phase1.stages.mtf.constants import MTF_TIMEFRAMES
        valid_mtf_modes = ['bars', 'indicators', 'both']
        if self.mtf_mode not in valid_mtf_modes:
            raise ValueError(
                f"mtf_mode must be one of {valid_mtf_modes}, got '{self.mtf_mode}'"
            )
        for tf in self.mtf_timeframes:
            if tf not in MTF_TIMEFRAMES:
                raise ValueError(
                    f"Unsupported MTF timeframe: '{tf}'. "
                    f"Supported: {list(MTF_TIMEFRAMES.keys())}"
                )

        # Handle horizon configuration
        # Priority: horizon_config > label_horizons
        if self.horizon_config is not None:
            # Sync label_horizons from horizon_config
            self.label_horizons = self.horizon_config.horizons
            # Validate horizon_config
            horizon_issues = self.horizon_config.validate()
            if horizon_issues:
                raise ValueError(f"HorizonConfig validation failed: {horizon_issues}")
        else:
            # Validate label_horizons directly
            if not self.label_horizons:
                raise ValueError("At least one label horizon must be specified")
            for h in self.label_horizons:
                if h not in SUPPORTED_HORIZONS:
                    logger.warning(
                        f"Horizon {h} not in SUPPORTED_HORIZONS {SUPPORTED_HORIZONS}. "
                        f"Auto-generated barrier params will be used."
                    )

        # Auto-scale purge and embargo bars based on horizons
        if self.auto_scale_purge_embargo:
            self.purge_bars, self.embargo_bars = auto_scale_purge_embargo(
                self.label_horizons
            )
            logger.debug(
                f"Auto-scaled purge/embargo for horizons {self.label_horizons}: "
                f"purge={self.purge_bars}, embargo={self.embargo_bars}"
            )

        # Validate ratios sum to 1.0
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not (0.99 <= total_ratio <= 1.01):
            raise ValueError(f"Train/val/test ratios must sum to 1.0, got {total_ratio}")

        # Validate date format if provided
        if self.start_date:
            try:
                datetime.strptime(self.start_date, "%Y-%m-%d")
            except ValueError:
                raise ValueError(f"start_date must be in YYYY-MM-DD format, got {self.start_date}")

        if self.end_date:
            try:
                datetime.strptime(self.end_date, "%Y-%m-%d")
            except ValueError:
                raise ValueError(f"end_date must be in YYYY-MM-DD format, got {self.end_date}")

        # Validate symbols
        if not self.symbols:
            raise ValueError("At least one symbol must be specified")

    @property
    def data_dir(self) -> Path:
        """Data directory for this run."""
        return self.project_root / "data"

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def clean_data_dir(self) -> Path:
        return self.data_dir / "clean"

    @property
    def features_dir(self) -> Path:
        return self.data_dir / "features"

    @property
    def final_data_dir(self) -> Path:
        return self.data_dir / "final"

    @property
    def splits_dir(self) -> Path:
        return self.data_dir / "splits"

    @property
    def run_dir(self) -> Path:
        """Directory for this specific run."""
        return self.project_root / "runs" / self.run_id

    @property
    def run_config_dir(self) -> Path:
        """Config directory for this run."""
        return self.run_dir / "config"

    @property
    def run_logs_dir(self) -> Path:
        """Logs directory for this run."""
        return self.run_dir / "logs"

    @property
    def run_artifacts_dir(self) -> Path:
        """Artifacts directory for this run."""
        return self.run_dir / "artifacts"

    @property
    def results_dir(self) -> Path:
        """Results directory."""
        return self.project_root / "results"

    def create_directories(self):
        """Create all required directories for this run."""
        directories = [
            self.raw_data_dir,
            self.clean_data_dir,
            self.features_dir,
            self.final_data_dir,
            self.splits_dir,
            self.run_dir,
            self.run_config_dir,
            self.run_logs_dir,
            self.run_artifacts_dir,
            self.results_dir,
        ]

        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {dir_path}")

    def save_config(self, path: Optional[Path] = None) -> Path:
        """
        Save configuration to JSON file.

        Args:
            path: Path to save config. If None, saves to run_config_dir/config.json

        Returns:
            Path where config was saved
        """
        if path is None:
            self.create_directories()
            path = self.run_config_dir / "config.json"

        # Convert to dict and handle Path objects
        config_dict = asdict(self)
        config_dict['project_root'] = str(self.project_root)

        # Add metadata
        config_dict['_metadata'] = {
            'created_at': datetime.now().isoformat(),
            'config_version': '1.0',
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"Configuration saved to {path}")
        return path

    @classmethod
    def load_config(cls, path: Path) -> 'PipelineConfig':
        """
        Load configuration from JSON file.

        Args:
            path: Path to config JSON file

        Returns:
            PipelineConfig instance
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, 'r') as f:
            config_dict = json.load(f)

        # Remove metadata if present
        config_dict.pop('_metadata', None)

        # Convert project_root back to Path
        if 'project_root' in config_dict:
            config_dict['project_root'] = Path(config_dict['project_root'])

        logger.info(f"Configuration loaded from {path}")
        return cls(**config_dict)

    @classmethod
    def load_from_run_id(cls, run_id: str, project_root: Optional[Path] = None) -> 'PipelineConfig':
        """
        Load configuration from a run ID.

        Args:
            run_id: Run identifier
            project_root: Project root path (defaults to /home/user/Research)

        Returns:
            PipelineConfig instance
        """
        if project_root is None:
            project_root = Path(__file__).parent.parent.resolve()
        else:
            project_root = Path(project_root)

        config_path = project_root / "runs" / run_id / "config" / "config.json"
        return cls.load_config(config_path)

    def validate(self) -> List[str]:
        """
        Validate configuration and return list of issues.

        Returns:
            List of validation error messages (empty if valid)
        """
        from src.phase1.config import SUPPORTED_TIMEFRAMES, validate_feature_set_config

        issues = []

        # Check timeframe
        if self.target_timeframe not in SUPPORTED_TIMEFRAMES:
            issues.append(
                f"target_timeframe '{self.target_timeframe}' is not supported. "
                f"Supported: {SUPPORTED_TIMEFRAMES}"
            )

        # Check ratios
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not (0.99 <= total_ratio <= 1.01):
            issues.append(f"Train/val/test ratios must sum to 1.0, got {total_ratio}")

        if not (0 < self.train_ratio < 1):
            issues.append(f"train_ratio must be between 0 and 1, got {self.train_ratio}")

        if not (0 < self.val_ratio < 1):
            issues.append(f"val_ratio must be between 0 and 1, got {self.val_ratio}")

        if not (0 < self.test_ratio < 1):
            issues.append(f"test_ratio must be between 0 and 1, got {self.test_ratio}")

        # Check symbols
        if not self.symbols:
            issues.append("At least one symbol must be specified")

        # Check label horizons
        if not self.label_horizons:
            issues.append("At least one label horizon must be specified")

        for horizon in self.label_horizons:
            if horizon < 1:
                issues.append(f"Label horizon must be >= 1, got {horizon}")

        # Note: Barrier parameters now validated in config.py

        if self.max_bars_ahead < max(self.label_horizons):
            issues.append(f"max_bars_ahead ({self.max_bars_ahead}) must be >= max horizon ({max(self.label_horizons)})")

        # Check purge/embargo
        if self.purge_bars < 0:
            issues.append(f"purge_bars must be >= 0, got {self.purge_bars}")

        if self.embargo_bars < 0:
            issues.append(f"embargo_bars must be >= 0, got {self.embargo_bars}")

        # Check GA parameters
        if self.ga_population_size < 2:
            issues.append(f"ga_population_size must be >= 2, got {self.ga_population_size}")

        if self.ga_generations < 1:
            issues.append(f"ga_generations must be >= 1, got {self.ga_generations}")

        if not (0 <= self.ga_crossover_rate <= 1):
            issues.append(f"ga_crossover_rate must be between 0 and 1, got {self.ga_crossover_rate}")

        if not (0 <= self.ga_mutation_rate <= 1):
            issues.append(f"ga_mutation_rate must be between 0 and 1, got {self.ga_mutation_rate}")

        if self.ga_elite_size >= self.ga_population_size:
            issues.append(f"ga_elite_size ({self.ga_elite_size}) must be < ga_population_size ({self.ga_population_size})")

        # Check feature parameters
        if not self.sma_periods:
            issues.append("At least one SMA period must be specified")

        if not self.ema_periods:
            issues.append("At least one EMA period must be specified")

        if not self.atr_periods:
            issues.append("At least one ATR period must be specified")

        # Check MTF parameters
        from src.phase1.stages.mtf.constants import MTF_TIMEFRAMES
        valid_mtf_modes = ['bars', 'indicators', 'both']
        if self.mtf_mode not in valid_mtf_modes:
            issues.append(
                f"mtf_mode must be one of {valid_mtf_modes}, got '{self.mtf_mode}'"
            )
        for tf in self.mtf_timeframes:
            if tf not in MTF_TIMEFRAMES:
                issues.append(
                    f"Unsupported MTF timeframe: '{tf}'. "
                    f"Supported: {list(MTF_TIMEFRAMES.keys())}"
                )

        if self.rsi_period < 2:
            issues.append(f"rsi_period must be >= 2, got {self.rsi_period}")

        issues.extend(validate_feature_set_config(self.feature_set))

        return issues

    def summary(self) -> str:
        """Generate a human-readable summary of the configuration."""
        return f"""
Pipeline Configuration Summary
==============================
Run ID: {self.run_id}
Description: {self.description}

Data Parameters:
  - Symbols: {', '.join(self.symbols)}
  - Date Range: {self.start_date or 'N/A'} to {self.end_date or 'N/A'}
  - Target Timeframe: {self.target_timeframe}
  - Synthetic Data: {self.use_synthetic_data}

Features:
  - Feature Set: {self.feature_set}
  - SMA Periods: {self.sma_periods}
  - EMA Periods: {self.ema_periods}
  - ATR Periods: {self.atr_periods}
  - RSI Period: {self.rsi_period}

Multi-Timeframe (MTF):
  - Timeframes: {', '.join(self.mtf_timeframes)}
  - Mode: {self.mtf_mode}

Labeling:
  - Horizons: {self.label_horizons}
  - Barrier Params: config.BARRIER_PARAMS (symbol-specific)
  - Max Bars Ahead: {self.max_bars_ahead}

Splits:
  - Train: {self.train_ratio:.1%}
  - Validation: {self.val_ratio:.1%}
  - Test: {self.test_ratio:.1%}
  - Purge Bars: {self.purge_bars}
  - Embargo Bars: {self.embargo_bars}

GA Settings (Phase 2):
  - Population: {self.ga_population_size}
  - Generations: {self.ga_generations}
  - Crossover Rate: {self.ga_crossover_rate}
  - Mutation Rate: {self.ga_mutation_rate}
  - Elite Size: {self.ga_elite_size}

Paths:
  - Project Root: {self.project_root}
  - Run Directory: {self.run_dir}
  - Data Directory: {self.data_dir}
"""

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = asdict(self)
        config_dict['project_root'] = str(self.project_root)
        return config_dict


def create_default_config(
    symbols: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    run_id: Optional[str] = None,
    **kwargs
) -> PipelineConfig:
    """
    Create a default configuration with optional overrides.

    Args:
        symbols: List of symbols to trade
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        run_id: Run identifier (auto-generated if None)
        **kwargs: Additional parameters to override defaults

    Returns:
        PipelineConfig instance
    """
    config_kwargs = {}

    if symbols is not None:
        config_kwargs['symbols'] = symbols

    if start_date is not None:
        config_kwargs['start_date'] = start_date

    if end_date is not None:
        config_kwargs['end_date'] = end_date

    if run_id is not None:
        config_kwargs['run_id'] = run_id

    # Merge with additional kwargs
    config_kwargs.update(kwargs)

    return PipelineConfig(**config_kwargs)


if __name__ == "__main__":
    # Example usage - configure logging for standalone execution
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    # Create default config
    config = create_default_config(
        symbols=['MES', 'MGC', 'MNQ'],
        start_date='2020-01-01',
        end_date='2024-12-31',
        description='Test run with 3 symbols'
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
