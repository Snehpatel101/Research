"""
PreprocessingGraph - Serializable preprocessing pipeline for train/serve parity.

This module ensures the exact same preprocessing applied during training
is applied at inference time, maintaining train/serve parity.

The preprocessing graph captures:
1. Data cleaning configuration (resampling, gap handling, outliers)
2. Feature engineering configuration (indicator periods, wavelets, MTF)
3. Regime detection configuration
4. Scaling configuration (per-column parameters from training)

Usage:
    # During training - capture the preprocessing graph
    graph = PreprocessingGraph.from_pipeline_config(pipeline_config)
    graph.save(output_path / "preprocessing_graph.json")

    # During inference - apply same preprocessing
    graph = PreprocessingGraph.load(model_bundle_path / "preprocessing_graph.json")
    features = graph.transform(raw_ohlcv_df)
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Version for backward compatibility
PREPROCESSING_GRAPH_VERSION = "1.0.0"


@dataclass
class CleaningConfig:
    """Data cleaning configuration."""

    source_timeframe: str = "1min"
    target_timeframe: str = "5min"
    gap_fill_method: str = "forward"
    max_gap_fill_minutes: int = 5
    outlier_method: str = "atr"
    atr_threshold: float = 5.0
    zscore_threshold: float = 5.0
    iqr_multiplier: float = 3.0
    calendar_aware: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CleaningConfig:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class MTFConfig:
    """Multi-timeframe configuration."""

    enabled: bool = True
    base_timeframe: str = "5min"
    mtf_timeframes: list[str] = field(default_factory=lambda: ["15min", "60min"])
    mode: str = "both"  # bars, indicators, both
    include_ohlcv: bool = True
    include_indicators: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MTFConfig:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class WaveletConfig:
    """Wavelet decomposition configuration."""

    enabled: bool = True
    wavelet_type: str = "db4"
    level: int = 3
    window: int = 64
    include_volume: bool = True
    include_energy: bool = True
    include_volatility: bool = True
    include_trend: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WaveletConfig:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class IndicatorConfig:
    """Feature indicator configuration with period settings."""

    # Period scaling
    scale_periods: bool = True
    base_timeframe: str = "5min"

    # Indicator periods (can be overridden)
    sma_periods: list[int] = field(default_factory=lambda: [10, 20, 50])
    ema_periods: list[int] = field(default_factory=lambda: [10, 20, 50])
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    stochastic_k: int = 14
    stochastic_d: int = 3
    williams_r_period: int = 14
    roc_periods: list[int] = field(default_factory=lambda: [10, 20])
    cci_period: int = 20
    mfi_period: int = 14
    atr_periods: list[int] = field(default_factory=lambda: [14, 20])
    bollinger_period: int = 20
    keltner_period: int = 20
    hvol_periods: list[int] = field(default_factory=lambda: [10, 20, 50])
    parkinson_period: int = 20
    garman_klass_period: int = 20
    rs_vol_period: int = 20
    yz_vol_period: int = 20
    volume_sma_period: int = 20
    adx_period: int = 14
    supertrend_period: int = 10

    # NaN handling
    nan_threshold: float = 0.9

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IndicatorConfig:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class RegimeConfig:
    """Regime detection configuration."""

    enabled: bool = True
    regime_types: list[str] = field(
        default_factory=lambda: ["volatility", "trend", "structure"]
    )
    # Volatility regime params
    vol_lookback: int = 20
    vol_threshold_low: float = 0.5
    vol_threshold_high: float = 1.5
    # Trend regime params
    trend_short_period: int = 10
    trend_long_period: int = 50
    # Structure regime params
    structure_lookback: int = 20

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegimeConfig:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ScalingConfig:
    """Scaling configuration captured from training."""

    scaler_type: str = "robust"
    clip_outliers: bool = True
    clip_range: tuple[float, float] = (-5.0, 5.0)
    robust_quantile_range: tuple[float, float] = (25.0, 75.0)
    apply_log_to_price_volume: bool = True
    feature_columns: list[str] = field(default_factory=list)
    # Per-column scaling parameters (median, IQR, etc.)
    scaling_params: dict[str, dict[str, float]] = field(default_factory=dict)
    # Scaler state file reference (relative path within bundle)
    scaler_file: str = "scaler.pkl"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Convert tuples to lists for JSON serialization
        d["clip_range"] = list(self.clip_range)
        d["robust_quantile_range"] = list(self.robust_quantile_range)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScalingConfig:
        # Convert lists back to tuples
        if "clip_range" in data:
            data["clip_range"] = tuple(data["clip_range"])
        if "robust_quantile_range" in data:
            data["robust_quantile_range"] = tuple(data["robust_quantile_range"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PreprocessingGraphConfig:
    """Complete preprocessing graph configuration."""

    version: str = PREPROCESSING_GRAPH_VERSION
    created_at: str = ""
    horizon: int = 20
    symbol: str = ""
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    mtf: MTFConfig = field(default_factory=MTFConfig)
    wavelets: WaveletConfig = field(default_factory=WaveletConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    # Hash of the configuration for validation
    config_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "horizon": self.horizon,
            "symbol": self.symbol,
            "cleaning": self.cleaning.to_dict(),
            "indicators": self.indicators.to_dict(),
            "mtf": self.mtf.to_dict(),
            "wavelets": self.wavelets.to_dict(),
            "regime": self.regime.to_dict(),
            "scaling": self.scaling.to_dict(),
            "config_hash": self.config_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreprocessingGraphConfig:
        return cls(
            version=data.get("version", PREPROCESSING_GRAPH_VERSION),
            created_at=data.get("created_at", ""),
            horizon=data.get("horizon", 20),
            symbol=data.get("symbol", ""),
            cleaning=CleaningConfig.from_dict(data.get("cleaning", {})),
            indicators=IndicatorConfig.from_dict(data.get("indicators", {})),
            mtf=MTFConfig.from_dict(data.get("mtf", {})),
            wavelets=WaveletConfig.from_dict(data.get("wavelets", {})),
            regime=RegimeConfig.from_dict(data.get("regime", {})),
            scaling=ScalingConfig.from_dict(data.get("scaling", {})),
            config_hash=data.get("config_hash", ""),
        )

    def compute_hash(self) -> str:
        """Compute hash of configuration for validation."""
        # Exclude created_at and config_hash from hash computation
        hash_data = self.to_dict()
        hash_data.pop("created_at", None)
        hash_data.pop("config_hash", None)
        hash_str = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()[:16]


class PreprocessingGraph:
    """
    Serializable preprocessing pipeline for train/serve parity.

    Captures the complete preprocessing configuration used during training
    and provides methods to apply the same preprocessing at inference time.

    The graph ensures that:
    1. The same resampling is applied (1min -> 5min)
    2. The same indicators are computed with identical periods
    3. The same MTF features are generated
    4. The same scaling parameters are applied

    Attributes:
        config: PreprocessingGraphConfig with all settings
        scaler: Fitted scaler instance (loaded from bundle)
        _is_fitted: Whether the graph has been fitted/loaded
    """

    def __init__(self, config: PreprocessingGraphConfig) -> None:
        """
        Initialize PreprocessingGraph.

        Args:
            config: Complete preprocessing configuration
        """
        self.config = config
        self._scaler: Any = None
        self._is_fitted = False

    @classmethod
    def from_pipeline_config(
        cls,
        pipeline_config: dict[str, Any],
        feature_columns: list[str] | None = None,
        scaling_params: dict[str, dict[str, float]] | None = None,
        symbol: str = "",
        horizon: int = 20,
    ) -> PreprocessingGraph:
        """
        Create from Phase 1 pipeline configuration.

        Args:
            pipeline_config: Dictionary with pipeline stage configurations
            feature_columns: List of feature column names
            scaling_params: Per-column scaling parameters
            symbol: Trading symbol
            horizon: Prediction horizon

        Returns:
            PreprocessingGraph configured for the pipeline
        """
        # Extract cleaning config
        clean_cfg = pipeline_config.get("clean", {})
        cleaning = CleaningConfig(
            source_timeframe=clean_cfg.get("timeframe", "1min"),
            target_timeframe=clean_cfg.get("target_timeframe", "5min"),
            gap_fill_method=clean_cfg.get("gap_fill_method", "forward"),
            max_gap_fill_minutes=clean_cfg.get("max_gap_fill_minutes", 5),
            outlier_method=clean_cfg.get("outlier_method", "atr"),
            atr_threshold=clean_cfg.get("atr_threshold", 5.0),
            zscore_threshold=clean_cfg.get("zscore_threshold", 5.0),
            iqr_multiplier=clean_cfg.get("iqr_multiplier", 3.0),
            calendar_aware=clean_cfg.get("calendar_aware", True),
        )

        # Extract feature config
        feat_cfg = pipeline_config.get("features", {})
        indicators = IndicatorConfig(
            scale_periods=feat_cfg.get("scale_periods", True),
            base_timeframe=feat_cfg.get("base_timeframe", "5min"),
            nan_threshold=feat_cfg.get("nan_threshold", 0.9),
        )

        # Extract MTF config
        mtf_cfg = pipeline_config.get("mtf", {})
        mtf = MTFConfig(
            enabled=mtf_cfg.get("enable_mtf", True),
            base_timeframe=mtf_cfg.get("base_timeframe", "5min"),
            mtf_timeframes=mtf_cfg.get("mtf_timeframes", ["15min", "60min"]),
            mode=mtf_cfg.get("mode", "both"),
            include_ohlcv=mtf_cfg.get("include_ohlcv", True),
            include_indicators=mtf_cfg.get("include_indicators", True),
        )

        # Extract wavelet config
        wav_cfg = pipeline_config.get("wavelets", {})
        wavelets = WaveletConfig(
            enabled=wav_cfg.get("enable_wavelets", True),
            wavelet_type=wav_cfg.get("wavelet_type", "db4"),
            level=wav_cfg.get("wavelet_level", 3),
            window=wav_cfg.get("wavelet_window", 64),
        )

        # Extract regime config
        reg_cfg = pipeline_config.get("regime", {})
        regime = RegimeConfig(
            enabled=reg_cfg.get("enabled", True),
            regime_types=reg_cfg.get("regime_types", ["volatility", "trend", "structure"]),
        )

        # Extract scaling config
        scale_cfg = pipeline_config.get("scaling", {})
        scaling = ScalingConfig(
            scaler_type=scale_cfg.get("scaler_type", "robust"),
            clip_outliers=scale_cfg.get("clip_outliers", True),
            clip_range=tuple(scale_cfg.get("clip_range", [-5.0, 5.0])),
            robust_quantile_range=tuple(
                scale_cfg.get("robust_quantile_range", [25.0, 75.0])
            ),
            apply_log_to_price_volume=scale_cfg.get("apply_log_to_price_volume", True),
            feature_columns=feature_columns or [],
            scaling_params=scaling_params or {},
        )

        config = PreprocessingGraphConfig(
            version=PREPROCESSING_GRAPH_VERSION,
            created_at=datetime.now().isoformat(),
            horizon=horizon,
            symbol=symbol,
            cleaning=cleaning,
            indicators=indicators,
            mtf=mtf,
            wavelets=wavelets,
            regime=regime,
            scaling=scaling,
        )
        config.config_hash = config.compute_hash()

        graph = cls(config)
        graph._is_fitted = True
        return graph

    @classmethod
    def from_training_run(cls, run_path: Path) -> PreprocessingGraph:
        """
        Create from a completed training run's artifacts.

        Args:
            run_path: Path to training run directory containing:
                - preprocessing_graph.json
                - scaler.pkl

        Returns:
            PreprocessingGraph loaded from run artifacts
        """
        run_path = Path(run_path)

        # Load the graph config
        graph_path = run_path / "preprocessing_graph.json"
        if not graph_path.exists():
            raise FileNotFoundError(
                f"Preprocessing graph not found at {graph_path}. "
                "Ensure the training run saved the preprocessing graph."
            )

        graph = cls.load(graph_path)

        # Load scaler if available
        scaler_path = run_path / graph.config.scaling.scaler_file
        if scaler_path.exists():
            graph._load_scaler(scaler_path)

        return graph

    def _load_scaler(self, path: Path) -> None:
        """Load fitted scaler from disk."""
        with open(path, "rb") as f:
            self._scaler = pickle.load(f)
        logger.info(f"Loaded scaler from {path}")

    def _save_scaler(self, path: Path) -> None:
        """Save fitted scaler to disk."""
        if self._scaler is not None:
            with open(path, "wb") as f:
                pickle.dump(self._scaler, f)
            logger.info(f"Saved scaler to {path}")

    def set_scaler(self, scaler: Any) -> None:
        """
        Set the fitted scaler instance.

        Args:
            scaler: Fitted scaler (e.g., FeatureScaler, RobustScaler)
        """
        self._scaler = scaler
        self._is_fitted = True

    def transform(
        self,
        raw_df: pd.DataFrame,
        skip_cleaning: bool = False,
        skip_scaling: bool = False,
    ) -> pd.DataFrame:
        """
        Apply preprocessing to raw OHLCV data.

        Steps:
        1. Validate input schema (OHLCV columns required)
        2. Resample to target timeframe (if not skipped)
        3. Generate features (indicators, wavelets, regimes)
        4. Generate MTF features
        5. Apply scaling (if scaler available and not skipped)

        Args:
            raw_df: DataFrame with raw OHLCV data. Must have columns:
                   [datetime, open, high, low, close, volume]
            skip_cleaning: If True, skip the resampling step (data already cleaned)
            skip_scaling: If True, skip the scaling step

        Returns:
            DataFrame with features ready for model prediction

        Raises:
            ValueError: If required columns are missing
            RuntimeError: If graph not fitted and scaling requested
        """
        # Validate input
        self._validate_input(raw_df)

        df = raw_df.copy()

        # Step 1: Cleaning / Resampling
        if not skip_cleaning:
            df = self._apply_cleaning(df)

        # Step 2: Feature engineering
        df = self._apply_features(df)

        # Step 3: MTF features
        if self.config.mtf.enabled:
            df = self._apply_mtf(df)

        # Step 4: Regime detection
        if self.config.regime.enabled:
            df = self._apply_regime(df)

        # Step 5: Handle NaN values (drop rows)
        df = df.dropna()

        # Step 6: Scaling
        if not skip_scaling and self._scaler is not None:
            df = self._apply_scaling(df)

        # Step 7: Select feature columns if specified
        if self.config.scaling.feature_columns:
            available_cols = [
                c for c in self.config.scaling.feature_columns if c in df.columns
            ]
            missing_cols = set(self.config.scaling.feature_columns) - set(
                available_cols
            )
            if missing_cols:
                logger.warning(
                    f"Missing {len(missing_cols)} feature columns: "
                    f"{list(missing_cols)[:5]}..."
                )
            df = df[available_cols]

        return df

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate input DataFrame has required columns."""
        required_cols = {"open", "high", "low", "close", "volume"}

        # Check for datetime column or index
        has_datetime = "datetime" in df.columns or isinstance(
            df.index, pd.DatetimeIndex
        )

        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required OHLCV columns: {missing}. "
                f"Expected: {required_cols}"
            )
        if not has_datetime:
            raise ValueError(
                "DataFrame must have 'datetime' column or DatetimeIndex"
            )

    def _apply_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply data cleaning and resampling."""
        try:
            from src.phase1.stages.clean.cleaner import DataCleaner

            # Create a temporary cleaner for resampling
            # Note: In production, this would use the saved cleaner config
            cfg = self.config.cleaning

            # For inference, we typically receive already-cleaned data
            # Just resample if needed
            if "datetime" in df.columns:
                df = df.set_index("datetime")

            source_freq = cfg.source_timeframe
            target_freq = cfg.target_timeframe

            if source_freq != target_freq:
                # Resample to target timeframe
                target_pandas_freq = self._get_pandas_freq(target_freq)
                df_resampled = df.resample(target_pandas_freq).agg(
                    {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    }
                )
                df = df_resampled.dropna()
                logger.debug(
                    f"Resampled from {source_freq} to {target_freq}: {len(df)} bars"
                )

            df = df.reset_index()
            return df

        except ImportError:
            logger.warning("DataCleaner not available, skipping cleaning step")
            return df

    def _apply_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering."""
        try:
            # Import feature functions
            from src.phase1.stages.features.microstructure import (
                add_microstructure_features,
            )
            from src.phase1.stages.features.momentum import (
                add_cci,
                add_macd,
                add_mfi,
                add_roc,
                add_rsi,
                add_stochastic,
                add_williams_r,
            )
            from src.phase1.stages.features.moving_averages import add_ema, add_sma
            from src.phase1.stages.features.price_features import (
                add_price_ratios,
                add_returns,
            )
            from src.phase1.stages.features.regime import add_regime_features
            from src.phase1.stages.features.temporal import add_temporal_features
            from src.phase1.stages.features.trend import add_adx, add_supertrend
            from src.phase1.stages.features.volatility import (
                add_atr,
                add_bollinger_bands,
                add_garman_klass_volatility,
                add_historical_volatility,
                add_keltner_channels,
                add_parkinson_volatility,
                add_rogers_satchell_volatility,
                add_yang_zhang_volatility,
            )
            from src.phase1.stages.features.volume import (
                add_dollar_volume,
                add_volume_features,
                add_vwap,
            )

            cfg = self.config.indicators
            metadata: dict[str, str] = {}

            # Apply features in order
            df = add_returns(df, metadata)
            df = add_price_ratios(df, metadata)
            df = add_sma(df, metadata, periods=cfg.sma_periods)
            df = add_ema(df, metadata, periods=cfg.ema_periods)
            df = add_rsi(df, metadata, period=cfg.rsi_period)
            df = add_macd(
                df,
                metadata,
                fast_period=cfg.macd_fast,
                slow_period=cfg.macd_slow,
                signal_period=cfg.macd_signal,
            )
            df = add_stochastic(
                df, metadata, k_period=cfg.stochastic_k, d_period=cfg.stochastic_d
            )
            df = add_williams_r(df, metadata, period=cfg.williams_r_period)
            df = add_roc(df, metadata, periods=cfg.roc_periods)
            df = add_cci(df, metadata, period=cfg.cci_period)
            df = add_mfi(df, metadata, period=cfg.mfi_period)
            df = add_atr(df, metadata, periods=cfg.atr_periods)
            df = add_bollinger_bands(df, metadata, period=cfg.bollinger_period)
            df = add_keltner_channels(df, metadata, period=cfg.keltner_period)
            df = add_historical_volatility(df, metadata, periods=cfg.hvol_periods)
            df = add_parkinson_volatility(df, metadata, period=cfg.parkinson_period)
            df = add_garman_klass_volatility(
                df, metadata, period=cfg.garman_klass_period
            )
            df = add_rogers_satchell_volatility(df, metadata, period=cfg.rs_vol_period)
            df = add_yang_zhang_volatility(df, metadata, period=cfg.yz_vol_period)
            df = add_volume_features(df, metadata, period=cfg.volume_sma_period)
            df = add_vwap(df, metadata)
            df = add_dollar_volume(df, metadata)
            df = add_adx(df, metadata, period=cfg.adx_period)
            df = add_supertrend(df, metadata, period=cfg.supertrend_period)
            df = add_temporal_features(df, metadata)
            df = add_regime_features(df, metadata)
            df = add_microstructure_features(df, metadata)

            # Apply wavelets if enabled
            if self.config.wavelets.enabled and len(df) >= self.config.wavelets.window:
                try:
                    from src.phase1.stages.features.wavelets import add_wavelet_features

                    wav_cfg = self.config.wavelets
                    df = add_wavelet_features(
                        df,
                        metadata,
                        price_col="close",
                        volume_col="volume",
                        wavelet=wav_cfg.wavelet_type,
                        level=wav_cfg.level,
                        window=wav_cfg.window,
                        include_volume=wav_cfg.include_volume,
                        include_energy=wav_cfg.include_energy,
                        include_volatility=wav_cfg.include_volatility,
                        include_trend=wav_cfg.include_trend,
                    )
                except ImportError:
                    logger.warning("PyWavelets not available, skipping wavelet features")

            logger.debug(f"Applied {len(metadata)} feature groups")
            return df

        except ImportError as e:
            logger.error(f"Feature engineering import error: {e}")
            raise RuntimeError(
                "Feature engineering modules not available. "
                "Ensure src/phase1/stages/features is installed."
            ) from e

    def _apply_mtf(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply multi-timeframe features."""
        try:
            from src.phase1.stages.mtf.generator import MTFFeatureGenerator

            cfg = self.config.mtf
            mtf_gen = MTFFeatureGenerator(
                base_timeframe=cfg.base_timeframe,
                mtf_timeframes=cfg.mtf_timeframes,
                mode=cfg.mode,
            )

            df = mtf_gen.generate_mtf_features(df)
            logger.debug(f"Applied MTF features for {cfg.mtf_timeframes}")
            return df

        except ImportError as e:
            logger.warning(f"MTF generator not available: {e}")
            return df

    def _apply_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply regime detection."""
        # Regime features are already added in _apply_features via add_regime_features
        # This method is for any additional regime-specific processing
        return df

    def _apply_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted scaler to features."""
        if self._scaler is None:
            logger.warning("No scaler available, skipping scaling")
            return df

        # Get numeric columns for scaling
        feature_cols = [
            c for c in df.columns if c not in ["datetime", "date", "time", "symbol"]
        ]
        numeric_cols = df[feature_cols].select_dtypes(
            include=[np.number]
        ).columns.tolist()

        if not numeric_cols:
            return df

        # Apply scaler transform
        try:
            if hasattr(self._scaler, "transform"):
                df[numeric_cols] = self._scaler.transform(df[numeric_cols])
            else:
                logger.warning("Scaler does not have transform method")
        except Exception as e:
            logger.error(f"Scaling error: {e}")
            raise RuntimeError(f"Failed to apply scaling: {e}") from e

        return df

    def _get_pandas_freq(self, timeframe: str) -> str:
        """Convert timeframe string to pandas frequency."""
        freq_map = {
            "1min": "1min",
            "5min": "5min",
            "10min": "10min",
            "15min": "15min",
            "20min": "20min",
            "30min": "30min",
            "45min": "45min",
            "60min": "60min",
            "1h": "1h",
        }
        return freq_map.get(timeframe, timeframe)

    def save(self, path: Path) -> None:
        """
        Save preprocessing graph to JSON.

        Args:
            path: Path to save the graph configuration
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Update hash before saving
        self.config.config_hash = self.config.compute_hash()

        with open(path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        logger.info(f"Saved preprocessing graph to {path}")

        # Save scaler alongside if available
        if self._scaler is not None:
            scaler_path = path.parent / self.config.scaling.scaler_file
            self._save_scaler(scaler_path)

    @classmethod
    def load(cls, path: Path) -> PreprocessingGraph:
        """
        Load preprocessing graph from JSON.

        Args:
            path: Path to the graph configuration file

        Returns:
            PreprocessingGraph loaded from file
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Preprocessing graph not found at {path}")

        with open(path) as f:
            data = json.load(f)

        config = PreprocessingGraphConfig.from_dict(data)

        # Validate hash
        expected_hash = config.compute_hash()
        if config.config_hash and config.config_hash != expected_hash:
            logger.warning(
                f"Config hash mismatch. Expected {expected_hash}, got {config.config_hash}. "
                "Configuration may have been modified."
            )

        graph = cls(config)

        # Try to load scaler
        scaler_path = path.parent / config.scaling.scaler_file
        if scaler_path.exists():
            graph._load_scaler(scaler_path)

        graph._is_fitted = True
        logger.info(f"Loaded preprocessing graph from {path}")
        return graph

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.config.to_dict()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreprocessingGraph:
        """Create from dictionary."""
        config = PreprocessingGraphConfig.from_dict(data)
        return cls(config)

    def validate(self) -> dict[str, Any]:
        """
        Validate the preprocessing graph.

        Returns:
            Dictionary with validation results
        """
        issues: list[str] = []

        # Check version compatibility
        if self.config.version != PREPROCESSING_GRAPH_VERSION:
            issues.append(
                f"Version mismatch: graph {self.config.version} vs "
                f"current {PREPROCESSING_GRAPH_VERSION}"
            )

        # Check scaler
        if not self._is_fitted:
            issues.append("Graph not fitted (no scaler loaded)")

        # Check feature columns
        if not self.config.scaling.feature_columns:
            issues.append("No feature columns specified")

        # Check hash
        expected_hash = self.config.compute_hash()
        if self.config.config_hash and self.config.config_hash != expected_hash:
            issues.append("Configuration hash mismatch")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "version": self.config.version,
            "horizon": self.config.horizon,
            "symbol": self.config.symbol,
            "n_features": len(self.config.scaling.feature_columns),
            "has_scaler": self._scaler is not None,
        }

    def __repr__(self) -> str:
        return (
            f"PreprocessingGraph(version={self.config.version}, "
            f"symbol={self.config.symbol}, "
            f"horizon={self.config.horizon}, "
            f"features={len(self.config.scaling.feature_columns)}, "
            f"fitted={self._is_fitted})"
        )


# Constants for bundle integration
PREPROCESSING_GRAPH_FILE = "preprocessing_graph.json"

__all__ = [
    "PreprocessingGraph",
    "PreprocessingGraphConfig",
    "CleaningConfig",
    "MTFConfig",
    "WaveletConfig",
    "IndicatorConfig",
    "RegimeConfig",
    "ScalingConfig",
    "PREPROCESSING_GRAPH_VERSION",
    "PREPROCESSING_GRAPH_FILE",
]
