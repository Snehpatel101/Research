"""
Main FeatureEngineer class for feature engineering pipeline.

This module provides the FeatureEngineer class that orchestrates
all feature engineering operations and manages the complete pipeline.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

# MTF Features - import from sibling module
from ..mtf import add_mtf_features
from .microstructure import add_microstructure_features
from .momentum import add_cci, add_macd, add_mfi, add_roc, add_rsi, add_stochastic, add_williams_r
from .moving_averages import add_ema, add_sma
from .nan_handling import clean_nan_columns

# Import all feature modules
from .price_features import add_autocorrelation, add_clv, add_price_ratios, add_returns
from .regime import add_regime_features
from .scaling import PeriodScaler, create_period_config
from .temporal import add_temporal_features
from .trend import add_adx, add_supertrend
from .volatility import (
    add_atr,
    add_bollinger_bands,
    add_garman_klass_volatility,
    add_higher_moments,
    add_historical_volatility,
    add_keltner_channels,
    add_parkinson_volatility,
    add_rogers_satchell_volatility,
    add_yang_zhang_volatility,
)

# Re-import here for wrapper methods
from .volume import add_dollar_volume, add_volume_features, add_vwap
from .wavelets import PYWT_AVAILABLE, add_wavelet_features

# Configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class FeatureEngineer:
    """
    Comprehensive feature engineering for financial time series.

    This class orchestrates the complete feature engineering pipeline,
    generating 50+ technical indicators across multiple categories:
    - Price-based features (returns, ratios)
    - Moving averages (SMA, EMA)
    - Momentum indicators (RSI, MACD, Stochastic, etc.)
    - Volatility indicators (ATR, Bollinger Bands, etc.)
    - Volume indicators (OBV, VWAP, etc.)
    - Trend indicators (ADX, Supertrend, etc.)
    - Temporal features (time encoding)
    - Regime indicators (volatility, trend)
    - Cross-asset features (MES-MGC correlation, beta, etc.)
    - Multi-timeframe features (MTF - indicators from higher TFs)
    """

    def __init__(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        timeframe: str = '5min',
        enable_mtf: bool = True,
        mtf_timeframes: list | None = None,
        mtf_include_ohlcv: bool = True,
        mtf_include_indicators: bool = True,
        scale_periods: bool = True,
        base_timeframe: str = '5min',
        enable_wavelets: bool = True,
        wavelet_type: str = 'db4',
        wavelet_level: int = 3,
        wavelet_window: int = 64,
        nan_threshold: float = 0.9
    ):
        """
        Initialize feature engineer.

        Parameters
        ----------
        input_dir : Union[str, Path]
            Path to cleaned data directory
        output_dir : Union[str, Path]
            Path to output directory
        timeframe : str, default '5min'
            Data timeframe (e.g., '1min', '5min', '15min')
        enable_mtf : bool, default True
            Whether to compute multi-timeframe features
        mtf_timeframes : list, optional
            List of higher timeframes for MTF features.
            Default: ['15min', '60min']
        mtf_include_ohlcv : bool, default True
            Include OHLCV data from higher TFs
        mtf_include_indicators : bool, default True
            Include indicators computed on higher TFs
        scale_periods : bool, default True
            Whether to scale indicator periods based on timeframe.
            When True, indicator periods are scaled to maintain
            consistent lookback duration across timeframes.
        base_timeframe : str, default '5min'
            Base timeframe that indicator periods are defined for.
            Only used when scale_periods=True.
        enable_wavelets : bool, default True
            Whether to compute wavelet decomposition features.
            Requires PyWavelets library.
        wavelet_type : str, default 'db4'
            Wavelet family to use ('db4', 'sym5', 'coif3', 'haar').
        wavelet_level : int, default 3
            Decomposition level (3 = 4 frequency bands).
        wavelet_window : int, default 64
            Rolling window size for causal wavelet computation.
        nan_threshold : float, default 0.9
            Columns with NaN rate above this threshold are dropped before
            row-wise NaN removal. Range: 0.0 to 1.0. Set to 1.0 to disable
            column dropping (original behavior).
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        # Validate input directory exists
        if not self.input_dir.exists():
            raise FileNotFoundError(
                f"Input directory does not exist: {self.input_dir}. "
                f"Expected cleaned data from DataCleaner stage."
            )

        self.timeframe = timeframe

        # Period scaling configuration
        self.scale_periods = scale_periods
        self.base_timeframe = base_timeframe
        if scale_periods:
            self.period_scaler = PeriodScaler(timeframe, base_timeframe)
            self.period_config = self.period_scaler.config
        else:
            self.period_scaler = None
            self.period_config = create_period_config(base_timeframe, base_timeframe)

        # MTF configuration
        self.enable_mtf = enable_mtf
        self.mtf_timeframes = mtf_timeframes or ['15min', '60min']
        self.mtf_include_ohlcv = mtf_include_ohlcv
        self.mtf_include_indicators = mtf_include_indicators

        # Wavelet configuration
        self.enable_wavelets = enable_wavelets and PYWT_AVAILABLE
        self.wavelet_type = wavelet_type
        self.wavelet_level = wavelet_level
        self.wavelet_window = wavelet_window
        if enable_wavelets and not PYWT_AVAILABLE:
            logger.warning("Wavelets disabled: PyWavelets not installed")

        # NaN handling configuration
        if not 0.0 <= nan_threshold <= 1.0:
            raise ValueError(f"nan_threshold must be between 0.0 and 1.0, got {nan_threshold}")
        self.nan_threshold = nan_threshold

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Feature metadata
        self.feature_metadata: dict[str, str] = {}

        logger.info("Initialized FeatureEngineer")
        logger.info(f"Input dir: {self.input_dir}")
        logger.info(f"Output dir: {self.output_dir}")
        logger.info(f"Timeframe: {self.timeframe}, scale_periods: {self.scale_periods}")
        if self.scale_periods and self.timeframe != self.base_timeframe:
            logger.info(f"Period scaling: {self.base_timeframe} -> {self.timeframe}")
        logger.info(f"MTF enabled: {self.enable_mtf}, timeframes: {self.mtf_timeframes}")
        logger.info(
            f"Wavelets enabled: {self.enable_wavelets}, "
            f"type: {self.wavelet_type}, level: {self.wavelet_level}"
        )
        logger.info(f"NaN threshold: {self.nan_threshold} (columns with >{self.nan_threshold*100:.0f}% NaN dropped)")

    def engineer_features(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> tuple[pd.DataFrame, dict]:
        """
        Complete feature engineering pipeline.

        Each symbol is processed independently - no cross-symbol correlation.
        This ensures symbol isolation as required by the ML Factory design.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with cleaned OHLCV data
        symbol : str
            Symbol name

        Returns
        -------
        Tuple[pd.DataFrame, Dict]
            (DataFrame with features, feature report)
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting feature engineering for: {symbol}")
        logger.info(f"{'='*60}\n")

        initial_rows = len(df)
        initial_cols = len(df.columns)

        df = df.copy()

        # Reset feature metadata for this run
        self.feature_metadata = {}

        # Get scaled periods for this timeframe
        pc = self.period_config

        # Add all features with scaled periods
        df = add_returns(df, self.feature_metadata)
        df = add_price_ratios(df, self.feature_metadata)
        df = add_sma(df, self.feature_metadata, periods=pc.get('sma'))
        df = add_ema(df, self.feature_metadata, periods=pc.get('ema'))
        df = add_rsi(df, self.feature_metadata, period=pc.get('rsi', [14])[0])
        df = add_macd(
            df, self.feature_metadata,
            fast_period=pc.get('macd_fast', [12])[0],
            slow_period=pc.get('macd_slow', [26])[0],
            signal_period=pc.get('macd_signal', [9])[0]
        )
        df = add_stochastic(
            df, self.feature_metadata,
            k_period=pc.get('stochastic_k', [14])[0],
            d_period=pc.get('stochastic_d', [3])[0]
        )
        df = add_williams_r(df, self.feature_metadata, period=pc.get('williams_r', [14])[0])
        df = add_roc(df, self.feature_metadata, periods=pc.get('roc'))
        df = add_cci(df, self.feature_metadata, period=pc.get('cci', [20])[0])
        df = add_mfi(df, self.feature_metadata, period=pc.get('mfi', [14])[0])
        df = add_atr(df, self.feature_metadata, periods=pc.get('atr'))
        df = add_bollinger_bands(df, self.feature_metadata, period=pc.get('bollinger', [20])[0])
        df = add_keltner_channels(df, self.feature_metadata, period=pc.get('keltner', [20])[0])
        df = add_historical_volatility(df, self.feature_metadata, periods=pc.get('hvol'))
        df = add_parkinson_volatility(df, self.feature_metadata, period=pc.get('parkinson', [20])[0])
        df = add_garman_klass_volatility(df, self.feature_metadata, period=pc.get('garman_klass', [20])[0])
        df = add_rogers_satchell_volatility(df, self.feature_metadata, period=pc.get('rs_vol', [20])[0])
        df = add_yang_zhang_volatility(df, self.feature_metadata, period=pc.get('yz_vol', [20])[0])
        df = add_volume_features(df, self.feature_metadata, period=pc.get('volume_sma', [20])[0])
        df = add_vwap(df, self.feature_metadata)
        df = add_dollar_volume(df, self.feature_metadata)
        df = add_adx(df, self.feature_metadata, period=pc.get('adx', [14])[0])
        df = add_supertrend(
            df, self.feature_metadata,
            period=pc.get('supertrend_period', [10])[0]
        )
        df = add_temporal_features(df, self.feature_metadata)
        df = add_regime_features(df, self.feature_metadata)

        # Add new statistical features
        df = add_autocorrelation(df, self.feature_metadata)
        df = add_clv(df, self.feature_metadata)
        df = add_higher_moments(df, self.feature_metadata)

        # Add microstructure proxy features (liquidity, spread, price impact from OHLCV)
        df = add_microstructure_features(df, self.feature_metadata)

        # Add Wavelet decomposition features
        wavelet_cols_added = 0
        if self.enable_wavelets and len(df) >= self.wavelet_window:
            try:
                logger.info(
                    f"Adding wavelet features ({self.wavelet_type}, "
                    f"level={self.wavelet_level}, window={self.wavelet_window})"
                )
                df = add_wavelet_features(
                    df,
                    self.feature_metadata,
                    price_col='close',
                    volume_col='volume',
                    wavelet=self.wavelet_type,
                    level=self.wavelet_level,
                    window=self.wavelet_window,
                    include_volume=True,
                    include_energy=True,
                    include_volatility=True,
                    include_trend=True
                )
                # Count wavelet columns added
                wavelet_cols_added = len([
                    c for c in df.columns if c.startswith('wavelet_')
                ])
                logger.info(f"Added {wavelet_cols_added} wavelet feature columns")
            except Exception as e:
                logger.warning(
                    f"Wavelet feature generation failed: {e}. Continuing without wavelet features."
                )
        elif self.enable_wavelets:
            logger.warning(
                f"Skipping wavelet features: insufficient data "
                f"({len(df)} rows < {self.wavelet_window} window required)"
            )

        # Add Multi-Timeframe (MTF) features
        # Filter MTF timeframes to only include those > base timeframe
        mtf_cols_added = 0
        if self.enable_mtf and len(df) >= 500:
            try:
                from src.phase1.config.features import parse_timeframe_to_minutes
                base_minutes = parse_timeframe_to_minutes(self.timeframe)
                # Filter MTF timeframes to only those strictly greater than base
                valid_mtf_timeframes = [
                    tf for tf in self.mtf_timeframes
                    if parse_timeframe_to_minutes(tf) > base_minutes
                ]
                if valid_mtf_timeframes:
                    logger.info(f"Adding MTF features for timeframes: {valid_mtf_timeframes}")
                    df = add_mtf_features(
                        df,
                        feature_metadata=self.feature_metadata,
                        base_timeframe=self.timeframe,
                        mtf_timeframes=valid_mtf_timeframes,
                        include_ohlcv=self.mtf_include_ohlcv,
                        include_indicators=self.mtf_include_indicators
                    )
                    # Count MTF columns added (dynamic suffixes based on config)
                    mtf_suffixes = []
                    for tf in valid_mtf_timeframes:
                        if tf.endswith('min'):
                            mtf_suffixes.append(f"_{tf.replace('min', 'm')}")
                        elif tf in ['1h', '60min']:
                            mtf_suffixes.append('_1h')
                    mtf_cols_added = len([
                        c for c in df.columns
                        if any(c.endswith(s) for s in mtf_suffixes)
                    ])
                    logger.info(f"Added {mtf_cols_added} MTF feature columns")
                else:
                    logger.info(f"No MTF timeframes > base {self.timeframe}, skipping MTF features")
            except Exception as e:
                logger.warning(f"MTF feature generation failed: {e}. Continuing without MTF features.")
        elif self.enable_mtf:
            logger.warning(
                f"Skipping MTF features: insufficient data ({len(df)} rows < 500 required)"
            )

        # Audit NaN values and clean problematic columns before row-wise dropna
        # This prevents all-NaN columns from wiping the entire dataset
        df, nan_audit = clean_nan_columns(
            df,
            symbol=symbol,
            nan_threshold=self.nan_threshold
        )
        rows_dropped = nan_audit['rows_dropped']
        cols_dropped = nan_audit['cols_dropped']

        # Get MTF column names
        mtf_suffixes = ['_15m', '_30m', '_1h']
        mtf_col_names = [
            c for c in df.columns
            if any(c.endswith(s) for s in mtf_suffixes)
        ]

        # Get microstructure feature names
        micro_col_names = [c for c in df.columns if c.startswith('micro_')]

        feature_report = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'initial_rows': initial_rows,
            'initial_columns': initial_cols,
            'final_rows': len(df),
            'final_columns': len(df.columns),
            'features_added': len(df.columns) - initial_cols,
            'rows_dropped_for_nan': rows_dropped,
            'cols_dropped_for_nan': cols_dropped,
            'nan_audit': nan_audit,
            'symbol_isolation': True,  # Each symbol processed independently
            'mtf_features': self.enable_mtf and mtf_cols_added > 0,
            'mtf_feature_count': mtf_cols_added,
            'mtf_timeframes': self.mtf_timeframes if mtf_cols_added > 0 else [],
            'mtf_feature_names': mtf_col_names,
            'wavelet_features': self.enable_wavelets and wavelet_cols_added > 0,
            'wavelet_feature_count': wavelet_cols_added,
            'wavelet_config': {
                'type': self.wavelet_type,
                'level': self.wavelet_level,
                'window': self.wavelet_window
            } if wavelet_cols_added > 0 else {},
            'wavelet_feature_names': [c for c in df.columns if c.startswith('wavelet_')],
            'microstructure_features': len(micro_col_names) > 0,
            'microstructure_feature_count': len(micro_col_names),
            'microstructure_feature_names': micro_col_names,
            'date_range': {
                'start': df['datetime'].min().isoformat(),
                'end': df['datetime'].max().isoformat()
            }
        }

        logger.info(f"\nFeature engineering complete for {symbol}")
        logger.info(f"Columns: {initial_cols} -> {len(df.columns)} (+{feature_report['features_added']} features, -{cols_dropped} dropped for NaN)")
        logger.info(f"Rows: {initial_rows:,} -> {len(df):,} (-{rows_dropped:,} for NaN)")
        logger.info("Symbol isolation: each symbol processed independently")
        if mtf_cols_added > 0:
            logger.info(f"MTF features: {mtf_cols_added} columns from {self.mtf_timeframes}")
        if wavelet_cols_added > 0:
            logger.info(f"Wavelet features: {wavelet_cols_added} columns ({self.wavelet_type})")
        if len(micro_col_names) > 0:
            logger.info(f"Microstructure features: {len(micro_col_names)} columns")

        return df, feature_report

    def save_features(
        self,
        df: pd.DataFrame,
        symbol: str,
        feature_report: dict
    ) -> tuple[Path, Path]:
        """
        Save features and metadata.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with features
        symbol : str
            Symbol name
        feature_report : Dict
            Feature report dict

        Returns
        -------
        Tuple[Path, Path]
            (data_path, metadata_path)
        """
        # Save features
        data_path = self.output_dir / f"{symbol}_features.parquet"
        df.to_parquet(
            data_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        logger.info(f"Saved features to: {data_path}")

        # Save metadata
        metadata = {
            'report': feature_report,
            'features': self.feature_metadata
        }

        metadata_path = self.output_dir / f"{symbol}_feature_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Saved metadata to: {metadata_path}")

        return data_path, metadata_path

    def process_file(self, file_path: str | Path) -> dict:
        """
        Process a single file.

        Parameters
        ----------
        file_path : Union[str, Path]
            Path to cleaned data file

        Returns
        -------
        Dict
            Feature report
        """
        file_path = Path(file_path)
        symbol = file_path.stem.split('_')[0].upper()

        # Load data
        df = pd.read_parquet(file_path)

        # Engineer features
        df, feature_report = self.engineer_features(df, symbol)

        # Save results
        self.save_features(df, symbol, feature_report)

        return feature_report

    def process_directory(self, pattern: str = "*.parquet") -> dict[str, dict]:
        """
        Process all files in directory.

        Parameters
        ----------
        pattern : str, default "*.parquet"
            File pattern to match

        Returns
        -------
        Dict[str, Dict]
            Dictionary mapping symbols to feature reports
        """
        files = list(self.input_dir.glob(pattern))

        if not files:
            logger.warning(f"No files found matching pattern: {pattern}")
            return {}

        logger.info(f"Found {len(files)} files to process")

        results = {}
        errors = []

        for file_path in files:
            try:
                feature_report = self.process_file(file_path)
                symbol = feature_report['symbol']
                results[symbol] = feature_report

            except Exception as e:
                errors.append({
                    'file': str(file_path.name),
                    'error': str(e),
                    'type': type(e).__name__
                })
                logger.error(f"Error processing {file_path.name}: {e}", exc_info=True)

        if errors:
            error_summary = f"{len(errors)}/{len(files)} files failed"
            logger.error(f"Feature engineering completed with errors: {error_summary}")
            raise RuntimeError(f"{error_summary}. Errors: {errors[:5]}")

        return results

    def process_multi_symbol(
        self,
        symbol_files: dict[str, str | Path]
    ) -> dict[str, dict]:
        """
        Process multiple symbols independently (no cross-symbol correlation).

        Each symbol is processed in isolation, ensuring no data leakage
        between symbols as required by the ML Factory design.

        Parameters
        ----------
        symbol_files : Dict[str, Union[str, Path]]
            Dict mapping symbol names to file paths
            e.g., {'MES': 'path/to/mes.parquet', 'MGC': 'path/to/mgc.parquet'}

        Returns
        -------
        Dict[str, Dict]
            Dictionary mapping symbols to feature reports
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {len(symbol_files)} symbols (isolated, no cross-correlation)")
        logger.info(f"{'='*60}\n")

        # Process each symbol independently
        results = {}
        errors = []

        for symbol, file_path in symbol_files.items():
            file_path = Path(file_path)
            if not file_path.exists():
                logger.warning(f"File not found for {symbol}: {file_path}")
                continue

            try:
                logger.info(f"\nProcessing {symbol.upper()}...")
                df = pd.read_parquet(file_path)
                logger.info(f"Loaded {symbol.upper()}: {len(df):,} rows")

                # Engineer features (symbol isolated)
                df_features, feature_report = self.engineer_features(df, symbol.upper())

                # Save results
                self.save_features(df_features, symbol.upper(), feature_report)
                results[symbol.upper()] = feature_report

            except Exception as e:
                errors.append({
                    'symbol': symbol,
                    'error': str(e),
                    'type': type(e).__name__
                })
                logger.error(f"Error processing {symbol}: {e}", exc_info=True)

        if errors:
            error_summary = f"{len(errors)}/{len(symbol_files)} symbols failed"
            logger.error(f"Multi-symbol processing completed with errors: {error_summary}")
            raise RuntimeError(f"{error_summary}. Errors: {errors[:5]}")

        return results


if __name__ == '__main__':
    from .cli import main
    main()
