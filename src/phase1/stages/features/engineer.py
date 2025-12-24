"""
Main FeatureEngineer class for feature engineering pipeline.

This module provides the FeatureEngineer class that orchestrates
all feature engineering operations and manages the complete pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict, Tuple, Optional, List
import logging
from datetime import datetime
import json

# Import all feature modules
from .price_features import add_returns, add_price_ratios
from .moving_averages import add_sma, add_ema
from .momentum import (
    add_rsi, add_macd, add_stochastic, add_williams_r,
    add_roc, add_cci, add_mfi
)
from .volatility import (
    add_atr, add_bollinger_bands, add_keltner_channels,
    add_historical_volatility, add_parkinson_volatility,
    add_garman_klass_volatility
)
# Re-import here for wrapper methods
from .volume import add_volume_features, add_vwap, add_obv
from .trend import add_adx, add_supertrend
from .temporal import add_temporal_features, add_session_features
from .regime import add_regime_features
from .cross_asset import add_cross_asset_features
from .microstructure import add_microstructure_features
from .scaling import PeriodScaler, create_period_config
from .wavelets import add_wavelet_features, PYWT_AVAILABLE

# Import config for cross-asset feature flag
from src.phase1.config.features import CROSS_ASSET_FEATURES

# MTF Features - import from sibling module
from ..mtf import add_mtf_features, MTFFeatureGenerator

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
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        timeframe: str = '5min',
        enable_mtf: bool = True,
        mtf_timeframes: Optional[list] = None,
        mtf_include_ohlcv: bool = True,
        mtf_include_indicators: bool = True,
        scale_periods: bool = True,
        base_timeframe: str = '5min',
        enable_wavelets: bool = True,
        wavelet_type: str = 'db4',
        wavelet_level: int = 3,
        wavelet_window: int = 64
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
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
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

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Feature metadata
        self.feature_metadata: Dict[str, str] = {}

        logger.info("Initialized FeatureEngineer")
        logger.info(f"Input dir: {self.input_dir}")
        logger.info(f"Output dir: {self.output_dir}")
        logger.info(f"Timeframe: {self.timeframe}, scale_periods: {self.scale_periods}")
        if self.scale_periods and self.timeframe != self.base_timeframe:
            logger.info(f"Period scaling: {self.base_timeframe} -> {self.timeframe}")
        logger.info(f"MTF enabled: {self.enable_mtf}, timeframes: {self.mtf_timeframes}")
        logger.info(f"Cross-asset features enabled: {CROSS_ASSET_FEATURES.get('enabled', True)}")
        logger.info(
            f"Wavelets enabled: {self.enable_wavelets}, "
            f"type: {self.wavelet_type}, level: {self.wavelet_level}"
        )

    def engineer_features(
        self,
        df: pd.DataFrame,
        symbol: str,
        cross_asset_data: Optional[Dict[str, np.ndarray]] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete feature engineering pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with cleaned OHLCV data
        symbol : str
            Symbol name
        cross_asset_data : Optional[Dict[str, np.ndarray]]
            Optional dict with 'mes_close' and 'mgc_close' arrays
            for cross-asset feature computation. Arrays must be
            aligned with df index (same length and timestamps).

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
        df = add_volume_features(df, self.feature_metadata, period=pc.get('volume_sma', [20])[0])
        df = add_vwap(df, self.feature_metadata)
        df = add_adx(df, self.feature_metadata, period=pc.get('adx', [14])[0])
        df = add_supertrend(
            df, self.feature_metadata,
            period=pc.get('supertrend_period', [10])[0]
        )
        df = add_temporal_features(df, self.feature_metadata)
        df = add_regime_features(df, self.feature_metadata)

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

        # Add cross-asset features (MES-MGC)
        mes_close = cross_asset_data.get('mes_close') if cross_asset_data else None
        mgc_close = cross_asset_data.get('mgc_close') if cross_asset_data else None

        df = add_cross_asset_features(
            df,
            self.feature_metadata,
            mes_close=mes_close,
            mgc_close=mgc_close,
            current_symbol=symbol
        )

        # Add Multi-Timeframe (MTF) features
        mtf_cols_added = 0
        if self.enable_mtf and len(df) >= 500:
            try:
                logger.info(f"Adding MTF features for timeframes: {self.mtf_timeframes}")
                df = add_mtf_features(
                    df,
                    feature_metadata=self.feature_metadata,
                    base_timeframe=self.timeframe if self.timeframe in ['5min'] else '5min',
                    mtf_timeframes=self.mtf_timeframes,
                    include_ohlcv=self.mtf_include_ohlcv,
                    include_indicators=self.mtf_include_indicators
                )
                # Count MTF columns added
                mtf_suffixes = ['_15m', '_30m', '_1h']
                mtf_cols_added = len([
                    c for c in df.columns
                    if any(c.endswith(s) for s in mtf_suffixes)
                ])
                logger.info(f"Added {mtf_cols_added} MTF feature columns")
            except Exception as e:
                logger.warning(f"MTF feature generation failed: {e}. Continuing without MTF features.")
        elif self.enable_mtf:
            logger.warning(
                f"Skipping MTF features: insufficient data ({len(df)} rows < 500 required)"
            )

        # Drop rows with NaN (mainly from initial indicator warmup periods)
        # Note: Cross-asset features are NaN when only one symbol is present
        # We use subset to exclude cross-asset columns from dropna if they are all NaN
        cross_asset_cols = ['mes_mgc_correlation_20', 'mes_mgc_spread_zscore',
                           'mes_mgc_beta', 'relative_strength']
        non_cross_asset_cols = [c for c in df.columns if c not in cross_asset_cols]

        rows_before_dropna = len(df)
        # Only drop NaN based on non-cross-asset columns
        df = df.dropna(subset=non_cross_asset_cols)
        rows_dropped = rows_before_dropna - len(df)

        if len(df) == 0:
            raise ValueError(
                f"All rows dropped after removing NaN values. "
                f"Insufficient data for feature calculation. "
                f"Original rows: {rows_before_dropna}, "
                f"Required: ~200+ rows for longest rolling window."
            )

        if rows_dropped > rows_before_dropna * 0.5:
            logger.warning(
                f"Dropped {rows_dropped} rows ({rows_dropped/rows_before_dropna*100:.1f}%) "
                f"due to NaN values. Check feature warmup periods."
            )

        # Check if cross-asset features were computed
        # Cross-asset is only computed when:
        # 1. Config flag is enabled (CROSS_ASSET_FEATURES['enabled'] = True)
        # 2. Both MES and MGC data are provided
        has_cross_asset = (
            CROSS_ASSET_FEATURES.get('enabled', True) and
            cross_asset_data is not None and
            'mes_close' in cross_asset_data and
            'mgc_close' in cross_asset_data
        )

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
            'cross_asset_features': has_cross_asset,
            'cross_asset_enabled_in_config': CROSS_ASSET_FEATURES.get('enabled', True),
            'cross_asset_feature_names': cross_asset_cols if has_cross_asset else [],
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
        logger.info(f"Columns: {initial_cols} -> {len(df.columns)} (+{feature_report['features_added']} features)")
        logger.info(f"Rows: {initial_rows:,} -> {len(df):,} (-{rows_dropped:,} for NaN)")
        if has_cross_asset:
            logger.info(f"Cross-asset features: {cross_asset_cols}")
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
        feature_report: Dict
    ) -> Tuple[Path, Path]:
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

    def process_file(self, file_path: Union[str, Path]) -> Dict:
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

    def process_directory(self, pattern: str = "*.parquet") -> Dict[str, Dict]:
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
        symbol_files: Dict[str, Union[str, Path]],
        compute_cross_asset: bool = True
    ) -> Dict[str, Dict]:
        """
        Process multiple symbols with cross-asset feature computation.

        This method loads data for multiple symbols, aligns them by timestamp,
        and computes cross-asset features when both MES and MGC are present.

        Parameters
        ----------
        symbol_files : Dict[str, Union[str, Path]]
            Dict mapping symbol names to file paths
            e.g., {'MES': 'path/to/mes.parquet', 'MGC': 'path/to/mgc.parquet'}
        compute_cross_asset : bool, default True
            Whether to compute cross-asset features

        Returns
        -------
        Dict[str, Dict]
            Dictionary mapping symbols to feature reports
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {len(symbol_files)} symbols with cross-asset features")
        logger.info(f"{'='*60}\n")

        # Load all data
        symbol_data = {}
        for symbol, file_path in symbol_files.items():
            file_path = Path(file_path)
            if file_path.exists():
                df = pd.read_parquet(file_path)
                symbol_data[symbol.upper()] = df
                logger.info(f"Loaded {symbol.upper()}: {len(df):,} rows")
            else:
                logger.warning(f"File not found for {symbol}: {file_path}")

        # Check if we can compute cross-asset features
        has_mes = 'MES' in symbol_data
        has_mgc = 'MGC' in symbol_data
        can_compute_cross_asset = compute_cross_asset and has_mes and has_mgc

        cross_asset_data = None

        if can_compute_cross_asset:
            logger.info("Aligning MES and MGC data for cross-asset features...")

            mes_df = symbol_data['MES'].copy()
            mgc_df = symbol_data['MGC'].copy()

            # Set datetime as index for alignment
            mes_df = mes_df.set_index('datetime')
            mgc_df = mgc_df.set_index('datetime')

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

                # Prepare cross-asset data
                cross_asset_data = {
                    'mes_close': mes_aligned['close'].values,
                    'mgc_close': mgc_aligned['close'].values
                }
            else:
                logger.warning("No common timestamps found, skipping cross-asset features")
                can_compute_cross_asset = False

        # Process each symbol
        results = {}
        errors = []
        for symbol, df in symbol_data.items():
            try:
                logger.info(f"\nProcessing {symbol}...")

                # Engineer features with cross-asset data
                df_features, feature_report = self.engineer_features(
                    df,
                    symbol,
                    cross_asset_data=cross_asset_data if can_compute_cross_asset else None
                )

                # Save results
                self.save_features(df_features, symbol, feature_report)
                results[symbol] = feature_report

            except Exception as e:
                errors.append({
                    'symbol': symbol,
                    'error': str(e),
                    'type': type(e).__name__
                })
                logger.error(f"Error processing {symbol}: {e}", exc_info=True)

        if errors:
            error_summary = f"{len(errors)}/{len(symbol_data)} symbols failed"
            logger.error(f"Multi-symbol processing completed with errors: {error_summary}")
            raise RuntimeError(f"{error_summary}. Errors: {errors[:5]}")

        return results


if __name__ == '__main__':
    from .cli import main
    main()
