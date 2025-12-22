"""
Main FeatureEngineer class for feature engineering pipeline.

This module provides the FeatureEngineer class that orchestrates
all feature engineering operations and manages the complete pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict, Tuple, Optional
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
from .volume import add_volume_features, add_vwap
from .trend import add_adx, add_supertrend
from .temporal import add_temporal_features
from .regime import add_regime_features
from .cross_asset import add_cross_asset_features

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
    """

    def __init__(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        timeframe: str = '1min'
    ):
        """
        Initialize feature engineer.

        Parameters
        ----------
        input_dir : Union[str, Path]
            Path to cleaned data directory
        output_dir : Union[str, Path]
            Path to output directory
        timeframe : str, default '1min'
            Data timeframe (e.g., '1min', '5min')
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.timeframe = timeframe

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Feature metadata
        self.feature_metadata: Dict[str, str] = {}

        logger.info("Initialized FeatureEngineer")
        logger.info(f"Input dir: {self.input_dir}")
        logger.info(f"Output dir: {self.output_dir}")

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

        # Add all features
        df = add_returns(df, self.feature_metadata)
        df = add_price_ratios(df, self.feature_metadata)
        df = add_sma(df, self.feature_metadata)
        df = add_ema(df, self.feature_metadata)
        df = add_rsi(df, self.feature_metadata)
        df = add_macd(df, self.feature_metadata)
        df = add_stochastic(df, self.feature_metadata)
        df = add_williams_r(df, self.feature_metadata)
        df = add_roc(df, self.feature_metadata)
        df = add_cci(df, self.feature_metadata)
        df = add_mfi(df, self.feature_metadata)
        df = add_atr(df, self.feature_metadata)
        df = add_bollinger_bands(df, self.feature_metadata)
        df = add_keltner_channels(df, self.feature_metadata)
        df = add_historical_volatility(df, self.feature_metadata)
        df = add_parkinson_volatility(df, self.feature_metadata)
        df = add_garman_klass_volatility(df, self.feature_metadata)
        df = add_volume_features(df, self.feature_metadata)
        df = add_vwap(df, self.feature_metadata)
        df = add_adx(df, self.feature_metadata)
        df = add_supertrend(df, self.feature_metadata)
        df = add_temporal_features(df, self.feature_metadata)
        df = add_regime_features(df, self.feature_metadata)

        # Add cross-asset features (MES-MGC)
        mes_close = None
        mgc_close = None
        if cross_asset_data is not None:
            mes_close = cross_asset_data.get('mes_close')
            mgc_close = cross_asset_data.get('mgc_close')
        df = add_cross_asset_features(
            df,
            self.feature_metadata,
            mes_close=mes_close,
            mgc_close=mgc_close,
            current_symbol=symbol
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
        has_cross_asset = (cross_asset_data is not None and
                          'mes_close' in cross_asset_data and
                          'mgc_close' in cross_asset_data)

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
            'cross_asset_feature_names': cross_asset_cols if has_cross_asset else [],
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

    # =========================================================================
    # Wrapper methods for testing - delegate to standalone functions
    # =========================================================================

    def add_sma(self, df: pd.DataFrame) -> pd.DataFrame:
        """Wrapper for add_sma function."""
        return add_sma(df, self.feature_metadata)

    def add_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Wrapper for add_ema function."""
        return add_ema(df, self.feature_metadata)

    def add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Wrapper for add_rsi function."""
        return add_rsi(df, self.feature_metadata)

    def add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Wrapper for add_macd function."""
        return add_macd(df, self.feature_metadata)

    def add_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Wrapper for add_stochastic function."""
        return add_stochastic(df, self.feature_metadata)

    def add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Wrapper for add_atr function."""
        return add_atr(df, self.feature_metadata)

    def add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Wrapper for add_bollinger_bands function."""
        return add_bollinger_bands(df, self.feature_metadata)

    def add_keltner_channels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Wrapper for add_keltner_channels function."""
        return add_keltner_channels(df, self.feature_metadata)

    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Wrapper for add_volume_features function."""
        return add_volume_features(df, self.feature_metadata)

    def add_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Wrapper for add_vwap function."""
        return add_vwap(df, self.feature_metadata)

    def add_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """Wrapper for add_adx function."""
        return add_adx(df, self.feature_metadata)

    def add_mfi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Wrapper for add_mfi function."""
        return add_mfi(df, self.feature_metadata)

    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Wrapper for add_returns function."""
        return add_returns(df, self.feature_metadata)

    def add_price_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Wrapper for add_price_ratios function."""
        return add_price_ratios(df, self.feature_metadata)


def main():
    """
    Example usage of FeatureEngineer.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Stage 3: Feature Engineering')
    parser.add_argument('--input-dir', type=str, default='data/clean',
                        help='Input data directory')
    parser.add_argument('--output-dir', type=str, default='data/features',
                        help='Output directory')
    parser.add_argument('--timeframe', type=str, default='1min',
                        help='Data timeframe')
    parser.add_argument('--pattern', type=str, default='*.parquet',
                        help='File pattern to match')
    parser.add_argument('--multi-symbol', action='store_true',
                        help='Process MES and MGC together with cross-asset features')
    parser.add_argument('--symbols', type=str, nargs='+', default=['MES', 'MGC'],
                        help='Symbols to process (for multi-symbol mode)')

    args = parser.parse_args()

    # Initialize feature engineer
    engineer = FeatureEngineer(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        timeframe=args.timeframe
    )

    if args.multi_symbol:
        # Build symbol_files dict
        input_dir = Path(args.input_dir)
        symbol_files = {}
        for symbol in args.symbols:
            # Look for files matching the symbol
            matches = list(input_dir.glob(f"{symbol.lower()}*.parquet")) + \
                      list(input_dir.glob(f"{symbol.upper()}*.parquet"))
            if matches:
                symbol_files[symbol.upper()] = matches[0]
                print(f"Found {symbol.upper()}: {matches[0]}")
            else:
                print(f"Warning: No file found for {symbol}")

        if symbol_files:
            results = engineer.process_multi_symbol(symbol_files, compute_cross_asset=True)
        else:
            print("No files found for specified symbols")
            results = {}
    else:
        # Process all files independently
        results = engineer.process_directory(pattern=args.pattern)

    print("\n" + "="*60)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*60)
    for symbol, report in results.items():
        print(f"\n{symbol}:")
        print(f"  Features added: {report['features_added']}")
        print(f"  Final columns: {report['final_columns']}")
        print(f"  Final rows: {report['final_rows']:,}")
        print(f"  Cross-asset features: {report.get('cross_asset_features', False)}")
        print(f"  Date range: {report['date_range']['start']} to {report['date_range']['end']}")


if __name__ == '__main__':
    main()
