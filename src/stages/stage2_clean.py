"""
Stage 2: Data Cleaning Module
Production-ready data cleaning with gap detection, outlier removal, and quality checks.

This module handles:
- Gap detection and quantification
- Gap filling strategies (forward fill, interpolation)
- Duplicate timestamp detection and removal
- Outlier detection (z-score, IQR methods)
- Spike removal (ATR-based)
- Contract roll/stitch handling for futures
- Comprehensive quality reporting
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
from numba import jit

# Configure logging - use NullHandler to avoid duplicate logs when imported as module
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@jit(nopython=True)
def calculate_atr_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Average True Range using Numba for performance.

    Parameters:
    -----------
    high : High prices
    low : Low prices
    close : Close prices
    period : ATR period

    Returns:
    --------
    np.ndarray : ATR values
    """
    n = len(high)
    tr = np.zeros(n)
    atr = np.zeros(n)

    # Calculate True Range
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)

    # Calculate ATR
    atr[period] = np.mean(tr[1:period+1])

    for i in range(period + 1, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period

    return atr


class DataCleaner:
    """
    Comprehensive data cleaning for financial time series.
    """

    def __init__(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        timeframe: str = '1min',
        gap_fill_method: str = 'forward',
        max_gap_fill_minutes: int = 5,
        outlier_method: str = 'atr',
        atr_threshold: float = 5.0,
        zscore_threshold: float = 5.0,
        iqr_multiplier: float = 3.0
    ):
        """
        Initialize data cleaner.

        Parameters:
        -----------
        input_dir : Path to input data directory
        output_dir : Path to output directory
        timeframe : Data timeframe (e.g., '1min', '5min')
        gap_fill_method : Method for gap filling ('forward', 'interpolate', 'none')
        max_gap_fill_minutes : Maximum gap to fill (in minutes)
        outlier_method : Outlier detection method ('atr', 'zscore', 'iqr', 'all')
        atr_threshold : ATR multiplier for spike detection
        zscore_threshold : Z-score threshold for outlier detection
        iqr_multiplier : IQR multiplier for outlier detection
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.timeframe = timeframe
        self.gap_fill_method = gap_fill_method
        self.max_gap_fill_minutes = max_gap_fill_minutes
        self.outlier_method = outlier_method
        self.atr_threshold = atr_threshold
        self.zscore_threshold = zscore_threshold
        self.iqr_multiplier = iqr_multiplier

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Parse timeframe to minutes
        self.freq_minutes = self._parse_timeframe(timeframe)

        logger.info(f"Initialized DataCleaner")
        logger.info(f"Input dir: {self.input_dir}")
        logger.info(f"Output dir: {self.output_dir}")
        logger.info(f"Timeframe: {timeframe} ({self.freq_minutes} minutes)")

    def _parse_timeframe(self, timeframe: str) -> int:
        """Parse timeframe string to minutes."""
        timeframe = timeframe.lower()
        if 'min' in timeframe:
            return int(timeframe.replace('min', ''))
        elif 'h' in timeframe:
            return int(timeframe.replace('h', '')) * 60
        elif 'd' in timeframe:
            return int(timeframe.replace('d', '')) * 60 * 24
        else:
            return 1  # Default to 1 minute

    def detect_gaps(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Detect gaps in time series.

        Parameters:
        -----------
        df : Input DataFrame with datetime index

        Returns:
        --------
        tuple : (DataFrame with gap info, gap report dict)
        """
        logger.info("Detecting gaps in time series...")

        df = df.copy()
        df = df.sort_values('datetime').reset_index(drop=True)

        # Calculate time differences
        df['time_diff'] = df['datetime'].diff()

        # Expected frequency
        expected_freq = pd.Timedelta(minutes=self.freq_minutes)

        # Identify gaps (where time_diff > expected_freq)
        gap_mask = df['time_diff'] > expected_freq

        gaps = []
        if gap_mask.any():
            gap_indices = np.where(gap_mask)[0]

            for idx in gap_indices:
                gap_start = df.loc[idx - 1, 'datetime']
                gap_end = df.loc[idx, 'datetime']
                gap_duration = df.loc[idx, 'time_diff']
                missing_bars = int(gap_duration / expected_freq) - 1

                gaps.append({
                    'gap_start': gap_start,
                    'gap_end': gap_end,
                    'duration': str(gap_duration),
                    'missing_bars': missing_bars
                })

        # Gap report
        total_expected_bars = (df['datetime'].max() - df['datetime'].min()) / expected_freq
        total_actual_bars = len(df)
        total_missing_bars = int(total_expected_bars - total_actual_bars)

        gap_report = {
            'total_gaps': len(gaps),
            'total_missing_bars': total_missing_bars,
            'expected_bars': int(total_expected_bars),
            'actual_bars': total_actual_bars,
            'completeness_pct': (total_actual_bars / total_expected_bars * 100) if total_expected_bars > 0 else 100,
            'gaps': gaps[:100]  # Limit to first 100 gaps for reporting
        }

        logger.info(f"Found {len(gaps)} gaps")
        logger.info(f"Total missing bars: {total_missing_bars:,}")
        logger.info(f"Completeness: {gap_report['completeness_pct']:.2f}%")

        # Drop temp column
        df = df.drop('time_diff', axis=1)

        return df, gap_report

    def fill_gaps(self, df: pd.DataFrame, max_fill_bars: Optional[int] = None) -> pd.DataFrame:
        """
        Fill gaps in time series.

        Parameters:
        -----------
        df : Input DataFrame
        max_fill_bars : Maximum number of bars to fill (None = use max_gap_fill_minutes)

        Returns:
        --------
        pd.DataFrame : DataFrame with filled gaps
        """
        if self.gap_fill_method == 'none':
            logger.info("Gap filling disabled")
            return df

        logger.info(f"Filling gaps using method: {self.gap_fill_method}")

        df = df.copy()
        df = df.sort_values('datetime').reset_index(drop=True)

        if max_fill_bars is None:
            max_fill_bars = self.max_gap_fill_minutes // self.freq_minutes

        # Create complete datetime range
        date_range = pd.date_range(
            start=df['datetime'].min(),
            end=df['datetime'].max(),
            freq=f'{self.freq_minutes}min'
        )

        # Reindex to complete range
        df_complete = df.set_index('datetime').reindex(date_range).reset_index()
        df_complete = df_complete.rename(columns={'index': 'datetime'})

        # Identify which rows were filled
        filled_mask = df_complete['close'].isna()
        n_filled = filled_mask.sum()

        if n_filled > 0:
            logger.info(f"Filling {n_filled:,} missing bars...")

            if self.gap_fill_method == 'forward':
                # Forward fill with limit
                df_complete[['open', 'high', 'low', 'close']] = \
                    df_complete[['open', 'high', 'low', 'close']].ffill(limit=max_fill_bars)
                df_complete['volume'] = df_complete['volume'].fillna(0)

            elif self.gap_fill_method == 'interpolate':
                # Linear interpolation
                df_complete[['open', 'high', 'low', 'close']] = \
                    df_complete[['open', 'high', 'low', 'close']].interpolate(method='linear', limit=max_fill_bars)
                df_complete['volume'] = df_complete['volume'].fillna(0)

            # Copy symbol column if exists
            if 'symbol' in df_complete.columns:
                df_complete['symbol'] = df_complete['symbol'].ffill()

            # Drop any remaining NaN rows (gaps too large to fill)
            remaining_na = df_complete['close'].isna().sum()
            if remaining_na > 0:
                logger.info(f"Dropping {remaining_na} rows with gaps > {max_fill_bars} bars")
                df_complete = df_complete.dropna(subset=['close'])

        return df_complete

    def detect_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Detect and remove duplicate timestamps.

        Parameters:
        -----------
        df : Input DataFrame

        Returns:
        --------
        tuple : (DataFrame without duplicates, duplicate report)
        """
        logger.info("Detecting duplicate timestamps...")

        df = df.copy()

        # Find duplicates
        duplicate_mask = df.duplicated(subset=['datetime'], keep='first')
        n_duplicates = duplicate_mask.sum()

        duplicate_report = {
            'n_duplicates': int(n_duplicates),
            'duplicate_pct': (n_duplicates / len(df) * 100) if len(df) > 0 else 0
        }

        if n_duplicates > 0:
            logger.warning(f"Found {n_duplicates} duplicate timestamps ({duplicate_report['duplicate_pct']:.2f}%)")

            # Get some examples
            duplicate_examples = df[df.duplicated(subset=['datetime'], keep=False)].head(10)
            duplicate_report['examples'] = duplicate_examples['datetime'].astype(str).tolist()

            # Remove duplicates (keep first occurrence)
            df = df[~duplicate_mask].reset_index(drop=True)
            logger.info(f"Removed duplicates. Remaining rows: {len(df):,}")
        else:
            logger.info("No duplicates found")

        return df, duplicate_report

    def detect_outliers_zscore(self, series: pd.Series, threshold: float = 5.0) -> np.ndarray:
        """
        Detect outliers using z-score method.

        Parameters:
        -----------
        series : Data series
        threshold : Z-score threshold

        Returns:
        --------
        np.ndarray : Boolean mask of outliers
        """
        mean = series.mean()
        std = series.std()

        if std == 0:
            return np.zeros(len(series), dtype=bool)

        z_scores = np.abs((series - mean) / std)
        return z_scores > threshold

    def detect_outliers_iqr(self, series: pd.Series, multiplier: float = 3.0) -> np.ndarray:
        """
        Detect outliers using IQR method.

        Parameters:
        -----------
        series : Data series
        multiplier : IQR multiplier

        Returns:
        --------
        np.ndarray : Boolean mask of outliers
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        return (series < lower_bound) | (series > upper_bound)

    def detect_spikes_atr(self, df: pd.DataFrame, threshold: float = 5.0, period: int = 14) -> np.ndarray:
        """
        Detect price spikes using ATR method.

        Parameters:
        -----------
        df : DataFrame with OHLC data
        threshold : ATR multiplier threshold
        period : ATR period

        Returns:
        --------
        np.ndarray : Boolean mask of spikes
        """
        # Calculate ATR using Numba
        atr = calculate_atr_numba(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            period
        )

        # Calculate price changes
        price_change = np.abs(df['close'].diff().values)

        # Detect spikes (price change > threshold * ATR)
        spikes = np.zeros(len(df), dtype=bool)
        spikes[period:] = price_change[period:] > (threshold * atr[period:])

        return spikes

    def clean_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Detect and remove outliers/spikes.

        Parameters:
        -----------
        df : Input DataFrame

        Returns:
        --------
        tuple : (Cleaned DataFrame, outlier report)
        """
        logger.info(f"Detecting outliers using method: {self.outlier_method}")

        df = df.copy()
        initial_rows = len(df)

        outlier_mask = np.zeros(len(df), dtype=bool)

        outlier_report = {
            'methods': {},
            'total_outliers': 0,
            'outlier_pct': 0
        }

        # ATR-based spike detection
        if self.outlier_method in ['atr', 'all']:
            spike_mask = self.detect_spikes_atr(df, threshold=self.atr_threshold)
            n_spikes = spike_mask.sum()

            outlier_report['methods']['atr_spikes'] = {
                'n_outliers': int(n_spikes),
                'threshold': self.atr_threshold
            }

            logger.info(f"ATR method: Found {n_spikes} spikes")
            outlier_mask |= spike_mask

        # Z-score based detection
        if self.outlier_method in ['zscore', 'all']:
            # Check returns instead of raw prices
            returns = df['close'].pct_change()
            zscore_mask = self.detect_outliers_zscore(returns, threshold=self.zscore_threshold)
            n_zscore = zscore_mask.sum()

            outlier_report['methods']['zscore'] = {
                'n_outliers': int(n_zscore),
                'threshold': self.zscore_threshold
            }

            logger.info(f"Z-score method: Found {n_zscore} outliers")
            outlier_mask |= zscore_mask

        # IQR based detection
        if self.outlier_method in ['iqr', 'all']:
            returns = df['close'].pct_change()
            iqr_mask = self.detect_outliers_iqr(returns, multiplier=self.iqr_multiplier)
            n_iqr = iqr_mask.sum()

            outlier_report['methods']['iqr'] = {
                'n_outliers': int(n_iqr),
                'threshold': self.iqr_multiplier
            }

            logger.info(f"IQR method: Found {n_iqr} outliers")
            outlier_mask |= iqr_mask

        # Remove outliers
        total_outliers = outlier_mask.sum()
        outlier_report['total_outliers'] = int(total_outliers)
        outlier_report['outlier_pct'] = (total_outliers / initial_rows * 100) if initial_rows > 0 else 0

        if total_outliers > 0:
            logger.warning(f"Removing {total_outliers} total outliers ({outlier_report['outlier_pct']:.2f}%)")

            # Store outlier examples
            outlier_examples = df[outlier_mask].head(10)
            outlier_report['examples'] = outlier_examples[['datetime', 'close']].to_dict('records')

            # Remove outliers
            df = df[~outlier_mask].reset_index(drop=True)
            logger.info(f"Remaining rows: {len(df):,}")
        else:
            logger.info("No outliers detected")

        return df, outlier_report

    def handle_contract_rolls(self, df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Detect and handle contract rolls for futures.

        Parameters:
        -----------
        df : Input DataFrame
        symbol : Symbol name

        Returns:
        --------
        tuple : (Adjusted DataFrame, roll report)
        """
        logger.info("Checking for contract rolls...")

        df = df.copy()

        # Calculate returns
        returns = df['close'].pct_change()

        # Detect large gaps that might be rolls (e.g., > 5% overnight move)
        roll_threshold = 0.05  # 5%
        potential_rolls = np.abs(returns) > roll_threshold

        n_potential_rolls = potential_rolls.sum()

        roll_report = {
            'potential_rolls': int(n_potential_rolls),
            'roll_dates': []
        }

        if n_potential_rolls > 0:
            logger.info(f"Found {n_potential_rolls} potential contract rolls")

            roll_indices = np.where(potential_rolls)[0]
            for idx in roll_indices:
                roll_date = df.loc[idx, 'datetime']
                price_jump = returns.iloc[idx]
                roll_report['roll_dates'].append({
                    'date': str(roll_date),
                    'price_jump_pct': float(price_jump * 100)
                })

            # Note: Actual roll adjustment would require contract-specific logic
            # For now, we just report potential rolls
            logger.info("Contract rolls detected but not adjusted (requires contract-specific logic)")
        else:
            logger.info("No contract rolls detected")

        return df, roll_report

    def clean_file(
        self,
        file_path: Union[str, Path],
        symbol: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete cleaning pipeline for a single file.

        Parameters:
        -----------
        file_path : Path to input file
        symbol : Symbol name

        Returns:
        --------
        tuple : (Cleaned DataFrame, cleaning report)
        """
        file_path = Path(file_path)

        if symbol is None:
            symbol = file_path.stem.split('_')[0].upper()

        logger.info(f"\n{'='*60}")
        logger.info(f"Starting cleaning for symbol: {symbol}")
        logger.info(f"{'='*60}\n")

        # Load data
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded {len(df):,} rows")

        cleaning_report = {
            'symbol': symbol,
            'source_file': str(file_path),
            'cleaning_timestamp': datetime.now().isoformat(),
            'initial_rows': len(df),
            'initial_date_range': {
                'start': df['datetime'].min().isoformat(),
                'end': df['datetime'].max().isoformat()
            }
        }

        # Step 1: Detect and remove duplicates
        df, duplicate_report = self.detect_duplicates(df)
        cleaning_report['duplicates'] = duplicate_report

        # Step 2: Detect gaps
        df, gap_report = self.detect_gaps(df)
        cleaning_report['gaps'] = gap_report

        # Step 3: Fill gaps
        rows_before_fill = len(df)
        df = self.fill_gaps(df)
        cleaning_report['gap_filling'] = {
            'method': self.gap_fill_method,
            'max_fill_bars': self.max_gap_fill_minutes // self.freq_minutes,
            'rows_added': len(df) - rows_before_fill
        }

        # Step 4: Clean outliers
        df, outlier_report = self.clean_outliers(df)
        cleaning_report['outliers'] = outlier_report

        # Step 5: Handle contract rolls (for futures)
        if any(x in symbol.upper() for x in ['MES', 'MNQ', 'MGC', 'ES', 'NQ', 'GC']):
            df, roll_report = self.handle_contract_rolls(df, symbol)
            cleaning_report['contract_rolls'] = roll_report

        # Final statistics
        cleaning_report['final_rows'] = len(df)
        cleaning_report['final_date_range'] = {
            'start': df['datetime'].min().isoformat(),
            'end': df['datetime'].max().isoformat()
        }
        cleaning_report['rows_removed'] = cleaning_report['initial_rows'] - len(df) + cleaning_report['gap_filling']['rows_added']
        cleaning_report['retention_pct'] = (len(df) / cleaning_report['initial_rows'] * 100) if cleaning_report['initial_rows'] > 0 else 0

        logger.info(f"\nCleaning complete for {symbol}")
        logger.info(f"Rows: {cleaning_report['initial_rows']:,} -> {cleaning_report['final_rows']:,}")
        logger.info(f"Retention: {cleaning_report['retention_pct']:.2f}%")

        return df, cleaning_report

    def save_results(
        self,
        df: pd.DataFrame,
        symbol: str,
        cleaning_report: Dict
    ) -> Tuple[Path, Path]:
        """
        Save cleaned data and report.

        Parameters:
        -----------
        df : Cleaned DataFrame
        symbol : Symbol name
        cleaning_report : Cleaning report dict

        Returns:
        --------
        tuple : (data_path, report_path)
        """
        # Save cleaned data
        data_path = self.output_dir / f"{symbol}.parquet"
        df.to_parquet(
            data_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        logger.info(f"Saved cleaned data to: {data_path}")

        # Save cleaning report
        report_path = self.output_dir / f"{symbol}_cleaning_report.json"
        with open(report_path, 'w') as f:
            json.dump(cleaning_report, f, indent=2, default=str)
        logger.info(f"Saved cleaning report to: {report_path}")

        return data_path, report_path

    def clean_directory(self, pattern: str = "*.parquet") -> Dict[str, Dict]:
        """
        Clean all files in input directory.

        Parameters:
        -----------
        pattern : File pattern to match

        Returns:
        --------
        dict : Dictionary mapping symbols to cleaning reports
        """
        files = list(self.input_dir.glob(pattern))

        if not files:
            logger.warning(f"No files found matching pattern: {pattern}")
            return {}

        logger.info(f"Found {len(files)} files to clean")

        results = {}

        for file_path in files:
            try:
                df, cleaning_report = self.clean_file(file_path)
                symbol = cleaning_report['symbol']
                self.save_results(df, symbol, cleaning_report)
                results[symbol] = cleaning_report

            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {e}", exc_info=True)
                continue

        # Save combined cleaning report
        combined_report_path = self.output_dir / "cleaning_report.json"
        with open(combined_report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nSaved combined report to: {combined_report_path}")

        return results


def main():
    """
    Example usage of DataCleaner.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Stage 2: Data Cleaning')
    parser.add_argument('--input-dir', type=str, default='data/raw',
                        help='Input data directory')
    parser.add_argument('--output-dir', type=str, default='data/clean',
                        help='Output directory')
    parser.add_argument('--timeframe', type=str, default='1min',
                        help='Data timeframe (e.g., 1min, 5min)')
    parser.add_argument('--gap-fill', type=str, default='forward',
                        choices=['forward', 'interpolate', 'none'],
                        help='Gap filling method')
    parser.add_argument('--max-gap-fill', type=int, default=5,
                        help='Maximum gap to fill (minutes)')
    parser.add_argument('--outlier-method', type=str, default='atr',
                        choices=['atr', 'zscore', 'iqr', 'all'],
                        help='Outlier detection method')
    parser.add_argument('--atr-threshold', type=float, default=5.0,
                        help='ATR threshold for spike detection')
    parser.add_argument('--pattern', type=str, default='*.parquet',
                        help='File pattern to match')

    args = parser.parse_args()

    # Initialize cleaner
    cleaner = DataCleaner(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        timeframe=args.timeframe,
        gap_fill_method=args.gap_fill,
        max_gap_fill_minutes=args.max_gap_fill,
        outlier_method=args.outlier_method,
        atr_threshold=args.atr_threshold
    )

    # Clean all files
    results = cleaner.clean_directory(pattern=args.pattern)

    print("\n" + "="*60)
    print("CLEANING SUMMARY")
    print("="*60)
    for symbol, report in results.items():
        print(f"\n{symbol}:")
        print(f"  Rows: {report['initial_rows']:,} -> {report['final_rows']:,}")
        print(f"  Retention: {report['retention_pct']:.2f}%")
        print(f"  Duplicates removed: {report['duplicates']['n_duplicates']}")
        print(f"  Gaps found: {report['gaps']['total_gaps']}")
        print(f"  Gap filling: {report['gap_filling']['rows_added']} bars added")
        print(f"  Outliers removed: {report['outliers']['total_outliers']}")


if __name__ == '__main__':
    main()
