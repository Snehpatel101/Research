"""
DataCleaner Class - Comprehensive data cleaning for financial time series.

This module provides the main DataCleaner class which handles:
- Gap detection and filling (via GapHandler)
- Duplicate detection and removal
- Outlier detection (z-score, IQR, ATR methods)
- Contract roll handling
- Cleaning report generation
- Batch processing of files

Author: ML Pipeline
Created: 2025-12-20
Updated: 2025-12-22 - Extracted gap handling to gap_handler.py
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .gap_handler import GapHandler
from .utils import calculate_atr_numba, resample_ohlcv, validate_ohlc

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class DataCleaner:
    """
    Comprehensive data cleaning for financial time series.

    Supports Multi-Timeframe (MTF) resampling through the target_timeframe parameter.
    Input data is expected to be 1-minute bars by default (timeframe='1min'), which
    can then be resampled to any target_timeframe (e.g., '5min', '15min', '30min').
    """

    def __init__(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        timeframe: str = '1min',
        target_timeframe: str = '5min',
        gap_fill_method: str = 'forward',
        max_gap_fill_minutes: int = 5,
        outlier_method: str = 'atr',
        atr_threshold: float = 5.0,
        zscore_threshold: float = 5.0,
        iqr_multiplier: float = 3.0,
        calendar_aware: bool = True
    ):
        """
        Initialize data cleaner.

        Parameters:
        -----------
        input_dir : Path to input data directory
        output_dir : Path to output directory
        timeframe : Source data timeframe (e.g., '1min', '5min')
        target_timeframe : Target timeframe for resampling (e.g., '5min', '15min')
            Supported: '1min', '5min', '10min', '15min', '20min', '30min', '45min', '60min'
        gap_fill_method : Method for gap filling ('forward', 'interpolate', 'none')
        max_gap_fill_minutes : Maximum gap to fill (in minutes)
        outlier_method : Outlier detection method ('atr', 'zscore', 'iqr', 'all')
        atr_threshold : ATR multiplier for spike detection
        zscore_threshold : Z-score threshold for outlier detection
        iqr_multiplier : IQR multiplier for outlier detection
        calendar_aware : Whether to use calendar-aware gap handling (skip market closures)

        Raises:
        -------
        ValueError : If target_timeframe is not supported
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        # Validate input directory exists
        if not self.input_dir.exists():
            raise FileNotFoundError(
                f"Input directory does not exist: {self.input_dir}. "
                f"Please run the ingest stage first or provide a valid path."
            )

        self.timeframe = timeframe
        self.target_timeframe = target_timeframe
        self.gap_fill_method = gap_fill_method
        self.max_gap_fill_minutes = max_gap_fill_minutes
        self.outlier_method = outlier_method
        self.atr_threshold = atr_threshold
        self.zscore_threshold = zscore_threshold
        self.iqr_multiplier = iqr_multiplier
        self.calendar_aware = calendar_aware

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Parse source timeframe to minutes
        self.freq_minutes = self._parse_timeframe(timeframe)

        # Validate and parse target timeframe
        self._validate_target_timeframe(target_timeframe)
        self.target_freq_minutes = self._parse_timeframe(target_timeframe)

        # Initialize gap handler (dependency injection)
        self._gap_handler = GapHandler(
            freq_minutes=self.freq_minutes,
            gap_fill_method=self.gap_fill_method,
            max_gap_fill_minutes=self.max_gap_fill_minutes,
            calendar_aware=self.calendar_aware
        )

        logger.info("Initialized DataCleaner")
        logger.info(f"Input dir: {self.input_dir}")
        logger.info(f"Output dir: {self.output_dir}")
        logger.info(f"Source timeframe: {timeframe} ({self.freq_minutes} minutes)")
        logger.info(f"Target timeframe: {target_timeframe} ({self.target_freq_minutes} minutes)")

    def _validate_target_timeframe(self, target_timeframe: str) -> None:
        """
        Validate that target_timeframe is supported.

        Parameters:
        -----------
        target_timeframe : str
            Target timeframe to validate

        Raises:
        -------
        ValueError : If target_timeframe is not supported
        """
        from src.phase1.config import validate_timeframe
        validate_timeframe(target_timeframe)

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

    def detect_gaps(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Detect gaps in time series.

        Delegates to the internal GapHandler for gap detection.

        Parameters:
        -----------
        df : Input DataFrame with datetime index

        Returns:
        --------
        tuple : (DataFrame with gap info, gap report dict)
        """
        return self._gap_handler.detect_gaps(df)

    def fill_gaps(self, df: pd.DataFrame, max_fill_bars: int | None = None) -> pd.DataFrame:
        """
        Fill gaps in time series.

        Delegates to the internal GapHandler for gap filling.

        Parameters:
        -----------
        df : Input DataFrame
        max_fill_bars : Maximum number of bars to fill (None = use max_gap_fill_minutes)

        Returns:
        --------
        pd.DataFrame : DataFrame with filled gaps
        """
        return self._gap_handler.fill_gaps(df, max_fill_bars)

    def detect_duplicates(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
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
            period=period
        )

        # Calculate close-to-close changes
        close_changes = np.abs(df['close'].pct_change())

        # Spikes are where close change exceeds threshold * ATR
        atr_threshold = (threshold * atr / df['close'].values)
        spikes = close_changes > atr_threshold

        return spikes.fillna(False).values

    def clean_outliers(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Detect and remove outliers based on configured method.

        Parameters:
        -----------
        df : Input DataFrame with OHLC data

        Returns:
        --------
        tuple : (Cleaned DataFrame, outlier report)
        """
        logger.info(f"Detecting outliers using method: {self.outlier_method}")

        df = df.copy()
        initial_rows = len(df)

        outliers_by_method = {}
        total_outlier_mask = pd.Series(False, index=df.index)

        if self.outlier_method in ['atr', 'all']:
            spike_mask = self.detect_spikes_atr(df, self.atr_threshold)
            outliers_by_method['atr'] = int(spike_mask.sum())
            total_outlier_mask |= spike_mask

        if self.outlier_method in ['zscore', 'all']:
            zscore_mask = self.detect_outliers_zscore(df['close'].pct_change().fillna(0), self.zscore_threshold)
            outliers_by_method['zscore'] = int(zscore_mask.sum())
            total_outlier_mask |= zscore_mask

        if self.outlier_method in ['iqr', 'all']:
            iqr_mask = self.detect_outliers_iqr(df['close'].pct_change().fillna(0), self.iqr_multiplier)
            outliers_by_method['iqr'] = int(iqr_mask.sum())
            total_outlier_mask |= iqr_mask

        n_outliers = total_outlier_mask.sum()
        outlier_report = {
            'method': self.outlier_method,
            'total_outliers': int(n_outliers),
            'outlier_pct': (n_outliers / len(df) * 100) if len(df) > 0 else 0,
            'by_method': outliers_by_method
        }

        if n_outliers > 0:
            logger.warning(f"Found {n_outliers} outliers ({outlier_report['outlier_pct']:.2f}%)")
            df = df[~total_outlier_mask].reset_index(drop=True)
            logger.info(f"After outlier removal: {len(df):,} rows")

        return df, outlier_report

    def handle_contract_rolls(self, df: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, dict]:
        """
        Detect and handle contract rolls for futures data.

        Parameters:
        -----------
        df : Input DataFrame with OHLC data
        symbol : Symbol name

        Returns:
        --------
        tuple : (Processed DataFrame, contract roll report)
        """
        logger.info("Checking for contract rolls...")

        df = df.copy()

        # Detect large price gaps (potential contract rolls)
        price_changes = df['close'].pct_change().abs()
        large_gap_threshold = 0.10  # 10% gap
        potential_rolls = price_changes > large_gap_threshold

        roll_report = {
            'potential_rolls': int(potential_rolls.sum()),
            'details': []
        }

        if potential_rolls.any():
            roll_indices = np.where(potential_rolls)[0]
            for idx in roll_indices:
                roll_report['details'].append({
                    'datetime': str(df.iloc[idx]['datetime']),
                    'price_change_pct': float(price_changes.iloc[idx] * 100)
                })
            logger.warning(f"Detected {potential_rolls.sum()} potential contract rolls")

        return df, roll_report

    def resample_data(
        self,
        df: pd.DataFrame,
        target_timeframe: str | None = None,
        include_metadata: bool = True
    ) -> pd.DataFrame:
        """
        Resample data to the target timeframe.

        This method uses the generic resample_ohlcv function to convert data
        from the source timeframe to the target timeframe. If no target_timeframe
        is provided, uses the instance's target_timeframe attribute.

        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame with OHLCV data
        target_timeframe : str, optional
            Target timeframe for resampling. If None, uses self.target_timeframe
        include_metadata : bool
            If True, adds 'timeframe' column to output

        Returns:
        --------
        pd.DataFrame : Resampled OHLCV data

        Raises:
        -------
        ValueError : If target timeframe is not valid or data is empty

        Examples:
        ---------
        >>> cleaner = DataCleaner(input_dir, output_dir, target_timeframe='15min')
        >>> df_15min = cleaner.resample_data(df_1min)
        >>> df_30min = cleaner.resample_data(df_1min, target_timeframe='30min')
        """
        if target_timeframe is None:
            target_timeframe = self.target_timeframe

        # Skip resampling if source equals target
        source_minutes = self.freq_minutes
        target_minutes = self._parse_timeframe(target_timeframe)

        if source_minutes == target_minutes:
            logger.info(f"Source ({self.timeframe}) equals target ({target_timeframe}), skipping resampling")
            if include_metadata:
                df = df.copy()
                df['timeframe'] = target_timeframe
            return df

        # Validate target >= source
        if target_minutes < source_minutes:
            raise ValueError(
                f"Cannot downsample from {self.timeframe} ({source_minutes}min) "
                f"to {target_timeframe} ({target_minutes}min). "
                f"Target timeframe must be >= source timeframe."
            )

        # Use the generic resampling function
        return resample_ohlcv(df, target_timeframe, include_metadata)

    def clean_file(self, file_path: Path) -> tuple[pd.DataFrame, dict]:
        """
        Complete cleaning pipeline for a single file.

        Parameters:
        -----------
        file_path : Path to input file

        Returns:
        --------
        tuple : (Cleaned DataFrame, cleaning report)
        """
        logger.info(f"\nCleaning {file_path.name}...")

        # Load data
        if str(file_path).endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_parquet(file_path)

        # Parse symbol from filename
        symbol = file_path.stem
        logger.info(f"Symbol: {symbol}, Initial rows: {len(df):,}")

        initial_rows = len(df)

        # Ensure datetime column
        if 'datetime' not in df.columns:
            for col in ['timestamp', 'date', 'time', 'DateTime', 'Timestamp']:
                if col in df.columns:
                    df = df.rename(columns={col: 'datetime'})
                    break

        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)

        # Step 1: Validate OHLC
        df = validate_ohlc(df)

        # Step 2: Detect duplicates
        df, duplicate_report = self.detect_duplicates(df)

        # Step 3: Detect gaps
        df, gap_report = self.detect_gaps(df)

        # Step 4: Fill gaps
        df = self.fill_gaps(df)

        gap_filling_report = {'rows_added': max(0, len(df) - (initial_rows - duplicate_report['n_duplicates']))}

        # Step 5: Clean outliers
        df, outlier_report = self.clean_outliers(df)

        # Step 6: Handle contract rolls
        df, roll_report = self.handle_contract_rolls(df, symbol)

        # Step 7: Final OHLC validation
        df = validate_ohlc(df)

        # Add symbol column
        df['symbol'] = symbol

        final_rows = len(df)
        retention_pct = (final_rows / initial_rows * 100) if initial_rows > 0 else 0

        # Compile cleaning report
        cleaning_report = {
            'symbol': symbol,
            'initial_rows': initial_rows,
            'final_rows': final_rows,
            'retention_pct': retention_pct,
            'duplicates': duplicate_report,
            'gaps': gap_report,
            'gap_filling': gap_filling_report,
            'outliers': outlier_report,
            'contract_rolls': roll_report
        }

        logger.info(f"Final rows: {final_rows:,} (retention: {retention_pct:.2f}%)")

        return df, cleaning_report

    def save_results(self, df: pd.DataFrame, symbol: str, cleaning_report: dict) -> tuple[Path, Path]:
        """
        Save cleaned data and report.

        Parameters:
        -----------
        df : Cleaned DataFrame
        symbol : Symbol name
        cleaning_report : Cleaning report dictionary

        Returns:
        --------
        tuple : (data_path, report_path)
        """
        # Save cleaned data
        data_path = self.output_dir / f"{symbol}_clean.parquet"
        df.to_parquet(data_path, index=False)
        logger.info(f"Saved cleaned data to: {data_path}")

        # Save cleaning report
        report_path = self.output_dir / f"{symbol}_cleaning_report.json"
        with open(report_path, 'w') as f:
            json.dump(cleaning_report, f, indent=2, default=str)
        logger.info(f"Saved cleaning report to: {report_path}")

        return data_path, report_path

    def clean_directory(self, pattern: str = "*.parquet") -> dict[str, dict]:
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
        errors = []

        for file_path in files:
            try:
                df, cleaning_report = self.clean_file(file_path)
                symbol = cleaning_report['symbol']
                self.save_results(df, symbol, cleaning_report)
                results[symbol] = cleaning_report

            except Exception as e:
                errors.append({
                    'file': str(file_path.name),
                    'error': str(e),
                    'type': type(e).__name__
                })
                logger.error(f"Error processing {file_path.name}: {e}", exc_info=True)

        if errors:
            error_summary = f"{len(errors)}/{len(files)} files failed cleaning"
            logger.error(f"Cleaning completed with errors: {error_summary}")
            raise RuntimeError(f"{error_summary}. Errors: {errors[:5]}")

        # Save combined cleaning report
        combined_report_path = self.output_dir / "cleaning_report.json"
        with open(combined_report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nSaved combined report to: {combined_report_path}")

        return results
