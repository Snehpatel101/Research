"""
Stage 1: Data Ingestion Module
Production-ready data ingestion with standardization and validation.

This module handles:
- Loading raw OHLCV data from CSV or Parquet
- Column name standardization
- Timezone conversion to UTC
- Data type validation
- Output to standardized Parquet format
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict, List, Optional, Tuple
import logging
from datetime import datetime
import pytz

# Configure logging - use NullHandler to avoid duplicate logs when imported as module
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class SecurityError(Exception):
    """Exception raised for security violations such as path traversal attempts."""
    pass


class DataIngestor:
    """
    Handles ingestion of raw market data with standardization and validation.
    """

    # Standard column names
    STANDARD_COLS = ['datetime', 'open', 'high', 'low', 'close', 'volume']

    # Common column name mappings (case-insensitive)
    COLUMN_MAPPINGS = {
        'timestamp': 'datetime',
        'time': 'datetime',
        'date': 'datetime',
        'dt': 'datetime',
        'o': 'open',
        'h': 'high',
        'l': 'low',
        'c': 'close',
        'v': 'volume',
        'vol': 'volume',
        'trade_volume': 'volume',
    }

    # Timezone mappings
    TIMEZONE_MAP = {
        'EST': 'America/New_York',
        'EDT': 'America/New_York',
        'CST': 'America/Chicago',
        'CDT': 'America/Chicago',
        'PST': 'America/Los_Angeles',
        'PDT': 'America/Los_Angeles',
        'GMT': 'GMT',
        'UTC': 'UTC',
    }

    def __init__(
        self,
        raw_data_dir: Union[str, Path],
        output_dir: Union[str, Path],
        source_timezone: str = 'UTC',
        symbol_col: Optional[str] = 'symbol'
    ):
        """
        Initialize data ingestor.

        Parameters:
        -----------
        raw_data_dir : Path to raw data directory
        output_dir : Path to output directory
        source_timezone : Source timezone of the data (default: 'UTC')
        symbol_col : Column name containing symbol (default: 'symbol')
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.source_timezone = source_timezone
        self.symbol_col = symbol_col

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized DataIngestor")
        logger.info(f"Raw data dir: {self.raw_data_dir}")
        logger.info(f"Output dir: {self.output_dir}")

    def _validate_path(self, file_path: Path, allowed_dirs: List[Path]) -> Path:
        """
        Validate that a file path is safe and within allowed directories.

        Parameters:
        -----------
        file_path : Path to validate
        allowed_dirs : List of allowed directory paths

        Returns:
        --------
        Path : Resolved absolute path

        Raises:
        -------
        SecurityError : If path is outside allowed directories or contains suspicious patterns
        """
        # Check for suspicious patterns in the input path string (before resolution)
        # Focus on relative path traversal patterns
        path_str = str(file_path)
        suspicious_patterns = ['..', '~']
        for pattern in suspicious_patterns:
            if pattern in path_str:
                logger.warning(f"Suspicious path pattern detected: {pattern} in {path_str}")
                raise SecurityError(f"Path contains suspicious pattern '{pattern}': {path_str}")

        # Resolve to absolute path to prevent path traversal
        try:
            resolved_path = file_path.resolve()
        except (OSError, RuntimeError) as e:
            raise SecurityError(f"Invalid file path: {file_path}. Error: {e}")

        # Validate that resolved path is within allowed directories
        is_allowed = False
        for allowed_dir in allowed_dirs:
            try:
                resolved_allowed = allowed_dir.resolve()
                # Check if the file is within the allowed directory
                resolved_path.relative_to(resolved_allowed)
                is_allowed = True
                break
            except ValueError:
                # Path is not relative to this allowed directory
                continue
            except (OSError, RuntimeError) as e:
                logger.warning(f"Error resolving allowed directory {allowed_dir}: {e}")
                continue

        if not is_allowed:
            allowed_dirs_str = ', '.join(str(d) for d in allowed_dirs)
            raise SecurityError(
                f"Access denied: Path '{resolved_path}' is outside allowed directories: {allowed_dirs_str}"
            )

        return resolved_path

    def load_data(
        self,
        file_path: Union[str, Path],
        file_format: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load data from CSV or Parquet file.

        Parameters:
        -----------
        file_path : Path to data file
        file_format : File format ('csv' or 'parquet'). Auto-detected if None.

        Returns:
        --------
        pd.DataFrame : Loaded data

        Raises:
        -------
        SecurityError : If path validation fails
        FileNotFoundError : If file does not exist
        ValueError : If file format is unsupported
        """
        file_path = Path(file_path)

        # Validate path is within allowed directories
        allowed_dirs = [self.raw_data_dir, self.output_dir]
        validated_path = self._validate_path(file_path, allowed_dirs)

        if not validated_path.exists():
            raise FileNotFoundError(f"File not found: {validated_path}")

        # Auto-detect format
        if file_format is None:
            file_format = validated_path.suffix.lower().replace('.', '')

        logger.info(f"Loading {file_format.upper()} file: {validated_path.name}")

        try:
            if file_format == 'csv':
                df = pd.read_csv(validated_path)
            elif file_format == 'parquet':
                df = pd.read_parquet(validated_path)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")

            logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
            return df

        except pd.errors.ParserError as e:
            logger.error(f"Error parsing CSV file: {e}")
            raise
        except (OSError, IOError) as e:
            logger.error(f"Error reading file: {e}")
            raise
        except ValueError as e:
            logger.error(f"Invalid file format or data: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading file: {e}")
            raise

    def standardize_columns(self, df: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
        """
        Standardize column names to expected format.

        Parameters:
        -----------
        df : Input DataFrame
        copy : If True, create a copy of the DataFrame. If False, modify in place.

        Returns:
        --------
        pd.DataFrame : DataFrame with standardized column names
        """
        logger.info("Standardizing column names...")

        if copy:
            df = df.copy()

        # Convert all column names to lowercase for mapping
        df.columns = df.columns.str.lower().str.strip()

        # Apply column mappings
        rename_dict = {}
        for col in df.columns:
            if col in self.COLUMN_MAPPINGS:
                rename_dict[col] = self.COLUMN_MAPPINGS[col]

        if rename_dict:
            df = df.rename(columns=rename_dict)
            logger.info(f"Renamed columns: {rename_dict}")

        # Check for required columns
        missing_cols = set(self.STANDARD_COLS) - set(df.columns)
        if missing_cols:
            logger.warning(f"Missing expected columns: {missing_cols}")
            # Check if we have the columns under different names
            available_cols = df.columns.tolist()
            logger.info(f"Available columns: {available_cols}")

        return df

    def validate_ohlcv_relationships(
        self,
        df: pd.DataFrame,
        auto_fix: bool = True,
        dry_run: bool = False,
        copy: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Validate OHLC relationships (high >= low, etc.).

        Parameters:
        -----------
        df : Input DataFrame
        auto_fix : If True, automatically fix violations. If False, only report them.
        dry_run : If True, show what would be fixed without applying changes.
                  Implies auto_fix=False for the actual modifications.
        copy : If True, create a copy of the DataFrame. If False, modify in place.

        Returns:
        --------
        tuple : (validated DataFrame, validation report with detailed fix information)
        """
        logger.info("Validating OHLCV relationships...")
        if dry_run:
            logger.info("  [DRY RUN MODE] - No changes will be applied")
        elif not auto_fix:
            logger.info("  [REPORT ONLY MODE] - Violations will be reported but not fixed")

        if copy:
            df = df.copy()

        validation_report = {
            'total_rows': len(df),
            'violations': {},
            'fixes_applied': {},
            'mode': 'dry_run' if dry_run else ('auto_fix' if auto_fix else 'report_only')
        }

        should_fix = auto_fix and not dry_run

        # Check 1: High >= Low
        high_low_violations = df['high'] < df['low']
        n_violations = high_low_violations.sum()
        if n_violations > 0:
            logger.warning(f"Found {n_violations} rows where high < low")
            validation_report['violations']['high_lt_low'] = int(n_violations)
            if should_fix:
                mask = high_low_violations
                df.loc[mask, ['high', 'low']] = df.loc[mask, ['low', 'high']].values
                validation_report['fixes_applied']['high_lt_low'] = int(n_violations)
                logger.info(f"  Fixed {n_violations} rows by swapping high/low values")
            elif dry_run:
                logger.info(f"  [DRY RUN] Would fix {n_violations} rows by swapping high/low values")

        # Check 2: High >= Open
        high_open_violations = df['high'] < df['open']
        n_violations = high_open_violations.sum()
        if n_violations > 0:
            logger.warning(f"Found {n_violations} rows where high < open")
            validation_report['violations']['high_lt_open'] = int(n_violations)
            if should_fix:
                df.loc[high_open_violations, 'high'] = df.loc[high_open_violations, ['high', 'open']].max(axis=1)
                validation_report['fixes_applied']['high_lt_open'] = int(n_violations)
                logger.info(f"  Fixed {n_violations} rows by setting high = max(high, open)")
            elif dry_run:
                logger.info(f"  [DRY RUN] Would fix {n_violations} rows by setting high = max(high, open)")

        # Check 3: High >= Close
        high_close_violations = df['high'] < df['close']
        n_violations = high_close_violations.sum()
        if n_violations > 0:
            logger.warning(f"Found {n_violations} rows where high < close")
            validation_report['violations']['high_lt_close'] = int(n_violations)
            if should_fix:
                df.loc[high_close_violations, 'high'] = df.loc[high_close_violations, ['high', 'close']].max(axis=1)
                validation_report['fixes_applied']['high_lt_close'] = int(n_violations)
                logger.info(f"  Fixed {n_violations} rows by setting high = max(high, close)")
            elif dry_run:
                logger.info(f"  [DRY RUN] Would fix {n_violations} rows by setting high = max(high, close)")

        # Check 4: Low <= Open
        low_open_violations = df['low'] > df['open']
        n_violations = low_open_violations.sum()
        if n_violations > 0:
            logger.warning(f"Found {n_violations} rows where low > open")
            validation_report['violations']['low_gt_open'] = int(n_violations)
            if should_fix:
                df.loc[low_open_violations, 'low'] = df.loc[low_open_violations, ['low', 'open']].min(axis=1)
                validation_report['fixes_applied']['low_gt_open'] = int(n_violations)
                logger.info(f"  Fixed {n_violations} rows by setting low = min(low, open)")
            elif dry_run:
                logger.info(f"  [DRY RUN] Would fix {n_violations} rows by setting low = min(low, open)")

        # Check 5: Low <= Close
        low_close_violations = df['low'] > df['close']
        n_violations = low_close_violations.sum()
        if n_violations > 0:
            logger.warning(f"Found {n_violations} rows where low > close")
            validation_report['violations']['low_gt_close'] = int(n_violations)
            if should_fix:
                df.loc[low_close_violations, 'low'] = df.loc[low_close_violations, ['low', 'close']].min(axis=1)
                validation_report['fixes_applied']['low_gt_close'] = int(n_violations)
                logger.info(f"  Fixed {n_violations} rows by setting low = min(low, close)")
            elif dry_run:
                logger.info(f"  [DRY RUN] Would fix {n_violations} rows by setting low = min(low, close)")

        # Check 6: Negative prices
        negative_price_mask = (df['open'] <= 0) | (df['high'] <= 0) | (df['low'] <= 0) | (df['close'] <= 0)
        n_violations = negative_price_mask.sum()
        if n_violations > 0:
            logger.warning(f"Found {n_violations} rows with negative or zero prices")
            validation_report['violations']['negative_prices'] = int(n_violations)
            if should_fix:
                df = df[~negative_price_mask]
                validation_report['fixes_applied']['negative_prices'] = int(n_violations)
                logger.info(f"  Fixed by removing {n_violations} rows with negative/zero prices")
            elif dry_run:
                logger.info(f"  [DRY RUN] Would remove {n_violations} rows with negative/zero prices")

        # Check 7: Negative volume
        if 'volume' in df.columns:
            negative_volume = df['volume'] < 0
            n_violations = negative_volume.sum()
            if n_violations > 0:
                logger.warning(f"Found {n_violations} rows with negative volume")
                validation_report['violations']['negative_volume'] = int(n_violations)
                if should_fix:
                    df.loc[negative_volume, 'volume'] = 0
                    validation_report['fixes_applied']['negative_volume'] = int(n_violations)
                    logger.info(f"  Fixed {n_violations} rows by setting negative volume to 0")
                elif dry_run:
                    logger.info(f"  [DRY RUN] Would set {n_violations} negative volume values to 0")

        validation_report['rows_after_validation'] = len(df)
        validation_report['total_fixes_applied'] = sum(validation_report['fixes_applied'].values())

        if dry_run:
            logger.info(f"Validation complete (DRY RUN). No changes applied.")
            logger.info(f"  Would fix {validation_report['total_fixes_applied']} violations")
        else:
            logger.info(f"Validation complete. Rows: {validation_report['total_rows']} -> {validation_report['rows_after_validation']}")
            if auto_fix:
                logger.info(f"  Applied {validation_report['total_fixes_applied']} fixes")

        return df, validation_report

    def handle_timezone(self, df: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
        """
        Convert datetime to UTC timezone.

        Parameters:
        -----------
        df : Input DataFrame
        copy : If True, create a copy of the DataFrame. If False, modify in place.

        Returns:
        --------
        pd.DataFrame : DataFrame with UTC datetime
        """
        logger.info(f"Converting timezone from {self.source_timezone} to UTC...")

        if copy:
            df = df.copy()

        # Ensure datetime column exists and is datetime type
        if 'datetime' not in df.columns:
            raise ValueError("No 'datetime' column found in data")

        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
            df['datetime'] = pd.to_datetime(df['datetime'])

        # Handle timezone conversion
        if df['datetime'].dt.tz is None:
            # Naive datetime - localize to source timezone first
            source_tz = self.TIMEZONE_MAP.get(self.source_timezone, self.source_timezone)
            try:
                tz = pytz.timezone(source_tz)
                df['datetime'] = df['datetime'].dt.tz_localize(tz)
                logger.info(f"Localized to {source_tz}")
            except Exception as e:
                logger.warning(f"Could not localize to {source_tz}: {e}. Assuming UTC.")
                df['datetime'] = df['datetime'].dt.tz_localize('UTC')

        # Convert to UTC
        if df['datetime'].dt.tz.zone != 'UTC':
            df['datetime'] = df['datetime'].dt.tz_convert('UTC')
            logger.info("Converted to UTC")

        # Remove timezone info (store as naive UTC)
        df['datetime'] = df['datetime'].dt.tz_localize(None)

        return df

    def validate_data_types(self, df: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
        """
        Validate and convert data types.

        Parameters:
        -----------
        df : Input DataFrame
        copy : If True, create a copy of the DataFrame. If False, modify in place.

        Returns:
        --------
        pd.DataFrame : DataFrame with validated data types
        """
        logger.info("Validating data types...")

        if copy:
            df = df.copy()

        # Datetime
        if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
            df['datetime'] = pd.to_datetime(df['datetime'])

        # OHLC columns should be float
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Volume should be integer
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype('int64')

        # Check for any NaN values introduced
        nan_counts = df[['open', 'high', 'low', 'close']].isna().sum()
        if nan_counts.any():
            logger.warning(f"NaN values found after type conversion:\n{nan_counts[nan_counts > 0]}")
            # Drop rows with NaN in OHLC
            df = df.dropna(subset=['open', 'high', 'low', 'close'])
            logger.info(f"Dropped rows with NaN. Remaining: {len(df):,}")

        return df

    def ingest_file(
        self,
        file_path: Union[str, Path],
        symbol: Optional[str] = None,
        validate: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete ingestion pipeline for a single file.

        Parameters:
        -----------
        file_path : Path to raw data file
        symbol : Symbol name (if None, extracted from filename or data)
        validate : Whether to validate OHLCV relationships

        Returns:
        --------
        tuple : (processed DataFrame, metadata dict)
        """
        file_path = Path(file_path)

        # Extract symbol from filename if not provided
        if symbol is None:
            # Check if symbol column exists in data
            try:
                sample_df = pd.read_parquet(file_path)
                if self.symbol_col and self.symbol_col in sample_df.columns:
                    # Symbol is in the data, will be extracted later
                    symbol = sample_df[self.symbol_col].iloc[0] if len(sample_df) > 0 else None
            except (OSError, IOError, FileNotFoundError) as e:
                # File might be CSV or unreadable, will extract from filename instead
                logger.debug(f"Could not read parquet file to extract symbol: {e}")
            except (KeyError, IndexError) as e:
                # Symbol column not found in data or empty dataframe
                logger.debug(f"Symbol column not found in data: {e}")

            if symbol is None:
                # Extract from filename (e.g., "MES_1m.parquet" -> "MES")
                symbol = file_path.stem.split('_')[0].upper()

        logger.info(f"\n{'='*60}")
        logger.info(f"Starting ingestion for symbol: {symbol}")
        logger.info(f"{'='*60}\n")

        metadata = {
            'symbol': symbol,
            'source_file': str(file_path),
            'ingestion_timestamp': datetime.now().isoformat(),
            'source_timezone': self.source_timezone
        }

        # Load data
        df = self.load_data(file_path)
        metadata['raw_rows'] = len(df)

        # Create a single copy at the start to avoid 4x memory overhead
        # All subsequent operations use copy=False for in-place modifications
        df = df.copy()

        # Standardize columns (in-place)
        df = self.standardize_columns(df, copy=False)

        # Validate data types (in-place)
        df = self.validate_data_types(df, copy=False)
        metadata['rows_after_type_validation'] = len(df)

        # Handle timezone (in-place)
        df = self.handle_timezone(df, copy=False)

        # Sort by datetime
        df = df.sort_values('datetime').reset_index(drop=True)

        # Validate OHLCV relationships (in-place)
        if validate:
            df, validation_report = self.validate_ohlcv_relationships(df, copy=False)
            metadata['validation'] = validation_report

        # Add symbol column if not present
        if self.symbol_col and self.symbol_col not in df.columns:
            df[self.symbol_col] = symbol

        # Select and order columns
        output_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        if self.symbol_col and self.symbol_col in df.columns:
            output_cols.append(self.symbol_col)
        df = df[output_cols]

        metadata['final_rows'] = len(df)
        metadata['date_range'] = {
            'start': df['datetime'].min().isoformat(),
            'end': df['datetime'].max().isoformat()
        }
        metadata['columns'] = df.columns.tolist()

        logger.info(f"\nIngestion complete for {symbol}")
        logger.info(f"Final shape: {df.shape}")
        logger.info(f"Date range: {metadata['date_range']['start']} to {metadata['date_range']['end']}")

        return df, metadata

    def save_parquet(
        self,
        df: pd.DataFrame,
        symbol: str,
        metadata: Optional[Dict] = None
    ) -> Path:
        """
        Save DataFrame to Parquet format.

        Parameters:
        -----------
        df : DataFrame to save
        symbol : Symbol name
        metadata : Optional metadata to include

        Returns:
        --------
        Path : Path to saved file
        """
        output_path = self.output_dir / f"{symbol}.parquet"

        logger.info(f"Saving to: {output_path}")

        # Save with compression
        df.to_parquet(
            output_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )

        # Save metadata separately if provided
        if metadata:
            import json
            metadata_path = self.output_dir / f"{symbol}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Metadata saved to: {metadata_path}")

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.2f} MB")

        return output_path

    def ingest_directory(
        self,
        pattern: str = "*.parquet",
        validate: bool = True
    ) -> Dict[str, Dict]:
        """
        Ingest all files matching pattern in raw data directory.

        Parameters:
        -----------
        pattern : File pattern to match (default: "*.parquet")
        validate : Whether to validate OHLCV relationships

        Returns:
        --------
        dict : Dictionary mapping symbols to their metadata
        """
        files = list(self.raw_data_dir.glob(pattern))

        if not files:
            logger.warning(f"No files found matching pattern: {pattern}")
            return {}

        logger.info(f"Found {len(files)} files to ingest")

        results = {}

        for file_path in files:
            try:
                df, metadata = self.ingest_file(file_path, validate=validate)
                symbol = metadata['symbol']
                self.save_parquet(df, symbol, metadata)
                results[symbol] = metadata

            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {e}", exc_info=True)
                continue

        logger.info(f"\nIngestion complete. Processed {len(results)} files successfully.")
        return results


def main():
    """
    Example usage of DataIngestor.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Stage 1: Data Ingestion')
    parser.add_argument('--raw-dir', type=str, default='data/raw',
                        help='Raw data directory')
    parser.add_argument('--output-dir', type=str, default='data/raw',
                        help='Output directory')
    parser.add_argument('--pattern', type=str, default='*.parquet',
                        help='File pattern to match')
    parser.add_argument('--timezone', type=str, default='UTC',
                        help='Source timezone')
    parser.add_argument('--no-validate', action='store_true',
                        help='Skip OHLCV validation')

    args = parser.parse_args()

    # Initialize ingestor
    ingestor = DataIngestor(
        raw_data_dir=args.raw_dir,
        output_dir=args.output_dir,
        source_timezone=args.timezone
    )

    # Ingest all files
    results = ingestor.ingest_directory(
        pattern=args.pattern,
        validate=not args.no_validate
    )

    print("\n" + "="*60)
    print("INGESTION SUMMARY")
    print("="*60)
    for symbol, metadata in results.items():
        print(f"\n{symbol}:")
        print(f"  Rows: {metadata['raw_rows']:,} -> {metadata['final_rows']:,}")
        print(f"  Date range: {metadata['date_range']['start']} to {metadata['date_range']['end']}")
        if 'validation' in metadata:
            violations = metadata['validation'].get('violations', {})
            if violations:
                print(f"  Violations fixed: {sum(violations.values())}")


if __name__ == '__main__':
    main()
