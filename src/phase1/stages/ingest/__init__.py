"""
Stage 1: Data Ingestion Module
Production-ready data ingestion with standardization and validation.

This module handles:
- Loading raw OHLCV data from CSV or Parquet
- Column name standardization
- Timezone conversion to UTC
- Data type validation
- Output to standardized Parquet format

Public API:
- DataIngestor: Main ingestion class
- SecurityError: Exception for security violations
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import pandas as pd

from .loaders import load_data, save_parquet
from .transformers import (
    COLUMN_MAPPINGS,
    STANDARD_COLS,
    TIMEZONE_MAP,
    handle_timezone,
    standardize_columns,
)
from .validators import (
    SecurityError,
    validate_data_types,
    validate_ohlcv_relationships,
    validate_path,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Re-export for backward compatibility
__all__ = [
    'DataIngestor',
    'SecurityError',
    'STANDARD_COLS',
    'COLUMN_MAPPINGS',
    'TIMEZONE_MAP',
]


class DataIngestor:
    """
    Handles ingestion of raw market data with standardization and validation.

    This class orchestrates the complete ingestion pipeline:
    1. Load raw data from CSV/Parquet
    2. Standardize column names
    3. Validate and convert data types
    4. Handle timezone conversion to UTC
    5. Validate OHLCV relationships
    6. Save to standardized Parquet format

    Example:
    --------
    >>> ingestor = DataIngestor(
    ...     raw_data_dir='data/raw',
    ...     output_dir='data/processed',
    ...     source_timezone='America/New_York'
    ... )
    >>> df, metadata = ingestor.ingest_file('data/raw/MES_1m.parquet')
    >>> ingestor.save_parquet(df, 'MES', metadata)
    """

    # Expose constants at class level for backward compatibility
    STANDARD_COLS = STANDARD_COLS
    COLUMN_MAPPINGS = COLUMN_MAPPINGS
    TIMEZONE_MAP = TIMEZONE_MAP

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

        # Validate raw data directory exists
        if not self.raw_data_dir.exists():
            raise FileNotFoundError(
                f"Raw data directory does not exist: {self.raw_data_dir}. "
                f"Please place raw OHLCV files in this directory."
            )

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initialized DataIngestor")
        logger.info(f"Raw data dir: {self.raw_data_dir}")
        logger.info(f"Output dir: {self.output_dir}")

    def _validate_path(self, file_path: Path, allowed_dirs=None):
        """Validate path is within allowed directories."""
        if allowed_dirs is None:
            allowed_dirs = [self.raw_data_dir, self.output_dir]
        return validate_path(file_path, allowed_dirs)

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
        """
        allowed_dirs = [self.raw_data_dir, self.output_dir]
        return load_data(file_path, allowed_dirs, file_format)

    def standardize_columns(self, df: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
        """Standardize column names to expected format."""
        return standardize_columns(df, self.COLUMN_MAPPINGS, copy)

    def validate_ohlcv_relationships(
        self,
        df: pd.DataFrame,
        auto_fix: bool = True,
        dry_run: bool = False,
        copy: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """Validate OHLC relationships (high >= low, etc.)."""
        return validate_ohlcv_relationships(df, auto_fix, dry_run, copy)

    def handle_timezone(self, df: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
        """Convert datetime to UTC timezone."""
        return handle_timezone(df, self.source_timezone, self.TIMEZONE_MAP, copy)

    def validate_data_types(self, df: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
        """Validate and convert data types."""
        return validate_data_types(df, copy)

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
            symbol = self._extract_symbol(file_path)

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

    def _extract_symbol(self, file_path: Path) -> str:
        """Extract symbol from file path or data."""
        symbol = None

        # Check if symbol column exists in data
        try:
            sample_df = pd.read_parquet(file_path)
            if self.symbol_col and self.symbol_col in sample_df.columns:
                symbol = sample_df[self.symbol_col].iloc[0] if len(sample_df) > 0 else None
        except (OSError, IOError, FileNotFoundError) as e:
            logger.debug(f"Could not read parquet file to extract symbol: {e}")
        except (KeyError, IndexError) as e:
            logger.debug(f"Symbol column not found in data: {e}")

        if symbol is None:
            # Extract from filename (e.g., "MES_1m.parquet" -> "MES")
            symbol = file_path.stem.split('_')[0].upper()

        return symbol

    def save_parquet(
        self,
        df: pd.DataFrame,
        symbol: str,
        metadata: Optional[Dict] = None
    ) -> Path:
        """Save DataFrame to Parquet format."""
        return save_parquet(df, symbol, self.output_dir, metadata)

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
        errors = []

        for file_path in files:
            try:
                df, metadata = self.ingest_file(file_path, validate=validate)
                symbol = metadata['symbol']
                self.save_parquet(df, symbol, metadata)
                results[symbol] = metadata

            except Exception as e:
                errors.append({
                    'file': str(file_path.name),
                    'error': str(e),
                    'type': type(e).__name__
                })
                logger.error(f"Error processing {file_path.name}: {e}", exc_info=True)

        if errors:
            error_summary = f"{len(errors)}/{len(files)} files failed ingestion"
            logger.error(f"Ingestion completed with errors: {error_summary}")
            raise RuntimeError(f"{error_summary}. Errors: {errors[:5]}")

        logger.info(f"\nIngestion complete. Processed {len(results)} files successfully.")
        return results
