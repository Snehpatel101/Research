"""
File loading functions for the ingestion pipeline.

Handles:
- Loading CSV and Parquet files
- Saving to Parquet format
"""

import json
import logging
from pathlib import Path

import pandas as pd

from .validators import validate_path

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def load_data(
    file_path: str | Path, allowed_dirs: list[Path], file_format: str | None = None
) -> pd.DataFrame:
    """
    Load data from CSV or Parquet file.

    Parameters:
    -----------
    file_path : Path to data file
    allowed_dirs : List of allowed directories for path validation
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
    validated_path = validate_path(file_path, allowed_dirs)

    if not validated_path.exists():
        raise FileNotFoundError(f"File not found: {validated_path}")

    # Auto-detect format
    if file_format is None:
        file_format = validated_path.suffix.lower().replace(".", "")

    logger.info(f"Loading {file_format.upper()} file: {validated_path.name}")

    try:
        if file_format == "csv":
            df = pd.read_csv(validated_path)
        elif file_format == "parquet":
            df = pd.read_parquet(validated_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        # Check for empty DataFrame
        if len(df) == 0:
            raise ValueError(f"Loaded file is empty: {validated_path}")

        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
        return df

    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file: {e}")
        raise
    except OSError as e:
        logger.error(f"Error reading file: {e}")
        raise
    except ValueError as e:
        logger.error(f"Invalid file format or data: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading file: {e}")
        raise


def save_parquet(
    df: pd.DataFrame, symbol: str, output_dir: Path, metadata: dict | None = None
) -> Path:
    """
    Save DataFrame to Parquet format.

    Parameters:
    -----------
    df : DataFrame to save
    symbol : Symbol name
    output_dir : Output directory path
    metadata : Optional metadata to include

    Returns:
    --------
    Path : Path to saved file
    """
    output_path = output_dir / f"{symbol}.parquet"

    logger.info(f"Saving to: {output_path}")

    # Save with compression
    df.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)

    # Save metadata separately if provided
    if metadata:
        metadata_path = output_dir / f"{symbol}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to: {metadata_path}")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"File size: {file_size_mb:.2f} MB")

    return output_path
