"""
Stage 1: Data Ingestion Module
Production-ready data ingestion with standardization and validation.

This module is a thin wrapper around src/stages/ingest/ for backward compatibility.
All implementation is in the ingest submodule.

This module handles:
- Loading raw OHLCV data from CSV or Parquet
- Column name standardization
- Timezone conversion to UTC
- Data type validation
- Output to standardized Parquet format
"""

# Re-export public API from submodule
from .ingest import (
    COLUMN_MAPPINGS,
    STANDARD_COLS,
    TIMEZONE_MAP,
    DataIngestor,
    SecurityError,
)

__all__ = [
    'DataIngestor',
    'SecurityError',
    'STANDARD_COLS',
    'COLUMN_MAPPINGS',
    'TIMEZONE_MAP',
]


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
