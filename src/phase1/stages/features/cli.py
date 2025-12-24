"""
CLI for Feature Engineering.

This module provides command-line interface for feature engineering.
"""

import argparse
from pathlib import Path

from .engineer import FeatureEngineer


def main():
    """
    Example usage of FeatureEngineer.
    """
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
