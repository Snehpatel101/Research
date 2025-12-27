"""
Data integrity validation checks.
"""
import logging
import re
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def check_duplicate_timestamps(
    df: pd.DataFrame, issues_found: List[str]
) -> Dict:
    """
    Check for duplicate timestamps per symbol.

    Args:
        df: DataFrame to check
        issues_found: List to append issues to (mutated)

    Returns:
        Dictionary with duplicate timestamp counts
    """
    logger.info("\n1. Checking for duplicate timestamps...")

    if 'symbol' in df.columns:
        dup_counts = {}
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol]
            dups = symbol_df['datetime'].duplicated().sum()
            dup_counts[symbol] = dups
            if dups > 0:
                issues_found.append(f"{symbol}: {dups} duplicate timestamps")
                logger.warning(f"  {symbol}: {dups:,} duplicate timestamps")
            else:
                logger.info(f"  {symbol}: No duplicates")
        return dup_counts
    else:
        dups = df['datetime'].duplicated().sum()
        if dups > 0:
            issues_found.append(f"Found {dups} duplicate timestamps")
            logger.warning(f"  Found {dups:,} duplicate timestamps")
        else:
            logger.info("  No duplicate timestamps")
        return {'total': dups}


def check_nan_values(df: pd.DataFrame, issues_found: List[str]) -> Dict:
    """
    Check for NaN values in the DataFrame.

    Each symbol is processed in isolation - there are no cross-asset features.
    Forward return columns are expected to have NaN values in the last horizon bars.

    Args:
        df: DataFrame to check
        issues_found: List to append issues to (mutated)

    Returns:
        Dictionary with NaN counts per column
    """
    logger.info("\n2. Checking for NaN values...")

    nan_counts = df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]

    if len(nan_cols) > 0:
        def _forward_return_horizon(name: str) -> int | None:
            match = re.match(r"^fwd_return(?:_log)?_h(\d+)$", name)
            if not match:
                return None
            return int(match.group(1))

        # Determine number of symbols (for forward return NaN calculations)
        n_symbols = df['symbol'].nunique() if 'symbol' in df.columns else 1

        # Separate expected forward return NaNs from unexpected NaNs
        expected_forward_return_nans = {}
        regular_nans = {}

        for col, count in nan_cols.items():
            horizon = _forward_return_horizon(col)
            # Forward returns should have horizon * n_symbols NaNs at the end
            if horizon is not None and count <= horizon * n_symbols:
                expected_forward_return_nans[col] = count
            else:
                regular_nans[col] = count

        if expected_forward_return_nans:
            logger.info(
                "  Found NaN values in forward return columns "
                "(expected in last horizon bars):"
            )
            for col, count in expected_forward_return_nans.items():
                pct = count / len(df) * 100
                logger.info(f"    {col}: {count:,} ({pct:.2f}%)")

        # Log regular NaNs as critical issues
        if regular_nans:
            logger.warning(f"  Found NaN values in {len(regular_nans)} features:")
            for col, count in regular_nans.items():
                pct = count / len(df) * 100
                logger.warning(f"    {col}: {count:,} ({pct:.2f}%)")
                issues_found.append(f"{col}: {count} NaN values ({pct:.2f}%)")

        if not regular_nans and expected_forward_return_nans:
            logger.info(
                "  No unexpected NaN values found "
                "(forward-return NaNs are expected in last horizon bars)"
            )

        # Return all NaN counts for reporting, but only regular ones are flagged as issues
        return {col: int(count) for col, count in nan_cols.items()}
    else:
        logger.info("  No NaN values found")
        return {}


def check_infinite_values(df: pd.DataFrame, issues_found: List[str]) -> Dict:
    """
    Check for infinite values in numeric columns.

    Args:
        df: DataFrame to check
        issues_found: List to append issues to (mutated)

    Returns:
        Dictionary with infinite value counts per column
    """
    logger.info("\n3. Checking for infinite values...")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = {}

    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_counts[col] = int(inf_count)
            issues_found.append(f"{col}: {inf_count} infinite values")
            logger.warning(f"  {col}: {inf_count:,} infinite values")

    if not inf_counts:
        logger.info("  No infinite values found")

    return inf_counts


def analyze_time_gaps(df: pd.DataFrame) -> List[Dict]:
    """
    Analyze time gaps in the data.

    Args:
        df: DataFrame to check

    Returns:
        List of gap information dictionaries
    """
    logger.info("\n4. Analyzing time gaps...")

    gaps = []

    if 'symbol' in df.columns:
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].sort_values('datetime')
            time_diffs = symbol_df['datetime'].diff()
            median_gap = time_diffs.median()
            large_gaps = time_diffs[time_diffs > median_gap * 3]

            if len(large_gaps) > 0:
                gap_info = {
                    'symbol': symbol,
                    'count': int(len(large_gaps)),
                    'median_gap': str(median_gap),
                    'max_gap': str(time_diffs.max())
                }
                gaps.append(gap_info)
                logger.info(f"  {symbol}: {len(large_gaps)} large gaps (>{median_gap*3})")
    else:
        time_diffs = df.sort_values('datetime')['datetime'].diff()
        median_gap = time_diffs.median()
        large_gaps = time_diffs[time_diffs > median_gap * 3]

        if len(large_gaps) > 0:
            gap_info = {
                'count': int(len(large_gaps)),
                'median_gap': str(median_gap),
                'max_gap': str(time_diffs.max())
            }
            gaps.append(gap_info)
            logger.info(f"  Found {len(large_gaps)} large gaps")

    return gaps


def verify_date_range(df: pd.DataFrame) -> Dict:
    """
    Verify and report the date range of the data.

    Args:
        df: DataFrame to check

    Returns:
        Dictionary with date range information
    """
    logger.info("\n5. Date range verification...")

    date_range = {
        'start': str(df['datetime'].min()),
        'end': str(df['datetime'].max()),
        'duration_days': float((df['datetime'].max() - df['datetime'].min()).days),
        'total_bars': int(len(df))
    }

    logger.info(f"  Start: {date_range['start']}")
    logger.info(f"  End:   {date_range['end']}")
    logger.info(f"  Duration: {date_range['duration_days']:.1f} days")
    logger.info(f"  Total bars: {date_range['total_bars']:,}")

    return date_range


def check_data_integrity(df: pd.DataFrame, issues_found: List[str]) -> Dict:
    """
    Run all data integrity checks.

    Args:
        df: DataFrame to validate
        issues_found: List to append issues to (mutated)

    Returns:
        Dictionary with all integrity check results
    """
    logger.info("\n" + "=" * 60)
    logger.info("DATA INTEGRITY CHECKS")
    logger.info("=" * 60)

    results = {
        'duplicate_timestamps': check_duplicate_timestamps(df, issues_found),
        'nan_values': check_nan_values(df, issues_found),
        'infinite_values': check_infinite_values(df, issues_found),
        'gaps': analyze_time_gaps(df),
        'date_range': verify_date_range(df)
    }

    return results
