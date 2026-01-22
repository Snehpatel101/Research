"""
Data validation functions for the ingestion pipeline.

Handles:
- Path security validation
- OHLCV relationship validation
- Data type validation
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class SecurityError(Exception):
    """Exception raised for security violations such as path traversal attempts."""

    pass


def validate_path(file_path: Path, allowed_dirs: list[Path]) -> Path:
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
    suspicious_patterns = ["..", "~"]
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
        allowed_dirs_str = ", ".join(str(d) for d in allowed_dirs)
        raise SecurityError(
            f"Access denied: Path '{resolved_path}' is outside allowed directories: {allowed_dirs_str}"
        )

    return resolved_path


def validate_ohlcv_relationships(
    df: pd.DataFrame, auto_fix: bool = True, dry_run: bool = False, copy: bool = True
) -> tuple[pd.DataFrame, dict]:
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
        "total_rows": len(df),
        "violations": {},
        "fixes_applied": {},
        "mode": "dry_run" if dry_run else ("auto_fix" if auto_fix else "report_only"),
    }

    should_fix = auto_fix and not dry_run

    # Check 1: High >= Low
    high_low_violations = df["high"] < df["low"]
    n_violations = high_low_violations.sum()
    if n_violations > 0:
        logger.warning(f"Found {n_violations} rows where high < low")
        validation_report["violations"]["high_lt_low"] = int(n_violations)
        if should_fix:
            mask = high_low_violations
            df.loc[mask, ["high", "low"]] = df.loc[mask, ["low", "high"]].values
            validation_report["fixes_applied"]["high_lt_low"] = int(n_violations)
            logger.info(f"  Fixed {n_violations} rows by swapping high/low values")
        elif dry_run:
            logger.info(f"  [DRY RUN] Would fix {n_violations} rows by swapping high/low values")

    # Check 2: High >= Open
    high_open_violations = df["high"] < df["open"]
    n_violations = high_open_violations.sum()
    if n_violations > 0:
        logger.warning(f"Found {n_violations} rows where high < open")
        validation_report["violations"]["high_lt_open"] = int(n_violations)
        if should_fix:
            df.loc[high_open_violations, "high"] = df.loc[
                high_open_violations, ["high", "open"]
            ].max(axis=1)
            validation_report["fixes_applied"]["high_lt_open"] = int(n_violations)
            logger.info(f"  Fixed {n_violations} rows by setting high = max(high, open)")
        elif dry_run:
            logger.info(
                f"  [DRY RUN] Would fix {n_violations} rows by setting high = max(high, open)"
            )

    # Check 3: High >= Close
    high_close_violations = df["high"] < df["close"]
    n_violations = high_close_violations.sum()
    if n_violations > 0:
        logger.warning(f"Found {n_violations} rows where high < close")
        validation_report["violations"]["high_lt_close"] = int(n_violations)
        if should_fix:
            df.loc[high_close_violations, "high"] = df.loc[
                high_close_violations, ["high", "close"]
            ].max(axis=1)
            validation_report["fixes_applied"]["high_lt_close"] = int(n_violations)
            logger.info(f"  Fixed {n_violations} rows by setting high = max(high, close)")
        elif dry_run:
            logger.info(
                f"  [DRY RUN] Would fix {n_violations} rows by setting high = max(high, close)"
            )

    # Check 4: Low <= Open
    low_open_violations = df["low"] > df["open"]
    n_violations = low_open_violations.sum()
    if n_violations > 0:
        logger.warning(f"Found {n_violations} rows where low > open")
        validation_report["violations"]["low_gt_open"] = int(n_violations)
        if should_fix:
            df.loc[low_open_violations, "low"] = df.loc[low_open_violations, ["low", "open"]].min(
                axis=1
            )
            validation_report["fixes_applied"]["low_gt_open"] = int(n_violations)
            logger.info(f"  Fixed {n_violations} rows by setting low = min(low, open)")
        elif dry_run:
            logger.info(
                f"  [DRY RUN] Would fix {n_violations} rows by setting low = min(low, open)"
            )

    # Check 5: Low <= Close
    low_close_violations = df["low"] > df["close"]
    n_violations = low_close_violations.sum()
    if n_violations > 0:
        logger.warning(f"Found {n_violations} rows where low > close")
        validation_report["violations"]["low_gt_close"] = int(n_violations)
        if should_fix:
            df.loc[low_close_violations, "low"] = df.loc[
                low_close_violations, ["low", "close"]
            ].min(axis=1)
            validation_report["fixes_applied"]["low_gt_close"] = int(n_violations)
            logger.info(f"  Fixed {n_violations} rows by setting low = min(low, close)")
        elif dry_run:
            logger.info(
                f"  [DRY RUN] Would fix {n_violations} rows by setting low = min(low, close)"
            )

    # Check 6: Negative prices
    negative_price_mask = (
        (df["open"] <= 0) | (df["high"] <= 0) | (df["low"] <= 0) | (df["close"] <= 0)
    )
    n_violations = negative_price_mask.sum()
    if n_violations > 0:
        logger.warning(f"Found {n_violations} rows with negative or zero prices")
        validation_report["violations"]["negative_prices"] = int(n_violations)
        if should_fix:
            df = df[~negative_price_mask]
            validation_report["fixes_applied"]["negative_prices"] = int(n_violations)
            logger.info(f"  Fixed by removing {n_violations} rows with negative/zero prices")
        elif dry_run:
            logger.info(f"  [DRY RUN] Would remove {n_violations} rows with negative/zero prices")

    # Check 7: Negative volume
    if "volume" in df.columns:
        negative_volume = df["volume"] < 0
        n_violations = negative_volume.sum()
        if n_violations > 0:
            logger.warning(f"Found {n_violations} rows with negative volume")
            validation_report["violations"]["negative_volume"] = int(n_violations)
            if should_fix:
                df.loc[negative_volume, "volume"] = 0
                validation_report["fixes_applied"]["negative_volume"] = int(n_violations)
                logger.info(f"  Fixed {n_violations} rows by setting negative volume to 0")
            elif dry_run:
                logger.info(f"  [DRY RUN] Would set {n_violations} negative volume values to 0")

    validation_report["rows_after_validation"] = len(df)
    validation_report["total_fixes_applied"] = sum(validation_report["fixes_applied"].values())

    if dry_run:
        logger.info("Validation complete (DRY RUN). No changes applied.")
        logger.info(f"  Would fix {validation_report['total_fixes_applied']} violations")
    else:
        logger.info(
            f"Validation complete. Rows: {validation_report['total_rows']} -> {validation_report['rows_after_validation']}"
        )
        if auto_fix:
            logger.info(f"  Applied {validation_report['total_fixes_applied']} fixes")

    return df, validation_report


def validate_data_types(df: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
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
    if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        df["datetime"] = pd.to_datetime(df["datetime"])

    # OHLC columns should be float
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Volume should be integer
    if "volume" in df.columns:
        volume_numeric = pd.to_numeric(df["volume"], errors="coerce")
        nan_count = volume_numeric.isna().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} invalid volume values, setting to 0")
        df["volume"] = volume_numeric.fillna(0).astype("int64")

    # Check for any NaN values introduced
    nan_counts = df[["open", "high", "low", "close"]].isna().sum()
    if nan_counts.any():
        logger.warning(f"NaN values found after type conversion:\n{nan_counts[nan_counts > 0]}")
        # Drop rows with NaN in OHLC
        df = df.dropna(subset=["open", "high", "low", "close"])
        logger.info(f"Dropped rows with NaN. Remaining: {len(df):,}")

    return df
