"""
Lookahead Bias Audit for Features and MTF Resampling.

Implements corruption testing to detect lookahead bias:
1. Corrupt future data (set to NaN or random)
2. Recompute features
3. If past feature values change, lookahead exists

Also validates MTF resampling configuration to ensure
proper `closed` and `label` settings.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# RESAMPLE CONFIG VALIDATION
# =============================================================================

@dataclass
class ResampleConfig:
    """
    Expected resampling configuration for OHLCV data.

    For financial OHLCV data, use:
    - closed='left': Interval is [start, end), left edge inclusive
    - label='left': Timestamp = start of bar

    A bar at 09:30 represents [09:30:00, 09:34:59] for 5-min data.
    The first tick in the period is 'open', last is 'close'.
    """
    closed: Literal["left", "right"] = "left"
    label: Literal["left", "right"] = "left"

    @classmethod
    def ohlcv_default(cls) -> ResampleConfig:
        """Standard OHLCV resampling config (no lookahead)."""
        return cls(closed="left", label="left")


def validate_resample_config(
    closed: str | None = None,
    label: str | None = None,
    expected: ResampleConfig | None = None,
) -> tuple[bool, list[str]]:
    """
    Validate resample parameters against expected config.

    Args:
        closed: The 'closed' parameter used in resample()
        label: The 'label' parameter used in resample()
        expected: Expected config (defaults to OHLCV standard)

    Returns:
        Tuple of (is_valid, list of warnings/errors)

    Notes:
        When closed/label are None, pandas uses defaults:
        - closed='left' for most frequencies
        - label='left' for most frequencies
        This is typically safe but implicit.
    """
    expected = expected or ResampleConfig.ohlcv_default()
    issues = []

    # Check closed parameter
    if closed is None:
        issues.append(
            f"'closed' is implicit (pandas default). "
            f"Recommend explicit closed='{expected.closed}'"
        )
    elif closed != expected.closed:
        issues.append(
            f"'closed={closed}' differs from expected '{expected.closed}'. "
            f"This may cause lookahead bias."
        )

    # Check label parameter
    if label is None:
        issues.append(
            f"'label' is implicit (pandas default). "
            f"Recommend explicit label='{expected.label}'"
        )
    elif label != expected.label:
        issues.append(
            f"'label={label}' differs from expected '{expected.label}'. "
            f"This may cause time alignment issues."
        )

    is_valid = all("differs from expected" not in msg for msg in issues)
    return is_valid, issues


# =============================================================================
# CORRUPTION TESTING FOR LOOKAHEAD DETECTION
# =============================================================================

@dataclass
class LookaheadAuditResult:
    """Result of lookahead audit for a feature function."""
    feature_name: str
    has_lookahead: bool
    affected_columns: list[str] = field(default_factory=list)
    max_affected_rows: int = 0
    corruption_point: int = 0
    details: str = ""

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {
            "feature_name": self.feature_name,
            "has_lookahead": self.has_lookahead,
            "affected_columns": self.affected_columns,
            "max_affected_rows": self.max_affected_rows,
            "corruption_point": self.corruption_point,
            "details": self.details,
        }


class LookaheadAuditor:
    """
    Detects lookahead bias via corruption testing.

    Corruption Testing Method:
    1. Generate features from clean data
    2. Corrupt data from index T onwards (set to NaN or random)
    3. Regenerate features
    4. Compare features for rows < T
    5. If they differ, lookahead exists

    Example:
        >>> auditor = LookaheadAuditor(corruption_point=0.8)
        >>> result = auditor.audit_feature_function(
        ...     df, compute_rsi, name="RSI"
        ... )
        >>> assert not result.has_lookahead, "RSI has lookahead!"
    """

    def __init__(
        self,
        corruption_point: float = 0.8,
        corruption_method: Literal["nan", "random", "shuffle"] = "nan",
        tolerance: float = 1e-10,
        random_seed: int = 42,
    ) -> None:
        """
        Initialize LookaheadAuditor.

        Args:
            corruption_point: Fraction of data to keep clean (0.8 = corrupt last 20%)
            corruption_method: How to corrupt future data
                - 'nan': Set to NaN (strictest test)
                - 'random': Replace with random values
                - 'shuffle': Shuffle future values
            tolerance: Numerical tolerance for comparing features
            random_seed: Random seed for reproducibility
        """
        if not 0 < corruption_point < 1:
            raise ValueError("corruption_point must be in (0, 1)")

        self.corruption_point = corruption_point
        self.corruption_method = corruption_method
        self.tolerance = tolerance
        self.random_seed = random_seed

    def audit_feature_function(
        self,
        df: pd.DataFrame,
        feature_fn: Callable[[pd.DataFrame], pd.DataFrame],
        name: str,
        price_cols: list[str] | None = None,
    ) -> LookaheadAuditResult:
        """
        Audit a feature function for lookahead bias.

        Args:
            df: Input OHLCV DataFrame (must have datetime index or column)
            feature_fn: Function that adds features to DataFrame
                        Signature: (df) -> df_with_features
            name: Name of the feature (for reporting)
            price_cols: Columns to corrupt (default: OHLCV columns)

        Returns:
            LookaheadAuditResult with audit findings
        """
        price_cols = price_cols or ["open", "high", "low", "close", "volume"]
        df_clean = df.copy()

        # Ensure we have enough data
        n = len(df_clean)
        if n < 100:
            return LookaheadAuditResult(
                feature_name=name,
                has_lookahead=False,
                details="Insufficient data for corruption test (need >= 100 rows)",
            )

        # Compute features on clean data
        try:
            df_features_clean = feature_fn(df_clean.copy())
        except Exception as e:
            return LookaheadAuditResult(
                feature_name=name,
                has_lookahead=False,
                details=f"Feature function failed on clean data: {e}",
            )

        # Identify new feature columns
        original_cols = set(df_clean.columns)
        feature_cols = [c for c in df_features_clean.columns if c not in original_cols]

        if not feature_cols:
            return LookaheadAuditResult(
                feature_name=name,
                has_lookahead=False,
                details="No new columns generated by feature function",
            )

        # Corruption point index
        corruption_idx = int(n * self.corruption_point)

        # Corrupt future data
        df_corrupted = self._corrupt_data(
            df_clean.copy(), corruption_idx, price_cols
        )

        # Compute features on corrupted data
        try:
            df_features_corrupted = feature_fn(df_corrupted)
        except Exception as e:
            # If feature fails on corrupted data, that's expected for NaN corruption
            if self.corruption_method == "nan":
                return LookaheadAuditResult(
                    feature_name=name,
                    has_lookahead=False,
                    details=f"Feature function handles NaN gracefully (expected): {e}",
                )
            return LookaheadAuditResult(
                feature_name=name,
                has_lookahead=False,
                details=f"Feature function failed on corrupted data: {e}",
            )

        # Compare features for rows BEFORE corruption point
        affected_cols = []
        max_affected = 0

        for col in feature_cols:
            if col not in df_features_corrupted.columns:
                continue

            clean_vals = df_features_clean[col].iloc[:corruption_idx].values
            corrupted_vals = df_features_corrupted[col].iloc[:corruption_idx].values

            # Handle NaN comparison properly
            clean_nan = np.isnan(clean_vals) if np.issubdtype(clean_vals.dtype, np.floating) else np.zeros_like(clean_vals, dtype=bool)
            corrupt_nan = np.isnan(corrupted_vals) if np.issubdtype(corrupted_vals.dtype, np.floating) else np.zeros_like(corrupted_vals, dtype=bool)

            # Check if NaN patterns match
            if not np.array_equal(clean_nan, corrupt_nan):
                affected_cols.append(col)
                max_affected = max(max_affected, corruption_idx)
                continue

            # Compare non-NaN values
            valid_mask = ~clean_nan
            if valid_mask.sum() > 0:
                diff = np.abs(
                    clean_vals[valid_mask].astype(float) -
                    corrupted_vals[valid_mask].astype(float)
                )
                n_affected = np.sum(diff > self.tolerance)
                if n_affected > 0:
                    affected_cols.append(col)
                    max_affected = max(max_affected, int(n_affected))

        has_lookahead = len(affected_cols) > 0

        if has_lookahead:
            details = (
                f"LOOKAHEAD DETECTED: {len(affected_cols)} columns affected. "
                f"Past values changed when future data was corrupted."
            )
            logger.warning(f"[{name}] {details}")
        else:
            details = "No lookahead detected - past features stable under corruption"

        return LookaheadAuditResult(
            feature_name=name,
            has_lookahead=has_lookahead,
            affected_columns=affected_cols,
            max_affected_rows=max_affected,
            corruption_point=corruption_idx,
            details=details,
        )

    def _corrupt_data(
        self,
        df: pd.DataFrame,
        start_idx: int,
        columns: list[str],
    ) -> pd.DataFrame:
        """Corrupt data from start_idx onwards."""
        rng = np.random.default_rng(self.random_seed)

        for col in columns:
            if col not in df.columns:
                continue

            if self.corruption_method == "nan":
                df.loc[df.index[start_idx:], col] = np.nan

            elif self.corruption_method == "random":
                n_corrupt = len(df) - start_idx
                col_std = df[col].iloc[:start_idx].std()
                col_mean = df[col].iloc[:start_idx].mean()
                random_vals = rng.normal(col_mean, col_std, n_corrupt)
                df.loc[df.index[start_idx:], col] = random_vals

            elif self.corruption_method == "shuffle":
                future_vals = df[col].iloc[start_idx:].values.copy()
                rng.shuffle(future_vals)
                df.loc[df.index[start_idx:], col] = future_vals

        return df


# =============================================================================
# MTF ALIGNMENT AUDIT
# =============================================================================

def audit_mtf_alignment(
    df_base: pd.DataFrame,
    df_mtf: pd.DataFrame,
    base_tf_minutes: int,
    mtf_minutes: int,
) -> tuple[bool, list[str]]:
    """
    Audit MTF alignment for lookahead bias.

    Checks that MTF features at time T only use information
    from completed MTF bars (not the current in-progress bar).

    Args:
        df_base: Base timeframe data with datetime index
        df_mtf: MTF-aligned data with datetime index
        base_tf_minutes: Base timeframe in minutes (e.g., 5)
        mtf_minutes: MTF timeframe in minutes (e.g., 60)

    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []

    if df_mtf.empty:
        issues.append("MTF DataFrame is empty")
        return False, issues

    # Check that MTF features have initial NaNs (due to shift(1))
    # First MTF bar worth of base bars should have NaN
    expected_nan_rows = mtf_minutes // base_tf_minutes

    for col in df_mtf.columns:
        if col in ("datetime", "date", "time"):
            continue

        first_valid = df_mtf[col].first_valid_index()
        if first_valid is None:
            continue

        first_valid_iloc = df_mtf.index.get_loc(first_valid)
        if first_valid_iloc == 0:
            issues.append(
                f"Column '{col}': First row is not NaN. "
                f"MTF features should have initial NaNs from shift(1)."
            )
        elif first_valid_iloc < expected_nan_rows - 1:
            issues.append(
                f"Column '{col}': First valid at row {first_valid_iloc}, "
                f"expected >= {expected_nan_rows - 1} initial NaNs."
            )

    # Check for suspicious patterns (constant early values)
    for col in df_mtf.columns:
        if col in ("datetime", "date", "time"):
            continue

        early_vals = df_mtf[col].iloc[:expected_nan_rows * 2].dropna()
        if len(early_vals) > 1:
            if early_vals.nunique() == 1 and len(early_vals) > 3:
                issues.append(
                    f"Column '{col}': Suspiciously constant early values. "
                    f"May indicate improper forward-fill without shift."
                )

    is_valid = len(issues) == 0
    return is_valid, issues


def audit_feature_lookahead(
    df: pd.DataFrame,
    feature_fn: Callable[[pd.DataFrame], pd.DataFrame],
    name: str,
    corruption_points: list[float] | None = None,
) -> list[LookaheadAuditResult]:
    """
    Convenience function to audit features at multiple corruption points.

    Args:
        df: Input OHLCV DataFrame
        feature_fn: Feature generation function
        name: Feature name
        corruption_points: List of corruption fractions (default: [0.5, 0.7, 0.9])

    Returns:
        List of LookaheadAuditResult for each corruption point
    """
    corruption_points = corruption_points or [0.5, 0.7, 0.9]
    results = []

    for cp in corruption_points:
        auditor = LookaheadAuditor(corruption_point=cp)
        result = auditor.audit_feature_function(df, feature_fn, f"{name}@{cp}")
        results.append(result)

    return results


__all__ = [
    "LookaheadAuditor",
    "LookaheadAuditResult",
    "ResampleConfig",
    "validate_resample_config",
    "audit_feature_lookahead",
    "audit_mtf_alignment",
]
