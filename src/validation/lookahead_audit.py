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
import pandas as pd  # type: ignore[import-untyped]

from src.core.exceptions import LookaheadError

logger = logging.getLogger(__name__)


class LookaheadBiasError(LookaheadError):
    """Raised when lookahead bias is detected and blocking mode is enabled.

    This exception is raised by lookahead audit functions when:
    1. Lookahead bias is detected (affected columns found)
    2. The `raise_on_lookahead=True` parameter is set

    The exception contains the full LookaheadAuditResult for inspection.

    Example:
        >>> auditor = LookaheadAuditor(corruption_point=0.8)
        >>> try:
        ...     result = auditor.audit_feature_function(
        ...         df, compute_rsi, name="RSI", raise_on_lookahead=True
        ...     )
        ... except LookaheadBiasError as e:
        ...     print(f"Lookahead detected: {e.result.affected_columns}")
    """

    def __init__(self, result: LookaheadAuditResult) -> None:
        self.result = result
        super().__init__(
            f"Lookahead bias detected in '{result.feature_name}': "
            f"{len(result.affected_columns)} columns affected. {result.details}"
        )


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
            f"'label' is implicit (pandas default). " f"Recommend explicit label='{expected.label}'"
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
        raise_on_lookahead: bool = True,
    ) -> LookaheadAuditResult:
        """
        Audit a feature function for lookahead bias.

        Args:
            df: Input OHLCV DataFrame (must have datetime index or column)
            feature_fn: Function that adds features to DataFrame
                        Signature: (df) -> df_with_features
            name: Name of the feature (for reporting)
            price_cols: Columns to corrupt (default: OHLCV columns)
            raise_on_lookahead: If True, raise LookaheadBiasError when lookahead
                is detected. Default is True (blocking mode for production safety).

        Returns:
            LookaheadAuditResult with audit findings

        Raises:
            LookaheadBiasError: If raise_on_lookahead=True and lookahead detected.
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
        df_corrupted = self._corrupt_data(df_clean.copy(), corruption_idx, price_cols)

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
            clean_nan = (
                np.isnan(clean_vals)
                if np.issubdtype(clean_vals.dtype, np.floating)
                else np.zeros_like(clean_vals, dtype=bool)
            )
            corrupt_nan = (
                np.isnan(corrupted_vals)
                if np.issubdtype(corrupted_vals.dtype, np.floating)
                else np.zeros_like(corrupted_vals, dtype=bool)
            )

            # Check if NaN patterns match
            if not np.array_equal(clean_nan, corrupt_nan):
                affected_cols.append(col)
                max_affected = max(max_affected, corruption_idx)
                continue

            # Compare non-NaN values
            valid_mask = ~clean_nan
            if valid_mask.sum() > 0:
                diff = np.abs(
                    clean_vals[valid_mask].astype(float) - corrupted_vals[valid_mask].astype(float)
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

        result = LookaheadAuditResult(
            feature_name=name,
            has_lookahead=has_lookahead,
            affected_columns=affected_cols,
            max_affected_rows=max_affected,
            corruption_point=corruption_idx,
            details=details,
        )

        if raise_on_lookahead and result.has_lookahead:
            raise LookaheadBiasError(result)

        return result

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

    def scan_dependency_propagation(
        self,
        feature_dependencies: dict[str, list[str]],
        tainted_features: list[str] | set[str],
    ) -> PropagationScanResult:
        """Scan the feature dependency DAG for lookahead propagation.

        Convenience method that delegates to the module-level
        :func:`scan_dependency_propagation` function.  See that function
        for full documentation.

        Args:
            feature_dependencies: The ``FEATURE_DEPENDENCIES`` dict.
            tainted_features: Feature families with known direct lookahead.

        Returns:
            PropagationScanResult with direct, propagated and clean features.
        """
        return scan_dependency_propagation(feature_dependencies, tainted_features)


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

        early_vals = df_mtf[col].iloc[: expected_nan_rows * 2].dropna()
        if len(early_vals) > 1 and early_vals.nunique() == 1 and len(early_vals) > 3:
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
    raise_on_lookahead: bool = True,
) -> list[LookaheadAuditResult]:
    """
    Convenience function to audit features at multiple corruption points.

    Args:
        df: Input OHLCV DataFrame
        feature_fn: Feature generation function
        name: Feature name
        corruption_points: List of corruption fractions (default: [0.5, 0.7, 0.9])
        raise_on_lookahead: If True, raise LookaheadBiasError when lookahead
            is detected at any corruption point.
            Default is True (blocking mode for production safety).

    Returns:
        List of LookaheadAuditResult for each corruption point

    Raises:
        LookaheadBiasError: If raise_on_lookahead=True and lookahead detected
            at any corruption point. Raises on first detection.
    """
    corruption_points = corruption_points or [0.5, 0.7, 0.9]
    results = []

    for cp in corruption_points:
        auditor = LookaheadAuditor(corruption_point=cp)
        result = auditor.audit_feature_function(
            df, feature_fn, f"{name}@{cp}", raise_on_lookahead=raise_on_lookahead
        )
        results.append(result)

    return results


# =============================================================================
# H3: CROSS-FEATURE LOOKAHEAD PROPAGATION SCAN
# =============================================================================


@dataclass
class PropagationScanResult:
    """Result of a dependency-propagation lookahead scan.

    Attributes:
        tainted_features: Feature families with *direct* lookahead issues.
        propagated_features: Feature families tainted through upstream
            dependencies (not directly, but via the DAG).
        propagation_paths: Mapping from each propagated feature to the
            chain of dependencies that connects it to a tainted source.
        clean_features: Feature families with no direct or propagated
            lookahead issues.
    """

    tainted_features: list[str] = field(default_factory=list)
    propagated_features: list[str] = field(default_factory=list)
    propagation_paths: dict[str, list[str]] = field(default_factory=dict)
    clean_features: list[str] = field(default_factory=list)

    @property
    def all_affected(self) -> list[str]:
        """All features affected by lookahead (direct + propagated)."""
        return sorted(set(self.tainted_features) | set(self.propagated_features))

    @property
    def has_propagation(self) -> bool:
        """True if any feature is tainted through propagation."""
        return len(self.propagated_features) > 0

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {
            "tainted_features": self.tainted_features,
            "propagated_features": self.propagated_features,
            "propagation_paths": self.propagation_paths,
            "clean_features": self.clean_features,
            "all_affected": self.all_affected,
            "has_propagation": self.has_propagation,
        }


def _topological_sort(dependencies: dict[str, list[str]]) -> list[str]:
    """Compute a topological ordering of feature families from a dependency dict.

    Uses Kahn's algorithm.  The special wildcard ``"*"`` dependency (used by
    ``mtf`` to mean "all other families") is expanded before sorting.

    Args:
        dependencies: Mapping of ``feature_family -> [upstream_families]``.

    Returns:
        List of feature families in topological (computation) order.

    Raises:
        ValueError: If there is a cycle in the dependency graph.
    """
    all_nodes = set(dependencies.keys())

    # Expand wildcard dependencies: "*" means "depends on all other families"
    resolved: dict[str, list[str]] = {}
    for node, deps in dependencies.items():
        if "*" in deps:
            resolved[node] = [d for d in all_nodes if d != node]
        else:
            # Only include deps that are actually defined nodes
            resolved[node] = [d for d in deps if d in all_nodes]

    # Build in-degree map
    in_degree: dict[str, int] = dict.fromkeys(all_nodes, 0)
    for node, deps in resolved.items():
        in_degree[node] = len(deps)

    # Seed queue with nodes that have no dependencies
    queue: list[str] = sorted([node for node, deg in in_degree.items() if deg == 0])
    order: list[str] = []

    while queue:
        current = queue.pop(0)
        order.append(current)
        # For every node that depends on `current`, decrement in-degree
        for node, deps in resolved.items():
            if current in deps:
                in_degree[node] -= 1
                if in_degree[node] == 0:
                    queue.append(node)
                    queue.sort()  # deterministic ordering

    if len(order) != len(all_nodes):
        missing = all_nodes - set(order)
        raise ValueError(
            f"Cycle detected in feature dependency graph. " f"Nodes involved: {sorted(missing)}"
        )

    return order


def scan_dependency_propagation(
    feature_dependencies: dict[str, list[str]],
    tainted_features: list[str] | set[str],
) -> PropagationScanResult:
    """Scan the feature dependency DAG for lookahead propagation.

    Given a set of feature families known to have *direct* lookahead issues
    (``tainted_features``), this function traces the dependency graph to
    find all downstream features that are *indirectly* tainted because they
    depend on a tainted upstream.

    The scan uses a topological sort so that taint is propagated level by
    level -- if A is tainted and B depends on A and C depends on B, then
    both B and C are flagged and the full propagation path is recorded.

    Args:
        feature_dependencies: The ``FEATURE_DEPENDENCIES`` dict from
            ``engineer.py``.  Keys are feature family names; values are
            lists of upstream family names.
        tainted_features: Feature families known to have direct lookahead
            issues (e.g. from corruption testing with ``LookaheadAuditor``).

    Returns:
        PropagationScanResult describing direct, propagated and clean features.

    Example:
        >>> from src.data.pipeline.stages.features.engineer import (
        ...     FEATURE_DEPENDENCIES,
        ... )
        >>> result = scan_dependency_propagation(
        ...     FEATURE_DEPENDENCIES,
        ...     tainted_features=["returns"],
        ... )
        >>> # "rsi" depends on "returns", so it should be propagated
        >>> assert "rsi" in result.propagated_features
    """
    tainted_set = set(tainted_features)
    all_nodes = set(feature_dependencies.keys())

    # Expand wildcard deps
    resolved: dict[str, list[str]] = {}
    for node, deps in feature_dependencies.items():
        if "*" in deps:
            resolved[node] = [d for d in all_nodes if d != node]
        else:
            resolved[node] = [d for d in deps if d in all_nodes]

    # Get topological order
    topo_order = _topological_sort(feature_dependencies)

    # Track which features are tainted (direct or propagated) and paths
    effective_taint: set[str] = set(tainted_set)
    propagated: set[str] = set()
    propagation_paths: dict[str, list[str]] = {}

    for feature in topo_order:
        if feature in tainted_set:
            # Already directly tainted -- skip propagation check
            continue

        # Check if any upstream is tainted (directly or via propagation)
        tainted_upstreams = [dep for dep in resolved.get(feature, []) if dep in effective_taint]

        if tainted_upstreams:
            effective_taint.add(feature)
            propagated.add(feature)

            # Build the propagation path: pick the first tainted upstream
            # and prepend its path (if any)
            source = tainted_upstreams[0]
            if source in propagation_paths:
                propagation_paths[feature] = propagation_paths[source] + [source, feature]
            else:
                propagation_paths[feature] = [source, feature]

            logger.warning(
                f"[PropagationScan] Feature '{feature}' tainted via upstream "
                f"{tainted_upstreams}: path={propagation_paths[feature]}"
            )

    clean = sorted(all_nodes - effective_taint)

    return PropagationScanResult(
        tainted_features=sorted(tainted_set & all_nodes),
        propagated_features=sorted(propagated),
        propagation_paths=propagation_paths,
        clean_features=clean,
    )


# =============================================================================
# M6: RESAMPLING PARITY VERIFICATION
# =============================================================================


@dataclass
class ResamplingParityResult:
    """Result of resampling parity verification.

    Attributes:
        is_match: True if all resampling parameters match.
        mismatches: List of human-readable mismatch descriptions.
        training_params: The training resampling parameters that were checked.
        inference_params: The inference resampling parameters that were checked.
    """

    is_match: bool
    mismatches: list[str] = field(default_factory=list)
    training_params: dict[str, str] = field(default_factory=dict)
    inference_params: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {
            "is_match": self.is_match,
            "mismatches": self.mismatches,
            "training_params": self.training_params,
            "inference_params": self.inference_params,
        }


def verify_resampling_parity(
    training_config: dict[str, str],
    inference_config: dict[str, str],
    *,
    raise_on_mismatch: bool = True,
) -> ResamplingParityResult:
    """Verify that resampling parameters match between training and inference.

    Ensures that the exact same resampling convention (``closed``, ``label``,
    ``origin``, ``offset``, etc.) is used in both the training data pipeline
    and the inference/bundle pipeline.  A mismatch can cause subtle
    distribution shift that silently degrades model performance.

    The function compares all keys present in *either* config dict.  If a key
    is present in one config but absent in the other, that counts as a
    mismatch.

    Args:
        training_config: Resampling parameters used during training.
            Expected keys: ``"closed"``, ``"label"``; optional:
            ``"origin"``, ``"offset"``, ``"freq"``.
        inference_config: Resampling parameters used during inference.
            Same expected keys as ``training_config``.
        raise_on_mismatch: If True (default), raise ``ValueError`` when a
            mismatch is detected.

    Returns:
        ResamplingParityResult with comparison details.

    Raises:
        ValueError: If ``raise_on_mismatch=True`` and parameters differ.

    Example:
        >>> train_cfg = {"closed": "left", "label": "left"}
        >>> infer_cfg = {"closed": "left", "label": "left"}
        >>> result = verify_resampling_parity(train_cfg, infer_cfg)
        >>> assert result.is_match
        >>> # Mismatch example:
        >>> bad_cfg = {"closed": "right", "label": "left"}
        >>> verify_resampling_parity(train_cfg, bad_cfg)  # raises ValueError
    """
    all_keys = sorted(set(training_config.keys()) | set(inference_config.keys()))

    # Ensure at least the critical keys are checked
    critical_keys = ["closed", "label"]
    for key in critical_keys:
        if key not in all_keys:
            all_keys.append(key)
    all_keys = sorted(set(all_keys))

    mismatches: list[str] = []

    for key in all_keys:
        train_val = training_config.get(key)
        infer_val = inference_config.get(key)

        if train_val is None and infer_val is None:
            # Neither config specifies this key -- nothing to compare
            continue

        if train_val is None:
            mismatches.append(
                f"Parameter '{key}' is set in inference ('{infer_val}') "
                f"but missing from training config."
            )
        elif infer_val is None:
            mismatches.append(
                f"Parameter '{key}' is set in training ('{train_val}') "
                f"but missing from inference config."
            )
        elif str(train_val) != str(infer_val):
            mismatches.append(
                f"Parameter '{key}' mismatch: "
                f"training='{train_val}' vs inference='{infer_val}'."
            )

    is_match = len(mismatches) == 0

    result = ResamplingParityResult(
        is_match=is_match,
        mismatches=mismatches,
        training_params=dict(training_config),
        inference_params=dict(inference_config),
    )

    if not is_match:
        mismatch_detail = "; ".join(mismatches)
        logger.error(f"[ResamplingParity] Mismatch detected: {mismatch_detail}")
        if raise_on_mismatch:
            raise ValueError(
                f"Resampling parity violation: {mismatch_detail}. "
                f"Training and inference must use identical resampling parameters "
                f"to avoid distribution shift."
            )
    else:
        logger.info("[ResamplingParity] Training and inference resampling parameters match.")

    return result


__all__ = [
    "LookaheadAuditor",
    "LookaheadAuditResult",
    "LookaheadBiasError",
    "ResampleConfig",
    "validate_resample_config",
    "audit_feature_lookahead",
    "audit_mtf_alignment",
    "PropagationScanResult",
    "scan_dependency_propagation",
    "ResamplingParityResult",
    "verify_resampling_parity",
]
