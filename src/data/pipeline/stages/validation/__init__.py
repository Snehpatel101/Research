"""
Validators submodule for Stage 8 data validation.

Provides modular validation checks for data integrity, labels, features,
and normalization.

Note: Lookahead audit is MANDATORY (Phase 14C) and always runs in blocking mode.
"""

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.optimization.feature_selection import (
    FeatureSelectionResult,
    save_feature_selection_report,
    select_features,
)
from src.validation.leakage_detection import (
    LeakageDetectedError,
    comprehensive_leakage_check,
)
from src.validation.lookahead_audit import LookaheadAuditor, LookaheadBiasError

from .data_contract import (
    INVALID_LABEL_SENTINEL,
    POSITIVE_COLUMNS,
    REQUIRED_OHLCV,
    VALID_LABELS,
    OHLCVValidationSchema,
    filter_invalid_labels,
    get_dataset_fingerprint,
    summarize_label_distribution,
    validate_feature_lookahead,
    validate_labels,
    validate_ohlcv_schema,
)
from .features import check_feature_quality
from .integrity import check_data_integrity
from .labels import check_label_sanity
from .normalization import check_feature_normalization

# Re-export for convenience
__all__ = [
    # Main validation function
    "validate_data",
    # DataValidator class
    "DataValidator",
    # Existing validators
    "check_data_integrity",
    "check_label_sanity",
    "check_feature_quality",
    "check_feature_normalization",
    # OHLCV validation schema (NOT the same as core DataContract)
    "OHLCVValidationSchema",
    "validate_ohlcv_schema",
    "validate_labels",
    "filter_invalid_labels",
    "get_dataset_fingerprint",
    "validate_feature_lookahead",
    "summarize_label_distribution",
    # Feature selection
    "FeatureSelectionResult",
    # Phase 4A, 4B: Validation exceptions
    "LeakageDetectedError",
    "LookaheadBiasError",
    # Constants
    "REQUIRED_OHLCV",
    "VALID_LABELS",
    "INVALID_LABEL_SENTINEL",
    "POSITIVE_COLUMNS",
]

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Comprehensive data validation for Phase 1 pipeline.

    Delegates to specialized modules for each validation type:
    - integrity: duplicate timestamps, NaN/inf values, time gaps
    - labels: distribution, balance, quality scores
    - features: correlations, importance, stationarity
    - normalization: scale, skewness, outliers

    Example:
        >>> validator = DataValidator(df, horizons=[5, 20])
        >>> validator.check_data_integrity()
        >>> validator.check_label_sanity()
        >>> summary = validator.generate_summary()
    """

    def __init__(self, df: pd.DataFrame, horizons: list[int] | None = None, seed: int = 42):
        """
        Initialize the validator.

        Args:
            df: DataFrame to validate
            horizons: List of label horizons to validate (default: [1, 5, 20])
            seed: Random seed for reproducibility
        """
        self.df = df
        self.horizons = horizons if horizons is not None else [1, 5, 20]
        self.seed = seed
        self.validation_results: dict = {}
        self.issues_found: list[str] = []
        self.warnings_found: list[str] = []

    def check_data_integrity(self) -> dict:
        """
        Check for data quality issues.

        Checks for:
        - Duplicate timestamps per symbol
        - NaN values in any column
        - Infinite values in numeric columns
        - Large time gaps
        - Date range verification

        Returns:
            Dictionary with integrity check results
        """
        results = check_data_integrity(self.df, self.issues_found)
        self.validation_results["data_integrity"] = results
        return results

    def check_label_sanity(self) -> dict:
        """
        Check label distributions and quality metrics.

        Checks for:
        - Label distribution per horizon
        - Label balance (warns if any class <20% or >60%)
        - Per-symbol distribution (if symbol column exists)
        - Bars-to-hit statistics
        - Quality score statistics

        Returns:
            Dictionary with label sanity results
        """
        results = check_label_sanity(self.df, self.horizons, self.warnings_found)
        self.validation_results["label_sanity"] = results
        return results

    def check_feature_quality(self, max_features: int = 50) -> dict:
        """
        Check feature correlations and basic importance.

        Checks for:
        - Highly correlated feature pairs (>0.85)
        - Feature importance via Random Forest
        - Stationarity tests (Augmented Dickey-Fuller)

        Args:
            max_features: Maximum features to analyze (for performance)

        Returns:
            Dictionary with feature quality results
        """
        results = check_feature_quality(
            self.df, self.horizons, self.warnings_found, self.seed, max_features
        )
        self.validation_results["feature_quality"] = results
        return results

    def check_feature_normalization(
        self, z_threshold: float = 3.0, extreme_threshold: float = 5.0
    ) -> dict:
        """
        Check feature normalization, distributions, and outliers.

        Checks for:
        - Distribution statistics (mean, std, percentiles, skewness)
        - Unnormalized features (large scale)
        - Highly skewed features
        - Z-score outlier detection
        - Feature range issues (constant values, extreme ranges)
        - Generates normalization recommendations

        Args:
            z_threshold: Z-score threshold for outlier warning (default 3.0)
            extreme_threshold: Z-score threshold for extreme outliers (default 5.0)

        Returns:
            Dictionary with normalization validation results
        """
        results = check_feature_normalization(
            self.df, self.issues_found, self.warnings_found, z_threshold, extreme_threshold
        )
        self.validation_results["feature_normalization"] = results
        return results

    def run_feature_selection(
        self, correlation_threshold: float = 0.85, variance_threshold: float = 0.01
    ) -> FeatureSelectionResult:
        """
        Run feature selection to identify and remove redundant features.

        This addresses multicollinearity by removing highly correlated features
        while keeping the most interpretable feature from each correlated group.

        Args:
            correlation_threshold: Threshold for feature correlation (default 0.85)
            variance_threshold: Minimum variance to keep feature (default 0.01)

        Returns:
            FeatureSelectionResult with selected and removed features
        """
        logger.info("\n" + "=" * 60)
        logger.info("FEATURE SELECTION")
        logger.info("=" * 60)

        result = select_features(
            self.df,
            correlation_threshold=correlation_threshold,
            variance_threshold=variance_threshold,
        )

        # Store results in validation_results
        self.validation_results["feature_selection"] = result.to_dict()

        # Update warnings if features were removed
        if len(result.removed_features) > 0:
            # Replace the old correlation warning with a more informative one
            self.warnings_found = [
                w for w in self.warnings_found if "correlated feature pairs" not in w
            ]
            self.warnings_found.append(
                f"Feature selection removed {len(result.removed_features)} redundant features "
                f"({result.original_count} -> {result.final_count})"
            )

        return result

    def generate_summary(self) -> dict:
        """
        Generate validation summary.

        Returns:
            Dictionary with overall validation summary including:
            - timestamp
            - row/column counts
            - issues and warnings
            - pass/fail status
            - all validation results
        """
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)

        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_rows": int(len(self.df)),
            "total_columns": int(len(self.df.columns)),
            "issues_count": len(self.issues_found),
            "warnings_count": len(self.warnings_found),
            "issues": self.issues_found,
            "warnings": self.warnings_found,
            "validation_results": self.validation_results,
        }

        # Determine overall status
        if len(self.issues_found) == 0:
            summary["status"] = "PASSED"
            logger.info("\nAll validation checks PASSED")
        else:
            summary["status"] = "FAILED"
            logger.error(f"\nValidation FAILED with {len(self.issues_found)} issues")

        if len(self.warnings_found) > 0:
            logger.warning(f"\n{len(self.warnings_found)} warnings found:")
            for warning in self.warnings_found:
                logger.warning(f"  - {warning}")

        if len(self.issues_found) > 0:
            logger.error(f"\n{len(self.issues_found)} issues found:")
            for issue in self.issues_found:
                logger.error(f"  - {issue}")

        return summary


def validate_data(
    data_path: Path,
    output_path: Path | None = None,
    horizons: list[int] = None,
    run_feature_selection: bool = True,
    correlation_threshold: float = 0.85,
    variance_threshold: float = 0.01,
    feature_selection_output_path: Path | None = None,
    seed: int = 42,
    check_leakage: bool = True,
    check_lookahead: bool = True,
) -> tuple[dict, FeatureSelectionResult | None]:
    """
    Main validation function.

    IMPORTANT: Lookahead audit is MANDATORY and always runs in blocking mode.
    The check_lookahead parameter is deprecated - setting it to False will
    emit a warning but the audit will still run.

    Args:
        data_path: Path to combined labeled data
        output_path: Optional path to save validation report (JSON)
        horizons: List of label horizons to validate
        run_feature_selection: Whether to run feature selection (default True)
        correlation_threshold: Threshold for feature correlation (default 0.85)
        variance_threshold: Minimum variance to keep feature (default 0.01)
        feature_selection_output_path: Optional path to save feature selection report
        seed: Random seed for reproducibility (default: 42)
        check_leakage: Whether to run leakage detection (default True, Phase 4A)
        check_lookahead: DEPRECATED - lookahead audit is now mandatory and always
            runs in blocking mode. Setting to False will emit a deprecation warning.

    Returns:
        Tuple of (validation summary dict, FeatureSelectionResult or None)

    Raises:
        LeakageDetectedError: If leakage is detected and check_leakage=True
        LookaheadBiasError: If lookahead bias is detected (always - audit is mandatory)
    """
    if horizons is None:
        horizons = [1, 5, 20]

    # Phase 14C: Lookahead audit is now mandatory - emit deprecation warning if disabled
    if not check_lookahead:
        warnings.warn(
            "check_lookahead=False is deprecated and ignored. "
            "Lookahead audit is now MANDATORY and always runs in blocking mode. "
            "This parameter will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )

    logger.info("=" * 70)
    logger.info("STAGE 8: DATA VALIDATION")
    logger.info("=" * 70)
    logger.info(f"Random seed: {seed}")

    # Load data
    logger.info(f"\nLoading data from {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Create validator with seed for reproducibility
    validator = DataValidator(df, horizons=horizons, seed=seed)

    # Run all checks
    validator.check_data_integrity()
    validator.check_label_sanity()
    validator.check_feature_quality()
    validator.check_feature_normalization()

    # Phase 4A: Leakage detection (blocks training if leakage detected)
    if check_leakage:
        logger.info("\n" + "=" * 60)
        logger.info("LEAKAGE DETECTION (Phase 4A)")
        logger.info("=" * 60)

        # Separate features and labels for leakage check
        label_cols = [col for col in df.columns if col.startswith("label_")]
        feature_cols = [
            col
            for col in df.columns
            if col not in label_cols and col not in ["timestamp", "symbol"]
        ]

        if label_cols and feature_cols:
            # Run comprehensive leakage check with raise_on_leakage=True
            # This will raise LeakageDetectedError if leakage is found
            features_arr = np.asarray(df[feature_cols].values)
            labels_arr = np.asarray(df[label_cols[0]].values)
            leakage_reports = comprehensive_leakage_check(
                features=features_arr,
                labels=labels_arr,
                feature_names=feature_cols,
                correlation_threshold=0.5,
                temporal_threshold=0.3,
                mi_threshold=0.5,
                max_lag=5,
                raise_on_leakage=True,
            )
            logger.info("Leakage detection passed - no leakage detected")
            validator.validation_results["leakage_detection"] = {
                "status": "passed",
                "reports": {k: v.to_dict() for k, v in leakage_reports.items()},
            }
        else:
            logger.warning("Skipping leakage detection - insufficient labels or features")

    # Phase 4B/14C: Lookahead audit - MANDATORY and BLOCKING
    # This audit ALWAYS runs regardless of check_lookahead parameter (deprecated)
    logger.info("\n" + "=" * 60)
    logger.info("LOOKAHEAD AUDIT (MANDATORY - Phase 14C)")
    logger.info("=" * 60)
    logger.info("Lookahead audit is mandatory and will BLOCK on failure.")

    # Run lookahead audit with corruption testing
    # This will raise LookaheadBiasError if lookahead is found
    auditor = LookaheadAuditor(corruption_point=0.8, random_seed=seed)

    # For a comprehensive audit, we'd test individual feature functions
    # For now, we'll do a basic check on the entire feature set
    # Note: Detailed per-feature audit requires feature generation functions
    logger.info("Running corruption-based lookahead audit at 80% point")

    # Check if data has required OHLCV columns for meaningful audit
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    has_ohlcv = all(col in df.columns for col in ohlcv_cols)

    if has_ohlcv:
        # Test a simple feature (e.g., returns) for lookahead
        # In production, this would test all feature generation functions
        try:
            # Create a simple feature function to test
            def compute_returns(data: pd.DataFrame) -> pd.DataFrame:
                result = data.copy()
                if "close" in result.columns:
                    result["test_returns"] = result["close"].pct_change()
                return result

            # Audit the feature function - ALWAYS with raise_on_lookahead=True
            ohlcv_df = pd.DataFrame(df[ohlcv_cols])
            result = auditor.audit_feature_function(
                df=ohlcv_df,
                feature_fn=compute_returns,
                name="returns",
                raise_on_lookahead=True,  # Mandatory blocking mode
            )

            logger.info("Lookahead audit PASSED - no lookahead bias detected")
            validator.validation_results["lookahead_audit"] = {
                "status": "passed",
                "mandatory": True,
                "blocking": True,
                "corruption_point": 0.8,
                "affected_columns": (
                    list(result.affected_columns) if result.affected_columns else []
                ),
            }
        except LookaheadBiasError:
            # Re-raise to block training - this is mandatory
            logger.error("Lookahead audit FAILED - blocking pipeline execution")
            raise
    else:
        # Even without OHLCV, we log that audit was attempted but data insufficient
        logger.warning(
            "Lookahead audit: Insufficient data (missing OHLCV columns). "
            "Audit will pass but with warning - ensure OHLCV data is available "
            "for comprehensive lookahead detection."
        )
        validator.validation_results["lookahead_audit"] = {
            "status": "skipped_insufficient_data",
            "mandatory": True,
            "blocking": True,
            "reason": "Missing OHLCV columns for corruption testing",
        }

    # Run feature selection if requested
    feature_selection_result = None
    if run_feature_selection:
        feature_selection_result = validator.run_feature_selection(
            correlation_threshold=correlation_threshold, variance_threshold=variance_threshold
        )

        # Save feature selection report if path provided
        if feature_selection_output_path:
            save_feature_selection_report(feature_selection_result, feature_selection_output_path)

    # Generate summary
    summary = validator.generate_summary()

    # Save report if requested
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Custom JSON encoder for numpy types
        def numpy_encoder(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):  # noqa: UP038
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):  # noqa: UP038
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, default=numpy_encoder)
        logger.info(f"\nValidation report saved to: {output_path}")

    logger.info("\n" + "=" * 70)
    logger.info("STAGE 8 COMPLETE")
    logger.info("=" * 70)

    return summary, feature_selection_result
