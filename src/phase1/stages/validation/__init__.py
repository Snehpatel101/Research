"""
Validators submodule for Stage 8 data validation.

Provides modular validation checks for data integrity, labels, features,
and normalization.
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.phase1.utils.feature_selection import (
    FeatureSelectionResult,
    save_feature_selection_report,
    select_features,
)

from .data_contract import (
    INVALID_LABEL_SENTINEL,
    POSITIVE_COLUMNS,
    REQUIRED_OHLCV,
    VALID_LABELS,
    DataContract,
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
    'validate_data',
    # DataValidator class
    'DataValidator',
    # Existing validators
    'check_data_integrity',
    'check_label_sanity',
    'check_feature_quality',
    'check_feature_normalization',
    # Data contract validation
    'DataContract',
    'validate_ohlcv_schema',
    'validate_labels',
    'filter_invalid_labels',
    'get_dataset_fingerprint',
    'validate_feature_lookahead',
    'summarize_label_distribution',
    # Feature selection
    'FeatureSelectionResult',
    # Constants
    'REQUIRED_OHLCV',
    'VALID_LABELS',
    'INVALID_LABEL_SENTINEL',
    'POSITIVE_COLUMNS',
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

    def __init__(
        self,
        df: pd.DataFrame,
        horizons: list[int] | None = None,
        seed: int = 42
    ):
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
        self.validation_results['data_integrity'] = results
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
        self.validation_results['label_sanity'] = results
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
        self.validation_results['feature_quality'] = results
        return results

    def check_feature_normalization(
        self,
        z_threshold: float = 3.0,
        extreme_threshold: float = 5.0
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
            self.df, self.issues_found, self.warnings_found,
            z_threshold, extreme_threshold
        )
        self.validation_results['feature_normalization'] = results
        return results

    def run_feature_selection(
        self,
        correlation_threshold: float = 0.85,
        variance_threshold: float = 0.01
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
        logger.info("\n" + "="*60)
        logger.info("FEATURE SELECTION")
        logger.info("="*60)

        result = select_features(
            self.df,
            correlation_threshold=correlation_threshold,
            variance_threshold=variance_threshold
        )

        # Store results in validation_results
        self.validation_results['feature_selection'] = result.to_dict()

        # Update warnings if features were removed
        if len(result.removed_features) > 0:
            # Replace the old correlation warning with a more informative one
            self.warnings_found = [
                w for w in self.warnings_found
                if 'correlated feature pairs' not in w
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
            'timestamp': datetime.now().isoformat(),
            'total_rows': int(len(self.df)),
            'total_columns': int(len(self.df.columns)),
            'issues_count': len(self.issues_found),
            'warnings_count': len(self.warnings_found),
            'issues': self.issues_found,
            'warnings': self.warnings_found,
            'validation_results': self.validation_results
        }

        # Determine overall status
        if len(self.issues_found) == 0:
            summary['status'] = 'PASSED'
            logger.info("\nAll validation checks PASSED")
        else:
            summary['status'] = 'FAILED'
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
    horizons: list[int] = [1, 5, 20],
    run_feature_selection: bool = True,
    correlation_threshold: float = 0.85,
    variance_threshold: float = 0.01,
    feature_selection_output_path: Path | None = None,
    seed: int = 42
) -> tuple[dict, FeatureSelectionResult | None]:
    """
    Main validation function.

    Args:
        data_path: Path to combined labeled data
        output_path: Optional path to save validation report (JSON)
        horizons: List of label horizons to validate
        run_feature_selection: Whether to run feature selection (default True)
        correlation_threshold: Threshold for feature correlation (default 0.85)
        variance_threshold: Minimum variance to keep feature (default 0.01)
        feature_selection_output_path: Optional path to save feature selection report
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (validation summary dict, FeatureSelectionResult or None)
    """
    logger.info("="*70)
    logger.info("STAGE 8: DATA VALIDATION")
    logger.info("="*70)
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

    # Run feature selection if requested
    feature_selection_result = None
    if run_feature_selection:
        feature_selection_result = validator.run_feature_selection(
            correlation_threshold=correlation_threshold,
            variance_threshold=variance_threshold
        )

        # Save feature selection report if path provided
        if feature_selection_output_path:
            save_feature_selection_report(
                feature_selection_result,
                feature_selection_output_path
            )

    # Generate summary
    summary = validator.generate_summary()

    # Save report if requested
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Custom JSON encoder for numpy types
        def numpy_encoder(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            raise TypeError(f'Object of type {type(obj)} is not JSON serializable')

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=numpy_encoder)
        logger.info(f"\nValidation report saved to: {output_path}")

    logger.info("\n" + "="*70)
    logger.info("STAGE 8 COMPLETE")
    logger.info("="*70)

    return summary, feature_selection_result
