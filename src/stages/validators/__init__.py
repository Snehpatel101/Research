"""
Validators submodule for Stage 8 data validation.

Provides modular validation checks for data integrity, labels, features,
and normalization.
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from .integrity import check_data_integrity
from .labels import check_label_sanity
from .features import check_feature_quality
from .normalization import check_feature_normalization
from .data_contract import (
    DataContract,
    validate_ohlcv_schema,
    validate_labels,
    filter_invalid_labels,
    get_dataset_fingerprint,
    validate_feature_lookahead,
    summarize_label_distribution,
    REQUIRED_OHLCV,
    VALID_LABELS,
    INVALID_LABEL_SENTINEL,
    POSITIVE_COLUMNS,
)

# Re-export for convenience
__all__ = [
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
        horizons: Optional[List[int]] = None,
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
        self.validation_results: Dict = {}
        self.issues_found: List[str] = []
        self.warnings_found: List[str] = []

    def check_data_integrity(self) -> Dict:
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

    def check_label_sanity(self) -> Dict:
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

    def check_feature_quality(self, max_features: int = 50) -> Dict:
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
    ) -> Dict:
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

    def generate_summary(self) -> Dict:
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
