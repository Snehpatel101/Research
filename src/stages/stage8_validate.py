"""
Stage 8: Comprehensive Data Validation

Performs integrity, label sanity, feature quality checks, and feature selection.
This module is a thin wrapper around the validators submodule.
"""
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.horizon_config import LOOKBACK_HORIZONS
from src.stages.validators import DataValidator
from src.utils.feature_selection import (
    FeatureSelectionResult,
    save_feature_selection_report,
    select_features,
)

warnings.filterwarnings('ignore')

# Configure logging - use NullHandler to avoid duplicate logs when imported as module
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class DataValidatorWithFeatureSelection(DataValidator):
    """
    Extended DataValidator with feature selection capability.

    Inherits all validation methods from DataValidator and adds
    feature selection functionality.
    """

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
        logger.info("\n" + "=" * 60)
        logger.info("FEATURE SELECTION")
        logger.info("=" * 60)

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


def _numpy_encoder(obj):
    """Custom JSON encoder for numpy types."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f'Object of type {type(obj)} is not JSON serializable')


def validate_data(
    data_path: Path,
    output_path: Optional[Path] = None,
    horizons: Optional[List[int]] = None,
    run_feature_selection: bool = True,
    correlation_threshold: float = 0.85,
    variance_threshold: float = 0.01,
    feature_selection_output_path: Optional[Path] = None,
    seed: int = 42
) -> Tuple[Dict, Optional[FeatureSelectionResult]]:
    """
    Main validation function.

    Args:
        data_path: Path to combined labeled data
        output_path: Optional path to save validation report (JSON)
        horizons: List of label horizons to validate (default: LOOKBACK_HORIZONS)
        run_feature_selection: Whether to run feature selection (default True)
        correlation_threshold: Threshold for feature correlation (default 0.85)
        variance_threshold: Minimum variance to keep feature (default 0.01)
        feature_selection_output_path: Optional path to save feature selection report
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (validation summary dict, FeatureSelectionResult or None)
    """
    if horizons is None:
        horizons = LOOKBACK_HORIZONS

    logger.info("=" * 70)
    logger.info("STAGE 8: DATA VALIDATION")
    logger.info("=" * 70)
    logger.info(f"Random seed: {seed}")

    # Load data
    logger.info(f"\nLoading data from {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Create validator with seed for reproducibility
    validator = DataValidatorWithFeatureSelection(df, horizons=horizons, seed=seed)

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
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=_numpy_encoder)
        logger.info(f"\nValidation report saved to: {output_path}")

    logger.info("\n" + "=" * 70)
    logger.info("STAGE 8 COMPLETE")
    logger.info("=" * 70)

    return summary, feature_selection_result


def main():
    """Run validation on default configuration."""
    from src.config import FINAL_DATA_DIR, RANDOM_SEED, RESULTS_DIR

    data_path = FINAL_DATA_DIR / "combined_final_labeled.parquet"
    output_path = RESULTS_DIR / "validation_report.json"
    feature_selection_path = RESULTS_DIR / "feature_selection_report.json"

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return

    summary, feature_selection_result = validate_data(
        data_path,
        output_path,
        run_feature_selection=True,
        feature_selection_output_path=feature_selection_path,
        seed=RANDOM_SEED
    )

    logger.info(f"\nValidation status: {summary['status']}")
    logger.info(f"Issues: {summary['issues_count']}")
    logger.info(f"Warnings: {summary['warnings_count']}")

    if feature_selection_result:
        logger.info("\nFeature Selection Results:")
        logger.info(f"  Original features: {feature_selection_result.original_count}")
        logger.info(f"  Selected features: {feature_selection_result.final_count}")
        logger.info(f"  Removed features: {len(feature_selection_result.removed_features)}")


if __name__ == "__main__":
    main()
