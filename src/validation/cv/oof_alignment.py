"""
OOF alignment validation for heterogeneous ensembles.

This module handles alignment validation between OOF predictions from
different model types (tabular vs sequence) which may have different
coverage ranges due to sequence lookback requirements.

For heterogeneous ensembles, different models may produce OOF predictions
for different sample ranges:
- Tabular models: Full coverage (indices 0 to N-1)
- Sequence models: Partial coverage (indices seq_len-1 to N-1)

This module validates alignment and extracts the common sample range
that all models can predict.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from src.core.contracts import ModelContract, get_model_contract

if TYPE_CHECKING:
    from .oof_core import OOFPrediction

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class OOFAlignmentResult:
    """
    Result of OOF alignment validation.

    Contains information about the common sample range across all models
    and any alignment issues detected.

    Attributes:
        is_aligned: Whether all models can be aligned (have overlapping ranges)
        issues: List of alignment issues found
        sample_counts: Number of valid samples per model
        common_start_idx: Start index of common range (inclusive)
        common_end_idx: End index of common range (exclusive)
        n_common_samples: Number of samples in common range
        offsets: Offset from common_start for each model's OOF array
    """

    is_aligned: bool
    issues: list[str] = field(default_factory=list)
    sample_counts: dict[str, int] = field(default_factory=dict)
    common_start_idx: int = 0
    common_end_idx: int = 0
    n_common_samples: int = 0
    offsets: dict[str, int] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Concise string representation."""
        status = "ALIGNED" if self.is_aligned else "MISALIGNED"
        return (
            f"OOFAlignmentResult({status}, "
            f"common_range=[{self.common_start_idx}:{self.common_end_idx}], "
            f"n_common={self.n_common_samples}, "
            f"n_models={len(self.sample_counts)})"
        )


# =============================================================================
# OOF COVERAGE TRACKING
# =============================================================================


@dataclass
class ModelCoverage:
    """
    Coverage information for a single model's OOF predictions.

    Attributes:
        model_name: Name of the model
        start_idx: First valid sample index (inclusive)
        end_idx: Last valid sample index (exclusive)
        n_samples: Number of valid samples
        n_total: Total number of samples in dataset
    """

    model_name: str
    start_idx: int
    end_idx: int
    n_samples: int
    n_total: int

    @property
    def coverage_ratio(self) -> float:
        """Fraction of total samples covered."""
        if self.n_total == 0:
            return 0.0
        return self.n_samples / self.n_total


# =============================================================================
# OOF ALIGNMENT VALIDATOR
# =============================================================================


class OOFAlignmentValidator:
    """
    Validates and aligns OOF predictions from heterogeneous model ensembles.

    Handles the case where different models have different coverage ranges:
    - Tabular models typically cover all samples
    - Sequence models lose samples at the start due to lookback requirements

    This validator computes the common range (intersection) that all models
    can predict, enabling proper stacking ensemble training.

    Example:
        >>> validator = OOFAlignmentValidator()
        >>> validator.register_oof("xgboost", None, n_total=1000)  # Full coverage
        >>> validator.register_oof("lstm", None, n_total=1000, sequence_length=60)  # Starts at 59
        >>> result = validator.validate()
        >>> if result.is_aligned:
        ...     aligned_oofs = validator.align_oof_predictions(oof_dict, result)
    """

    def __init__(self) -> None:
        """Initialize validator with empty model coverages."""
        self.model_coverages: dict[str, ModelCoverage] = {}

    def register_oof(
        self,
        model_name: str,
        oof_indices: np.ndarray | None,
        n_total_samples: int,
        sequence_length: int | None = None,
    ) -> None:
        """
        Register OOF coverage for a model.

        Args:
            model_name: Name of the model
            oof_indices: Array of valid sample indices, or None to infer from sequence_length
            n_total_samples: Total number of samples in the dataset
            sequence_length: Sequence length for sequence models. If set and oof_indices is None,
                coverage is computed as (sequence_length-1, n_total_samples).
                If both are None, coverage is (0, n_total_samples).
        """
        if n_total_samples <= 0:
            raise ValueError(f"n_total_samples must be positive, got {n_total_samples}")

        if oof_indices is not None:
            # Explicit indices provided
            if len(oof_indices) == 0:
                start_idx = 0
                end_idx = 0
                n_samples = 0
            else:
                start_idx = int(oof_indices.min())
                end_idx = int(oof_indices.max()) + 1  # Exclusive end
                n_samples = len(oof_indices)
        elif sequence_length is not None and sequence_length > 0:
            # Infer from sequence length (sequence models start later)
            start_idx = sequence_length - 1
            end_idx = n_total_samples
            n_samples = max(0, end_idx - start_idx)
        else:
            # Full coverage (tabular models)
            start_idx = 0
            end_idx = n_total_samples
            n_samples = n_total_samples

        coverage = ModelCoverage(
            model_name=model_name,
            start_idx=start_idx,
            end_idx=end_idx,
            n_samples=n_samples,
            n_total=n_total_samples,
        )

        self.model_coverages[model_name] = coverage
        logger.debug(
            f"Registered OOF coverage for {model_name}: "
            f"[{start_idx}:{end_idx}] ({n_samples}/{n_total_samples} samples, "
            f"{coverage.coverage_ratio:.1%} coverage)"
        )

    def validate(self) -> OOFAlignmentResult:
        """
        Validate alignment across all registered models.

        Computes the common range (intersection) of all model coverages
        and identifies any alignment issues.

        Returns:
            OOFAlignmentResult with alignment status and common range info
        """
        if not self.model_coverages:
            return OOFAlignmentResult(
                is_aligned=False,
                issues=["No models registered for alignment validation"],
            )

        issues: list[str] = []
        sample_counts: dict[str, int] = {}

        # Collect coverage info
        for name, coverage in self.model_coverages.items():
            sample_counts[name] = coverage.n_samples

        # Check consistent n_total across models
        n_totals = {cov.n_total for cov in self.model_coverages.values()}
        if len(n_totals) > 1:
            issues.append(
                f"Inconsistent total sample counts across models: {n_totals}. "
                "All models should be trained on the same dataset."
            )

        # Compute common range (intersection of all coverage ranges)
        # common_start = max of all start indices
        # common_end = min of all end indices
        common_start = max(cov.start_idx for cov in self.model_coverages.values())
        common_end = min(cov.end_idx for cov in self.model_coverages.values())
        n_common = max(0, common_end - common_start)

        if n_common == 0:
            issues.append(
                f"No common sample range found. Ranges do not overlap. "
                f"common_start={common_start}, common_end={common_end}"
            )

        # Compute offsets: how far into each model's OOF array does the common range start
        # offset[model] = common_start - model_start
        offsets: dict[str, int] = {}
        for name, coverage in self.model_coverages.items():
            offsets[name] = common_start - coverage.start_idx

        # Check for significant coverage loss
        n_total = max(n_totals) if n_totals else 0
        if n_total > 0:
            coverage_ratio = n_common / n_total
            if coverage_ratio < 0.80:
                issues.append(
                    f"Low common coverage: {coverage_ratio:.1%} ({n_common}/{n_total} samples). "
                    "Consider using models with similar lookback requirements."
                )

        # Log alignment summary
        logger.info(
            f"OOF alignment: common_range=[{common_start}:{common_end}], "
            f"n_common={n_common} samples"
        )
        for name, coverage in self.model_coverages.items():
            logger.debug(
                f"  {name}: range=[{coverage.start_idx}:{coverage.end_idx}], "
                f"offset={offsets[name]}"
            )

        is_aligned = n_common > 0 and len(issues) == 0

        return OOFAlignmentResult(
            is_aligned=is_aligned,
            issues=issues,
            sample_counts=sample_counts,
            common_start_idx=common_start,
            common_end_idx=common_end,
            n_common_samples=n_common,
            offsets=offsets,
        )

    def align_oof_predictions(
        self,
        oof_dict: dict[str, np.ndarray],
        alignment: OOFAlignmentResult | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Extract aligned portions from each model's OOF predictions.

        Given OOF prediction arrays from different models with potentially
        different coverage ranges, extract the common aligned portion.

        Args:
            oof_dict: Dictionary mapping model_name -> OOF prediction array.
                Each array should have shape (n_model_samples,) or (n_model_samples, n_classes).
            alignment: Pre-computed alignment result. If None, validate() is called.

        Returns:
            Dictionary mapping model_name -> aligned OOF array.
            All arrays will have the same number of samples (n_common_samples).

        Raises:
            ValueError: If alignment fails or arrays have unexpected shapes.
        """
        if alignment is None:
            alignment = self.validate()

        if not alignment.is_aligned:
            raise ValueError(
                f"Cannot align OOF predictions: {alignment.issues}. "
                "Ensure all models have overlapping coverage ranges."
            )

        aligned_oofs: dict[str, np.ndarray] = {}

        for model_name, oof_array in oof_dict.items():
            if model_name not in self.model_coverages:
                raise ValueError(
                    f"Model '{model_name}' not registered. "
                    f"Registered models: {list(self.model_coverages.keys())}"
                )

            coverage = self.model_coverages[model_name]
            offset = alignment.offsets[model_name]
            n_common = alignment.n_common_samples

            # Validate array length matches expected coverage
            expected_len = coverage.n_samples
            actual_len = len(oof_array)
            if actual_len != expected_len:
                raise ValueError(
                    f"OOF array length mismatch for {model_name}: "
                    f"expected {expected_len}, got {actual_len}"
                )

            # Extract aligned portion
            # offset is the distance from the model's start to common_start
            # We extract oof[offset:offset+n_common]
            end_idx = offset + n_common
            if end_idx > actual_len:
                raise ValueError(
                    f"Cannot extract aligned portion for {model_name}: "
                    f"end_idx={end_idx} exceeds array length {actual_len}"
                )

            aligned_oofs[model_name] = oof_array[offset:end_idx]
            logger.debug(
                f"Aligned {model_name}: extracted [{offset}:{end_idx}] "
                f"from array of length {actual_len}"
            )

        return aligned_oofs


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def compute_oof_coverage(
    model_name: str,
    oof_predictions: np.ndarray,
    n_total_samples: int,
) -> tuple[float, int]:
    """
    Compute OOF coverage statistics for a model.

    Args:
        model_name: Name of the model (used for logging)
        oof_predictions: OOF prediction array
        n_total_samples: Total number of samples in dataset

    Returns:
        Tuple of (coverage_ratio, n_missing) where:
            - coverage_ratio: Fraction of samples with valid predictions
            - n_missing: Number of samples without predictions
    """
    if n_total_samples <= 0:
        return 0.0, 0

    n_valid = len(oof_predictions)
    n_missing = n_total_samples - n_valid
    coverage_ratio = n_valid / n_total_samples

    logger.debug(
        f"{model_name}: coverage={coverage_ratio:.1%}, "
        f"valid={n_valid}, missing={n_missing}"
    )

    return coverage_ratio, n_missing


def validate_oof_for_stacking(
    oof_results: dict[str, "OOFPrediction"],
    min_coverage: float = 0.95,
) -> tuple[bool, list[str]]:
    """
    Validate OOF predictions are suitable for stacking ensemble.

    Checks that:
    1. All models have sufficient coverage (>= min_coverage)
    2. Sample counts are consistent across models (after alignment)
    3. Predictions from different models can be aligned

    Args:
        oof_results: Dictionary mapping model_name -> OOFPrediction
        min_coverage: Minimum acceptable coverage ratio (default 0.95)

    Returns:
        Tuple of (is_valid, issues) where:
            - is_valid: True if all validations pass
            - issues: List of validation issues found
    """
    issues: list[str] = []

    if not oof_results:
        return False, ["No OOF results provided"]

    # Check individual model coverage
    for model_name, oof_pred in oof_results.items():
        if oof_pred.coverage < min_coverage:
            issues.append(
                f"{model_name}: coverage {oof_pred.coverage:.1%} < "
                f"minimum {min_coverage:.1%}"
            )

    # Check sample count alignment
    sample_counts = {name: len(pred.predictions) for name, pred in oof_results.items()}
    unique_counts = set(sample_counts.values())

    if len(unique_counts) > 1:
        # Different sample counts - this is expected for heterogeneous ensembles
        # with sequence models, but we should validate they can be aligned
        validator = OOFAlignmentValidator()

        for model_name, oof_pred in oof_results.items():
            n_samples = len(oof_pred.predictions)
            # Try to get sequence length from model contract
            try:
                contract = get_model_contract(model_name)
                seq_len = contract.sequence_length if contract.requires_sequences else None
            except ValueError:
                seq_len = None

            # If we don't have explicit indices, infer from contract
            validator.register_oof(
                model_name=model_name,
                oof_indices=None,
                n_total_samples=n_samples,
                sequence_length=seq_len,
            )

        alignment_result = validator.validate()
        if not alignment_result.is_aligned:
            issues.extend(alignment_result.issues)

        # Check if common range has sufficient coverage
        if alignment_result.n_common_samples > 0:
            max_samples = max(sample_counts.values())
            common_coverage = alignment_result.n_common_samples / max_samples
            if common_coverage < min_coverage:
                issues.append(
                    f"Common aligned range coverage {common_coverage:.1%} < "
                    f"minimum {min_coverage:.1%}"
                )

    is_valid = len(issues) == 0
    return is_valid, issues


__all__ = [
    "OOFAlignmentResult",
    "OOFAlignmentValidator",
    "compute_oof_coverage",
    "validate_oof_for_stacking",
]
