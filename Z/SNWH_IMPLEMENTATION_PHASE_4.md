# SNwH Implementation: Phase 4 - OOF Integrity

## Overview

Phase 4 implements strict OOF (Out-of-Fold) validation to prevent the coverage mismatches identified in `oof_sequence.py:194-206`. This ensures heterogeneous ensembles stack correctly.

**Critical Issue**: Sequence models lose `(seq_len - 1)` samples at the start due to windowing. If tabular and sequence OOF predictions are stacked without alignment, the meta-learner sees mismatched samples.

---

## 4.1 OOF Alignment Validator

### New File: `src/cross_validation/oof_alignment.py`

```python
"""
OOF Alignment Validator - Ensures OOF predictions from different models align.

Critical for heterogeneous ensembles where:
- Tabular models produce OOF for all N samples
- Sequence models produce OOF for (N - seq_len + 1) samples
- OOF must be aligned before stacking
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.contracts import ModelContract, get_model_contract

logger = logging.getLogger(__name__)


@dataclass
class OOFAlignmentResult:
    """Result of OOF alignment validation."""

    # Validity
    is_aligned: bool = False
    issues: list[str] = field(default_factory=list)

    # Sample counts per model
    sample_counts: dict[str, int] = field(default_factory=dict)

    # Common sample range
    common_start_idx: int = 0
    common_end_idx: int = 0
    n_common_samples: int = 0

    # Offset information
    offsets: dict[str, int] = field(default_factory=dict)

    def __repr__(self) -> str:
        status = "ALIGNED" if self.is_aligned else "MISALIGNED"
        return (
            f"OOFAlignmentResult({status}, "
            f"common={self.n_common_samples}, "
            f"offsets={self.offsets})"
        )


class OOFAlignmentValidator:
    """
    Validates and aligns OOF predictions from multiple models.

    For heterogeneous ensembles, different model types may have different
    sample coverage:
    - Tabular (2D): All N samples
    - Sequence (3D): N - (seq_len - 1) samples (missing first seq_len-1)
    - Multi-stream (4D): Similar to sequence

    This validator:
    1. Computes the common sample range across all models
    2. Validates that OOF predictions can be aligned
    3. Provides alignment instructions for stacking
    """

    def __init__(self):
        """Initialize validator."""
        self.model_coverages: dict[str, tuple[int, int, int]] = {}  # model -> (start, end, count)

    def register_oof(
        self,
        model_name: str,
        oof_indices: np.ndarray | None,
        n_total_samples: int,
        sequence_length: int | None = None,
    ) -> None:
        """
        Register OOF prediction coverage for a model.

        Args:
            model_name: Name of the model
            oof_indices: Array of sample indices covered by OOF (None = all)
            n_total_samples: Total number of samples in the dataset
            sequence_length: Sequence length if applicable
        """
        if oof_indices is not None:
            start_idx = int(oof_indices.min())
            end_idx = int(oof_indices.max()) + 1
            count = len(oof_indices)
        elif sequence_length:
            # Sequence models miss first (seq_len - 1) samples
            start_idx = sequence_length - 1
            end_idx = n_total_samples
            count = n_total_samples - (sequence_length - 1)
        else:
            # Tabular models cover all samples
            start_idx = 0
            end_idx = n_total_samples
            count = n_total_samples

        self.model_coverages[model_name] = (start_idx, end_idx, count)

        logger.debug(
            f"Registered OOF for {model_name}: "
            f"indices [{start_idx}, {end_idx}), count={count}"
        )

    def validate(self) -> OOFAlignmentResult:
        """
        Validate that all registered OOF predictions can be aligned.

        Returns:
            OOFAlignmentResult with alignment information
        """
        if not self.model_coverages:
            return OOFAlignmentResult(
                is_aligned=False,
                issues=["No OOF predictions registered"],
            )

        issues = []

        # Compute common range (intersection of all coverages)
        common_start = max(cov[0] for cov in self.model_coverages.values())
        common_end = min(cov[1] for cov in self.model_coverages.values())

        if common_start >= common_end:
            issues.append(
                f"No common sample range: start={common_start} >= end={common_end}"
            )
            return OOFAlignmentResult(
                is_aligned=False,
                issues=issues,
                sample_counts={m: c[2] for m, c in self.model_coverages.items()},
            )

        n_common = common_end - common_start

        # Compute offset for each model
        offsets = {}
        for model_name, (start, end, count) in self.model_coverages.items():
            offsets[model_name] = common_start - start

        # Validate alignment is possible
        sample_counts = {m: c[2] for m, c in self.model_coverages.items()}

        # Check if any model has too few samples in common range
        min_samples = min(sample_counts.values())
        if n_common < min_samples * 0.9:  # Allow 10% loss
            issues.append(
                f"Common range ({n_common}) is less than 90% of "
                f"smallest model ({min_samples})"
            )

        return OOFAlignmentResult(
            is_aligned=len(issues) == 0,
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
        Align OOF predictions to common sample range.

        Args:
            oof_dict: Dict mapping model_name -> OOF array
            alignment: Pre-computed alignment (or compute fresh)

        Returns:
            Dict mapping model_name -> aligned OOF array

        Raises:
            ValueError: If alignment is not possible
        """
        if alignment is None:
            alignment = self.validate()

        if not alignment.is_aligned:
            raise ValueError(
                f"Cannot align OOF predictions: {alignment.issues}"
            )

        aligned = {}
        for model_name, oof in oof_dict.items():
            if model_name not in self.model_coverages:
                raise ValueError(f"Model '{model_name}' not registered")

            offset = alignment.offsets[model_name]
            start = offset
            end = start + alignment.n_common_samples

            if end > len(oof):
                raise ValueError(
                    f"Model '{model_name}': cannot extract "
                    f"[{start}:{end}] from array of length {len(oof)}"
                )

            aligned[model_name] = oof[start:end]

        # Validate all aligned arrays have same length
        lengths = {m: len(a) for m, a in aligned.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"Aligned arrays have different lengths: {lengths}")

        logger.info(
            f"Aligned {len(aligned)} OOF predictions to "
            f"{alignment.n_common_samples} common samples"
        )

        return aligned


def compute_oof_coverage(
    model_name: str,
    oof_predictions: np.ndarray,
    n_total_samples: int,
) -> tuple[float, int]:
    """
    Compute OOF coverage statistics.

    Args:
        model_name: Name of the model
        oof_predictions: OOF prediction array (may have NaN for missing)
        n_total_samples: Total samples in dataset

    Returns:
        (coverage_ratio, n_missing)
    """
    if oof_predictions.ndim > 1:
        # For probability arrays, check first column
        valid = ~np.isnan(oof_predictions[:, 0])
    else:
        valid = ~np.isnan(oof_predictions)

    n_valid = int(valid.sum())
    n_missing = n_total_samples - n_valid
    coverage = n_valid / n_total_samples

    return coverage, n_missing


def validate_oof_for_stacking(
    oof_results: dict[str, "OOFPrediction"],
    min_coverage: float = 0.95,
) -> tuple[bool, list[str]]:
    """
    Validate OOF results are suitable for stacking.

    Args:
        oof_results: Dict mapping model_name -> OOFPrediction
        min_coverage: Minimum required coverage ratio

    Returns:
        (is_valid, list_of_issues)
    """
    from src.cross_validation.oof_core import OOFPrediction

    issues = []

    # Check coverage
    for model_name, oof in oof_results.items():
        if oof.coverage < min_coverage:
            issues.append(
                f"{model_name}: coverage {oof.coverage:.1%} < {min_coverage:.1%}"
            )

    # Check sample counts match
    counts = {m: len(oof.predictions) for m, oof in oof_results.items()}
    unique_counts = set(counts.values())

    if len(unique_counts) > 1:
        # Different counts - need alignment
        max_count = max(counts.values())
        min_count = min(counts.values())

        if min_count < max_count * 0.9:
            issues.append(
                f"Sample count mismatch: {counts}. "
                f"Alignment may lose >10% samples."
            )

    return len(issues) == 0, issues


__all__ = [
    "OOFAlignmentResult",
    "OOFAlignmentValidator",
    "compute_oof_coverage",
    "validate_oof_for_stacking",
]
```

---

## 4.2 Enhanced OOF Generation with Alignment

### File: `src/cross_validation/oof_sequence.py`

**Modifications around lines 194-206 to add strict validation**

```python
# ADD this validation after computing coverage (around line 206)

def generate_sequence_oof(
    self,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    config: dict[str, Any],
    seq_len: int,
    sample_weights: pd.Series | None = None,
    label_end_times: pd.Series | None = None,
    symbol_column: str | None = "symbol",
    strict_validation: bool = True,  # NEW parameter
) -> OOFPrediction:
    """
    Generate OOF predictions for a sequence model.

    ... existing docstring ...

    Args:
        ... existing args ...
        strict_validation: If True, raise error on coverage issues (default True)
    """
    # ... existing code until line 206 ...

    # Validate coverage (expected to be < 100% for sequence models due to lookback)
    coverage = float((~np.isnan(oof_preds)).mean())
    n_missing = int(np.isnan(oof_preds).sum())

    # Calculate expected coverage based on sequence length and boundaries
    n_boundaries = (
        len(seq_builder._symbol_boundaries) if seq_builder._symbol_boundaries is not None else 0
    )
    n_segments = n_boundaries + 1
    expected_missing = n_segments * seq_len
    expected_coverage = max(0.0, 1.0 - (expected_missing / n_samples))

    coverage_shortfall = expected_coverage - coverage

    # NEW: Strict validation for SNwH
    if strict_validation and coverage_shortfall > COVERAGE_WARNING_THRESHOLD:
        raise ValueError(
            f"{model_name}: OOF coverage {coverage:.1%} is unacceptably low "
            f"(expected ~{expected_coverage:.1%}). "
            f"Missing {n_missing} samples. "
            f"This will cause stacking alignment issues. "
            f"Set strict_validation=False to proceed with warning."
        )

    # ... rest of existing code ...

    # NEW: Store original indices for alignment
    valid_indices = np.where(~np.isnan(oof_preds))[0]

    return OOFPrediction(
        model_name=model_name,
        predictions=oof_df,
        fold_info=fold_info,
        coverage=coverage,
        # NEW: Additional alignment metadata
        original_indices=valid_indices,  # Indices that have predictions
        sequence_length=seq_len,
        n_total_samples=n_samples,
    )
```

---

## 4.3 Stacking Dataset Builder with Alignment

### File: `src/cross_validation/oof_stacking.py`

**Modifications to ensure aligned stacking**

```python
# ADD this class to handle heterogeneous stacking alignment

class HeterogeneousStackingBuilder:
    """
    Builds stacking datasets for heterogeneous ensembles.

    Handles alignment between tabular and sequence model OOF predictions
    to ensure the meta-learner sees correctly matched samples.
    """

    def __init__(
        self,
        purge_bars: int = 60,
        embargo_bars: int = 1440,
    ):
        """
        Initialize builder.

        Args:
            purge_bars: Purge bars for leakage prevention
            embargo_bars: Embargo bars for serial correlation
        """
        self.purge_bars = purge_bars
        self.embargo_bars = embargo_bars
        self.validator = OOFAlignmentValidator()

    def build_stacking_dataset(
        self,
        oof_results: dict[str, "OOFPrediction"],
        y: np.ndarray | pd.Series,
        sample_weights: np.ndarray | pd.Series | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Build aligned stacking dataset from heterogeneous OOF predictions.

        Args:
            oof_results: Dict mapping model_name -> OOFPrediction
            y: True labels (full dataset)
            sample_weights: Optional sample weights (full dataset)

        Returns:
            (X_stack, y_aligned, weights_aligned)

        Raises:
            ValueError: If alignment is not possible
        """
        from src.cross_validation.oof_core import OOFPrediction

        # Convert to numpy
        y = np.asarray(y)
        if sample_weights is not None:
            sample_weights = np.asarray(sample_weights)

        # Register coverage for each model
        for model_name, oof in oof_results.items():
            self.validator.register_oof(
                model_name=model_name,
                oof_indices=getattr(oof, 'original_indices', None),
                n_total_samples=len(y),
                sequence_length=getattr(oof, 'sequence_length', None),
            )

        # Validate alignment
        alignment = self.validator.validate()

        if not alignment.is_aligned:
            raise ValueError(
                f"Cannot build stacking dataset: {alignment.issues}"
            )

        # Extract and align OOF predictions
        aligned_oofs = {}
        for model_name, oof in oof_results.items():
            # Get probability columns
            prob_cols = [c for c in oof.predictions.columns if "_prob_" in c]
            probs = oof.predictions[prob_cols].values

            # Handle NaN values (missing predictions)
            valid_mask = ~np.isnan(probs[:, 0])
            valid_probs = probs[valid_mask]

            # Apply alignment offset
            offset = alignment.offsets[model_name]
            start = offset
            end = start + alignment.n_common_samples

            if end > len(valid_probs):
                logger.warning(
                    f"{model_name}: trimming to {len(valid_probs)} valid predictions "
                    f"(requested {alignment.n_common_samples})"
                )
                end = len(valid_probs)

            aligned_oofs[model_name] = valid_probs[start:end]

        # Stack aligned predictions
        n_models = len(aligned_oofs)
        n_samples = alignment.n_common_samples

        # Verify all have same length after trimming
        actual_lengths = {m: len(a) for m, a in aligned_oofs.items()}
        min_length = min(actual_lengths.values())

        if any(l != min_length for l in actual_lengths.values()):
            logger.warning(
                f"Trimming all OOF to {min_length} samples for alignment"
            )
            for m in aligned_oofs:
                aligned_oofs[m] = aligned_oofs[m][:min_length]

        n_samples = min_length

        # Build stacking feature matrix
        # Shape: (n_samples, n_models * n_classes)
        X_stack = np.hstack([aligned_oofs[m] for m in sorted(aligned_oofs.keys())])

        # Align labels and weights
        y_aligned = y[alignment.common_start_idx:alignment.common_start_idx + n_samples]

        weights_aligned = None
        if sample_weights is not None:
            weights_aligned = sample_weights[
                alignment.common_start_idx:alignment.common_start_idx + n_samples
            ]

        logger.info(
            f"Built stacking dataset: X{X_stack.shape}, "
            f"aligned from {len(y)} to {n_samples} samples "
            f"(offset={alignment.common_start_idx})"
        )

        return X_stack, y_aligned, weights_aligned

    def get_alignment_summary(self) -> dict[str, Any]:
        """Get summary of alignment decisions."""
        alignment = self.validator.validate()
        return {
            "is_aligned": alignment.is_aligned,
            "issues": alignment.issues,
            "sample_counts": alignment.sample_counts,
            "common_range": (alignment.common_start_idx, alignment.common_end_idx),
            "n_common_samples": alignment.n_common_samples,
            "offsets": alignment.offsets,
        }


__all__ = [
    # ... existing exports ...
    "HeterogeneousStackingBuilder",
]
```

---

## 4.4 Integration with oof_core.py

### File: `src/cross_validation/oof_core.py`

**Add alignment metadata to OOFPrediction dataclass**

```python
@dataclass
class OOFPrediction:
    """Container for OOF prediction results."""

    model_name: str
    predictions: pd.DataFrame  # Columns: datetime, model_prob_*, model_pred, model_confidence
    fold_info: list[dict[str, Any]]
    coverage: float

    # NEW: Alignment metadata for heterogeneous stacking
    original_indices: np.ndarray | None = None  # Indices that have valid predictions
    sequence_length: int | None = None  # For sequence models
    n_total_samples: int | None = None  # Total samples in source data

    @property
    def n_valid(self) -> int:
        """Number of samples with valid predictions."""
        if self.original_indices is not None:
            return len(self.original_indices)
        return len(self.predictions)

    @property
    def alignment_offset(self) -> int:
        """
        Offset from start of dataset to first valid prediction.

        For tabular models: 0
        For sequence models: seq_len - 1
        """
        if self.original_indices is not None and len(self.original_indices) > 0:
            return int(self.original_indices.min())
        if self.sequence_length:
            return self.sequence_length - 1
        return 0

    def get_aligned_probabilities(
        self,
        start_idx: int,
        n_samples: int,
    ) -> np.ndarray:
        """
        Get probability array aligned to specified range.

        Args:
            start_idx: Start index in original dataset
            n_samples: Number of samples to extract

        Returns:
            Aligned probability array
        """
        prob_cols = [c for c in self.predictions.columns if "_prob_" in c]
        probs = self.predictions[prob_cols].values

        # Adjust for alignment offset
        local_start = start_idx - self.alignment_offset
        local_end = local_start + n_samples

        if local_start < 0 or local_end > len(probs):
            raise ValueError(
                f"Requested range [{start_idx}, {start_idx + n_samples}) "
                f"exceeds valid range [{self.alignment_offset}, "
                f"{self.alignment_offset + len(probs)})"
            )

        return probs[local_start:local_end]
```

---

## Summary: Phase 4 Changes

| File | Type | Purpose |
|------|------|---------|
| `src/cross_validation/oof_alignment.py` | NEW | OOFAlignmentValidator, alignment utilities |
| `src/cross_validation/oof_sequence.py` | MODIFY | Add strict_validation, store original_indices |
| `src/cross_validation/oof_stacking.py` | MODIFY | Add HeterogeneousStackingBuilder |
| `src/cross_validation/oof_core.py` | MODIFY | Add alignment metadata to OOFPrediction |

## Dependencies

- Phase 0-3 must be complete
- Requires `src/contracts/` for model contracts

## Key Design Decisions

1. **Explicit Alignment**: Never silently drop samples - always compute and log alignment
2. **Strict by Default**: Raise errors on coverage issues unless explicitly overridden
3. **Preserved Indices**: OOF results track which original samples have predictions
4. **Common Range**: Stacking uses intersection of all model coverages

## Alignment Rules

| Model Type | OOF Coverage | Alignment Offset |
|------------|--------------|------------------|
| Tabular (2D) | 100% | 0 |
| Sequence (3D) | ~(N - seq_len + 1) / N | seq_len - 1 |
| Multi-stream (4D) | ~(N - seq_len + 1) / N | seq_len - 1 |

## Usage Example

```python
from src.cross_validation.oof_alignment import (
    OOFAlignmentValidator,
    validate_oof_for_stacking,
)
from src.cross_validation.oof_stacking import HeterogeneousStackingBuilder

# Validate before building
is_valid, issues = validate_oof_for_stacking(oof_results)
if not is_valid:
    print(f"Issues: {issues}")

# Build aligned stacking dataset
builder = HeterogeneousStackingBuilder()
X_stack, y_aligned, w_aligned = builder.build_stacking_dataset(
    oof_results={"xgboost": xgb_oof, "lstm": lstm_oof, "patchtst": patchtst_oof},
    y=y_train,
    sample_weights=weights,
)

# X_stack.shape = (n_common_samples, n_models * n_classes)
```

## Next Steps

After Phase 4 is implemented, proceed to Phase 5 (Feature Strategy Integration) which will:
1. Wire MODEL_FEATURE_STRATEGIES into trainer config flow
2. Implement per-model feature selection
3. Add feature optimization integration
