"""
OOF Alignment utilities for heterogeneous ensembles.

PHASE_2: Handles alignment of OOF predictions from models with different
sample coverage (tabular vs sequence vs multi-stream).

Problem Context:
- Tabular models (2D): 100% sample coverage
- Sequence models (3D): ~98% coverage (loses seq_len-1 samples from start)
- Multi-stream models (4D): Similar to sequence models

For stacking ensembles with mixed model types, OOF predictions must be
aligned to common indices before they can be used as meta-learner features.

Example:
    # After training individual models with CV
    oof_xgb = trainer.get_oof_result("xgboost")    # 10000 samples
    oof_lstm = trainer.get_oof_result("lstm")       # 9941 samples (seq_len=60)
    oof_patch = trainer.get_oof_result("patchtst")  # 9941 samples

    # Align for stacking
    aligned = align_oof_predictions([oof_xgb, oof_lstm, oof_patch])
    X_stack = aligned.stacking_features  # Shape: (9941, 9 + 2)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.core.constants import (
    DEFAULT_SEQUENCE_LENGTH,
    LABEL_CLASSES,
    MODEL_DATA_RANKS,
    N_CLASSES,
)
from src.core.interfaces import OOFResult

# =============================================================================
# ALIGNED OOF RESULT
# =============================================================================


@dataclass
class AlignedOOFResult:
    """
    Aligned OOF predictions from multiple models.

    All models' predictions are aligned to common_indices,
    with NaN for samples where a model has no prediction.

    Attributes:
        probabilities: Shape (n_common, n_models * n_classes). Flattened
            probability matrix where each model contributes n_classes columns.
        predictions: Shape (n_common, n_models). Predicted class labels,
            with -999 indicating missing predictions.
        common_indices: Array of sample indices present in the aligned result.
        model_names: List of model names in order.
        coverage: Per-model coverage ratio (0.0 to 1.0).
        n_common: Number of aligned samples.
        n_models: Number of models.
        n_classes: Number of prediction classes.
    """

    probabilities: np.ndarray  # Shape: (n_common, n_models * n_classes)
    predictions: np.ndarray  # Shape: (n_common, n_models)
    common_indices: np.ndarray  # Common sample indices
    model_names: list[str]
    coverage: dict[str, float]  # Per-model coverage ratio
    n_common: int
    n_models: int
    n_classes: int = N_CLASSES

    # Sentinel value for missing predictions
    MISSING_PREDICTION: int = -999

    @property
    def stacking_features(self) -> np.ndarray:
        """
        Get features for stacking (prob columns + derived features).

        Returns:
            Array of shape (n_common, n_models * n_classes + 2) containing:
            - All probability columns (n_models * n_classes)
            - Mean confidence per sample (1)
            - Prediction agreement ratio (1)
        """
        # Base: all probability columns
        features = [self.probabilities]

        # Derived: mean confidence per sample
        # Reshape to (n_common, n_models, n_classes) to get max per model
        probs_reshaped = self.probabilities.reshape(self.n_common, self.n_models, self.n_classes)
        # Max probability per model (confidence)
        confidences = np.nanmax(probs_reshaped, axis=2)  # (n_common, n_models)
        # Mean confidence across models (ignoring NaN)
        mean_confidence = np.nanmean(confidences, axis=1, keepdims=True)
        features.append(mean_confidence)

        # Derived: prediction agreement (how many models agree on class)
        agreement = self._compute_agreement()
        features.append(agreement)

        return np.hstack(features)

    def _compute_agreement(self) -> np.ndarray:
        """
        Compute prediction agreement ratio per sample.

        For each sample, calculates the fraction of models that agree
        on the most common prediction (excluding missing predictions).

        Returns:
            Array of shape (n_common, 1) with agreement ratios.
        """
        agreement = np.zeros((self.n_common, 1), dtype=np.float32)

        for i in range(self.n_common):
            preds = self.predictions[i]
            # Filter out missing predictions
            valid = preds[preds != self.MISSING_PREDICTION]

            if len(valid) > 0:
                unique, counts = np.unique(valid, return_counts=True)
                agreement[i] = counts.max() / len(valid)
            else:
                # No valid predictions - set to NaN
                agreement[i] = np.nan

        return agreement

    def get_feature_names(self) -> list[str]:
        """
        Get column names for stacking features.

        Returns:
            List of feature names corresponding to stacking_features columns.
        """
        names = []
        class_names = list(LABEL_CLASSES.values())  # ["short", "neutral", "long"]

        # Probability columns
        for model in self.model_names:
            for cls in class_names:
                names.append(f"{model}_prob_{cls}")

        # Derived features
        names.append("mean_confidence")
        names.append("prediction_agreement")

        return names

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert stacking features to a DataFrame.

        Returns:
            DataFrame with stacking features and common_indices as index.
        """
        return pd.DataFrame(
            self.stacking_features,
            index=self.common_indices,
            columns=self.get_feature_names(),
        )

    def get_model_probabilities(self, model_name: str) -> np.ndarray:
        """
        Get probability columns for a specific model.

        Args:
            model_name: Name of the model.

        Returns:
            Array of shape (n_common, n_classes).

        Raises:
            ValueError: If model_name not found.
        """
        if model_name not in self.model_names:
            raise ValueError(f"Model '{model_name}' not found. " f"Available: {self.model_names}")

        idx = self.model_names.index(model_name)
        start_col = idx * self.n_classes
        end_col = start_col + self.n_classes
        return self.probabilities[:, start_col:end_col]

    def get_valid_mask(self) -> np.ndarray:
        """
        Get mask of samples with valid predictions from ALL models.

        Returns:
            Boolean array of shape (n_common,).
        """
        # Check if any probability is NaN for each sample
        # Reshape and check across all models/classes
        probs_reshaped = self.probabilities.reshape(self.n_common, self.n_models, self.n_classes)
        # Valid if no NaN in any model/class
        return ~np.any(np.isnan(probs_reshaped), axis=(1, 2))

    def drop_incomplete(self) -> AlignedOOFResult:
        """
        Return new AlignedOOFResult with only complete samples.

        Samples where any model has missing predictions are removed.

        Returns:
            New AlignedOOFResult with only complete samples.
        """
        valid_mask = self.get_valid_mask()
        n_valid = valid_mask.sum()

        return AlignedOOFResult(
            probabilities=self.probabilities[valid_mask],
            predictions=self.predictions[valid_mask],
            common_indices=self.common_indices[valid_mask],
            model_names=self.model_names,
            coverage=dict.fromkeys(self.model_names, 1.0),  # All complete
            n_common=n_valid,
            n_models=self.n_models,
            n_classes=self.n_classes,
        )

    def summary(self) -> str:
        """
        Get summary string of the aligned result.

        Returns:
            Human-readable summary string.
        """
        lines = [
            "AlignedOOFResult Summary:",
            f"  Models: {self.n_models}",
            f"  Samples: {self.n_common}",
            f"  Classes: {self.n_classes}",
            f"  Feature columns: {self.n_models * self.n_classes + 2}",
            "  Coverage:",
        ]
        for name, cov in self.coverage.items():
            lines.append(f"    {name}: {cov:.2%}")

        valid_count = self.get_valid_mask().sum()
        lines.append(f"  Complete samples: {valid_count} ({valid_count/self.n_common:.2%})")

        return "\n".join(lines)


# =============================================================================
# OOF ALIGNER
# =============================================================================


class OOFAligner:
    """
    Aligns OOF predictions from heterogeneous models.

    This class handles the alignment of out-of-fold predictions from models
    that may have different sample coverage due to their data requirements:

    - Tabular models (2D): Full coverage, use all samples
    - Sequence models (3D): Lose (seq_len - 1) samples from the start
    - Multi-stream models (4D): Similar loss to sequence models

    Example:
        aligner = OOFAligner()
        aligned = aligner.align([oof_xgb, oof_lstm, oof_patchtst])
        X_stack = aligned.stacking_features

        # Or use target indices
        target_idx = np.arange(1000, 10000)
        aligned = aligner.align(oof_results, target_indices=target_idx)
    """

    def __init__(self, n_classes: int = N_CLASSES):
        """
        Initialize the OOF aligner.

        Args:
            n_classes: Number of prediction classes (default: 3).
        """
        self.n_classes = n_classes

    def align(
        self,
        oof_results: list[OOFResult],
        target_indices: np.ndarray | None = None,
        strategy: str = "intersection",
    ) -> AlignedOOFResult:
        """
        Align OOF predictions from multiple models.

        Args:
            oof_results: List of OOFResult from each model.
            target_indices: Optional target indices to align to.
                If None, uses intersection of all model indices.
            strategy: Alignment strategy when target_indices is None:
                - "intersection": Use indices common to ALL models (default)
                - "union": Use indices present in ANY model (allows NaN)

        Returns:
            AlignedOOFResult with aligned predictions.

        Raises:
            ValueError: If oof_results is empty or strategy is invalid.
        """
        if not oof_results:
            raise ValueError("Empty oof_results list")

        if strategy not in ("intersection", "union"):
            raise ValueError(f"Invalid strategy: {strategy}. Use 'intersection' or 'union'")

        model_names = [r.model_name for r in oof_results]
        n_models = len(oof_results)

        # Determine common indices
        if target_indices is not None:
            common_indices = np.asarray(target_indices, dtype=np.int64)
        elif strategy == "intersection":
            common_indices = self._find_intersection_indices(oof_results)
        else:  # union
            common_indices = self._find_union_indices(oof_results)

        n_common = len(common_indices)

        if n_common == 0:
            raise ValueError(
                "No common indices found. Check that OOF results have overlapping indices."
            )

        # Allocate aligned arrays
        aligned_probs = np.full((n_common, n_models * self.n_classes), np.nan, dtype=np.float32)
        aligned_preds = np.full(
            (n_common, n_models), AlignedOOFResult.MISSING_PREDICTION, dtype=np.int64
        )
        coverage: dict[str, float] = {}

        # Create index mapping for fast lookup
        idx_map = {idx: i for i, idx in enumerate(common_indices)}

        # Align each model's predictions
        for m, oof in enumerate(oof_results):
            model_coverage = 0

            for i, idx in enumerate(oof.indices):
                if idx in idx_map:
                    j = idx_map[idx]
                    start_col = m * self.n_classes
                    end_col = start_col + self.n_classes
                    aligned_probs[j, start_col:end_col] = oof.probabilities[i]
                    aligned_preds[j, m] = oof.predictions[i]
                    model_coverage += 1

            coverage[oof.model_name] = model_coverage / n_common if n_common > 0 else 0.0

        return AlignedOOFResult(
            probabilities=aligned_probs,
            predictions=aligned_preds,
            common_indices=common_indices,
            model_names=model_names,
            coverage=coverage,
            n_common=n_common,
            n_models=n_models,
            n_classes=self.n_classes,
        )

    def _find_intersection_indices(self, oof_results: list[OOFResult]) -> np.ndarray:
        """
        Find indices common to all models.

        Args:
            oof_results: List of OOFResult.

        Returns:
            Sorted array of common indices.
        """
        if not oof_results:
            return np.array([], dtype=np.int64)

        # Start with first model's indices
        common = set(oof_results[0].indices)

        # Intersect with each subsequent model
        for oof in oof_results[1:]:
            common &= set(oof.indices)

        return np.array(sorted(common), dtype=np.int64)

    def _find_union_indices(self, oof_results: list[OOFResult]) -> np.ndarray:
        """
        Find indices present in any model.

        Args:
            oof_results: List of OOFResult.

        Returns:
            Sorted array of all unique indices.
        """
        if not oof_results:
            return np.array([], dtype=np.int64)

        all_indices: set = set()
        for oof in oof_results:
            all_indices |= set(oof.indices)

        return np.array(sorted(all_indices), dtype=np.int64)

    def compute_offset(
        self,
        model_name: str,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    ) -> int:
        """
        Compute index offset for a model based on its data rank.

        For sequence and multi-stream models, the first (seq_len - 1)
        samples cannot be used because they lack sufficient history.

        Args:
            model_name: Model name (case-insensitive).
            sequence_length: Sequence length used (default: 60).

        Returns:
            Index offset (0 for tabular, seq_len-1 for sequence/multi-stream).

        Examples:
            >>> aligner = OOFAligner()
            >>> aligner.compute_offset("xgboost")
            0
            >>> aligner.compute_offset("lstm", sequence_length=60)
            59
            >>> aligner.compute_offset("patchtst", sequence_length=60)
            59
        """
        rank = MODEL_DATA_RANKS.get(model_name.lower(), 2)

        if rank == 2:
            return 0
        else:
            return sequence_length - 1

    def estimate_common_samples(
        self,
        total_samples: int,
        model_names: list[str],
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    ) -> int:
        """
        Estimate the number of common samples after alignment.

        Useful for planning before training to understand sample loss.

        Args:
            total_samples: Total samples in the original dataset.
            model_names: List of model names to be aligned.
            sequence_length: Sequence length for sequence/multi-stream models.

        Returns:
            Estimated number of common samples.

        Example:
            >>> aligner = OOFAligner()
            >>> aligner.estimate_common_samples(10000, ["xgboost", "lstm"], 60)
            9941
        """
        max_offset = max(self.compute_offset(name, sequence_length) for name in model_names)
        return max(0, total_samples - max_offset)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def align_oof_predictions(
    oof_results: list[OOFResult],
    target_indices: np.ndarray | None = None,
    strategy: str = "intersection",
) -> AlignedOOFResult:
    """
    Convenience function to align OOF predictions.

    Args:
        oof_results: List of OOFResult from each model.
        target_indices: Optional target indices to align to.
        strategy: Alignment strategy ("intersection" or "union").

    Returns:
        AlignedOOFResult with aligned predictions.

    Example:
        >>> aligned = align_oof_predictions([oof_xgb, oof_lstm, oof_patch])
        >>> X_stack = aligned.stacking_features
    """
    aligner = OOFAligner()
    return aligner.align(oof_results, target_indices, strategy)


def compute_coverage_stats(
    oof_results: list[OOFResult],
) -> dict[str, dict[str, int | float]]:
    """
    Compute coverage statistics for OOF results.

    Args:
        oof_results: List of OOFResult from each model.

    Returns:
        Dictionary with per-model coverage statistics.

    Example:
        >>> stats = compute_coverage_stats([oof_xgb, oof_lstm])
        >>> print(stats)
        {
            'xgboost': {'n_samples': 10000, 'coverage': 1.0},
            'lstm': {'n_samples': 9941, 'coverage': 0.9941}
        }
    """
    if not oof_results:
        return {}

    # Find max samples (assumed to be from full-coverage model)
    max_samples = max(oof.n_samples for oof in oof_results)

    stats: dict[str, dict[str, int | float]] = {}
    for oof in oof_results:
        stats[oof.model_name] = {
            "n_samples": oof.n_samples,
            "coverage": oof.n_samples / max_samples if max_samples > 0 else 0.0,
            "original_coverage": oof.coverage,
        }

    return stats


def validate_oof_results(oof_results: list[OOFResult]) -> list[str]:
    """
    Validate OOF results for alignment compatibility.

    Args:
        oof_results: List of OOFResult to validate.

    Returns:
        List of warning messages (empty if all OK).
    """
    warnings: list[str] = []

    if not oof_results:
        warnings.append("Empty OOF results list")
        return warnings

    # Check for duplicate model names
    names = [r.model_name for r in oof_results]
    if len(names) != len(set(names)):
        duplicates = [n for n in names if names.count(n) > 1]
        warnings.append(f"Duplicate model names: {set(duplicates)}")

    # Check for consistent n_classes
    n_classes_set = {r.n_classes for r in oof_results}
    if len(n_classes_set) > 1:
        warnings.append(f"Inconsistent n_classes: {n_classes_set}")

    # Check for overlapping indices
    if len(oof_results) >= 2:
        common = set(oof_results[0].indices)
        for oof in oof_results[1:]:
            common &= set(oof.indices)

        if len(common) == 0:
            warnings.append("No overlapping indices between models")
        elif len(common) < min(r.n_samples for r in oof_results) * 0.5:
            warnings.append(
                f"Low overlap: only {len(common)} common indices "
                f"(less than 50% of smallest model)"
            )

    return warnings


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main classes
    "AlignedOOFResult",
    "OOFAligner",
    # Convenience functions
    "align_oof_predictions",
    "compute_coverage_stats",
    "validate_oof_results",
]
