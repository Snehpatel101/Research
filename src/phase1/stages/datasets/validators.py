"""
Model-Ready Validation for TimeSeriesDataContainer.

Validates Phase 1 outputs are ready for Phase 2 model training.

CLI: python -m src.stages.datasets.validators --path data/splits/scaled --horizon 20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.phase1.stages.datasets.container import TimeSeriesDataContainer

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Constants
VALID_LABELS = frozenset({-1, 0, 1, -99})
INVALID_LABEL = -99
FEATURE_CLIP_RANGE = (-5.0, 5.0)
MIN_CLASS_FRACTION = 0.10
WEIGHT_RANGE = (0.4, 1.6)
EXPECTED_BAR_INTERVAL = timedelta(minutes=5)
MIN_SEQUENCES_PER_SYMBOL = 100


@dataclass
class ValidationResult:
    """Result of model-ready validation with errors (blocking) and warnings."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return f"ValidationResult({'VALID' if self.is_valid else 'INVALID'}, errors={len(self.errors)}, warnings={len(self.warnings)})"


def _validate_features(container: TimeSeriesDataContainer, result: ValidationResult) -> None:
    """Check features: no NaN/Inf, expected range, no constant features."""
    for name, split in container.splits.items():
        if not split.feature_columns:
            result.add_error(f"{name}: No feature columns found")
            continue
        X = split.df[split.feature_columns]
        # NaN check
        nan_features = X.isna().sum()
        nan_features = nan_features[nan_features > 0]
        if len(nan_features) > 0:
            result.add_error(
                f"{name}: {len(nan_features)} features have NaN. Top: {dict(nan_features.nlargest(5))}"
            )
        # Inf check
        if np.isinf(X.values).any():
            result.add_error(f"{name}: Inf values found in features")
        # Range check
        vals = X.values[~np.isnan(X.values) & ~np.isinf(X.values)]
        if len(vals) > 0:
            vmin, vmax = vals.min(), vals.max()
            if vmin < FEATURE_CLIP_RANGE[0] - 0.01 or vmax > FEATURE_CLIP_RANGE[1] + 0.01:
                result.add_warning(
                    f"{name}: Features outside [{FEATURE_CLIP_RANGE[0]}, {FEATURE_CLIP_RANGE[1]}]: [{vmin:.3f}, {vmax:.3f}]"
                )
        # Constant features
        constant = X.std()[X.std() == 0].index.tolist()
        if constant:
            result.add_warning(f"{name}: {len(constant)} constant features: {constant[:5]}")


def _validate_labels(container: TimeSeriesDataContainer, result: ValidationResult) -> None:
    """Check labels: valid values, class balance, invalid count."""
    for name, split in container.splits.items():
        if split.label_column not in split.df.columns:
            result.add_error(f"{name}: Label column '{split.label_column}' not found")
            continue
        labels = split.df[split.label_column]
        # Unexpected values
        unexpected = set(labels.unique()) - VALID_LABELS
        if unexpected:
            result.add_error(f"{name}: Unexpected label values: {unexpected}")
        # Invalid count
        invalid_count = (labels == INVALID_LABEL).sum()
        if invalid_count > 0:
            result.add_warning(
                f"{name}: {invalid_count} invalid labels ({100*invalid_count/len(labels):.1f}%) present"
            )
        # Class balance
        valid = labels[labels != INVALID_LABEL]
        if len(valid) == 0:
            result.add_error(f"{name}: No valid labels")
            continue
        counts = valid.value_counts()
        balance = {}
        for cls in [-1, 0, 1]:
            cnt = counts.get(cls, 0)
            frac = cnt / len(valid)
            balance[cls] = {"count": int(cnt), "fraction": float(frac)}
            if 0 < frac < MIN_CLASS_FRACTION:
                result.add_warning(
                    f"{name}: Class {cls} has only {frac:.1%} ({cnt}/{len(valid)}). Consider class weighting."
                )
        result.metadata.setdefault("label_distribution", {})[name] = balance


def _validate_sequences(
    container: TimeSeriesDataContainer, result: ValidationResult, seq_len: int
) -> None:
    """Check datetime continuity and sufficient sequences per symbol."""
    for name, split in container.splits.items():
        if split.datetime_column not in split.df.columns:
            result.add_warning(f"{name}: Datetime column '{split.datetime_column}' not found")
            continue
        if split.symbol_column in split.df.columns:
            for sym in split.df[split.symbol_column].unique():
                sym_df = split.df[split.df[split.symbol_column] == sym]
                dt = pd.to_datetime(sym_df[split.datetime_column])
                if len(dt) > 1:
                    gaps = (dt.diff().dropna() > pd.Timedelta(EXPECTED_BAR_INTERVAL) * 1.5).sum()
                    if gaps > 0:
                        result.add_warning(f"{name}/{sym}: {gaps} datetime gaps detected")
                n_seq = max(0, len(sym_df) - seq_len + 1)
                if n_seq < MIN_SEQUENCES_PER_SYMBOL:
                    result.add_warning(
                        f"{name}/{sym}: Only {n_seq} sequences (< {MIN_SEQUENCES_PER_SYMBOL})"
                    )
                result.metadata.setdefault("sequences_per_symbol", {}).setdefault(name, {})[
                    sym
                ] = n_seq


def _validate_integration(container: TimeSeriesDataContainer, result: ValidationResult) -> None:
    """Check cross-split consistency: feature columns match, weights in range."""
    splits = list(container.splits.items())

    # Weight range check (always run, even for single split)
    for name, split in splits:
        if split.weight_column in split.df.columns:
            w = split.df[split.weight_column]
            if w.min() < WEIGHT_RANGE[0] - 0.01:
                result.add_warning(f"{name}: Weight min ({w.min():.3f}) below {WEIGHT_RANGE[0]}")
            if w.max() > WEIGHT_RANGE[1] + 0.01:
                result.add_warning(f"{name}: Weight max ({w.max():.3f}) above {WEIGHT_RANGE[1]}")

    # Scaler metadata check (always run)
    if container.metadata:
        fit_split = container.metadata.get("scaling", {}).get("fitted_on", "")
        if fit_split and fit_split != "train":
            result.add_error(
                f"Scaler fitted on '{fit_split}', expected 'train' (potential leakage)"
            )

    # Cross-split checks need at least 2 splits
    if len(splits) < 2:
        return

    ref_name, ref_split = splits[0]
    ref_features = set(ref_split.feature_columns)
    for name, split in splits[1:]:
        current = set(split.feature_columns)
        missing, extra = ref_features - current, current - ref_features
        if missing:
            result.add_error(
                f"{name}: Missing {len(missing)} features from {ref_name}: {list(missing)[:5]}"
            )
        if extra:
            result.add_error(
                f"{name}: Extra {len(extra)} features not in {ref_name}: {list(extra)[:5]}"
            )


def validate_model_ready(container: TimeSeriesDataContainer, seq_len: int = 60) -> ValidationResult:
    """
    Validate Phase 1 outputs are ready for model training.

    Args:
        container: TimeSeriesDataContainer from Phase 1
        seq_len: Expected sequence length for validation

    Returns:
        ValidationResult with is_valid, errors, warnings, metadata
    """
    result = ValidationResult()
    result.metadata.update(
        {
            "validated_at": datetime.now().isoformat(),
            "horizon": container.horizon,
            "n_features": container.n_features,
            "splits": container.available_splits,
        }
    )
    _validate_features(container, result)
    _validate_labels(container, result)
    _validate_sequences(container, result, seq_len)
    _validate_integration(container, result)
    for name, split in container.splits.items():
        result.metadata.setdefault("split_sizes", {})[name] = split.n_samples
    logger.info(f"Validation complete: {result}")
    return result


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Validate Phase 1 outputs for model training")
    parser.add_argument("--path", default="data/splits/scaled", help="Scaled splits directory")
    parser.add_argument("--horizon", type=int, default=20, help="Label horizon")
    parser.add_argument("--seq-len", type=int, default=60, help="Sequence length")
    parser.add_argument("--output", help="JSON report output path")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s"
    )
    from src.phase1.stages.datasets.container import TimeSeriesDataContainer

    path = Path(args.path)
    if not path.is_dir():
        logger.error(f"Directory not found: {path}")
        return 1

    logger.info(f"Loading container from {path} (horizon={args.horizon})")
    try:
        container = TimeSeriesDataContainer.from_parquet_dir(
            path, args.horizon, exclude_invalid_labels=False
        )
    except Exception as e:
        logger.error(f"Failed to load: {e}")
        return 1

    result = validate_model_ready(container, args.seq_len)

    print("\n" + "=" * 70)
    print(f"MODEL-READY VALIDATION: {'VALID' if result.is_valid else 'INVALID'}")
    print(
        f"Horizon: {args.horizon} | Splits: {container.available_splits} | Features: {container.n_features}"
    )
    print("=" * 70)
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for e in result.errors:
            print(f"  [ERROR] {e}")
    if result.warnings:
        print(f"\nWARNINGS ({len(result.warnings)}):")
        for w in result.warnings:
            print(f"  [WARN] {w}")
    if not result.errors and not result.warnings:
        print("\nAll checks passed!")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nReport: {args.output}")

    return 0 if result.is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
