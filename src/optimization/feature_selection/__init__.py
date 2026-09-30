"""
Feature Selection Package for OHLCV Time-Series ML.

The live selection is the orchestrator's train-only pipeline
(``src/models/training/feature_selection.py``): MDA ranking, timeframe budget,
low-variance filter, rank-ordered decorrelation, per-model contract head. This
package holds its building blocks plus the per-fold selector of ``ml cv``.

Main Components:
    Selection:
        WalkForwardFeatureSelector: Per-fold MDA/MDI selection over purged CV splits
        FeatureSelectorConfig: Its configuration
        FeatureSelectionResult: Its result

    Filters and ranking:
        filter_low_variance: Remove near-constant features
        select_decorrelated_by_rank: Greedy rank-ordered decorrelation
        apply_timeframe_budget: Cap MTF features per timeframe

    Governance diagnostics (opt-in, read-only):
        BootstrapFeatureStability, LabelPerturbationTester, RobustnessScorer,
        FeatureRegistry, FeatureLifecycleState

Reference: Lopez de Prado (2018) "Advances in Financial Machine Learning"
"""

from .bootstrap_stability import (
    BootstrapFeatureStability,
    BootstrapStabilityResult,
    StabilitySummary,
)
from .config import FeatureSelectorConfig
from .filtering import filter_low_variance, select_decorrelated_by_rank
from .label_perturbation import (
    LabelPerturbationTester,
    PerturbationResult,
    PerturbationSummary,
)
from .lifecycle import FeatureLifecycleState
from .registry import FeatureRecord, FeatureRegistry, RunUpdate
from .result import FeatureSelectionResult
from .robustness_scoring import RobustnessScorer
from .timeframe_budget import apply_timeframe_budget
from .walk_forward import WalkForwardFeatureSelector

__all__ = [
    # Walk-forward selection
    "FeatureSelectionResult",
    "FeatureSelectorConfig",
    "WalkForwardFeatureSelector",
    # Filters
    "filter_low_variance",
    "select_decorrelated_by_rank",
    "apply_timeframe_budget",
    # Feature lifecycle / registry
    "FeatureLifecycleState",
    "FeatureRecord",
    "FeatureRegistry",
    "RunUpdate",
    # Bootstrap stability
    "BootstrapFeatureStability",
    "BootstrapStabilityResult",
    "StabilitySummary",
    # Label perturbation
    "LabelPerturbationTester",
    "PerturbationResult",
    "PerturbationSummary",
    # Robustness scoring
    "RobustnessScorer",
]
