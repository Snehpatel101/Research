"""
Unified Configuration Validator - Central validation for all pipeline configs.

Provides a single entry point for validating Phase 1-4 configurations,
with clear error messages and actionable suggestions.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATION RESULTS
# =============================================================================


class ValidationResult:
    """Result of a configuration validation check."""

    def __init__(
        self,
        is_valid: bool,
        errors: list[str] | None = None,
        warnings: list[str] | None = None,
        suggestions: list[str] | None = None,
    ):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.suggestions = suggestions or []

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def add_suggestion(self, message: str) -> None:
        """Add a suggestion message."""
        self.suggestions.append(message)

    def merge(self, other: ValidationResult) -> None:
        """Merge another validation result into this one."""
        if not other.is_valid:
            self.is_valid = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.suggestions.extend(other.suggestions)

    def __str__(self) -> str:
        """Generate human-readable validation report."""
        lines = []

        if self.is_valid:
            lines.append("✅ VALIDATION PASSED")
        else:
            lines.append("❌ VALIDATION FAILED")

        if self.errors:
            lines.append("\nERRORS:")
            for error in self.errors:
                lines.append(f"  ❌ {error}")

        if self.warnings:
            lines.append("\nWARNINGS:")
            for warning in self.warnings:
                lines.append(f"  ⚠️  {warning}")

        if self.suggestions:
            lines.append("\nSUGGESTIONS:")
            for suggestion in self.suggestions:
                lines.append(f"  💡 {suggestion}")

        return "\n".join(lines)


# =============================================================================
# PHASE 1: DATA PIPELINE VALIDATION
# =============================================================================


def validate_pipeline_config(config: Any) -> ValidationResult:
    """
    Validate Phase 1 pipeline configuration.

    Args:
        config: PipelineConfig instance or dict

    Returns:
        ValidationResult with errors/warnings/suggestions

    Example:
        >>> from src.phase1.pipeline_config import PipelineConfig
        >>> config = PipelineConfig()
        >>> result = validate_pipeline_config(config)
        >>> if not result.is_valid:
        ...     print(result)
    """
    result = ValidationResult(is_valid=True)

    # Convert to dict if needed
    if hasattr(config, "__dict__"):
        config_dict = {
            k: v for k, v in config.__dict__.items() if not k.startswith("_")
        }
    else:
        config_dict = config

    # Check project root
    project_root = config_dict.get("project_root")
    if project_root:
        root_str = str(project_root)
        if root_str.endswith("src") or root_str.endswith("src/"):
            result.add_error(
                f"Project root incorrectly set to '{project_root}'. "
                "Should be repository root, not src/"
            )
            result.add_suggestion("Use default project root (omit from config)")

    # Check symbol configuration
    symbols = config_dict.get("symbols", [])
    if not symbols:
        result.add_error("No symbols specified")
        result.add_suggestion("Set symbols = ['MES'] or ['MGC'] in config")
    elif len(symbols) > 1:
        allow_batch = config_dict.get("allow_batch_symbols", False)
        if not allow_batch:
            result.add_error(
                f"Multiple symbols specified but allow_batch_symbols=False: {symbols}"
            )
            result.add_suggestion(
                "Single-contract architecture: process one symbol per run"
            )

    # Check horizons
    from src.common.horizon_config import ACTIVE_HORIZONS
    horizons = config_dict.get("horizons") or config_dict.get("label_horizons")
    if not horizons:
        result.add_warning(f"No label horizons specified (will use default {ACTIVE_HORIZONS})")
    elif not all(isinstance(h, int) and h > 0 for h in horizons):
        result.add_error(f"Invalid horizons: {horizons}. Must be positive integers")

    # Check split ratios
    train_ratio = config_dict.get("train_ratio", 0.7)
    val_ratio = config_dict.get("val_ratio", 0.15)
    test_ratio = config_dict.get("test_ratio", 0.15)

    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        result.add_error(
            f"Split ratios don't sum to 1.0: "
            f"train={train_ratio}, val={val_ratio}, test={test_ratio} "
            f"(sum={total_ratio})"
        )

    if train_ratio < 0.5:
        result.add_warning(
            f"Train ratio is small ({train_ratio}). Recommended: >= 0.6"
        )

    # Check purge/embargo
    purge_bars = config_dict.get("purge_bars")
    embargo_bars = config_dict.get("embargo_bars")

    if purge_bars is not None and purge_bars < 0:
        result.add_error(f"Invalid purge_bars: {purge_bars}. Must be >= 0")

    if embargo_bars is not None and embargo_bars < 0:
        result.add_error(f"Invalid embargo_bars: {embargo_bars}. Must be >= 0")

    if purge_bars is not None and embargo_bars is not None:
        if purge_bars == 0 and embargo_bars == 0:
            result.add_warning(
                "Both purge and embargo are 0. High risk of data leakage!"
            )
            result.add_suggestion(
                "Use auto-scaling: purge = max_horizon * 3, embargo = 1440"
            )

    # Check paths
    data_dir = config_dict.get("data_dir")
    if data_dir:
        data_path = Path(data_dir)
        raw_data = data_path / "raw"
        if not raw_data.exists():
            result.add_warning(f"Raw data directory does not exist: {raw_data}")
            result.add_suggestion("Create data/raw/ and add OHLCV data files")

    return result


# =============================================================================
# PHASE 2: TRAINER VALIDATION
# =============================================================================


def validate_trainer_config(config: dict[str, Any]) -> ValidationResult:
    """
    Validate Phase 2 trainer configuration.

    Args:
        config: Trainer configuration dict with keys:
            - model_name: str
            - horizon: int
            - config: dict (model-specific params)
            - output_dir: Path

    Returns:
        ValidationResult with errors/warnings/suggestions

    Example:
        >>> config = {
        ...     "model_name": "xgboost",
        ...     "horizon": 20,
        ...     "config": {"n_estimators": 100},
        ... }
        >>> result = validate_trainer_config(config)
        >>> assert result.is_valid
    """
    result = ValidationResult(is_valid=True)

    # Check model name
    model_name = config.get("model_name")
    if not model_name:
        result.add_error("No model_name specified")
        result.add_suggestion("Use --model xgboost or similar")
    else:
        # Check if model is registered
        try:
            from src.models import ModelRegistry

            if not ModelRegistry.is_registered(model_name):
                available = ModelRegistry.list_all()
                result.add_error(
                    f"Model '{model_name}' not registered. Available: {available}"
                )
        except ImportError:
            result.add_warning("Could not import ModelRegistry for validation")

    # Check horizon
    horizon = config.get("horizon")
    if not horizon:
        result.add_error("No horizon specified")
        result.add_suggestion("Use --horizon 20 or similar")
    elif not isinstance(horizon, int) or horizon <= 0:
        result.add_error(f"Invalid horizon: {horizon}. Must be positive integer")
    else:
        # Check horizon is in standard set
        from src.common.horizon_config import LABEL_HORIZONS

        if horizon not in LABEL_HORIZONS:
            result.add_warning(
                f"Horizon {horizon} not in standard set {LABEL_HORIZONS}. "
                "Ensure Phase 1 generated labels for this horizon"
            )

    # Check sequence length for neural models
    if model_name in ["lstm", "gru", "tcn", "transformer"]:
        seq_len = config.get("config", {}).get("seq_len")
        if not seq_len:
            result.add_error(f"Neural model '{model_name}' requires seq_len parameter")
            result.add_suggestion("Use --seq-len 30 or similar")
        elif seq_len < 10:
            result.add_warning(f"Sequence length {seq_len} is very short. Consider >= 20")

    # Check output directory
    output_dir = config.get("output_dir")
    if output_dir:
        output_path = Path(output_dir)
        if not output_path.exists():
            result.add_warning(f"Output directory does not exist: {output_dir}")
            result.add_suggestion("Directory will be created automatically")

    return result


# =============================================================================
# PHASE 3: CROSS-VALIDATION VALIDATION
# =============================================================================


def validate_cv_config(config: dict[str, Any]) -> ValidationResult:
    """
    Validate Phase 3 cross-validation configuration.

    Args:
        config: CV configuration dict with keys:
            - models: List[str]
            - horizons: List[int]
            - n_splits: int
            - purge_bars: int
            - embargo_bars: int

    Returns:
        ValidationResult with errors/warnings/suggestions

    Example:
        >>> config = {
        ...     "models": ["xgboost", "lightgbm"],
        ...     "horizons": [20],
        ...     "n_splits": 5,
        ... }
        >>> result = validate_cv_config(config)
        >>> assert result.is_valid
    """
    result = ValidationResult(is_valid=True)

    # Check models
    models = config.get("models", [])
    if not models:
        result.add_error("No models specified for cross-validation")
        result.add_suggestion("Use --models xgboost,lightgbm")
    else:
        # Validate each model
        try:
            from src.models import ModelRegistry

            for model in models:
                if not ModelRegistry.is_registered(model):
                    available = ModelRegistry.list_all()
                    result.add_error(
                        f"Model '{model}' not registered. Available: {available}"
                    )
        except ImportError:
            result.add_warning("Could not import ModelRegistry for validation")

    # Check horizons
    horizons = config.get("horizons", [])
    if not horizons:
        result.add_error("No horizons specified for cross-validation")
        result.add_suggestion("Use --horizons 5,10,15,20")
    elif not all(isinstance(h, int) and h > 0 for h in horizons):
        result.add_error(f"Invalid horizons: {horizons}. Must be positive integers")

    # Check n_splits
    n_splits = config.get("n_splits", 5)
    if n_splits < 2:
        result.add_error(f"Invalid n_splits: {n_splits}. Must be >= 2")
    elif n_splits < 3:
        result.add_warning(f"n_splits={n_splits} is small. Recommended: >= 5")

    # Check purge/embargo
    purge_bars = config.get("purge_bars")
    embargo_bars = config.get("embargo_bars")

    if purge_bars == 0 and embargo_bars == 0:
        result.add_warning(
            "Both purge and embargo are 0. High risk of data leakage in CV!"
        )
        result.add_suggestion(
            "Use auto-scaling from src.common.horizon_config.auto_scale_purge_embargo"
        )

    # Check Optuna config
    if config.get("tune", False):
        n_trials = config.get("n_trials", 50)
        if n_trials < 10:
            result.add_warning(
                f"n_trials={n_trials} is small for tuning. Recommended: >= 50"
            )
        elif n_trials > 500:
            result.add_warning(
                f"n_trials={n_trials} is large. May take very long time"
            )

    return result


# =============================================================================
# PHASE 4: ENSEMBLE VALIDATION
# =============================================================================


def validate_ensemble_config(config: dict[str, Any]) -> ValidationResult:
    """
    Validate Phase 4 ensemble configuration.

    Args:
        config: Ensemble configuration dict with keys:
            - model_name: str (voting/stacking/blending)
            - base_model_names: List[str]
            - horizon: int
            - stacking_data: Optional[str] (CV run ID)

    Returns:
        ValidationResult with errors/warnings/suggestions

    Example:
        >>> config = {
        ...     "model_name": "stacking",
        ...     "base_model_names": ["xgboost", "lightgbm"],
        ...     "horizon": 20,
        ... }
        >>> result = validate_ensemble_config(config)
        >>> assert result.is_valid
    """
    result = ValidationResult(is_valid=True)

    # Check ensemble type
    model_name = config.get("model_name")
    if model_name not in ["voting", "stacking", "blending"]:
        result.add_error(
            f"Invalid ensemble type: {model_name}. "
            "Must be voting, stacking, or blending"
        )
        return result

    # Check base models
    base_models = config.get("base_model_names", [])
    if not base_models:
        result.add_error("No base_model_names specified for ensemble")
        result.add_suggestion("Use --base-models xgboost,lightgbm,catboost")
        return result

    if len(base_models) < 2:
        result.add_error(
            f"Need at least 2 base models for ensemble, got {len(base_models)}"
        )
        return result

    # Use ensemble validator from Phase 4
    try:
        from src.models.ensemble.validator import (
            validate_ensemble_config as ensemble_validator,
        )

        is_valid, error_msg = ensemble_validator(base_models)
        if not is_valid:
            result.add_error(f"Ensemble compatibility error:\n{error_msg}")
    except ImportError:
        result.add_warning("Could not import ensemble validator")

    # Check horizon
    horizon = config.get("horizon")
    if not horizon:
        result.add_error("No horizon specified for ensemble")
    elif not isinstance(horizon, int) or horizon <= 0:
        result.add_error(f"Invalid horizon: {horizon}. Must be positive integer")

    # Check Phase 3 stacking data (if provided)
    stacking_data = config.get("stacking_data")
    if stacking_data:
        phase3_base = Path(config.get("phase3_base_dir", "data/stacking"))
        stacking_dir = phase3_base / stacking_data / "stacking"

        if not stacking_dir.exists():
            result.add_error(
                f"Phase 3 stacking data not found: {stacking_dir}\n"
                "Run Phase 3 CV first to generate stacking data"
            )
            result.add_suggestion(
                f"python scripts/run_cv.py --models {','.join(base_models[:2])} "
                f"--horizons {horizon} --output-name {stacking_data}"
            )
        else:
            # Check stacking dataset exists for horizon
            if horizon:
                stacking_file = stacking_dir / f"stacking_dataset_h{horizon}.parquet"
                if not stacking_file.exists():
                    result.add_error(
                        f"Stacking dataset not found for horizon {horizon}: {stacking_file}"
                    )
                    result.add_suggestion(
                        f"Run CV with --horizons {horizon} to generate this dataset"
                    )

    # Ensemble-specific checks
    if model_name == "stacking":
        n_folds = config.get("config", {}).get("n_folds", 5)
        if not stacking_data and n_folds < 3:
            result.add_warning(
                f"n_folds={n_folds} is small for stacking. Recommended: >= 5"
            )

    return result


# =============================================================================
# UNIFIED VALIDATION
# =============================================================================


def run_all_validations(
    config_dict: dict[str, Any], phases: list[str] | None = None
) -> ValidationResult:
    """
    Run all applicable validations on a configuration.

    Args:
        config_dict: Configuration dictionary or object
        phases: List of phases to validate (default: detect from config)
                Options: ["phase1", "phase2", "phase3", "phase4"]

    Returns:
        Combined ValidationResult

    Example:
        >>> config = {
        ...     "pipeline": {...},  # Phase 1 config
        ...     "trainer": {...},   # Phase 2 config
        ...     "cv": {...},        # Phase 3 config
        ...     "ensemble": {...},  # Phase 4 config
        ... }
        >>> result = run_all_validations(config)
        >>> print(result)
    """
    result = ValidationResult(is_valid=True)

    # Auto-detect phases if not specified
    if phases is None:
        phases = []
        if "pipeline" in config_dict or "symbols" in config_dict:
            phases.append("phase1")
        if "trainer" in config_dict or "model_name" in config_dict:
            phases.append("phase2")
        if "cv" in config_dict or "n_splits" in config_dict:
            phases.append("phase3")
        if "ensemble" in config_dict or "base_model_names" in config_dict:
            phases.append("phase4")

    # Run phase-specific validations
    if "phase1" in phases:
        phase1_config = config_dict.get("pipeline", config_dict)
        phase1_result = validate_pipeline_config(phase1_config)
        result.merge(phase1_result)

    if "phase2" in phases:
        phase2_config = config_dict.get("trainer", config_dict)
        phase2_result = validate_trainer_config(phase2_config)
        result.merge(phase2_result)

    if "phase3" in phases:
        phase3_config = config_dict.get("cv", config_dict)
        phase3_result = validate_cv_config(phase3_config)
        result.merge(phase3_result)

    if "phase4" in phases:
        phase4_config = config_dict.get("ensemble", config_dict)
        phase4_result = validate_ensemble_config(phase4_config)
        result.merge(phase4_result)

    return result


def generate_validation_report(
    config_dict: dict[str, Any], phases: list[str] | None = None
) -> str:
    """
    Generate human-readable validation report.

    Args:
        config_dict: Configuration to validate
        phases: Phases to validate (default: auto-detect)

    Returns:
        Formatted validation report string

    Example:
        >>> config = {"model_name": "xgboost", "horizon": 20}
        >>> report = generate_validation_report(config, phases=["phase2"])
        >>> print(report)
    """
    result = run_all_validations(config_dict, phases)

    lines = [
        "=" * 70,
        "ML Pipeline Configuration Validation Report",
        "=" * 70,
        "",
    ]

    if phases:
        lines.append(f"Phases Validated: {', '.join(phases)}")
    else:
        lines.append("Phases Validated: auto-detected")

    lines.append("")
    lines.append(str(result))
    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


# =============================================================================
# QUICK VALIDATION HELPERS
# =============================================================================


def quick_validate(
    phase: str, **kwargs: Any
) -> tuple[bool, str | None]:
    """
    Quick validation helper for command-line scripts.

    Args:
        phase: Phase to validate ("phase1", "phase2", "phase3", "phase4")
        **kwargs: Configuration parameters

    Returns:
        Tuple of (is_valid, error_message)
        If valid: (True, None)
        If invalid: (False, error_message)

    Example:
        >>> is_valid, error = quick_validate(
        ...     "phase2",
        ...     model_name="xgboost",
        ...     horizon=20
        ... )
        >>> if not is_valid:
        ...     print(f"Error: {error}")
    """
    validators = {
        "phase1": validate_pipeline_config,
        "phase2": validate_trainer_config,
        "phase3": validate_cv_config,
        "phase4": validate_ensemble_config,
    }

    if phase not in validators:
        return False, f"Unknown phase: {phase}. Must be one of {list(validators.keys())}"

    try:
        result = validators[phase](kwargs)
        if result.is_valid:
            return True, None
        else:
            return False, str(result)
    except Exception as e:
        return False, f"Validation error: {e}"


__all__ = [
    "ValidationResult",
    "validate_pipeline_config",
    "validate_trainer_config",
    "validate_cv_config",
    "validate_ensemble_config",
    "run_all_validations",
    "generate_validation_report",
    "quick_validate",
]
