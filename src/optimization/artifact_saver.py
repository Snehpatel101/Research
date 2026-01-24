"""
FeatureSpec Artifact Saver - Persist optimization results for reproducibility.

Phase 3 of the SNwH (Unified Multi-Timeframe Model Factory) implementation.

This module provides functions to save FeatureSpec artifacts after Optuna
optimization, enabling:
    - Full reproducibility of best configurations
    - Experiment tracking with unique run IDs
    - Easy loading for inference pipelines

Directory Structure:
    experiments/
    └── {run_id}/
        └── feature_specs/
            ├── manifest.json                    # Index of all saved specs
            ├── best_{model_name}.json           # Best spec for quick access
            ├── {model_name}_trial_0.json        # Individual trial specs
            ├── {model_name}_trial_5.json
            └── ...

Usage:
    from src.optimization.artifact_saver import (
        save_feature_specs,
        load_best_feature_spec,
        load_feature_spec,
        list_saved_specs,
    )

    # After optimization
    study, best_result = run_5d_optimization(...)
    saved_paths = save_feature_specs(study, run_id="exp_001")

    # During inference
    spec = load_best_feature_spec("exp_001", "xgboost")
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import optuna

from src.core.contracts import FeatureSpec

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_OUTPUT_DIR = "experiments"
FEATURE_SPECS_SUBDIR = "feature_specs"
MANIFEST_FILENAME = "manifest.json"


# =============================================================================
# SAVING FUNCTIONS
# =============================================================================


def save_feature_specs(
    study: optuna.Study,
    run_id: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    save_all_trials: bool = False,
    top_k: int = 5,
) -> dict[str | int, Path]:
    """
    Save FeatureSpec artifacts from a completed Optuna study.

    Saves the best spec and optionally top-K or all trial specs to disk
    in a standardized directory structure for reproducibility.

    Args:
        study: Completed Optuna study with feature_spec user attributes
        run_id: Experiment run identifier (e.g., "exp_001", "2024-01-15_boosting")
        output_dir: Base output directory (default: "experiments")
        save_all_trials: If True, save all trials; otherwise save top_k
        top_k: Number of top trials to save by value (default: 5)

    Returns:
        Dict mapping trial_number (or "best") -> saved Path

    Example:
        >>> study, _ = run_5d_optimization(...)
        >>> paths = save_feature_specs(study, run_id="exp_001", top_k=10)
        >>> print(paths["best"])
        experiments/exp_001/feature_specs/best_xgboost.json
    """
    specs_dir = Path(output_dir) / run_id / FEATURE_SPECS_SUBDIR
    specs_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str | int, Path] = {}

    # Get completed trials sorted by value (best first, handling None)
    completed_trials = [t for t in study.trials if t.value is not None]
    sorted_trials = sorted(completed_trials, key=lambda t: t.value or float("-inf"), reverse=True)

    # Determine which trials to save
    trials_to_save = sorted_trials if save_all_trials else sorted_trials[:top_k]

    model_name = ""  # Track for best spec naming

    for trial in trials_to_save:
        spec_dict = trial.user_attrs.get("feature_spec")
        if not spec_dict:
            logger.debug(f"Trial {trial.number} has no feature_spec, skipping")
            continue

        spec = FeatureSpec.from_dict(spec_dict)
        model_name = spec.model_name or "unknown"

        # Save individual trial spec
        filename = f"{model_name}_trial_{trial.number}.json"
        path = specs_dir / filename
        spec.save(str(path))
        saved_paths[trial.number] = path
        logger.debug(f"Saved trial {trial.number} spec to {path}")

    # Save best spec with special naming for easy access
    if study.best_trial and "feature_spec" in study.best_trial.user_attrs:
        best_spec_dict = study.best_trial.user_attrs["feature_spec"]
        best_spec = FeatureSpec.from_dict(best_spec_dict)
        best_model_name = best_spec.model_name or model_name or "unknown"

        best_path = specs_dir / f"best_{best_model_name}.json"
        best_spec.save(str(best_path))
        saved_paths["best"] = best_path
        logger.info(f"Saved best spec to {best_path}")

    # Create and save manifest
    manifest = _create_manifest(study, saved_paths, run_id)
    manifest_path = specs_dir / MANIFEST_FILENAME
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info(f"Saved manifest to {manifest_path}")

    return saved_paths


def _create_manifest(
    study: optuna.Study,
    saved_paths: dict[str | int, Path],
    run_id: str,
) -> dict:
    """
    Create a manifest documenting saved specs.

    Args:
        study: Optuna study
        saved_paths: Mapping of trial numbers to saved paths
        run_id: Experiment run identifier

    Returns:
        Manifest dictionary
    """
    # Get best trial info
    best_trial_number = None
    best_value = None
    best_model_name = None

    if study.best_trial:
        best_trial_number = study.best_trial.number
        best_value = study.best_value
        best_spec_dict = study.best_trial.user_attrs.get("feature_spec", {})
        best_model_name = best_spec_dict.get("model_name", "unknown")

    # Build manifest
    manifest = {
        "run_id": run_id,
        "study_name": study.study_name,
        "created_at": datetime.now().isoformat(),
        "n_total_trials": len(study.trials),
        "n_completed_trials": len([t for t in study.trials if t.value is not None]),
        "n_saved_specs": len([k for k in saved_paths if k != "best"]),
        "best_trial": best_trial_number,
        "best_value": best_value,
        "best_model": best_model_name,
        "direction": study.direction.name if hasattr(study, "direction") else "MAXIMIZE",
        "specs": {str(k): str(v) for k, v in saved_paths.items()},
    }

    # Add summary of barrier parameters from best trial
    if study.best_trial:
        manifest["best_barrier_params"] = {
            "upper_mult": study.best_trial.params.get("upper_mult"),
            "lower_mult": study.best_trial.params.get("lower_mult"),
            "max_holding_bars": study.best_trial.params.get("max_holding_bars"),
        }

    return manifest


# =============================================================================
# LOADING FUNCTIONS
# =============================================================================


def load_best_feature_spec(
    run_id: str,
    model_name: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> FeatureSpec:
    """
    Load the best FeatureSpec for a model from a run.

    This is the primary function for loading specs during inference.

    Args:
        run_id: Experiment run identifier
        model_name: Model name (e.g., "xgboost", "lightgbm")
        output_dir: Base output directory (default: "experiments")

    Returns:
        FeatureSpec from the best trial

    Raises:
        FileNotFoundError: If the spec file doesn't exist

    Example:
        >>> spec = load_best_feature_spec("exp_001", "xgboost")
        >>> print(spec.selected_features)
    """
    path = Path(output_dir) / run_id / FEATURE_SPECS_SUBDIR / f"best_{model_name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Best spec for {model_name} not found at {path}. "
            f"Available specs: {list_saved_specs(run_id, output_dir)}"
        )
    return FeatureSpec.load(str(path))


def load_feature_spec(
    run_id: str,
    model_name: str,
    trial_number: int,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> FeatureSpec:
    """
    Load a specific trial's FeatureSpec.

    Args:
        run_id: Experiment run identifier
        model_name: Model name (e.g., "xgboost")
        trial_number: Optuna trial number
        output_dir: Base output directory (default: "experiments")

    Returns:
        FeatureSpec from the specified trial

    Raises:
        FileNotFoundError: If the spec file doesn't exist

    Example:
        >>> spec = load_feature_spec("exp_001", "xgboost", trial_number=42)
    """
    path = (
        Path(output_dir) / run_id / FEATURE_SPECS_SUBDIR / f"{model_name}_trial_{trial_number}.json"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Spec for trial {trial_number} not found at {path}. "
            f"Available specs: {list_saved_specs(run_id, output_dir)}"
        )
    return FeatureSpec.load(str(path))


def load_manifest(
    run_id: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict:
    """
    Load the manifest for a run.

    Args:
        run_id: Experiment run identifier
        output_dir: Base output directory

    Returns:
        Manifest dictionary

    Raises:
        FileNotFoundError: If manifest doesn't exist
    """
    path = Path(output_dir) / run_id / FEATURE_SPECS_SUBDIR / MANIFEST_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found at {path}")

    with open(path) as f:
        return json.load(f)


def list_saved_specs(
    run_id: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> list[str]:
    """
    List all saved spec files for a run.

    Args:
        run_id: Experiment run identifier
        output_dir: Base output directory

    Returns:
        List of spec filenames (without path)
    """
    specs_dir = Path(output_dir) / run_id / FEATURE_SPECS_SUBDIR
    if not specs_dir.exists():
        return []

    return sorted([f.name for f in specs_dir.glob("*.json") if f.name != MANIFEST_FILENAME])


def list_runs(output_dir: str = DEFAULT_OUTPUT_DIR) -> list[str]:
    """
    List all run IDs in the experiments directory.

    Args:
        output_dir: Base output directory

    Returns:
        List of run IDs that have feature_specs saved
    """
    base_dir = Path(output_dir)
    if not base_dir.exists():
        return []

    runs = []
    for item in base_dir.iterdir():
        if item.is_dir() and (item / FEATURE_SPECS_SUBDIR / MANIFEST_FILENAME).exists():
            runs.append(item.name)

    return sorted(runs)


# =============================================================================
# INTEGRATION WITH run_5d_optimization
# =============================================================================


def save_optimization_results(
    study: optuna.Study,
    run_id: str | None = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    save_all_trials: bool = False,
    top_k: int = 5,
    auto_generate_run_id: bool = True,
) -> tuple[str, dict[str | int, Path]]:
    """
    Save optimization results with automatic run ID generation.

    Convenience wrapper around save_feature_specs that generates a run_id
    if not provided.

    Args:
        study: Completed Optuna study
        run_id: Optional run identifier (generated if not provided)
        output_dir: Base output directory
        save_all_trials: If True, save all trials
        top_k: Number of top trials to save
        auto_generate_run_id: If True and run_id is None, generate from timestamp

    Returns:
        Tuple of (run_id, saved_paths)

    Example:
        >>> study, _ = run_5d_optimization(...)
        >>> run_id, paths = save_optimization_results(study)
        >>> print(f"Saved to {run_id}")
    """
    if run_id is None and auto_generate_run_id:
        # Generate run_id from study name and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_prefix = study.study_name.replace(" ", "_")[:20] if study.study_name else "opt"
        run_id = f"{study_prefix}_{timestamp}"

    if run_id is None:
        raise ValueError("run_id must be provided or auto_generate_run_id must be True")

    saved_paths = save_feature_specs(
        study=study,
        run_id=run_id,
        output_dir=output_dir,
        save_all_trials=save_all_trials,
        top_k=top_k,
    )

    return run_id, saved_paths


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Saving
    "save_feature_specs",
    "save_optimization_results",
    # Loading
    "load_best_feature_spec",
    "load_feature_spec",
    "load_manifest",
    # Discovery
    "list_saved_specs",
    "list_runs",
    # Constants
    "DEFAULT_OUTPUT_DIR",
    "FEATURE_SPECS_SUBDIR",
    "MANIFEST_FILENAME",
]
