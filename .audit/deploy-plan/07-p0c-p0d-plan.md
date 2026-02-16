# 07 - Detailed Implementation Plan: P0-C (Deploy Packaging) and P0-D (Notebook Integration)

**Date:** 2026-02-15
**Agent:** 7/10 (Integration Planner)
**Purpose:** Exact task specifications for P0-C and P0-D with file paths, API signatures, dependency order, backward compatibility, and acceptance criteria. Includes exact code for notebook cells and dataclass definitions.

---

## References

- Reports 01-06 in `.audit/deploy-plan/`
- `PHASE0-DEPLOYABLE-ARTIFACT-PLAN.md` and `HIGH-LEVEL-DEPLOYABLE-ARTIFACT-ARCHITECTURE.md`
- `src/factory.py` L233-331 (run phases), L673-704 (_create_bundle)
- `src/config/experiment.py` L141-150 (BundlingSection)
- `src/inference/__init__.py` (current exports)
- `notebooks/ml_factory_colab.ipynb` (8 cells: 0=markdown, 1-7=code)

---

## Hard Constraints (Reiterated)

1. **factory.py changes minimal** -- add a Phase 5 (deploy) after existing Phase 4 (bundling)
2. **Notebook cells must work for BOTH single-model and ensemble runs**
3. **Deploy manifest must be loadable without the full src/ package** (pure JSON)
4. **Backward compat**: existing factory runs without deploy config still work
5. **No code modification** in this planning document

---

# P0-C: Deploy Packaging

---

## P0-C-1: Add `DeployManifest` Dataclass

### Task ID

`P0-C-1`

### Files to Create

- `/home/jake/Desktop/Research/src/inference/deploy.py` (NEW, ~180 lines)

### API / Exact Definition

```python
"""
Deploy packaging for ML Factory.

Creates a deploy/ directory structure with per-horizon artifacts,
manifest.json, and validation reports after factory bundling completes.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HorizonArtifactEntry:
    """Per-horizon entry in the deploy manifest."""

    horizon: int
    artifact_type: str  # "model" or "ensemble"
    model_key: str  # e.g. "xgboost_h20" or "ensemble_h20"
    model_family: str  # e.g. "boosting", "neural", "ensemble"
    bundle_path: str  # Relative path from deploy/ to artifact dir
    feature_count: int = 0
    sequence_length: int = 0
    requires_sequences: bool = False
    requires_4d: bool = False
    scaling_source: str = "bundle"
    metrics: dict[str, float] = field(default_factory=dict)
    validation_passed: bool = False
    validation_path: str = ""  # Relative path to validation.json


@dataclass
class DeployManifest:
    """
    Run-level deployment manifest.

    This is a pure-JSON-serializable dataclass that can be loaded
    without importing any src/ modules. All paths are relative to
    the deploy/ directory.
    """

    run_id: str
    created_at: str
    horizons: list[int]
    selected_artifacts: dict[str, HorizonArtifactEntry]  # key: "h{horizon}"
    runtime_profile: str = "native"  # "native", "onnx", "torchscript"
    total_models_trained: int = 0
    best_model_overall: str = ""
    ensemble_available: bool = False
    compatibility: dict[str, str] = field(default_factory=lambda: {
        "min_python": "3.10",
        "bundle_version": "1.3.0",
    })

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for json.dump."""
        result = {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "horizons": self.horizons,
            "runtime_profile": self.runtime_profile,
            "total_models_trained": self.total_models_trained,
            "best_model_overall": self.best_model_overall,
            "ensemble_available": self.ensemble_available,
            "compatibility": self.compatibility,
            "selected_artifacts": {},
        }
        for key, entry in self.selected_artifacts.items():
            result["selected_artifacts"][key] = asdict(entry)
        return result

    def save(self, path: Path) -> None:
        """Write manifest.json to the given directory."""
        manifest_path = path / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Deploy manifest written: {manifest_path}")

    @classmethod
    def load(cls, path: Path) -> DeployManifest:
        """Load manifest.json from a deploy/ directory. Pure JSON, no src/ imports needed."""
        manifest_path = path / "manifest.json" if path.is_dir() else path
        with open(manifest_path) as f:
            data = json.load(f)

        artifacts = {}
        for key, entry_data in data.get("selected_artifacts", {}).items():
            artifacts[key] = HorizonArtifactEntry(**entry_data)

        return cls(
            run_id=data["run_id"],
            created_at=data["created_at"],
            horizons=data["horizons"],
            selected_artifacts=artifacts,
            runtime_profile=data.get("runtime_profile", "native"),
            total_models_trained=data.get("total_models_trained", 0),
            best_model_overall=data.get("best_model_overall", ""),
            ensemble_available=data.get("ensemble_available", False),
            compatibility=data.get("compatibility", {}),
        )
```

### Metadata/Schema Changes

The `manifest.json` is a new file written into `deploy/`. Example JSON:

```json
{
  "run_id": "exp_20260215_143022",
  "created_at": "2026-02-15T14:30:22",
  "horizons": [20],
  "runtime_profile": "native",
  "total_models_trained": 12,
  "best_model_overall": "xgboost_h20",
  "ensemble_available": true,
  "compatibility": {
    "min_python": "3.10",
    "bundle_version": "1.3.0"
  },
  "selected_artifacts": {
    "h20": {
      "horizon": 20,
      "artifact_type": "ensemble",
      "model_key": "ensemble_h20",
      "model_family": "ensemble",
      "bundle_path": "h20/artifact",
      "feature_count": 150,
      "sequence_length": 0,
      "requires_sequences": false,
      "requires_4d": false,
      "scaling_source": "bundle",
      "metrics": {"val_f1": 0.62, "val_accuracy": 0.58},
      "validation_passed": true,
      "validation_path": "h20/validation.json"
    }
  }
}
```

### Dependency Order

- **Blocks:** P0-C-2 (deploy directory creation), P0-C-5 (deploy helper)
- **Blocked by:** Nothing (standalone dataclass)

### Backward Compatibility

- Purely additive -- new file, no existing code changes
- `DeployManifest.load()` works with pure `json` module; no src/ imports needed
- Old factory runs without deploy config are unaffected

### Acceptance Criteria

1. `python -c "from src.inference.deploy import DeployManifest, HorizonArtifactEntry; print('OK')"` succeeds
2. Round-trip test: `DeployManifest.load(path)` after `manifest.save(path)` produces equivalent object
3. `manifest.json` is valid JSON loadable with `json.load()` and no src/ imports
4. `ruff check src/inference/deploy.py` passes
5. File includes `from __future__ import annotations`

### Effort Estimate

**M** -- 100-130 LOC

---

## P0-C-2: Deploy Directory Creation in factory.py (Phase 5)

### Task ID

`P0-C-2`

### Files to Modify

- `/home/jake/Desktop/Research/src/factory.py` (MODIFY)

### API / Insertion Point

**Insert after Line 296** (after `self._save_checkpoint_bundling(bundle_path)`) and **before Line 298** (before `# Build result`).

The `run()` method docstring at L237-241 changes from "4 phases" to "5 phases":

```python
        Phases:
        1. Data Pipeline: Prepare features and labels
        2. Training: Train models and ensemble
        3. Evaluation: Compute metrics (optional backtest)
        4. Bundling: Create deployment artifacts (optional)
        5. Deploy: Select best artifacts and create deploy/ structure (optional)
```

Phase labels in log messages change from `[Phase N/4]` to `[Phase N/5]`.

New code block inserted after Phase 4 bundling:

```python
            # Phase 5: Deploy packaging (optional)
            deploy_path = None
            if bundle_path is not None:
                self._log("\n[Phase 5/5] Deploy Packaging")
                deploy_path = self._create_deploy(
                    training_result=training_result,
                    bundle_path=bundle_path,
                )
```

New `ExperimentResult` field (add after `bundle_path` at L94):

```python
    deploy_path: Path | None = None
```

Pass `deploy_path` into ExperimentResult construction at L300-312:

```python
                deploy_path=deploy_path,
```

New private method `_create_deploy()` added after `_create_bundle()` (after L704):

```python
    def _create_deploy(
        self,
        training_result: Any,
        bundle_path: Path,
    ) -> Path | None:
        """
        Create deploy/ directory with selected best artifact per horizon.

        Phase 5: After bundling, selects the best artifact per horizon
        (ensemble if valid, else best model by val_f1), copies it to
        deploy/h{horizon}/artifact/, and writes deploy/manifest.json.

        Args:
            training_result: TrainingRunResult from Phase 2
            bundle_path: Path to bundles/ directory from Phase 4

        Returns:
            Path to deploy/ directory, or None if deploy creation fails
        """
        try:
            from src.inference.deploy import (
                DeployManifest,
                HorizonArtifactEntry,
                select_deploy_artifact,
                validate_deploy_artifact,
            )

            deploy_dir = self.output_dir / "deploy"
            deploy_dir.mkdir(exist_ok=True)

            # Determine horizons from training result
            horizons = []
            if hasattr(training_result, "config") and hasattr(training_result.config, "horizons"):
                horizons = list(training_result.config.horizons)
            if not horizons:
                # Infer from model result keys (e.g. "xgboost_h20" -> 20)
                for key in training_result.model_results:
                    parts = key.rsplit("_h", 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        h = int(parts[1])
                        if h not in horizons:
                            horizons.append(h)
                horizons.sort()

            selected_artifacts: dict[str, HorizonArtifactEntry] = {}

            for horizon in horizons:
                horizon_dir = deploy_dir / f"h{horizon}"
                horizon_dir.mkdir(exist_ok=True)
                artifact_dir = horizon_dir / "artifact"

                # Select best artifact for this horizon
                entry = select_deploy_artifact(
                    training_result=training_result,
                    bundle_path=bundle_path,
                    horizon=horizon,
                    artifact_dir=artifact_dir,
                )

                if entry is not None:
                    # Validate the deployed artifact
                    validation = validate_deploy_artifact(
                        artifact_dir=artifact_dir,
                        artifact_type=entry.artifact_type,
                    )
                    validation_path = horizon_dir / "validation.json"
                    with open(validation_path, "w") as f:
                        json.dump(validation, f, indent=2)

                    entry.validation_passed = validation.get("passed", False)
                    entry.validation_path = f"h{horizon}/validation.json"
                    selected_artifacts[f"h{horizon}"] = entry

                    self._log(
                        f"  h{horizon}: {entry.artifact_type} "
                        f"({entry.model_key}) -> {artifact_dir}"
                    )

            # Write run-level manifest
            manifest = DeployManifest(
                run_id=self.config.run_id,
                created_at=datetime.now().isoformat(),
                horizons=horizons,
                selected_artifacts=selected_artifacts,
                total_models_trained=training_result.n_models,
                best_model_overall=training_result.best_model or "",
                ensemble_available=training_result.ensemble_result is not None,
            )
            manifest.save(deploy_dir)

            self._log(f"  Deploy manifest: {deploy_dir / 'manifest.json'}")
            self._log(f"  {len(selected_artifacts)} artifact(s) deployed")

            return deploy_dir

        except Exception as e:
            logger.warning(f"Deploy packaging failed: {e}")
            self._log(f"  Deploy packaging failed: {e}")
            return None
```

Also add `import json` at the top if not already present (it is not currently imported in factory.py).

### Dependency Order

- **Blocks:** P0-D-1 through P0-D-5 (notebook cells reference deploy_path)
- **Blocked by:** P0-C-1 (DeployManifest), P0-C-3 (select_deploy_artifact), P0-C-4 (validate_deploy_artifact)

### Backward Compatibility

- Phase 5 only runs if `bundle_path is not None` (bundling must have succeeded)
- If deploy packaging fails, it returns None and factory run still succeeds
- `ExperimentResult.deploy_path` defaults to `None` -- existing code that doesn't use it is unaffected
- No changes to `ExperimentResult.summary()` required (deploy_path is informational)

### Acceptance Criteria

1. After a successful `factory.run()` with bundling enabled, `result.deploy_path` is a `Path` pointing to a `deploy/` directory
2. `deploy/manifest.json` exists and is valid JSON
3. `deploy/h{horizon}/artifact/` contains the selected bundle
4. `deploy/h{horizon}/validation.json` exists
5. Without bundling enabled (`create_bundle=False`), `deploy_path` is `None` and no deploy/ directory is created
6. `ruff check src/factory.py` passes

### Effort Estimate

**L** -- 80-100 LOC (new method + minor edits to run())

---

## P0-C-3: Artifact Selector Logic

### Task ID

`P0-C-3`

### Files to Modify

- `/home/jake/Desktop/Research/src/inference/deploy.py` (MODIFY -- add to file created in P0-C-1)

### API / Exact Definition

Add the following function to `src/inference/deploy.py`:

```python
def select_deploy_artifact(
    training_result: Any,
    bundle_path: Path,
    horizon: int,
    artifact_dir: Path,
    selection_metric: str = "val_f1",
    min_base_models_for_ensemble: int = 2,
) -> HorizonArtifactEntry | None:
    """
    Select the best deployable artifact for a given horizon.

    Selection policy:
    1. If ensemble exists, has >= min_base_models base models,
       and its metric >= best single model metric -> select ensemble
    2. Otherwise -> select best single model by selection_metric

    After selection, copies/links the bundle directory to artifact_dir.

    Args:
        training_result: TrainingRunResult from training phase.
            Must have .model_results dict and optionally .ensemble_result.
        bundle_path: Path to bundles/ directory containing saved ModelBundles
        horizon: Which horizon to select for (e.g. 20)
        artifact_dir: Destination directory (deploy/h{horizon}/artifact/)
        selection_metric: Metric to rank models by (default: val_f1)
        min_base_models_for_ensemble: Minimum base models needed to
            consider the ensemble (default: 2)

    Returns:
        HorizonArtifactEntry describing the selected artifact,
        or None if no suitable artifact found for this horizon.
    """
    import shutil

    # Collect single-model candidates for this horizon
    candidates: list[tuple[str, dict[str, float]]] = []
    for key, result in training_result.model_results.items():
        # Parse horizon from key like "xgboost_h20"
        parts = key.rsplit("_h", 1)
        if len(parts) == 2 and parts[1].isdigit() and int(parts[1]) == horizon:
            metrics = result.metrics if hasattr(result, "metrics") else {}
            candidates.append((key, metrics))

    if not candidates:
        logger.warning(f"No model results found for horizon {horizon}")
        return None

    # Sort by selection metric (descending)
    candidates.sort(
        key=lambda x: x[1].get(selection_metric, 0.0),
        reverse=True,
    )
    best_single_key, best_single_metrics = candidates[0]
    best_single_score = best_single_metrics.get(selection_metric, 0.0)

    # Check ensemble viability
    use_ensemble = False
    ensemble_key = f"ensemble_h{horizon}"
    ensemble_metrics: dict[str, float] = {}

    if (
        training_result.ensemble_result is not None
        and len(candidates) >= min_base_models_for_ensemble
    ):
        ensemble_metrics = (
            training_result.ensemble_result.metrics
            if hasattr(training_result.ensemble_result, "metrics")
            else {}
        )
        ensemble_score = ensemble_metrics.get(selection_metric, 0.0)
        if ensemble_score >= best_single_score:
            use_ensemble = True

    # Resolve source bundle path
    if use_ensemble:
        # Look for ensemble bundle in bundles/ directory
        source_candidates = [
            bundle_path / "ensemble",
            bundle_path / f"ensemble_h{horizon}",
        ]
        source = None
        for sc in source_candidates:
            if sc.exists():
                source = sc
                break

        if source is None:
            # Fall back to best single model
            logger.warning(
                f"Ensemble selected but no ensemble bundle found at {bundle_path}. "
                f"Falling back to best single model: {best_single_key}"
            )
            use_ensemble = False

    if not use_ensemble:
        # Best single model
        model_name = best_single_key.rsplit("_h", 1)[0]
        source_candidates = [
            bundle_path / best_single_key,
            bundle_path / f"{model_name}_h{horizon}",
        ]
        source = None
        for sc in source_candidates:
            if sc.exists():
                source = sc
                break

        if source is None:
            logger.warning(f"Bundle directory not found for {best_single_key} in {bundle_path}")
            return None

    # Copy bundle to artifact_dir
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    shutil.copytree(source, artifact_dir)

    # Determine metadata from the copied bundle
    n_features = 0
    seq_len = 0
    requires_seq = False
    requires_4d_flag = False
    model_family = "unknown"

    manifest_file = artifact_dir / "manifest.json"
    if manifest_file.exists():
        try:
            with open(manifest_file) as f:
                bundle_manifest = json.load(f)
            meta = bundle_manifest.get("metadata", bundle_manifest)
            n_features = meta.get("n_features", 0)
            seq_len = meta.get("sequence_length", 0)
            requires_seq = meta.get("requires_sequences", False)
            requires_4d_flag = meta.get("requires_4d", False)
            model_family = meta.get("model_family", "unknown")
        except Exception:
            pass

    selected_key = ensemble_key if use_ensemble else best_single_key
    selected_metrics = ensemble_metrics if use_ensemble else best_single_metrics

    return HorizonArtifactEntry(
        horizon=horizon,
        artifact_type="ensemble" if use_ensemble else "model",
        model_key=selected_key,
        model_family="ensemble" if use_ensemble else model_family,
        bundle_path=f"h{horizon}/artifact",
        feature_count=n_features,
        sequence_length=seq_len,
        requires_sequences=requires_seq,
        requires_4d=requires_4d_flag,
        metrics=dict(selected_metrics),
    )
```

### Decision Logic Summary

```
For each horizon:
  1. Collect all ModelTrainingResults matching "_h{horizon}"
  2. Sort by val_f1 descending -> best_single
  3. If ensemble_result exists AND base_model_count >= 2:
     a. Compare ensemble val_f1 vs best_single val_f1
     b. If ensemble >= best_single -> use ensemble
  4. Else -> use best single model
  5. Copy selected bundle dir to deploy/h{horizon}/artifact/
  6. Return HorizonArtifactEntry with metadata
```

### Dependency Order

- **Blocks:** P0-C-2 (factory _create_deploy calls this)
- **Blocked by:** P0-C-1 (HorizonArtifactEntry dataclass)

### Backward Compatibility

- New function, no existing behavior changed
- Uses `getattr` and `.get()` defensively for all result access
- If no candidates found, returns None (graceful degradation)

### Acceptance Criteria

1. With 3 models trained for horizon 20 and no ensemble: selects model with highest val_f1
2. With 3 models + ensemble for horizon 20, ensemble score > best single: selects ensemble
3. With 3 models + ensemble for horizon 20, ensemble score < best single: selects single model
4. With ensemble but only 1 model trained: selects single model (min_base_models_for_ensemble=2)
5. Copies correct bundle directory to artifact_dir
6. Returns None if no model results for given horizon
7. `ruff check src/inference/deploy.py` passes

### Effort Estimate

**L** -- 100-140 LOC

---

## P0-C-4: Validation Report Generation

### Task ID

`P0-C-4`

### Files to Modify

- `/home/jake/Desktop/Research/src/inference/deploy.py` (MODIFY -- add to same file)

### API / Exact Definition

```python
def validate_deploy_artifact(
    artifact_dir: Path,
    artifact_type: str = "model",
    sample_rows: int = 100,
) -> dict[str, Any]:
    """
    Per-artifact validation: load bundle, check integrity, optionally
    run a smoke predict on synthetic data.

    Checks performed:
    1. Directory exists and contains expected files (manifest.json, model files)
    2. Bundle loads successfully via ModelBundle.load() or EnsembleBundle.load()
    3. bundle.validate() returns a report
    4. (Optional) predict() on synthetic input returns expected output shape/type

    Args:
        artifact_dir: Path to the artifact directory (deploy/h{horizon}/artifact/)
        artifact_type: "model" or "ensemble"
        sample_rows: Number of synthetic rows for smoke test (0 to skip)

    Returns:
        dict with keys:
            passed: bool -- overall pass/fail
            checks: list of {name, passed, message} dicts
            timing_seconds: float -- total validation time
            artifact_type: str
            artifact_dir: str
    """
    import time

    start = time.time()
    checks: list[dict[str, Any]] = []

    # Check 1: Directory exists
    dir_ok = artifact_dir.exists() and artifact_dir.is_dir()
    checks.append({
        "name": "directory_exists",
        "passed": dir_ok,
        "message": str(artifact_dir) if dir_ok else f"Missing: {artifact_dir}",
    })
    if not dir_ok:
        return _validation_result(checks, start, artifact_type, artifact_dir)

    # Check 2: manifest.json exists
    manifest_exists = (artifact_dir / "manifest.json").exists()
    checks.append({
        "name": "manifest_exists",
        "passed": manifest_exists,
        "message": "manifest.json found" if manifest_exists else "manifest.json missing",
    })

    # Check 3: Load bundle
    bundle = None
    try:
        if artifact_type == "ensemble":
            from src.inference.ensemble_bundle import EnsembleBundle
            bundle = EnsembleBundle.load(artifact_dir)
        else:
            from src.inference.bundle import ModelBundle
            bundle = ModelBundle.load(artifact_dir)
        checks.append({
            "name": "bundle_loads",
            "passed": True,
            "message": f"Loaded as {type(bundle).__name__}",
        })
    except Exception as e:
        checks.append({
            "name": "bundle_loads",
            "passed": False,
            "message": f"Load failed: {e}",
        })

    # Check 4: bundle.validate()
    if bundle is not None and hasattr(bundle, "validate"):
        try:
            val_report = bundle.validate()
            val_ok = val_report.get("valid", True)
            checks.append({
                "name": "bundle_validate",
                "passed": val_ok,
                "message": str(val_report),
            })
        except Exception as e:
            checks.append({
                "name": "bundle_validate",
                "passed": False,
                "message": f"validate() failed: {e}",
            })

    # Check 5: Smoke predict with synthetic data (model bundles only, skip for ensemble)
    if bundle is not None and artifact_type == "model" and sample_rows > 0:
        try:
            import numpy as np

            meta = getattr(bundle, "metadata", None)
            n_feat = meta.n_features if meta else 10
            is_seq = meta.requires_sequences if meta else False
            is_4d = meta.requires_4d if meta else False
            seq_len = meta.sequence_length if meta and meta.sequence_length > 0 else 60

            # Build synthetic input of appropriate shape
            if is_4d:
                n_tf = meta.n_timeframes if meta and meta.n_timeframes > 0 else 3
                X = np.random.randn(1, n_tf, seq_len, n_feat).astype(np.float32)
            elif is_seq:
                X = np.random.randn(1, seq_len, n_feat).astype(np.float32)
            else:
                X = np.random.randn(sample_rows, n_feat).astype(np.float32)

            output = bundle.predict(X, calibrate=False)
            checks.append({
                "name": "smoke_predict",
                "passed": output is not None,
                "message": f"predict() returned {type(output).__name__}",
            })
        except Exception as e:
            checks.append({
                "name": "smoke_predict",
                "passed": False,
                "message": f"predict() failed: {e}",
            })

    return _validation_result(checks, start, artifact_type, artifact_dir)


def _validation_result(
    checks: list[dict[str, Any]],
    start_time: float,
    artifact_type: str,
    artifact_dir: Path,
) -> dict[str, Any]:
    """Build the validation result dict."""
    import time

    elapsed = time.time() - start_time
    all_passed = all(c["passed"] for c in checks)
    return {
        "passed": all_passed,
        "checks": checks,
        "timing_seconds": round(elapsed, 3),
        "artifact_type": artifact_type,
        "artifact_dir": str(artifact_dir),
    }
```

### Dependency Order

- **Blocks:** P0-C-2 (factory _create_deploy calls this)
- **Blocked by:** P0-C-1 (lives in same file), P0-B-1 (ModelBundle.predict must work)

### Backward Compatibility

- New function, no existing behavior changed
- Uses try/except for all bundle operations (graceful on failure)
- Smoke predict uses `predict()` with synthetic ndarray (not predict_from_raw), so it works even before P0-B is fully complete for all models

### Acceptance Criteria

1. For a valid ModelBundle directory: returns `{"passed": true, "checks": [...]}`
2. For a missing directory: returns `{"passed": false}` with `directory_exists` check failed
3. For a corrupt bundle: returns `{"passed": false}` with `bundle_loads` check failed
4. `validation.json` output is valid JSON
5. `ruff check src/inference/deploy.py` passes

### Effort Estimate

**M** -- 80-100 LOC

---

## P0-C-5: Deploy Helper Function (`load_deploy_artifact`)

### Task ID

`P0-C-5`

### Files to Modify

- `/home/jake/Desktop/Research/src/inference/deploy.py` (MODIFY -- add to same file)

### API / Exact Definition

```python
def load_deploy_artifact(
    deploy_dir: str | Path,
    horizon: int | None = None,
) -> Any:
    """
    Load the selected deploy artifact from a deploy/ directory.

    This is the primary user-facing function for loading a trained
    artifact for inference:

        from src.inference.deploy import load_deploy_artifact
        artifact = load_deploy_artifact("experiments/runs/exp_001/deploy", horizon=20)
        prediction = artifact.predict_from_raw(raw_bars_df)

    Args:
        deploy_dir: Path to the deploy/ directory (contains manifest.json)
        horizon: Which horizon's artifact to load. If None and only one
                 horizon exists, loads that one. If None and multiple
                 exist, raises ValueError.

    Returns:
        ModelBundle or EnsembleBundle, depending on what was selected.

    Raises:
        FileNotFoundError: If deploy_dir or manifest.json doesn't exist
        ValueError: If horizon is None and multiple horizons exist
        ValueError: If requested horizon not found in manifest
    """
    deploy_dir = Path(deploy_dir)

    if not deploy_dir.exists():
        raise FileNotFoundError(f"Deploy directory not found: {deploy_dir}")

    # Load manifest
    manifest = DeployManifest.load(deploy_dir)

    # Resolve horizon
    if horizon is None:
        if len(manifest.horizons) == 1:
            horizon = manifest.horizons[0]
        elif len(manifest.horizons) == 0:
            raise ValueError("No horizons found in deploy manifest")
        else:
            raise ValueError(
                f"Multiple horizons in manifest: {manifest.horizons}. "
                f"Specify horizon= parameter."
            )

    key = f"h{horizon}"
    if key not in manifest.selected_artifacts:
        raise ValueError(
            f"Horizon {horizon} not found in deploy manifest. "
            f"Available: {manifest.horizons}"
        )

    entry = manifest.selected_artifacts[key]
    artifact_path = deploy_dir / entry.bundle_path

    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Artifact directory not found: {artifact_path}"
        )

    # Load appropriate bundle type
    if entry.artifact_type == "ensemble":
        from src.inference.ensemble_bundle import EnsembleBundle
        return EnsembleBundle.load(artifact_path)
    else:
        from src.inference.bundle import ModelBundle
        return ModelBundle.load(artifact_path)
```

### Dependency Order

- **Blocks:** P0-D-1 (notebook inference demo cell uses this)
- **Blocked by:** P0-C-1 (DeployManifest), P0-C-2 (deploy directory must exist)

### Backward Compatibility

- Purely additive -- new function
- Works with both ModelBundle and EnsembleBundle

### Acceptance Criteria

1. `load_deploy_artifact("path/to/deploy", horizon=20)` returns a ModelBundle or EnsembleBundle
2. With single horizon, `load_deploy_artifact("path/to/deploy")` works without specifying horizon
3. With multiple horizons and no horizon specified: raises `ValueError`
4. With nonexistent directory: raises `FileNotFoundError`
5. `ruff check src/inference/deploy.py` passes

### Effort Estimate

**S** -- 40-50 LOC

---

## P0-C-6: Update `src/inference/__init__.py` Exports

### Task ID

`P0-C-6`

### Files to Modify

- `/home/jake/Desktop/Research/src/inference/__init__.py` (MODIFY)

### API Changes

Add after the `server` imports (after L139):

```python
# Deploy packaging (PHASE_0C)
from src.inference.deploy import (
    DeployManifest,
    HorizonArtifactEntry,
    load_deploy_artifact,
    select_deploy_artifact,
    validate_deploy_artifact,
)
```

Add to `__all__` list:

```python
    # Deploy (PHASE_0C)
    "DeployManifest",
    "HorizonArtifactEntry",
    "load_deploy_artifact",
    "select_deploy_artifact",
    "validate_deploy_artifact",
```

### Dependency Order

- **Blocks:** Nothing
- **Blocked by:** P0-C-1, P0-C-3, P0-C-4, P0-C-5

### Effort Estimate

**S** -- 10-15 LOC

---

## P0-C-7: Add `deploy_artifact` Toggle to BundlingSection

### Task ID

`P0-C-7`

### Files to Modify

- `/home/jake/Desktop/Research/src/config/experiment.py` (MODIFY)

### API Changes

Add new field to `BundlingSection` at L146 (after `create_bundle`):

```python
@dataclass
class BundlingSection:
    """
    Bundling-related configuration section.
    """

    create_bundle: bool = True
    deploy_artifact: bool = True            # NEW: Create deploy/ directory with selected artifact
    bundle_format: str = "directory"        # directory, tar.gz
    include_oof: bool = True
    include_feature_importance: bool = True
```

Update `from_dict()` parsing (~L297):

```python
        config_dict["bundling"] = BundlingSection(
            create_bundle=bundle_section_dict.get("create_bundle", True),
            deploy_artifact=bundle_section_dict.get("deploy_artifact", True),  # NEW
            ...
        )
```

Update `to_dict()` serialization (~L366):

```python
            "bundling": {
                "create_bundle": self.bundling.create_bundle,
                "deploy_artifact": self.bundling.deploy_artifact,  # NEW
                ...
            },
```

Then in `factory.py` `_create_deploy()`, gate on this config:

```python
            # Phase 5: Deploy packaging (optional)
            deploy_path = None
            if bundle_path is not None and self.config.bundling.deploy_artifact:
                ...
```

### Dependency Order

- **Blocks:** Nothing
- **Blocked by:** P0-C-2

### Backward Compatibility

- `deploy_artifact` defaults to `True` -- new behavior is opt-out
- Existing configs without `deploy_artifact` key deserialize with `True` default via `.get()`
- Setting `create_bundle=False` already skips bundling and thus deploy

### Acceptance Criteria

1. Default config has `deploy_artifact=True`
2. Setting `deploy_artifact=False` skips deploy directory creation
3. Existing config JSON without `deploy_artifact` loads successfully (defaults to True)
4. `ruff check src/config/experiment.py` passes

### Effort Estimate

**S** -- 10-15 LOC

---

# P0-D: Notebook Integration

All notebook changes go into `/home/jake/Desktop/Research/notebooks/ml_factory_colab.ipynb`.

Current cells: 0 (markdown), 1-7 (code). New cells are inserted after Cell 7 (save/download).

---

## P0-D-1: Inference Demo Cell (Cell 8)

### Task ID

`P0-D-1`

### Files to Modify

- `/home/jake/Desktop/Research/notebooks/ml_factory_colab.ipynb` (NEW CELL after cell-7)

### Exact Cell Code

```python
# =============================================================
# CELL 8: INFERENCE DEMO - Load artifact and predict
# =============================================================
from pathlib import Path
import pandas as pd

if "result" not in dir() or result is None or not result.success:
    print("No successful result. Run Cells 1-5 first.")
else:
    deploy_dir = None

    # Try deploy/ path first (Phase 5 output)
    if hasattr(result, "deploy_path") and result.deploy_path and Path(result.deploy_path).exists():
        deploy_dir = Path(result.deploy_path)
    elif result.output_dir and (Path(result.output_dir) / "deploy").exists():
        deploy_dir = Path(result.output_dir) / "deploy"

    if deploy_dir is not None:
        print("=" * 60)
        print("INFERENCE DEMO")
        print("=" * 60)

        # Load manifest
        import json
        manifest_path = deploy_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            print(f"Run ID:    {manifest['run_id']}")
            print(f"Horizons:  {manifest['horizons']}")
            print(f"Profile:   {manifest.get('runtime_profile', 'native')}")
            print()

            # Load artifact for each horizon
            for h_key, entry in manifest.get("selected_artifacts", {}).items():
                horizon = entry["horizon"]
                print(f"--- Horizon {horizon} ---")
                print(f"  Type:    {entry['artifact_type']}")
                print(f"  Model:   {entry['model_key']}")
                print(f"  Family:  {entry['model_family']}")
                print(f"  Metrics: {entry.get('metrics', {})}")
                print(f"  Valid:   {'PASS' if entry.get('validation_passed') else 'FAIL'}")
                print()

                # Load and run inference demo
                artifact_path = deploy_dir / entry["bundle_path"]
                if artifact_path.exists() and entry["artifact_type"] == "model":
                    try:
                        from src.inference.bundle import ModelBundle

                        bundle = ModelBundle.load(artifact_path)
                        print(f"  Bundle loaded: {bundle.metadata.model_name}")
                        print(f"  Features: {bundle.metadata.n_features}")
                        print(f"  Sequences: {bundle.metadata.requires_sequences}")
                        print(f"  4D: {bundle.metadata.requires_4d}")

                        # Demo prediction with last N bars of raw data
                        n_bars = max(200, bundle.metadata.sequence_length * 2 or 200)
                        if "raw_data" in dir() and len(raw_data) >= n_bars:
                            sample = raw_data.tail(n_bars).copy()
                            sample.index.name = "datetime"

                            import time
                            start = time.time()

                            try:
                                pred = bundle.predict_from_raw(sample)
                                elapsed = time.time() - start

                                print(f"\n  predict_from_raw() on {n_bars} bars:")
                                print(f"    Time:       {elapsed:.3f}s")
                                print(f"    Output type: {type(pred).__name__}")
                                if hasattr(pred, "predictions"):
                                    print(f"    Predictions: {pred.predictions[:5]}")
                                if hasattr(pred, "class_probabilities") and pred.class_probabilities is not None:
                                    print(f"    Probabilities shape: {pred.class_probabilities.shape}")
                            except Exception as e:
                                print(f"\n  predict_from_raw() failed: {e}")
                                print("  (This is expected if P0-B is not yet implemented for this model type)")
                        else:
                            print(f"\n  Skipping demo predict (need {n_bars} bars, have {len(raw_data) if 'raw_data' in dir() else 0})")

                    except Exception as e:
                        print(f"  Failed to load bundle: {e}")

                elif artifact_path.exists() and entry["artifact_type"] == "ensemble":
                    try:
                        from src.inference.ensemble_bundle import EnsembleBundle
                        bundle = EnsembleBundle.load(artifact_path)
                        print(f"  Ensemble loaded: {bundle.metadata.meta_learner_name}")
                        print(f"  Base models: {bundle.metadata.base_model_names}")
                    except Exception as e:
                        print(f"  Failed to load ensemble: {e}")

                print()
        else:
            print("No deploy manifest found.")

    elif result.bundle_path and Path(result.bundle_path).exists():
        # Fallback: list bundles directly (no deploy/ yet)
        print("=" * 60)
        print("BUNDLE INVENTORY (no deploy/ directory)")
        print("=" * 60)
        bundle_dir = Path(result.bundle_path)
        for item in sorted(bundle_dir.iterdir()):
            if item.is_dir():
                manifest = item / "manifest.json"
                if manifest.exists():
                    with open(manifest) as f:
                        m = json.load(f)
                    meta = m.get("metadata", {})
                    print(f"  {item.name}: {meta.get('model_family', '?')} "
                          f"| feat={meta.get('n_features', '?')} "
                          f"| seq={meta.get('requires_sequences', False)}")
        print("\nNote: Run with deploy_artifact=True to get the deploy/ directory.")
    else:
        print("No bundles or deploy directory found.")
```

### Dependency Order

- **Blocks:** Nothing
- **Blocked by:** P0-C-2 (deploy directory must be created by factory), P0-C-1 (manifest schema)

### Backward Compatibility

- Works with BOTH deploy/ and plain bundles/ directory (fallback path)
- Works for single-model and ensemble runs
- Gracefully handles missing predict_from_raw support (catches exception, prints message)
- If raw_data variable doesn't exist (e.g. user cleared state), skips demo prediction

### Acceptance Criteria

1. After successful factory run with deploy: shows manifest info + per-horizon artifact details
2. After successful factory run without deploy: falls back to bundle listing
3. predict_from_raw() demo runs for tabular models (may fail for neural until P0-B completes)
4. No cell execution errors even when bundles/deploy are missing

### Effort Estimate

**L** -- cell is ~100 lines

---

## P0-D-2: Deploy Export Cell (Cell 9)

### Task ID

`P0-D-2`

### Files to Modify

- `/home/jake/Desktop/Research/notebooks/ml_factory_colab.ipynb` (NEW CELL after cell-8/P0-D-1)

### Exact Cell Code

```python
# =============================================================
# CELL 9: EXPORT DEPLOY ARTIFACT
# =============================================================
from pathlib import Path
import shutil

if "result" not in dir() or result is None or not result.success:
    print("No successful result. Run Cells 1-5 first.")
else:
    deploy_dir = None
    if hasattr(result, "deploy_path") and result.deploy_path and Path(result.deploy_path).exists():
        deploy_dir = Path(result.deploy_path)
    elif result.output_dir and (Path(result.output_dir) / "deploy").exists():
        deploy_dir = Path(result.output_dir) / "deploy"

    if deploy_dir is not None:
        # Zip ONLY the deploy/ directory (not the full output)
        zip_name = f"{EXPERIMENT_NAME}_deploy"
        if IN_COLAB:
            zip_path = f"/content/{zip_name}"
        else:
            zip_path = str(deploy_dir.parent / zip_name)

        shutil.make_archive(zip_path, "zip", deploy_dir)
        zip_file = f"{zip_path}.zip"
        size_mb = Path(zip_file).stat().st_size / 1e6

        print("=" * 60)
        print("DEPLOY ARTIFACT EXPORT")
        print("=" * 60)
        print(f"  Source:  {deploy_dir}")
        print(f"  Archive: {zip_file}")
        print(f"  Size:    {size_mb:.1f} MB")

        if IN_COLAB:
            try:
                from google.colab import files
                files.download(zip_file)
                print("\n  Download started automatically.")
            except Exception:
                print(f"\n  Manual download: files.download('{zip_file}')")
        else:
            print(f"\n  Archive ready at: {zip_file}")
    else:
        print("No deploy directory found. Run with deploy_artifact=True in Cell 2 config.")
        if result.output_dir:
            print(f"Full output directory: {result.output_dir}")
```

### Dependency Order

- **Blocks:** Nothing
- **Blocked by:** P0-C-2 (deploy directory)

### Backward Compatibility

- Only zips deploy/ (not full output), so downloads are smaller and deployment-focused
- Falls back gracefully if no deploy directory

### Acceptance Criteria

1. Produces a `.zip` containing only the `deploy/` directory contents
2. In Colab: triggers auto-download
3. Locally: prints the archive path
4. If no deploy/ exists: prints helpful message

### Effort Estimate

**S** -- 35-40 lines

---

## P0-D-3: Validation Cell (Cell 10)

### Task ID

`P0-D-3`

### Files to Modify

- `/home/jake/Desktop/Research/notebooks/ml_factory_colab.ipynb` (NEW CELL after cell-9/P0-D-2)

### Exact Cell Code

```python
# =============================================================
# CELL 10: VALIDATION REPORT
# =============================================================
from pathlib import Path
import json

if "result" not in dir() or result is None or not result.success:
    print("No successful result. Run Cells 1-5 first.")
else:
    deploy_dir = None
    if hasattr(result, "deploy_path") and result.deploy_path and Path(result.deploy_path).exists():
        deploy_dir = Path(result.deploy_path)
    elif result.output_dir and (Path(result.output_dir) / "deploy").exists():
        deploy_dir = Path(result.output_dir) / "deploy"

    if deploy_dir is not None:
        print("=" * 60)
        print("DEPLOY VALIDATION REPORT")
        print("=" * 60)

        manifest_path = deploy_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)

            all_passed = True
            for h_key, entry in manifest.get("selected_artifacts", {}).items():
                horizon = entry["horizon"]
                val_path = deploy_dir / entry.get("validation_path", "")

                print(f"\n--- Horizon {horizon} ({entry['artifact_type']}: {entry['model_key']}) ---")

                if val_path.exists():
                    with open(val_path) as f:
                        val = json.load(f)

                    status = "PASS" if val["passed"] else "FAIL"
                    print(f"  Overall: {status}")
                    print(f"  Time:    {val.get('timing_seconds', '?')}s")

                    for check in val.get("checks", []):
                        icon = "OK" if check["passed"] else "FAIL"
                        print(f"  [{icon}] {check['name']}: {check['message'][:80]}")

                    if not val["passed"]:
                        all_passed = False
                else:
                    print("  Validation file not found.")
                    all_passed = False

            print("\n" + "=" * 60)
            print(f"Overall: {'ALL PASSED' if all_passed else 'SOME CHECKS FAILED'}")
            print("=" * 60)
        else:
            print("No deploy manifest found at", deploy_dir)
    else:
        print("No deploy directory found.")
```

### Dependency Order

- **Blocks:** Nothing
- **Blocked by:** P0-C-2 (deploy directory), P0-C-4 (validation reports)

### Backward Compatibility

- Read-only cell; only reads existing JSON files
- Works for both model and ensemble artifacts

### Acceptance Criteria

1. Displays per-horizon validation results from validation.json
2. Shows per-check pass/fail with messages
3. Shows overall pass/fail summary
4. Handles missing validation files gracefully

### Effort Estimate

**S** -- 45-50 lines

---

## P0-D-4: Config Additions (Cell 2 Modification)

### Task ID

`P0-D-4`

### Files to Modify

- `/home/jake/Desktop/Research/notebooks/ml_factory_colab.ipynb` (MODIFY Cell 2)

### Exact Changes to Cell 2

Add after the `GENERATE_REPORT = True` line (in the `--- EVALUATION ---` section) and before `--- EXPERIMENT ---`:

```python
# --- BUNDLING & DEPLOY ---
BUNDLING_ENABLED = True             # Create model bundles after training
DEPLOY_ARTIFACT = True              # Create deploy/ directory with selected best artifact
```

Then update Cell 5 (Training cell) to pass these through. In the `ExperimentConfig` constructor, the `bundling` section is NOT currently passed (it uses defaults). We need to add it.

### Exact Changes to Cell 5

In the `ExperimentConfig(...)` constructor, add after `evaluation=EvaluationSection(...)`:

```python
    # Note: BundlingSection imported at top of cell
    bundling=BundlingSection(
        create_bundle=BUNDLING_ENABLED,
        deploy_artifact=DEPLOY_ARTIFACT,
    ),
```

And add to the imports at the top of Cell 5:

```python
from src.config.experiment import (
    ExperimentConfig,
    DataSection,
    TrainingSection,
    EvaluationSection,
    BundlingSection,  # NEW
)
```

### Dependency Order

- **Blocks:** Nothing
- **Blocked by:** P0-C-7 (BundlingSection.deploy_artifact field)

### Backward Compatibility

- Both `BUNDLING_ENABLED` and `DEPLOY_ARTIFACT` default to `True` -- same as current behavior (bundling was already enabled by default)
- Users who don't change these toggles get the new deploy behavior automatically
- Setting `BUNDLING_ENABLED = False` disables both bundling and deploy (since deploy depends on bundles)

### Acceptance Criteria

1. Default notebook config produces deploy/ directory
2. Setting `BUNDLING_ENABLED = False` skips bundling and deploy
3. Setting `DEPLOY_ARTIFACT = False` creates bundles but skips deploy
4. All existing config toggles still work unchanged

### Effort Estimate

**S** -- 10-15 LOC across two cells

---

## P0-D-5: Drive Persistence Cell (Cell 11)

### Task ID

`P0-D-5`

### Files to Modify

- `/home/jake/Desktop/Research/notebooks/ml_factory_colab.ipynb` (NEW CELL after cell-10/P0-D-3)

### Exact Cell Code

```python
# =============================================================
# CELL 11: SAVE DEPLOY ARTIFACT TO GOOGLE DRIVE
# =============================================================
from pathlib import Path
import shutil

if "result" not in dir() or result is None or not result.success:
    print("No successful result.")
elif not IN_COLAB:
    print("Not running in Colab. Deploy artifacts saved locally.")
    if hasattr(result, "deploy_path") and result.deploy_path:
        print(f"  Deploy dir: {result.deploy_path}")
    elif result.output_dir:
        print(f"  Output dir: {result.output_dir}")
else:
    # Mount Google Drive if not already mounted
    drive_root = Path("/content/drive/MyDrive")
    if not drive_root.exists():
        try:
            from google.colab import drive
            drive.mount("/content/drive")
            print("Google Drive mounted.")
        except Exception as e:
            print(f"Could not mount Drive: {e}")
            print("Run this in a cell first:  from google.colab import drive; drive.mount('/content/drive')")

    if drive_root.exists():
        deploy_dir = None
        if hasattr(result, "deploy_path") and result.deploy_path and Path(result.deploy_path).exists():
            deploy_dir = Path(result.deploy_path)
        elif result.output_dir and (Path(result.output_dir) / "deploy").exists():
            deploy_dir = Path(result.output_dir) / "deploy"

        if deploy_dir is not None:
            # Save to Drive
            drive_dest = drive_root / "ml_factory_results" / EXPERIMENT_NAME / "deploy"
            drive_dest.parent.mkdir(parents=True, exist_ok=True)

            if drive_dest.exists():
                shutil.rmtree(drive_dest)
            shutil.copytree(deploy_dir, drive_dest)

            n_files = sum(1 for f in drive_dest.rglob("*") if f.is_file())
            size_mb = sum(f.stat().st_size for f in drive_dest.rglob("*") if f.is_file()) / 1e6

            print("=" * 60)
            print("SAVED TO GOOGLE DRIVE")
            print("=" * 60)
            print(f"  Destination: {drive_dest}")
            print(f"  Files:       {n_files}")
            print(f"  Size:        {size_mb:.1f} MB")
            print()
            print("To load later:")
            print(f"  from src.inference.deploy import load_deploy_artifact")
            print(f"  artifact = load_deploy_artifact('{drive_dest}')")
            print(f"  pred = artifact.predict_from_raw(raw_bars_df)")
        else:
            print("No deploy directory found. Run with DEPLOY_ARTIFACT = True.")
    else:
        print("Google Drive not available.")
```

### Dependency Order

- **Blocks:** Nothing
- **Blocked by:** P0-C-2 (deploy directory), P0-D-1 (comes after inference demo)

### Backward Compatibility

- Colab-only cell; locally it just prints the deploy path
- Does not modify any existing Drive data without explicit overwrite
- Prints the exact Python code needed to reload the artifact later

### Acceptance Criteria

1. In Colab with Drive mounted: copies deploy/ to Drive, prints path
2. In Colab without Drive: prints helpful mount instructions
3. Locally: prints local deploy path
4. Shows reload instructions with `load_deploy_artifact()`

### Effort Estimate

**S** -- 45-50 lines

---

# Dependency Graph

```
P0-C Tasks (Deploy Packaging):

  P0-C-1 (DeployManifest dataclass)     -- no deps
      |
      +------> P0-C-3 (artifact selector)   -- depends on C-1
      |            |
      +------> P0-C-4 (validation report)   -- depends on C-1
      |            |
      +------> P0-C-5 (load helper)         -- depends on C-1
      |            |
      v            v
  P0-C-2 (factory.py Phase 5)           -- depends on C-1, C-3, C-4
      |
      v
  P0-C-6 (__init__.py exports)          -- depends on C-1, C-3, C-4, C-5
  P0-C-7 (BundlingSection toggle)       -- depends on C-2


P0-D Tasks (Notebook Integration):

  P0-D-4 (config additions, Cell 2+5)   -- depends on C-7
      |
  P0-D-1 (inference demo, Cell 8)       -- depends on C-2
      |
  P0-D-2 (deploy export, Cell 9)        -- depends on C-2
      |
  P0-D-3 (validation cell, Cell 10)     -- depends on C-2, C-4
      |
  P0-D-5 (Drive persistence, Cell 11)   -- depends on C-2


Full P0-C/P0-D dependency on P0-A/P0-B:

  P0-A + P0-B (Foundation + Inference)
      |
      v
  P0-C (Deploy Packaging)
      |
      v
  P0-D (Notebook Integration)
```

---

# Recommended Execution Order

```
Batch 1 (no deps):
  P0-C-1  Create DeployManifest + HorizonArtifactEntry dataclasses

Batch 2 (depends on Batch 1):
  P0-C-3  Artifact selector function
  P0-C-4  Validation report function
  P0-C-5  load_deploy_artifact helper

Batch 3 (depends on Batch 2):
  P0-C-2  factory.py Phase 5 integration
  P0-C-7  BundlingSection.deploy_artifact toggle

Batch 4 (depends on Batch 3):
  P0-C-6  __init__.py export updates
  P0-D-4  Notebook config additions (Cell 2+5)

Batch 5 (depends on Batch 4):
  P0-D-1  Inference demo cell (Cell 8)
  P0-D-2  Deploy export cell (Cell 9)
  P0-D-3  Validation cell (Cell 10)
  P0-D-5  Drive persistence cell (Cell 11)
```

---

# Total Effort Summary

| Task | ID | Effort | LOC Range | Files |
|------|----|--------|-----------|-------|
| DeployManifest dataclass | P0-C-1 | M | 100-130 | 1 new |
| Factory Phase 5 | P0-C-2 | L | 80-100 | 1 modify |
| Artifact selector | P0-C-3 | L | 100-140 | 1 modify (same new file) |
| Validation report | P0-C-4 | M | 80-100 | 1 modify (same new file) |
| Deploy helper | P0-C-5 | S | 40-50 | 1 modify (same new file) |
| __init__.py exports | P0-C-6 | S | 10-15 | 1 modify |
| BundlingSection toggle | P0-C-7 | S | 10-15 | 1 modify |
| Inference demo cell | P0-D-1 | L | ~100 | 1 modify (notebook) |
| Deploy export cell | P0-D-2 | S | 35-40 | 1 modify (notebook) |
| Validation cell | P0-D-3 | S | 45-50 | 1 modify (notebook) |
| Config additions | P0-D-4 | S | 10-15 | 1 modify (notebook) |
| Drive persistence cell | P0-D-5 | S | 45-50 | 1 modify (notebook) |
| **Total** | | | **655-805** | **1 new, 3 modify, 1 notebook** |

---

# Key Design Decisions

## 1. Single New File for Deploy Logic

All deploy packaging logic lives in `src/inference/deploy.py`. This keeps the scope contained:
- `DeployManifest` + `HorizonArtifactEntry` dataclasses
- `select_deploy_artifact()` function
- `validate_deploy_artifact()` function
- `load_deploy_artifact()` user-facing function

## 2. Factory Phase 5 Is Fail-Safe

Phase 5 (deploy) wraps everything in try/except. If deploy packaging fails:
- `ExperimentResult.deploy_path` is `None`
- The factory run still succeeds
- Bundles are still available in `bundles/`

## 3. Manifest Is Pure JSON

`DeployManifest` serializes to/from plain JSON. The `load()` classmethod uses only `json.load()` and standard Python types. This means the manifest can be read by:
- Any Python script without importing `src/`
- JavaScript, Go, Rust, or any language with JSON support
- CI/CD pipelines that just need to know what was deployed

## 4. Notebook Cells Are Defensive

Every notebook cell checks:
- `result` exists and is successful
- `deploy_path` or `deploy_dir` exists
- Falls back to bundles/ if no deploy/
- Catches exceptions from bundle loading and predict_from_raw()

This ensures the notebook never crashes on partial results.

## 5. Artifact Selection Is Deterministic

The selection policy is explicit and logged:
1. Ensemble wins if it exists, has enough base models, and scores >= best single model
2. Otherwise best single model by `val_f1`
3. Decision is recorded in `manifest.json` with the metric snapshot

## 6. Backward Compat via Defaults

- `BundlingSection.deploy_artifact` defaults to `True`
- `ExperimentResult.deploy_path` defaults to `None`
- Factory Phase 5 only runs if Phase 4 succeeded
- Old configs without `deploy_artifact` key get `True` via `.get()`

---

*This document is a planning artifact. No code has been modified.*
