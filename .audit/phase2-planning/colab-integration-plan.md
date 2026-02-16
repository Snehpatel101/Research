# Colab Integration Plan — Phase 2

**Date:** 2026-02-15
**Input:** Phase 1 audit (CONSOLIDATED-FINDINGS.md sections 2.2, 3.3; colab-readiness.md)
**Scope:** Notebook changes to support inference demos, bundle-only exports, and Colab constraints

---

## 1. Inference Demo Cell

### Design

Add **Cell 8: Inference Demo** after Cell 7 (Save & Download). This cell loads a trained bundle and runs predictions on sample data from the training run.

### Code Outline

```python
# =============================================================
# CELL 8: INFERENCE DEMO — Load Bundle & Predict
# =============================================================
from pathlib import Path
from src.inference.bundle import ModelBundle
from src.inference.ensemble_bundle import EnsembleBundle
import pandas as pd
import numpy as np

if "result" not in dir() or result is None or not result.success:
    print("No successful result — skip inference demo.")
else:
    bundle_dir = Path(result.output_dir) / "bundles"
    if not bundle_dir.exists():
        print(f"No bundles found at {bundle_dir}")
    else:
        # --- Discover available bundles ---
        bundle_dirs = sorted([
            d for d in bundle_dir.iterdir()
            if d.is_dir() and (d / "manifest.json").exists()
        ])
        print(f"Found {len(bundle_dirs)} bundles:")
        for bd in bundle_dirs:
            print(f"  - {bd.name}")

        # --- Load best individual model bundle ---
        best_model = result.best_model
        best_bundle_dir = None
        for bd in bundle_dirs:
            if best_model and best_model.lower() in bd.name.lower():
                best_bundle_dir = bd
                break
        if best_bundle_dir is None and bundle_dirs:
            best_bundle_dir = bundle_dirs[0]  # fallback to first

        if best_bundle_dir:
            print(f"\nLoading bundle: {best_bundle_dir.name}")
            bundle = ModelBundle.load(best_bundle_dir)
            print(f"  Model:    {bundle.metadata.model_name}")
            print(f"  Family:   {bundle.metadata.model_family}")
            print(f"  Horizon:  H{bundle.metadata.horizon}")
            print(f"  Features: {bundle.metadata.n_features}")
            print(f"  Sequences: {bundle.metadata.requires_sequences}")
            print(f"  4D:       {bundle.metadata.requires_4d}")

            # --- Predict on sample data (tabular models only for now) ---
            if not bundle.metadata.requires_sequences and not bundle.metadata.requires_4d:
                # Use last N rows of training data as sample
                sample_size = min(100, len(raw_data))
                sample_df = raw_data.tail(sample_size).copy()

                if bundle.preprocessing_graph is not None:
                    features = bundle.preprocess(sample_df)
                    preds = bundle.predict(features)
                    print(f"\n  Predictions on {len(preds.class_predictions)} samples:")
                    unique, counts = np.unique(preds.class_predictions, return_counts=True)
                    for cls, cnt in zip(unique, counts):
                        label = {0: "SHORT", 1: "HOLD", 2: "LONG"}.get(int(cls), str(cls))
                        print(f"    {label}: {cnt} ({cnt/len(preds.class_predictions)*100:.1f}%)")
                    print(f"  Mean confidence: {preds.confidence.mean():.4f}")
                else:
                    print("\n  [No preprocessing graph — raw-to-prediction requires Phase 2 adapter integration]")
                    print("  Use bundle.predict(X_preshaped) with pre-computed feature arrays.")
            else:
                print(f"\n  [Neural/Transformer model — requires adapter integration for raw inference]")
                print(f"  Use bundle.predict(X_3d_or_4d) with pre-shaped tensors.")

        # --- Check for ensemble bundle ---
        ensemble_dir = bundle_dir / "ensemble"
        if not ensemble_dir.exists():
            # Try finding any directory with "ensemble" in name
            ensemble_candidates = [d for d in bundle_dirs if "ensemble" in d.name.lower()]
            if ensemble_candidates:
                ensemble_dir = ensemble_candidates[0]

        if ensemble_dir.exists() and (ensemble_dir / "manifest.json").exists():
            print(f"\nEnsemble bundle found: {ensemble_dir.name}")
            try:
                ens_bundle = EnsembleBundle.load(ensemble_dir)
                print(f"  Meta-learner:  {ens_bundle.metadata.meta_learner_name}")
                print(f"  Base models:   {ens_bundle.metadata.base_model_names}")
                print(f"  Stacking features: {ens_bundle.metadata.n_stacking_features}")
            except Exception as e:
                print(f"  [Could not load ensemble bundle: {e}]")

        print("\n--- Inference demo complete ---")
```

### Key Decisions

- **Tabular models only** for the initial raw-data demo — tabular models (4 of 14, including 3 boosting + random_forest) have working `predict_from_raw()`. Neural/transformer models will show a placeholder message until the Phase 2 adapter integration lands.
- **Uses `raw_data` from Cell 4** as sample input — avoids requiring the user to provide separate test data.
- **Graceful degradation** — if no preprocessing graph exists, explains what the user needs to do manually.

---

## 2. Inference-Only Export

### Design

Add a **Cell 9: Download Inference Bundle Only** that strips cache/checkpoints and packages just the `bundles/` directory.

### Code Outline

```python
# =============================================================
# CELL 9: DOWNLOAD INFERENCE BUNDLE ONLY
# =============================================================
from pathlib import Path
import shutil

if "result" not in dir() or result is None or not result.success:
    print("No successful result to export.")
elif result.output_dir:
    bundle_dir = Path(result.output_dir) / "bundles"
    if not bundle_dir.exists() or not any(bundle_dir.iterdir()):
        print("No bundles found. Was bundling enabled in config?")
    else:
        # Count bundle contents
        bundle_subdirs = [d for d in bundle_dir.iterdir() if d.is_dir()]
        n_files = sum(1 for f in bundle_dir.rglob("*") if f.is_file())

        # Create inference-only zip (bundles/ only, no cache/checkpoints/raw data)
        zip_name = f"/content/{EXPERIMENT_NAME}_inference_bundle"
        shutil.make_archive(zip_name, "zip", bundle_dir)
        zip_path = Path(f"{zip_name}.zip")

        full_zip = Path(f"/content/{EXPERIMENT_NAME}_results.zip")
        full_size = full_zip.stat().st_size / 1e6 if full_zip.exists() else 0

        print(f"Inference bundle: {zip_path}")
        print(f"  Models:   {len(bundle_subdirs)}")
        print(f"  Files:    {n_files}")
        print(f"  Size:     {zip_path.stat().st_size / 1e6:.1f} MB")
        if full_size > 0:
            print(f"  (vs full: {full_size:.1f} MB — {zip_path.stat().st_size / full_zip.stat().st_size * 100:.0f}% of full)")

        if IN_COLAB:
            try:
                from google.colab import files
                files.download(str(zip_path))
            except Exception:
                print(f"\nManual download: files.download('{zip_path}')")
else:
    print("No output directory found.")
```

### What Gets Stripped

| Included | Excluded |
|----------|----------|
| `bundles/*/manifest.json` | `cache/` (data pipeline parquet, training pkl) |
| `bundles/*/model/` | `checkpoints/` |
| `bundles/*/scaler.pkl` | `experiment_config.yaml` |
| `bundles/*/features.json` | `*.png` plots |
| `bundles/*/metadata.json` | Raw data copies |
| `bundles/*/preprocessing_graph.json` | Evaluation JSONs |
| `bundles/*/feature_spec.json` | |
| `bundles/*/calibrator.pkl` | |

---

## 3. Auto-Packaging Flow

### Current State

```
MLFactory.run()
  └── Phase 4: _create_bundle()
        └── BundleBuilder.build_from_training_result(training_result)
              └── Saves to output_dir/bundles/{model_name}_h{horizon}/
```

This already runs automatically. The issue is that the **notebook doesn't surface it**.

### Changes Required

1. **Cell 5 (MLFactory Run)**: No code changes needed — `config.bundling.create_bundle` defaults to `True`, so bundles are already created.

2. **Cell 6 (Results)**: Add bundle summary section after existing results display:

```python
    # --- Bundle Summary ---
    if result.bundle_path and Path(result.bundle_path).exists():
        bundle_dir = Path(result.bundle_path)
        bundle_subdirs = sorted([
            d for d in bundle_dir.iterdir()
            if d.is_dir() and (d / "manifest.json").exists()
        ])
        if bundle_subdirs:
            print("-" * 40)
            print(f"Inference Bundles ({len(bundle_subdirs)} created)")
            print("-" * 40)
            for bd in bundle_subdirs:
                import json
                meta_path = bd / "metadata.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                    family = meta.get("model_family", "?")
                    n_feat = meta.get("n_features", "?")
                    print(f"  {bd.name}: {family}, {n_feat} features")
                else:
                    print(f"  {bd.name}")
            total_size = sum(f.stat().st_size for f in bundle_dir.rglob("*") if f.is_file())
            print(f"  Total bundle size: {total_size / 1e6:.1f} MB")
```

3. **No changes to `src/factory.py`** — the auto-packaging flow works. The gap is purely in notebook visibility.

---

## 4. Bundling Config Exposure

### Current State

Cell 2 builds `ExperimentConfig` but never sets `BundlingSection` fields. The defaults (`create_bundle=True`, `bundle_format="directory"`, `include_oof=True`, `include_feature_importance=True`) silently apply.

### Change to Cell 2

Add a bundling section to the configuration cell:

```python
# --- BUNDLING ---
CREATE_BUNDLE = True              # Create inference bundles after training
BUNDLE_FORMAT = "directory"       # "directory" or "tar.gz"
INCLUDE_OOF = True                # Include out-of-fold predictions in bundle
INCLUDE_FEATURE_IMPORTANCE = True  # Include feature importance scores
```

### Change to Cell 5

Update `ExperimentConfig` construction to include bundling:

```python
from src.config.experiment import BundlingSection

config = ExperimentConfig(
    # ... existing fields ...
    bundling=BundlingSection(
        create_bundle=CREATE_BUNDLE,
        bundle_format=BUNDLE_FORMAT,
        include_oof=INCLUDE_OOF,
        include_feature_importance=INCLUDE_FEATURE_IMPORTANCE,
    ),
)
```

---

## 5. Drive Persistence

### Design

Add a **Cell 7b: Save Bundles to Google Drive** between the current Cell 7 and new Cell 8.

### Code Outline

```python
# =============================================================
# CELL 7b: SAVE TO GOOGLE DRIVE (optional)
# =============================================================
# Run this cell to mount Google Drive and save bundles for persistence.
# Colab's filesystem is EPHEMERAL — files are lost on disconnect.
# Google Drive keeps your bundles across sessions.

import os

if IN_COLAB:
    drive_mounted = os.path.exists("/content/drive/MyDrive")

    if not drive_mounted:
        print("Mounting Google Drive...")
        from google.colab import drive
        drive.mount("/content/drive")
        drive_mounted = os.path.exists("/content/drive/MyDrive")

    if drive_mounted and "result" in dir() and result and result.success:
        from pathlib import Path
        import shutil

        # Save bundles (not full output)
        drive_base = Path("/content/drive/MyDrive/ml_factory_results")
        drive_dest = drive_base / EXPERIMENT_NAME / "bundles"
        drive_dest.mkdir(parents=True, exist_ok=True)

        bundle_src = Path(result.output_dir) / "bundles"
        if bundle_src.exists():
            # Copy bundles to Drive
            if (drive_dest).exists():
                shutil.rmtree(drive_dest)
            shutil.copytree(bundle_src, drive_dest)
            print(f"Bundles saved to Drive: {drive_dest}")

            # Also save config for reference
            config_src = Path(result.output_dir) / "experiment_config.yaml"
            if config_src.exists():
                shutil.copy2(config_src, drive_base / EXPERIMENT_NAME / "experiment_config.yaml")

            n_bundles = sum(1 for d in drive_dest.iterdir() if d.is_dir())
            drive_size = sum(f.stat().st_size for f in drive_dest.rglob("*") if f.is_file())
            print(f"  {n_bundles} bundles, {drive_size / 1e6:.1f} MB")
            print(f"\nTo reload in a future session:")
            print(f"  from src.inference.bundle import ModelBundle")
            print(f"  bundle = ModelBundle.load('{drive_dest}/xgboost_h20')")
        else:
            print("No bundles found to save.")
    elif not drive_mounted:
        print("Drive mount failed or was cancelled.")
    else:
        print("No successful result to save.")
else:
    print("Not in Colab — files persist locally at:", result.output_dir if "result" in dir() and result else "N/A")
```

### Key Decision

- **Only bundles** go to Drive (not the full output_dir), to reduce Drive usage and speed up the copy.
- **Explicit mount** — unlike current Cell 7 which only uses Drive if pre-mounted, this cell actively mounts Drive.
- **Reload instructions** — prints the exact import/load code for future sessions.

---

## 6. Colab Constraint Mitigations

### 6.1 Memory Warnings

**Add to Cell 3 (Validation):**

```python
# Memory estimation warning
n_neural = len([m for m in MODELS if m in NEURAL_MODELS])
if TRAINING_MODE == "walk_forward" and WF_N_WINDOWS >= 5 and len(MODELS) >= 8:
    warnings.append(
        f"Walk-forward with {WF_N_WINDOWS} windows and {len(MODELS)} models may exceed "
        f"Colab's ~12GB RAM. Consider: fewer models, fewer windows, or Colab Pro (25GB)."
    )
if n_neural >= 6 and torch.cuda.is_available():
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    if gpu_mem < 16:
        warnings.append(
            f"GPU has {gpu_mem:.0f}GB VRAM. Running {n_neural} neural models may OOM. "
            f"Consider training in batches (boosting first, neural second)."
        )
```

### 6.2 Torch Version Check

**Add to Cell 1 (Setup), after GPU check:**

```python
# Torch version check
import torch
REQUIRED_TORCH = "2.2.0"
if torch.__version__ < REQUIRED_TORCH:
    print(f"WARNING: torch {torch.__version__} installed, but >={REQUIRED_TORCH} required.")
    print(f"  Neural models may fail. Upgrade with: pip install torch>={REQUIRED_TORCH}")
```

### 6.3 Ephemeral Filesystem Handling

**Add to Cell 5 (MLFactory Run), before `factory.run()`:**

```python
# Ephemeral filesystem warning
if IN_COLAB:
    print("NOTE: Colab filesystem is ephemeral. Save bundles to Drive (Cell 7b)")
    print("      or download them (Cell 7/9) before disconnecting.\n")
```

### 6.4 pandas Version Check

Already handled by `requirements-colab.txt` pinning `pandas==2.2.2`. No additional mitigation needed.

---

## 7. Cell-by-Cell Additions

### Final Notebook Structure

| Cell | Status | Description |
|------|--------|-------------|
| **Cell 0** (Markdown) | MODIFY | Add inference section to Quick Start |
| **Cell 1** (Setup) | MODIFY | Add torch version check after GPU check |
| **Cell 2** (Config) | MODIFY | Add bundling config section (4 toggles) |
| **Cell 3** (Validation) | MODIFY | Add memory/VRAM warnings |
| **Cell 4** (Data Load) | NO CHANGE | |
| **Cell 5** (MLFactory Run) | MODIFY | Add `BundlingSection` to config; add ephemeral FS warning |
| **Cell 6** (Results) | MODIFY | Add bundle summary section at end |
| **Cell 7** (Save & Download) | NO CHANGE | (keeps existing full-zip download) |
| **Cell 7b** (Drive Persistence) | **NEW** | Mount Drive, save bundles only |
| **Cell 8** (Inference Demo) | **NEW** | Load bundle, run predictions, display results |
| **Cell 9** (Inference-Only Export) | **NEW** | Download bundles-only zip |

### Modification Details

**Cell 0 (Markdown) — Add to Quick Start:**
```markdown
## After Training
7. **Cell 7b** - Save bundles to Google Drive (persistent)
8. **Cell 8** - Run inference demo (load bundle → predict)
9. **Cell 9** - Download inference-only bundle (smaller zip)
```

**Cell 1 — Insert after GPU check (line ~end):**
- Torch version validation (6 lines, see section 6.2)

**Cell 2 — Insert before `# BUILD MODEL LIST`:**
- 4 bundling toggles (see section 4)

**Cell 3 — Insert after existing warnings:**
- Memory estimation warning (see section 6.1)

**Cell 5 — Two changes:**
1. Import `BundlingSection` and add to `ExperimentConfig` constructor
2. Add ephemeral filesystem note before `factory.run()` (see section 6.3)

**Cell 6 — Append after existing output:**
- Bundle summary block (see section 3, change #2)

**Cell 7b — NEW cell:**
- Drive mount + bundle save (see section 5)

**Cell 8 — NEW cell:**
- Inference demo (see section 1)

**Cell 9 — NEW cell:**
- Inference-only export (see section 2)

---

## 8. Single Model vs Ensemble

### How the Inference Demo Handles Both Cases

**Cell 8** handles both through a two-stage approach:

#### Stage 1: Best Individual Model

1. Find bundle matching `result.best_model` in `bundles/` directory
2. Load via `ModelBundle.load(path)`
3. For tabular models (boosting): run `bundle.preprocess()` + `bundle.predict()` if preprocessing graph exists
4. For neural/transformer: print placeholder message explaining adapter integration is needed
5. Display prediction distribution (SHORT/HOLD/LONG counts) and mean confidence

#### Stage 2: Ensemble (if exists)

1. Look for ensemble bundle in `bundles/ensemble*` directory
2. Load via `EnsembleBundle.load(path)`
3. Display metadata (meta-learner name, base models, stacking features)
4. **Do not run ensemble predictions** in the demo — ensemble requires base model predictions as input, which requires all base models to predict first. This is a multi-step flow better suited for a dedicated inference script.

#### Decision Matrix

| Scenario | Individual Demo | Ensemble Demo |
|----------|----------------|---------------|
| Only boosting models trained | Full raw→predict demo | N/A (no ensemble with 1 model) |
| Boosting + neural, no ensemble | Boosting: full demo; Neural: metadata only | N/A |
| Boosting + neural + ensemble | Best model demo (likely boosting) | Metadata display |
| Only neural models | Metadata + shape info only | N/A or metadata |
| Ensemble with all models | Best model demo | Full metadata display |

#### Post-Phase-2 Enhancement

Once the universal inference pipeline (adapter integration) lands:
- Neural/transformer models will get full raw→predict demos
- Ensemble bundle will get end-to-end `predict_from_base_features()` demo
- A unified `UniversalInferencePipeline.predict_from_raw(df)` call will replace the manual routing

---

## Dependencies

| This Plan Item | Depends On | Blocks |
|---------------|-----------|--------|
| Inference Demo (tabular) | Nothing — works today for boosting | Nothing |
| Inference Demo (neural) | Phase 2: Adapter integration in inference | Full 12-model demo |
| Inference-Only Export | Nothing — works today | Nothing |
| Bundling Config | Nothing — `BundlingSection` exists | Nothing |
| Drive Persistence | Nothing | Nothing |
| Constraint Mitigations | Nothing | Nothing |

**All items except neural inference demos can be implemented immediately** — they don't depend on the adapter integration work.

---

## Implementation Priority

1. **Cell 2 + Cell 5**: Expose bundling config (5 min, low risk)
2. **Cell 6**: Add bundle summary (5 min, low risk)
3. **Cell 7b**: Drive persistence (new cell, 10 min, low risk)
4. **Cell 9**: Inference-only export (new cell, 10 min, low risk)
5. **Cell 8**: Inference demo (new cell, 15 min, medium risk — depends on bundle state)
6. **Cells 0/1/3**: Constraint mitigations and docs (10 min, low risk)

Total estimated notebook changes: ~200 lines of new code across 3 new cells + modifications to 5 existing cells.
