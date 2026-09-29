# Colab Notebook & Integration Readiness Audit

**Auditor:** colab-auditor
**Date:** 2026-02-15
**Notebook:** `notebooks/ml_factory_colab.ipynb`

---

## Overview

The Colab notebook is a well-structured 8-cell (1 markdown + 7 code) end-to-end training workflow. It clones the repo, configures experiments, runs the full MLFactory pipeline, displays results, and zips/downloads outputs. The notebook **does** trigger bundling via MLFactory's Phase 4, but the Colab cells themselves have **no explicit awareness** of inference bundles — they treat `result.output_dir` as a flat artifact directory.

---

## Notebook Flow (Cell by Cell)

### Cell 0 (Markdown)
Quick-start documentation. Lists all 12 models, GPU benefit tiers, and data format (MGC 1-minute).

### Cell 1: Setup
- Detects Colab via `"google.colab" in sys.modules`
- **Fresh clones** the repo every run (`shutil.rmtree` + `git clone`)
- Installs from `requirements-colab.txt` (Colab-optimized, avoids upgrading pre-installed packages)
- Adds repo to `sys.path`, verifies `MLFactory` and `ExperimentConfig` imports
- Checks GPU availability, prints device info

### Cell 2: Configuration
- All user-facing toggles in one cell: model selection (12 boolean flags), training params, walk-forward settings, feature config, ensemble, Optuna, backtest, reporting
- Builds `MODELS` list from toggle flags
- Defaults: walk-forward with 5 windows, expanding, 50 Optuna trials, all 12 models enabled

### Cell 3: Validation
- Validates model names against allowed set
- Checks data file exists
- Validates walk-forward math (min_train + n_windows * test <= 1.0)
- Warns about neural models without GPU, especially TFT (~10h on CPU)

### Cell 4: Data Loading
- Loads parquet or CSV
- Normalizes column names, validates OHLCV columns
- Handles datetime index detection (column or index)
- Reports missing values, previews data

### Cell 5: MLFactory Run
- Constructs `ExperimentConfig` with all cell-2 settings
- Creates `MLFactory(config, enable_checkpoints=True)`
- Calls `factory.run()`
- Catches `KeyboardInterrupt` for resume capability
- **Key:** The config uses defaults for `BundlingSection` → `create_bundle=True` by default

### Cell 6: Results & Visualization
- Displays model metrics table, ensemble metrics, backtest results
- Finds and displays up to 6 PNG plots from output directory
- Shows `result.bundle_path` and `result.output_dir` if present

### Cell 7: Save & Download
- Zips `result.output_dir` and triggers Colab download via `google.colab.files.download()`
- Tries to copy to Google Drive if mounted (`/content/drive/MyDrive/ml_factory_results/`)
- For local runs, just prints the path

---

## Post-Training State

### What MLFactory Produces (4 Phases)
1. **Data Pipeline** → cached parquet in `output_dir/cache/data_pipeline.parquet`
2. **Training** → pickled `TrainingRunResult` in `output_dir/cache/training_result.pkl`
3. **Evaluation** → backtest metrics in `output_dir/cache/evaluation.json`
4. **Bundling** → inference bundles in `output_dir/bundles/`

### Bundle Contents (per model)
Each bundle directory (e.g., `bundles/xgboost_h20/`) contains:
- `manifest.json` — version, model name, feature hash
- `model/` — serialized trained model
- `scaler.pkl` — fitted feature scaler
- `calibrator.pkl` — probability calibrator (optional)
- `features.json` — feature column names
- `metadata.json` — horizon, training date, metrics
- `preprocessing_graph.json` — preprocessing config for raw OHLCV inference
- `feature_spec.json` — feature specification for parity

### What the Notebook Downloads
The zip contains **everything** in `output_dir` — which includes the bundles **plus** checkpoints, cached data, config YAML, and plots. There is no separate "inference-only" packaging step.

---

## Colab-Specific Code & Constraints

### Colab Detection
```python
IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")
```

### Colab Constraints
| Constraint | Impact |
|-----------|--------|
| **Ephemeral filesystem** | Everything in `/content/` is lost on disconnect. The notebook handles this with Drive save + zip download |
| **GPU session limits** | Free tier: ~12h, T4. Pro: longer, A100. Affects TFT training feasibility |
| **Memory** | T4 has 15GB VRAM, ~12GB RAM. Walk-forward with all 12 models may OOM |
| **Package conflicts** | `pandas==2.2.2` pinned to avoid breaking `google-colab` package. `river` excluded due to pandas version conflict |
| **No persistent state** | Cannot do multi-session training without Drive mounting |

### Drive Integration
- Saves to `/content/drive/MyDrive/ml_factory_results/{experiment_name}/`
- Only if Drive is already mounted (no explicit mount cell)
- Copies the **entire** output directory, not just bundles

---

## Integration Gaps

### 1. No Inference Demo Cell
The notebook stops at "display results and download zip." There is no cell that:
- Loads a bundle back from disk
- Runs inference on new data
- Demonstrates the prediction API (`bundle.predict(X_new)`)

### 2. Bundle Path Not Prominently Surfaced
`result.bundle_path` is printed at the bottom of Cell 6, but users may not understand what it is or how to use it. The download in Cell 7 doesn't separate bundles from training artifacts.

### 3. No Inference-Only Package
The downloaded zip includes everything (~cache, checkpoints, raw data parquet). For deployment, users need only the `bundles/` directory. There's no option to download just the inference bundle.

### 4. BundlingSection Not Exposed in Configuration Cell
Cell 2 configures data, training, evaluation — but does not expose `BundlingSection` settings (`create_bundle`, `bundle_format`, `include_oof`, `include_feature_importance`). Users can't control bundling without editing Cell 5's config construction.

### 5. No Drive Mount Cell
If a user wants persistent storage, they must know to run `drive.mount('/content/drive')` before training. The notebook detects Drive if mounted but doesn't help mount it.

### 6. No Model Export for External Use
No ONNX export, no TorchScript, no standalone inference script. Bundles use pickle serialization — they require the full `src/` package to load (`ModelBundle.load()` imports from `src.models.base`, `src.models.registry`).

### 7. Resumability Documentation Gap
Cell 5 catches `KeyboardInterrupt` and mentions `factory.resume_from_checkpoint()`, but there's no Cell 5b that actually demonstrates resume. After a Colab disconnect, users would need to re-run setup and manually call resume.

### 8. Walk-Forward Config Not Validated Against Colab Memory
With 5 walk-forward windows and 12 models, memory usage can spike. The notebook warns about GPU for neural models but doesn't estimate or warn about RAM constraints.

---

## Requirements Analysis

### requirements-colab.txt
- **Correctly** pins `pandas==2.2.2` to avoid Colab conflicts
- **Correctly** excludes `river` (pandas version conflict documented)
- Installs: xgboost, lightgbm, catboost, optuna, numba, PyWavelets, typer, rich
- **Missing from Colab requirements:** No `pytz` (though Colab may have it pre-installed)
- Relies on Colab pre-installing: numpy, pandas, scipy, scikit-learn, torch, tqdm, pyarrow, PyYAML, matplotlib, joblib

### pyproject.toml
- `requires-python = ">=3.11"` — Colab currently runs Python 3.10/3.11, should be compatible
- Has `[project.optional-dependencies] serving = ["flask>=2.3.0"]` — not needed for training but would be needed for inference server
- `river>=0.21.0` in main deps but excluded from Colab (documented conflict)

### Version Compatibility Risk
- `torch>=2.2.0` required — Colab typically has 2.1.x or 2.2.x pre-installed. If Colab's torch is older, pip won't upgrade it (not in requirements-colab.txt), which could cause runtime issues

---

## Summary of Findings

| Category | Status | Details |
|----------|--------|---------|
| **Training pipeline** | GOOD | Full end-to-end flow works, walk-forward supported |
| **Bundling** | PARTIAL | MLFactory creates bundles by default, but notebook doesn't surface or validate them |
| **Inference demo** | MISSING | No cell to load bundle and run predictions |
| **Download/export** | PARTIAL | Downloads everything as zip; no inference-only package |
| **Colab compatibility** | GOOD | Dedicated requirements file, GPU detection, Drive save |
| **Resume support** | PARTIAL | Checkpoint code exists but no demo cell |
| **Requirements** | GOOD | Well-documented, Colab-aware, conflict handling |

### Priority Gaps for Inference-Ready Bundles
1. **HIGH:** Add inference demo cell (load bundle → predict on new data)
2. **HIGH:** Add "Download Inference Bundle Only" option to Cell 7
3. **MEDIUM:** Expose BundlingSection config in Cell 2
4. **MEDIUM:** Add Drive mount helper cell
5. **LOW:** Add ONNX/TorchScript export option for deployment without src/ dependency
6. **LOW:** Add resume demo cell for Colab disconnect recovery
