# Configuration Analysis & Source of Truth

**Status:** Draft / Analysis  
**Date:** 2026-01-13  
**Based on:** Codebase analysis and `FINDINGS 2.md`

## 🚨 Core Problem
The repository currently contains multiple potential sources of configuration, leading to ambiguity about which files actually control the pipeline's behavior.

## 🔍 Findings

### 1. The Real Source of Truth (Active)
The pipeline currently relies on Python-based configuration, not the YAML files in `config/`.

*   **Pipeline Config:** `src/pipeline_cli.py` and `src/phase1/config/` appear to be the active drivers for the data pipeline.
*   **Model Training:** `src/models/config/trainer_config.py` defines the `TrainerConfig` dataclass. Entry points largely construct this directly rather than merging from YAMLs.

### 2. The "Ghost" Configs (Inactive/Reference)
The files in `config/` (root) seem to be largely decorative or legacy templates in the current state, unless explicitly loaded by a specific script (which does not appear to be the default path).

*   `config/training.yaml` (Defaults to MES/MGC, but pipeline enforces single-symbol).
*   `config/cv.yaml` (Mentions embargo: 60, but code uses 1440).
*   `config/experiments/*`, `config/ensembles/*` (No clear references in active code paths).

## 🛠 Recommended Actions

### Short Term (Documentation)
1.  **Mark `config/` as Reference:** Explicitly document that `config/*.yaml` files are currently **templates** or **reference** implementation details, and that changing them *might not* affect the running pipeline unless the specific script loading them is used.
2.  **Centralize Config:** If the goal is to use YAMLs, a refactor is needed to ensure `pipeline_cli.py` and `train_model.py` explicitly load and merge these YAMLs before execution.

### Long Term (Refactor)
1.  **Adopt `Hydra` or `OmegaConf`:** Strictly enforce a pattern where CLI args override YAML config, which overrides Code defaults.
2.  **Purge Unused Configs:** Archive configuration files that are not wired into the `src/` logic.

## 📝 Summary for Developers
**If you need to change pipeline parameters today:**
Look at `src/pipeline_cli.py` or the argument parsers in `scripts/`. Do **not** assume changing `config/training.yaml` will have an immediate effect without verifying the loading logic.
