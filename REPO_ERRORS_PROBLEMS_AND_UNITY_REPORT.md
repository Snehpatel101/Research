# Repo Audit: Errors, Misalignments, and “Lack of Unity”

**Scope of this report:** repo-wide, but weighted toward `src/` (code wiring, runtime breakages, structural problems). No code changes were made.

**High-level intent (reconstructed from `CLAUDE.md`, `docs/`, and `src/`):** this repository is meant to be an *adaptive ML pipeline / factory* for futures OHLCV data, supporting single-model training and ensemble/meta-learner training under “institutional” standards (leakage controls, reproducibility, standardized artifacts).

---

## 0) Executive Summary (What’s broken, in one page)

### What you’re trying to build
- A single-contract (one symbol per run) OHLCV **model factory**:
  - Phase 1: ingest → clean/resample → features + MTF → labels (triple barrier) → optimize label params → splits → scaling → datasets → validation/reporting
  - Phase 2+: model registry + trainer (tabular + sequence + ensembles), plus cross-validation/OOF stacking, plus inference bundling/serving.

### What you actually have
- The “shape” of a model factory is present in `src/` (pipeline runner + stages + model registry + trainer + inference bundle).
- But the repo is **not runnable as-is** in the current environment:
  - dependencies are missing (Typer, scikit-learn, numba, xgboost, etc.)
  - some CLI modules import non-existent modules
  - docs + config references point to scripts/modules that don’t exist.

### The core “unity” problem
There isn’t one authoritative truth for:
- **how to run the system** (CLI vs “scripts/” that don’t exist vs docs that contradict each other),
- **how configuration works** (Python dataclasses vs YAML configs, and which layer drives what),
- **what’s implemented** (13 vs 23 models, MTF partial vs complete, etc.),
- **what module names are** (`stages` vs `src.phase1.stages`, etc.).

---

## 1) Hard Breakages (P0 — prevents running)

### 1.1 Missing dependencies (runtime + tests)

**Observed**
- `./pipeline --help` fails immediately:
  - `src/pipeline_cli.py` imports `src.cli`, which imports `typer`, but `typer` is not installed.
- `.venv/bin/python -m pytest -q` fails during test collection due to missing packages:
  - `sklearn` (scikit-learn)
  - `numba`
  - `xgboost`
  - (and likely more if those were installed)

**Evidence**
- Wrapper entrypoint: `pipeline` (root) runs `python3 src/pipeline_cli.py`
- Error on invocation: `src/cli/__init__.py` → `ModuleNotFoundError: No module named 'typer'`
- Test collection errors show missing `sklearn`, `numba`, `xgboost`

**Impact**
- The pipeline CLI cannot run.
- Tests cannot be executed, so the repo can’t currently certify correctness.

**Likely cause**
- `.venv` exists but isn’t fully provisioned.
- The system python also lacks requirements.

---

### 1.2 Root packaging is inconsistent (missing README)

**Observed**
- `pyproject.toml` declares: `readme = "README.md"`
- but `README.md` is missing at repo root.

**Impact**
- Editable installs / packaging metadata can break or be incomplete.
- This also signals a repo “entry narrative” gap (root README is usually the canonical quickstart).

---

### 1.3 CLI “status” path is broken by incorrect imports

**Observed**
- `src/cli/status_commands.py` tries to import modules that do not exist:
  - `from .. import pipeline_config` (no `src/pipeline_config.py`)
  - `from .. import manifest` (no `src/manifest.py`)

**Impact**
- Even if dependencies were installed, `pipeline status ...` is likely non-functional.

**Why this matters**
- This is an example of the repo having “two truths”: the run command path uses `src/phase1/pipeline_config.py`, but status tries to use a different (missing) layout.

---

## 2) Major Misalignments (P1 — “lack of unity”)

### 2.1 Documentation contradicts itself about implementation status

There are direct contradictions across first-party docs:

- `docs/README.md` claims **23 models** and **MTF complete** (9 TF ladder).
- `docs/planning/PROJECT_CHARTER.md` claims **13 models deployed** and **MTF partially implemented**.
- `config/README.md` / `config/INDEX.md` also talk in “13 model” terms and reference **scripts that don’t exist** (e.g., `scripts/train_model.py`).
- `CLAUDE.md` claims MTF is complete and speaks in “Phases 1–7 complete” terms.

**Impact**
- A new developer can’t tell what the repo is *actually* capable of.
- You lose trust in docs, which kills velocity.

**Recommendation (process-level)**
- Pick one canonical statement of truth (e.g., `docs/README.md` or a root `README.md`) and make everything else either:
  - derived from it, or
  - explicitly marked “historical / archive”.

---

### 2.2 “Ghost module name” problem: `stages` doesn’t exist

**Observed**
- Many docs/tests use imports like:
  - `from stages import DataIngestor, DataCleaner, FeatureEngineer`
  - `from stages.labeling...`
- But there is no `stages/` package at repo root.
- The actual code lives under `src/phase1/stages/`.

**Impact**
- Tests that import `stages` will fail even with deps installed.
- Docs that show `stages.*` usage mislead users into dead ends.

**Root cause**
- Refactor incomplete: module name changed but references weren’t updated or an alias shim wasn’t provided.

---

### 2.3 Pipeline run isolation is contradicted by actual artifact paths

**Intent stated in code**
- `src/phase1/config/pipeline_paths.py` explicitly promotes run isolation:
  - “All outputs: run-scoped under `runs/{run_id}/data/`”

**Observed**
- Stage 1 writes validated data globally under:
  - `data/raw/validated/{SYMBOL}_1m_validated.parquet`
  - (see `src/phase1/stages/ingest/run.py`)

**Impact**
- Run reproducibility and isolation get blurred:
  - multiple runs can overwrite/contaminate “validated” state
  - run artifacts are split between global and run-scoped locations

**This may be intentional**
- If Stage 1 is meant as a cache, it needs to be **explicitly** documented as such and treated as a derived dataset (with versioning / checksum).

---

### 2.4 Config split-brain (Python dataclasses vs YAML configs)

**Observed**
- Pipeline config is primarily Python (`src/phase1/pipeline_config.py` and `src/phase1/config/*`).
- Training config is primarily YAML (`config/models/*.yaml`, `config/pipeline/*.yaml`) and loaded by `src/models/config/*`.

**Impact**
- It’s unclear which knobs matter for which flow:
  - Phase 1 pipeline vs Phase 2 trainer vs CV vs ensembles
  - CLI options vs config YAML defaults vs code defaults

**Concrete example**
- `config/pipeline/cv.yaml` sets `purged_kfold.embargo: 60`
- Phase 1 config defaults/patterns use `embargo_bars: 1440` (and can auto-scale)
- Without a single config merge layer, you can easily end up with incompatible leakage controls.

---

## 3) Code-Level Problems in `src/` (P1/P2)

### 3.1 Import-time fragility from “auto-register everything”

**Observed**
- `src/models/__init__.py` imports `boosting`, `neural`, etc. to force registration.
- Those submodules import heavy optional deps (xgboost/torch/etc.) at import time.

**Impact**
- If *any* one of those deps is missing, importing `src.models` can fail entirely.
- This blocks partial usage (e.g., “run only pipeline stages” or “use only classical models”).

**Recommendation**
- Make optional deps truly optional: register models lazily or guard imports with try/except + conditional registration (like CatBoost does).

---

### 3.2 Internal docs/strings contradict the “single contract isolation” claim

**Observed**
- `CLAUDE.md` and top docs emphasize strict single-symbol isolation.
- But some docstrings still mention “cross-asset features (MES-MGC correlation, beta, etc.)”
  - Example: `src/phase1/stages/features/run.py` docstring

**Impact**
- Unclear whether cross-symbol features are intended (they violate the isolation principle) or are stale narrative.

---

### 3.3 Export list lies: `__all__` includes missing functions

**Observed**
- `src/phase1/stages/features/numba_functions.py` exports:
  - `calculate_rolling_correlation_numba`, `calculate_rolling_beta_numba`
- But these functions do not exist in the module.

**Impact**
- Indicates incomplete refactor or dropped functionality.
- If any code tries to import these from the module, it will break.

---

## 4) Static Analysis Snapshot (Ruff + Syntax)

### 4.1 Syntax
- `python3 -m compileall -q src tests` completes successfully (no syntax errors).

### 4.2 Ruff
`ruff check src tests --statistics` reports **many** issues (854 total earlier; stats show the big buckets):
- Import hygiene: `I001` unsorted imports (181)
- Unused imports/vars: `F401` (172), `F841` (106)
- Undefined names: `F821` (21)
- Undefined exports: `F822` (2)
- Exception chaining hygiene: `B904` (23)
- Lots of whitespace / formatting drift

**Interpretation**
- This isn’t just “style” debt: `F821` / `F822` often point to real wiring/refactor issues.
- The volume suggests the repo has been moving fast without a single enforced quality gate.

---

## 5) Repo-Level Organization Problems (Non-code, but blocking clarity)

### 5.1 Docs moved but git state is messy
`git status` shows many deleted docs and many “archive” re-additions/untracked docs.

**Impact**
- It’s impossible to know which documentation set is authoritative.

### 5.2 Root directory clutter and naming drift
Root contains multiple planning/finding docs with inconsistent naming and spacing (`FINDINGS 1.MD`, `SNEH IMPROVEMENT PLAN/`, etc.).

**Impact**
- Adds noise and reduces discoverability.
- Encourages “random notes” rather than a unified documentation structure.

---

## 6) Where You’re Going (A sane consolidation path)

This is a suggested “unity” roadmap (documentation/process level; not code changes in this report):

1. **Define the canonical run path**
   - Decide: is the supported entry point `./pipeline ...` (Typer) or something else?
   - Ensure docs reference only that path.

2. **Install/lock dependencies**
   - Ensure `.venv` matches `pyproject.toml` / `requirements.txt`.
   - Add a single “bootstrap” instruction in the root README.

3. **Pick one module namespace**
   - Either make a real top-level `stages` shim (if you want that API), or update all docs/tests to `src.phase1.stages`.

4. **Unify configuration hierarchy**
   - One explicit merge order (example):
     `config/*.yaml` defaults → env overrides → CLI overrides → validated dataclasses

5. **Reconcile docs “truth tables”**
   - Model count, MTF status, and phase completeness must match the code.
   - Everything else goes to `docs/archive/`.

---

## 7) P0/P1 Checklist (Quick triage)

If you want the repo to feel “institutional” quickly, these are the first few wins:
- Make `./pipeline --help` work on a fresh env (dependencies + import wiring).
- Fix CLI status commands imports (so “run → status → validate” works).
- Remove/replace `stages` references across docs/tests.
- Add a root `README.md` that matches `pyproject.toml` and points to *one* quickstart.

