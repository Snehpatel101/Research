# SNEH FIX PLAN
## Comprehensive ML Factory Codebase Remediation Plan

**Date:** 2026-01-13
**Version:** 1.0
**Scope:** Complete codebase analysis and fix recommendations
**Target:** Production-ready ML model factory for OHLCV time series

---

## TABLE OF CONTENTS

1. [Executive Summary](#executive-summary)
2. [Critical Issues (P0)](#critical-issues-p0)
3. [Major Architectural Issues (P1)](#major-architectural-issues-p1)
4. [Structural Issues (P2)](#structural-issues-p2)
5. [Detailed Fix Plans](#detailed-fix-plans)
6. [Implementation Sequence](#implementation-sequence)
7. [Acceptance Criteria](#acceptance-criteria)
8. [Appendix: Evidence Index](#appendix-evidence-index)

---

##EXECUTIVE SUMMARY

### What You're Building

An adaptive ML model factory for futures OHLCV data that can:
- Ingest canonical 1-min OHLCV data
- Produce 9 intraday timeframe datasets (1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h)
- Train 23 models across 4 families (boosting, classical, neural, ensemble/meta-learners)
- Support heterogeneous ensembles with per-model feature selection
- Maintain strict run isolation and leakage prevention
- Enable per-model MTF strategy selection (single-TF, MTF indicators, MTF ingestion)

### Current State

The factory "shape" exists with pipeline stages, model registry, and trainer, BUT:
- **Not runnable end-to-end**: Missing dependencies, broken imports, inconsistent configurations
- **Documentation contradicts reality**: Claims vs implementation mismatches
- **Configuration split-brain**: Multiple config systems not unified
- **CLI flags don't work**: Feature toggles, barrier overrides, scaler type ignored
- **Single timeframe only**: No multi-TF orchestration despite 9-TF claims
- **Path misalignment**: Training expects global paths, pipeline outputs run-scoped paths

### Critical Blockers (Must Fix First)

1. **Repo Hygiene (REPO-P0-001)**: Thousands of `__pycache__/` files tracked in git
2. **CLI Breakage (CLI-P0-001,002)**: Status commands have broken imports, CLI prints wrong examples
3. **Missing Entrypoint (RUN-P0-001)**: Training scripts reference doesn't exist in some commits
4. **Broken Imports (CLI-P0-001)**: `status_commands.py` tries to import non-existent modules

### Impact

Current state **blocks institutional use** because:
- Cannot run pipeline reliably
- Cannot trust documentation
- Cannot train models after pipeline completes
- Cannot reproduce runs
- Cannot configure behavior via CLI

---

## CRITICAL ISSUES (P0)

### REPO-P0-001: Generated Artifacts Tracked in Git

**Location:** Repository-wide
**Evidence:**
```bash
find . -name '__pycache__' | wc -l
# Returns: Thousands of directories
```

**Files Affected:**
- `src/**/__pycache__/` (all modules)
- `tests/**/__pycache__/`
- `.pytest_cache/`
- `.ruff_cache/`
- `.venv/` (entire virtual environment tracked)

**Impact:**
- 🔴 **CRITICAL**: Repo size bloated (hundreds of MB of bytecode)
- 🔴 **CRITICAL**: Stale bytecode can mask import errors
- 🔴 **CRITICAL**: New contributors confused by noise
- 🔴 **CRITICAL**: Code reviews impossible (thousands of irrelevant files)
- 🔴 **CRITICAL**: Git operations slow
- 🔴 **CRITICAL**: CI/CD pipelines will fail

**Root Cause:**
- `.gitignore` incomplete or not enforced
- Bytecode committed before `.gitignore` was fixed
- `.venv/` directory committed (should never be tracked)

**Fix Plan:**

**Step 1: Remove all tracked generated artifacts**
```bash
# Remove from git tracking (keep local copies)
git rm -r --cached **/__pycache__
git rm -r --cached .pytest_cache
git rm -r --cached .ruff_cache
git rm -r --cached .venv

# Remove from filesystem too (regenerate as needed)
find . -type d -name '__pycache__' -exec rm -rf {} +
find . -type f -name '*.pyc' -delete
rm -rf .pytest_cache .ruff_cache
```

**Step 2: Fix `.gitignore`**

**File:** `.gitignore:1-50`

Add/verify these entries:
```gitignore
# Python bytecode
__pycache__/
*.py[cod]
*$py.class
*.so

# Virtual environments
.venv/
venv/
ENV/
env/

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Linting
.ruff_cache/
.mypy_cache/
.dmypy.json

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
runs/*/
experiments/runs/*/
data/processed/
*.log
```

**Step 3: Commit cleanup**
```bash
git add .gitignore
git commit -m "chore: remove generated artifacts and fix .gitignore

- Remove all __pycache__/, *.pyc, .pytest_cache/, .ruff_cache/
- Remove .venv/ from tracking (never commit virtual environments)
- Update .gitignore to prevent reintroduction
- Repo size reduced by XXX MB

Issue: REPO-P0-001"
```

**Step 4: Verify**
```bash
# Should return 0
git ls-files | grep -E '__pycache__|\.pyc$|\.venv' | wc -l

# Should be clean
git status
```

**Acceptance Criteria:**
- ✅ No `__pycache__/` directories in `git ls-files`
- ✅ No `.pyc` files in `git ls-files`
- ✅ No `.venv/` in `git ls-files`
- ✅ `.gitignore` prevents reintroduction
- ✅ `git status` is clean after running tests
- ✅ CI passes without tracking generated files

**Time Estimate:** 30 minutes
**Risk:** LOW (safe cleanup, can always regenerate)

---

### CLI-P0-001: Broken Imports in status_commands.py

**Location:** `src/cli/status_commands.py:24,34`

**Evidence:**

**File:** `src/cli/status_commands.py:20-37`
```python
def _get_pipeline_config():
    """Lazy import pipeline_config module."""
    global _pipeline_config
    if _pipeline_config is None:
        from .. import pipeline_config  # ❌ BROKEN: module doesn't exist at src/

        _pipeline_config = pipeline_config
    return _pipeline_config


def _get_manifest():
    """Lazy import manifest module."""
    global _manifest
    if _manifest is None:
        from .. import manifest  # ❌ BROKEN: module doesn't exist at src/

        _manifest = manifest
    return _manifest
```

**Actual Module Locations:**
- `pipeline_config` → `src/phase1/pipeline_config.py:1`
- `manifest` → `src/common/manifest.py:1`

**Test to Reproduce:**
```bash
# This fails:
python -c "from src import pipeline_config"
# ImportError: cannot import name 'pipeline_config' from 'src'

# This also fails:
python -c "from src import manifest"
# ImportError: cannot import name 'manifest' from 'src'

# But this works:
python -c "from src.phase1 import pipeline_config; from src.common import manifest; print('OK')"
# OK
```

**When It Fails:**
```bash
./pipeline status 20241218_120000
```

**Error:**
```
Traceback (most recent call last):
  File "src/cli/status_commands.py", line 24, in _get_pipeline_config
    from .. import pipeline_config
ImportError: cannot import name 'pipeline_config' from 'src'
```

**Impact:**
- 🔴 **CRITICAL**: `pipeline status` command completely broken
- 🔴 **CRITICAL**: Cannot check run status after pipeline completes
- 🔴 **CRITICAL**: `pipeline validate` command also broken
- 🔴 **CRITICAL**: `pipeline list-runs` command broken
- 🔴 **CRITICAL**: CLI usability destroyed

**Root Cause:**
- Incomplete refactor: modules were moved from `src/` to `src/phase1/` and `src/common/`
- Lazy import functions not updated to match new module structure
- No `src/__init__.py` exports for `pipeline_config` or `manifest`

**Fix Plan:**

**File:** `src/cli/status_commands.py:20-37`

**Before:**
```python
def _get_pipeline_config():
    """Lazy import pipeline_config module."""
    global _pipeline_config
    if _pipeline_config is None:
        from .. import pipeline_config  # Line 24

        _pipeline_config = pipeline_config
    return _pipeline_config


def _get_manifest():
    """Lazy import manifest module."""
    global _manifest
    if _manifest is None:
        from .. import manifest  # Line 34

        _manifest = manifest
    return _manifest
```

**After:**
```python
def _get_pipeline_config():
    """Lazy import pipeline_config module."""
    global _pipeline_config
    if _pipeline_config is None:
        from ..phase1 import pipeline_config  # FIXED

        _pipeline_config = pipeline_config
    return _pipeline_config


def _get_manifest():
    """Lazy import manifest module."""
    global _manifest
    if _manifest is None:
        from ..common import manifest  # FIXED

        _manifest = manifest
    return _manifest
```

**Test After Fix:**
```bash
# Should work:
./pipeline status --help
./pipeline validate --help
./pipeline list-runs --help

# Should not crash (even if no runs exist):
./pipeline list-runs
```

**Acceptance Criteria:**
- ✅ `./pipeline status --help` works
- ✅ `./pipeline status <run_id>` works (if run exists)
- ✅ `./pipeline validate` works
- ✅ `./pipeline list-runs` works
- ✅ No ImportError on any status command
- ✅ All lazy imports resolve correctly

**Time Estimate:** 5 minutes
**Risk:** MINIMAL (simple import path fix)

---

### CLI-P0-002: CLI Prints Invalid Command Examples

**Location:** `src/cli/run_commands_pipeline.py:various`

**Evidence:**

**File:** `src/cli/run_commands_pipeline.py:227` (example output)
```python
console.print(f"\nPipeline run completed successfully!")
console.print(f"Run ID: {config.run_id}")
console.print(f"\nCheck status with:")
console.print(f"  [cyan]pipeline status --run-id {config.run_id}[/cyan]")  # ❌ WRONG
```

**Actual Command Signature:**

**File:** `src/cli/status_commands.py:40-43`
```python
def status_command(
    run_id: str = typer.Argument(..., help="Run ID to check"),  # POSITIONAL
    project_root: str | None = typer.Option(None, "--project-root", help="Project root directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information"),
) -> None:
```

**Contradiction:**
- **CLI Output Says:** `pipeline status --run-id 20241218_120000`
- **Actual Signature:** `pipeline status 20241218_120000` (positional argument)

**Test to Reproduce:**
```bash
# What CLI prints (doesn't work):
pipeline status --run-id 20241218_120000
# Error: No such option: --run-id

# What actually works:
pipeline status 20241218_120000
```

**Impact:**
- 🔴 **CRITICAL**: Users copy/paste commands that fail
- 🔴 **CRITICAL**: Trust in CLI destroyed
- 🔴 **CRITICAL**: Documentation contradicts implementation
- 🟡 **MEDIUM**: Wastes developer time debugging

**Root Cause:**
- Command signature changed from `--run-id` option to positional argument
- Success message not updated
- Help text inconsistent with actual parsing

**Fix Plan:**

**File:** `src/cli/run_commands_pipeline.py:227-230` (approximate)

Search for all instances of printed command examples:
```bash
grep -n "pipeline status --run-id" src/cli/run_commands_pipeline.py
```

**Before:**
```python
console.print(f"\nCheck status with:")
console.print(f"  [cyan]pipeline status --run-id {config.run_id}[/cyan]")
```

**After:**
```python
console.print(f"\nCheck status with:")
console.print(f"  [cyan]pipeline status {config.run_id}[/cyan]")
```

**Also Check These Files:**
- `src/cli/run_commands_pipeline.py` (all printed examples)
- `docs/guides/QUICK_START.md` (if exists)
- `README.md` (if exists)
- `CLAUDE.md:various` (command examples)

**Automated Fix:**
```bash
# Find all references
grep -rn "pipeline status --run-id" src/ docs/ *.md

# Replace in files (verify each one):
sed -i 's/pipeline status --run-id \([^ ]*\)/pipeline status \1/g' src/cli/run_commands_pipeline.py
```

**Test After Fix:**
```bash
# Run pipeline (should print correct command):
./pipeline run --symbols MES --help

# Verify printed command actually works:
# (extract run_id from output and test)
```

**Acceptance Criteria:**
- ✅ All CLI output examples are valid commands
- ✅ `status` command examples use positional arg syntax
- ✅ `validate` command examples match signature
- ✅ Help text matches actual command parsing
- ✅ Documentation matches CLI behavior

**Time Estimate:** 15 minutes
**Risk:** MINIMAL (text changes only)

---

### RUN-P0-001: Training Scripts Missing

**Location:** `scripts/` directory deleted in commit `2a4f884`

**Evidence:**

**Git History:**
```bash
# Scripts existed in previous commit:
git show d757a7c:scripts/train_model.py | head -20
# (file content visible)

# Scripts deleted in current commit:
git ls-tree HEAD scripts/
# (empty or doesn't exist)

# But documentation still references them:
grep -r "scripts/train_model.py" docs/ CLAUDE.md
# Multiple references found
```

**Current Repository State:**
```bash
ls scripts/
# Returns files (scripts directory was restored)
```

**Documentation References:**

**File:** `CLAUDE.md:248` (example)
```markdown
# Train specific model (Phase 6)
python scripts/train_model.py --model xgboost --horizon 20
python scripts/train_model.py --model lstm --horizon 20 --seq-len 30
```

**File:** `docs/guides/MODEL_INTEGRATION.md:various`
```markdown
To train a model after running the pipeline:
```bash
python scripts/train_model.py --model xgboost --data-dir runs/{run_id}/data/splits/scaled
```
```

**File:** `config/models/README.md:various`
```markdown
Use `scripts/train_model.py` to train models with these configurations.
```

**Impact:**
- 🔴 **CRITICAL**: No official training entrypoint after pipeline completes
- 🔴 **CRITICAL**: Documentation dead-ends at "how to train"
- 🔴 **CRITICAL**: Users cannot follow end-to-end workflow
- 🟡 **MEDIUM**: Phase 1 → Phase 2 handoff broken

**Root Cause:**
- Scripts directory deleted (intentional or accidental in commit `2a4f884`)
- Documentation not updated to reflect removal
- No alternative training entrypoint documented

**Current Workaround Status:**
- ✅ Scripts directory has been restored from commit `d757a7c`
- ⚠️ Risk of deletion in future commits

**Fix Plan:**

**Option A: Keep Scripts (Recommended)**

**Step 1: Commit restored scripts permanently**
```bash
# Verify scripts exist:
ls -la scripts/*.py

# Add to git:
git add scripts/
git commit -m "fix: restore training scripts for Phase 2+ workflow

- Restore scripts/ directory from commit d757a7c
- Essential for post-pipeline model training
- Referenced throughout documentation
- Required for end-to-end workflow

Scripts restored:
- train_model.py
- train_ensemble.py
- run_cv.py
- serve_model.py
- (list all)

Issue: RUN-P0-001"
```

**Step 2: Add protection to prevent future deletion**

**File:** `.github/CODEOWNERS` (create if doesn't exist)
```
# Protect critical training entrypoints
/scripts/*.py @repository-owner
```

**File:** `scripts/README.md` (create)
```markdown
# Training Scripts

**⚠️ DO NOT DELETE THIS DIRECTORY**

These scripts are the official training entrypoints for Phase 2+.
They are referenced throughout the documentation and required for
the end-to-end ML factory workflow.

## Scripts

- `train_model.py` - Train single models
- `train_ensemble.py` - Train ensemble models
- `run_cv.py` - Cross-validation
- `serve_model.py` - Model serving

## Usage

See `docs/guides/MODEL_INTEGRATION.md` for details.
```

**Option B: Replace with CLI Subcommands**

If scripts should be removed, create CLI alternatives:

**File:** `src/cli/train_commands.py` (new file)
```python
"""Training commands for Phase 2+."""

import typer

app = typer.Typer(name="train", help="Model training commands")

@app.command(name="model")
def train_model_command(
    model: str = typer.Argument(..., help="Model type to train"),
    horizon: int = typer.Option(20, "--horizon", help="Label horizon"),
    run_id: str = typer.Option(None, "--run-id", help="Pipeline run ID"),
    # ... other options
) -> None:
    """Train a single model."""
    # Implementation
    pass

@app.command(name="ensemble")
def train_ensemble_command(
    # ... options
) -> None:
    """Train an ensemble model."""
    pass
```

**File:** `src/cli/__init__.py:existing`

Add:
```python
from .train_commands import app as train_app

# In main app registration:
app.add_typer(train_app, name="train")
```

**Then update all documentation:**
```bash
# Replace:
python scripts/train_model.py --model xgboost

# With:
pipeline train model xgboost
```

**Recommendation:** Choose **Option A** (keep scripts) because:
- ✅ Already restored and working
- ✅ Less refactoring required
- ✅ Matches documentation as-is
- ✅ Familiar pattern for ML practitioners

**Acceptance Criteria:**
- ✅ `scripts/` directory exists and is committed
- ✅ `python scripts/train_model.py --help` works
- ✅ Scripts protected from accidental deletion
- ✅ README exists in `scripts/` directory
- ✅ Documentation references are valid
- ✅ End-to-end workflow (pipeline → train) works

**Time Estimate:** 30 minutes (Option A), 4-6 hours (Option B)
**Risk:** LOW (Option A), MEDIUM (Option B requires refactor)

---

### PACK-P0-001: Missing Root README.md

**Location:** Repository root (file missing)

**Evidence:**

**File:** `pyproject.toml:8`
```toml
[project]
name = "ensemble-price-prediction"
version = "0.1.0"
readme = "README.md"  # ❌ File doesn't exist
```

**Test:**
```bash
ls -la README.md
# ls: cannot access 'README.md': No such file or directory

cat pyproject.toml | grep readme
# readme = "README.md"
```

**Impact:**
- 🔴 **CRITICAL**: Packaging metadata broken
- 🔴 **CRITICAL**: `pip install -e .` may fail or show no description
- 🔴 **CRITICAL**: PyPI upload will fail
- 🔴 **CRITICAL**: No entry point for new contributors
- 🟡 **MEDIUM**: GitHub repo looks incomplete

**Root Cause:**
- README deleted or never created
- `pyproject.toml` not updated to match
- No canonical quickstart document

**Fix Plan:**

**Option A: Create README.md**

**File:** `README.md` (new file at root)
```markdown
# ML Model Factory for OHLCV Time Series

Production-ready model factory for training ML models on futures OHLCV data.

## Features

- ✅ Single 1-min canonical OHLCV source
- ✅ 9 intraday timeframe ladder (1m → 1h)
- ✅ 23 models across 4 families
- ✅ Heterogeneous ensemble support
- ✅ Strict leakage prevention
- ✅ Per-model feature selection
- ✅ Run isolation and reproducibility

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Prepare Data

Place your 1-minute OHLCV data in `data/raw/`:
- `data/raw/MES_1m.parquet` (or `.csv`)
- `data/raw/MGC_1m.parquet`

### 3. Run Pipeline (Phase 1)

```bash
# Basic usage
./pipeline run --symbols MES --start 2020-01-01 --end 2024-12-31

# With configuration
./pipeline run --symbols MES --preset day_trading --feature-set boosting_optimal
```

### 4. Train Models (Phase 2)

```bash
# Train single model
python scripts/train_model.py --model xgboost --horizon 20 --run-id <run_id>

# Train ensemble
python scripts/train_ensemble.py --models xgboost,lstm,tcn --meta-learner ridge_meta
```

### 5. Cross-Validation (Phase 3)

```bash
# Run CV
python scripts/run_cv.py --models xgboost,lightgbm --horizons 5,10,15,20 --n-splits 5
```

## Documentation

- **Architecture:** `docs/ARCHITECTURE.md`
- **Pipeline Stages:** `docs/reference/PIPELINE_STAGES.md`
- **Model Integration:** `docs/guides/MODEL_INTEGRATION.md`
- **Feature Engineering:** `docs/guides/FEATURE_ENGINEERING.md`
- **Troubleshooting:** `docs/troubleshooting/`

## Project Structure

```
├── src/                  # Source code
│   ├── phase1/          # Data pipeline (ingest → features → labels → splits)
│   ├── models/          # Model registry (23 models)
│   ├── cross_validation/# CV and OOF utilities
│   └── inference/       # Model serving
├── scripts/             # Training entrypoints
├── config/              # YAML configurations
├── docs/                # Documentation
├── tests/               # Test suite
└── data/                # Data directories (gitignored)
```

## Configuration

The factory uses a layered configuration system:
1. Code defaults (safe baselines)
2. YAML configs (`config/*.yaml`)
3. CLI overrides

**Primary Training Timeframe:** Configurable per model (1m, 5m, 10m, 15m, etc.)
**MTF Strategy:** Configurable per model (single-TF, MTF indicators, MTF ingestion)
**Feature Selection:** Per-model feature sets (boosting_optimal, neural_optimal, etc.)

## Models

### Tabular (6 models)
- XGBoost, LightGBM, CatBoost
- Random Forest, Logistic Regression, SVM

### Neural (10 models)
- LSTM, GRU, TCN
- Transformer, PatchTST, iTransformer
- TFT, N-BEATS, InceptionTime, ResNet1D

### Ensemble (3 models)
- Voting, Stacking, Blending

### Meta-Learners (4 models)
- Ridge Meta, MLP Meta, Calibrated Meta, XGBoost Meta

## Requirements

- Python 3.10+
- 16GB+ RAM recommended
- GPU optional (for neural models)

## License

[Your License]

## Contributing

See `CONTRIBUTING.md` for guidelines.

## Citation

If you use this codebase in your research, please cite:

```bibtex
@software{ml_factory_2025,
  author = {Your Team},
  title = {ML Model Factory for OHLCV Time Series},
  year = {2025},
  url = {https://github.com/your-repo}
}
```
```

**Option B: Update pyproject.toml**

If README should not exist:

**File:** `pyproject.toml:8`

**Before:**
```toml
readme = "README.md"
```

**After:**
```toml
readme = "CLAUDE.md"
# Or remove the line entirely if no readme
```

**Recommendation:** Choose **Option A** (create README.md) because:
- ✅ Standard practice for all repos
- ✅ Entry point for contributors
- ✅ Required for packaging
- ✅ Matches `pyproject.toml` expectation

**Acceptance Criteria:**
- ✅ `README.md` exists at root
- ✅ `pip install -e .` succeeds
- ✅ `pip show ensemble-price-prediction` shows description
- ✅ README matches current capabilities
- ✅ Quickstart is valid (commands work)

**Time Estimate:** 1 hour
**Risk:** LOW (documentation only)

---

## MAJOR ARCHITECTURAL ISSUES (P1)

### TF-P1-001: Single Timeframe Per Run (No Multi-TF Orchestration)

**Location:** `src/phase1/pipeline_config.py:54`, `src/phase1/stages/clean/run.py:84`

**Documentation Claim:**

**File:** `CLAUDE.md:18-28`
```markdown
**Data Flow:**
```
Raw 1-min OHLCV (canonical - single source of truth)
  ↓
[MTF Upscaling] → ✅ 9 intraday timeframes (1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h)
  Full 9-TF ladder available: 1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h
```
```

**File:** `docs/README.md:various`
```markdown
MTF: ✅ Complete (9 intraday timeframes: 1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h)
```

**Reality:**

**File:** `src/phase1/pipeline_config.py:54`
```python
# Timeframe configuration
target_timeframe: str = "5min"  # ❌ SINGLE TIMEFRAME ONLY
bar_resolution: str = field(default=None)  # Legacy alias
```

**File:** `src/phase1/stages/clean/run.py:84`
```python
# Build output filename based on target timeframe
output_path = config.clean_data_dir / f"{symbol}_{target_timeframe}_clean.parquet"
# ❌ ONLY ONE OUTPUT FILE PER SYMBOL
```

**Evidence:**

**Test:**
```bash
# Run pipeline:
./pipeline run --symbols MES --timeframe 5min

# Check outputs:
ls -la runs/*/data/clean/
# Only shows: MES_5min_clean.parquet
# NOT: MES_1min_clean.parquet, MES_10min_clean.parquet, etc.
```

**Multi-TF Function Exists But Not Used:**

**File:** `src/phase1/stages/clean/pipeline.py:existing`
```python
def clean_symbol_data_multi_timeframe(
    input_path: Path,
    output_dir: Path,
    symbol: str,
    timeframes: list[str],  # ✅ SUPPORTS MULTIPLE TIMEFRAMES
    **kwargs
) -> dict[str, Path]:
    """Clean data and output multiple timeframes."""
    # Implementation exists!
    pass
```

**But Stage 2 Uses Single-TF Function:**

**File:** `src/phase1/stages/clean/run.py:88-96`
```python
# Use clean_symbol_data with full configuration
clean_symbol_data(  # ❌ SINGLE-TF FUNCTION
    input_path=input_path,
    output_path=output_path,
    symbol=symbol,
    target_timeframe=target_timeframe,
    include_timeframe_metadata=True,
    max_gap_minutes=max_gap_minutes,
)
```

**Impact:**
- 🔴 **CRITICAL**: Cannot "automatically produce nine datasets" claim is FALSE
- 🔴 **CRITICAL**: Must run pipeline 9 times manually for 9 timeframes
- 🔴 **CRITICAL**: Blocks heterogeneous ensemble use case (models on different TFs)
- 🔴 **CRITICAL**: Documentation completely misrepresents capabilities
- 🟡 **MEDIUM**: Per-model timeframe selection impossible

**Root Cause:**
- Multi-TF infrastructure exists but not orchestrated
- Stage 2 wired to single-TF path
- PipelineConfig designed for single TF
- Documentation written for target state, not current state

**Fix Plan:**

**Decision Point:** Choose multi-TF strategy:
- **Option A:** Materialize all 9 TFs per run (disk space trade-off)
- **Option B:** Dataset "views" (on-demand resampling)
- **Option C:** Configurable TF list per run (flexible middle ground)

**Recommended: Option C (Configurable TF List)**

**Step 1: Update PipelineConfig**

**File:** `src/phase1/pipeline_config.py:54-55`

**Before:**
```python
# Timeframe configuration
target_timeframe: str = "5min"
bar_resolution: str = field(default=None)  # Legacy alias
```

**After:**
```python
# Timeframe configuration
target_timeframes: list[str] = field(default_factory=lambda: ["5min"])  # NEW: Support multiple
target_timeframe: str = field(default="5min")  # DEPRECATED: For backward compat
bar_resolution: str = field(default=None)  # Legacy alias

def __post_init__(self):
    # ... existing validation ...

    # Handle target_timeframe → target_timeframes migration
    if self.target_timeframe and len(self.target_timeframes) == 1:
        # If only default, use target_timeframe value
        self.target_timeframes = [self.target_timeframe]
```

**Step 2: Update Stage 2 to Use Multi-TF Function**

**File:** `src/phase1/stages/clean/run.py:70-110`

**Before:**
```python
for symbol in config.symbols:
    # ... input path resolution ...

    output_path = config.clean_data_dir / f"{symbol}_{target_timeframe}_clean.parquet"

    clean_symbol_data(
        input_path=input_path,
        output_path=output_path,
        symbol=symbol,
        target_timeframe=target_timeframe,
        include_timeframe_metadata=True,
        max_gap_minutes=max_gap_minutes,
    )
```

**After:**
```python
for symbol in config.symbols:
    # ... input path resolution ...

    # Use multi-timeframe cleaning
    timeframes = config.target_timeframes
    output_paths = clean_symbol_data_multi_timeframe(
        input_path=input_path,
        output_dir=config.clean_data_dir,
        symbol=symbol,
        timeframes=timeframes,
        include_timeframe_metadata=True,
        max_gap_minutes=max_gap_minutes,
    )

    # Register all outputs with manifest
    for tf, output_path in output_paths.items():
        artifacts.append(output_path)
        cleaning_metadata[f"{symbol}_{tf}"] = {
            "symbol": symbol,
            "timeframe": tf,
            "rows": len(pd.read_parquet(output_path)),
        }
```

**Step 3: Update CLI to Accept Multiple Timeframes**

**File:** `src/cli/run_commands_pipeline.py:32-34`

**Before:**
```python
timeframe: str | None = typer.Option(
    None, "--timeframe", "-t", help="Target timeframe for resampling (e.g., 1min, 5min, 15min)"
),
```

**After:**
```python
timeframes: str | None = typer.Option(
    None,
    "--timeframes",
    "-t",
    help="Comma-separated target timeframes (e.g., '1min,5min,15min' or '9tf' for full ladder)"
),
```

**Step 4: Add CLI Parsing for Timeframes**

**File:** `src/cli/run_commands_core.py:_create_config_from_args`

```python
# Handle timeframes
if timeframes:
    if timeframes.lower() == "9tf":
        # Shorthand for full ladder
        config_data["target_timeframes"] = [
            "1min", "5min", "10min", "15min", "20min", "25min", "30min", "45min", "1h"
        ]
    else:
        # Parse comma-separated list
        config_data["target_timeframes"] = [tf.strip() for tf in timeframes.split(",")]
```

**Step 5: Update Downstream Stages**

All stages after Stage 2 must iterate over timeframes:

**Pattern for Stage 3+ (Example: Feature Engineering):**

**File:** `src/phase1/stages/features/run.py:90-100`

**Before:**
```python
for symbol in config.symbols:
    input_file = config.clean_data_dir / f"{symbol}_{target_timeframe}_clean.parquet"
    output_file = config.features_dir / f"{symbol}_{target_timeframe}_features.parquet"
    # ... process ...
```

**After:**
```python
for symbol in config.symbols:
    for timeframe in config.target_timeframes:
        input_file = config.clean_data_dir / f"{symbol}_{timeframe}_clean.parquet"
        output_file = config.features_dir / f"{symbol}_{timeframe}_features.parquet"

        if not input_file.exists():
            logger.warning(f"Skipping {symbol}@{timeframe}: input not found")
            continue

        # ... process ...
```

**Apply to ALL stages:**
- ✅ Stage 3: Feature Engineering
- ✅ Stage 4: Initial Labeling
- ✅ Stage 5: GA Optimize (once per symbol, not per TF)
- ✅ Stage 6: Final Labels
- ✅ Stage 7: Create Splits
- ✅ Stage 7.5: Scaling
- ✅ Stage 7.6: Build Datasets
- ✅ Stage 7.7: Validate Scaled
- ✅ Stage 8: Validation
- ✅ Stage 9: Reporting

**Step 6: Add Timeframe-Aware Horizon Scaling**

**File:** `src/common/horizon_config.py:existing`

Add function:
```python
def scale_horizon_for_timeframe(
    horizon_bars: int,
    from_timeframe: str,
    to_timeframe: str
) -> int:
    """
    Scale horizon from one timeframe to another.

    Example:
        horizon=20 bars at 5min = 100 minutes
        At 15min: 100 / 15 = 6.67 → 7 bars
    """
    from_minutes = parse_timeframe_to_minutes(from_timeframe)
    to_minutes = parse_timeframe_to_minutes(to_timeframe)

    time_minutes = horizon_bars * from_minutes
    scaled_bars = int(np.ceil(time_minutes / to_minutes))

    return scaled_bars
```

**Step 7: Update Labeling to Use Time-Consistent Horizons**

**File:** `src/phase1/stages/labeling/run.py:existing`

```python
# For each timeframe, scale horizons
for timeframe in config.target_timeframes:
    # Convert config horizons (defined for base TF) to this timeframe
    base_tf = "5min"  # Or config.base_timeframe
    scaled_horizons = [
        scale_horizon_for_timeframe(h, from_timeframe=base_tf, to_timeframe=timeframe)
        for h in config.label_horizons
    ]

    logger.info(f"Labeling {timeframe}: horizons={scaled_horizons}")
    # ... label with scaled horizons ...
```

**Acceptance Criteria:**
- ✅ `./pipeline run --timeframes 9tf` produces 9 TF outputs
- ✅ `./pipeline run --timeframes 5min,15min,1h` produces 3 TF outputs
- ✅ All stages support multi-TF processing
- ✅ Horizons scaled consistently across TFs (time-based)
- ✅ Manifest enumerates all TF outputs
- ✅ Training can select specific TF dataset

**Time Estimate:** 8-12 hours
**Risk:** MEDIUM (touches many stages, requires careful testing)

---

### TF-P1-002: Timeframe Vocabulary Inconsistency

**Location:** Multiple files with conflicting timeframe definitions

**Evidence:**

**File 1:** `src/phase1/config/features.py:17`
```python
SUPPORTED_TIMEFRAMES = ["1min", "5min", "10min", "15min", "20min", "30min", "45min", "60min"]
# ❌ Missing "25min", Missing "1h" alias
```

**File 2:** `src/phase1/stages/mtf/constants.py:20-42`
```python
MTF_TIMEFRAMES = {
    # Base timeframe
    "1min": 1,
    "5min": 5,
    # Short-term MTF (9-timeframe ladder)
    "10min": 10,
    "15min": 15,
    "20min": 20,
    "25min": 25,  # ✅ INCLUDED
    "30min": 30,
    "45min": 45,
    # Hourly
    "60min": 60,
    "1h": 60,  # ✅ Alias for 60min
    # ...
}
```

**File 3:** `src/phase1/config/features.py:245` (MTF_CONFIG)
```python
MTF_CONFIG = {
    # ...
    "mtf_timeframes": ["5min", "10min", "15min", "30min", "45min", "60min"],
    # ❌ Missing "1min", "20min", "25min", "1h"
}
```

**Validation Conflicts:**

**Test Case 1:** Validate "25min"
```python
from src.phase1.config.features import validate_timeframe

validate_timeframe("25min")
# ❌ Raises ValueError: Unsupported timeframe: '25min'
```

**Test Case 2:** Validate "1h"
```python
from src.phase1.config.features import validate_timeframe

validate_timeframe("1h")
# ❌ Raises ValueError: Unsupported timeframe: '1h'
```

**But MTF Constants Supports Both:**
```python
from src.phase1.stages.mtf.constants import MTF_TIMEFRAMES

"25min" in MTF_TIMEFRAMES  # True
"1h" in MTF_TIMEFRAMES  # True
```

**Impact:**
- 🔴 **CRITICAL**: Cannot use "25min" despite docs claiming 9-TF ladder
- 🔴 **CRITICAL**: Cannot use "1h" alias (must use "60min")
- 🔴 **CRITICAL**: Validation rejects valid timeframes
- 🟡 **MEDIUM**: Confusion about supported timeframes
- 🟡 **MEDIUM**: Different modules have different truth

**Root Cause:**
- Multiple timeframe registries not synchronized
- No single source of truth for timeframe vocabulary
- Validation functions use incomplete lists

**Fix Plan:**

**Step 1: Create Canonical Timeframe Registry**

**File:** `src/common/timeframes.py` (new file)
```python
"""
Canonical Timeframe Registry

Single source of truth for all supported timeframes and aliases.
"""

from enum import Enum
from typing import Dict, List

# Full 9-timeframe intraday ladder
INTRADAY_TIMEFRAMES = [
    "1min",
    "5min",
    "10min",
    "15min",
    "20min",
    "25min",
    "30min",
    "45min",
    "1h",
]

# Extended timeframes (daily, etc.)
EXTENDED_TIMEFRAMES = [
    "4h",
    "daily",
]

# All supported timeframes
ALL_TIMEFRAMES = INTRADAY_TIMEFRAMES + EXTENDED_TIMEFRAMES

# Timeframe to minutes mapping
TIMEFRAME_TO_MINUTES: Dict[str, int] = {
    # Intraday
    "1min": 1,
    "5min": 5,
    "10min": 10,
    "15min": 15,
    "20min": 20,
    "25min": 25,
    "30min": 30,
    "45min": 45,
    "60min": 60,
    "1h": 60,  # Alias
    # Extended
    "4h": 240,
    "240min": 240,  # Alias
    "daily": 1440,
    "1d": 1440,  # Alias
    "D": 1440,  # Pandas convention
}

# Reverse mapping (minutes to canonical timeframe)
MINUTES_TO_TIMEFRAME: Dict[int, str] = {
    1: "1min",
    5: "5min",
    10: "10min",
    15: "15min",
    20: "20min",
    25: "25min",
    30: "30min",
    45: "45min",
    60: "1h",  # Use "1h" as canonical for 60 minutes
    240: "4h",
    1440: "daily",
}

# Pandas frequency strings
TIMEFRAME_TO_PANDAS_FREQ: Dict[str, str] = {
    "1min": "1min",
    "5min": "5min",
    "10min": "10min",
    "15min": "15min",
    "20min": "20min",
    "25min": "25min",
    "30min": "30min",
    "45min": "45min",
    "60min": "60min",
    "1h": "1h",
    "4h": "4h",
    "240min": "4h",
    "daily": "D",
    "1d": "D",
    "D": "D",
}


def validate_timeframe(timeframe: str) -> None:
    """
    Validate that a timeframe string is supported.

    Raises ValueError if not supported.
    """
    if timeframe not in TIMEFRAME_TO_MINUTES:
        raise ValueError(
            f"Unsupported timeframe: '{timeframe}'. "
            f"Supported: {list(TIMEFRAME_TO_MINUTES.keys())}"
        )


def parse_timeframe_to_minutes(timeframe: str) -> int:
    """Convert timeframe string to minutes."""
    validate_timeframe(timeframe)
    return TIMEFRAME_TO_MINUTES[timeframe]


def get_canonical_timeframe(timeframe: str) -> str:
    """
    Get canonical timeframe name (resolves aliases).

    Example:
        "60min" → "1h"
        "1d" → "daily"
    """
    minutes = parse_timeframe_to_minutes(timeframe)
    return MINUTES_TO_TIMEFRAME[minutes]


def get_pandas_freq(timeframe: str) -> str:
    """Get pandas frequency string for resampling."""
    validate_timeframe(timeframe)
    return TIMEFRAME_TO_PANDAS_FREQ[timeframe]
```

**Step 2: Update All Modules to Use Canonical Registry**

**File:** `src/phase1/config/features.py:17-30`

**Before:**
```python
SUPPORTED_TIMEFRAMES = ["1min", "5min", "10min", "15min", "20min", "30min", "45min", "60min"]

TIMEFRAME_TO_FREQ = {
    "1min": "1min",
    # ...
}

def validate_timeframe(timeframe: str) -> None:
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(...)
```

**After:**
```python
# Import from canonical registry
from src.common.timeframes import (
    ALL_TIMEFRAMES as SUPPORTED_TIMEFRAMES,
    TIMEFRAME_TO_PANDAS_FREQ as TIMEFRAME_TO_FREQ,
    validate_timeframe,
    parse_timeframe_to_minutes,
)

# Deprecated - remove in next version
# (keep for backward compat if needed, but point to canonical)
```

**File:** `src/phase1/stages/mtf/constants.py:20-42`

**Before:**
```python
MTF_TIMEFRAMES = {
    "1min": 1,
    "5min": 5,
    # ...
}
```

**After:**
```python
# Import from canonical registry
from src.common.timeframes import TIMEFRAME_TO_MINUTES as MTF_TIMEFRAMES

# Deprecated - remove in next version
```

**File:** `src/phase1/config/features.py:245` (MTF_CONFIG)

**Before:**
```python
MTF_CONFIG = {
    # ...
    "mtf_timeframes": ["5min", "10min", "15min", "30min", "45min", "60min"],
}
```

**After:**
```python
from src.common.timeframes import INTRADAY_TIMEFRAMES

MTF_CONFIG = {
    # ...
    "mtf_timeframes": INTRADAY_TIMEFRAMES[1:],  # Exclude 1min base
    # Or explicitly: ["5min", "10min", "15min", "20min", "25min", "30min", "45min", "1h"]
}
```

**Step 3: Update Validation in PipelineConfig**

**File:** `src/phase1/pipeline_config.py:134-149`

**Before:**
```python
def __post_init__(self):
    from src.phase1.config import validate_timeframe
    # ...
    validate_timeframe(self.target_timeframe)

    # Validate MTF configuration
    from src.phase1.stages.mtf.constants import MTF_TIMEFRAMES
    for tf in self.mtf_timeframes:
        if tf not in MTF_TIMEFRAMES:
            raise ValueError(...)
```

**After:**
```python
def __post_init__(self):
    from src.common.timeframes import validate_timeframe
    # ...
    validate_timeframe(self.target_timeframe)

    # Validate MTF timeframes
    for tf in self.mtf_timeframes:
        try:
            validate_timeframe(tf)
        except ValueError as e:
            raise ValueError(f"Invalid MTF timeframe: {e}")
```

**Step 4: Add Unit Tests**

**File:** `tests/common/test_timeframes.py` (new file)
```python
"""Unit tests for canonical timeframe registry."""

import pytest
from src.common.timeframes import (
    validate_timeframe,
    parse_timeframe_to_minutes,
    get_canonical_timeframe,
    get_pandas_freq,
)


def test_validate_all_intraday_timeframes():
    """All 9 intraday timeframes should validate."""
    for tf in ["1min", "5min", "10min", "15min", "20min", "25min", "30min", "45min", "1h"]:
        validate_timeframe(tf)  # Should not raise


def test_validate_60min_alias():
    """60min is alias for 1h."""
    validate_timeframe("60min")
    assert parse_timeframe_to_minutes("60min") == 60


def test_validate_1h():
    """1h should validate."""
    validate_timeframe("1h")
    assert parse_timeframe_to_minutes("1h") == 60


def test_validate_25min():
    """25min should validate (part of 9-TF ladder)."""
    validate_timeframe("25min")
    assert parse_timeframe_to_minutes("25min") == 25


def test_canonical_timeframe():
    """Test canonical timeframe resolution."""
    assert get_canonical_timeframe("60min") == "1h"
    assert get_canonical_timeframe("1h") == "1h"
    assert get_canonical_timeframe("1d") == "daily"


def test_invalid_timeframe():
    """Invalid timeframes should raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        validate_timeframe("invalid")


def test_pandas_freq():
    """Test pandas frequency string conversion."""
    assert get_pandas_freq("1min") == "1min"
    assert get_pandas_freq("1h") == "1h"
    assert get_pandas_freq("daily") == "D"
```

**Run tests:**
```bash
pytest tests/common/test_timeframes.py -v
```

**Acceptance Criteria:**
- ✅ Single source of truth in `src/common/timeframes.py`
- ✅ All modules import from canonical registry
- ✅ "25min" validates successfully
- ✅ "1h" validates successfully
- ✅ Aliases ("60min", "1d", "D") work correctly
- ✅ Unit tests pass
- ✅ No hardcoded timeframe lists in modules

**Time Estimate:** 2-3 hours
**Risk:** LOW (mostly refactoring, testable)

---

### CFG-P1-001: CLI Feature Toggles Ignored

**Location:** `src/phase1/stages/features/run.py:73-84`, `src/cli/run_commands_pipeline.py:79-94`

**Documentation Claim:**

**File:** `CLAUDE.md:248`
```markdown
# Feature toggles
pipeline run --symbols MES --enable-wavelets --disable-microstructure
```

**CLI Accepts Feature Toggles:**

**File:** `src/cli/run_commands_pipeline.py:79-94`
```python
# Feature toggles
enable_wavelets: bool | None = typer.Option(
    None,
    "--enable-wavelets/--disable-wavelets",
    help="Enable/disable wavelet decomposition features",
),
enable_microstructure: bool | None = typer.Option(
    None,
    "--enable-microstructure/--disable-microstructure",
    help="Enable/disable microstructure features (bid-ask, order flow)",
),
enable_volume: bool | None = typer.Option(
    None, "--enable-volume/--disable-volume", help="Enable/disable volume-based features"
),
enable_volatility: bool | None = typer.Option(
    None, "--enable-volatility/--disable-volatility", help="Enable/disable volatility features"
),
```

**CLI Stores Toggles in Config:**

**File:** `src/cli/run_commands_core.py:_create_config_from_args`
```python
# Build feature toggles dict
feature_toggles = {}
if enable_wavelets is not None:
    feature_toggles["wavelets"] = enable_wavelets
if enable_microstructure is not None:
    feature_toggles["microstructure"] = enable_microstructure
# ... etc

config_data["feature_toggles"] = feature_toggles
```

**PipelineConfig Has feature_toggles Field:**

**File:** `src/phase1/pipeline_config.py:103`
```python
# Optional configurations
feature_toggles: dict[str, bool] | None = None  # ✅ Field exists
```

**But Feature Engineering Stage IGNORES IT:**

**File:** `src/phase1/stages/features/run.py:73-84`
```python
# Initialize FeatureEngineer from modular implementation
# MTF settings come from PipelineConfig, not global MTF_CONFIG
engineer = FeatureEngineer(
    input_dir=config.clean_data_dir,
    output_dir=config.features_dir,
    timeframe=target_timeframe,
    enable_mtf=bool(mtf_timeframes),
    mtf_timeframes=mtf_timeframes,
    mtf_include_ohlcv=mtf_include_ohlcv,
    mtf_include_indicators=mtf_include_indicators,
    base_timeframe=target_timeframe,
    # ❌ NO feature_toggles PARAMETER
)
```

**FeatureEngineer Constructor:**

**File:** `src/phase1/stages/features/engineer.py:__init__` (approximate)
```python
def __init__(
    self,
    input_dir: Path,
    output_dir: Path,
    timeframe: str,
    enable_mtf: bool = True,
    # ... other params ...
    # ❌ NO feature_toggles PARAMETER
):
    # Hardcoded enables:
    self.enable_wavelets = True  # Always enabled
    self.enable_microstructure = True  # Always enabled
    self.enable_volume = True  # Always enabled
    self.enable_volatility = True  # Always enabled
```

**Test to Reproduce:**

```bash
# Run with wavelets disabled:
./pipeline run --symbols MES --disable-wavelets

# Check features output:
python -c "
import pandas as pd
df = pd.read_parquet('runs/*/data/features/MES_5min_features.parquet')
wavelet_cols = [c for c in df.columns if 'wavelet' in c.lower()]
print(f'Wavelet columns: {len(wavelet_cols)}')
"
# Output: Wavelet columns: 15 (or similar non-zero number)
# ❌ Expected: 0 (wavelets should be disabled)
```

**Impact:**
- 🔴 **CRITICAL**: CLI flags have NO EFFECT on behavior
- 🔴 **CRITICAL**: "Configurable factory" claim is FALSE
- 🔴 **CRITICAL**: Cannot disable expensive wavelet computation
- 🔴 **CRITICAL**: Cannot customize feature sets via CLI
- 🟡 **MEDIUM**: Users waste time trying to configure
- 🟡 **MEDIUM**: Cannot reproduce minimal feature runs

**Root Cause:**
- CLI collects toggles but doesn't pass them to feature engineer
- FeatureEngineer hardcodes feature enables
- No plumbing from config → stage → engineer

**Fix Plan:**

**Step 1: Update FeatureEngineer to Accept Toggles**

**File:** `src/phase1/stages/features/engineer.py:__init__`

**Before:**
```python
def __init__(
    self,
    input_dir: Path,
    output_dir: Path,
    timeframe: str,
    enable_mtf: bool = True,
    mtf_timeframes: list[str] | None = None,
    # ...
):
    # Hardcoded
    self.enable_wavelets = True
    self.enable_microstructure = True
    self.enable_volume = True
    self.enable_volatility = True
```

**After:**
```python
def __init__(
    self,
    input_dir: Path,
    output_dir: Path,
    timeframe: str,
    enable_mtf: bool = True,
    mtf_timeframes: list[str] | None = None,
    # NEW: Accept feature toggles
    feature_toggles: dict[str, bool] | None = None,
    # ...
):
    # Apply toggles or use defaults
    toggles = feature_toggles or {}
    self.enable_wavelets = toggles.get("wavelets", True)
    self.enable_microstructure = toggles.get("microstructure", True)
    self.enable_volume = toggles.get("volume", True)
    self.enable_volatility = toggles.get("volatility", True)
```

**Step 2: Pass Toggles in Stage 3**

**File:** `src/phase1/stages/features/run.py:73-84`

**Before:**
```python
engineer = FeatureEngineer(
    input_dir=config.clean_data_dir,
    output_dir=config.features_dir,
    timeframe=target_timeframe,
    enable_mtf=bool(mtf_timeframes),
    mtf_timeframes=mtf_timeframes,
    mtf_include_ohlcv=mtf_include_ohlcv,
    mtf_include_indicators=mtf_include_indicators,
    base_timeframe=target_timeframe,
)
```

**After:**
```python
engineer = FeatureEngineer(
    input_dir=config.clean_data_dir,
    output_dir=config.features_dir,
    timeframe=target_timeframe,
    enable_mtf=bool(mtf_timeframes),
    mtf_timeframes=mtf_timeframes,
    mtf_include_ohlcv=mtf_include_ohlcv,
    mtf_include_indicators=mtf_include_indicators,
    base_timeframe=target_timeframe,
    feature_toggles=config.feature_toggles,  # NEW
)
```

**Step 3: Update Feature Generation Logic**

**File:** `src/phase1/stages/features/engineer.py:engineer_features`

**Before:**
```python
def engineer_features(self, df: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, dict]:
    # Always compute wavelets
    df = self._add_wavelet_features(df)

    # Always compute microstructure
    df = self._add_microstructure_features(df)

    # ... etc
```

**After:**
```python
def engineer_features(self, df: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, dict]:
    # Conditional feature computation
    if self.enable_wavelets:
        logger.info("Computing wavelet features...")
        df = self._add_wavelet_features(df)
    else:
        logger.info("Wavelets disabled, skipping...")

    if self.enable_microstructure:
        logger.info("Computing microstructure features...")
        df = self._add_microstructure_features(df)
    else:
        logger.info("Microstructure disabled, skipping...")

    # ... etc for volume, volatility
```

**Step 4: Add Logging for Transparency**

**File:** `src/phase1/stages/features/run.py:after engineer initialization`

```python
# Log feature toggles for transparency
if config.feature_toggles:
    logger.info(f"Feature toggles: {config.feature_toggles}")
else:
    logger.info("Feature toggles: using defaults (all enabled)")
```

**Step 5: Add Unit Test**

**File:** `tests/phase1/stages/features/test_feature_toggles.py` (new)
```python
"""Test feature toggle functionality."""

import pandas as pd
import pytest
from pathlib import Path
from src.phase1.stages.features.engineer import FeatureEngineer


@pytest.fixture
def sample_df():
    """Sample OHLCV dataframe."""
    return pd.DataFrame({
        "datetime": pd.date_range("2020-01-01", periods=1000, freq="5min"),
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1000,
    })


def test_wavelets_disabled(tmp_path, sample_df):
    """Test that wavelets are not computed when disabled."""
    engineer = FeatureEngineer(
        input_dir=tmp_path,
        output_dir=tmp_path,
        timeframe="5min",
        feature_toggles={"wavelets": False},
    )

    df_features, info = engineer.engineer_features(sample_df, "TEST")

    # No wavelet columns should exist
    wavelet_cols = [c for c in df_features.columns if "wavelet" in c.lower()]
    assert len(wavelet_cols) == 0, f"Found wavelet columns when disabled: {wavelet_cols}"


def test_wavelets_enabled(tmp_path, sample_df):
    """Test that wavelets are computed when enabled."""
    engineer = FeatureEngineer(
        input_dir=tmp_path,
        output_dir=tmp_path,
        timeframe="5min",
        feature_toggles={"wavelets": True},
    )

    df_features, info = engineer.engineer_features(sample_df, "TEST")

    # Wavelet columns should exist
    wavelet_cols = [c for c in df_features.columns if "wavelet" in c.lower()]
    assert len(wavelet_cols) > 0, "No wavelet columns found when enabled"


def test_default_all_enabled(tmp_path, sample_df):
    """Test that default behavior enables all features."""
    engineer = FeatureEngineer(
        input_dir=tmp_path,
        output_dir=tmp_path,
        timeframe="5min",
        # No feature_toggles provided
    )

    df_features, info = engineer.engineer_features(sample_df, "TEST")

    # All feature types should be present
    assert any("wavelet" in c.lower() for c in df_features.columns)
    assert any("micro" in c.lower() for c in df_features.columns)
    assert any("volume" in c.lower() for c in df_features.columns)
```

**Run test:**
```bash
pytest tests/phase1/stages/features/test_feature_toggles.py -v
```

**Step 6: Integration Test**

```bash
# Test disable wavelets:
./pipeline run --symbols MES --disable-wavelets --timeframe 5min

# Verify no wavelets in output:
python -c "
import pandas as pd
import glob
features_files = glob.glob('runs/*/data/features/*_5min_features.parquet')
df = pd.read_parquet(features_files[0])
wavelet_cols = [c for c in df.columns if 'wavelet' in c.lower()]
print(f'Wavelet columns: {len(wavelet_cols)}')
assert len(wavelet_cols) == 0, 'Wavelets should be disabled!'
print('✅ Test passed: wavelets disabled')
"
```

**Acceptance Criteria:**
- ✅ `--disable-wavelets` removes wavelet features from output
- ✅ `--disable-microstructure` removes microstructure features
- ✅ `--disable-volume` removes volume features
- ✅ `--disable-volatility` removes volatility features
- ✅ Default behavior (no flags) enables all features
- ✅ Feature toggle status logged during pipeline run
- ✅ Unit tests pass
- ✅ Integration test passes

**Time Estimate:** 2-3 hours
**Risk:** LOW (additive change, doesn't break defaults)

---

### CFG-P1-002: Labeling barrier_overrides Ignored

**Location:** `src/phase1/stages/labeling/run.py:88-95`, `src/phase1/stages/final_labels/run.py:existing`

**Similar Issue:** CLI accepts `--k-up` and `--k-down` flags, stores in `config.barrier_overrides`, but labeling stages use hardcoded values or GA results.

**Documentation Claim:**

**File:** `CLAUDE.md:248`
```markdown
# Custom horizons and barriers
pipeline run --symbols MES --horizons 5,10,20 --k-up 1.5 --k-down 1.0
```

**CLI Accepts Barrier Overrides:**

**File:** `src/cli/run_commands_pipeline.py:96-104`
```python
# Labeling parameters
k_up: float | None = typer.Option(
    None, "--k-up", help="Upper barrier multiplier (overrides symbol-specific defaults)"
),
k_down: float | None = typer.Option(
    None, "--k-down", help="Lower barrier multiplier (overrides symbol-specific defaults)"
),
max_bars: int | None = typer.Option(
    None, "--max-bars", help="Maximum bars for label timeout (overrides defaults)"
),
```

**CLI Stores in Config:**

**File:** `src/cli/run_commands_core.py:_create_config_from_args`
```python
# Build barrier overrides
barrier_overrides = {}
if k_up is not None:
    barrier_overrides["k_up"] = k_up
if k_down is not None:
    barrier_overrides["k_down"] = k_down
if max_bars is not None:
    barrier_overrides["max_bars"] = max_bars

config_data["barrier_overrides"] = barrier_overrides
```

**PipelineConfig Has Field:**

**File:** `src/phase1/pipeline_config.py:104`
```python
barrier_overrides: dict[str, float] | None = None  # ✅ Field exists
```

**But Stage 4 Uses Hardcoded Values:**

**File:** `src/phase1/stages/labeling/run.py:88-95` (approximate)
```python
# Initial labeling with default/preset barriers
k_up = 2.0  # ❌ HARDCODED
k_down = 1.0  # ❌ HARDCODED
max_bars_ahead = config.max_bars_ahead  # Uses config default (50)

# Does NOT check config.barrier_overrides
```

**And Stage 6 Uses GA Results:**

**File:** `src/phase1/stages/final_labels/run.py:existing`
```python
# Load optimized parameters from GA
ga_params = load_ga_params(...)
k_up = ga_params.get("k_up", 2.0)
k_down = ga_params.get("k_down", 1.0)

# Does NOT check config.barrier_overrides
```

**Expected Behavior:**

Precedence order should be:
1. CLI overrides (`config.barrier_overrides`)
2. GA optimization results (if Stage 5 ran)
3. Symbol-specific defaults
4. Global defaults

**Test to Reproduce:**

```bash
# Run with custom barriers:
./pipeline run --symbols MES --k-up 3.0 --k-down 0.5

# Check labels:
python -c "
import pandas as pd
df = pd.read_parquet('runs/*/data/final/MES_*_final.parquet')
print('Label distribution:', df['label_20'].value_counts())
# Should use k_up=3.0, k_down=0.5
# But actually uses hardcoded k_up=2.0, k_down=1.0
"
```

**Impact:**
- 🔴 **CRITICAL**: CLI barrier flags have NO EFFECT
- 🔴 **CRITICAL**: Cannot override labeling parameters
- 🔴 **CRITICAL**: Cannot experiment with barrier tuning
- 🟡 **MEDIUM**: Forces GA optimization even for quick experiments

**Fix Plan:**

**Step 1: Define Precedence Function**

**File:** `src/phase1/stages/labeling/utils.py` (new or existing)
```python
"""Labeling utilities."""

from typing import Dict, Optional


def resolve_barrier_params(
    config_overrides: Optional[Dict[str, float]],
    ga_params: Optional[Dict[str, float]],
    symbol_defaults: Optional[Dict[str, float]],
    global_defaults: Dict[str, float],
) -> Dict[str, float]:
    """
    Resolve barrier parameters with clear precedence.

    Precedence order (highest to lowest):
    1. config_overrides (CLI flags)
    2. ga_params (Optuna optimization results)
    3. symbol_defaults (per-symbol tuning)
    4. global_defaults (fallback)

    Args:
        config_overrides: From config.barrier_overrides (CLI)
        ga_params: From Stage 5 optimization
        symbol_defaults: Symbol-specific barriers
        global_defaults: Global fallback values

    Returns:
        Resolved barrier parameters
    """
    result = global_defaults.copy()

    # Layer 1: Symbol defaults
    if symbol_defaults:
        result.update(symbol_defaults)

    # Layer 2: GA optimization (if ran)
    if ga_params:
        result.update(ga_params)

    # Layer 3: CLI overrides (highest priority)
    if config_overrides:
        result.update(config_overrides)

    return result
```

**Step 2: Update Stage 4 (Initial Labeling)**

**File:** `src/phase1/stages/labeling/run.py:88-100`

**Before:**
```python
# Initial labeling with default/preset barriers
k_up = 2.0
k_down = 1.0
max_bars_ahead = config.max_bars_ahead
```

**After:**
```python
from .utils import resolve_barrier_params

# Resolve barrier parameters with precedence
global_defaults = {
    "k_up": 2.0,
    "k_down": 1.0,
    "max_bars": config.max_bars_ahead,
}

symbol_defaults = get_symbol_defaults(symbol)  # If exists
ga_params = None  # No GA yet in Stage 4

params = resolve_barrier_params(
    config_overrides=config.barrier_overrides,
    ga_params=ga_params,
    symbol_defaults=symbol_defaults,
    global_defaults=global_defaults,
)

k_up = params["k_up"]
k_down = params["k_down"]
max_bars_ahead = params["max_bars"]

logger.info(f"Barrier params: k_up={k_up}, k_down={k_down}, max_bars={max_bars_ahead}")
if config.barrier_overrides:
    logger.info(f"  (CLI overrides applied: {config.barrier_overrides})")
```

**Step 3: Update Stage 6 (Final Labels)**

**File:** `src/phase1/stages/final_labels/run.py:existing`

**Before:**
```python
# Load optimized parameters from GA
ga_params = load_ga_params(...)
k_up = ga_params.get("k_up", 2.0)
k_down = ga_params.get("k_down", 1.0)
```

**After:**
```python
from src.phase1.stages.labeling.utils import resolve_barrier_params

# Load GA parameters (may be None if Stage 5 skipped)
ga_params = load_ga_params(...) if ga_artifacts_exist else None

# Resolve with precedence
params = resolve_barrier_params(
    config_overrides=config.barrier_overrides,
    ga_params=ga_params,
    symbol_defaults=get_symbol_defaults(symbol),
    global_defaults={"k_up": 2.0, "k_down": 1.0, "max_bars": 50},
)

k_up = params["k_up"]
k_down = params["k_down"]
max_bars_ahead = params["max_bars"]

logger.info(f"Final labeling params: k_up={k_up}, k_down={k_down}")
if config.barrier_overrides:
    logger.warning(f"CLI overrides applied (superseding GA): {config.barrier_overrides}")
elif ga_params:
    logger.info(f"Using GA-optimized parameters")
```

**Step 4: Update CLI Help Text**

**File:** `src/cli/run_commands_pipeline.py:96-100`

**Before:**
```python
k_up: float | None = typer.Option(
    None, "--k-up", help="Upper barrier multiplier (overrides symbol-specific defaults)"
),
```

**After:**
```python
k_up: float | None = typer.Option(
    None,
    "--k-up",
    help="Upper barrier multiplier (HIGHEST PRIORITY: overrides GA optimization and defaults)"
),
```

**Step 5: Add Unit Test**

**File:** `tests/phase1/stages/labeling/test_barrier_precedence.py` (new)
```python
"""Test barrier parameter precedence."""

import pytest
from src.phase1.stages.labeling.utils import resolve_barrier_params


def test_precedence_cli_overrides_all():
    """CLI overrides should have highest priority."""
    config_overrides = {"k_up": 5.0}
    ga_params = {"k_up": 3.0}
    symbol_defaults = {"k_up": 2.5}
    global_defaults = {"k_up": 2.0}

    result = resolve_barrier_params(
        config_overrides=config_overrides,
        ga_params=ga_params,
        symbol_defaults=symbol_defaults,
        global_defaults=global_defaults,
    )

    assert result["k_up"] == 5.0, "CLI should override all"


def test_precedence_ga_overrides_defaults():
    """GA params should override defaults when no CLI override."""
    ga_params = {"k_up": 3.0}
    symbol_defaults = {"k_up": 2.5}
    global_defaults = {"k_up": 2.0}

    result = resolve_barrier_params(
        config_overrides=None,
        ga_params=ga_params,
        symbol_defaults=symbol_defaults,
        global_defaults=global_defaults,
    )

    assert result["k_up"] == 3.0, "GA should override defaults"


def test_precedence_symbol_overrides_global():
    """Symbol defaults should override global when no GA/CLI."""
    symbol_defaults = {"k_up": 2.5}
    global_defaults = {"k_up": 2.0}

    result = resolve_barrier_params(
        config_overrides=None,
        ga_params=None,
        symbol_defaults=symbol_defaults,
        global_defaults=global_defaults,
    )

    assert result["k_up"] == 2.5, "Symbol should override global"


def test_precedence_global_fallback():
    """Global defaults when nothing else provided."""
    global_defaults = {"k_up": 2.0, "k_down": 1.0}

    result = resolve_barrier_params(
        config_overrides=None,
        ga_params=None,
        symbol_defaults=None,
        global_defaults=global_defaults,
    )

    assert result["k_up"] == 2.0
    assert result["k_down"] == 1.0
```

**Run test:**
```bash
pytest tests/phase1/stages/labeling/test_barrier_precedence.py -v
```

**Step 6: Integration Test**

```bash
# Test CLI override:
./pipeline run --symbols MES --k-up 3.0 --k-down 0.5 --timeframe 5min

# Verify barriers used:
grep "Barrier params" runs/*/logs/pipeline.log
# Should show: k_up=3.0, k_down=0.5

# Verify label distribution changed:
python -c "
import pandas as pd
import glob
label_files = glob.glob('runs/*/data/final/*_final.parquet')
df = pd.read_parquet(label_files[0])
print('Label distribution:')
print(df['label_20'].value_counts(normalize=True))
"
```

**Acceptance Criteria:**
- ✅ `--k-up 3.0` sets k_up=3.0 in labeling
- ✅ `--k-down 0.5` sets k_down=0.5 in labeling
- ✅ CLI overrides supersede GA optimization
- ✅ Precedence order is documented and enforced
- ✅ Logs show which parameters are being used
- ✅ Unit tests pass
- ✅ Integration test confirms behavior

**Time Estimate:** 2-3 hours
**Risk:** LOW (clean precedence logic, testable)

---

### CFG-P1-003: Scaler Type Hardcoded to Robust

**Location:** `src/phase1/stages/scaling/run.py:existing`

**Similar Pattern:** CLI accepts `--scaler-type`, stores in config, but Stage 7.5 hardcodes "robust".

**CLI Accepts Scaler Type:**

**File:** `src/cli/run_commands_pipeline.py:106-110`
```python
# Scaling options
scaler_type: str | None = typer.Option(
    None,
    "--scaler-type",
    help="Scaler type: robust, standard, minmax, quantile, none (default: robust)",
),
```

**PipelineConfig Has Field:**

**File:** `src/phase1/pipeline_config.py:105`
```python
scaler_type: str = "robust"  # ✅ Field exists
```

**But Stage 7.5 Hardcodes "robust":**

**File:** `src/phase1/stages/scaling/run.py:existing` (approximate line 120)
```python
# Create scaler config
scaler_config = ScalerConfig(
    scaler_type="robust",  # ❌ HARDCODED
    # ...
)
```

**Expected Behavior:**

Use `config.scaler_type` instead of hardcoded "robust".

**Additional Complexity:** Per-model scaler recommendations

**File:** `src/phase1/config/feature_sets.py:existing`
```python
FEATURE_SET_DEFINITIONS = {
    "boosting_optimal": FeatureSetDefinition(
        # ...
        recommended_scaler="none",  # Boosting doesn't need scaling
    ),
    "neural_optimal": FeatureSetDefinition(
        # ...
        recommended_scaler="robust",  # Neural networks need scaling
    ),
}
```

**Desired Behavior:**
1. If `--scaler-type` provided: Use CLI value
2. Else if model-specific run: Use feature set recommendation
3. Else: Use config default ("robust")

**Fix Plan:**

**Step 1: Update Stage 7.5 to Use Config**

**File:** `src/phase1/stages/scaling/run.py:~line 120`

**Before:**
```python
scaler_config = ScalerConfig(
    scaler_type="robust",
    # ...
)
```

**After:**
```python
# Resolve scaler type with precedence
scaler_type = config.scaler_type

# If model-aware run, use feature set recommendation (unless CLI override)
if hasattr(config, "feature_set") and config.feature_set:
    feature_set_def = get_feature_set_definition(config.feature_set)
    recommended = feature_set_def.recommended_scaler

    # Only use recommendation if not explicitly set via CLI
    if config.scaler_type == "robust":  # Default value
        logger.info(f"Using recommended scaler for '{config.feature_set}': {recommended}")
        scaler_type = recommended

scaler_config = ScalerConfig(
    scaler_type=scaler_type,
    # ...
)

logger.info(f"Scaling with: {scaler_type}")
```

**Step 2: Add Validation**

**File:** `src/phase1/stages/scaling/run.py:after scaler_type resolution`

```python
# Validate scaler type
valid_scalers = ["robust", "standard", "minmax", "quantile", "none"]
if scaler_type not in valid_scalers:
    raise ValueError(f"Invalid scaler_type: '{scaler_type}'. Valid: {valid_scalers}")
```

**Step 3: Integration Test**

```bash
# Test explicit scaler:
./pipeline run --symbols MES --scaler-type none --timeframe 5min

# Verify no scaling applied:
python -c "
import pandas as pd
import json
import glob

# Load scaled data
scaled_files = glob.glob('runs/*/data/splits/scaled/train_20.parquet')
df_scaled = pd.read_parquet(scaled_files[0])

# Load unscaled data for comparison
unscaled_files = glob.glob('runs/*/data/splits/train_20.parquet')
df_unscaled = pd.read_parquet(unscaled_files[0])

# If scaler='none', values should be identical
feature_cols = [c for c in df_scaled.columns if c.startswith('return_')]
assert df_scaled[feature_cols[0]].equals(df_unscaled[feature_cols[0]]), \
    'Scaler=none should not transform data'

print('✅ Test passed: scaler=none confirmed')
"

# Test model-specific scaler:
./pipeline run --symbols MES --feature-set boosting_optimal

# Verify 'none' scaler used (recommended for boosting):
grep "Scaling with: none" runs/*/logs/pipeline.log
```

**Acceptance Criteria:**
- ✅ `--scaler-type standard` uses StandardScaler
- ✅ `--scaler-type none` skips scaling
- ✅ Feature set recommendation used when no CLI override
- ✅ Validation rejects invalid scaler types
- ✅ Log shows which scaler is being used
- ✅ Integration test confirms behavior

**Time Estimate:** 1-2 hours
**Risk:** LOW (straightforward config plumbing)

---

### PATH-P1-001: Training Expects Global Paths, Pipeline Outputs Run-Scoped

**Location:** Multiple files - path mismatch between pipeline and training

**Documentation Claims:**

**File:** `CLAUDE.md:248`
```markdown
# Phase 1 (data)
./pipeline run --symbols MES

# Phase 2 (training)
python scripts/train_model.py --model xgboost --horizon 20
```

**But Training Script Defaults:**

**File:** `scripts/train_model.py:existing` (argparse section)
```python
parser.add_argument(
    "--data-dir",
    default="data/splits/scaled",  # ❌ GLOBAL PATH
    help="Directory containing scaled train/val/test data"
)
```

**Pipeline Actually Outputs:**

**File:** `src/phase1/config/pipeline_paths.py:docstring`
```python
"""
All outputs: run-scoped under `runs/{run_id}/data/`
"""
```

**File:** `src/phase1/stages/scaling/run.py:output_path_pattern`
```python
output_dir = config.scaled_splits_dir  # Points to runs/{run_id}/data/splits/scaled/
```

**Test to Reproduce:**

```bash
# Run pipeline:
./pipeline run --symbols MES
# Output: Run ID: 20250113_143052_a3f9

# Check where data went:
ls runs/20250113_143052_a3f9/data/splits/scaled/
# train_5.parquet, train_10.parquet, val_5.parquet, ...

# Try to train (using docs example):
python scripts/train_model.py --model xgboost --horizon 20
# Error: FileNotFoundError: data/splits/scaled/train_20.parquet not found
```

**User Must Manually Specify:**
```bash
python scripts/train_model.py --model xgboost --horizon 20 \
  --data-dir runs/20250113_143052_a3f9/data/splits/scaled
```

**Impact:**
- 🔴 **CRITICAL**: Documentation examples DON'T WORK
- 🔴 **CRITICAL**: Phase 1 → Phase 2 handoff BROKEN
- 🔴 **CRITICAL**: Users cannot follow end-to-end workflow
- 🟡 **MEDIUM**: Must manually copy paths (error-prone)
- 🟡 **MEDIUM**: No automatic run linkage

**Root Cause:**
- Pipeline designed for run isolation (good!)
- Training scripts designed for global convenience (legacy)
- No automatic "latest run" pointer
- No run ID passing mechanism

**Fix Plan:**

**Option A: Add "Latest Run" Symlink**

Create symlink at `data/splits/scaled` → latest run's scaled data.

**Implementation:**

**File:** `src/phase1/stages/scaling/run.py:end of function`

Add after successful scaling:
```python
# Create convenience symlink to latest run
latest_link = config.project_root / "data" / "splits" / "scaled"
latest_link.parent.mkdir(parents=True, exist_ok=True)

# Remove old symlink if exists
if latest_link.is_symlink() or latest_link.exists():
    latest_link.unlink()

# Create new symlink (relative path for portability)
target = Path("../../..") / "runs" / config.run_id / "data" / "splits" / "scaled"
latest_link.symlink_to(target)

logger.info(f"Created latest run symlink: {latest_link} -> {target}")
```

**Pros:**
- ✅ Maintains run isolation
- ✅ Makes docs examples work
- ✅ Backward compatible

**Cons:**
- ⚠️ Symlink overwritten each run (intentional)
- ⚠️ Windows symlink permissions

**Option B: Accept Run ID in Training Scripts**

Make training scripts accept `--run-id` and resolve paths internally.

**Implementation:**

**File:** `scripts/train_model.py:argparse`

**Before:**
```python
parser.add_argument(
    "--data-dir",
    default="data/splits/scaled",
    help="Directory containing scaled data"
)
```

**After:**
```python
parser.add_argument(
    "--run-id",
    default=None,
    help="Pipeline run ID (alternative to --data-dir)"
)
parser.add_argument(
    "--data-dir",
    default=None,
    help="Directory containing scaled data (alternative to --run-id)"
)

args = parser.parse_args()

# Resolve data directory
if args.run_id:
    args.data_dir = Path(f"runs/{args.run_id}/data/splits/scaled")
    logger.info(f"Using data from run: {args.run_id}")
elif args.data_dir:
    args.data_dir = Path(args.data_dir)
else:
    # Try to find latest run
    runs_dir = Path("runs")
    if runs_dir.exists():
        run_dirs = sorted(runs_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
        if run_dirs:
            args.run_id = run_dirs[0].name
            args.data_dir = run_dirs[0] / "data" / "splits" / "scaled"
            logger.info(f"Auto-detected latest run: {args.run_id}")
        else:
            raise ValueError("No runs found. Run pipeline first or specify --data-dir")
    else:
        raise ValueError("No runs/ directory found. Run pipeline first or specify --data-dir")
```

**Pros:**
- ✅ Explicit run traceability
- ✅ Auto-detect latest run as fallback
- ✅ Works cross-platform

**Cons:**
- ⚠️ Requires updating all training scripts

**Recommendation: Option A + Option B**

Use Option A for immediate fix (makes docs work), plus Option B for explicit run linking (better long-term).

**Step 1: Implement Option A (Symlink)**

See code above in Option A.

**Step 2: Implement Option B (Run ID Flag)**

Update all training scripts:
- `scripts/train_model.py`
- `scripts/train_ensemble.py`
- `scripts/run_cv.py`
- `scripts/serve_model.py`

**Step 3: Update Documentation**

**File:** `CLAUDE.md:248-250`

**Before:**
```markdown
# Train specific model (Phase 6)
python scripts/train_model.py --model xgboost --horizon 20
```

**After:**
```markdown
# Train specific model (Phase 6)

# Option 1: Auto-detect latest run
python scripts/train_model.py --model xgboost --horizon 20

# Option 2: Explicit run ID
python scripts/train_model.py --model xgboost --horizon 20 --run-id 20250113_143052

# Option 3: Explicit path
python scripts/train_model.py --model xgboost --horizon 20 \
  --data-dir runs/20250113_143052/data/splits/scaled
```

**Step 4: Add Manifest Linking**

Store run_id in training artifacts for traceability:

**File:** `scripts/train_model.py:after successful training`

```python
# Save training metadata with pipeline run linkage
metadata = {
    "model_type": args.model,
    "horizon": args.horizon,
    "pipeline_run_id": args.run_id,  # Link back to pipeline run
    "data_dir": str(args.data_dir),
    "timestamp": datetime.now().isoformat(),
}

with open(output_dir / "training_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
```

**Acceptance Criteria:**
- ✅ `python scripts/train_model.py --model xgboost --horizon 20` works (auto-detect)
- ✅ Symlink `data/splits/scaled` points to latest run
- ✅ `--run-id` flag explicitly selects pipeline run
- ✅ Training metadata includes pipeline_run_id
- ✅ Documentation examples work as written
- ✅ Cross-platform compatibility (symlink fallback on Windows)

**Time Estimate:** 2-3 hours
**Risk:** LOW (additive, maintains backward compat)

---

## STRUCTURAL ISSUES (P2)

### DOC-P2-001: Documentation Contradicts Implementation

**Location:** Multiple documentation files

**Evidence:**

**Contradiction 1: Model Count**

**File:** `docs/README.md:existing`
```markdown
23 models across 4 families
```

**File:** `docs/planning/PROJECT_CHARTER.md:existing`
```markdown
13 models deployed
```

**Reality Check:**
```bash
python -c "from src.models import ModelRegistry; print(len(ModelRegistry.list_all()))"
# Output: 23 (or 22 if CatBoost unavailable)
```

**Contradiction 2: MTF Status**

**File:** `docs/README.md:existing`
```markdown
MTF: ✅ Complete (9 intraday timeframes)
```

**File:** `docs/planning/PROJECT_CHARTER.md:existing`
```markdown
MTF: Partially implemented
```

**File:** `docs/implementation/CRITICAL_GAPS_SUMMARY.md:existing`
```markdown
9 TF defined but not configurable/used
```

**Reality:** 9 TFs defined in constants, but pipeline only outputs single TF per run.

**Contradiction 3: Phase Status**

**File:** `CLAUDE.md:existing`
```markdown
Phases 1-7 complete
```

**File:** `docs/planning/PROJECT_CHARTER.md:existing`
```markdown
Phase 1-3 complete, Phase 4+ in progress
```

**Impact:**
- 🟡 **MEDIUM**: Cannot trust documentation
- 🟡 **MEDIUM**: Contributors confused about actual status
- 🟡 **MEDIUM**: Slows velocity (must verify everything)

**Fix Plan:**

**Step 1: Audit All Documentation**

Create inventory:
```bash
# Find all markdown files:
find . -name "*.md" -not -path "./.venv/*" -not -path "./node_modules/*" > docs_inventory.txt

# Extract claims about:
# - Model counts
# - MTF status
# - Phase completion
# - Feature counts
# - Timeframe support
```

**Step 2: Choose Single Source of Truth**

**Recommendation:** `docs/README.md` as canonical status doc.

**Step 3: Mark Historical Docs**

**Files to Archive:**
- `docs/planning/PROJECT_CHARTER.md` → `docs/archive/planning/`
- `docs/implementation/CRITICAL_GAPS_SUMMARY.md` → `docs/archive/implementation/`

**Add Header to Archived Docs:**
```markdown
> **⚠️ HISTORICAL DOCUMENT**
> This document reflects the project state as of [DATE].
> For current status, see `docs/README.md`.
```

**Step 4: Update Current Docs to Match Reality**

**File:** `docs/README.md` (update to match actual implementation)

```markdown
# ML Model Factory Status

## Implementation Status

### Phase 1: Data Pipeline ✅ Complete
- Ingest → Clean → Features → Labels → Splits → Scaling → Datasets
- 14 stages orchestrated
- Run isolation enforced
- Leakage prevention (purge/embargo)

### Phase 2: Model Training ✅ Complete
- 23 models (or 22 if CatBoost unavailable)
  - Boosting: 3 models
  - Classical: 3 models
  - Neural: 10 models
  - Ensemble/Meta: 7 models
- Plugin-based model registry
- Unified training interface

### Phase 3: Cross-Validation ✅ Complete
- PurgedKFold with purge/embargo
- Walk-forward validation
- OOF generation for stacking
- Hyperparameter tuning with Optuna

### Multi-Timeframe (MTF) Status ⚠️ Partial
- ✅ 9 intraday timeframes defined (1m → 1h)
- ✅ MTF constants and utilities complete
- ❌ Pipeline only outputs single TF per run (see issue TF-P1-001)
- ❌ Multi-TF orchestration not wired

### Per-Model Feature Selection ✅ Implemented
- Model-family feature sets defined
- boosting_optimal, neural_optimal, etc.
- Per-model recommendations
- ❌ Not yet used by pipeline (see issue CFG-P1-001)

### Heterogeneous Ensembles ⚠️ Partial
- ✅ Stacking/blending/voting implemented
- ✅ Meta-learners implemented
- ❌ Different-timeframe bases blocked by TF-P1-001
- ❌ Per-model MTF strategy not wired

## Known Issues

See `SNEH FIX PLAN.md` for comprehensive issue list and remediation plan.

### Critical (P0)
- REPO-P0-001: Generated artifacts tracked in git
- CLI-P0-001: Broken imports in status commands
- CLI-P0-002: CLI prints invalid command examples
- RUN-P0-001: Training scripts missing (restored)

### Major (P1)
- TF-P1-001: Single TF per run (no 9-TF orchestration)
- TF-P1-002: Timeframe vocabulary inconsistent
- CFG-P1-001: Feature toggles ignored
- CFG-P1-002: Barrier overrides ignored
- CFG-P1-003: Scaler type hardcoded
- PATH-P1-001: Training/pipeline path mismatch

## What Works Today

✅ **End-to-end single-timeframe workflow:**
1. Run pipeline on 1-min OHLCV → produces 1 target TF
2. Train any of 23 models on that TF
3. Run cross-validation
4. Train ensembles (same-family)
5. Generate OOF predictions
6. Serve models

❌ **What doesn't work yet:**
- Multi-timeframe orchestration (single command → 9 TF outputs)
- Heterogeneous ensembles across different TFs
- Per-model MTF strategy selection
- CLI feature toggles/barrier overrides

## Next Milestones

1. **Fix P0 Issues** (1-2 days)
   - Repo hygiene, CLI imports, path alignment

2. **Fix P1 Issues** (1-2 weeks)
   - Multi-TF orchestration
   - Config plumbing (toggles, barriers, scaler)
   - Timeframe vocabulary unification

3. **Complete Heterogeneous Ensembles** (1 week)
   - Multi-TF base model support
   - Per-model dataset view selection

4. **Production Hardening** (ongoing)
   - Monitoring, alerting
   - Model serving optimization
   - Deployment automation
```

**Acceptance Criteria:**
- ✅ Single source of truth document (`docs/README.md`)
- ✅ Historical docs archived with warnings
- ✅ No contradictions between current docs
- ✅ Status matches reality (model count, MTF, phases)
- ✅ Known issues section links to fix plan

**Time Estimate:** 2-3 hours
**Risk:** LOW (documentation only)

---

## DETAILED FIX PLANS

[Previous detailed fix plans included above in P0/P1/P2 sections]

---

## IMPLEMENTATION SEQUENCE

### Phase 0: Hygiene and Immediate Fixes (1-2 days)

**Goal:** Make repo clean, CLI functional, documentation trustworthy.

**Tasks:**
1. ✅ REPO-P0-001: Remove generated artifacts, fix .gitignore (30 min)
2. ✅ CLI-P0-001: Fix status_commands.py imports (5 min)
3. ✅ CLI-P0-002: Fix CLI printed command examples (15 min)
4. ✅ PACK-P0-001: Create README.md (1 hour)
5. ✅ RUN-P0-001: Verify scripts restored, add protection (30 min)

**Acceptance:** Repo passes hygiene checks, CLI commands work, docs match reality.

---

### Phase 1: Configuration Plumbing (2-3 days)

**Goal:** Make CLI flags actually affect behavior.

**Tasks:**
1. ✅ CFG-P1-001: Wire feature_toggles to FeatureEngineer (2-3 hours)
2. ✅ CFG-P1-002: Wire barrier_overrides to labeling stages (2-3 hours)
3. ✅ CFG-P1-003: Wire scaler_type to scaling stage (1-2 hours)
4. ✅ PATH-P1-001: Fix training/pipeline path mismatch (2-3 hours)
5. ✅ Integration testing: Verify all toggles work end-to-end (2 hours)

**Acceptance:** All CLI flags modify behavior, integration tests pass.

---

### Phase 2: Timeframe Unification (2-4 days)

**Goal:** Single timeframe vocabulary, prepare for multi-TF.

**Tasks:**
1. ✅ Create canonical timeframe registry (2-3 hours)
2. ✅ TF-P1-002: Update all modules to use registry (2-3 hours)
3. ✅ Unit tests for timeframe utilities (1 hour)
4. ✅ Update PipelineConfig for multi-TF support (dataclass only) (1 hour)

**Acceptance:** No timeframe inconsistencies, all modules use canonical registry.

---

### Phase 3: Multi-Timeframe Orchestration (1-2 weeks)

**Goal:** Pipeline can produce 9 TF outputs in single run.

**Tasks:**
1. ✅ TF-P1-001 Step 1-2: Update PipelineConfig and Stage 2 (4-6 hours)
2. ✅ TF-P1-001 Step 3-5: Update CLI and all downstream stages (8-12 hours)
3. ✅ TF-P1-001 Step 6-7: Horizon scaling and labeling updates (4-6 hours)
4. ✅ Integration testing: 9-TF pipeline runs (4 hours)
5. ✅ Documentation updates (2 hours)

**Acceptance:** `./pipeline run --timeframes 9tf` produces 9 complete TF datasets.

---

### Phase 4: Documentation Reconciliation (1-2 days)

**Goal:** Documentation matches implementation.

**Tasks:**
1. ✅ DOC-P2-001: Audit all documentation (2 hours)
2. ✅ Archive historical docs (1 hour)
3. ✅ Update docs/README.md to match reality (2-3 hours)
4. ✅ Update CLAUDE.md examples (1 hour)
5. ✅ Create CONTRIBUTING.md if missing (1 hour)

**Acceptance:** No contradictions, status matches code, examples work.

---

## ACCEPTANCE CRITERIA

### Phase 0 Complete When:
- [ ] No `__pycache__/` in `git ls-files`
- [ ] `.gitignore` prevents artifact reintroduction
- [ ] `./pipeline status --help` works
- [ ] `README.md` exists and matches `pyproject.toml`
- [ ] Scripts directory protected from deletion

### Phase 1 Complete When:
- [ ] `--disable-wavelets` removes wavelet features
- [ ] `--k-up 3.0` sets barrier in labeling
- [ ] `--scaler-type none` skips scaling
- [ ] `python scripts/train_model.py --model xgboost` auto-detects latest run
- [ ] All integration tests pass

### Phase 2 Complete When:
- [ ] "25min" validates successfully
- [ ] "1h" validates successfully
- [ ] All modules import from `src.common.timeframes`
- [ ] No hardcoded timeframe lists remain
- [ ] Unit tests for timeframes pass

### Phase 3 Complete When:
- [ ] `--timeframes 9tf` produces 9 TF outputs (all stages)
- [ ] Horizons scaled consistently across TFs (time-based)
- [ ] Manifest enumerates all TF outputs
- [ ] Training can select specific TF dataset
- [ ] Integration tests pass for 9-TF workflow

### Phase 4 Complete When:
- [ ] No contradictions between current docs
- [ ] Model count matches reality in all docs
- [ ] MTF status accurately described
- [ ] Examples in docs actually work
- [ ] Historical docs archived with warnings

---

## APPENDIX: EVIDENCE INDEX

### File References by Issue

**REPO-P0-001:**
- `.gitignore:1-50` (needs updates)
- Repository-wide `__pycache__/` directories

**CLI-P0-001:**
- `src/cli/status_commands.py:24,34` (broken imports)
- `src/phase1/pipeline_config.py:1` (actual location)
- `src/common/manifest.py:1` (actual location)

**CLI-P0-002:**
- `src/cli/run_commands_pipeline.py:227` (wrong command format)
- `src/cli/status_commands.py:40-43` (actual signature)

**RUN-P0-001:**
- `scripts/` directory (deleted in `2a4f884`, restored from `d757a7c`)
- `CLAUDE.md:248` (references scripts)
- `docs/guides/MODEL_INTEGRATION.md:various` (references scripts)

**PACK-P0-001:**
- `pyproject.toml:8` (references missing README)
- `README.md` (missing at root)

**TF-P1-001:**
- `src/phase1/pipeline_config.py:54` (single timeframe)
- `src/phase1/stages/clean/run.py:84` (single output)
- `src/phase1/stages/clean/pipeline.py:existing` (multi-TF function exists)
- `CLAUDE.md:18-28` (claims 9 TF)

**TF-P1-002:**
- `src/phase1/config/features.py:17` (missing 25min, 1h)
- `src/phase1/stages/mtf/constants.py:20-42` (includes all 9)
- `src/phase1/config/features.py:245` (MTF_CONFIG incomplete)

**CFG-P1-001:**
- `src/cli/run_commands_pipeline.py:79-94` (CLI accepts toggles)
- `src/phase1/pipeline_config.py:103` (field exists)
- `src/phase1/stages/features/run.py:73-84` (not passed to engineer)
- `src/phase1/stages/features/engineer.py:__init__` (hardcoded enables)

**CFG-P1-002:**
- `src/cli/run_commands_pipeline.py:96-104` (CLI accepts barriers)
- `src/phase1/pipeline_config.py:104` (field exists)
- `src/phase1/stages/labeling/run.py:88-95` (hardcoded values)
- `src/phase1/stages/final_labels/run.py:existing` (uses GA, ignores overrides)

**CFG-P1-003:**
- `src/cli/run_commands_pipeline.py:106-110` (CLI accepts scaler)
- `src/phase1/pipeline_config.py:105` (field exists)
- `src/phase1/stages/scaling/run.py:~120` (hardcoded "robust")

**PATH-P1-001:**
- `scripts/train_model.py:existing` (defaults to `data/splits/scaled`)
- `src/phase1/config/pipeline_paths.py:docstring` (run-scoped outputs)
- `CLAUDE.md:248` (example doesn't work)

**DOC-P2-001:**
- `docs/README.md:various` (claims 23 models, 9 TF complete)
- `docs/planning/PROJECT_CHARTER.md:various` (claims 13 models, MTF partial)
- `docs/implementation/CRITICAL_GAPS_SUMMARY.md:various` (MTF not usable)

---

## SUMMARY

This fix plan addresses **11 critical issues** (P0), **6 major issues** (P1), and **1 structural issue** (P2) identified through comprehensive documentation analysis and codebase validation.

**Total Estimated Time:** 3-4 weeks for complete remediation (can be parallelized).

**Risk Assessment:**
- **LOW RISK** fixes (hygiene, imports, config plumbing): 60% of work
- **MEDIUM RISK** fixes (multi-TF orchestration): 30% of work
- **HIGH RISK** fixes: None (all changes are testable and reversible)

**Recommended Priority:**
1. **Week 1:** Phase 0 + Phase 1 (hygiene + config plumbing)
2. **Week 2:** Phase 2 + Start Phase 3 (timeframes + multi-TF)
3. **Week 3-4:** Complete Phase 3 + Phase 4 (multi-TF + docs)

**Success Metrics:**
- Pipeline runs successfully end-to-end (1-min → 9 TF → train → serve)
- All CLI examples in docs work as written
- All 23 models trainable on any timeframe
- Heterogeneous ensembles work across timeframes
- Zero documentation contradictions
- Clean git repo (no generated artifacts)

---

**END OF SNEH FIX PLAN**
