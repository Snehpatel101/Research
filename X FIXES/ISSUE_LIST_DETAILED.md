# Detailed Issue List (Repo + `src/` Deep Dive)

This is the detailed issue list of inconsistencies and structural flaws that currently prevent the repo from cleanly achieving:

> Upload one dataset → automatically produce nine timeframe datasets → train a single model or ensemble (with per-model feature practices, meta-learners, and model management).

## P0 — Hard blockers (must fix first)

### REPO-P0-001 — Repo contains thousands of generated artifacts (`__pycache__`, `*.pyc`)

- Evidence: `find . -name '__pycache__' -o -name '*.pyc'` returns thousands of entries across `src/` and `tests/`.
- Impact:
  - Massive noise hides real structure.
  - Easy to accidentally rely on stale bytecode.
  - New contributors cannot tell what’s real code vs generated output.
- Fix direction:
  - Remove tracked caches; ensure `.gitignore` ignores `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.ruff_cache/`, `.venv/`.

### CLI-P0-001 — `pipeline status` is broken by incorrect lazy imports

- Evidence: `src/cli/status_commands.py` does `from .. import pipeline_config` and `from .. import manifest`, but `src/__init__.py` does not expose `pipeline_config` or `manifest` modules.
- Impact: status/validation UX is broken even if the pipeline run itself completes; users can’t introspect runs reliably.
- Fix direction: unify imports to `src.phase1.pipeline_config` and `src.common.manifest` (or expose a single stable import surface).

### CLI-P0-002 — CLI prints invalid command examples

- Evidence: `src/cli/run_commands_pipeline.py` prints `pipeline status --run-id {run_id}`, but `status` expects positional `run_id`.
- Impact: users copy/paste a command that does not work; breaks trust in the CLI.
- Fix direction: align CLI output text with actual command signature (or change signature).

### RUN-P0-001 — Training quickstart references a missing runner (`scripts/train_model.py`)

- Evidence:
  - `CLAUDE.md`, many docs under `docs/`, and `config/*/README.md` reference `scripts/train_model.py`.
  - The `scripts/` directory was deleted at `HEAD` (commit `2a4f884`), but exists in the prior commit `d757a7c`.
- Impact: when `scripts/` is missing, there is no obvious end-to-end “train a model after pipeline” entrypoint.
- Status in this workspace: `scripts/` has been restored from `d757a7c`.
- Fix direction: commit/keep the restored scripts and still converge on one authoritative entrypoint (CLI subcommand or script) so docs don’t drift again.

## P1 — Major blockers for “9 datasets” + “adaptive factory”

### TF-P1-001 — Only one base timeframe per pipeline run (no multi-TF orchestration)

- Evidence: `src/phase1/pipeline_config.py` has a single `target_timeframe: str`; Stage 2 (`src/phase1/stages/clean/run.py`) outputs only `{symbol}_{target_timeframe}_clean.parquet`.
- Impact: cannot “automatically produce nine datasets” from one upload; requires manual repeated runs.
- Fix direction: introduce a first-class multi-timeframe run concept (either materialize 9 TF outputs or implement dataset “views”).

### TF-P1-002 — Multi-timeframe cleaning exists but is not wired into the pipeline

- Evidence: `src/phase1/stages/clean/pipeline.py` defines `clean_symbol_data_multi_timeframe()`, but Stage 2 uses `clean_symbol_data()` only.
- Impact: the repo already contains the core primitive needed for “one upload → many TF outputs”, but orchestration ignores it.
- Fix direction: promote the multi-timeframe function into Stage 2 execution path or create a dedicated “MTF materialization” stage.

### TF-P1-003 — Timeframe vocabulary is inconsistent across modules

- Evidence:
  - `src/phase1/config/features.py` validation does not include `25min` and does not treat `1h` as supported.
  - `src/phase1/stages/mtf/constants.py` includes `25min` and `1h` in `MTF_TIMEFRAMES`.
- Impact: even if you wire “9 TF” output, some TF strings will be rejected upstream.
- Fix direction: one canonical timeframe registry + alias mapping + single validation function used everywhere.

### MTF-P1-001 — MTF configuration exists in multiple places with conflicting defaults

- Evidence:
  - `src/phase1/stages/mtf/constants.py` defines supported TFs and defaults.
  - `src/phase1/stages/mtf/__init__.py` docstring describes a different supported/default set.
  - `src/phase1/config/features.py` contains an `MTF_CONFIG` dict with its own base TF and TF list.
- Impact: developers can’t tell which file controls MTF behavior; “change the config” may not change runtime behavior.
- Fix direction: keep one MTF config authority (prefer the code-path actually used by the pipeline), and mark the rest as deprecated/reference.

### CFG-P1-002 — CLI “feature toggles” are stored but ignored by feature engineering

- Evidence:
  - CLI creates `PipelineConfig.feature_toggles` (`src/cli/run_commands_core.py`).
  - Stage 3 (`src/phase1/stages/features/run.py`) does not read `config.feature_toggles` and always constructs `FeatureEngineer(...)` with defaults (wavelets/microstructure/etc enabled by default).
- Impact: the system appears configurable but behavior does not change; prevents “configure any shoe”.
- Fix direction: plumb `feature_toggles` into `FeatureEngineer` (and/or into the individual feature modules) and make the feature output deterministic per toggle set.

### CFG-P1-003 — CLI labeling overrides are stored but ignored by labeling stages

- Evidence:
  - CLI stores `PipelineConfig.barrier_overrides` (`src/cli/run_commands_core.py`).
  - Stage 4 uses hardcoded initial params (`k_up=2.0`, `k_down=1.0`) (`src/phase1/stages/labeling/run.py`).
  - Stage 6 uses GA results or defaults; does not consult `config.barrier_overrides` (`src/phase1/stages/final_labels/run.py`).
- Impact: “override labeling parameters” flags do not actually override labeling; blocks rapid experimentation and proper configuration-driven behavior.
- Fix direction: define a clear precedence order: overrides > GA > defaults, and enforce it in Stage 4/6.

### CFG-P1-004 — `--scaler-type` exists but scaling stage is hardcoded to robust

- Evidence:
  - CLI supports `--scaler-type` and stores `PipelineConfig.scaler_type`.
  - Stage 7.5 constructs `ScalerConfig(scaler_type="robust", ...)` unconditionally (`src/phase1/stages/scaling/run.py`).
- Impact: scaling policy cannot be configured; conflicts with feature-set guidance (e.g., boosting recommended scaler “none”).
- Fix direction: make scaling stage derive scaler config from `PipelineConfig.scaler_type` and/or feature set/model family.

### PATH-P1-001 — Pipeline outputs run-scoped artifacts but training examples assume global paths

- Evidence:
  - Pipeline path mixin defines `runs/{run_id}/data/*` outputs (`src/phase1/config/pipeline_paths.py`).
  - `TimeSeriesDataContainer` docs and examples reference `data/splits/scaled` (`src/phase1/stages/datasets/container.py`).
- Reinforced by root-level docs/scripts:
  - `CLAUDE.md` describes “Processed: `data/splits/scaled/`”.
  - `scripts/train_model.py` defaults to `--data-dir data/splits/scaled`.
  - `scripts/run_cv.py` assumes “Phase 1 data in data/splits/scaled/”.
- Impact: users can’t follow docs to train on the data they just produced.
- Fix direction: define an explicit “artifact locator” contract: pipeline run → scaled splits dir → dataset manifests.

### SCRIPT-P1-001 — `scripts/verify_pipeline_final.py` is not portable and conflicts with repo contracts

- Evidence:
  - Hardcoded absolute path: `DATA_DIR = Path("/Users/sneh/research/...")`.
  - Assumes a specific directory layout (`data/splits/final_correct/scaled`) and file names like `scaling_config.json`.
  - Assumes 5-minute bars when computing purge/embargo gaps (divides minutes by 5).
- Impact:
  - The “final verification” script cannot run in this repo without manual edits.
  - It encodes assumptions (multi-symbol, fixed bar duration, legacy filenames) that contradict the “configurable factory” goal.
- Fix direction:
  - Convert to use repo-relative paths and `--data-dir` args.
  - Make timeframe assumptions explicit (or infer from config/metadata).
  - Align expected filenames with the pipeline’s actual outputs.

### PATH-P1-002 — Stage 1 writes global “validated” artifacts, contradicting run isolation

- Evidence:
  - `PipelinePathMixin` declares “All outputs run-scoped under runs/{run_id}/data/” (`src/phase1/config/pipeline_paths.py`).
  - Stage 1 writes to `data/raw/validated/{symbol}_1m_validated.parquet` (global) (`src/phase1/stages/ingest/run.py`).
- Impact: reproducibility boundaries are ambiguous; runs can share/overwrite “validated” state without an explicit cache/versioning policy.
- Fix direction: either (a) make validated data run-scoped, or (b) explicitly define it as a content-addressed cache with checksums/versioning.

### DOC-P1-001 — Docs disagree about model counts and MTF status

- Evidence:
  - `docs/README.md` claims 23 models and 9 TF complete.
  - `docs/planning/PROJECT_CHARTER.md` claims 13 models and MTF partially implemented.
  - `docs/implementation/CRITICAL_GAPS_SUMMARY.md` says “9 TF defined but not configurable/used”.
- Impact: contributors cannot trust documentation to reflect reality, slowing any re-org/refactor.
- Fix direction: choose one canonical truth doc; mark others as historical/archive or update them to match.

### STAGE-P1-001 — “14 pipeline stages” is claimed, but the runnable stage registry defines fewer

- Evidence:
  - Docs refer to “14 stages” (e.g., `docs/reference/PIPELINE_STAGES.md`).
  - The orchestrated pipeline (`src/pipeline/stage_registry.py`) defines 9 main stages (+ fractional 7.5/7.6/7.7).
  - Additional capabilities exist under `src/phase1/stages/` (sessions, regime, etc.) but are not first-class stages in the runner.
- Impact: “what runs when” is unclear; contributors can’t tell if modules are standalone utilities, internal steps, or intended stages.
- Fix direction: publish the canonical stage list from the runner and treat other components as implementation details (or promote them into explicit stages).

### API-P1-001 — “stages” public import surface is implied but does not exist

- Evidence:
  - Tests import `from stages import ...` (`tests/verify_modules.py`, `tests/test_stages.py`).
  - Code docstrings show usage like `from stages.regime import ...` (`src/phase1/stages/regime/__init__.py`, others).
- Impact: tests/docs suggest a stable API that isn’t implemented; new devs cannot import what docs show.
- Fix direction: either (a) create a real `stages` package as a compatibility layer, or (b) purge all references and standardize on `src.phase1.stages`.

### CFG-P1-001 — Multiple configuration systems are present but not unified

- Evidence:
  - Phase 1 driven by `PipelineConfig` dataclass.
  - Training has both `TrainerConfig` dataclass and YAML loaders (`src/models/config/*` + `config/models/*.yaml`).
  - Root `config/pipeline/*.yaml` exists but is not clearly used by the main CLI path.
- Impact: “one shoe does not fit all” becomes “nobody knows which shoe is being worn”.
- Fix direction: one explicit config merge layer and a published precedence order.

## P2 — Structural/quality issues that compound the above

### PACK-P2-001 — Packaging metadata references missing root `README.md`

- Evidence: `pyproject.toml` declares `readme = "README.md"` but no root `README.md` exists.
- Impact: editable installs/build metadata can break; also signals absence of a single canonical quickstart.
- Fix direction: add a root `README.md` (or update pyproject to point to an existing readme).

### DOC-P2-001 — Internal docstrings contradict implementation (MTF support, imports)

- Evidence: `src/phase1/stages/mtf/__init__.py` and `src/phase1/stages/mtf/generator.py` docstrings describe TFs and defaults that differ from `src/phase1/stages/mtf/constants.py`.
- Impact: even maintainers cannot rely on local docs to understand behavior.
- Fix direction: audit/update docstrings after selecting the canonical timeframe vocabulary and defaults.

### CLI-P2-001 — CLI defaults are inconsistent with single-symbol enforcement

- Evidence: default symbols in some commands are `"MES,MGC"` while `PipelineConfig` rejects multi-symbol by default.
- Impact: “out of the box” commands can fail with validation errors.
- Fix direction: align CLI defaults with config constraints (or change constraints and document it).

### DATA-P2-001 — Feature scaling likely scales raw MTF OHLCV columns unintentionally

- Evidence: scaling stage excludes `open/high/low/close/volume` but not `open_10m`, `close_1h`, etc. (`src/phase1/stages/scaling/run.py`).
- Impact: model-family-specific preprocessing is muddied; “transformer_raw” style feature sets can be distorted.
- Fix direction: scaler policy should be tied to the chosen feature set/model family, not inferred from column dtypes alone.

### MODEL-P2-001 — Import-time model registration is not resilient to missing optional deps

- Evidence:
  - `src/models/__init__.py` imports `boosting`, `neural`, etc. to trigger registration.
  - Those subpackages import model implementations eagerly (e.g., `src/models/boosting/__init__.py` imports XGBoost/LightGBM/CatBoost modules).
- Impact:
  - “Use only parts of the factory” becomes brittle if any optional dependency is missing.
  - Makes it harder to run Phase 1 pipeline in minimal environments.
- Fix direction: lazy/conditional registration and optional dependency guards for all model families (not just CatBoost).

### SERVE-P2-001 — Serving docs reference missing scripts and inconsistent frameworks

- Evidence:
  - `src/inference/server.py` docstring references `python scripts/serve_model.py ...` (missing).
  - Higher-level docs reference FastAPI in places, but `src/inference/server.py` is Flask-oriented.
- Impact: “train → serve” parity is hard to validate because the official entrypoint is unclear/non-existent when `scripts/` is missing.
- Status in this workspace: `scripts/serve_model.py` has been restored from `d757a7c`.
- Fix direction: choose one serving interface, provide one entrypoint, and update docs to match (then keep it from being deleted again).

### PERF-P2-001 — Parquet checksum strategy is likely too expensive for real datasets

- Evidence: `src/common/manifest.py` computes parquet checksums by loading the entire parquet into pandas and hashing `df.to_json(...)`.
- Impact: checksum computation becomes a bottleneck or memory failure on large OHLCV datasets; undermines “robust factory” claims.
- Fix direction: hash parquet bytes (or stable metadata + row group hashes), or use a streaming approach that doesn’t materialize full dataframes.

### ORG-P2-001 — Overlapping/duplicated “feature selection” and “validation” modules across `src/`

- Evidence (non-exhaustive):
  - Feature selection logic appears in multiple places: `src/feature_selection/`, `src/models/feature_selection/`, `src/cross_validation/feature_selector.py`, and `src/phase1/utils/feature_selection.py`.
  - Validation logic exists in both pipeline-stage validators (`src/phase1/stages/*/validators.py`) and generic utilities (`src/utils/config_validator.py`, etc.).
- Impact: unclear ownership and drift between implementations; increases refactor cost and makes “proper feature practices” hard to enforce uniformly.
- Fix direction: pick canonical owners:
  - Phase 1 data validation under `src/phase1/stages/validation/`,
  - Model/training-time feature selection under `src/models/feature_selection/`,
  - CV-specific selection under `src/cross_validation/`,
  and document the boundaries explicitly.

## What to do next (prioritized)

1. Decide what “nine datasets” means (materialized vs views); see `X FIXES/DATASET_TIMEFRAME_STRATEGY.md`.
2. Establish a single timeframe registry and validation function used by Phase 1 + MTF + training.
3. Fix CLI import surfaces and printed command examples (status/validate/run).
4. Create one authoritative training entrypoint (CLI subcommand or script) and remove dead references.
5. Clean repo hygiene (remove caches/bytecode; stop tracking `.venv` and other generated directories).
