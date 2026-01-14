# Inconsistency Catalog (Docs vs Code vs Config vs Paths)

This is a concrete catalog of contradictions that create the repo’s “split-brain” feeling and prevent clean end-to-end usage.

## 1) “9 Timeframes” vs multiple competing definitions

- `docs/README.md` claims 9 of 9 timeframes: `1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h`.
- `src/phase1/stages/mtf/constants.py` defines a “full ladder” (`FULL_MTF_TIMEFRAMES`) matching that claim, but **default** `DEFAULT_MTF_TIMEFRAMES` is a 7-TF intraday ladder (`10min..1h`).
- `src/phase1/stages/mtf/__init__.py` docstring claims supported TFs include `4h` and `daily` and describes a different default set.
- `src/phase1/config/features.py` has its own timeframe list (`SUPPORTED_TIMEFRAMES`) that does **not** include `25min`, and treats `1h` as a pandas alias but not as a supported timeframe for validation.

Impact: even before “nine datasets”, the project has *multiple incompatible timeframe vocabularies*.

## 2) “Canonical dataset” vs what the pipeline actually outputs

- Docs commonly describe `data/processed/*` outputs.
- Code implements run-scoped outputs under `runs/{run_id}/data/*` via `src/phase1/config/pipeline_paths.py`.
- But Stage 1 writes validated data globally to `data/raw/validated/*` (`src/phase1/stages/ingest/run.py`), contradicting the “run-scoped outputs” principle.

Impact: reproducibility boundaries are unclear (global cache vs run outputs).

## 3) CLI output strings vs CLI interface reality

- `src/cli/run_commands_pipeline.py` prints: `pipeline status --run-id {run_id}`.
- `src/cli/status_commands.py` defines `status_command(run_id: typer.Argument(...))` (positional arg), not `--run-id`.

Impact: copy/paste from CLI output leads to failure.

## 4) CLI defaults contradict PipelineConfig constraints

- `src/phase1/pipeline_config.py` enforces single-symbol runs unless `allow_batch_symbols=True`.
- `src/cli/status_commands.py` and `validate_command` default to `"MES,MGC"` (multi-symbol).

Impact: default CLI behavior can immediately violate the config constraints.

## 5) “scripts/train_model.py” references vs repo state

References exist across:
- `CLAUDE.md`, multiple docs under `docs/`, and multiple `config/*` READMEs.
- The `scripts/` directory was missing at `HEAD` (it was deleted in commit `2a4f884`).
- It can be restored from the prior commit `d757a7c` (and is now restored in this workspace).

Impact: most “how to train” instructions dead-end.

## 5b) “scripts/serve_model.py” references vs repo state

- `src/inference/server.py` docstring references `python scripts/serve_model.py ...`.
- Same underlying issue as (5): `scripts/` was missing at `HEAD`, but can be restored from `d757a7c` (and is now restored in this workspace).

Impact: serving/bundling instructions also dead-end, even though inference code exists.

## 6) `stages.*` import surface exists in docs/tests, but not in code layout

- Multiple tests import `from stages import ...` (e.g., `tests/verify_modules.py`).
- Several `src/phase1/stages/*/__init__.py` docstrings show examples like `from stages.regime import ...`.
- Actual implementation lives under `src/phase1/stages/...` (no top-level `stages` package).

Impact: docs/tests imply a public import surface that doesn’t exist, damaging usability and trust.

## 7) Training expects global paths; pipeline produces run-scoped paths

- `src/phase1/stages/datasets/container.py` usage examples reference `data/splits/scaled`.
- Pipeline produces `runs/{run_id}/data/splits/scaled/*` (via `PipelinePathMixin`).
- Root-level guidance reinforces the global path:
  - `CLAUDE.md` describes “Processed: `data/splits/scaled/`”.
  - `scripts/train_model.py` and `scripts/run_cv.py` default to `data/splits/scaled`.

Impact: training examples don’t match pipeline outputs; users must “guess” the correct path.

## 8) MTF docstrings disagree with MTF constants

- `src/phase1/stages/mtf/generator.py` docstring describes supported TFs like `15min,30min,1h,4h,daily`.
- `src/phase1/stages/mtf/constants.py` currently defines broader support and a different default.

Impact: internal code documentation is not aligned with actual behavior.

## 9) CLI advertises config toggles that don’t affect behavior

- CLI exposes flags for wavelets/microstructure/volume/volatility and stores them in `PipelineConfig.feature_toggles`.
- Stage 3 feature engineering does not consult those toggles (it instantiates `FeatureEngineer` with defaults).

Impact: “configurable factory” is undermined because CLI options appear to work but do not change outputs.

## 10) CLI advertises labeling/scaling overrides that don’t affect behavior

- CLI stores `PipelineConfig.barrier_overrides` and `PipelineConfig.scaler_type`.
- Labeling and scaling stages use hardcoded values / GA outputs and do not consult the overrides.

Impact: the repo appears highly parameterized, but key knobs are not actually connected to the pipeline.

## 11) Repo includes generated artifacts as tracked files

- `__pycache__/` and `*.pyc` are present throughout `src/` and `tests/` (thousands of files).
- `.venv/`, `.pytest_cache/`, `.ruff_cache/` exist in-repo.

Impact: noise hides signal, complicates reviews, and encourages accidental coupling to one machine/environment.
