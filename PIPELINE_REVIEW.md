# Src Pipeline Review and OHLCV Training Standards (Expanded)

## Executive Summary
This review covers the operational pipeline under `src/` with emphasis on the active data pipeline (`src/data/pipeline`) and the training orchestration path (`src/orchestrator.py` + `src/models/training`). The codebase contains two parallel pipelines (Phase 1 data prep and Phase 2 training) that are not wired together in a single entry point. Several configuration and multi-timeframe gaps will break runs or produce inconsistent artifacts unless you are very deliberate about how you invoke the system and what data you feed it.

Security, tests, and MLflow style tracking are explicitly out of scope, per request.

## Recent Origin Pull (Latest Upstream Change)
- Pulled fast-forward to commit `27a5120` (fix: remove extra `open_prices` argument from `triple_barrier_numba` calls).
- Summary: `triple_barrier_numba` expects 7 args (close, high, low, atr, k_up, k_down, max_bars); call sites were passing an extra `open_prices` array.
- Files touched: `src/data/pipeline/stages/labeling/run.py`, `src/data/pipeline/stages/ga_optimize/fitness.py`, `src/data/pipeline/stages/ga_optimize/optuna_optimizer.py` (2 locations), `src/data/pipeline/stages/final_labels/core.py`.
- Impact: removes a runtime arity mismatch that would have thrown `TypeError` during labeling and optimization; functional intent is unchanged aside from fixing the call signature.

## Scope and What "src pipeline" Means Here
- `src/pipeline/` contains only bytecode (`__pycache__`) and no Python sources.
- The active, maintainable pipeline is `src/data/pipeline` (Phase 1: data preparation for OHLCV ML).
- The training orchestration path is `src/orchestrator.py` + `src/models/training/*` (Phase 2: model training and ensemble handling). It expects data already labeled and featurized.
- The CLI entry at `src/cli/commands/pipeline.py` references a missing `src/ml_pipeline` module, so the CLI "pipeline" commands are currently not functional.

## Pipeline Topology (Actual Code Path)
### Phase 1: Data Pipeline
Stage order from `src/data/pipeline/stage_registry.py`:
1. data_generation
2. data_cleaning
3. feature_engineering
4. initial_labeling
5. ga_optimize
6. final_labels
7. create_splits
7.5 feature_scaling
7.6 build_datasets
7.7 validate_scaled
8. validate
9. generate_report

### Phase 2: Training Orchestration
- `src/orchestrator.py` loads a parquet file via `PipelineConfig.data_path`, then calls `UnifiedTrainingOrchestrator.train(df)`.
- `UnifiedTrainingOrchestrator` uses `UnifiedDataPreparation` adapters to split, scale, and shape features for each model.
- It does NOT call Phase 1 data pipeline stages.

## Configuration Surfaces
### 1) Core Training Config
- `src/core/config.py::PipelineConfig` (training, model, CV, ensemble)

### 2) Phase 1 Data Config
- `src/data/pipeline/data_config.py::DataConfig` (data pipeline only)

### 3) Config Adapter
- `src/data/pipeline/config_adapter.py::to_data_config` (core -> data pipeline)

### 4) CLI
- `src/cli/run_commands_pipeline.py` uses `DataConfig` via `create_default_config` (the Phase 1 pipeline), not the training config.
- `src/cli/commands/pipeline.py` tries to use `src/ml_pipeline` (missing).

## Output and Artifact Structure
The data pipeline uses run-scoped paths via `PipelinePathMixin`:
- Global raw inputs: `data/raw`
- Per-run outputs: `runs/{run_id}/data/` with subfolders:
  - clean
  - features
  - final
  - splits
  - artifacts
  - logs

The training orchestration (`UnifiedTrainingOrchestrator`) writes to:
- `output_dir / run_id` with OOF caches, trained model artifacts, etc.

These two output schemes are independent.

---

# Findings (Ordered by Severity)

## Critical
1) Invalid MTF mode in adapter breaks core -> data pipeline conversion.
- Evidence: `src/data/pipeline/config_adapter.py:40` sets `mtf_mode="aligned"`.
- Validation requires `mtf_mode` in `{"bars","indicators","both"}`: `src/data/pipeline/data_config.py:239`.
- Impact: Any use of `to_data_config()` will raise a `ValueError` and abort the run.

2) CLI "pipeline" commands reference a non-existent module.
- Evidence: `src/cli/commands/pipeline.py:39` and `src/cli/commands/pipeline.py:98` import `src.ml_pipeline`.
- `src/ml_pipeline` has no Python sources (directory is empty).
- Impact: `pipeline run`, `pipeline data`, and `pipeline status` commands will crash at import time.

## High
3) `project_root` defaults to `src/`, causing data path drift.
- Evidence: `src/data/pipeline/data_config.py:210` sets `project_root = Path(__file__).parent.parent.parent.resolve()` (the `src/` directory).
- Adapter sets `project_root = Path(config.output_dir).parent`: `src/data/pipeline/config_adapter.py:51`.
- Impact: `data/raw` will resolve to `src/data/raw` or `experiments/data/raw` instead of repo-level `data/raw`. This commonly results in missing input files or outputs written into source directories.

4) `start_date` / `end_date` are validated but never applied.
- Evidence: Config fields exist (`src/data/pipeline/data_config.py:83`), but Stage 2 loads and processes full data without filtering (`src/data/pipeline/stages/clean/pipeline.py:119`).
- Impact: date range settings are silently ignored; backtests or training can unintentionally include future data.

5) Gap filling fabricates bars and forward-fills volume with no calendar awareness.
- Evidence: `src/data/pipeline/stages/clean/utils.py:158` builds a full 1-min index, then `ffill` fills all columns including volume (`src/data/pipeline/stages/clean/utils.py:166`).
- Evidence: Stage 2 uses the simple gap filler, not the calendar-aware handler (`src/data/pipeline/stages/clean/pipeline.py:148`).
- Impact: missing sessions and weekend closures become synthetic bars with non-zero volume and flat prices; indicators and labels are materially distorted.

6) Multi-timeframe runs break at later stages (7.7, 8, 9).
- Stage 7.7 assumes scaled splits in `splits/scaled` instead of `splits/{tf}/scaled`: `src/data/pipeline/stages/scaled_validation/run.py:83`.
- Stage 8 assumes `combined_final_labeled.parquet` (single TF), which is not produced in multi-TF mode: `src/data/pipeline/stages/validation/run.py:51`.
- Stage 9 assumes `combined_final_labeled.parquet` and `splits/split_config.json` exist in root splits dir: `src/data/pipeline/stages/reporting/run.py:183` and `src/data/pipeline/stages/reporting/run.py:186`.
- Impact: multi-TF runs will fail or validate/report the wrong dataset.

7) Training orchestration expects labeled features but does not run the data pipeline.
- Evidence: `src/orchestrator.py:129` loads a parquet and immediately calls training; no Phase 1 stages are invoked.
- Training expects "Input DataFrame with features and labels": `src/models/training/unified_orchestrator.py:358`.
- Impact: If you pass raw OHLCV to `MLPipeline`, training will fail due to missing labels/features, or produce undefined behavior if labels are wrong.

## Medium
8) Hard-coded timezone in ingestion.
- Evidence: `src/data/pipeline/stages/ingest/run.py:80` passes `source_timezone="UTC"` with no config override.
- Impact: non-UTC vendor data is misaligned with sessions, splits, and labels.

9) Labeling depends on `atr_14`, but feature toggles can disable volatility features.
- Evidence: `src/data/pipeline/stages/features/run.py:83` reads feature toggles; `src/data/pipeline/stages/labeling/run.py:39` requires `atr_14`.
- Impact: valid configs can still fail at Stage 4.

10) `max_bars_ahead` is validated but not enforced in labeling or optimization.
- Evidence: `max_bars_ahead` exists and is validated (`src/data/pipeline/data_config.py:146`, `src/data/pipeline/config/pipeline_validation.py:53`), but `max_bars` in labeling defaults to `horizon * 3` and is not bounded by `max_bars_ahead` (`src/data/pipeline/stages/labeling/run.py:234`).
- Impact: actual lookahead window can exceed expectations; configuration semantics are misleading.

11) Adapter ignores training config choices.
- Evidence: `to_data_config` hard-codes `target_timeframe="5min"` and `scaler_type="standard"` (`src/data/pipeline/config_adapter.py:38`, `src/data/pipeline/config_adapter.py:47`).
- Impact: core PipelineConfig values for timeframe and scaling are ignored when bridging to the data pipeline.

12) Stage 8 validation runs feature selection on full data (train+val+test).
- Evidence: `validate_data` is called on the combined dataset; no split filtering occurs (`src/data/pipeline/stages/validation/run.py:66`).
- Impact: if you use Stage 8 outputs to pick features, you are selecting based on future data. This may be acceptable for research but is a leakage risk in production training.

13) Initial labeling defaults ignore symbol/horizon specific configs.
- Evidence: Stage 4 uses hard-coded `k_up=2.0`, `k_down=1.0` defaults when no override (`src/data/pipeline/stages/labeling/run.py:234`), rather than `barriers_config` defaults.
- Impact: label distributions can be skewed early, making GA optimization less stable and less consistent with policy.

## Low
14) GA safe optimization uses only `train_ratio`, not purge/embargo.
- Evidence: `run_optuna_optimization_safe` slices by `train_ratio` (`src/data/pipeline/stages/ga_optimize/optuna_optimizer.py:528`), while splits later add purge/embargo (`src/data/pipeline/stages/splits/core.py:242`).
- Impact: optimization can include data that training later discards; subtle mismatch in effective training distribution.

15) Documentation drift.
- Evidence: `docs/reference/PIPELINE_STAGES.md` describes stages not present in `stage_registry.py` (sessions stage, etc.).
- Impact: mental model and actual behavior diverge, leading to misconfiguration.

---

# Stage-by-Stage Deep Dive (Behavior + Risks)

## Stage 1: Ingest (`data_generation`)
**Inputs**: `data/raw/{symbol}_1m.parquet` or `.csv`
**Outputs**: `data/raw/validated/{symbol}_1m_validated.parquet`
**Core operations**:
- Standardize OHLCV columns, validate types, normalize timestamps, fix OHLC violations.
- Adds symbol column if missing.

**Observations**:
- Validated data is stored in a global folder (`data/raw/validated`), not run-scoped.
- `source_timezone` is fixed to UTC and cannot be configured.

**Risks**:
- Changing raw data between runs will change validated files and silently alter downstream outputs.
- Non-UTC data becomes misaligned without explicit timezone control.

## Stage 2: Clean (`data_cleaning`)
**Inputs**: `data/raw/validated/{symbol}_1m_validated.parquet` (fallback to raw if missing)
**Outputs**: `runs/{run_id}/data/clean/{symbol}_{tf}_clean.parquet`
**Core operations**:
- Gap detection, gap filling, resampling to target TF(s).
- Adds roll flags and session_id.

**Observations**:
- Uses `fill_gaps_simple` (ffill on a 1-min index) even when the more robust `GapHandler` exists.
- `max_gap_minutes` is only configurable via `getattr` default, not a formal DataConfig field.

**Risks**:
- Synthetic bars across closures and forward-filled volume bias downstream indicators and labels.
- Inability to configure gap policy from config limits control for different instruments.

## Stage 3: Feature Engineering (`feature_engineering`)
**Inputs**: `runs/{run_id}/data/clean/{symbol}_{tf}_clean.parquet`
**Outputs**: `runs/{run_id}/data/features/{symbol}_{tf}_features.parquet`
**Core operations**:
- Generates a large, mostly fixed feature superset.
- MTF features are added for higher timeframes only (per current TF).

**Observations**:
- `feature_generation` config does not directly change the number of generated features; it only affects feature set selection later.
- Feature toggles can disable volatility or wavelet features, but there is no validation that labeling requirements are still met.

**Risks**:
- Mismatches between feature generation and labeling prerequisites (ATR).
- Feature superset generation is computationally heavy even when a smaller set is desired.

## Stage 4: Initial Labeling (`initial_labeling`)
**Inputs**: `features/{symbol}_{tf}_features.parquet`
**Outputs**: `runs/{run_id}/data/labels/{symbol}_{tf}_labels_init.parquet`
**Core operations**:
- Triple barrier labeling for each horizon using fixed defaults unless overrides are set.
- Records label provenance metadata.

**Observations**:
- Defaults are fixed (k_up=2.0, k_down=1.0) and ignore symbol/horizon defaults from `barriers_config`.
- No enforcement of `max_bars_ahead` in labeling logic.

**Risks**:
- Default barrier choices may be inconsistent with your stated policy.
- Horizons can imply lookahead windows larger than intended.

## Stage 5: GA / Optuna Optimization (`ga_optimize`)
**Inputs**: `labels_init.parquet`
**Outputs**: `runs/{run_id}/artifacts/ga_results/*_ga_h{h}_best.json`
**Core operations**:
- Optuna TPE search for k_up/k_down/max_bars using a subset of training data.

**Observations**:
- Safe mode uses the first `train_ratio` portion of data, ignoring purge/embargo.
- Stores validation fitness computed on full data (after optimization) for reporting only.

**Risks**:
- The reported best_fitness uses data that would be "future" relative to training boundaries.

## Stage 6: Final Labels (`final_labels`)
**Inputs**: `features/{symbol}_{tf}_features.parquet`, GA results
**Outputs**: `final/{symbol}_{tf}_labeled.parquet`
**Core operations**:
- Applies GA-optimized (or default) barrier params to generate labels and quality metrics.

**Observations**:
- CLI barrier overrides apply globally to all horizons.
- Default fallback uses fixed values rather than symbol/horizon-specific defaults.

**Risks**:
- Global overrides can produce uneven label distributions across horizons.

## Stage 7: Splits (`create_splits`)
**Inputs**: `final/{symbol}_{tf}_labeled.parquet`
**Outputs**: `splits/{tf}/train_indices.npy` etc.
**Core operations**:
- Chronological split with purge and embargo buffers.
- Saves combined labeled data for each timeframe.

**Observations**:
- Pipeline enforces single-symbol by default; multi-symbol is allowed only with explicit override.
- Combines symbol DataFrames by time if batch symbols are allowed.

**Risks**:
- If multiple symbols have different calendars, combined chronological splits can interleave bars in ways that are not comparable.

## Stage 7.5: Scaling (`feature_scaling`)
**Inputs**: `combined_final_labeled.parquet` + split indices
**Outputs**: `splits/{tf}/scaled/*_scaled.parquet` and scaler artifacts
**Core operations**:
- Fits scalers on training split only, then transforms val/test.

**Observations**:
- Identifies feature columns by exclusion list, which is reasonable but not tied to feature set definitions.
- Writes per-timeframe outputs when multi-TF is enabled.

**Risks**:
- If feature toggles remove columns referenced by downstream feature sets, later stages fail.

## Stage 7.6: Build Datasets (`build_datasets`)
**Inputs**: `splits/{tf}/scaled/*.parquet`
**Outputs**: dataset folders by feature set and horizon
**Core operations**:
- Validates schema consistency across splits.
- Writes per-feature-set datasets for each horizon.

**Observations**:
- Correctly validates feature schema and fails early if mismatched.
- Uses `feature_generation` to decide which feature sets to export.

**Risks**:
- If `feature_generation` requests features that were disabled by toggles, it fails during `validate_feature_set_columns`.

## Stage 7.7: Scaled Validation (`validate_scaled`)
**Inputs**: `splits/scaled/*.parquet`
**Outputs**: drift reports
**Observations**:
- Not multi-TF aware. In multi-TF runs, scaled data are stored under `splits/{tf}/scaled`.

**Risks**:
- Fails outright or validates the wrong data in multi-TF configurations.

## Stage 8: Validation (`validate`)
**Inputs**: `final/combined_final_labeled.parquet`
**Outputs**: validation report + optional feature selection report

**Observations**:
- Uses combined data (train+val+test) for feature selection.
- Data contract checks exist but are not used in `validate_data`.

**Risks**:
- Not multi-TF aware.
- Feature selection can leak future data if used to drive training.

## Stage 9: Reporting (`generate_report`)
**Inputs**: `final/combined_final_labeled.parquet`, `splits/split_config.json`
**Outputs**: markdown report

**Observations**:
- Assumes single TF file locations.
- Example paths in the report are not run-scoped.

**Risks**:
- Not usable in multi-TF mode.

---

# Cross-Cutting Architecture Issues

## Parallel Pipelines (Phase 1 vs Phase 2)
- The data pipeline (Phase 1) generates labeled, scaled, split datasets and drift reports.
- The training orchestrator (Phase 2) expects a DataFrame with features and labels already present.
- There is no bridge that calls Phase 1 automatically from `src/orchestrator.py`.

Practical implication: You must either
1) Run Phase 1 manually and feed its outputs into training, or
2) Pass already prepared data into `MLPipeline`, or
3) Extend `MLPipeline` to invoke the Phase 1 pipeline first.

## Multi-Timeframe Inconsistency
- Stages 2-7.6 are multi-TF aware.
- Stages 7.7, 8, and 9 are not multi-TF aware and assume single-file paths.

Practical implication: Multi-TF runs will likely fail after scaling, or silently validate/report only the single-TF combined file.

## Config Semantics Gaps
- `feature_generation` describes generation scope but is only used for dataset export.
- `max_bars_ahead` suggests a lookahead cap but is not enforced in labeling.
- `target_timeframe` and `scaler_type` are ignored by the core-to-data adapter.

Practical implication: configuration does not reliably describe actual pipeline behavior without manual cross-checks.

---

# OHLCV Model Training Standards and Best Practices (Industry Baselines)

These are standard conventions across quantitative finance and regulated environments. They are not legal advice, but they are common expectations for OHLCV modeling pipelines.

## A) Data Governance and Model Risk Management
- **BCBS 239**: Principles for risk data aggregation and reporting (accuracy, completeness, timeliness).
- **SR 11-7 / OCC 2011-12**: Model risk management expectations (data governance, validation, monitoring).
- **EBA/ECB MRM guidance** (EU contexts): emphasis on data quality, traceability, and robustness.

## B) Market Data Identifier Standards
- **ISO 8601** for timestamps (including timezone offsets).
- **ISO 10383 (MIC)** for exchange/venue identifiers.
- **ISO 4217** for currency codes.
- **ISO 10962 (CFI)** and **ISO 6166 (ISIN)** for instrument classification and identification.

## C) OHLCV Bar Construction Standards
- **Bar alignment**: Define whether bars are left-closed (timestamp at bar start) or right-closed (timestamp at bar end). Use one convention consistently.
- **Aggregation rules**:
  - Open = first trade
  - High = max trade
  - Low = min trade
  - Close = last trade
  - Volume = sum of executed trade size
- **Volume integrity**: do not forward-fill volume. For synthetic bars, set volume to zero and flag the bar.

## D) Timestamp and Calendar Standards
- Use exchange calendar logic (holidays, early closes, maintenance windows).
- Explicitly document DST handling and timezone conversion.
- For futures (CME Globex), daily maintenance and weekend closures must be respected; missing bars should remain missing unless explicitly backfilled by the vendor.

## E) Futures-Specific Standards
- **Continuous contract methodology** must be explicit (back-adjusted, ratio-adjusted, or unadjusted roll).
- **Roll detection** should be based on volume/open interest or explicit roll schedules, not only price gaps.
- Keep raw contract identifiers in metadata if you are stitching multiple expiries.

## F) Labeling and Leakage Control
- Label horizons must be consistent with split boundaries (purge and embargo).
- Any hyperparameter search (barrier tuning, feature selection) should be done only on training data.
- Document labeling assumptions: entry at next bar open, stop/target definitions, cost/slippage assumptions.

## G) Split and Evaluation Standards
- **Chronological splits** are the default for time series.
- **Purging and embargo** should scale with the maximum label horizon and bar interval.
- Report actual split date ranges and percentage loss due to purge/embargo.

## H) Feature Engineering Standards
- Feature calculations should be strictly causal (no use of future bars).
- Maintain explicit feature categories and stable naming conventions.
- Track which features are enabled per run when toggles are used.

## I) Data Quality Checks (Minimum Set)
- Duplicate timestamps, missing bars, and NaN/Inf checks.
- OHLC integrity: high >= low, high >= open/close, low <= open/close.
- Outlier detection (z-score, IQR, ATR-based).
- Label distribution sanity (avoid extreme class imbalance unless intentional).

---

# If You Are Building an ML Factory: Operational Implications

## What Works Today
- The Phase 1 data pipeline is operational for single-TF runs when you point it at the correct `data/raw` directory.
- The labeling + GA optimization flow is functional and produces provenance metadata.
- The scaling stage correctly fits on train only and validates leakage.

## What Will Break or Produce Inconsistent Outputs
- Multi-timeframe runs will fail at validation and reporting.
- Training orchestration will not work on raw OHLCV input without manual feature/label preparation.
- CLI pipeline commands will fail because `src/ml_pipeline` is missing.
- Configuration is not consistent across the core and data pipelines.

## Minimum Fixes to Stabilize the Factory (No MLflow, No Tests)
1) Fix the adapter MTF mode and honor `target_timeframe` and `scaler_type`.
2) Align `project_root` to repo root so `data/raw` resolves correctly.
3) Decide whether `start_date` / `end_date` should be enforced; if yes, filter at Stage 2.
4) Replace `fill_gaps_simple` with calendar-aware gap handling and volume-safe synthetic bars.
5) Make Stage 7.7/8/9 multi-TF aware or explicitly disable them when multi-TF is enabled.
6) Provide a single entry point that runs Phase 1 then Phase 2, or clearly document the manual handoff.

---

# Evidence Index (Quick Reference)
- Adapter invalid MTF mode: `src/data/pipeline/config_adapter.py:40`
- MTF mode validation: `src/data/pipeline/data_config.py:239`
- CLI imports missing module: `src/cli/commands/pipeline.py:39`, `src/cli/commands/pipeline.py:98`
- project_root default: `src/data/pipeline/data_config.py:210`
- start/end date fields: `src/data/pipeline/data_config.py:83`
- date filtering absent: `src/data/pipeline/stages/clean/pipeline.py:119`
- gap fill ffill + 1-min index: `src/data/pipeline/stages/clean/utils.py:158`, `src/data/pipeline/stages/clean/utils.py:166`
- Stage 7.7 single-TF path: `src/data/pipeline/stages/scaled_validation/run.py:83`
- Stage 8 single-TF path: `src/data/pipeline/stages/validation/run.py:51`
- Stage 9 single-TF path: `src/data/pipeline/stages/reporting/run.py:183`
- Training expects labeled features: `src/models/training/unified_orchestrator.py:358`
- Orchestrator loads parquet only: `src/orchestrator.py:167`
- ATR requirement: `src/data/pipeline/stages/labeling/run.py:39`
- Volatility toggle: `src/data/pipeline/stages/features/run.py:83`
- Initial labeling defaults: `src/data/pipeline/stages/labeling/run.py:234`
- max_bars_ahead validation: `src/data/pipeline/config/pipeline_validation.py:53`
