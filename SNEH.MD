# SNEH.MD
## Purpose
- Provide a meticulous, actionable fix list for SRC and repository robustness.
- Capture config conflicts, hard coded defaults, and hidden assumptions.
- Define a consistent format so fixes can be tracked and validated.
- Serve as a single reference for high priority reliability work.
## Scope
- Source code under src/ and pipeline stages.
- Configuration under config/ and CLI defaults.
- Training and inference pipelines.
- Documentation, testing, and operational gaps.
## Document format rules
- Each issue has an ID, severity, problem, evidence, impact, and fix.
- Evidence references file paths or behaviors, not opinions.
- Fixes are phrased as changes that can be verified.
- No filler or blank line padding.
- Use ASCII and consistent headings for scan-ability.
## Current state summary
- Multiple config sources exist for models, features, and defaults.
- Timeframe handling and MTF support are partially unified.
- Feature selection exists but is not clearly wired end to end.
- Docs and implementation status disagree in several places.
- There is no single canonical registry for model requirements.
## Canonical decisions needed
- Pick the authoritative source for model defaults.
- Pick the authoritative source for feature set definitions.
- Establish a single authoritative pipeline config (editable in Jupyter) and deprecate all other defaults.
- Normalize timeframes to canonical strings across the pipeline.
- Define how feature selection is configured and persisted.
- Define a rule for config precedence and surface it in logs.
## Issue catalog
## Category: Configuration and Defaults
### CFG-001: Competing model defaults across code and YAML
- Severity: High
- Problem: model defaults are defined in multiple places and can diverge.
- Evidence: src/phase1/config/model_config.py, config/models/*.yaml, src/models/config/trainer_config.py.
- Impact: training can run with untracked defaults, reducing reproducibility.
- Fix: choose one canonical source and generate or validate the others.
### CFG-002: Pipeline feature_set default conflicts with training default
- Severity: High
- Problem: pipeline uses feature_set="full" while trainer defaults to "boosting_optimal".
- Evidence: src/phase1/pipeline_config.py, src/models/config/trainer_config.py.
- Impact: dataset build and model training can use different feature sets silently.
- Fix: align defaults or require explicit selection at runtime.
### CFG-011: No single authoritative pipeline config
- Severity: High
- Problem: config is scattered across code defaults, YAMLs, and CLI flags with no single source of truth.
- Evidence: src/phase1/pipeline_config.py, config/models/*.yaml, src/models/config/trainer_config.py.
- Impact: notebook edits cannot reliably change end-to-end behavior.
- Fix: define one canonical config file (notebook-editable) and generate/validate all other configs from it.
### CFG-003: Legacy feature_set comments are misleading
- Severity: Medium
- Problem: pipeline_config describes feature_set as "full/minimal/custom" but actual sets are different.
- Evidence: src/phase1/pipeline_config.py and src/phase1/config/feature_sets.py.
- Impact: users pass invalid values or assume wrong feature coverage.
- Fix: update comments and validation messages to match current definitions.
### CFG-004: Timeframe validation uses multiple registries
- Severity: High
- Problem: timeframes are validated in more than one module.
- Evidence: src/common/timeframes.py and src/phase1/config/features.py.
- Impact: new timeframes can pass in one place and fail in another.
- Fix: route all validation through a single timeframe module.
### CFG-005: CLI examples include timeframes rejected by validation
- Severity: High
- Problem: CLI mentions 4h and daily while validators do not accept them.
- Evidence: src/cli/run_commands_pipeline.py, src/common/timeframes.py.
- Impact: CLI appears to support values that will fail at runtime.
- Fix: either support the values in validation or remove them from docs and help.
### CFG-006: Output timeframes list uses 60min but alias 1h appears elsewhere
- Severity: Medium
- Problem: output_timeframes uses 60min while other lists include 1h.
- Evidence: src/cli/run_commands_core.py and src/common/timeframes.py.
- Impact: file naming can drift or produce mixed suffixes across stages.
- Fix: normalize all timeframe inputs to canonical strings at the boundary.
### CFG-007: MTF default list includes aliases without de-duplication
- Severity: Medium
- Problem: MTF lists can include both 1h and 60min.
- Evidence: src/phase1/stages/mtf/constants.py and src/common/timeframes.py.
- Impact: duplicate features or wasted compute.
- Fix: normalize and de-duplicate MTF lists before use.
### CFG-008: Environment overrides are not surfaced in run metadata
- Severity: Medium
- Problem: environment overrides can change config without visibility.
- Evidence: src/models/config/merging.py.
- Impact: config drift is hard to diagnose and reproduce.
- Fix: log and persist applied overrides in run artifacts.
### CFG-009: Feature selection YAML configs are unused
- Severity: Low
- Problem: config/features/model_features.yaml and selection_methods.yaml appear unused.
- Evidence: no references in src/.
- Impact: stale configs can mislead users and docs.
- Fix: wire them in or archive them explicitly with a notice.
### CFG-010: mtf_disable mapped to mtf_enable is not documented
- Severity: Low
- Problem: CLI uses mtf_disable which flips to mtf_enable in core.
- Evidence: src/cli/run_commands_pipeline.py.
- Impact: behavior is correct but unclear to users.
- Fix: document the mapping in CLI help or rename for clarity.
## Category: Models and Training
### MOD-001: Model requirements list is incomplete
- Severity: High
- Problem: model_config does not include several models present in config/models.
- Evidence: src/phase1/config/model_config.py vs config/models/*.yaml.
- Impact: Phase 1 prep does not fully represent Phase 2 training needs.
- Fix: add missing models to model_config or retire the file.
### MOD-002: patchtst feature_set mismatch
- Severity: Medium
- Problem: patchtst uses transformer_raw in model_config but neural_optimal in YAML.
- Evidence: src/phase1/config/model_config.py and config/models/patchtst.yaml.
- Impact: data preparation and training may disagree on feature coverage.
- Fix: align patchtst feature_set across all sources.
### MOD-003: Model family definitions are spread across layers
- Severity: Medium
- Problem: model families are defined in registry, model_config, and YAML.
- Evidence: src/models/registry.py, src/phase1/config/model_config.py, config/models/*.yaml.
- Impact: inconsistent family labels can break routing and feature selection.
- Fix: centralize family classification and validate on load.
### MOD-004: Sequence length defaults differ by location
- Severity: Medium
- Problem: sequence_length defaults vary between trainer and model_config.
- Evidence: src/models/config/trainer_config.py and src/phase1/config/model_config.py.
- Impact: models may train with unintended window sizes when YAML is absent.
- Fix: enforce sequence_length via model requirements or require explicit config.
### MOD-005: Trainer does not validate feature_set against model requirements
- Severity: Medium
- Problem: trainer accepts any feature_set string without model-family checks.
- Evidence: src/models/trainer.py resolves feature_set but does not validate compatibility.
- Impact: incompatible feature sets can be used without warning.
- Fix: add validation against model_config requirements.
### MOD-006: Feature selection behavior depends on model.requires_sequences
- Severity: Low
- Problem: feature selection is disabled for sequence models regardless of config.
- Evidence: src/models/trainer.py.
- Impact: user intent may be ignored without explicit warning.
- Fix: log a clear message and allow opt-in for advanced users.
### MOD-007: Ensemble base model compatibility is not enforced at config time
- Severity: Medium
- Problem: base model lists can mix incompatible data shapes.
- Evidence: src/models/trainer.py heterogeneous checks occur at runtime.
- Impact: failure happens late after data prep.
- Fix: validate base model compatibility during config creation.
### MOD-008: Training artifacts do not record model requirement versions
- Severity: Low
- Problem: run artifacts do not persist which model_config version was used.
- Evidence: no explicit metadata file ties to model_config entries.
- Impact: hard to reproduce training with evolving defaults.
- Fix: store a normalized requirements snapshot in artifacts.
## Category: Feature Engineering and MTF
### FEAT-001: Feature engineering docstring conflicts with symbol isolation policy
- Severity: Low
- Problem: docstring references cross-asset features despite isolation policy.
- Evidence: src/phase1/stages/features/run.py and src/phase1/config/features.py.
- Impact: docs mislead users and reviewers.
- Fix: update docstring to reflect symbol isolation.
### FEAT-002: MTF base_timeframe in config is not tied to target_timeframe
- Severity: Medium
- Problem: MTF_CONFIG base_timeframe is static while pipeline target_timeframe is dynamic.
- Evidence: src/phase1/config/features.py and src/phase1/pipeline_config.py.
- Impact: MTF assumptions can drift from pipeline settings.
- Fix: derive MTF base_timeframe from pipeline config and normalize once.
### FEAT-003: MTF timeframe parsing does not handle daily values
- Severity: Medium
- Problem: parse_timeframe_to_minutes accepts min and h, not daily.
- Evidence: src/phase1/config/features.py and src/common/timeframes.py.
- Impact: daily values in docs or configs fail at runtime.
- Fix: add daily parsing or remove daily from supported lists.
### FEAT-004: Feature sets are prefix based and fragile
- Severity: Medium
- Problem: feature set definitions rely on name prefixes and can miss new columns.
- Evidence: src/phase1/config/feature_sets.py and src/phase1/utils/feature_sets.py.
- Impact: features can silently drop when naming changes.
- Fix: add tests and a manifest-based mapping for feature sets.
### FEAT-005: Feature toggles are not recorded in dataset manifests
- Severity: Low
- Problem: toggles change feature generation but are not embedded in artifacts.
- Evidence: src/phase1/stages/features/run.py and datasets manifest output.
- Impact: feature provenance is incomplete.
- Fix: persist toggles in feature_set_manifest or run metadata.
### FEAT-006: MTF feature suffixes vary by timeframe alias
- Severity: Medium
- Problem: suffixes use 1h or 60m depending on input.
- Evidence: src/phase1/stages/features/engineer.py and timeframes lists.
- Impact: downstream feature set resolution can be inconsistent.
- Fix: canonicalize suffixes after normalization.
### FEAT-007: MTF enable logic depends on data length threshold
- Severity: Low
- Problem: MTF feature generation is skipped when rows < 500.
- Evidence: src/phase1/stages/features/engineer.py.
- Impact: short runs may produce different feature sets without warning.
- Fix: log a structured warning and add a manifest flag.
### FEAT-008: Feature set coverage for sequence models is unclear
- Severity: Medium
- Problem: transformer_raw and neural_optimal definitions overlap without guidance.
- Evidence: src/phase1/config/feature_sets.py.
- Impact: user choices are inconsistent across experiments.
- Fix: document recommended sets per model and enforce in trainer.
## Category: Labeling and Cross Validation
### LBL-001: Labeling depends on ATR feature
- Severity: High
- Problem: labeling requires atr_14 to exist in features.
- Evidence: src/phase1/stages/labeling/run.py.
- Impact: feature set changes can break labeling.
- Fix: validate required features before labeling and add clear error message.
### LBL-002: Barrier overrides lack provenance in initial labeling
- Severity: Medium
- Problem: initial labels apply overrides but do not store source metadata.
- Evidence: src/phase1/stages/labeling/run.py.
- Impact: audit trail is incomplete.
- Fix: persist source and override values in labeling metadata.
### LBL-003: GA results naming may not match timeframe aliases
- Severity: Medium
- Problem: GA files include timeframe strings and can mismatch alias usage.
- Evidence: src/phase1/stages/final_labels/run.py and timeframe handling.
- Impact: GA results may not be found for a given alias.
- Fix: normalize timeframes before file naming and lookup.
### LBL-004: Purge and embargo scaling depends on timeframe conversion
- Severity: Medium
- Problem: purge and embargo rely on timeframe conversion that is not centralized.
- Evidence: src/common/horizon_config.py and parse_timeframe_to_minutes.
- Impact: leakage controls can be inconsistent across timeframes.
- Fix: use canonical timeframe conversion for all purge and embargo scaling.
### LBL-005: Horizon validation does not check data length
- Severity: Low
- Problem: horizon values can exceed available bars without early warning.
- Evidence: src/phase1/config/features.py validate_horizons.
- Impact: labeling quality degrades or fails late.
- Fix: add data length checks before labeling.
### LBL-006: CV feature selection default behavior is unclear
- Severity: Medium
- Problem: CV can tune on full feature set before selection by default.
- Evidence: src/cross_validation/cv_runner.py.
- Impact: potential mismatch between tuning and final feature set.
- Fix: make tune_per_fold explicit in config and docs.
## Category: Data Pipeline and IO
### DATA-001: Timeframe aliases can break file discovery
- Severity: High
- Problem: files are named with timeframe strings that may vary by alias.
- Evidence: stage run files use {symbol}_{tf}_*.parquet naming.
- Impact: downstream stages fail to find upstream outputs.
- Fix: normalize timeframes before file naming and access.
### DATA-002: Raw data fallback bypasses schema validation
- Severity: Medium
- Problem: cleaning stage uses raw files when validated files are missing.
- Evidence: src/phase1/stages/clean/run.py.
- Impact: inconsistent input schema and quality.
- Fix: enforce minimal schema checks even on raw fallback.
### DATA-003: Metadata columns are not tightly controlled
- Severity: Low
- Problem: auto feature detection uses METADATA_COLUMNS list.
- Evidence: src/phase1/stages/datasets/container.py.
- Impact: new metadata columns can leak into features.
- Fix: version and validate metadata columns explicitly.
### DATA-004: Artifact manifests do not include config snapshot by default
- Severity: Medium
- Problem: artifacts list paths but not a complete config snapshot.
- Evidence: StageResult metadata is partial.
- Impact: limited reproducibility for runs.
- Fix: persist a full config.json with each run.
## Category: Documentation and Process
### DOC-001: MTF status and timeframe counts are inconsistent across docs
- Severity: Medium
- Problem: docs claim different MTF completeness and counts.
- Evidence: docs/README.md and docs/planning/PROJECT_CHARTER.md.
- Impact: readers get conflicting project status.
- Fix: align docs with current implementation and keep a single source of truth.
### DOC-002: Model inventory is not anchored to registry
- Severity: Medium
- Problem: docs list model counts that can drift from registry.
- Evidence: docs/README.md and registry list.
- Impact: wrong model counts in planning and reporting.
- Fix: auto-generate inventory from registry or config.
### DOC-003: Feature set guidance is spread across docs and archives
- Severity: Low
- Problem: multiple documents define feature selection rules.
- Evidence: docs/guides/FEATURE_ENGINEERING.md and docs/archive.
- Impact: users follow outdated guidance.
- Fix: consolidate guidance and mark archive as historical.
### DOC-004: Fix plans are not tied to test coverage
- Severity: Low
- Problem: fixes are listed without required tests.
- Evidence: SNEH fix plans and analysis docs.
- Impact: fixes ship without regression coverage.
- Fix: add a test requirement line for each fix item.
### DOC-005: Prune redundant or outdated docs
- Severity: Low
- Problem: multiple overlapping docs and archives are not clearly marked as historical.
- Evidence: docs/archive, docs/analysis, FINDINGS 1.MD, FINDINGS 2.md, CONFIGURATION_ANALYSIS.md, SNEH FIX PLAN.md.
- Impact: users follow obsolete guidance or duplicate fix plans.
- Fix: consolidate to one active fix plan and mark or move older docs to an explicit archive.
## Category: Testing and Validation
### TEST-001: No tests for config precedence
- Severity: High
- Problem: precedence between defaults, YAML, env, and CLI is not tested.
- Evidence: src/models/config/merging.py.
- Impact: regressions can silently change training behavior.
- Fix: add unit tests for precedence and override visibility.
### TEST-002: No tests for feature set resolution
- Severity: Medium
- Problem: feature set aliases and prefix resolution lack tests.
- Evidence: src/phase1/config/feature_sets.py and utils.
- Impact: feature sets can silently resolve to empty lists.
- Fix: add tests for each alias and required columns.
### TEST-003: No tests for labeling preconditions
- Severity: Medium
- Problem: labeling assumes ATR exists with no guard tests.
- Evidence: src/phase1/stages/labeling/run.py.
- Impact: feature set changes cause runtime failures.
- Fix: add tests that validate required features for labeling.
### TEST-004: Update tests for unified config and feature sets
- Severity: Medium
- Problem: tests do not assert the single-config contract or consistent feature set resolution.
- Evidence: missing tests in tests/ for pipeline config + trainer config alignment.
- Impact: regressions can reintroduce config drift and silent feature mismatch.
- Fix: add tests that validate canonical config propagation into pipeline, datasets, and trainer.
## Category: Operations and Observability
### OPS-001: Logging is inconsistent across stages
- Severity: Low
- Problem: stage logs are human readable but not structured.
- Evidence: stage run logs across src/phase1/stages.
- Impact: log parsing and monitoring are harder.
- Fix: add structured fields for run_id, stage, timeframe, and symbol.
### OPS-002: Dependency management lacks a recorded policy
- Severity: Low
- Problem: dependency updates and pinning strategy are not documented.
- Evidence: requirements.txt and pyproject.toml without policy doc.
- Impact: security and reproducibility risks.
- Fix: add a dependency update policy and periodic audit cadence.
## Fix sequencing plan
- Phase 0: Define canonical config sources and precedence rules.
- Phase 1: Normalize timeframes and file naming across stages.
- Phase 2: Align model requirements, YAML configs, and trainer defaults.
- Phase 3: Harden feature set resolution and feature selection pipelines.
- Phase 4: Update labeling preconditions and CV safeguards.
- Phase 5: Update docs, inventories, and automation.
- Phase 6: Add tests and CI gates for all critical paths.
## Validation plan
- Add unit tests for config precedence and alias resolution.
- Add integration tests for multi-TF output with alias normalization.
- Add labeling preflight checks for required columns and horizons.
- Add dataset build checks for feature set coverage and manifests.
- Add smoke tests for model training with each feature set.
- Add regression checks for MTF output filenames.
## Docs to prune (initial list)
- docs/archive (move to explicit historical references or delete).
- docs/analysis (consolidate to one active analysis doc).
- FINDINGS 1.MD and FINDINGS 2.md (merge into a single findings doc).
- CONFIGURATION_ANALYSIS.md (fold into the canonical config decision doc).
- SNEH FIX PLAN.md (superseded by this document or merged here).
## Tests to update (initial list)
- Config precedence and single-config propagation tests (pipeline -> datasets -> trainer).
- Feature set alias resolution and zero-feature fallback behavior.
- Timeframe normalization for output filenames and MTF paths.
- Labeling prerequisites when feature sets change.
- Dataset manifest content (feature toggles, config snapshot, and feature set manifest).
## Definition of done
- All configuration sources are either canonical or generated.
- One configurable config set drives the full pipeline and is notebook-editable.
- All timeframes are normalized to a canonical string before use.
- All feature sets resolve deterministically and are test covered.
- Training artifacts include a config snapshot and feature set manifest.
- Docs are aligned with the code and auto-generated where possible.
- CI blocks merges that break config precedence or labeling preconditions.
## Open questions
- Decision: use a single configurable config set as the authoritative source, editable from Jupyter when running the pipeline.
- Should 4h and daily be supported as first class timeframes.
- Should feature selection be enabled by default for non-sequence models.
- Do we want strict failure when feature sets resolve to zero columns.
- Should pipeline enforce a required feature list for labeling and CV.
## Change log
- 2026-01-14: Expanded SNEH.MD with structured issue catalog and plans.
