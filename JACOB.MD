JACOB - SRC PIPELINE REVIEW (2026-02-06)

Reviewed core pipeline orchestration and key stages. Findings below are concrete issues or mismatches observed in code.

Issues and Mismatches
- Stage 3 (feature engineering) can succeed even when some symbol/timeframe tasks fail; failures are logged but pipeline still proceeds without required artifacts. This risks silent data gaps.
  - Location: src/data/pipeline/stages/features/run.py
- Stage timeouts are configured in DataConfig but never enforced in PipelineRunner. The pipeline can hang despite timeout settings.
  - Locations: src/data/pipeline/data_config.py, src/data/pipeline/runner.py
- Stage transition validation exists but is never invoked. Only per-stage schema validation runs; transition requirements are unused.
  - Location: src/data/pipeline/schemas.py (validate_stage_transition)
  - Runner only calls: _validate_stage_output in src/data/pipeline/runner.py
- Documentation is stale for stages. README references files that no longer exist under current stages layout.
  - Location: src/data/pipeline/stages/README.md
  - References: stage7_splits.py, stage8_validate.py, baseline_backtest.py
- Stage 10 (Evaluation) is exported in stages __init__ but not in the stage registry, so it never runs in the pipeline.
  - Locations: src/data/pipeline/stages/__init__.py, src/data/pipeline/stage_registry.py

Notes
- No TODO/FIXME/HACK markers found under src/data/pipeline.
