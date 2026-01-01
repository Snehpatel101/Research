# Phase 1 Pipeline Fixes and Alignment Notes

This doc captures the concrete fixes applied to align the Phase 1 pipeline with
current behavior, paths, and configurability. It also documents the remaining
assumptions and verification steps.

## What Was Fixed

1. **CLI stage alignment**
   - `pipeline rerun --from labeling` now maps to `initial_labeling`.
   - Added common aliases for `final_labels`, `ga_optimize`, `feature_scaling`,
     and `build_datasets`.
   - `pipeline status` now pulls stage definitions from the stage registry,
     so progress counts and stage names match the actual pipeline.

2. **Labeling report output path + horizons**
   - `generate_labeling_report` now writes to `config.results_dir` instead of
     `src/phase1/results`.
   - The report iterates over configured horizons rather than `[5, 20]`.

3. **Regime-adaptive labeling support**
   - Added `REGIME_CONFIG`, `REGIME_BARRIER_ADJUSTMENTS`, and
     `get_regime_adjusted_barriers`.
   - Adaptive labeling now has real adjustment logic instead of falling back
     to defaults.

4. **Project root alignment**
   - `PipelineConfig.project_root` now defaults to repo root (not `src/`).
     This keeps `data/`, `runs/`, and `results/` in the intended locations.

5. **Unified purge/embargo auto-scaling**
   - Phase 1 now delegates to `src.common.horizon_config.auto_scale_purge_embargo`.
     This eliminates drift between multiple implementations.

6. **Compatibility config facade**
   - Added `src/config.py` and `src/phase1/config/runtime.py` as a single, stable
     import surface for tests and standalone scripts.

## Config Defaults (Current)

- Horizons: `[5, 10, 15, 20]`
- Auto purge/embargo: derived from `src.common.horizon_config`
- Target timeframe: `5min`
- Split ratios: `train=0.70`, `val=0.15`, `test=0.15`

If you want to override these, prefer `PipelineConfig` or a config file for
runs; the `config` facade is meant for compatibility and defaults.

## Verification Commands

- Targeted tests used during validation:

```bash
pytest tests/test_dynamic_horizons.py \
  tests/phase_1_tests/stages/test_regime_detection.py \
  tests/phase_1_tests/stages/test_stage2_mtf_resampling.py -q
```

- Optional CLI smoke checks (if data is present):

```bash
./pipeline status <run_id>
./pipeline rerun <run_id> --from initial_labeling
```

## Notes for Older References

- Replace `labeling` with `initial_labeling` or `final_labels` in scripts.
- If you used `config` imports before, they now resolve via `src/config.py`.

