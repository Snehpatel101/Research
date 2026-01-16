# Oracle Architecture Review - 2026-01-15

## Executive Summary

The ML Model Factory architecture is **unusually strong for a bespoke trading ML codebase**. Core factory separation (canonical data → deterministic transforms → model adapters → unified training/eval) plus serious leakage defenses are well implemented.

## Key Architectural Strengths

1. **Single Canonical Source + Deterministic Derivations**
   - 1m OHLCV "single source of truth"
   - Derived MTF ladder reduces data inconsistency

2. **Model Plugin Registry + Unified Contracts**
   - `src/models/registry.py` + `BaseModel` keeps extensibility high

3. **Leakage-Aware CV + OOF as First-Class**
   - `src/cross_validation/purged_kfold.py` with timeframe-aware embargo
   - OOF generation/caching aligned with professional quant ML

4. **Artifact Integrity + Reproducibility Hooks**
   - Phase-1 artifact manifests/checksums (`src/common/manifest.py`)
   - Model artifact checksums (`src/models/training/checksums.py`)

5. **Train/Serve Parity Foundations**
   - `src/inference/bundle.py` and `src/inference/preprocessing_graph.py`

## Critical Gaps (Priority Order)

### 1. Lineage Unification (Medium: 1-2 days)
- Require training runs to reference pipeline run artifact set
- Store `{pipeline_run_id, dataset_hash, feature_set_name, timeframe}` in training artifacts
- Verify checksums before loading

### 2. Feature Store Wiring (Large: 3+ days)
- Wire Phase-1 feature outputs through `src/feature_store/`
- Goal: one canonical read API for training/backtests/inference

### 3. Standardized Evaluation Reports (Short: 1-4 hours)
- Produce comparable report artifact per model/run (JSON + markdown)
- Include: data version, CV scheme, embargo/purge, costs model, stability metrics

### 4. Production Monitoring (Large: 3+ days)
- Add paper trading / shadow mode hooks
- Monitor: prediction distribution drift, feature drift, realized PnL, calibration decay

### 5. Timestamp Alignment Guarantees (Medium: 1-2 days)
- Upgrade from index-based trimming to explicit datetime joins for OOF
- Fail fast on timestamp mismatches in heterogeneous stacking

## Anti-Patterns to Watch

1. **Split Artifact Universes**
   - Phase-1 outputs in `runs/...`, training in `experiments/runs/...`
   - Need hard link via pipeline run ID + checksums

2. **Heterogeneous Ensemble Alignment Risk**
   - Sequence windows may cause timestamp misalignment
   - Treat timestamp alignment as invariant

3. **Config Sprawl Risk**
   - Add stronger schema validation
   - Snapshot "effective config" in every run artifact

## Trading-Specific Concerns

1. **Subtle Leakage via Label Optimization**
   - Use true outer holdout for tuning model+label params

2. **Non-Stationarity**
   - Make retrain policy explicit (drift triggers, roll forward, model retirement)

3. **Live Constraints**
   - Encode inference assumptions (latency, partial bars, contract roll)
   - Fail fast on violations

## Escalation Triggers (When to Invest in Complex Infra)

- Multiple concurrent symbols/portfolios
- Continuous deployment (daily/weekly)
- Online features (sub-minute, microstructure, order book)

## Recommended Next Steps

| Priority | Task | Effort |
|----------|------|--------|
| P0 | Lineage unification (pipeline→training) | 1-2 days |
| P0 | Timestamp alignment checks for stacking | 1-2 days |
| P1 | Standardized evaluation reports | 4 hours |
| P2 | Feature store wiring | 3+ days |
| P2 | Production monitoring pipeline | 3+ days |

---

*Source: Oracle agent strategic review*
*Date: 2026-01-15*
