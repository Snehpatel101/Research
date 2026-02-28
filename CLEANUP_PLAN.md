# Cleanup Plan: ML Factory

**Status:** Phase 84 COMPLETE (Signal Quality — logloss metrics + binary classification mode)
**Last Updated:** 2026-02-28

---

## Completed Phases (24-49)

See **COMPLETION.md** for full details on all completed phases.

| Phase | Description | Status | Completed |
|-------|-------------|--------|-----------|
| 24 | Feature Computation Caching (ADX/DI, microstructure, supertrend) | ✅ COMPLETE | 2026-01-29 |
| 25 | Data Validation Hardening (fail-fast validation) | ✅ COMPLETE | 2026-01-29 |
| 26 | Type Safety & Code Quality (Any types, return annotations) | ✅ COMPLETE | 2026-01-29 |
| 27 | Architecture Consolidation (class deduplication) | ✅ COMPLETE | 2026-01-29 |
| 28 | Compute Performance (numba, parallelization, GARCH, caching) | ✅ COMPLETE | 2026-01-30 |
| 29 | Memory Performance (cache bounds, log_returns consolidation) | ✅ COMPLETE | 2026-01-29 |
| 30 | Advanced Architecture (transformer family, derived constants, SMA/EMA/STD caching) | ✅ COMPLETE | 2026-01-30 |
| 31 | Code Polish (TODOs, constants, adapters, feature DAG) | ✅ COMPLETE | 2026-01-31 |
| 32 | Critical Fixes (model families, data leakage, numerical stability) | ✅ COMPLETE | 2026-02-01 |
| 33 | Performance & Architecture (evaluators, layer violations, optimizations) | ✅ COMPLETE | 2026-02-01 |
| 34 | Cleanup & Consolidation (orphaned files, MTF defaults, verification) | ✅ COMPLETE | 2026-02-01 |
| 35 | Production Hardening (exception logging, pickle security) | ✅ COMPLETE | 2026-02-02 |
| 36 | Pipeline Runtime Issues (label -99, sqrt, autocorr) | ✅ COMPLETE | 2026-02-02 |
| 37 | Runtime Warning Fixes (Additional sqrt/autocorr protection) | ✅ COMPLETE | 2026-02-02 |
| 39 | Sequence Model Data Shape Fix (route 3D/4D to run_prepared) | ✅ COMPLETE | 2026-02-04 |
| 40 | Skip Hyperparameter Tuning for Sequence Models | ✅ COMPLETE | 2026-02-04 |
| 41 | Critical Vectorization Fixes (wavelets, entropy) | ✅ COMPLETE | 2026-02-04 |
| 42 | Memory Leak Fixes (TCN training crash) | ✅ COMPLETE | 2026-02-06 |
| 43 | Pipeline Robustness + TCN Timeframe Fix | ✅ COMPLETE | 2026-02-07 |
| 44 | Label Column Preservation During Resampling | ✅ COMPLETE | 2026-02-07 |
| 45 | Codebase Cohesion Overhaul | ✅ COMPLETE | 2026-02-11 |
| 46 | Full Pipeline Cleanup & Test Consolidation | ✅ COMPLETE | 2026-02-11 |
| 47 | Critical Pipeline Fixes (data leakage, thread safety, notebook) | ✅ COMPLETE | 2026-02-12 |
| 48 | Medium Pipeline Fixes (evaluators, feature selection, orphaned files) | ✅ COMPLETE | 2026-02-12 |
| 49 | Ruff Clean Sweep (SIM/E402/UP047, black formatting) | ✅ COMPLETE | 2026-02-12 |
| 50 | Speed Optimizations, Config Cleanup & MGC Readiness | ✅ COMPLETE | 2026-02-13 |
| 51 | Deploy Artifact — Single-Call Production Inference | ✅ COMPLETE | 2026-02-15 |
| 52 | Universal Inference Pipeline, Special Mode Bundles & Safe Pickle | ✅ COMPLETE | 2026-02-15 |
| 53 | Security Hardening, SymbolConfig Extraction & Resample Safety | ✅ COMPLETE | 2026-02-16 |
| 54 | E2E Pipeline Fixes: Trainer.save, per-model features, 4D multi-stream | ✅ COMPLETE | 2026-02-16 |
| 55 | Deploy Manifest model_name Fix & primary_model Selection | ✅ COMPLETE | 2026-02-16 |
| 56 | Backtest Pipeline Fix: _extract_predictions & Timestamp Alignment | ✅ COMPLETE | 2026-02-16 |
| 57 | 4D OOF Generation — Cross-Family Ensemble Support | ✅ COMPLETE | 2026-02-16 |
| 58 | Feature Selection Pipeline Overhaul (pre-filters, per-model) | ✅ COMPLETE | 2026-02-17 |
| 59 | MDA Feature Ranking + Test Split Crash Fix | ✅ COMPLETE | 2026-02-17 |
| 60 | DatetimeIndex Pipeline Fix & Cross-Family Ensembles | ✅ COMPLETE | 2026-02-19 |
| 62 | OPTIMIZATIONPLAN Complete — Final 5 Optimizations | ✅ COMPLETE | 2026-02-19 |
| 63 | CODEBASE_AUDIT Complete — All 12 Audit Fixes | ✅ COMPLETE | 2026-02-19 |
| 64 | E2E Pipeline Smoke Test — 12 Models x 2 Modes (6 bugs fixed) | ✅ COMPLETE | 2026-02-19 |
| 65 | Pipeline Audit & Test Suite Cleanup (212/212 tests) | ✅ COMPLETE | 2026-02-19 |
| 66 | Financial Rigor — ONC Clustering, Transaction Costs, DSR Gate, CPCV | ✅ COMPLETE | 2026-02-20 |
| 67 | Consistency Hardening — 14 Inconsistencies Fixed + 3D OOF Chunking | ✅ COMPLETE | 2026-02-20 |
| 68 | Performance Optimizations — 9 Items (~4.4x H100 Speedup) | ✅ COMPLETE | 2026-02-20 |
| 69 | Calibrator Single-Class Crash Fix | ✅ COMPLETE | 2026-02-21 |
| 70 | Lint Fixes (14 ruff + 15 black, 22 files) | ✅ COMPLETE | 2026-02-21 |
| 71 | Comprehensive Notebook Overhaul (12 fixes, 25 cells) | ✅ COMPLETE | 2026-02-21 |
| 72 | Memory Cleanup — OOM Prevention for Large Datasets (5 fixes, 4 files) | ✅ COMPLETE | 2026-02-22 |
| 73 | Scaler Serialization Fix + Notebook Warnings | ✅ COMPLETE | 2026-02-22 |
| 74 | Memory Optimization + Training Bug Fixes + Notebook Visualizations | ✅ COMPLETE | 2026-02-22 |
| 75 | OOM Root Cause Fix + Pipeline Bug Fixes (11 items) | ✅ COMPLETE | 2026-02-22 |
| 76 | Walk-Forward Feature Selection Fix + Float32 Scaler | ✅ COMPLETE | 2026-02-22 |
| 77 | Pipeline Audit Fixes (6 items across 6 files) | ✅ COMPLETE | 2026-02-22 |
| 78 | Deep Memory Fixes (4 items, ~19 GB saved per neural model) | ✅ COMPLETE | 2026-02-23 |
| 79 | In-Place Scaling + Factory Float32 (4 items, saves ~27 GB peak) | ✅ COMPLETE | 2026-02-22 |
| 80 | Audit-Driven Fixes: Label Balance, Memory, Early Stopping, TCN (20 files) | ✅ COMPLETE | 2026-02-26 |
| 81 | Fix 5 Dead Notebook Cells (calibration, leakage, features, equity, agreement) | ✅ COMPLETE | 2026-02-26 |
| 82 | Checkpoint Resume — 4D additional_dfs Persistence | ✅ COMPLETE | 2026-02-26 |
| 83 | Audit Cleanup — min_frequency wiring + missed fixes + weight_norm | ✅ COMPLETE | 2026-02-26 |
| 84 | Signal Quality — Logloss Metrics + Binary Classification Mode | ✅ COMPLETE | 2026-02-28 |

**Phase 3 Master Implementation Plan: COMPLETE (26/26 tasks across Phases 51-52)**

**Summary Impact:** 54 phases complete (24-84), 220+ files modified, production-ready evaluators, pipeline time reduced from 5+ hours to 15-25 minutes, sequence models fully functional, critical vectorization and memory bottlenecks eliminated, pipeline robustness hardened, model timeframe contracts enforced, test suite consolidated, all data leakage fixed, ruff clean (0 errors), 10 speed optimizations (~50-60% runtime reduction), walk-forward validation enabled, MGC contract auto-detection, single-call deploy artifact inference, UniversalInferencePipeline for all 12 models, special mode bundles (walk-forward, regime, meta-labeling), safe pickle migration complete (all 38 sites), neural architecture versioning, SymbolConfig standalone class, deploy manifest model names fixed, backtest pipeline fully functional, all 8 cross-family ensemble combinations working, DatetimeIndex pipeline fix, codebase audit 12/12 fixes, financial rigor improvements (ONC, transaction costs, DSR gate, CPCV), consistency hardening (14 inconsistencies fixed, 3D OOF chunked processing for 1.7M+ row scalability), audit-driven label balance fix, checkpoint resume MTF persistence for 4D models.

---

## Phase Summary

| Phase | Focus | Priority | Effort | Status |
|-------|-------|----------|--------|--------|
| 24-34 | See above | VARIOUS | 11 days | ✅ ALL COMPLETE - See COMPLETION.md |
| 35 | Hardening (exception logging, pickle security) | HIGH | 1 day | ✅ COMPLETE |
| 36 | Pipeline Runtime Issues (label -99, sqrt, autocorr) | CRITICAL | 1 day | ✅ COMPLETE |
| 37 | Runtime Warning Fixes (Additional sqrt/autocorr protection) | HIGH | 1 day | ✅ COMPLETE |
| 39 | Sequence Model Data Shape Fix | CRITICAL | 1 session | ✅ COMPLETE |
| 40 | Skip Hyperparameter Tuning for Sequence Models | HIGH | 1 session | ✅ COMPLETE |
| 41 | Critical Vectorization Fixes (wavelets, entropy) | CRITICAL | 1 session | ✅ COMPLETE |
| 42 | Memory Leak Fixes (TCN training crash) | CRITICAL | 1 session | ✅ COMPLETE |
| 43 | Pipeline Robustness + TCN Timeframe Fix | HIGH | 2 sessions | ✅ COMPLETE |
| 44 | Label Column Preservation During Resampling | CRITICAL | 1 session | ✅ COMPLETE |
| 45 | Codebase Cohesion Overhaul | HIGH | 1 session | ✅ COMPLETE |
| 46 | Full Pipeline Cleanup & Test Consolidation | HIGH | 1 session | ✅ COMPLETE |
| 47 | Critical Pipeline Fixes (data leakage, thread safety) | CRITICAL | 1 session | ✅ COMPLETE |
| 48 | Medium Pipeline Fixes (evaluators, orphaned files) | HIGH | 1 session | ✅ COMPLETE |
| 49 | Ruff Clean Sweep (all lint issues) | HIGH | 1 session | ✅ COMPLETE |
| 50 | Speed Optimizations, Config Cleanup & MGC Readiness | HIGH | 1 session | ✅ COMPLETE |
| 51 | Deploy Artifact — Single-Call Production Inference | HIGH | 1 session | ✅ COMPLETE |
| 52 | Universal Inference Pipeline + Special Bundles + Safe Pickle | HIGH | 1 session | ✅ COMPLETE |
| 53 | Security Hardening, SymbolConfig, Resample Safety | HIGH | 1 session | ✅ COMPLETE |
| 54 | E2E Pipeline Fixes: Trainer.save, per-model features, 4D multi-stream | CRITICAL | 1 session | ✅ COMPLETE |
| 55 | Deploy Manifest model_name Fix & primary_model Selection | HIGH | 1 session | ✅ COMPLETE |
| 56 | Backtest Pipeline Fix: _extract_predictions & Timestamp Alignment | HIGH | 1 session | ✅ COMPLETE |
| 57 | 4D OOF Generation — Cross-Family Ensemble Support | CRITICAL | 1 session | ✅ COMPLETE |
| 58 | Feature Selection Pipeline Overhaul (pre-filters, per-model) | HIGH | 1 session | ✅ COMPLETE |
| 59 | MDA Feature Ranking + Test Split Crash Fix | HIGH | 1 session | ✅ COMPLETE |
| 60 | DatetimeIndex Pipeline Fix & Cross-Family Ensembles (7 bugs) | CRITICAL | 1 session | ✅ COMPLETE |
| 62 | OPTIMIZATIONPLAN Complete — Final 5 Optimizations | HIGH | 1 session | ✅ COMPLETE |
| 63 | CODEBASE_AUDIT Complete — All 12 Audit Fixes | HIGH | 1 session | ✅ COMPLETE |
| 64 | E2E Pipeline Smoke Test — 12 Models x 2 Modes | CRITICAL | 1 session | ✅ COMPLETE |
| 65 | Pipeline Audit & Test Suite Cleanup | HIGH | 1 session | ✅ COMPLETE |
| 66 | Financial Rigor (ONC, costs, DSR, CPCV) | HIGH | 1 session | ✅ COMPLETE |
| 67 | Consistency Hardening (14 inconsistencies + 3D OOF) | HIGH | 1 session | ✅ COMPLETE |
| 68 | Performance Optimizations (9 items, ~4.4x H100) | HIGH | 1 session | ✅ COMPLETE |
| 69 | Calibrator Single-Class Crash Fix | HIGH | 1 session | ✅ COMPLETE |
| 70 | Lint Fixes (14 ruff + 15 black) | MEDIUM | 1 session | ✅ COMPLETE |
| 71 | Notebook Overhaul (12 fixes, 25 cells) | HIGH | 1 session | ✅ COMPLETE |
| 72 | Memory Cleanup — OOM Prevention (5 fixes, 4 files) | HIGH | 1 session | ✅ COMPLETE |
| 73 | Scaler Serialization Fix + Notebook Warnings | HIGH | 1 session | ✅ COMPLETE |
| 74 | Memory Optimization + Training Bug Fixes + Notebook Visualizations | HIGH | 1 session | ✅ COMPLETE |
| 75 | OOM Root Cause Fix + Pipeline Bug Fixes (11 items) | CRITICAL | 1 session | ✅ COMPLETE |
| 76 | Walk-Forward Feature Selection Fix + Float32 Scaler | CRITICAL | 1 session | ✅ COMPLETE |
| 77 | Pipeline Audit Fixes (6 items, 6 files) | HIGH | 1 session | ✅ COMPLETE |
| 78 | Deep Memory Fixes (4 items, ~19 GB saved) | CRITICAL | 1 session | ✅ COMPLETE |
| 79 | In-Place Scaling + Factory Float32 (4 items, ~27 GB saved) | CRITICAL | 1 session | ✅ COMPLETE |
| 80 | Audit-Driven Fixes: Label Balance, Memory, Early Stopping, TCN | CRITICAL | 1 session | ✅ COMPLETE |
| 81 | Fix 5 Dead Notebook Cells | HIGH | 1 session | ✅ COMPLETE |
| 82 | Checkpoint Resume — 4D additional_dfs Persistence | CRITICAL | 1 session | ✅ COMPLETE |
| 83 | Audit Cleanup — min_frequency wiring + weight_norm | HIGH | 1 session | ✅ COMPLETE |
| 84 | Signal Quality — Logloss Metrics + Binary Classification | HIGH | 1 session | ✅ COMPLETE |

---

## Active Phases

**No active phases.** All phases through 84 are complete. All 17 audit items addressed. Test suite: 212/212 passing. See COMPLETION.md for details.

---

## Completed Recent Phases (Archive)

### Phase 43: Pipeline Robustness + TCN Timeframe Fix

**Status:** ✅ COMPLETE
**Priority:** HIGH (P1) - Production reliability + Memory crash fix
**Effort:** Two sessions (2026-02-06, 2026-02-07)
**Source:** Pipeline reliability hardening + TCN training crash
**Completed:** 2026-02-07

**Overview**

Enhanced pipeline reliability with fail-fast behavior, timeout enforcement, and stage transition validation. Prevents silent failures and pipeline hangs.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                   PHASE 43 ENHANCEMENTS IMPLEMENTED (2026-02-06)                 │
│                                                                                  │
│  ✅ ADDED - STAGE 3 FAIL-FAST OPTION:                                           │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  File: data/pipeline/stages/features/run.py                    │             │
│  │  Feature: Configurable fail-fast when tasks fail               │             │
│  │  Config: stage3_fail_on_partial, stage3_min_success_rate       │             │
│  │  Impact: Prevents silent data gaps from partial failures       │             │
│  └────────────────────────────────────────────────────────────────┘             │
│                                                                                  │
│  ✅ ADDED - TIMEOUT ENFORCEMENT:                                                │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  File: data/pipeline/runner.py                                 │             │
│  │  Feature: StageTimeoutError + _run_with_timeout()              │             │
│  │  Uses: signal.SIGALRM for Unix timeout enforcement             │             │
│  │  Config: stage_timeout_seconds, enable_stage_timeouts          │             │
│  │  Impact: Prevents pipeline hangs from stuck stages             │             │
│  └────────────────────────────────────────────────────────────────┘             │
│                                                                                  │
│  ✅ ADDED - STAGE TRANSITION VALIDATION:                                        │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  File: data/pipeline/runner.py                                 │             │
│  │  Feature: _validate_stage_transition() method                  │             │
│  │  Uses: schemas.py validate_stage_transition()                  │             │
│  │  Config: enable_transition_validation                          │             │
│  │  Impact: Catches data corruption between stages                │             │
│  └────────────────────────────────────────────────────────────────┘             │
│                                                                                  │
│  ✅ FIXED - STALE README:                                                       │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  File: data/pipeline/stages/README.md                          │             │
│  │  Fix: Complete rewrite matching actual stage structure         │             │
│  │  Removed: References to non-existent stage7/8/baseline files   │             │
│  │  Impact: Documentation matches reality                         │             │
│  └────────────────────────────────────────────────────────────────┘             │
│                                                                                  │
│  ✅ ADDED - STAGE 10 IN REGISTRY:                                               │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  File: data/pipeline/stage_registry.py                         │             │
│  │  Added: StageName.EVALUATION enum                              │             │
│  │  Status: Documented as optional (post-training)                │             │
│  │  Impact: Completes stage enumeration                           │             │
│  └────────────────────────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Tasks

| Task | File | Priority | Status | Description |
|------|------|----------|--------|-------------|
| 43-1 | `data/pipeline/stages/features/run.py` | HIGH | ✅ COMPLETE | Add fail-fast option for Stage 3 |
| 43-2 | `data/pipeline/runner.py` | HIGH | ✅ COMPLETE | Enforce timeout with signal.SIGALRM |
| 43-3 | `data/pipeline/runner.py` | HIGH | ✅ COMPLETE | Add stage transition validation |
| 43-4 | `data/pipeline/stages/README.md` | MEDIUM | ✅ COMPLETE | Update stale documentation |
| 43-5 | `data/pipeline/stage_registry.py` | MEDIUM | ✅ COMPLETE | Add Stage 10 to registry |
| 43-6 | `data/adapters/preparation.py` | CRITICAL | ✅ COMPLETE | Auto-resample to model's primary_timeframe |

### Success Metrics

| Metric | Before | After | How to Verify |
|--------|--------|-------|---------------|
| Partial failures | Silent (proceed with gaps) | Fail-fast (configurable) | Check stage3_fail_on_partial config |
| Pipeline hangs | No protection | Timeout enforced | Check stage_timeout_seconds config |
| Data corruption | Undetected between stages | Validated | Check enable_transition_validation |
| Documentation accuracy | Outdated (stage7/8 refs) | Current | Read stages/README.md |
| Stage enumeration | Incomplete (missing stage 10) | Complete | Check StageName enum |
| TCN memory | 230GB+ (crash) | ~25-35GB (working) | TCN trains on 1min input data |
| Model contracts | Ignored (wrong timeframe) | Enforced (auto-resample) | Log shows resampling message |

### Verification Results

| Check | Result | Notes |
|-------|--------|-------|
| Ruff linting | ✅ PASS | No new issues |
| Import tests | ✅ PASS | All modules importable |
| Config tests | ✅ PASS | New config fields recognized |

---

### Phase 42: Memory Leak Fixes

**Status:** ✅ COMPLETE
**Priority:** CRITICAL (P0) - Training crashed with 230GB+ RAM usage
**Effort:** Single session (2026-02-06)
**Source:** User-reported TCN training crash on 355K row dataset
**Completed:** 2026-02-06

**Overview**

Fixed critical memory leak during TCN training that caused 230GB+ RAM usage and crash on 355K row dataset. Memory reduced to ~25-35GB (~85% reduction).

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                   PHASE 42 FIXES IMPLEMENTED (2026-02-06)                        │
│                                                                                  │
│  ✅ FIXED - DATASET_TO_ARRAYS() MEMORY LEAK:                                    │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  File: models/data_preparation.py:120-191                      │             │
│  │  Problem: List accumulation held 355K tensors in memory        │             │
│  │  Fix: Pre-allocate arrays, in-place assignment, gc.collect()   │             │
│  │  Impact: ~8GB savings (50% reduction during data preparation)  │             │
│  └────────────────────────────────────────────────────────────────┘             │
│                                                                                  │
│  ✅ FIXED - DATALOADER WORKERS MEMORY DUPLICATION:                              │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  File: models/neural/base_rnn.py:312-313                       │             │
│  │  Problem: num_workers=4 caused 4x memory duplication (~32GB)   │             │
│  │  Fix: Changed defaults to num_workers=0, pin_memory=False      │             │
│  │  Impact: ~32GB savings                                          │             │
│  └────────────────────────────────────────────────────────────────┘             │
│                                                                                  │
│  ✅ FIXED - TRAINING DATA CLEANUP:                                              │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  File: models/training/trainer.py:953-963                      │             │
│  │  Problem: Training data stayed in memory during evaluation     │             │
│  │  Fix: Added del X_train, w_train + gc.collect()                │             │
│  │  Impact: ~8GB freed immediately after training                 │             │
│  └────────────────────────────────────────────────────────────────┘             │
│                                                                                  │
│  MEMORY ANALYSIS (355K rows, 100 features, 60 timesteps):                       │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  Before: 230GB+ (crash)                                         │             │
│  │  After: ~25-35GB (working)                                      │             │
│  │  Reduction: ~85%                                                │             │
│  └────────────────────────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Tasks

| Task | File | Priority | Status | Description |
|------|------|----------|--------|-------------|
| 42-1 | `models/data_preparation.py:120-191` | CRITICAL | ✅ COMPLETE | Fix dataset_to_arrays() list accumulation |
| 42-2 | `models/neural/base_rnn.py:312-313` | CRITICAL | ✅ COMPLETE | Reduce DataLoader workers to 0 |
| 42-3 | `models/neural/base_rnn.py:690-691` | HIGH | ✅ COMPLETE | Update fallback defaults |
| 42-4 | `models/training/trainer.py:953-963` | HIGH | ✅ COMPLETE | Add memory cleanup in run_prepared() |
| 42-5 | `models/training_utils.py:90-101` | MEDIUM | ✅ COMPLETE | Fix training_utils list pattern |

### Success Metrics

| Metric | Before | After | How to Verify |
|--------|--------|-------|---------------|
| Peak RAM usage | 230GB+ | ~25-35GB | Monitor during training |
| Training completion | Crash | Success | TCN trains on 355K rows |
| List accumulation | 355K tensors | 0 | Pre-allocated arrays |
| DataLoader workers | 4 | 0 | No worker memory duplication |

### Verification Results

| Check | Result | Notes |
|-------|--------|-------|
| Ruff linting | ✅ PASS | No new issues |
| Import tests | ✅ PASS | All modules importable |
| Memory test | ✅ PASS | TCN trains without crash |

---

### Phase 41: Critical Vectorization Fixes

**Status:** ✅ COMPLETE
**Priority:** CRITICAL (P0) - Pipeline was hanging for 5+ hours
**Effort:** Single session (2026-02-04)
**Source:** Production pipeline execution on 350K row dataset
**Completed:** 2026-02-04

**Overview**

Fixed 3 critical O(n²) bottlenecks that were causing 5+ hour pipeline hangs. All fixes use Numba JIT compilation for maximum performance.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                   PHASE 41 FIXES IMPLEMENTED (2026-02-04)                        │
│                                                                                  │
│  ✅ FIXED - WAVELET NORMALIZATION O(n²) BOTTLENECK:                             │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  File: data/pipeline/stages/features/wavelets.py               │             │
│  │  Problem: Expanding window creates ~175,000x redundant ops     │             │
│  │  Fix: Welford's O(n) online algorithm with Numba JIT           │             │
│  │  Impact: O(n²) → O(n), 175,000x fewer operations at 350K rows  │             │
│  └────────────────────────────────────────────────────────────────┘             │
│                                                                                  │
│  ✅ FIXED - SAMPLE ENTROPY NUMBA OPTIMIZATION:                                  │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  File: data/pipeline/stages/features/entropy.py                │             │
│  │  Problem: Python loops with no early exit                      │             │
│  │  Fix: _count_template_matches_numba() with early exit          │             │
│  │  Impact: ~20-50x speedup from Numba + early exit optimization  │             │
│  └────────────────────────────────────────────────────────────────┘             │
│                                                                                  │
│  ✅ FIXED - APPROXIMATE ENTROPY NUMBA OPTIMIZATION:                             │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  File: data/pipeline/stages/features/entropy.py                │             │
│  │  Problem: Python loops computing phi correlation               │             │
│  │  Fix: _phi_correlation_numba() with JIT compilation            │             │
│  │  Impact: ~20-50x speedup from Numba                            │             │
│  └────────────────────────────────────────────────────────────────┘             │
│                                                                                  │
│  ✅ FIXED - LEMPEL-ZIV COMPLEXITY STRING OPS:                                   │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  File: data/pipeline/stages/features/entropy.py                │             │
│  │  Problem: String concatenation in Python loops                 │             │
│  │  Fix: _lempel_ziv_complexity_numba() with array operations     │             │
│  │  Impact: ~10-20x speedup from array-based pattern matching     │             │
│  └────────────────────────────────────────────────────────────────┘             │
│                                                                                  │
│  PERFORMANCE IMPACT:                                                             │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  Before: 5+ hours for 350K rows with wavelets enabled          │             │
│  │  After: 15-25 minutes for 350K rows                            │             │
│  │  Speedup: ~12-20x overall pipeline improvement                 │             │
│  └────────────────────────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Tasks

| Task | File | Priority | Status | Description |
|------|------|----------|--------|-------------|
| 41-1 | `data/pipeline/stages/features/wavelets.py` | CRITICAL | ✅ COMPLETE | Wavelet normalization O(n) fix with Welford's algorithm |
| 41-2 | `data/pipeline/stages/features/entropy.py` | CRITICAL | ✅ COMPLETE | Sample/Approximate Entropy Numba optimization |
| 41-3 | `data/pipeline/stages/features/entropy.py` | CRITICAL | ✅ COMPLETE | Lempel-Ziv array-based optimization |

### Success Metrics

| Metric | Before | After | How to Verify |
|--------|--------|-------|---------------|
| Pipeline completion time | 5+ hours | 15-25 minutes | Full pipeline benchmark |
| Wavelet normalization | O(n²) | O(n) | Operation count analysis |
| Sample Entropy | Python loops | Numba JIT | Benchmark on 10K samples |
| Approximate Entropy | Python loops | Numba JIT | Benchmark on 10K samples |
| Lempel-Ziv | String ops | Array ops | Benchmark on 10K samples |

### Verification Results

| Check | Result | Notes |
|-------|--------|-------|
| Ruff linting | ✅ PASS | No new issues |
| Import tests | ✅ PASS | All modules importable |
| Pipeline completion | ✅ PASS | 350K rows completes in ~20 minutes |

---

### Phase 40: Skip Hyperparameter Tuning for Sequence Models

**Status:** ✅ COMPLETE
**Priority:** HIGH (P1) - Sequence models getting wrong hyperparameters
**Effort:** Single session (2026-02-04)
**Source:** Analysis of hyperparameter tuning for 3D/4D models
**Completed:** 2026-02-04

**Overview**

Fixed issue where hyperparameter tuning flattened 3D/4D data to 2D, producing hyperparameters optimized for the wrong data structure.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                   PHASE 40 FIX IMPLEMENTED (2026-02-04)                          │
│                                                                                  │
│  ✅ FIXED - SKIP TUNING FOR SEQUENCE MODELS:                                    │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  File: models/training/services/hyperparameter_tuning.py:67-80│             │
│  │  Problem: Optuna flattens 3D→2D, optimizing wrong structure   │             │
│  │  Fix: Check data_rank >= 3, return empty params with warning  │             │
│  │  Impact: Sequence models use defaults (safer than wrong ones) │             │
│  └────────────────────────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Tasks

| Task | File | Priority | Status | Description |
|------|------|----------|--------|-------------|
| 40-1 | `models/training/services/hyperparameter_tuning.py:67-80` | HIGH | ✅ COMPLETE | Skip tuning for 3D/4D data, use defaults |

### Success Metrics

| Metric | Before | After | How to Verify |
|--------|--------|-------|---------------|
| 3D/4D tuning | Flattened data | Skipped with warning | `n_trials_completed == 0` for 3D/4D |
| Hyperparameter quality | Wrong (optimized for 2D) | Safe (defaults) | Models train correctly |

### Verification Results

| Check | Result | Notes |
|-------|--------|-------|
| Ruff linting | ✅ PASS | No issues |
| Import tests | ✅ PASS | All modules importable |
| Manual test | ✅ PASS | 3D data skips tuning correctly |

---

### Phase 39: Sequence Model Data Shape Fix

**Status:** ✅ COMPLETE
**Priority:** CRITICAL (P0) - LSTM/TFT models completely broken
**Effort:** Single session (2026-02-04)
**Source:** Runtime shape error during model training
**Completed:** 2026-02-04

**Overview**

Fixed critical bug where sequential models failed with shape error due to double-processing of data.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                   PHASE 39 FIXES IMPLEMENTED (2026-02-04)                        │
│                                                                                  │
│  ✅ FIXED - DATA ROUTING FOR SEQUENCE MODELS:                                   │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  Root Cause: Double-processing of sequence data                │             │
│  │  1. Container flattened 3D→2D during build                     │             │
│  │  2. Trainer called get_pytorch_sequences() on flattened data   │             │
│  │  3. Result: Unusable data shape                                │             │
│  └────────────────────────────────────────────────────────────────┘             │
│                                                                                  │
│  ✅ SOLUTION - NEW PATHWAY:                                                     │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  File: models/training/trainer.py:885-1008                     │             │
│  │  Added: run_prepared() method for pre-shaped data              │             │
│  │  Bypasses: Container creation and reshaping                    │             │
│  │  Uses: Data arrays as-is for training                          │             │
│  └────────────────────────────────────────────────────────────────┘             │
│                                                                                  │
│  ✅ ROUTING LOGIC:                                                               │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  File: models/training/services/model_training.py:124-135      │             │
│  │  Logic: if data_rank >= 3: use run_prepared()                 │             │
│  │         else: use run() with container                         │             │
│  └────────────────────────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Tasks

| Task | File | Priority | Status | Description |
|------|------|----------|--------|-------------|
| 39-1 | `models/training/trainer.py:885-1008` | CRITICAL | ✅ COMPLETE | Add run_prepared() method |
| 39-2 | `models/training/trainer.py:994-997` | HIGH | ✅ COMPLETE | Fix _save_metrics() bug |
| 39-3 | `models/training/services/model_training.py:124-135` | CRITICAL | ✅ COMPLETE | Route 3D/4D to run_prepared() |

### Success Metrics

| Metric | Before | After | How to Verify |
|--------|--------|-------|---------------|
| LSTM training | Shape error | Success | Model trains without error |
| TFT training | Shape error | Success | Model trains without error |
| Data shape | (n, 13140) wrong | (n, 60, 219) correct | Verify with print statement |

### Verification Results

| Check | Result | Notes |
|-------|--------|-------|
| Ruff linting | ✅ PASS | No new issues |
| Import tests | ✅ PASS | All modules importable |
| Shape verification | ✅ PASS | 3D data remains 3D through training |

---

### Phase 37: Runtime Warning Fixes (Additional sqrt/autocorr protection)

**Status:** ✅ COMPLETE
**Priority:** HIGH (P1) - Runtime warnings during production pipeline execution
**Effort:** 1 day (actual)
**Source:** User-reported runtime warnings during pipeline execution (2026-02-02)
**Completed:** 2026-02-02

**Overview**

Additional runtime warning fixes discovered during production pipeline execution. Built on Phase 36's foundation to eliminate remaining edge cases in mathematical operations.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                   PHASE 37 FIXES IMPLEMENTED (2026-02-02)                        │
│                                                                                  │
│  ✅ FIXED - AUTOCORR DEGREES OF FREEDOM:                                        │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  File: models/training/modes/regime_aware.py:243               │             │
│  │  Problem: len(x) > 1 allows autocorr(lag=1) with 2 samples    │             │
│  │  Fix: Changed to len(x) >= 3 for sufficient degrees of freedom│             │
│  └────────────────────────────────────────────────────────────────┘             │
│                                                                                  │
│  ✅ FIXED - PARKINSON VOLATILITY SQRT:                                          │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  File: data/features/compute/volatility.py:307                 │             │
│  │  Problem: Edge cases caused negative values in sqrt            │             │
│  │  Fix: Added np.maximum(..., 0) protection                      │             │
│  └────────────────────────────────────────────────────────────────┘             │
│                                                                                  │
│  ✅ FIXED - CORWIN-SCHULTZ SPREAD SQRT:                                         │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  File: data/features/compute/microstructure.py:216             │             │
│  │  Problem: beta/gamma could be negative in sqrt operations      │             │
│  │  Fix: Added beta_safe/gamma_safe with np.maximum protection    │             │
│  └────────────────────────────────────────────────────────────────┘             │
│                                                                                  │
│  ✅ FIXED - EDGE SPREAD SQRT (NUMBA):                                           │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  File: pipeline/stages/features/microstructure_proxies.py:72   │             │
│  │  Problem: 1 - ratio**2 could be negative in sqrt               │             │
│  │  Fix: Changed to np.sqrt(max(0, 1 - ratio**2))                 │             │
│  └────────────────────────────────────────────────────────────────┘             │
│                                                                                  │
│  ✅ FIXED - ROLL SPREAD SQRT:                                                   │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  File: pipeline/stages/features/microstructure_proxies.py:131  │             │
│  │  Problem: -cov_lag1 could be positive (double negative)        │             │
│  │  Fix: Changed to 2 * np.sqrt(np.maximum(-cov_lag1, 0))         │             │
│  └────────────────────────────────────────────────────────────────┘             │
│                                                                                  │
│  ✅ FIXED - INCOMPLETE CONFIG FILE:                                             │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  File: config/global.yaml                                      │             │
│  │  Problem: Missing required TimeframeConfig fields              │             │
│  │  Fix: Completed global.yaml with all required sections         │             │
│  └────────────────────────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Tasks

| Task | File | Priority | Status | Description |
|------|------|----------|--------|-------------|
| 37-1 | `models/training/modes/regime_aware.py:243` | HIGH | ✅ COMPLETE | Fix autocorr degrees of freedom warning |
| 37-2 | `data/features/compute/volatility.py:307` | HIGH | ✅ COMPLETE | Add sqrt protection to Parkinson vol |
| 37-3 | `data/features/compute/microstructure.py:216` | HIGH | ✅ COMPLETE | Add sqrt protection to Corwin-Schultz |
| 37-4 | `pipeline/stages/features/microstructure_proxies.py:72` | HIGH | ✅ COMPLETE | Add sqrt protection to edge spread (numba) |
| 37-5 | `pipeline/stages/features/microstructure_proxies.py:131` | HIGH | ✅ COMPLETE | Add sqrt protection to roll spread |
| 37-6 | `config/global.yaml` | HIGH | ✅ COMPLETE | Complete global config with all required fields |

### Success Metrics

| Metric | Before | After | How to Verify |
|--------|--------|-------|---------------|
| Runtime warnings | 5 | 0 | No RuntimeWarning in pipeline output |
| Autocorr edge case | Warning on df<3 | Safe return | No "Degrees of freedom <= 0" warning |
| Sqrt edge cases | 4 warnings | 0 | All sqrt operations protected |
| Config initialization | ERROR | SUCCESS | TimeframeConfig.__init__() succeeds |

### Verification Results

| Check | Result | Notes |
|-------|--------|-------|
| Ruff linting | ✅ PASS | No new issues |
| Import tests | ✅ PASS | All modules importable |
| Runtime tests | ✅ PASS | No warnings during execution |

---

### Phase 36: Pipeline Runtime Issues

**Status:** ✅ COMPLETE
**Priority:** CRITICAL (P0) - Was blocking pipeline execution
**Effort:** 1 day
**Source:** Live pipeline execution on MES 1-min data (350,464 rows), 6-agent analysis (2026-02-02)
**Completed:** 2026-02-02

**Overview**

Pipeline failed after 10,782 seconds with label validation error. Initial static analysis incorrectly disproved claims, but actual pipeline execution confirmed all issues were real. All 4 fixes implemented and verified.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                   PHASE 36 FIXES IMPLEMENTED (2026-02-02)                        │
│                                                                                  │
│  ✅ FIXED - LABEL -99 FILTERING:                                                │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  Problem: -99 labels reached Optuna hyperparameter tuning      │             │
│  │  Fix 1: PreparedData.filter_invalid_labels() method added      │             │
│  │  Fix 2: HyperparameterTuningService filters before tuning      │             │
│  │  Fix 3: ModelTrainingService filters before training           │             │
│  └────────────────────────────────────────────────────────────────┘             │
│                                                                                  │
│  ✅ FIXED - SQRT OF NEGATIVE VALUES:                                            │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  Problem: Edge cases caused negative values inside sqrt        │             │
│  │  Fix: Added np.maximum(..., 0) before sqrt at 3 locations      │             │
│  │  - volatility.py: Garman-Klass, Rogers-Satchell, Yang-Zhang   │             │
│  └────────────────────────────────────────────────────────────────┘             │
│                                                                                  │
│  ✅ FIXED - AUTOCORRELATION LAG20 ALL NaN:                                      │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  Bug: window=20, lag=20 → len(x) > lag → 20 > 20 → False → NaN│             │
│  │  Fix: window=max(period, lag+2), condition len(x) >= lag+2    │             │
│  │  Note: Initial lag+1 fix was incomplete, required lag+2       │             │
│  └────────────────────────────────────────────────────────────────┘             │
│                                                                                  │
│  ✅ FIXED - MISSING CONFIG FILE:                                                │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  Created config/global.yaml with all default values            │             │
│  │  Eliminates 19+ "Failed to get config" warnings                │             │
│  └────────────────────────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Tasks

| Task | File | Priority | Status | Description |
|------|------|----------|--------|-------------|
| 36-1 | Multiple files | CRITICAL | ✅ COMPLETE | Filter -99 labels in PreparedData, tuning, training |
| 36-2 | `volatility.py:305,404,488` | HIGH | ✅ COMPLETE | Added np.maximum(..., 0) before sqrt |
| 36-3 | `price_features.py:147` | HIGH | ✅ COMPLETE | Fixed autocorr lag20 off-by-one bug |
| 36-4 | `config/global.yaml` | MEDIUM | ✅ COMPLETE | Created config file template |
| 36-5 | N/A | LOW | DEFERRED | LightGBM tuning handles per-dataset |

### Success Metrics

| Metric | Before | After | How to Verify |
|--------|--------|-------|---------------|
| Pipeline completion | FAILED | SUCCESS | Full pipeline run completes |
| sqrt warnings | 3 | 0 | No RuntimeWarning in output |
| NaN columns | 1 | 0 | return_autocorr_lag20 has values |
| Config warnings | 19 | 0 | No "Failed to get config" warnings |

### Verification Results (check-deep 5b - 2026-02-02)

| Agent | Result | Findings |
|-------|--------|----------|
| Code Review | ⚠️ WARN | 3 minor style issues: magic numbers, local imports |
| Contracts | ✅ PASS | All types, schemas, and API contracts verified |
| Integration | ✅ PASS | No circular deps, imports clean |
| Runtime | ✅ 4/4 PASS | All runtime tests pass after autocorr correction |

**Autocorrelation Fix Correction:**
Initial fix used `window=max(period, lag+1)` but check-deep verification revealed this was still off-by-one. Corrected to `window=max(period, lag+2)` and `len(x) >= lag+2`. NaN percentage reduced from 100% to 4.6%.

---

### Phase 35: Production Hardening

**Status:** ✅ COMPLETE
**Priority:** HIGH (P1)
**Effort:** 1 day (actual)
**Source:** Comprehensive pipeline review (6-agent analysis, 2026-02-02)
**Completed:** 2026-02-02

**Overview**

Remaining P1 items identified in comprehensive pipeline review that validated production readiness (7.5/10 overall):

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        PRODUCTION HARDENING TASKS                                │
│                                                                                  │
│  SILENT EXCEPTION HANDLING (26 locations):                                       │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  Add structured logging to all exception handlers               │             │
│  │  Current: except Exception: pass or return None                │             │
│  │  Target: except Exception as e: logger.error(...)             │             │
│  └────────────────────────────────────────────────────────────────┘             │
│                                                                                  │
│  UNSAFE DESERIALIZATION (45+ locations):                                         │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  pickle.load() and joblib.load() without validation            │             │
│  │  Recommendation: Add signature verification or safetensors      │             │
│  │  Priority: HIGH for production deployment                      │             │
│  └────────────────────────────────────────────────────────────────┘             │
│                                                                                  │
│  MTF CONSOLIDATION (COMPLETE):                                                   │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  ✅ Phase 34 consolidated to single source                     │             │
│  │  ✅ All modules import from constants.py                       │             │
│  └────────────────────────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Tasks

| Task | File Pattern | Priority | Description | Impact |
|------|--------------|----------|-------------|--------|
| 35-1 | Multiple files (26) | HIGH | Add structured logging to silent exception handlers | Debuggability |
| 35-2 | Multiple files (45+) | HIGH | Document/secure pickle loading or migrate to safetensors | Security |

### Success Metrics

| Metric | Before | After | How to Verify |
|--------|--------|-------|---------------|
| Silent exception handlers | 26 | 0 | `grep -r "except.*:" src/ \| grep -v logger` returns 0 |
| Undocumented pickle loads | 45+ | 0 | All pickle.load() has comment or uses verification |
| Code quality score | 7/10 | 8/10 | Improved debuggability and security |

---

## Phase 33: Performance & Architecture

**Status:** ✅ COMPLETE
**Priority:** HIGH
**Effort:** Single day (actual)
**Source:** Comprehensive ML pipeline review (9-agent analysis, 2026-02-01)
**Completed:** 2026-02-01


### Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    PERFORMANCE & ARCHITECTURE GAPS                               │
│                                                                                  │
│  INCOMPLETE IMPLEMENTATIONS (NotImplementedError):                               │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  validation/evaluation/cpcv_pbo_evaluator.py:52                │             │
│  │  validation/evaluation/cv_evaluator.py:51                      │             │
│  │  validation/evaluation/walk_forward_evaluator.py:51            │             │
│  │  → Three evaluator classes not implemented                     │             │
│  └────────────────────────────────────────────────────────────────┘             │
│                                                                                  │
│  LAYER VIOLATIONS (Core → Data):                                                │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  core/container.py:673 - imports MultiResolution4DAdapter      │             │
│  │  core/container.py:739 - imports MultiStreamAdapter            │             │
│  │  → Core layer should not depend on data layer                  │             │
│  └────────────────────────────────────────────────────────────────┘             │
│                                                                                  │
│  PERFORMANCE OPTIMIZATIONS REMAINING:                                            │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  CCI vectorization:       5-10x speedup                        │             │
│  │  Variance ratio:          10-20x speedup                       │             │
│  │  Order flow caching:      3-4x speedup                         │             │
│  │  Regime caching:          3x speedup                           │             │
│  │  Wavelet numba:           10-50x speedup                       │             │
│  │  Hurst O(n) algorithm:    5-10x speedup                        │             │
│  └────────────────────────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Tasks

| Task | File:Line | Priority | Description | Speedup |
|------|-----------|----------|-------------|---------|
| 33-1 | `validation/evaluation/cpcv_pbo_evaluator.py:52` | HIGH | Implement CPCV-PBO evaluator | N/A |
| 33-2 | `validation/evaluation/cv_evaluator.py:51` | HIGH | Implement CV evaluator | N/A |
| 33-3 | `validation/evaluation/walk_forward_evaluator.py:51` | HIGH | Implement walk-forward evaluator | N/A |
| 33-4 | `core/container.py:673` | HIGH | Remove MultiResolution4DAdapter import | N/A |
| 33-5 | `core/container.py:739` | HIGH | Remove MultiStreamAdapter import | N/A |
| 33-6 | `features/compute/momentum.py:322-341` | MEDIUM | Vectorize CCI computation | 5-10x |
| 33-7 | `features/compute/mean_reversion.py:250-300` | MEDIUM | Vectorize variance ratio | 10-20x |
| 33-8 | `features/compute/order_flow.py:53-103` | MEDIUM | Add caching to order flow features | 3-4x |
| 33-9 | `features/compute/regime.py:53-86,120-135` | MEDIUM | Add caching to regime features | 3x |
| 33-10 | `features/compute/wavelets.py:62-88` | MEDIUM | Apply numba to wavelet transform | 10-50x |
| 33-11 | `features/compute/mean_reversion.py:156-200` | MEDIUM | Replace Hurst with O(n) algorithm | 5-10x |

### Success Metrics

| Metric | Before | After | How to Verify |
|--------|--------|-------|---------------|
| NotImplementedError count | 3 | 0 | Run all validation tests |
| Core → Data imports | 2 | 0 | `grep "from src.data" src/core/` returns 0 |
| CCI computation time | Baseline | 10-20% | Profile CCI features |
| Variance ratio time | Baseline | 5-10% | Profile mean reversion |
| Order flow time | Baseline | 25-30% | Profile with caching |
| Wavelet time | Baseline | 2-5% | Profile with numba |
| **Overall pipeline speedup** | 100% | **60-70%** | Full pipeline benchmark |

---

## Phase 34: Cleanup & Consolidation

**Status:** ✅ COMPLETE
**Priority:** MEDIUM
**Effort:** Single day (actual)
**Source:** Comprehensive ML pipeline review (9-agent analysis, 2026-02-01)
**Completed:** 2026-02-01

### Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        ORPHANED FILES & DUPLICATES                               │
│                                                                                  │
│  ORPHANED FILES (0 imports):                                                     │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  core/features/__init__.py              - Empty placeholder    │             │
│  │  core/training/__init__.py              - Empty placeholder    │             │
│  │  core/types_pkg/__init__.py             - Unused re-export     │             │
│  │  data/store/lineage.py                  - Not integrated       │             │
│  │  data/store/versioning.py               - Not integrated       │             │
│  │  data/store/cache.py                    - Not integrated       │             │
│  │  pipeline/stages/features/cli.py        - Not connected        │             │
│  │  pipeline/stages/labeling/adaptive_barriers.py - Not used      │             │
│  └────────────────────────────────────────────────────────────────┘             │
│                                                                                  │
│  MTF TIMEFRAME INCONSISTENCIES:                                                  │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  core/constants.py:35       → ["5min", "15min", "60min"]      │             │
│  │  config/unified.py:270      → ["1min", "15min", "60min"]      │             │
│  │  adapters/multi_stream.py   → ["1min", "5min", "15min"]       │             │
│  │  → Three different defaults causing confusion                  │             │
│  └────────────────────────────────────────────────────────────────┘             │
│                                                                                  │
│  DATAFRAME FRAGMENTATION (Deferred from Phase 31):                               │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  117 patterns of df['col'] = value causing fragmentation       │             │
│  │  → Needs systematic batch concat refactoring                   │             │
│  └────────────────────────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Tasks

**6 Tasks Completed, 5 Tasks Disproven:**

| Task | File | Status | Description |
|------|------|--------|-------------|
| 34-1 | `core/features/__init__.py` | ✅ DELETED | Empty placeholder (0 imports) |
| 34-2 | `core/training/__init__.py` | ✅ DELETED | Empty placeholder (0 imports) |
| 34-3 | `core/types_pkg/__init__.py` | ✅ DELETED | Unused re-export layer (0 imports) |
| 34-4 | `data/store/lineage.py` | ❌ DISPROVEN | **IS integrated** - used by FeatureStore |
| 34-5 | `data/store/versioning.py` | ❌ DISPROVEN | **IS integrated** - used by FeatureStore |
| 34-6 | `data/store/cache.py` | ❌ DISPROVEN | **IS integrated** - used by FeatureStore |
| 34-7 | `pipeline/stages/features/cli.py` | ✅ DELETED | Standalone CLI not connected to unified CLI |
| 34-8 | `pipeline/stages/labeling/adaptive_barriers.py` | ❌ DISPROVEN | **IS integrated** - registered in factory |
| 34-9 | `core/constants.py` | ✅ UPDATED | Consolidated MTF defaults to `["1min", "5min", "15min", "60min"]` |
| 34-10 | `config/unified.py` + `adapters/multi_stream.py` | ✅ UPDATED | Both now import from constants |
| 34-11 | `features/compute/*.py` | ❌ DISPROVEN | **Already uses anti-fragmentation pattern** |

### Success Metrics

| Metric | Before | After | Result |
|--------|--------|-------|--------|
| Empty placeholder files | 3 | 0 | ✅ All deleted |
| Orphaned files verified | 5 claimed | 0 found | ✅ All integrated (claims disproven) |
| MTF default definitions | 3 | 1 | ✅ Single source in constants.py |
| Fragmentation patterns | 117 claimed | 0 found | ✅ Already using anti-fragmentation pattern |
| **Code cleanliness** | Good | **Excellent** | ✅ No dead code, single source of truth

---

## Execution Order

```
Phase 24 (Quick Wins) ────────┐
                              │
Phase 25 (Validation) ────────┼───▶ Can run in parallel (different files)
                              │
Phase 26 (Type Safety) ───────┘

                              ▼

Phase 27 (Architecture) ──────▶ Depends on 26 (type changes first)

                              ▼

Phase 28 (Compute) ───────────┐
                              ├───▶ Can run in parallel
Phase 29 (Memory) ────────────┘

                              ▼

Phase 30 (Adv Architecture) ──▶ Depends on 27

                              ▼

Phase 31 (Polish) ────────────▶ Ongoing, can start anytime
```

---

## Validation Commands

### Phase 24
```bash
# Profile trend features before/after
python -c "
import time
from src.data.features.compute import trend
import pandas as pd
df = pd.DataFrame({'high': [100]*1000, 'low': [99]*1000, 'close': [99.5]*1000})
start = time.time()
trend.compute_adx_14(df)
trend.compute_plus_di_14(df)
trend.compute_minus_di_14(df)
trend.compute_adx_strong_trend(df)
print(f'Time: {time.time()-start:.3f}s')
"
```

### Phase 25
```bash
# Verify validation is called
grep -r "validate_stage_transition" src/data/pipeline/stages/*/run.py
```

### Phase 26
```bash
# Count Any types in module-level caches and function signatures
grep -rn ": Any" src/ --include="*.py" | grep -v "test" | grep -v "dict\[str, Any\]" | wc -l
# Should be 0 (legitimate kwargs with dict[str, Any] excluded)

# Count bare excepts
grep -rn "except Exception:" src/ --include="*.py" | wc -l
```

### Phase 27
```bash
# Count class definitions
grep -r "class PredictionResult" src/ | wc -l  # Should be 1
grep -r "class AdapterResult" src/ | wc -l     # Should be 2 (documented exception)
grep -r "class DataContract" src/ | wc -l      # Should be 1
grep -r "class ModelContract" src/ | wc -l     # Should be 1
grep -r "class ModelContractViolation" src/ | wc -l  # Should be 1

# Test imports
python -c "from src.core.interfaces import PredictionResult; print('OK')"
python -c "from src.models.base import PredictionResult; print('OK')"
python -c "from src.inference.orchestrator import PredictionResult; print('OK')"

# Run tests
pytest tests/ -v  # Should pass all 42 tests
```

---

## Phase 66: Financial Rigor — ONC Clustering, Transaction Costs, DSR Gate, CPCV

4 improvements grounded in Lopez de Prado's Advances in Financial Machine Learning to increase prediction accuracy and reduce overfitting risk.

### Rationale
- ONC Clustered Feature Selection: Prevents substitution effect where correlated features dilute each other's MDA importance scores
- Transaction Costs in Optuna: Labels generated during optimization now include real trading costs, preventing selection of strategies that only profit on paper
- DSR Gate Enforcement: Prevents deployment of strategies whose Sharpe ratios are inflated by selection bias across multiple Optuna trials
- CPCV in Hyperparameter Tuning: More robust cross-validation with 15 backtest paths vs standard K-fold, reducing hyperparameter overfitting

### Success Metrics
| Metric | Before | After | Result |
|--------|--------|-------|--------|
| ONC clustering | Disabled (code existed) | Enabled (use_clustered_importance=True) | Wired |
| Transaction costs in Optuna | Disabled (apply_transaction_costs=False) | Enabled (True) | Wired |
| DSR gate | Advisory only (warnings) | Enforcement (ValueError on failure) | Conditional via config |
| CPCV support in tuning | Not available | Available via cv_method="cpcv" | Wired through pipeline |
| Tests passing | 212/212 | 212/212 | No regressions |

**Status: COMPLETE (2026-02-20)**

---

## Phase 63: CODEBASE_AUDIT Complete — All 12 Audit Fixes + 4 Smoke Test Bug Fixes

Addressed all Critical (C2-C4), High (H3-H5), Medium (M1-M2, M4, M6, M8), and Low (L3) items from CODEBASE_AUDIT.md. Additionally fixed 4 bugs discovered during comprehensive E2E smoke testing of all 12 models.

### Rationale
- Comprehensive codebase audit identified 12 actionable items across Critical, High, Medium, and Low severity
- C2: strict=True default for validators prevents silent schema drift
- C3: Symbol unknown error handling prevents silent failures
- C4: OOF leakage verification ensures train/test isolation
- H3: Lookahead propagation scan eliminates forward-looking bias
- H4: GitHub Actions CI/CD enables automated quality gates
- H5: Code deduplication eliminates 22 duplicate patterns across helpers and consumers
- M1: MDA-first feature filter improves feature selection quality
- M2: Orchestrator split reduces complexity (2470 lines to 747+553+738)
- M4: OOM degraded flag enables graceful degradation under memory pressure
- M6: Resampling parity ensures consistent cross-timeframe behavior
- M8: MDA threshold 500 to 200 enables MDA on smaller datasets
- L3: StrEnum modernization (51 classes across 33 files) improves type safety

### End-to-End Smoke Test — 4 Bugs Fixed

During comprehensive smoke testing (all 12 models, standard + walk-forward, 1 week MES, MTF, Optuna, backtesting):

- Walk-forward ensemble label alignment (`ensemble_service.py`) — fallback label extraction from source DataFrame
- Walk-forward 3D reshape for sequential models (`walk_forward.py`) — contract-aware reshaping + `_create_sequences`
- torch.compile state_dict prefix (`base_rnn.py`) — `removeprefix("_orig_mod.")` cleanup
- ClassVar for _PRESETS (`symbol.py`) — mutable dict default changed to `ClassVar`

### Success Metrics
| Metric | Before | After | Result |
|--------|--------|-------|--------|
| Audit fixes complete | 0/12 | 12/12 | ✅ |
| Smoke test bugs fixed | 0/4 | 4/4 | ✅ |
| Import cleanliness | unverified | 47/47 clean | ✅ |
| Circular imports | unverified | 0 | ✅ |
| StrEnum conversions | 0 | 51 classes (33 files) | ✅ |
| Orchestrator size | 2470 lines (1 file) | 747+553+738 lines (3 files) | ✅ |
| Duplicate patterns | 22 | 0 | ✅ |

**Status: COMPLETE (2026-02-19)**

---

## Phase 62: OPTIMIZATIONPLAN Complete — Final 5 Optimizations

Complete the remaining 5 optimizations from the verified pipeline optimization plan (21/21).
Projected pipeline runtime: ~42 min → ~6-8 min (80-85% faster). Zero accuracy impact.

### Rationale
- 2.1 (OOF fold caching) was the single biggest remaining win (~10-15 min saved)
- 3.2 (Hurst @njit) eliminated fragile conditional import dependency
- 2.3 leveraged existing in-memory handoff (5.2) for validation I/O
- 1.4 and 2.6 were minor cleanup completing the full plan

### Success Metrics
| Metric | Before | After | Result |
|--------|--------|-------|--------|
| Optimizations complete | 17.5/21 (83%) | 21/21 (100%) | ✅ |
| Remaining pd.Series.shift patterns | 2 | 0 | ✅ |
| Hurst @njit coverage | conditional | always | ✅ |
| OOF fold model caching | not implemented | implemented | ✅ |
| Validation I/O wired | partial | complete | ✅ |

**Status: COMPLETE (2026-02-19)**
**OPTIMIZATIONPLAN.md archived to COMPLETION.md and deleted.**

---

*See CLEANUP_TASKS.md for detailed file:line instructions*
*See COMPLETION.md for implementation details after completion*
