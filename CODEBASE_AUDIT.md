# ML Factory Codebase Audit

**Date:** 2026-02-17 | **Scope:** Full codebase, 13 agents (11 parallel + 2 sequential) | **Overall Grade: B+**

---

## Deliverable 1: What Changed (Phases 53-59) and What It Impacts

### Theme 1: Security Hardening

**What changed:** All 36 pickle deserialization sites were migrated from raw `joblib.load` to `safe_pickle_load` with allowlist enforcement. Zero raw load calls remain. SymbolConfig was extracted into a standalone class with MES/MGC/MNQ presets and explicit resample anti-lookahead parameters on all inference sites.

**What it impacts going forward:** The attack surface for arbitrary code execution via malicious model files is closed. Any new model persistence site must use the safe loader or it will be the only unsafe site in the codebase, making it easy to catch in review. The SymbolConfig presets mean new symbols must be added explicitly rather than falling through to silent defaults (though the fallback path still exists — see Deliverable 2, item C3).

### Theme 2: End-to-End Pipeline Reliability

**What changed:** Five distinct bugs were fixed across Phases 54-56. Model persistence (`Trainer.save()`), per-model feature selection replacing global truncation, 4D multi-stream data wiring for transformers, timeframe key normalization (1h to 60min), empty test split guards, deploy manifest model naming, and the backtest prediction extraction pipeline. Phase 57 added 4D OOF generation enabling cross-family ensembles (boosting + transformer).

**What it impacts going forward:** The pipeline now runs 10 of 12 models end-to-end without manual intervention. Cross-family ensembles are structurally possible for the first time. However, none of this is covered by automated tests, so any refactor to the OOF, backtest, or deployment paths risks silent regression. The fixes were verified manually; that verification is not repeatable without human effort.

### Theme 3: Feature Selection Overhaul

**What changed:** Phases 58-59 replaced variance-based feature ranking with Model-agnostic permutation importance (MDA), wired low-variance and correlation pre-filters into the orchestrator, and made feature selection per-model (respecting each model contract's `max_features`). A fallback to variance ranking exists if MDA fails.

**What it impacts going forward:** Feature subsets are now target-aware, which should improve model quality. The per-model approach prevents the previous bug where global feature truncation created conflicts between models with different dimensionality requirements. The MDA-first ordering is correct in principle, but the filter sequencing in the orchestrator runs low-variance and correlation filters before MDA, which means MDA operates on a pre-filtered set rather than the full feature space. This is suboptimal but not incorrect — it is a design tradeoff favoring speed over optimality.

### Theme 4: Robustness Guards

**What changed:** Embargo-bar overflow no longer crashes the test split (Phase 59). Empty dataframes from date-range filtering are handled. Optuna flags propagate correctly when `n_trials=0`. The backtest timestamp column mismatch is resolved.

**What it impacts going forward:** These guards make the pipeline more tolerant of edge-case data configurations (short histories, unusual date ranges, skip-optimization runs). But the guards are permissive — they skip steps silently rather than failing loudly. A user who misconfigures embargo bars will get a model trained without test evaluation and may not notice.

---

## Deliverable 2: Prioritized Risks, Regressions, and Architectural Inconsistencies

### CRITICAL — Must fix before production deployment

| # | Issue | Impact | What Could Go Wrong |
|---|-------|--------|---------------------|
| C1 | **Zero automated tests on the validation layer (42 files)** | The layer responsible for the system's #1 guarantee (no data leakage) has no automated test coverage. A regression here is undetectable until it causes financial loss. | A subtle off-by-one in purge/embargo logic goes undetected. Models appear profitable in backtest but fail in production because they were trained on future data. No test catches the regression when someone refactors the CV splitter. |
| C2 | **Config validation not enforced (strict=False)** | Invalid configurations produce silent incorrect results. A wrong symbol config means wrong point values, wrong P&L calculations, wrong position sizing. This is a direct path to financial loss. | A production config has a typo in a safety parameter. The system silently uses an unsafe default. The model trains on leaked data. Backtest looks great. Live performance is random. The root cause is a single character typo that strict validation would have caught instantly. |
| C3 | **Unknown symbol fallback uses MES defaults silently** | Trading ZB (Treasury bonds, $31.25/tick) with MES defaults ($12.50/tick) produces 2.5x wrong P&L calculations. System should fail, not guess. | A user adds a new symbol "MYM" (Micro Dow). The system falls back to generic defaults. Commission and tick size are wrong. Backtest shows profitability. In production, the actual transaction costs consume all the edge. |
| C4 | **No post-training OOF leakage validation** | The system validates leakage prevention in CV setup but never verifies it held after training completes. Subtle leakage through feature engineering or label construction would go undetected. | A model reports 85% accuracy on OOF predictions. The ensemble is deployed. In live trading it performs at 51% because the OOF accuracy was inflated by leakage. There is no automated check that would catch this before deployment. |
| C5 | **No E2E integration tests in pytest** | Manual verification scripts exist but are not in the automated test suite. Regressions across module boundaries are not caught automatically. | A change to adapter output shape breaks downstream model training but nothing catches it until a user runs the full pipeline manually. The Phase 54-56 bugs (5 distinct failures) were all integration-level — they would have been caught by E2E tests. |

### HIGH — Significant risk, fix in near term

| # | Issue | Impact | What Could Go Wrong |
|---|-------|--------|---------------------|
| H1 | **No train-test distribution shift detection** | Model deployed on data with different statistical properties than training data will produce unreliable predictions with no warning. | The model trains on a low-volatility period and is deployed into a high-volatility regime. OOF metrics looked good because train and validation sets shared the same regime. No distribution shift warning was raised. |
| H2 | **Label lookahead not verified** | Triple-barrier labels use future price data by design, but there's no automated check that the label construction window doesn't overlap with feature data in unexpected ways. | A contributor adds a momentum feature using `close.pct_change(5)` without shifting. It implicitly contains the current bar's close. The feature ranks highly via MDA (because it contains future information). The model appears excellent. It fails in production. |
| H3 | **Cross-feature lookahead not detected** | Feature engineering could inadvertently use future information through indicator chains. No automated scan exists for this. | Feature A depends on Feature B, and B has unshifted lookahead. Feature A inherits the lookahead but no audit catches it. |
| H4 | **No CI/CD pipeline** | Even the tests that exist don't run automatically on code changes. | A contributor pushes a commit that breaks imports. Nobody notices until the next person pulls and tries to run the pipeline. |
| H5 | **318 duplicate class/function definitions** | Duplicates create maintenance burden and divergence risk where one copy gets fixed but the other does not. | A bug fix is applied to one copy of a utility function. The other copy, used by a different code path, retains the bug. |
| H6 | **Resample config warnings only, not blocking** | Anti-lookahead resample parameters log warnings for suspicious configs but allow execution to proceed. | An automated nightly training run uses a config with incorrect resample parameters. The system logs a warning to a file nobody reads. Models are trained with lookahead. |
| H7 | **No CLI pre-flight validation** | Pipeline fails mid-execution on bad config rather than at startup. A multi-hour training run can fail after hours of compute. | User launches a 4-hour training run with an invalid horizon. Fails at hour 3 during evaluation. |
| H8 | **Missing horizon validation** | Any integer accepted as prediction horizon with no validation against the config's temporal constraints. | User requests horizon=999. Pipeline proceeds until it runs out of data, producing cryptic errors. |

### MEDIUM — Should fix, not immediately dangerous

| # | Issue | Impact |
|---|-------|--------|
| M1 | **Feature selection filter ordering suboptimal** | Low-variance → Correlation → MDA. MDA-first would produce better feature sets since MDA captures predictive value while correlation is target-agnostic. |
| M2 | **9 files over 1000 lines (orchestrator at 2,260)** | Maintenance and comprehension burden. Orchestrator effectively untestable at that size. |
| M3 | **Ensemble diversity not auto-run** | Ensemble validation exists but isn't part of the standard pipeline. Correlated models reduce ensemble value. |
| M4 | **OOM recovery silently reduces batch size** | Could change effective training dynamics without notification. Results may not be comparable to non-OOM runs. |
| M5 | **Embargo auto-scaling insufficient for large timeframes** | On 4-hour or daily bars, the default embargo may be too few bars. |
| M6 | **MTF resampling parity not verified training↔inference** | If resampling logic differs, inference predictions won't match training conditions. |
| M7 | **Primary model selection fallback could miss ensemble** | Deploy manifest selects best individual model; ensemble might outperform. |
| M8 | **MDA variance fallback threshold too conservative** | 500 rows threshold may be too high; smaller datasets forced to variance ranking. |

### LOW — Cleanup items

| # | Issue |
|---|-------|
| L1 | Neural model subclasses don't document inherited save/load |
| L2 | float('inf') returned from some metrics edge cases |
| L3 | 8 StrEnum modernization opportunities |
| L4 | 23 ruff issues (all optional modernization) |
| L5 | GA mentioned in docs but Optuna TPE is the actual implementation |

---

## Deliverable 3: Concrete Recommendations

### 3A. Phased Roadmap

#### Phase 60 — Testing Foundation (highest leverage, do first)

1. Write 5 integration tests covering the primary pipeline paths: (a) single-model boosting, (b) single-model transformer (4D), (c) multi-timeframe, (d) cross-family ensemble, (e) backtest-to-trades. Each test should run a minimal config (tiny dataset, 2 folds, no Optuna) and assert that the pipeline completes without error and produces outputs of the expected shape.

2. Write unit tests for the purged K-fold splitter verifying: no overlap between train and validation indices, embargo bars are excluded, and purge window is applied correctly. This directly addresses C1 and C4.

3. Write a test that deliberately introduces lookahead (an unshifted feature) and asserts that validation metrics are suspiciously high, establishing a baseline "leakage canary."

4. Set up a minimal CI configuration (GitHub Actions or equivalent) that runs `ruff check`, `black --check`, the import verification commands from CLAUDE.md, and the new test suite on every push.

#### Phase 61 — Config Enforcement (second highest leverage)

1. Change the default `strict` parameter to `True` for all config classes. Add an explicit opt-out (`strict=False`) for backward compatibility during migration, but log a deprecation warning when it is used.

2. Add a `validate()` method to SymbolConfig that raises an error (not a warning) when an unknown symbol is used without explicit parameters. Remove the silent fallback to defaults.

3. Make resample config validation blocking. Change warnings to errors for anti-lookahead parameters that are unset or inconsistent.

4. Add a CLI pre-flight validation step that runs config validation before any pipeline execution begins.

#### Phase 62 — Deduplication and Decomposition

1. Audit the 318 duplicate definitions. Categorize into: (a) true duplicates (delete one), (b) near-duplicates (consolidate), (c) intentional variants (document why). Target reducing to under 50.

2. Split the orchestrator (2,260 lines) into three files: orchestration coordination, feature selection orchestration, and training orchestration. Each under 800 lines.

3. Apply the same decomposition to any other file over 1,000 lines where the file contains more than one conceptual responsibility.

#### Phase 63 — Runtime Safety Validation

1. Implement post-training OOF leakage detection: verify that no OOF prediction index overlaps with its corresponding training fold indices, accounting for purge and embargo.

2. Add a train-test distribution shift detector using a simple two-sample test (KS or PSI) on the top 10 features by MDA importance.

3. Add runtime label lookahead verification: for each label column, verify that the label at time `t` does not use price data from time `t` or later.

### 3B. Refactor Plan

| Current State | Target State | Rationale |
|---------------|-------------|-----------|
| Orchestrator: 1 file, 2,260 lines | 3 files, under 800 each | Testability, reviewability |
| 318 duplicate definitions | Under 50, all documented | Single source of truth |
| Feature selection: variance → correlation → MDA | MDA on full feature space, then correlation on top-N | MDA is the most informative signal |
| Config `strict=False` default | Config `strict=True` default | Silent misconfiguration is the easiest way to introduce leakage |
| Warnings for safety violations | Errors for safety violations, with explicit override | A trading system should fail safe, not fail silent |

### 3C. Testing Priorities (ordered by damage-prevention value)

1. **Purged K-fold correctness** — If this is wrong, every model evaluation is fraudulent
2. **E2E pipeline smoke tests** (5 paths) — Catches integration failures that caused Phases 54-56
3. **OOF index integrity** — If OOF leaks, ensemble and backtest metrics are meaningless
4. **Config strict mode** — If config is wrong, everything downstream is wrong
5. **Adapter output shape contracts** — If adapters produce wrong shapes, models train on garbage
6. **Backtest P&L arithmetic** — If P&L is wrong, all trading decisions are wrong
7. **MTF alignment** — If timeframes are misaligned, features contain lookahead
8. **Safe pickle load enforcement** — Regression test that no raw `joblib.load` reappears

### 3D. Guardrails to Make Blocking

| Currently Warns | Should Block | Migration Path |
|-----------------|-------------|----------------|
| Unknown symbol config | Error: "Unknown symbol X. Define explicit SymbolConfig or use a preset." | Add symbol validation to pipeline entry point |
| Resample anti-lookahead params unset | Error: "Resample config requires explicit lookahead parameters." | Add validation to resample config constructor |
| Config misspelled keys (strict=False) | Error: "Unknown config key: lokahead_bars. Did you mean: lookahead_bars?" | Change default to strict=True |
| Embargo insufficient for timeframe | Warning with suggested minimum, error if below hard floor | Add `min_embargo_bars` per timeframe to SymbolConfig |
| OOM batch reduction | Warning + metric flag on output indicating reduced reliability | Add `training_degraded: true` flag to model metadata |

---

## Deliverable 4: Long-Term Standards

### 4A. Contracts and Interfaces

**Model Contract Discipline:** Every model must implement the existing `ModelContract` and `TrainerProtocol`. Any new model must pass a contract compliance test verifying: it accepts the declared input rank, respects `max_features`, produces outputs of the declared shape, and round-trips through save/load without data loss.

**Adapter Contract Discipline:** Every adapter must produce an `AdapterResult` with verified shape and dtype. Add a shape-assertion decorator that validates output dimensions match the model contract's expected input rank.

**Feature Contract:** Every feature should declare its lookback window and its shift. The pipeline should verify that no feature's effective lookback extends beyond available history, and that all shifts are >= 1 (no current-bar data in features). This contract does not exist today and should be introduced.

**Validation Contract:** Every validation function should return a structured result (pass/fail, metric value, threshold, explanation) rather than logging. This enables programmatic aggregation and deployment gating.

### 4B. Naming and Module Boundaries

**Dependency ordering (imports only go downward):**

```
core (types, contracts)             — depends on nothing
config                              — depends on core
data (adapters, features, pipeline) — depends on core, config
models                              — depends on core, config, data
optimization                        — depends on core, config, data, models
validation                          — depends on core, config, data
inference                           — depends on everything above
cli                                 — depends on everything above
```

Any import that violates this ordering is a dependency inversion and should be resolved by extracting the shared dependency into `core`.

**Naming Conventions:**
- Classes: PascalCase, noun phrases (`PurgedKFoldSplitter`, not `DoPurgedKFold`)
- Functions: snake_case, verb phrases (`compute_mda_importance`, not `mda_importance`)
- Config keys: snake_case, matching the parameter name they control exactly
- Test files: `test_` prefix, matching the file they test (`test_purged_kfold.py` tests `purged_kfold.py`)

**No file should exceed 800 lines.** If a file approaches this limit, it contains more than one responsibility and should be split.

### 4C. Keeping the Codebase Aligned as It Grows

**The Four-Document System works. Keep it.** DIRECTION.md, CLEANUP_PLAN.md, CLEANUP_TASKS.md, and COMPLETION.md provide governance that most production codebases lack entirely.

**Add a fifth document: TEST_PLAN.md.** Mirror the structure of CLEANUP_PLAN.md but track testing coverage by module. Each phase should include a "testing requirement" section. No phase is complete until its tests pass.

**Definition of Done for phases:**
1. Implementation complete
2. Tests written and passing
3. Linting and formatting clean
4. CLEANUP_TASKS.md updated
5. No new warnings-instead-of-errors introduced
6. No new files over 800 lines
7. No new duplicate definitions

**Quarterly Audit Cadence:** Run this same multi-agent audit quarterly. Track the overall grade and layer scores over time. The goal is monotonic improvement: B+ now, A- next quarter, A the quarter after.

### 4D. Governance Model for Changes

**Tier 1 — Core Changes (requires explicit approval):**
Any change to `src/core/types.py`, `src/core/contracts/`, the dependency ordering above, or the four root documents. These are architectural load-bearing walls.

**Tier 2 — Safety Changes (requires review + test):**
Any change to validation logic, purge/embargo parameters, anti-lookahead guards, config enforcement, or pickle loading. Every change must include a test that would fail if the safety guarantee were violated.

**Tier 3 — Feature Changes (requires test):**
Any new model, adapter, feature computation, or optimization strategy. Must include unit tests and must pass existing E2E smoke tests.

**Tier 4 — Maintenance Changes (standard review):**
Refactoring, documentation updates, formatting, dependency updates.

**The cardinal rule: warnings are technical debt. Every warning that is not an error is a future production incident.** The long-term standard should be: detect, block, explain, require explicit override.

---

## Appendix: Layer Health Scores

| Layer | Score | Assessment |
|-------|-------|------------|
| **Core (types, contracts, enums)** | 10/10 | Exemplary. Zero duplication, zero circular imports, clean re-exports. |
| **Data Pipeline** | 9.5/10 | Near-perfect. 12 stages, enforced dependencies, correct anti-lookahead, train-only scaling. |
| **Model Layer** | 9/10 | Consistent interface, unified protocol, full save/load, architecture versioning. |
| **Inference & Backtesting** | 9/10 | Production-grade. Realistic costs, circuit breakers, self-contained bundles. |
| **Optimization** | 7.5/10 | Good. MDA is right; filter ordering suboptimal; fallback thresholds need tuning. |
| **Validation & CV** | 7/10 | Solid structure, missing post-training checks and zero test coverage. |
| **Config & CLI** | 5.5/10 | Good class hierarchy, poor enforcement. Validation defaults to off. |
| **Testing** | 3/10 | 72 tests, 38% coverage, zero tests on critical subsystems, no CI/CD. |

---

## Appendix: Top 10 Strengths

1. Zero-leakage data pipeline (4 independent protection layers)
2. Core type system integrity (zero duplicates, zero circular imports)
3. Complete security posture (36 safe_pickle_load sites, zero raw loads)
4. Production-grade inference (UniversalInferencePipeline, bundles, circuit breakers)
5. Realistic financial backtesting (commission, exchange fees, 4 slippage models)
6. Cross-family ensemble support (2D boosting + 4D transformers)
7. Target-aware feature selection (MDA + per-model subsets)
8. Code quality discipline (96-99% doc coverage, clean formatting)
9. Comprehensive model coverage (12 models, unified interface, TrainerProtocol)
10. Documentation governance (4-doc system, 59 phases tracked)

---

*Audit conducted with Claude Opus 4.6 — 11 parallel specialist agents + 2 sequential consolidation agents*
