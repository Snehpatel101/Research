# Handoff — Stage 6: Feature Interfaces (F15)

**Date:** 2026-08-24
**Predecessor:** Stage 5 (`05_scaling_interfaces.md`)

---

## 1. Verification of predecessor

| Stage 5 claim | How I checked | Verdict |
|---|---|---|
| Suite 631p/2xf/0f (633) | Re-ran full suite | **CONFIRMED** |
| Scaler leakage refuted; affine cancellation exact | Re-ran `F4_double_scaling.py`, exit 0 | **CONFIRMED** |
| Both scalers reject non-affine at construction | 8 invariant tests pass | **CONFIRMED** |
| No production behaviour changed | `git show --stat` on the Stage 5 commit: only tests, comments, evidence | **CONFIRMED** |

Stage 5's instruction — *treat Phase 0 findings as hypotheses* — was applied
here. F15 survived re-verification, unlike the scaler finding.

## 2. F15 confirmed, with a sharper diagnosis

`MLFactory._run_data_pipeline` constructed `FeatureEngineer` with **2 of 18**
parameters:

```python
engineer = FeatureEngineer(input_dir=..., output_dir=...)
```

while the *other* caller (`features/run.py:242`) threads twelve. So every
`data.features.*` and `data.mtf.*` setting was dead on the documented entry
point. Agent 6's Phase 0 run showed the symptom: `cfg.data.mtf.enabled=False`
computed MTF anyway.

Beyond what Phase 0 reported, the defaults themselves were consequential:
- `enable_mtf=True` — MTF always on, ignoring config.
- `enable_wavelets=True` — the family Agent 6 saw producing **21 columns that
  are 100% NaN and are then dropped**. Pure wasted computation.
- `timeframe="5min"` on **1-minute** data. Since `base_timeframe` defaults to
  `"5min"` too, `PeriodScaler` gets a 1:1 ratio, so indicator periods are not
  numerically distorted — but the value drives the MTF `>base` filter, and it
  was unreachable from config.

## 3. Design decision: behaviour-preserving by construction

Wiring config through *could* silently change every existing result. I chose
the mapping so the **default `ExperimentConfig` reproduces the old feature set
exactly**:

| Config field | FeatureEngineer param | Default → old default? |
|---|---|---|
| `mtf.primary_timeframe` | `timeframe`, `base_timeframe` | `"5min"` → `"5min"` ✓ |
| `mtf.enabled` | `enable_mtf` | `True` → `True` ✓ |
| `mtf.timeframes` | `mtf_timeframes` | `["5min","15min","60min"]`, filtered to `>base` → `["15min","60min"]` ✓ |
| `mtf.aggregate_ohlcv/indicators` | `mtf_include_*` | `True` → `True` ✓ |
| `features.mode == "minimal"` | the four `enable_*` toggles | `"full"` → all `True` ✓ |

So: no surprise result changes, and the knobs become live. Both halves are
asserted, because either alone would be a false claim.

## 4. Live evidence

`docs/program/evidence/F5_feature_config_wiring.py` (exit 0):

```
A. DEFAULT config reproduces OLD behaviour
   old: 205 columns, 809 rows
   new: 205 columns, 809 rows
   columns only in old: none
   columns only in new: none
   IDENTICAL: True

B. NON-DEFAULT settings take effect
   mtf.enabled=True  -> 27 MTF columns
   mtf.enabled=False ->  0 MTF columns
   features.mode='full'    -> 205 columns
   features.mode='minimal' -> 127 columns   (saves 78, 38%)
```

## 5. Failures encountered

My first probe reported `mtf.enabled` as **dead wiring** — 0 MTF columns in
*both* the on and off cases. The wiring was fine; my probe was wrong. MTF
columns are suffixed via `get_timeframe_suffix()`, so `"60min"` becomes
`_1h`, not `_60min`. I was searching for a suffix that never exists.

This nearly produced a false negative on my own fix, so `MTF_SUFFIXES` is now
a named constant in the test with the mistake documented beside it.

## 6. Implementation changes

| File | Change |
|---|---|
| `src/factory.py` | Thread 10 feature/MTF settings into `FeatureEngineer`; `minimal` mode support |
| `tests/test_feature_config_wiring.py` | **new**, 5 tests (defaults-unchanged + settings-live + anti-regression guard) |
| `docs/program/evidence/F5_feature_config_wiring.py` | **new** |

## 7. Quantitative results

| Metric | Before | After |
|---|---|---|
| `FeatureEngineer` params passed by `MLFactory` | 2 / 18 | **12 / 18** |
| `data.mtf.*` settings honoured | 0 / 6 | **4 / 6** |
| `features.mode='minimal'` reachable | no | **yes** (205 → 127 cols) |
| Default-config feature set | 205 cols | **205 cols (identical)** |
| New tests | — | 5, running in **8.2s** |

The 8.2s matters: this is the fast tier Agent 6 designed and could not build,
because a cheap feature set was unreachable.

## 8. Unresolved risks

- **R1–R12 (carried).** R9 (prediction contract built, producers unmigrated)
  is still the largest.
- **R13 (new).** Two `data.mtf` fields remain unwired: `mode`
  (`"indicators"`) and `aggregate_ohlcv` is mapped but `mtf_min_rows`,
  `scale_periods`, `nan_threshold` and the wavelet parameters are still
  defaulted. 12/18 is progress, not completion.
- **R14 (new).** `features.families`, `sma_periods`, `ema_periods`,
  `atr_periods`, `rsi_period`, `macd_params`, `bb_period`, `bb_std` are
  **still dead** — `FeatureEngineer` has no parameters for them. Wiring those
  needs indicator-level plumbing, not just constructor arguments. This is the
  larger half of F15 and is deliberately not claimed as done.
- **R15 (new).** `timeframe="5min"` describing 1-minute data is now
  *configurable* but still *wrong by default*. Setting
  `mtf.primary_timeframe="1min"` would be more honest, but it changes the MTF
  `>base` filter and therefore results — a deliberate experiment, not a
  drive-by fix. Flagged for Stage 13 (contract `sequence_length`), which has
  the same "changes results, needs sign-off" character.

## 9. Instructions for the next agent (Stage 7 — target interfaces)

1. **Re-run first:** full suite (expect **636 passed, 2 xfailed, 0 failed**),
   plus `F3`, `F4`, `F5` evidence scripts, all exit 0.
2. **Do not remove the F1 xfail markers.**
3. R14 is the honest remainder of F15. If Stage 7 touches labelling config,
   check whether `labeling.*` has the same dead-config problem before
   assuming it works — same failure mode, different sub-config.
4. **When probing for columns, do not guess suffixes.** Use
   `get_timeframe_suffix()`. My wrong probe (§5) nearly reported a working fix
   as broken.
5. `features.mode='minimal'` is now available for fast tests. Prefer it in any
   new end-to-end test to keep the suite quick.
6. Still **UNKNOWN**: whether any ensemble beats its constituents and a
   baseline. Blocked on Stage 17 and Stage 12.
