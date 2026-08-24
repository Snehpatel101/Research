# Handoff — Stage 7: Target Interfaces (labelling config honesty)

**Date:** 2026-08-24
**Predecessor:** Stage 6 (`06_feature_interfaces.md`)

---

## 1. Verification of predecessor

| Stage 6 claim | How I checked | Verdict |
|---|---|---|
| Suite 636p/2xf/0f (638) | Re-ran full suite | **CONFIRMED** |
| Defaults unchanged, knobs live | Re-ran `F5_feature_config_wiring.py`, exit 0 | **CONFIRMED** |
| 12/18 FeatureEngineer params now passed | Read `factory.py` | **CONFIRMED** |
| R14: `features.*` indicator params still dead | Confirmed — no constructor params exist for them | **CONFIRMED** |

Stage 6's hypothesis — that `labeling.*` would have the same disease — was
correct, and the disease turned out to be worse.

## 2. Findings

### 2.1 Seven of thirteen labelling fields have ZERO consumers

Counted `labeling.<field>` / `.labeling.<field>` across `src/`:

| Field | Consumers | Old default | What it implies |
|---|---|---|---|
| `optimize_barriers` | **0** | **True** | barrier optimisation runs |
| `optimization_trials` | **0** | 100 | ...with 100 trials |
| `vertical_barrier_enabled` | **0** | True | time barrier is optional |
| `barrier_touch_is_exit` | **0** | True | exit semantics configurable |
| `use_dynamic_barriers` | **0** | True | barriers adapt |
| `min_return_threshold` | **0** | 0.0 | a min-move filter exists |
| `target_class_balance` | **0** | None | class balancing available |
| `upper_mult` / `lower_mult` / `max_holding_bars` | 2–3 | — | **live** |
| `atr_period` | 2 | — | **live** |
| `binary_mode` | 2 | — | **live** |
| `method` | 1 | — | validation only |

This is a stronger failure than F15. F15's settings were *passed nowhere*;
these are *read nowhere*. And `optimize_barriers=True` was the **default** —
every reader of this config was told an expensive optimisation step was
running when the field is inert.

### 2.2 The whole validation system was dead

`LabelingConfig.validate()` exists with real bounds checks. So do the other
config classes'. **All 20 `.validate()` call sites in `src/` are
`super().validate()` inside another config class.** Nothing outside the config
package ever invoked it.

That mattered for the fix: adding a warning to `validate()` would have been a
warning nobody would ever see — repeating the exact bug I was fixing.

## 3. Architectural decisions

- **Declared, not deleted.** `UNWIRED_FIELDS` enumerates the seven dead fields
  rather than removing them. Deleting config fields breaks users' saved YAML,
  and deletion is `DECISIONS.md` #8 — the user's call, not mine. Meanwhile a
  user who sets one is now told it does nothing.
- **Changed `optimize_barriers` default True → False.** Safe (zero consumers)
  and it stops the API asserting something false. A default that implies a
  working feature is a lie in the interface.
- **Switched validation ON at `MLFactory.__init__`.** This is the systemic
  half — it makes every pre-existing bounds check live too, not just my new
  ones.
- **Warnings, not exceptions.** Turning a never-executed validator into a hard
  error would reject configs that run fine today. Promoting to errors is a
  deliberate follow-up, not a side effect of switching it on.

## 4. Implementation changes

| File | Change |
|---|---|
| `src/config/data.py` | `UNWIRED_FIELDS` map; inert-field detection in `validate()`; `optimize_barriers` default → `False` |
| `src/factory.py` | `_log_config_issues()`, called from `__init__` — first real invocation of the validation chain |
| `tests/test_config_honesty.py` | **new**, 11 tests |

## 5. Live evidence

```
--- default config (should be SILENT) ---
(end)

--- config setting an UNWIRED field (should WARN) ---
WARNING config issue [data.labeling]: vertical_barrier_enabled=False has NO EFFECT: ...
WARNING config issue [data.labeling]: optimize_barriers=True has NO EFFECT: ...
WARNING config issue [data.labeling]: optimization_trials=500 has NO EFFECT: ...
```

Silent on defaults; precise and actionable when a user sets something inert.

## 6. Tests

`tests/test_config_honesty.py` — 11 passing:

- `test_unwired_map_matches_reality` — greps `src/` for each declared-dead
  field. **If someone wires one up without updating the map, this fails and
  says so.** The map cannot rot.
- `test_defaults_are_the_inert_values` — a default may never advertise a
  feature that does not exist.
- `test_optimize_barriers_no_longer_defaults_true` — pins the worst offender.
- `test_real_bounds_checks_still_work` — the new checks must not shadow the
  old ones.
- `TestValidationActuallyRuns` — validation fires at construction, and stays
  quiet on a default config.
- `TestBarrierParamsRemainLive` — the six working fields must never be
  reported inert.

## 7. Failures encountered

One of my own tests failed with
`TypeError: not all arguments converted during string formatting` — I wrote
`r.message % r.args if r.args else ...`, and the ternary bound wrong. The
warning was firing correctly all along (visible in the captured log).
`LogRecord.getMessage()` already applies the args. Fixed.

## 8. Quantitative results

| Metric | Before | After |
|---|---|---|
| Labelling fields that silently do nothing | 7 (undeclared) | 7 (**declared + warned**) |
| Misleading defaults | 1 (`optimize_barriers=True`) | **0** |
| Config `validate()` invocations outside the config package | **0** | **1** (every run) |
| New tests | — | 11 |

**No labelling behaviour changed.** Labels produced before and after this
stage are identical — the only behavioural change is that inert settings now
announce themselves.

## 9. Unresolved risks

- **R1–R15 (carried).** R9 (contract built, producers unmigrated) still
  largest.
- **R16 (new).** `_log_config_issues()` covers five `data.*` sub-configs.
  `training.*`, `optuna.*`, `calibration.*`, `checkpoint.*` and the ensemble
  configs are **not** validated, and `DECISIONS.md` #8 says several have the
  same dead-field problem. Extending coverage is mechanical; do it when
  touching those areas.
- **R17 (new).** Warnings are easy to ignore in a long run log. Once the
  unwired set is confirmed stable, consider promoting "user set an inert
  field" to a hard error — that is a genuine behaviour change and needs
  sign-off.
- **R18 (new).** The same audit has not been done for `FeatureConfig`
  (Stage 6 R14 lists 8 suspects), `ScalerConfig`, `SplitConfig`. Expect
  similar ratios.

## 10. Instructions for the next agent (Stage 8 — canonical model protocol)

1. **Re-run first:** full suite (expect **647 passed, 2 xfailed, 0 failed**),
   plus `F3`, `F4`, `F5` evidence scripts, all exit 0.
2. **Do not remove the F1 xfail markers.**
3. Stage 8 builds `ModelCapabilities` as the single source of truth. Phase 0
   measured the drift it must eliminate (V12): `patchtst`, `itransformer` and
   `transformer` each report family `neural` from the class and `transformer`
   from the contract; `mlp_meta` disagrees on `requires_scaling`.
   **Re-measure before fixing** — Stage 5 showed a Phase 0 finding can be
   wrong.
4. Put it in or beside `src/core/model_requirements.py` (moved there in
   Stage 3, verified light in Stage 4). Do **not** import `src.data` or
   `src.models` from it.
5. The `UNWIRED_FIELDS` pattern generalises. If Stage 8 finds capability
   fields nothing reads, declare them the same way rather than deleting.
6. Still **UNKNOWN**: whether any ensemble beats its constituents and a
   baseline. Blocked on Stage 17 and Stage 12.
