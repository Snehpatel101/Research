# Phase 1 — The 23 Sequential Stages

**Rule for every stage:** `inspect previous state → verify by execution →
investigate → modify → test → live-test → document evidence → hand off`.

No stage trusts its predecessor's claims. Each re-runs the predecessor's
evidence scripts before doing its own work. Handoffs go in
`docs/program/handoffs/NN_<name>.md` using `HANDOFF_TEMPLATE.md`.

---

## Deviation from the brief's ordering (explained, as required)

The brief lists stage 1 as "baseline/runtime characterization". **Phase 0
already delivered that** — measured suite (593/597), measured end-to-end run
(exit 0 / SUCCESS / MCC −0.0176), dataset inventory, and 16 verified findings.
Repeating it would be ceremony.

More importantly, Phase 0 proved the **measuring instruments are themselves
broken**: the lookahead auditor crashes (F13), the run summary prints
`F1=0.0000` regardless of truth (F16), macro-F1 credits pure noise (§4d), and
zero tests assert a model learned anything (§4e). Every later stage's "verify
your predecessor" duty depends on those instruments.

**So stage 1 becomes "Instrumentation & baseline hardening."** You cannot
verify stages 2–23 with instruments that lie. Everything else keeps the
brief's dependency order.

---

| # | Stage | Primary objective | Gate (must be true to proceed) |
|---|---|---|---|
| 1 | **Instrumentation & baseline hardening** | Fix F13/F16; land control fixtures + F1 repro as permanent tests; ban tautological tests | Suite green; controls separate noise/signal; F1 repro pinned as xfail |
| 2 | Architectural cleanup | Delete verified-dead code (**gated on user decision #1**); retire duplicate `AdapterResult`/`TrainingResult` | Suite green; no import regressions |
| 3 | Package/module organisation | Break the `src.core ↔ models ↔ data` cycle enough to import contracts without torch | `import src.core.contracts` without ML stack |
| 4 | **Canonical data contracts** | `PredictionSet` — keyed by source-bar/timestamp, explicit `valid` mask, retire `-99/-999/-1/NaN` sentinels | Round-trip tests; sentinel count → 0 |
| 5 | Preprocessing/transform interfaces | Scaler fit/apply as a declared, fold-scoped step | Scaler leak test passes |
| 6 | Feature interfaces | Feature-set selection reachable from `MLFactory` (**fixes F15**) | A cheap feature set is requestable → enables fast tier |
| 7 | Target interfaces | Label recipe as a first-class, fingerprinted object | Label/backtest parity test |
| 8 | **Canonical model protocol** | `TimeSeriesModel` Protocol + `ModelCapabilities` as single source of truth | All 23 models conform; drift (F10) structurally impossible |
| 9 | **Model registry / plugin architecture** | Open registry; delete `== 23` assertion (F11); entry-point discovery | A 24th model registers without editing `src/core` |
| 10 | Classical/statistical integration | Promote `logistic`/`random_forest`/`svm`; **add `DummyModel` baselines (F6)** | Baselines available to every evaluation |
| 11 | Tree/boosting integration | Conform to protocol; capability-declared | Conformance suite green |
| 12 | Deep-learning integration | Conform 3D models; **test metrics for all ranks (F7)** | 10/16 models gain held-out metrics |
| 13 | Advanced time-series integration | 4D models; resolve contract `sequence_length` (**user decision #3**) | TCN ≥ receptive field; no mode skew (F9) |
| 14 | Training orchestration | One orchestrator; **stop swallowing exceptions** (164 sites) | A failing model surfaces, not silently drops |
| 15 | Prediction/inference standardisation | Every path emits `PredictionSet` | Bundle round-trip preserves keys |
| 16 | Calibration & uncertainty | Calibration on held-out data only | Calibration improves log-loss on control signal set |
| 17 | **Ensemble primitives** | Fix F1 — key-based alignment; F1 repro flips xfail→pass | Two perfect models agree 100% |
| 18 | Advanced ensemble / meta-model | `EnsembleStrategy` registry replacing the 4-entry dict (F12); purged meta-split (F8) | ≥6 strategies; meta-CV purged |
| 19 | **Compatibility & composition validation** | `CompositionValidator` pre-flight; adjudicate 4D mixed-rank (**user decision #2**) | Invalid combos fail clearly *before* data loads |
| 20 | Evaluation & temporal validation | Baselines mandatory; MCC/accuracy-vs-majority headline; DM test + MCS | Ensemble-vs-constituent claims are significance-tested |
| 21 | Experiment/config/reproducibility | Manifest with `models:` pool + named `compositions:`; fit once, reuse OOF | `A+B+C→X` then `A+D+F→Y` without retraining A |
| 22 | Integration, observability, stress | Combination matrix; malformed inputs; repeated-run determinism | Matrix runs; determinism proven |
| 23 | **Final adversarial audit** | Break it deliberately; production-readiness matrix | Honest statement of proven / partial / broken / unvalidated |

---

## Standing rules

- **Metrics:** headline go/no-go claims use accuracy-vs-majority-baseline and
  MCC. Never macro-F1 alone — it gains +0.168 on pure noise (Phase 0 §4d).
- **Never** skip, disable, or weaken a test to reach green.
- **Never** route around a broken path to make a matrix pass. Reproduce →
  isolate → root-cause → repair → rerun → regression-test.
- A number without a stated baseline is not a result.
- Statistical validity ≠ runtime success. Report them separately.
