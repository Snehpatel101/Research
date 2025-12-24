# Documentation Hub

Use this directory for current, non-phase documentation. Phase specifications live in `docs/phases/` and `docs/phase1/` and should be treated as frozen unless explicitly updated.

## Quick Navigation

| Goal | Document |
|------|----------|
| Get started | `getting-started/QUICKSTART.md` |
| CLI reference | `getting-started/PIPELINE_CLI.md` |
| Architecture | `reference/ARCHITECTURE.md` |
| Feature catalog | `reference/FEATURES.md` |
| Slippage model | `reference/SLIPPAGE.md` |

## Quick Commands

```bash
# Run Phase 1
./pipeline run --symbols MES,MGC

# Check status
./pipeline status <run_id>

# Validate config and artifacts
./pipeline validate --run-id <run_id>
```

## Phase Status

| Phase | Status | Description | Spec |
|-------|--------|-------------|------|
| 1 | Implemented | Data prep + labeling | `phases/PHASE_1.md` |
| 2 | Planned | Model factory | `phases/PHASE_2.md` |
| 3 | Planned | CV + OOS predictions | `phases/PHASE_3.md` |
| 4 | Planned | Ensemble | `phases/PHASE_4.md` |
| 5 | Planned | Production | `phases/PHASE_5.md` |

## Key Defaults (Phase 1)

- Horizons: `[5, 10, 15, 20]`
- Timeframe: `5min`
- Splits: `70/15/15`
- Purge/embargo: auto-scaled from horizons (embargo defaults to 1440 bars unless overridden)

## Doc Structure

```
docs/
├── README.md
├── getting-started/
│   ├── QUICKSTART.md
│   └── PIPELINE_CLI.md
├── reference/
│   ├── ARCHITECTURE.md
│   ├── FEATURES.md
│   └── SLIPPAGE.md
├── phase1/
│   └── README.md
└── phases/
    ├── PHASE_1.md
    ├── PHASE_2.md
    ├── PHASE_3.md
    ├── PHASE_4.md
    └── PHASE_5.md
```
