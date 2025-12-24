# Universal ML Pipeline - Documentation

**Vision:** Train ANY model type on OHLCV data through a modular, phase-based architecture.

---

## Quick Navigation

| Goal | Document |
|------|----------|
| **Get started in 15 min** | [QUICKSTART.md](getting-started/QUICKSTART.md) |
| **Run the pipeline** | [PIPELINE_CLI.md](getting-started/PIPELINE_CLI.md) |
| **Understand the system** | [ARCHITECTURE.md](reference/ARCHITECTURE.md) |
| **Feature catalog** | [FEATURES.md](reference/FEATURES.md) |

---

## Phase Specifications

| Phase | Status | Description | Spec |
|-------|--------|-------------|------|
| **1** | COMPLETE | Data Preparation & Labeling | [PHASE_1.md](phases/PHASE_1.md) |
| **2** | IN DESIGN | Model Factory (train any model) | [PHASE_2.md](phases/PHASE_2.md) |
| **3** | PLANNED | Cross-Validation & OOS Predictions | [PHASE_3.md](phases/PHASE_3.md) |
| **4** | PLANNED | Ensemble Stacking & Meta-Learner | [PHASE_4.md](phases/PHASE_4.md) |
| **5** | PLANNED | Full Integration & Production | [PHASE_5.md](phases/PHASE_5.md) |
| **6** | FUTURE | Central Orchestrator + Frontend | TBD |

---

## Documentation Structure

```
docs/
├── README.md              # This file
├── getting-started/       # Onboarding
│   ├── QUICKSTART.md      # 15-min setup guide
│   └── PIPELINE_CLI.md    # Complete CLI reference
├── reference/             # Technical specifications
│   ├── ARCHITECTURE.md    # System design
│   ├── FEATURES.md        # Feature catalog (107 indicators)
│   └── SLIPPAGE.md        # Transaction cost modeling
├── phases/                # Frozen phase specifications
│   ├── PHASE_1.md         # Data prep (COMPLETE)
│   ├── PHASE_2.md         # Model training
│   ├── PHASE_3.md         # Cross-validation
│   ├── PHASE_4.md         # Ensemble
│   └── PHASE_5.md         # Production
└── archive/               # Historical docs (not maintained)
```

---

## Quick Commands

```bash
# Run Phase 1 pipeline
./pipeline run --symbols MES --stages all

# Check outputs
ls data/splits/scaled/

# Run tests
pytest tests/phase_1_tests/

# Use Phase 1 outputs in Python
from src.stages.datasets import TimeSeriesDataContainer
container = TimeSeriesDataContainer.from_parquet_dir('data/splits/scaled', horizon=20)
```

---

## Key Metrics (Phase 1)

| Metric | Value |
|--------|-------|
| **Codebase** | ~10K lines Python |
| **Features** | 107 technical indicators |
| **Horizons** | H5, H10, H15, H20 |
| **Split Ratio** | 70/15/15 |
| **Purge/Embargo** | 60/288 bars |
| **Output** | Model-ready parquet files |

---

## Engineering Principles

1. **Modularity** - No monoliths, clear separation
2. **650-line limit** - Forces good decomposition
3. **Fail fast** - Validate at boundaries
4. **Less code** - Simplicity wins
5. **Delete unused** - Git is the archive

---

## Archived Documentation

Historical documents (reports, fix logs, status updates) are preserved in `docs/archive/` for reference but are not actively maintained.
