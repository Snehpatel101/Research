# Documentation Index

## Quick Navigation

### Essential Docs
- **[Phase 1 Master Documentation](PHASE1_MASTER_DOCUMENTATION.md)** - Complete Phase 1 reference
- **[Lookahead Prevention Guide](LOOKAHEAD_PREVENTION_GUIDE.md)** - Critical leakage prevention concepts

### User Guides
1. [Getting Started](guides/00_GETTING_STARTED.md) - Overview and quick start
2. [Pipeline CLI](guides/01_PIPELINE_CLI.md) - Command reference
3. [Labeling Guide](guides/02_LABELING_GUIDE.md) - Triple-barrier labeling
4. [Validation Guide](guides/03_VALIDATION_GUIDE.md) - Quality checks

### Phase Specifications
- [Phase 1: Data Preparation](phases/PHASE_1_Data_Preparation_and_Labeling.md) - **Complete**
- [Phase 2: Training Base Models](phases/PHASE_2_Training_Base_Models.md)
- [Phase 3: Cross-Validation](phases/PHASE_3_Cross_Validation_OOS_Predictions.md)
- [Phase 4: Ensemble Meta-Learner](phases/PHASE_4_Train_Ensemble_Meta_Learner.md)
- [Phase 5: Final Integration](phases/PHASE_5_Full_Integration_Final_Test.md)

### Future Plans
- [Future Plans Index](future_plans/README.md)
- [Phase 2 Architecture](future_plans/phase2/) - Model training design
- [Session Normalization Design](future_plans/SESSION_AWARE_NORMALIZATION_DESIGN.md)
- [Data Ingestion Architecture](future_plans/DATA_INGESTION_ARCHITECTURE.md)

### Archive
- [Historical Documentation](archive/README.md) - Reviews, fixes, and dated docs

---

## Directory Structure

```
docs/
├── README.md                        # This file
├── PHASE1_MASTER_DOCUMENTATION.md   # Main Phase 1 reference
├── LOOKAHEAD_PREVENTION_GUIDE.md    # Leakage prevention
│
├── guides/                          # User guides
│   ├── 00_GETTING_STARTED.md
│   ├── 01_PIPELINE_CLI.md
│   ├── 02_LABELING_GUIDE.md
│   └── 03_VALIDATION_GUIDE.md
│
├── phases/                          # Phase specifications (DO NOT MODIFY)
│   ├── PHASE_1_*.md
│   ├── PHASE_2_*.md
│   ├── PHASE_3_*.md
│   ├── PHASE_4_*.md
│   └── PHASE_5_*.md
│
├── future_plans/                    # Enhancement designs
│   ├── phase2/                      # Phase 2 implementation guides
│   ├── SESSION_AWARE_NORMALIZATION_DESIGN.md
│   └── DATA_INGESTION_ARCHITECTURE.md
│
├── examples/                        # Example implementations
│
└── archive/                         # Historical documentation
```

---

## Current Status

**Phase 1: Complete** - Data preparation pipeline with 107 features, triple-barrier labeling, GA optimization, and purge/embargo splitting.

**Next: Phase 1 Enhancements** - See [ML_FACTORY_TODO.md](/ML_FACTORY_TODO.md) for dynamic factory features (sessions, MTF, horizons, labeling strategies).

---

## Quick Commands

```bash
./pipeline run --symbols MES,MGC
./pipeline status <run_id>
./pipeline --help
```

---

See [CLAUDE.md](/CLAUDE.md) for engineering rules and standards.
