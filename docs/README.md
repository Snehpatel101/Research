# Documentation Index

Welcome to the Ensemble Price Prediction Pipeline documentation.

## Quick Navigation

### 🚀 New Users Start Here
- [Getting Started Guide](guides/00_GETTING_STARTED.md) - Learn what this project does and run your first pipeline
- [Pipeline CLI Guide](guides/01_PIPELINE_CLI.md) - Complete command reference

### 📚 User Guides
All guides are numbered for progressive learning:

1. **[Getting Started](guides/00_GETTING_STARTED.md)** - Overview, concepts, and quick start
2. **[Pipeline CLI](guides/01_PIPELINE_CLI.md)** - Complete CLI command reference
3. **[Labeling Guide](guides/02_LABELING_GUIDE.md)** - Triple-barrier labeling and GA optimization
4. **[Validation Guide](guides/03_VALIDATION_GUIDE.md)** - Quality checks and testing
5. **[Migration Guide](guides/MIGRATION.md)** - Historical reference for standardization changes

### 📖 Phase Specifications
Canonical specifications for each phase of the project:

- **[Phase 1: Data Preparation and Labeling](phases/PHASE_1_Data_Preparation_and_Labeling.md)** ✅ Complete
- **[Phase 2: Training Base Models](phases/PHASE_2_Training_Base_Models.md)** 🚧 Next
- **[Phase 3: Cross-Validation OOS Predictions](phases/PHASE_3_Cross_Validation_OOS_Predictions.md)**
- **[Phase 4: Train Ensemble Meta-Learner](phases/PHASE_4_Train_Ensemble_Meta_Learner.md)**
- **[Phase 5: Full Integration Final Test](phases/PHASE_5_Full_Integration_Final_Test.md)**

### 🔧 Technical Reference

#### Architecture Documentation
- [Codebase Reorganization Summary](reference/architecture/CODEBASE_REORGANIZATION_SUMMARY.md) - Recent restructuring
- [Pipeline Comparison Report](reference/architecture/PIPELINE_COMPARISON.md) - Example vs main pipeline analysis
- [Standardization Plan](reference/architecture/STANDARDIZATION_PLAN.md) - Phase 11 standardization

#### Reviews and Audits
- [Phase 1 Comprehensive Review](reference/reviews/PHASE1_COMPREHENSIVE_REVIEW.md) - Multi-agent analysis
- [Phase 11 Verification Report](reference/reviews/PHASE11_VERIFICATION_REPORT.md) - Standardization verification
- [Stage 1 Comprehensive Review](reference/reviews/STAGE1_COMPREHENSIVE_REVIEW_REPORT.md) - Data ingestion review
- [Phase 4 Labeling Consolidation](reference/reviews/PHASE4_LABELING_CONSOLIDATION.md) - Labeling system review

#### Technical Specifications
- [Features Catalog](reference/technical/FEATURES_CATALOG.md) - All implemented features
- [Phase 1 Deliverables](reference/technical/PHASE1_DELIVERABLES.md) - Phase 1 goals and outcomes
- [Phase 1 Modules Summary](reference/technical/PHASE1_MODULES_SUMMARY.md) - Module-level documentation
- [Stage Modules README](reference/technical/STAGE_MODULES_README.md) - Stage implementation details

#### Task Specifications
- [Architecture Review Tasks](reference/tasks/TASK_ARCHITECTURE_REVIEW.md)
- [Data Engineering Tasks](reference/tasks/TASK_DATA_ENGINEERING.md)
- [Quantitative Analysis Tasks](reference/tasks/TASK_QUANT_ANALYSIS.md)
- [Security & Reliability Tasks](reference/tasks/TASK_SECURITY_RELIABILITY.md)
- [Testing & Quality Tasks](reference/tasks/TASK_TESTING_QUALITY.md)

### 📦 Examples
- [VWAP LSTM Examples](examples/vwap_lstm/) - Example implementations

### 🗄️ Archive
- [Historical Documentation](archive/README.md) - Outdated docs preserved for context

---

## Documentation Organization

```
docs/
├── README.md                    # ← You are here
│
├── guides/                      # User-facing guides (start here!)
│   ├── 00_GETTING_STARTED.md
│   ├── 01_PIPELINE_CLI.md
│   ├── 02_LABELING_GUIDE.md
│   ├── 03_VALIDATION_GUIDE.md
│   └── MIGRATION.md
│
├── phases/                      # Canonical phase specifications
│   ├── PHASE_1_Data_Preparation_and_Labeling.md
│   ├── PHASE_2_Training_Base_Models.md
│   ├── PHASE_3_Cross_Validation_OOS_Predictions.md
│   ├── PHASE_4_Train_Ensemble_Meta_Learner.md
│   └── PHASE_5_Full_Integration_Final_Test.md
│
├── reference/                   # Technical reference documentation
│   ├── architecture/            # System architecture
│   ├── reviews/                 # Historical reviews and audits
│   ├── technical/               # Technical specifications
│   └── tasks/                   # Task definitions
│
├── examples/                    # Example code and implementations
│   └── vwap_lstm/
│
└── archive/                     # Historical documentation
    └── README.md
```

---

## By Role

### I'm a User (Want to run the pipeline)
1. Start: [Getting Started Guide](guides/00_GETTING_STARTED.md)
2. Reference: [Pipeline CLI Guide](guides/01_PIPELINE_CLI.md)
3. Troubleshoot: [Validation Guide](guides/03_VALIDATION_GUIDE.md)

### I'm a Developer (Want to extend the code)
1. Architecture: [Codebase Reorganization Summary](reference/architecture/CODEBASE_REORGANIZATION_SUMMARY.md)
2. Standards: [CLAUDE.md](/CLAUDE.md) - Engineering rules
3. Specifications: [Phase 1 Spec](phases/PHASE_1_Data_Preparation_and_Labeling.md)

### I'm a Researcher (Want to understand the system)
1. Overview: [Getting Started Guide](guides/00_GETTING_STARTED.md)
2. Analysis: [Phase 1 Comprehensive Review](reference/reviews/PHASE1_COMPREHENSIVE_REVIEW.md)
3. Improvements: [Pipeline Comparison](reference/architecture/PIPELINE_COMPARISON.md)

### I'm a Quant (Want to improve labeling/features)
1. Labeling: [Labeling Guide](guides/02_LABELING_GUIDE.md)
2. Features: [Features Catalog](reference/technical/FEATURES_CATALOG.md)
3. Tasks: [Quantitative Analysis Tasks](reference/tasks/TASK_QUANT_ANALYSIS.md)

---

## Current Status

**Phase 1: ✅ Complete (Production-Ready)**
- Score: 7.5/10
- Triple-barrier labeling with GA optimization
- 50+ features, 3 horizons (H1, H5, H20)
- Quality-weighted labels
- Proper purge/embargo for leakage prevention

**Phase 2: 🚧 Next**
- Train N-HiTS, TFT, PatchTST base models
- Implement TimeSeriesDataset
- Hyperparameter tuning with Optuna

**Expected Performance (Phase 5 Test):**
- Sharpe: 0.5-1.2 (H20)
- Win Rate: 48-55%
- Max Drawdown: 8-18%

---

## Quick Commands

```bash
# Run pipeline
./pipeline run --symbols MES,MGC

# Check status
./pipeline status <run_id>

# Validate
./pipeline validate --symbols MES,MGC

# Get help
./pipeline --help
```

---

## Contributing

See [CLAUDE.md](/CLAUDE.md) for:
- Engineering rules (non-negotiables)
- File and complexity limits (650 lines max)
- Testing requirements
- Definition of done

---

## Version History

- **2025-12-21**: Documentation reorganization and consolidation
- **2025-12-20**: Phase 1 comprehensive review (Score: 7.5/10)
- **2025-12-19**: Codebase reorganization and standardization
- **Earlier**: Initial implementation and development

---

**Need help?** Start with [Getting Started](guides/00_GETTING_STARTED.md) or check the [CLI Guide](guides/01_PIPELINE_CLI.md).
