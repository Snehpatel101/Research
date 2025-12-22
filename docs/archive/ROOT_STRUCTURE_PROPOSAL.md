# Root Directory Organization Proposal

## Current Issues

The root directory currently contains **8 loose markdown files** that should be organized:

```
/home/jake/Desktop/Research/
├── CLAUDE.md ✓ (keep - project instructions)
├── README.md ✓ (keep - entry point)
├── FILE_REFACTORING_STATUS.md ❌ (should be in docs/archive/)
├── FINAL_SUMMARY.md ❌ (should be in docs/archive/)
├── MODULAR_ARCHITECTURE.md ❌ (should be in docs/reference/architecture/)
├── PHASE1_PIPELINE_REVIEW.md ❌ (should be in docs/reference/reviews/)
├── REFACTORING_SUMMARY_PHASE1.md ❌ (should be in docs/archive/)
├── VOLATILITY_STATIONARITY_FIX.md ❌ (should be in docs/archive/)
```

---

## Proposed Root Structure (Clean & Professional)

```
Research/
│
├── README.md                    # Entry point - quick start
├── CLAUDE.md                    # Engineering rules
├── LICENSE
├── pyproject.toml
├── pytest.ini
├── requirements.txt
├── setup.py
├── MANIFEST.in
├── .gitignore
├── pipeline .................... # CLI executable
│
├── src/ ....................... # Source code
│   ├── config.py
│   ├── manifest.py
│   ├── __init__.py
│   ├── pipeline/ .............. # Pipeline orchestration
│   ├── stages/ ................ # Data pipeline stages
│   ├── models/ ................ # [PHASE 2] Model implementations
│   ├── data/ .................. # [PHASE 2] Dataset loaders
│   ├── training/ .............. # [PHASE 2] Training infrastructure
│   └── utils/ ................. # Shared utilities
│
├── config/ .................... # Configuration files
│   ├── barriers/ .............. # Barrier parameters
│   ├── features/ .............. # Feature configs
│   └── models/ ................ # [PHASE 2] Model configs
│
├── data/ ...................... # Data directory
│   ├── raw/
│   ├── clean/
│   ├── features/
│   ├── final/
│   └── splits/
│
├── docs/ ...................... # All documentation
│   ├── README.md .............. # Documentation index
│   ├── guides/ ................ # User guides
│   ├── phases/ ................ # Phase specifications
│   ├── reference/ ............. # Technical reference
│   │   ├── architecture/
│   │   ├── reviews/
│   │   ├── technical/
│   │   └── tasks/
│   └── archive/ ............... # Historical docs
│
├── tests/ ..................... # All tests
│   ├── test_*.py
│   └── phase_1_tests/
│
├── scripts/ ................... # Utility scripts
│   └── verify_installation.sh
│
├── notebooks/ ................. # Jupyter notebooks
│
├── models/ .................... # Saved models
│   ├── base/
│   └── ensemble/
│
├── results/ ................... # Pipeline results
│   └── *.json, *.md
│
├── reports/ ................... # Generated reports
│
├── logs/ ...................... # Log files
│
└── runs/ ...................... # Pipeline run artifacts
```

---

## Migration Plan

### Step 1: Organize Root Markdown Files (10 minutes)

**Move to docs/archive/**
```bash
mv FILE_REFACTORING_STATUS.md docs/archive/
mv FINAL_SUMMARY.md docs/archive/
mv REFACTORING_SUMMARY_PHASE1.md docs/archive/
mv VOLATILITY_STATIONARITY_FIX.md docs/archive/
```

**Move to docs/reference/architecture/**
```bash
mv MODULAR_ARCHITECTURE.md docs/reference/architecture/
```

**Move to docs/reference/reviews/**
```bash
mv PHASE1_PIPELINE_REVIEW.md docs/reference/reviews/
```

**Update docs/archive/README.md:**
```markdown
# Archive

Historical documentation preserved for context:

- `FILE_REFACTORING_STATUS.md` - Dec 2025 refactoring tracking
- `FINAL_SUMMARY.md` - Phase 1 transformation summary
- `REFACTORING_SUMMARY_PHASE1.md` - Phase 1 refactoring details
- `VOLATILITY_STATIONARITY_FIX.md` - Volatility stationarity fix log
```

---

### Step 2: Consolidate Configuration (30 minutes)

**Create config/ package:**
```
config/
├── __init__.py ............ Export all configs
├── base.py ................ Base config + validation
├── barriers.py ............ Barrier parameters
├── features.py ............ Feature configs
├── paths.py ............... Path configuration
└── models/ ................ [PHASE 2] Model configs
    ├── xgboost.yaml
    ├── nhits.yaml
    └── tft.yaml
```

**Merge src/config.py + src/pipeline_config.py:**
- Move path definitions → `config/paths.py`
- Move barrier params → `config/barriers.py`
- Move feature configs → `config/features.py`
- Keep validation in `config/base.py`

---

### Step 3: Clean Up Legacy Files (5 minutes)

**Delete archived files:**
```bash
rm src/stages/feature_scaler_old.py  # 1,729 lines (replaced by package)
rm src/stages/stage2_clean_old.py    # 967 lines (replaced by package)
```

---

### Step 4: Update Documentation Links (15 minutes)

**Update README.md:**
```markdown
## Documentation
- [Getting Started](docs/guides/00_GETTING_STARTED.md)
- [Phase 1 Spec](docs/phases/PHASE_1_Data_Preparation_and_Labeling.md)
- [Architecture Review](docs/reference/reviews/ARCHITECTURE_REVIEW_PHASE1_PHASE2.md)
```

**Update docs/README.md:**
- Update links to moved files
- Add new Phase 2 documentation references

---

## Benefits of Proposed Structure

### 1. Clean Root Directory
- Only essential files (README, LICENSE, configs, executable)
- Professional appearance
- Easy navigation

### 2. Logical Organization
- All docs in `docs/` with clear hierarchy
- All configs in `config/` package
- All source in `src/` with clear modules

### 3. Phase 2 Ready
- Clear locations for new components:
  - `src/models/` for model implementations
  - `src/data/` for dataset loaders
  - `src/training/` for training orchestration
  - `config/models/` for model configs

### 4. Maintainability
- Historical docs preserved in `docs/archive/`
- Current docs easily discoverable
- Clear separation of reference vs guides

---

## One-Command Migration

Create a migration script:

```bash
#!/bin/bash
# migrate_root_structure.sh

set -e

echo "Migrating root structure..."

# Move markdown files
mv FILE_REFACTORING_STATUS.md docs/archive/
mv FINAL_SUMMARY.md docs/archive/
mv REFACTORING_SUMMARY_PHASE1.md docs/archive/
mv VOLATILITY_STATIONARITY_FIX.md docs/archive/
mv MODULAR_ARCHITECTURE.md docs/reference/architecture/
mv PHASE1_PIPELINE_REVIEW.md docs/reference/reviews/

# Delete legacy files
rm src/stages/feature_scaler_old.py
rm src/stages/stage2_clean_old.py

# Create config package structure
mkdir -p config/models

echo "Migration complete!"
echo "Next steps:"
echo "1. Update docs/archive/README.md"
echo "2. Update documentation links in README.md and docs/README.md"
echo "3. Test pipeline still runs: ./pipeline run --symbols MES,MGC"
```

---

## Verification Checklist

After migration:

- [ ] Root directory only has essential files
- [ ] All loose .md files organized in docs/
- [ ] Legacy _old.py files deleted
- [ ] Documentation links updated
- [ ] Pipeline still runs: `./pipeline run --symbols MES,MGC`
- [ ] Tests still pass: `python -m pytest tests/`

---

## Timeline

- **Step 1-3**: 45 minutes (file organization)
- **Step 4**: 15 minutes (update links)
- **Total**: 1 hour

**Recommended:** Do this immediately before starting Phase 2 work to have a clean foundation.
