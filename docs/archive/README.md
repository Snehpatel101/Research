# Archived Documentation

**Purpose:** This directory contains historical documentation that is no longer part of the active project documentation tree. Files are archived (not deleted) to preserve context for future reference.

## Archive Policy

Documents are archived when they:
- Describe unimplemented features in excessive detail
- Contain research notes not needed for operational tasks
- Are superseded by newer documentation
- Represent historical planning artifacts

**Note:** Archived docs are NOT maintained and may contain outdated information.

---

## Archive Structure

### roadmaps/
Long-term implementation roadmaps for future enhancements:
- `MTF_IMPLEMENTATION_ROADMAP.md` - Multi-timeframe enhancement plan (9-timeframe ladder, raw OHLCV ingestion for sequence models)
- `ADVANCED_MODELS_ROADMAP.md` - Advanced model implementations (PatchTST, iTransformer, TFT, N-BEATS, InceptionTime, ResNet)
- `MIGRATION_ROADMAP.md` - Detailed migration strategy for MTF enhancement

**Use when:** Planning implementation of advanced MTF strategies or model families

### guides/
Strategy guides for unimplemented features:
- `MTF_STRATEGY_GUIDE.md` - Multi-timeframe data strategies (single-TF, MTF indicators, MTF ingestion)

**Use when:** Understanding the architectural vision for multi-resolution data handling

### research/
Research analysis and literature reviews:
- `BEST_OHLCV_MODELS_2025.md` - Survey of SOTA models for OHLCV forecasting
- `BEST_OHLCV_FEATURES.md` - Feature engineering research
- `FEATURE_REQUIREMENTS_BY_MODEL.md` - Model-specific feature needs
- `FEATURE_SELECTION_METHODS.md` - Feature selection literature
- `ADVANCED_FEATURE_SELECTION.md` - Advanced selection techniques

**Use when:** Researching new model architectures or feature engineering approaches

### analysis/
Historical analysis snapshots:
- `PHASE1_FEATURE_ENGINEERING_REALITY.md` - Phase 1 feature analysis (2025-12-24)
- `IMPLEMENTATION_TASKS.md` - MTF enhancement task breakdown

**Use when:** Understanding the evolution of the pipeline design

### reports/
Historical audit and analysis reports:
- `ML_PIPELINE_AUDIT_REPORT.md` - Pipeline audit snapshot

**Use when:** Reviewing past system audits

### features/
Feature configuration archives:
- `MTF_FEATURE_CONFIGS.md` - MTF feature configuration details
- `FEATURE_CATALOG.md` - Historical feature catalog

**Use when:** Researching feature configuration patterns

### troubleshooting/
Issue guides for unimplemented features:
- `MTF_TROUBLESHOOTING.md` - MTF-specific troubleshooting (mostly for unimplemented Strategy 3)

**Use when:** Debugging multi-timeframe issues after Strategy 3 implementation

### reference/
Historical technical references:
- `PIPELINE_FIXES.md` - Historical pipeline fixes and rationale

**Use when:** Understanding why certain pipeline decisions were made

---

## How to Use Archived Docs

**Do NOT treat archived docs as current:**
- Always check active docs first (`docs/`)
- Archived docs may reference code that no longer exists
- Archived docs may describe features that were never implemented

**Good uses:**
- Research background for new features
- Understanding historical design decisions
- Planning future enhancements

**Bad uses:**
- Learning current system architecture (use `docs/ARCHITECTURE.md`)
- Following implementation guides (use `docs/guides/`)
- Troubleshooting current issues (use `docs/troubleshooting/`)

---

## Restoration Policy

If an archived document becomes relevant again (e.g., feature is now being implemented):
1. Review for accuracy against current codebase
2. Update to reflect current architecture
3. Move back to active docs with clear purpose
4. Update active docs index to reference it

**Do not restore without review and updates.**
