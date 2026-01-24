# Comprehensive Codebase Analysis - 2026-01-23

## Executive Summary

Multi-agent deep analysis performed on 2026-01-23 covering 6 major subsystems. The ML Factory demonstrates mature architecture with sophisticated data provenance, but has significant configuration complexity and technical debt requiring attention.

---

## Key Architectural Findings

### 1. Pipeline Architecture (12 Stages)

**Structure:**
```
Stage 1: Ingest    - Raw OHLCV validation/standardization
Stage 2: Clean     - Resampling, gap detection, outliers
Stage 3: Features  - 150+ technical indicators, MTF enrichment
Stage 4: Labeling  - Triple-barrier with ATR-based barriers
Stage 5: GA Opt    - Optuna TPE barrier optimization
Stage 6: Final     - Quality scores, sample weights
Stage 7: Splits    - Chronological train/val/test, purge/embargo
Stage 7.5: Scale   - Per-timeframe robust/standard/minmax
Stage 7.6: Dataset - Model-ready parquet splits
Stage 7.7: ScaleVal- Post-scaling drift detection
Stage 8: Validate  - Feature selection via correlation/variance
Stage 9: Report    - HTML report generation
```

**Key Finding:** Only 2/12 stages fully implement the clean run.py/core.py separation pattern. The rest have mixed concerns.

### 2. ML/Optimization System

**Optuna TPE Optimization:**
- Search Space: k_up [0.8-2.5], k_down [0.8-2.5], max_bars_mult [2.0-3.0]
- Safe Mode: Uses first 70% of data to prevent test leakage (default)
- Symbol-specific asymmetry: MES asymmetric, MGC symmetric seeding

**Fitness Function:**
| Component | Weight | Purpose |
|-----------|--------|---------|
| Neutral Score | PRIMARY | Target 20-30% neutral rate |
| Long/Short Balance | +2 | Reward 50-50 split |
| Speed Score | +1.5 | Prefer faster barrier hits |
| Profit Factor | +2 | MFE/MAE ratio |
| Transaction Penalty | -10 to +0.5 | Cost awareness |

**Meta-Labeling:** Primary classifier (70% recall target) -> Binary correctness -> Bet sizing

### 3. Configuration System

**Dual-Layer Architecture:**
```
Level 1: GlobalConfig (YAML-based, application-wide)
Level 2: UnifiedConfig (high-level API with sections)
Level 3: PipelineConfig (user-facing, Phase 0)
Level 4: DataConfig (internal Phase 1)
Level 5: Stage-specific (barriers, feature sets, regimes)
```

**Critical Issue (CFG-002):** 71+ duplicated `_get_global_or_default()` patterns scattered across codebase.

**Feature Sets (50+ Aliases):**
- `core_min` - 30-50 base features
- `boosting_optimal` - XGBoost/LightGBM optimized
- `neural_optimal` - LSTM/GRU normalized
- `transformer_raw` - Minimal for foundation models

### 4. Data Store/Versioning (7,522 lines)

**Components:**
- FeatureStore (897 lines) - Unified storage with Parquet caching
- FeatureCache (647 lines) - Content-addressable SHA256 storage
- LineageTracker (620 lines) - Full audit trail
- VersionManager (445 lines) - Semantic versioning
- Data Adapters (4,771 lines) - Model-specific format conversions

**Cache Key Formula:**
```
cache_key = SHA256(input_checksum + config_hash + symbol + version)
```

**Semantic Versioning:**
- Major bump: Schema changes (columns added/removed)
- Minor bump: Config changes (backward compatible)
- Patch bump: Recomputation (same output)

### 5. Features/Labeling System

**160+ Features in 9 Families:**
1. Raw OHLCV (4)
2. Momentum (~40) - RSI, MACD, Stochastic, MAs
3. Volatility (~30) - ATR, Bollinger, Historical Vol
4. Volume (~20) - VWAP, OBV, Volume Ratios
5. Microstructure (~30) - VPIN, Kyle's Lambda
6. Wavelets (~30) - Daubechies db4 decomposition
7. MTF (~20) - Multi-timeframe indicators
8. Regime (~10) - Volatility/Trend detection
9. Temporal - Hour/Day-of-week encoding

**Quality Scoring Weights:**
- Speed (20%), MAE (25%), MFE (20%), Pain/Gain (20%), Drawdown (15%)
- Sample weights: Tier 1 (1.5x), Tier 2 (1.0x), Tier 3 (0.5x)

**Anti-Lookahead:** shift(1) applied on ALL MTF features

### 6. Inference/Contracts System

**Contract Types:**
- DataContract: Schema validation, DataRank (2D/3D/4D), FeatureMode
- ModelContract: 23 models with required features, sequence length, scaler type
- ArtifactManifest: SHA256 hashes, environment capture, lineage

**Bundle Structure:**
```
bundle_dir/
  manifest.json
  metadata.json
  features.json
  scaler.pkl
  calibrator.pkl (optional)
  preprocessing_graph.json (optional)
  model/
```

**PreprocessingGraph:** 50+ parameters for train/serve parity

---

## Configuration System State

### Current State: Complex but Functional

**Hierarchy Issues:**
- UnifiedConfig vs PipelineConfig creates confusion
- No clear "single source of truth" for settings
- Cross-config validation missing (CompositeValidator needed)

**Migration Needed:**
- 71+ `_get_global_or_default()` patterns need consolidation to `get_config_value()`
- Constants scattered across multiple locations need centralization

**Working Components:**
- YAML-based GlobalConfig validated
- Feature set definitions (11 sets) functional
- Symbol-specific barrier configs operational

---

## Technical Debt Identified

### Priority 1 (Critical)

| ID | Issue | Location | Impact |
|----|-------|----------|--------|
| CFG-002 | 71+ duplicated config patterns | Multiple files | Maintenance nightmare |
| PIPE-008 | Single-symbol enforcement | Pipeline core | Blocks batch processing |
| SCALE-001 | Fragile string matching for feature columns | Scaling stage | Silent failures |

### Priority 2 (High)

| ID | Issue | Location | Impact |
|----|-------|----------|--------|
| VAL-001 | Hardcoded validation thresholds | Validation stage | Inflexibility |
| CFG-005 | No cross-config validation | Config system | Silent conflicts |
| ML-001 | Asymmetry bonus duplicated | fitness.py + optuna_optimizer.py | Inconsistency risk |

### Priority 3 (Medium)

| ID | Issue | Location | Impact |
|----|-------|----------|--------|
| LIN-001 | Lineage queries O(n) | LineageTracker | Performance at scale |
| PIT-001 | DateTime column name assumptions | FeatureStore | Brittleness |
| CONC-001 | No concurrent access testing | Data store | Production risk |

### Maintenance Debt

| ID | Issue | Effort |
|----|-------|--------|
| CORE-001 | Missing core.py for 10/12 stages | 5-7 days |
| DOC-001 | Feature sets reference guide needed | 1 day |
| GRAPH-001 | PreprocessingGraph has ~50 parameters | Refactor needed |

---

## Recommendations for Next Steps

### Immediate (This Week)

1. **CFG-002 Migration:** Create script to migrate `_get_global_or_default()` to centralized `get_config_value()`
2. **VAL-001:** Move validation thresholds (0.85 correlation, 0.01 variance) to config
3. **SCALE-001:** Replace string matching with explicit column lists

### Short-Term (Next 2 Weeks)

1. **Cross-Config Validation:** Implement CompositeValidator for config consistency
2. **Lineage Indexing:** Add SQLite index to LineageTracker for O(1) queries
3. **Feature Sets Documentation:** Create comprehensive reference guide

### Medium-Term (Next Month)

1. **Stage Pattern Extraction:** Extract core.py for remaining 10 stages
2. **Batch Processing:** Remove single-symbol constraint (PIPE-008)
3. **PreprocessingGraph Presets:** Create common preset configurations

### Long-Term

1. **Distributed Cache:** Add S3/GCS option for FeatureCache
2. **Concurrent Access:** Implement and test file locking
3. **Streaming Inference:** Add batch streaming option

---

## Model Status Summary

| Family | Count | Status |
|--------|-------|--------|
| Boosting | 3 | Complete |
| Classical | 3 | Complete |
| Neural | 10 | Complete |
| Ensemble | 3 | Complete |
| Meta-Learners | 4 | Complete |
| **Total** | **23** | **All Registered** |

---

## Pipeline Status Summary

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | OHLCV Ingestion | Complete |
| 2 | MTF Upscaling (9 TF) | Complete |
| 3 | Feature Engineering | Complete |
| 4 | Labeling + Splits | Complete |
| 5 | Adapters (2D/3D/4D) | Complete |
| 6 | Model Training (23 models) | Complete |
| 7 | Meta-Learner Stacking | Complete |

---

## Critical Bugs Status

All 5 previously documented bugs are FIXED:
1. HMM Lookahead Bias - shift(1) applied
2. GA Test Data Leakage - safe_mode default
3. Transaction Costs in Labels - Properly integrated
4. MTF/Regime shift(1) - Applied at output
5. LightGBM num_leaves - Constraint validated

No new critical bugs identified in this analysis.

---

**Analysis Performed By:** Phase 2 Agent #4
**Date:** 2026-01-23
**Phase 1 Inputs:** 6 agent handoffs analyzed
