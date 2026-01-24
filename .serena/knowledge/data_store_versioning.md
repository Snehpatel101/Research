# Data Store and Versioning System

**Created:** 2026-01-23 (from Phase 1 Agent #4 analysis)
**Total Lines:** 7,522

---

## Overview

Complete data provenance system providing:
- Feature storage with Parquet caching
- Content-addressable caching
- Full audit trail with lineage tracking
- Semantic versioning for datasets

---

## Core Components

### FeatureStore (897 lines)

Unified storage with Parquet caching.

**API:**
```python
from src.data_store import FeatureStore

store = FeatureStore(base_path="./data/features")

# Store features
store.put_features(df, symbol="MES", version="1.0.0")

# Retrieve features
df = store.get_features(symbol="MES", version="1.0.0")

# Point-in-time retrieval (for backtesting)
df = store.get_features_as_of(symbol="MES", as_of="2024-01-01")

# Lineage tracking
lineage = store.get_lineage(symbol="MES", version="1.0.0")

# Integrity validation
is_valid = store.validate_integrity(symbol="MES", version="1.0.0")
```

---

### FeatureCache (647 lines)

Content-addressable storage using SHA256 hashes.

**Cache Key Generation:**
```
cache_key = SHA256(input_checksum + config_hash + symbol + version)
```

**Properties:**
- Same input = same key (content-addressable)
- Automatic invalidation on config/input changes
- Snappy compression for Parquet files

---

### LineageTracker (620 lines)

Full audit trail for data transformations.

**TransformationType Enum:**
- INGESTION
- CLEANING
- RESAMPLING
- FEATURE_ENGINEERING
- LABELING
- SCALING
- SPLITTING

**Tracking:**
- source -> transformations -> output
- Graph queries: upstream/downstream dependencies

**Known Issue:** Lineage queries are O(n) - no indexing. Consider adding indexing for large lineage graphs.

---

### VersionManager (445 lines)

Semantic versioning for datasets.

**Version Bump Rules:**
- **Major bump:** Schema changes (columns added/removed)
- **Minor bump:** Config changes (backward compatible)
- **Patch bump:** Recomputation (same output)

---

## Data Adapters (4,771 lines)

Model-specific format conversions.

| Adapter | Shape | Models |
|---------|-------|--------|
| TabularAdapter | 2D `(N, F)` | XGBoost, LightGBM, CatBoost |
| SequenceAdapter | 3D `(N, T, F)` | LSTM, GRU, TCN |
| MultiStreamAdapter | 4D `(N, TF, T, F)` | PatchTST, iTransformer |

**AdapterFactory:** Unified entry point for all adapters.

---

## Key Issues Identified

1. **PIT Filtering:** Assumes specific datetime column names (may fail with non-standard schemas)
2. **Lineage Queries:** O(n) performance - no indexing
3. **Concurrent Access:** No concurrent access testing performed

---

## Phase 2 Recommendations

1. Add lineage indexing for large graphs
2. Test concurrent access patterns
3. Implement schema diff tracking
4. Add distributed cache option (S3/GCS)

---

**Last Updated:** 2026-01-23
