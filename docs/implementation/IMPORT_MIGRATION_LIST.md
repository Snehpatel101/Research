# Phase1 Import Migration List

**Generated:** 2026-01-21
**Total Imports Found:** 490
**Goal:** Migrate all `from src.phase1` imports to unified architecture

---

## Summary by Priority

| Category | Files | Import Statements | Priority |
|----------|-------|-------------------|----------|
| **CRITICAL: src/** | 35 | 73 | P0 - Must fix for pipeline |
| **SCRIPTS:** | 12 | 14 | P1 - CLI tools |
| **TESTS:** | 23 | ~150+ | P2 - Can update last |
| **INTERNAL: src/phase1/** | 50+ | 250+ | P3 - Internal refactoring |

---

## CRITICAL: src/ Directory Files (Must Fix for Pipeline)

### 1. `/Users/sneh/research/src/pipeline/runner.py` (13 imports)

**Current Imports:**
```python
from src.phase1.stages.clean.run import run_data_cleaning
from src.phase1.stages.datasets.run import run_build_datasets
from src.phase1.stages.features.run import run_feature_engineering
from src.phase1.stages.final_labels.run import run_final_labels
from src.phase1.stages.ga_optimize.run import run_ga_optimization
from src.phase1.stages.ingest.run import run_data_generation
from src.phase1.stages.labeling.run import run_initial_labeling
from src.phase1.stages.reporting.run import run_generate_report
from src.phase1.stages.scaled_validation.run import run_scaled_validation
from src.phase1.stages.scaling.run import run_feature_scaling
from src.phase1.stages.splits.run import run_create_splits
from src.phase1.stages.validation.run import run_validation
from src.phase1.lineage import PipelineLineage, create_dataset_checksum
```

**NEW Imports (Target Architecture):**
```python
from src.stages.clean import run_data_cleaning
from src.stages.datasets import run_build_datasets
from src.stages.features import run_feature_engineering
from src.stages.final_labels import run_final_labels
from src.stages.ga_optimize import run_ga_optimization
from src.stages.ingest import run_data_generation
from src.stages.labeling import run_initial_labeling
from src.stages.reporting import run_generate_report
from src.stages.scaled_validation import run_scaled_validation
from src.stages.scaling import run_feature_scaling
from src.stages.splits import run_create_splits
from src.stages.validation import run_validation
from src.core.lineage import PipelineLineage, create_dataset_checksum
```

---

### 2. `/Users/sneh/research/src/pipeline/__init__.py` (1 import)

**Current:**
```python
from src.phase1.pipeline_config import create_default_config
```

**NEW:**
```python
from src.config import create_default_config
```

---

### 3. `/Users/sneh/research/src/core/config.py` (1 import)

**Current:**
```python
from src.phase1.pipeline_config import PipelineConfig as Phase1PipelineConfig
```

**NEW:**
```python
from src.config import PipelineConfig as Phase1PipelineConfig
```

---

### 4. `/Users/sneh/research/src/core/container.py` (2 imports)

**Current:**
```python
from src.phase1.stages.datasets.sequences import SequenceDataset
from src.phase1.stages.datasets.adapters import MultiResolution4DAdapter
```

**NEW:**
```python
from src.adapters import SequenceDataset, MultiResolution4DAdapter
```

---

### 5. `/Users/sneh/research/src/ml_pipeline/unified.py` (3 imports)

**Current:**
```python
from src.phase1.pipeline_config import PipelineConfig  # 3 occurrences
```

**NEW:**
```python
from src.config import PipelineConfig
```

---

### 6. `/Users/sneh/research/src/config/pipeline/__init__.py` (8 imports)

**Current:**
```python
from src.phase1.config.barriers_config import (...)
from src.phase1.config.labeling_config import (...)
from src.phase1.config.labels import (...)
from src.phase1.config.feature_sets import (...)
from src.phase1.config.features import (...)
from src.phase1.config.regime_config import (...)
from src.phase1.config.runtime import (...)
from src.phase1.config.multi_model import (...)
```

**NEW:**
```python
from src.config.barriers import (...)
from src.config.labeling import (...)
from src.config.labels import (...)
from src.config.feature_sets import (...)
from src.config.features import (...)
from src.config.regime import (...)
from src.config.runtime import (...)
from src.config.multi_model import (...)
```

---

### 7. `/Users/sneh/research/src/config/unified.py` (1 import)

**Current:**
```python
from src.phase1.pipeline_config import PipelineConfig
```

**NEW:**
```python
from src.config import PipelineConfig
```

---

### 8. `/Users/sneh/research/src/optimization/labels.py` (1 import)

**Current:**
```python
from src.phase1.stages.labeling import TripleBarrierLabeler
```

**NEW:**
```python
from src.labeling import TripleBarrierLabeler
```

---

### 9. `/Users/sneh/research/src/training/modes/meta_labeling.py` (3 imports)

**Current:**
```python
from src.phase1.stages.datasets.container import TimeSeriesDataContainer
```

**NEW:**
```python
from src.core.container import TimeSeriesDataContainer
```

---

### 10. `/Users/sneh/research/src/training/modes/regime_aware.py` (4 imports)

**Current:**
```python
from src.phase1.stages.datasets.container import TimeSeriesDataContainer
```

**NEW:**
```python
from src.core.container import TimeSeriesDataContainer
```

---

### 11. `/Users/sneh/research/src/training/modes/walk_forward.py` (2 imports)

**Current:**
```python
from src.phase1.stages.datasets.container import TimeSeriesDataContainer
```

**NEW:**
```python
from src.core.container import TimeSeriesDataContainer
```

---

### 12. `/Users/sneh/research/src/training/unified_orchestrator.py` (3 imports)

**Current:**
```python
from src.phase1.stages.datasets import TimeSeriesDataContainer
```

**NEW:**
```python
from src.core import TimeSeriesDataContainer
```

---

### 13. `/Users/sneh/research/src/training/model_trainer.py` (1 import)

**Current:**
```python
from src.phase1.stages.datasets import TimeSeriesDataContainer
```

**NEW:**
```python
from src.core import TimeSeriesDataContainer
```

---

### 14. `/Users/sneh/research/src/training/orchestrator.py` (2 imports)

**Current:**
```python
from src.phase1.stages.datasets import TimeSeriesDataContainer
```

**NEW:**
```python
from src.core import TimeSeriesDataContainer
```

---

### 15. `/Users/sneh/research/src/training/regime_trainer.py` (2 imports)

**Current:**
```python
from src.phase1.stages.datasets import TimeSeriesDataContainer
```

**NEW:**
```python
from src.core import TimeSeriesDataContainer
```

---

### 16. `/Users/sneh/research/src/labeling/triple_barrier.py` (1 import)

**Current:**
```python
from src.phase1.config.barriers_config import get_tick_value, get_total_trade_cost
```

**NEW:**
```python
from src.config.barriers import get_tick_value, get_total_trade_cost
```

---

### 17. `/Users/sneh/research/src/features/strategy_manager.py` (2 imports)

**Current:**
```python
from src.phase1.utils.constants import METADATA_COLUMNS
from src.phase1.utils.feature_sets import _is_label_column
```

**NEW:**
```python
from src.core.constants import METADATA_COLUMNS
from src.features.utils import _is_label_column
```

---

### 18. `/Users/sneh/research/src/__init__.py` (1 import)

**Current:**
```python
from src.phase1.pipeline_config import create_default_config
```

**NEW:**
```python
from src.config import create_default_config
```

---

### 19. `/Users/sneh/research/src/utils/config_validator.py` (1 import - docstring)

**Current (docstring):**
```python
>>> from src.phase1.pipeline_config import PipelineConfig
```

**NEW:**
```python
>>> from src.config import PipelineConfig
```

---

### 20. `/Users/sneh/research/src/utils/colab_setup.py` (1 import)

**Current:**
```python
from src.phase1.stages.datasets.container import TimeSeriesDataContainer
```

**NEW:**
```python
from src.core.container import TimeSeriesDataContainer
```

---

### 21. `/Users/sneh/research/src/models/training/evaluation.py` (1 import)

**Current:**
```python
from src.phase1.stages.datasets.container import TimeSeriesDataContainer
```

**NEW:**
```python
from src.core.container import TimeSeriesDataContainer
```

---

### 22. `/Users/sneh/research/src/models/training/features.py` (3 imports)

**Current:**
```python
from src.phase1.stages.datasets.container import TimeSeriesDataContainer
from src.phase1.utils.constants import METADATA_COLUMNS
from src.phase1.utils.feature_sets import _is_label_column
```

**NEW:**
```python
from src.core.container import TimeSeriesDataContainer
from src.core.constants import METADATA_COLUMNS
from src.features.utils import _is_label_column
```

---

### 23. `/Users/sneh/research/src/models/training/trainer.py` (3 imports)

**Current:**
```python
from src.phase1.stages.datasets.container import TimeSeriesDataContainer
from src.phase1.lineage import PipelineLineage, validate_dataset_checksum
```

**NEW:**
```python
from src.core.container import TimeSeriesDataContainer
from src.core.lineage import PipelineLineage, validate_dataset_checksum
```

---

### 24. `/Users/sneh/research/src/models/regime_evaluation.py` (1 import)

**Current:**
```python
from src.phase1.stages.regime.trend import calculate_adx
```

**NEW:**
```python
from src.stages.regime.trend import calculate_adx
```

---

### 25. `/Users/sneh/research/src/models/meta_learner.py` (1 import)

**Current:**
```python
from src.phase1.stages.labeling.meta import MetaLabeler, BetSizeMethod
```

**NEW:**
```python
from src.labeling import MetaLabeler, BetSizeMethod
```

---

### 26. `/Users/sneh/research/src/models/training_utils.py` (1 import)

**Current:**
```python
from src.phase1.stages.datasets.container import TimeSeriesDataContainer
```

**NEW:**
```python
from src.core.container import TimeSeriesDataContainer
```

---

### 27. `/Users/sneh/research/src/models/trainer.py` (1 import - docstring)

**Current:**
```python
>>> from src.phase1.stages.datasets.container import TimeSeriesDataContainer
```

**NEW:**
```python
>>> from src.core.container import TimeSeriesDataContainer
```

---

### 28. `/Users/sneh/research/src/models/data_preparation.py` (1 import)

**Current:**
```python
from src.phase1.stages.datasets.container import TimeSeriesDataContainer
```

**NEW:**
```python
from src.core.container import TimeSeriesDataContainer
```

---

### 29. `/Users/sneh/research/src/cli/run_commands_core.py` (1 import)

**Current:**
```python
from src.phase1.config.runtime import detect_available_symbols
```

**NEW:**
```python
from src.config.runtime import detect_available_symbols
```

---

### 30. `/Users/sneh/research/src/inference/preprocessing_graph.py` (12 imports)

**Current:**
```python
from src.phase1.stages.clean.cleaner import DataCleaner
from src.phase1.stages.features.microstructure import (...)
from src.phase1.stages.features.momentum import (...)
from src.phase1.stages.features.moving_averages import add_ema, add_sma
from src.phase1.stages.features.price_features import (...)
from src.phase1.stages.features.regime import add_regime_features
from src.phase1.stages.features.temporal import add_temporal_features
from src.phase1.stages.features.trend import add_adx, add_supertrend
from src.phase1.stages.features.volatility import (...)
from src.phase1.stages.features.volume import (...)
from src.phase1.stages.features.wavelets import add_wavelet_features
from src.phase1.stages.mtf.generator import MTFFeatureGenerator
```

**NEW:**
```python
from src.stages.clean import DataCleaner
from src.features.compute.microstructure import (...)
from src.features.compute.momentum import (...)
from src.features.compute.moving_averages import add_ema, add_sma
from src.features.compute.price import (...)
from src.features.compute.regime import add_regime_features
from src.features.compute.temporal import add_temporal_features
from src.features.compute.trend import add_adx, add_supertrend
from src.features.compute.volatility import (...)
from src.features.compute.volume import (...)
from src.features.compute.wavelets import add_wavelet_features
from src.stages.mtf import MTFFeatureGenerator
```

---

### 31. `/Users/sneh/research/src/validation/__init__.py` (2 imports)

**Current:**
```python
from src.phase1.stages.meta_labeling import (...)
from src.phase1.stages.labeling import (...)
```

**NEW:**
```python
from src.labeling import (...)
from src.labeling import (...)
```

---

### 32. `/Users/sneh/research/src/feature_store/__init__.py` (1 import - docstring)

**Current:**
```python
>>> from src.phase1.stages.features import FeatureEngineer
```

**NEW:**
```python
>>> from src.features import FeatureEngineer
```

---

## SCRIPTS: scripts/ Directory Files

### 1. `/Users/sneh/research/scripts/train_ensemble.py`

**Current:**
```python
from src.phase1.stages.datasets.container import TimeSeriesDataContainer
```

**NEW:**
```python
from src.core.container import TimeSeriesDataContainer
```

---

### 2. `/Users/sneh/research/scripts/train_model.py`

**Current:**
```python
from src.phase1.stages.datasets.container import TimeSeriesDataContainer
```

**NEW:**
```python
from src.core.container import TimeSeriesDataContainer
```

---

### 3. `/Users/sneh/research/scripts/diagnose_model_performance.py`

**Current:**
```python
from src.phase1.stages.datasets.container import TimeSeriesDataContainer
```

**NEW:**
```python
from src.core.container import TimeSeriesDataContainer
```

---

### 4. `/Users/sneh/research/scripts/run_cv.py`

**Current:**
```python
from src.phase1.stages.datasets.container import TimeSeriesDataContainer
```

**NEW:**
```python
from src.core.container import TimeSeriesDataContainer
```

---

### 5. `/Users/sneh/research/scripts/test_feature_set_meta_learner.py`

**Current:**
```python
from src.phase1.config.feature_sets import (...)
from src.phase1.utils.feature_sets import resolve_feature_set
```

**NEW:**
```python
from src.config.feature_sets import (...)
from src.features.utils import resolve_feature_set
```

---

### 6. `/Users/sneh/research/scripts/train_regime_aware.py`

**Current:**
```python
from src.phase1.stages.datasets import TimeSeriesDataContainer
```

**NEW:**
```python
from src.core import TimeSeriesDataContainer
```

---

### 7. `/Users/sneh/research/scripts/run_walk_forward.py`

**Current:**
```python
from src.phase1.stages.datasets.container import TimeSeriesDataContainer
```

**NEW:**
```python
from src.core.container import TimeSeriesDataContainer
```

---

### 8. `/Users/sneh/research/scripts/refactor_notebook.py`

**Current:**
```python
from src.phase1.stages.validation.normalization import detect_outliers
from src.phase1.stages.validation import DataValidator
```

**NEW:**
```python
from src.validation import detect_outliers, DataValidator
```

---

### 9. `/Users/sneh/research/scripts/train_meta_labeling.py`

**Current:**
```python
from src.phase1.stages.datasets import TimeSeriesDataContainer
```

**NEW:**
```python
from src.core import TimeSeriesDataContainer
```

---

### 10. `/Users/sneh/research/scripts/run_cpcv_pbo.py`

**Current:**
```python
from src.phase1.stages.datasets.container import TimeSeriesDataContainer
```

**NEW:**
```python
from src.core.container import TimeSeriesDataContainer
```

---

## TESTS: tests/ Directory Files (Update Last)

### Key Test Files and Their Imports

| File | Import Type | Count |
|------|-------------|-------|
| `test_stages.py` | labeling, final_labels | 4 |
| `test_dynamic_horizons.py` | config, pipeline_config | 44 |
| `test_ga_bug_fixes.py` | ga_optimize, config | 2 |
| `test_ga_fixes_integration.py` | ga_optimize, labeling, config | 3 |
| `phase_1_tests/conftest.py` | pipeline_config | 1 |
| `phase_1_tests/test_feature_sets.py` | config.feature_sets, utils | 2 |
| `phase_1_tests/config/test_pipeline_config.py` | pipeline_config | 1 |
| `phase_1_tests/config/test_slippage.py` | config.barriers_config | 1 |
| `phase_1_tests/stages/test_indicator_scaling.py` | stages.features.scaling | 1 |
| `phase_1_tests/stages/ga_optimize/*.py` | ga_optimize | 10 |
| `phase_1_tests/stages/test_bar_builders.py` | stages.clean.bar_builders | 30+ |
| `phase_1_tests/stages/test_hmm_regime.py` | stages.regime | 20+ |
| `phase_1_tests/stages/test_stage2_mtf_resampling.py` | config, stages.clean, pipeline_config | 35+ |
| `phase_1_tests/stages/labeling/*.py` | stages.labeling | 20+ |
| `phase_1_tests/stages/test_sessions.py` | stages.sessions | 1 |
| `phase_1_tests/stages/test_stage3_feature_engineering_advanced.py` | stages.features | 2 |
| `phase_1_tests/stages/test_mtf_features.py` | stages.mtf | ~10 |
| `data_quality/test_ohlcv_validation.py` | stages.ingest.validators | 2 |
| `integration/test_cross_module.py` | stages.labeling | 1 |
| `integration/test_pipeline_fixes.py` | pipeline_config, config.regime_config | 2 |
| `integration/test_config_propagation.py` | pipeline_config, config | 15 |

---

## Import Migration Mapping Table

### Core Components

| Old Import Path | New Import Path |
|-----------------|-----------------|
| `src.phase1.pipeline_config.PipelineConfig` | `src.config.PipelineConfig` |
| `src.phase1.pipeline_config.create_default_config` | `src.config.create_default_config` |
| `src.phase1.pipeline_config.HorizonConfig` | `src.config.HorizonConfig` |
| `src.phase1.lineage.*` | `src.core.lineage.*` |

### Config Modules

| Old Import Path | New Import Path |
|-----------------|-----------------|
| `src.phase1.config.barriers_config.*` | `src.config.barriers.*` |
| `src.phase1.config.labeling_config.*` | `src.config.labeling.*` |
| `src.phase1.config.labels.*` | `src.config.labels.*` |
| `src.phase1.config.feature_sets.*` | `src.config.feature_sets.*` |
| `src.phase1.config.features.*` | `src.config.features.*` |
| `src.phase1.config.regime_config.*` | `src.config.regime.*` |
| `src.phase1.config.runtime.*` | `src.config.runtime.*` |
| `src.phase1.config.multi_model.*` | `src.config.multi_model.*` |

### Stages (Keep as stages but flatten)

| Old Import Path | New Import Path |
|-----------------|-----------------|
| `src.phase1.stages.clean.*` | `src.stages.clean.*` |
| `src.phase1.stages.datasets.*` | `src.core.*` (TimeSeriesDataContainer) |
| `src.phase1.stages.datasets.container.*` | `src.core.container.*` |
| `src.phase1.stages.datasets.sequences.*` | `src.adapters.sequences.*` |
| `src.phase1.stages.datasets.adapters.*` | `src.adapters.*` |
| `src.phase1.stages.features.*` | `src.features.compute.*` |
| `src.phase1.stages.final_labels.*` | `src.stages.final_labels.*` |
| `src.phase1.stages.ga_optimize.*` | `src.optimization.ga.*` |
| `src.phase1.stages.ingest.*` | `src.stages.ingest.*` |
| `src.phase1.stages.labeling.*` | `src.labeling.*` |
| `src.phase1.stages.meta_labeling.*` | `src.labeling.meta.*` |
| `src.phase1.stages.mtf.*` | `src.stages.mtf.*` |
| `src.phase1.stages.regime.*` | `src.stages.regime.*` |
| `src.phase1.stages.reporting.*` | `src.stages.reporting.*` |
| `src.phase1.stages.scaled_validation.*` | `src.stages.scaled_validation.*` |
| `src.phase1.stages.scaling.*` | `src.stages.scaling.*` |
| `src.phase1.stages.sessions.*` | `src.stages.sessions.*` |
| `src.phase1.stages.splits.*` | `src.stages.splits.*` |
| `src.phase1.stages.validation.*` | `src.validation.*` |

### Utils

| Old Import Path | New Import Path |
|-----------------|-----------------|
| `src.phase1.utils.constants.*` | `src.core.constants.*` |
| `src.phase1.utils.feature_sets.*` | `src.features.utils.*` |

---

## Migration Order

### Phase 1: Core Infrastructure (CRITICAL)
1. Move `src/phase1/pipeline_config.py` -> `src/config/pipeline_config.py`
2. Move `src/phase1/lineage.py` -> `src/core/lineage.py`
3. Move `src/phase1/config/` -> `src/config/` (merge)
4. Move `src/phase1/utils/` -> `src/core/` and `src/features/utils/`

### Phase 2: Stages Migration
1. Move `src/phase1/stages/datasets/container.py` -> `src/core/container.py` (already exists, ensure exports)
2. Move `src/phase1/stages/datasets/adapters/` -> `src/adapters/`
3. Move `src/phase1/stages/labeling/` -> `src/labeling/`
4. Move `src/phase1/stages/features/` -> `src/features/compute/`
5. Move remaining stages to `src/stages/`

### Phase 3: Update Imports
1. Update all CRITICAL src/ files
2. Update all scripts/
3. Update all tests/

### Phase 4: Cleanup
1. Remove `src/phase1/` directory
2. Update all documentation
3. Run full test suite

---

## Quick Grep Commands for Verification

```bash
# Count remaining phase1 imports after migration
grep -r "from src.phase1" --include="*.py" /Users/sneh/research/src | wc -l

# Find specific import patterns
grep -r "from src.phase1.stages.datasets" --include="*.py" /Users/sneh/research/src

# Verify new imports work
python -c "from src.core.container import TimeSeriesDataContainer; print('OK')"
```

---

## Notes

1. **Backward Compatibility**: Consider adding re-exports in `src/phase1/__init__.py` during transition
2. **Docstrings**: Several files have example imports in docstrings that also need updating
3. **Internal phase1 imports**: ~250+ imports within src/phase1/ itself need internal path updates
4. **Test restructuring**: Consider renaming `tests/phase_1_tests/` to `tests/stages/` or similar
