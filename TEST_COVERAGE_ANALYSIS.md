# Test Coverage Analysis and Recommendations

**Project:** /Users/sneh/research
**Analysis Date:** 2026-01-20
**Analyzed by:** Claude Code (Test Automation Expert)

---

## Executive Summary

### Coverage Statistics
- **Total Source Files:** 346 Python modules
- **Total Test Files:** 153 test modules
- **Files with Tests:** 168 (48.6%)
- **Files without Tests:** 178 (51.4%)
- **Total Test Functions:** ~7,651 assertions
- **Mock Usage:** 347 instances

### Overall Test Quality: **MODERATE** (6/10)

**Strengths:**
- Well-organized fixture architecture with minimal duplication
- Comprehensive integration test suite
- Good separation of unit vs integration tests
- Strong model testing coverage
- Proper use of pytest markers

**Critical Issues:**
- 51.4% of source code lacks corresponding tests
- Missing tests for critical infrastructure (CLI, pipeline orchestration, monitoring)
- Incomplete fixture coverage leads to scattered data generation
- Analysis scripts (not actual tests) mixed in test directory
- Over-reliance on mocking in some areas, under-mocking in others

---

## 1. Missing Tests Analysis

### 1.1 Critical Missing Test Coverage

#### **HIGH PRIORITY** - Core Infrastructure (0% coverage)

**CLI Module** (7 files, 0 tests)
```
src/cli/
├── unified_cli.py          # Main CLI entry point - NO TESTS
├── run_commands_core.py    # Core pipeline commands - NO TESTS
├── run_commands_pipeline.py # Pipeline orchestration - NO TESTS
├── status_commands.py      # Status reporting - NO TESTS
├── preset_commands.py      # Preset configurations - NO TESTS
├── run_commands_info.py    # Info commands - NO TESTS
└── utils.py                # CLI utilities - NO TESTS
```

**Impact:** The entire command-line interface has no test coverage. This is a major risk area.

**Monitoring Module** (5 files, 1 test)
```
src/monitoring/
├── alert_handler.py         # Alert system - NO TESTS
├── drift_detector.py        # Core drift detection - NO TESTS
├── drift_detectors.py       # Drift detector implementations - NO TESTS
├── drift_types.py           # Drift type definitions - NO TESTS
├── feature_drift_monitor.py # Feature monitoring - NO TESTS
└── tests/monitoring/test_drift.py # ONLY 1 test file exists
```

**Impact:** Production monitoring system is essentially untested.

**Pipeline Orchestration** (3 files, 0 tests)
```
src/coordination/
├── alignment.py             # Data alignment - NO TESTS
└── timeframe_coordinator.py # Multi-timeframe coordination - NO TESTS

src/ml_pipeline/
├── phase_registry.py        # Phase registration - NO TESTS
└── state_validator.py       # Pipeline state validation - NO TESTS
```

**Impact:** Core pipeline coordination logic lacks test coverage.

#### **MEDIUM PRIORITY** - Business Logic (Partial coverage)

**Cross-Validation Module** (12 files, 5 tests)
```
Missing tests:
- cv_orchestrator.py         # Main CV orchestrator
- cv_stacking.py            # Stacking CV logic
- cv_tuner.py               # Hyperparameter tuning
- oof_alignment.py          # OOF alignment
- oof_core.py               # Core OOF generation
- oof_io.py                 # OOF I/O operations
- oof_sequence.py           # Sequence-based OOF
- oof_stacking.py           # Stacking OOF
- oof_validation.py         # OOF validation
- param_spaces.py           # Parameter space definitions
- cv_feature_selection.py   # CV-based feature selection
- cv_dataclasses.py         # CV data structures
```

**Impact:** CV infrastructure is partially tested but key orchestration missing.

**Feature Engineering** (13 files, 0 tests)
```
src/features/compute/
├── microstructure.py        # Microstructure features - NO TESTS
├── momentum.py             # Momentum indicators - NO TESTS
├── moving_average.py       # Moving averages - NO TESTS
├── price.py                # Price features - NO TESTS
├── raw.py                  # Raw feature extraction - NO TESTS
├── temporal.py             # Temporal features - NO TESTS
├── trend.py                # Trend indicators - NO TESTS
├── volatility.py           # Volatility features - NO TESTS
├── volume.py               # Volume features - NO TESTS
└── wavelets.py             # Wavelet transforms - NO TESTS

src/features/
├── pruning.py              # Feature pruning - NO TESTS
├── strategies.py           # Feature strategies - NO TESTS
└── strategy_manager.py     # Strategy management - NO TESTS
```

**Impact:** Feature engineering has no unit tests, relying only on integration tests.

**Backtesting Module** (3 files, 2 tests)
```
Missing tests:
- costs.py                   # Transaction costs - NO TESTS
- equity_curve.py            # Equity curve calculation - NO TESTS
- position_sizing.py         # Position sizing - NO TESTS
```

**Impact:** Financial calculations lack dedicated unit tests.

**Inference Module** (6 files, 2 tests)
```
Missing tests:
- batch.py                   # Batch inference - NO TESTS
- bundle.py                  # Model bundling - NO TESTS
- ensemble_bundle.py         # Ensemble bundling - NO TESTS
- orchestrator.py            # Inference orchestration - NO TESTS
- preprocessing_graph.py     # Preprocessing pipeline - NO TESTS
- server.py                  # Inference server - NO TESTS
```

**Impact:** Production inference system lacks coverage.

**Training Orchestration** (9 files, 1 test)
```
Missing tests:
- config_loader.py           # Config loading - NO TESTS
- model_factory.py          # Model creation - NO TESTS
- model_trainer.py          # Training logic - NO TESTS
- orchestrator.py           # Training orchestration - NO TESTS
- regime_detector.py        # Regime detection - NO TESTS
- regime_trainer.py         # Regime-aware training - NO TESTS
- unified_orchestrator.py   # Unified training - NO TESTS
- modes/meta_labeling.py    # Meta-labeling mode - NO TESTS
- modes/regime_aware.py     # Regime-aware mode - NO TESTS
```

**Impact:** Training orchestration infrastructure untested.

#### **LOW PRIORITY** - Model Implementations (Good coverage overall)

**Models Module** (48 untested files, but high test count)

The models module has extensive test coverage for core functionality, but specific implementations lack dedicated tests:
- Individual neural network architectures (ResNet1D, InceptionTime, etc.)
- Model configuration utilities
- Calibration implementations
- Some ensemble meta-learners

**Note:** This is acceptable as models are tested through integration tests.

### 1.2 Complete List of Untested Modules by Category

<details>
<summary><strong>Adapters (3 files)</strong></summary>

- `adapters/multi_stream.py` - Multi-stream data handling
- `adapters/preparation.py` - Data preparation adapters
- `adapters/tabular.py` - Tabular data adapter
</details>

<details>
<summary><strong>Common Utilities (2 files)</strong></summary>

- `common/horizon_config.py` - Horizon configuration
- `common/split_ratios.py` - Data split ratio management
</details>

<details>
<summary><strong>Core Types (3 files)</strong></summary>

- `core/defaults.py` - Default configurations
- `core/interfaces.py` - Interface definitions
- `core/types.py` - Type definitions
</details>

<details>
<summary><strong>Evaluation (3 files)</strong></summary>

- `evaluation/cpcv_pbo_evaluator.py` - CPCV-PBO evaluation
- `evaluation/cv_evaluator.py` - CV evaluation
- `evaluation/walk_forward_evaluator.py` - Walk-forward evaluation
</details>

<details>
<summary><strong>Feature Selection (4 files)</strong></summary>

- `feature_selection/filtering.py` - Feature filtering
- `feature_selection/manager.py` - Feature selection manager
- `feature_selection/priority.py` - Feature priority
- `feature_selection/result.py` - Selection results
</details>

<details>
<summary><strong>Phase 1 (49 files)</strong></summary>

Major untested areas in Phase 1:
- All bar builders (dollar_bars, time_bars, volume_bars)
- Feature engineering stages
- Labeling implementations
- Meta-labeling
- MTF (multi-timeframe) generation
- Regime detection
- Session filtering
- Validation stages
</details>

<details>
<summary><strong>Utils (4 files)</strong></summary>

- `utils/checkpoint_manager.py` - Checkpoint management
- `utils/colab_setup.py` - Google Colab setup
- `utils/config_validator.py` - Config validation
- `utils/memory.py` - Memory utilities
</details>

---

## 2. Test Quality Issues

### 2.1 Non-Test Files in Test Directory

**Issue:** Analysis scripts mixed with tests
```
tests/analysis/
├── barrier_analysis.py        # Analysis script, NOT a test
└── barrier_visualization.py   # Visualization script, NOT a test
```

**Recommendation:** Move to `scripts/analysis/` or `notebooks/`

**Issue:** Validation scripts that should be tests
```
tests/phase4_validation.py     # Should be test_phase4_validation.py
tests/validate_phase2_adapters.py # Should be test_phase2_adapters.py
tests/verify_modules.py        # Should be test_module_imports.py
tests/verify_regime_implementation.py # Should be test_regime_implementation.py
```

**Recommendation:** Rename to follow `test_*.py` convention or move to scripts.

### 2.2 Fixture Inconsistencies

**Duplicate Fixtures Across Modules**

Found 2 duplicate fixture definitions that could cause confusion:

1. **`mock_data_container`** - Defined in:
   - `tests/integration/conftest.py`
   - `tests/models/conftest.py`

2. **`tmp_output_dir`** - Defined in:
   - `tests/integration/conftest.py`
   - `tests/models/conftest.py`
   - `tests/error_handling/conftest.py`

**Impact:** MODERATE - These duplicates provide slightly different implementations, which could lead to test inconsistencies.

**Recommendation:**
```python
# Create tests/conftest.py (root-level)
@pytest.fixture(scope="session")
def mock_data_container():
    """Shared mock data container for all tests."""
    return create_mock_container(n_train=200, n_val=50, n_features=20)

@pytest.fixture
def tmp_output_dir(tmp_path):
    """Shared temporary output directory."""
    output_dir = tmp_path / "experiments" / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
```

**Fixture Organization Quality: GOOD**

The project uses a well-structured fixture hierarchy:
- Root `conftest.py` for project-wide fixtures
- Module-specific `conftest.py` for domain fixtures
- 70 unique fixtures across 8 conftest files
- Minimal duplication (only 2 duplicates)

### 2.3 Mocking Issues

**Over-Mocking:** Some tests rely too heavily on mocks

Example from `tests/models/conftest.py`:
```python
def create_mock_container(...):
    """Creates extensive mock data container."""
    # 100+ lines of mock setup
    # Better: Use lightweight real data for unit tests
```

**Recommendation:** Use real lightweight data for unit tests, mocks for integration tests.

**Under-Mocking:** Some tests don't mock external dependencies

Example pattern found in integration tests:
```python
def test_full_pipeline(mock_data_container):
    trainer = Trainer(config)
    results = trainer.run(mock_data_container)  # Good: mocked data
    # Missing: Should mock file I/O, model registry, etc.
```

**Recommendation:**
- Mock all external I/O in unit tests
- Use real implementations in integration tests
- Create clear separation with pytest markers

### 2.4 Integration vs Unit Test Separation

**Current Separation: GOOD**

The project uses pytest markers:
```ini
# pytest.ini
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
    unit: marks tests as unit tests
```

**Issue:** Not all tests are properly marked.

**Test Distribution Analysis:**
```
tests/
├── unit/                    # Only 1 subdirectory (config tests)
├── integration/             # Well-organized integration tests
├── models/                  # Mix of unit and integration (not marked)
├── cross_validation/        # Mix of unit and integration (not marked)
└── phase_1_tests/           # Mix of unit and integration (not marked)
```

**Recommendations:**

1. **Mark all tests appropriately:**
```python
@pytest.mark.unit
def test_feature_calculation():
    """Unit test for feature calculation logic."""
    pass

@pytest.mark.integration
def test_full_training_pipeline():
    """Integration test for complete training flow."""
    pass

@pytest.mark.slow
@pytest.mark.integration
def test_multi_symbol_backtest():
    """Slow integration test for backtesting."""
    pass
```

2. **Run tests by category:**
```bash
# Fast unit tests only
pytest -m "unit and not slow"

# Integration tests
pytest -m integration

# Everything except slow tests
pytest -m "not slow"
```

---

## 3. Specific Testing Recommendations

### 3.1 HIGH PRIORITY: CLI Testing

**Problem:** Entire CLI has zero test coverage.

**Solution:** Implement CLI testing with CliRunner pattern

```python
# tests/cli/test_unified_cli.py

import pytest
from click.testing import CliRunner
from src.cli.unified_cli import cli

class TestCLICommands:
    """Test CLI command execution."""

    def test_help_command(self):
        """CLI should display help message."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'Usage:' in result.output

    def test_status_command(self):
        """Status command should display system status."""
        runner = CliRunner()
        result = runner.invoke(cli, ['status'])
        assert result.exit_code == 0
        # Add specific assertions

    @pytest.mark.integration
    def test_pipeline_run_command(self, tmp_path):
        """Pipeline run should execute successfully."""
        runner = CliRunner()
        config_path = tmp_path / "test_config.yaml"
        # Create test config
        result = runner.invoke(cli, ['pipeline', 'run', str(config_path)])
        assert result.exit_code == 0
```

**Estimated Effort:** 2-3 days
**Impact:** Critical - validates user-facing interface

### 3.2 HIGH PRIORITY: Monitoring System Testing

**Problem:** Production monitoring has minimal test coverage.

**Solution:** Implement comprehensive monitoring tests

```python
# tests/monitoring/test_drift_detection.py

import pytest
import numpy as np
import pandas as pd
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.drift_types import DriftType

class TestDriftDetector:
    """Test drift detection algorithms."""

    def test_detect_no_drift(self):
        """Should not detect drift in stable distribution."""
        reference_data = np.random.normal(0, 1, 1000)
        current_data = np.random.normal(0, 1, 1000)

        detector = DriftDetector()
        result = detector.detect(reference_data, current_data)

        assert result.drift_detected is False
        assert result.drift_type is None

    def test_detect_mean_shift(self):
        """Should detect mean shift drift."""
        reference_data = np.random.normal(0, 1, 1000)
        current_data = np.random.normal(2, 1, 1000)  # Mean shift

        detector = DriftDetector()
        result = detector.detect(reference_data, current_data)

        assert result.drift_detected is True
        assert result.drift_type == DriftType.MEAN_SHIFT

    def test_detect_variance_drift(self):
        """Should detect variance change."""
        reference_data = np.random.normal(0, 1, 1000)
        current_data = np.random.normal(0, 5, 1000)  # Variance increase

        detector = DriftDetector()
        result = detector.detect(reference_data, current_data)

        assert result.drift_detected is True
        assert result.drift_type == DriftType.VARIANCE_SHIFT

class TestFeatureDriftMonitor:
    """Test feature-level drift monitoring."""

    def test_monitor_multiple_features(self):
        """Should monitor drift across multiple features."""
        # Implementation
        pass

    def test_alert_generation(self):
        """Should generate alerts when drift detected."""
        # Implementation
        pass
```

**Estimated Effort:** 3-4 days
**Impact:** High - ensures production monitoring works correctly

### 3.3 MEDIUM PRIORITY: Feature Engineering Testing

**Problem:** Feature computation has no dedicated unit tests.

**Solution:** Property-based testing for feature calculations

```python
# tests/features/test_momentum_features.py

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st
from src.features.compute.momentum import calculate_rsi, calculate_macd

class TestMomentumFeatures:
    """Test momentum indicator calculations."""

    @given(
        prices=st.lists(st.floats(min_value=1, max_value=10000), min_size=50, max_size=500),
        period=st.integers(min_value=5, max_value=30)
    )
    def test_rsi_bounds(self, prices, period):
        """RSI should always be between 0 and 100."""
        df = pd.DataFrame({'close': prices})
        rsi = calculate_rsi(df, period=period)

        # Remove NaN values from warmup
        rsi_valid = rsi.dropna()

        assert (rsi_valid >= 0).all(), "RSI below 0"
        assert (rsi_valid <= 100).all(), "RSI above 100"

    def test_rsi_extreme_values(self):
        """RSI should reach extremes with constant price movement."""
        # All up moves -> RSI = 100
        prices_up = list(range(1, 101))
        df_up = pd.DataFrame({'close': prices_up})
        rsi_up = calculate_rsi(df_up, period=14)
        assert rsi_up.iloc[-1] == 100.0

        # All down moves -> RSI = 0
        prices_down = list(range(100, 0, -1))
        df_down = pd.DataFrame({'close': prices_down})
        rsi_down = calculate_rsi(df_down, period=14)
        assert rsi_down.iloc[-1] == 0.0

    def test_macd_crossover(self):
        """MACD should detect trend changes."""
        # Create uptrend then downtrend
        prices = [100] * 20 + list(range(100, 150, 2)) + list(range(150, 100, -2))
        df = pd.DataFrame({'close': prices})

        macd, signal = calculate_macd(df)
        crossovers = (macd > signal).astype(int).diff()

        # Should have at least one bullish and one bearish crossover
        assert (crossovers == 1).any(), "No bullish crossover"
        assert (crossovers == -1).any(), "No bearish crossover"
```

**Use Property-Based Testing:**
- Hypothesis library for edge cases
- Test invariants (bounds, monotonicity, etc.)
- Generate random but valid inputs

**Estimated Effort:** 5-7 days for all feature modules
**Impact:** Medium - catches calculation errors early

### 3.4 MEDIUM PRIORITY: Cross-Validation Testing

**Problem:** CV orchestration lacks comprehensive tests.

**Solution:** Test CV components in isolation and integration

```python
# tests/cross_validation/test_cv_orchestrator.py

import pytest
from src.cross_validation.cv_orchestrator import CVOrchestrator
from src.cross_validation.purged_kfold import PurgedKFoldConfig

class TestCVOrchestrator:
    """Test cross-validation orchestration."""

    def test_orchestrator_initialization(self):
        """Should initialize with valid configuration."""
        config = PurgedKFoldConfig(n_splits=5)
        orchestrator = CVOrchestrator(config)
        assert orchestrator.config.n_splits == 5

    def test_orchestrate_single_model(self, time_series_data, mock_model):
        """Should orchestrate CV for single model."""
        config = PurgedKFoldConfig(n_splits=3)
        orchestrator = CVOrchestrator(config)

        results = orchestrator.run(
            model=mock_model,
            X=time_series_data['X'],
            y=time_series_data['y']
        )

        assert len(results.fold_results) == 3
        assert results.mean_score > 0
        assert results.std_score >= 0

    def test_orchestrate_with_feature_selection(self, time_series_data):
        """Should integrate feature selection in CV."""
        # Implementation
        pass

    @pytest.mark.integration
    def test_orchestrate_ensemble(self, time_series_data):
        """Should orchestrate ensemble CV with stacking."""
        # Implementation
        pass
```

**Estimated Effort:** 3-4 days
**Impact:** Medium - ensures CV reliability

### 3.5 LOW PRIORITY: Backtesting Testing

**Problem:** Financial calculations lack dedicated tests.

**Solution:** Add tests with known outcomes

```python
# tests/backtesting/test_costs.py

import pytest
import pandas as pd
from src.backtesting.costs import calculate_slippage, calculate_commission

class TestTransactionCosts:
    """Test transaction cost calculations."""

    def test_fixed_slippage(self):
        """Fixed slippage should add constant cost per trade."""
        trades = pd.DataFrame({
            'quantity': [100, -50, 200],
            'price': [100.0, 101.0, 99.0]
        })

        slippage = calculate_slippage(trades, slippage_bps=10)

        # 10 bps = 0.1% per trade
        expected = [100 * 100.0 * 0.001, 50 * 101.0 * 0.001, 200 * 99.0 * 0.001]
        assert slippage.tolist() == pytest.approx(expected)

    def test_commission_calculation(self):
        """Commission should be calculated per share."""
        trades = pd.DataFrame({
            'quantity': [100, -50],
            'price': [100.0, 101.0]
        })

        commission = calculate_commission(trades, commission_per_share=0.01)

        expected = [100 * 0.01, 50 * 0.01]
        assert commission.tolist() == pytest.approx(expected)
```

**Estimated Effort:** 2-3 days
**Impact:** Low-Medium - validates critical financial logic

---

## 4. Testing Best Practices

### 4.1 Test Organization

**Current Structure:** Mixed patterns
**Recommended Structure:**

```
tests/
├── conftest.py                 # Root fixtures
├── unit/                       # Fast, isolated unit tests
│   ├── conftest.py
│   ├── features/
│   │   ├── test_momentum.py
│   │   ├── test_volatility.py
│   │   └── test_volume.py
│   ├── models/
│   │   ├── test_xgboost_model.py
│   │   ├── test_lstm_model.py
│   │   └── test_ensemble.py
│   ├── backtesting/
│   │   ├── test_costs.py
│   │   ├── test_metrics.py
│   │   └── test_position_sizing.py
│   └── validation/
│       ├── test_leakage_detection.py
│       └── test_statistical_tests.py
├── integration/                # Slower, multi-component tests
│   ├── conftest.py
│   ├── test_full_pipeline.py
│   ├── test_cv_pipeline.py
│   └── test_inference_pipeline.py
├── functional/                 # End-to-end functional tests
│   ├── test_cli_workflows.py
│   └── test_training_workflows.py
└── performance/                # Performance/benchmark tests
    └── test_model_speed.py
```

### 4.2 Test Naming Conventions

**Follow AAA Pattern:** Arrange, Act, Assert

```python
def test_feature_calculation_with_missing_values():
    """Feature calculation should handle missing values gracefully.

    Tests that:
    - Missing values are properly handled
    - No warnings are raised
    - Output has expected shape
    """
    # Arrange
    data = pd.DataFrame({
        'close': [100, np.nan, 102, 103],
        'volume': [1000, 2000, np.nan, 3000]
    })

    # Act
    features = calculate_features(data)

    # Assert
    assert features is not None
    assert len(features) == len(data)
    assert not features.isna().all().any()  # No all-NaN columns
```

**Test Naming:**
- `test_<what>_<condition>_<expected>`
- Be specific and descriptive
- Include edge cases in name

### 4.3 Fixture Best Practices

**Scope Appropriately:**
```python
@pytest.fixture(scope="session")  # Expensive, shared across all tests
def large_dataset():
    return load_large_dataset()

@pytest.fixture(scope="module")  # Shared within module
def trained_model():
    return train_model()

@pytest.fixture(scope="function")  # Default, new for each test
def temp_data():
    return generate_temp_data()
```

**Use Factories for Flexibility:**
```python
@pytest.fixture
def data_factory():
    """Factory to create data with different parameters."""
    def _factory(n_samples=1000, n_features=10, seed=42):
        np.random.seed(seed)
        return pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'f{i}' for i in range(n_features)]
        )
    return _factory

def test_with_small_data(data_factory):
    data = data_factory(n_samples=100, n_features=5)
    # Test with small data

def test_with_large_data(data_factory):
    data = data_factory(n_samples=10000, n_features=50)
    # Test with large data
```

### 4.4 Assertion Best Practices

**Use Specific Assertions:**
```python
# Bad
assert result == True

# Good
assert result is True

# Bad
assert len(predictions) > 0

# Good
assert len(predictions) == expected_length

# Use pytest.approx for floats
assert result == pytest.approx(expected, rel=1e-5)

# Use meaningful messages
assert predictions.shape == (100, 3), \
    f"Expected shape (100, 3), got {predictions.shape}"
```

**Test Error Conditions:**
```python
def test_invalid_input_raises_error():
    """Should raise ValueError for negative horizon."""
    with pytest.raises(ValueError, match="horizon must be positive"):
        create_model(horizon=-1)

def test_missing_data_raises_error():
    """Should raise KeyError for missing required column."""
    df = pd.DataFrame({'close': [1, 2, 3]})  # Missing 'open'
    with pytest.raises(KeyError, match="open"):
        calculate_features(df)
```

### 4.5 Parametrized Testing

**Use Parametrize for Multiple Cases:**
```python
@pytest.mark.parametrize("horizon,expected_length", [
    (5, 995),
    (10, 990),
    (20, 980),
])
def test_label_generation_different_horizons(horizon, expected_length):
    """Labels should account for horizon in length."""
    data = generate_data(1000)
    labels = generate_labels(data, horizon=horizon)
    assert len(labels) == expected_length

@pytest.mark.parametrize("model_name,expected_family", [
    ("xgboost", "boosting"),
    ("lightgbm", "boosting"),
    ("lstm", "neural"),
    ("gru", "neural"),
])
def test_model_family_detection(model_name, expected_family):
    """Models should be correctly categorized by family."""
    model = create_model(model_name)
    assert model.family == expected_family
```

---

## 5. Testing Strategy Recommendations

### 5.1 Test-Driven Development (TDD) for New Features

**When adding new features:**

1. **Write failing test first:**
```python
def test_new_feature_calculation():
    """New feature should calculate XYZ metric."""
    data = create_test_data()
    result = calculate_new_feature(data)
    assert result.shape == expected_shape
    assert result['new_metric'].mean() > 0
```

2. **Implement minimal code to pass:**
```python
def calculate_new_feature(data):
    # Minimal implementation
    return pd.DataFrame({'new_metric': [1, 2, 3]})
```

3. **Refactor with confidence:**
```python
def calculate_new_feature(data):
    # Optimized implementation
    return efficient_calculation(data)
```

### 5.2 Continuous Integration Testing

**Current Setup:** Basic pytest in CI (inferred from .github/)

**Recommended CI Pipeline:**

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.10, 3.11, 3.12]

    steps:
      - uses: actions/checkout@v2
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v2
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-xdist

      - name: Run unit tests (fast)
        run: pytest -m "unit and not slow" -n auto

      - name: Run integration tests
        run: pytest -m "integration and not slow"

      - name: Generate coverage report
        run: pytest --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

### 5.3 Test Coverage Goals

**Recommended Coverage Targets:**

| Category | Current | Target | Priority |
|----------|---------|--------|----------|
| CLI | 0% | 80% | HIGH |
| Monitoring | ~5% | 85% | HIGH |
| Pipeline Orchestration | ~10% | 75% | HIGH |
| Cross-Validation | ~40% | 80% | MEDIUM |
| Feature Engineering | ~0% | 70% | MEDIUM |
| Models (core) | ~70% | 85% | MEDIUM |
| Backtesting | ~30% | 75% | MEDIUM |
| Inference | ~20% | 75% | MEDIUM |
| Utilities | ~45% | 70% | LOW |
| **Overall** | **48%** | **75%** | - |

### 5.4 Testing Tools Recommendations

**Add to requirements-dev.txt:**
```txt
# Testing
pytest>=7.0
pytest-cov>=3.0
pytest-xdist>=2.5  # Parallel test execution
pytest-mock>=3.6
pytest-timeout>=2.1
hypothesis>=6.50  # Property-based testing

# Code Quality
black>=22.0
ruff>=0.0.250
mypy>=1.0

# Coverage
coverage[toml]>=6.0
```

**Add pyproject.toml configuration:**
```toml
[tool.pytest.ini_options]
minversion = "7.0"
addopts = [
    "-ra",
    "-q",
    "--strict-markers",
    "--strict-config",
    "--cov=src",
    "--cov-report=html",
    "--cov-report=term-missing",
]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]

markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
    "requires_gpu: marks tests requiring GPU",
]

[tool.coverage.run]
source = ["src"]
omit = [
    "*/tests/*",
    "*/__init__.py",
    "*/conftest.py",
]

[tool.coverage.report]
precision = 2
show_missing = true
skip_covered = false
```

---

## 6. Implementation Roadmap

### Phase 1: Critical Infrastructure (Weeks 1-2)

**Week 1: CLI Testing**
- [ ] Set up CLI test infrastructure
- [ ] Test all command entry points
- [ ] Test error handling in CLI
- [ ] Test configuration loading via CLI

**Week 2: Monitoring & Validation**
- [ ] Test drift detection algorithms
- [ ] Test alert system
- [ ] Test pipeline state validation
- [ ] Test data contract validation

**Deliverable:** 80% coverage for CLI and monitoring modules

### Phase 2: Core Business Logic (Weeks 3-5)

**Week 3: Cross-Validation**
- [ ] Test CV orchestrator
- [ ] Test OOF generation
- [ ] Test stacking CV
- [ ] Integration tests for full CV pipeline

**Week 4: Feature Engineering**
- [ ] Property-based tests for all indicators
- [ ] Test edge cases (NaN, inf, zeros)
- [ ] Test feature pruning
- [ ] Test feature selection

**Week 5: Backtesting**
- [ ] Test cost calculations
- [ ] Test equity curve generation
- [ ] Test position sizing
- [ ] Integration tests with actual trades

**Deliverable:** 75% coverage for CV, features, and backtesting

### Phase 3: Production Systems (Weeks 6-7)

**Week 6: Inference**
- [ ] Test batch inference
- [ ] Test model bundling
- [ ] Test preprocessing pipeline
- [ ] Performance tests

**Week 7: Training Orchestration**
- [ ] Test training modes
- [ ] Test regime-aware training
- [ ] Test config loading
- [ ] Integration tests

**Deliverable:** 70% coverage for inference and training

### Phase 4: Cleanup & Documentation (Week 8)

- [ ] Fix all fixture inconsistencies
- [ ] Add missing pytest markers
- [ ] Document testing guidelines
- [ ] Set up pre-commit hooks
- [ ] Configure CI/CD pipeline

**Deliverable:** Complete testing documentation and CI setup

---

## 7. Quick Wins (Can be done immediately)

### Fix 1: Rename Non-Test Files (5 minutes)
```bash
mv tests/analysis scripts/analysis
mv tests/phase4_validation.py tests/test_phase4_validation.py
mv tests/validate_phase2_adapters.py scripts/validate_phase2_adapters.py
mv tests/verify_modules.py scripts/verify_modules.py
```

### Fix 2: Consolidate Duplicate Fixtures (15 minutes)
```python
# tests/conftest.py
@pytest.fixture
def mock_data_container():
    """Shared mock data container."""
    from tests.models.conftest import create_mock_container
    return create_mock_container(n_train=200, n_val=50, n_features=20)

@pytest.fixture
def tmp_output_dir(tmp_path):
    """Shared temporary output directory."""
    output_dir = tmp_path / "experiments" / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
```

### Fix 3: Add Missing Markers (30 minutes)
```python
# Mark existing tests in tests/models/
@pytest.mark.unit
def test_xgboost_initialization():
    pass

@pytest.mark.integration
def test_full_training_pipeline():
    pass
```

### Fix 4: Add Pre-commit Hooks (10 minutes)
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black

  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.0.250
    hooks:
      - id: ruff

  - repo: local
    hooks:
      - id: pytest-unit
        name: pytest-unit
        entry: pytest -m "unit and not slow"
        language: system
        pass_filenames: false
        always_run: true
```

---

## 8. Summary and Next Steps

### Current State
- **48.6% test coverage** - Below industry standard (70-80%)
- **Well-organized fixture architecture** with minimal duplication
- **Good separation** of integration vs unit tests (with markers)
- **Critical gaps** in CLI, monitoring, and pipeline orchestration

### Immediate Actions (This Week)
1. Move analysis scripts out of tests/ directory
2. Consolidate duplicate fixtures
3. Add pytest markers to unmarked tests
4. Set up pre-commit hooks

### Short-term Goals (Next 2 Weeks)
1. Implement CLI testing (80% coverage)
2. Implement monitoring tests (85% coverage)
3. Add cross-validation orchestration tests (80% coverage)

### Long-term Goals (Next 2 Months)
1. Achieve 75% overall test coverage
2. Implement property-based testing for all feature calculations
3. Set up continuous integration with coverage reporting
4. Document testing best practices for the team

### Success Metrics
- [ ] 75% overall test coverage
- [ ] 100% coverage for critical paths (CLI, monitoring, validation)
- [ ] All tests marked with appropriate pytest markers
- [ ] CI/CD pipeline running all tests on every commit
- [ ] Test execution time < 5 minutes for unit tests
- [ ] Documentation of testing practices

---

## Appendix A: Testing Checklist Template

Use this checklist when adding new functionality:

### New Feature Testing Checklist

- [ ] **Unit Tests**
  - [ ] Test happy path
  - [ ] Test edge cases (empty, null, zero)
  - [ ] Test error conditions
  - [ ] Test boundary values
  - [ ] Property-based tests (if applicable)

- [ ] **Integration Tests**
  - [ ] Test interaction with other components
  - [ ] Test with real (not mocked) dependencies
  - [ ] Test configuration variations

- [ ] **Documentation**
  - [ ] Docstring explains what is tested
  - [ ] Complex tests have inline comments
  - [ ] Test name describes scenario

- [ ] **Code Quality**
  - [ ] Tests are independent (can run in any order)
  - [ ] No hardcoded paths or credentials
  - [ ] Proper use of fixtures
  - [ ] Appropriate pytest markers

---

## Appendix B: Example Test Suite

Complete example of well-structured test module:

```python
"""
Tests for RSI (Relative Strength Index) calculation.

Covers:
- Correct RSI calculation
- Boundary conditions (all gains, all losses)
- Edge cases (insufficient data, NaN values)
- Performance benchmarks
"""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st

from src.features.compute.momentum import calculate_rsi


# =============================================================================
# UNIT TESTS
# =============================================================================

@pytest.mark.unit
class TestRSICalculation:
    """Unit tests for RSI calculation logic."""

    def test_rsi_basic_calculation(self):
        """RSI should calculate correctly for known sequence."""
        # Known sequence with predictable RSI
        prices = [44, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84,
                  46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03,
                  46.41, 46.22, 45.64]
        df = pd.DataFrame({'close': prices})

        rsi = calculate_rsi(df, period=14)

        # Last RSI value should be approximately 51.78 (from textbook)
        assert rsi.iloc[-1] == pytest.approx(51.78, abs=0.5)

    def test_rsi_all_gains(self):
        """RSI should be 100 when all price moves are up."""
        prices = list(range(100, 200))  # All upward moves
        df = pd.DataFrame({'close': prices})

        rsi = calculate_rsi(df, period=14)

        assert rsi.iloc[-1] == pytest.approx(100.0, abs=0.01)

    def test_rsi_all_losses(self):
        """RSI should be 0 when all price moves are down."""
        prices = list(range(200, 100, -1))  # All downward moves
        df = pd.DataFrame({'close': prices})

        rsi = calculate_rsi(df, period=14)

        assert rsi.iloc[-1] == pytest.approx(0.0, abs=0.01)

    def test_rsi_warmup_period(self):
        """First N values should be NaN during warmup."""
        prices = list(range(100, 150))
        df = pd.DataFrame({'close': prices})
        period = 14

        rsi = calculate_rsi(df, period=period)

        # First 'period' values should be NaN
        assert rsi.iloc[:period].isna().all()
        # Rest should be valid
        assert not rsi.iloc[period:].isna().any()


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

@pytest.mark.unit
class TestRSIEdgeCases:
    """Test RSI behavior with edge cases."""

    def test_rsi_insufficient_data(self):
        """Should handle insufficient data gracefully."""
        prices = [100, 101, 102]  # Too few for period=14
        df = pd.DataFrame({'close': prices})

        rsi = calculate_rsi(df, period=14)

        assert rsi.isna().all()

    def test_rsi_with_nan_values(self):
        """Should handle NaN values in input."""
        prices = [100, 101, np.nan, 103, 104]
        df = pd.DataFrame({'close': prices})

        rsi = calculate_rsi(df, period=3)

        # Should not crash
        assert rsi is not None
        # NaN should propagate or be handled
        assert len(rsi) == len(prices)

    def test_rsi_constant_prices(self):
        """Should handle constant prices (no change)."""
        prices = [100] * 50
        df = pd.DataFrame({'close': prices})

        rsi = calculate_rsi(df, period=14)

        # RSI should be 50 (neutral) or NaN when no change
        valid_rsi = rsi.dropna()
        if len(valid_rsi) > 0:
            assert valid_rsi.iloc[-1] == pytest.approx(50.0, abs=0.1)


# =============================================================================
# PROPERTY-BASED TESTS
# =============================================================================

@pytest.mark.unit
class TestRSIProperties:
    """Property-based tests for RSI invariants."""

    @given(
        prices=st.lists(
            st.floats(min_value=1, max_value=10000, allow_nan=False),
            min_size=50,
            max_size=500
        ),
        period=st.integers(min_value=5, max_value=30)
    )
    def test_rsi_always_bounded(self, prices, period):
        """RSI should always be between 0 and 100."""
        df = pd.DataFrame({'close': prices})
        rsi = calculate_rsi(df, period=period)

        valid_rsi = rsi.dropna()
        if len(valid_rsi) > 0:
            assert (valid_rsi >= 0).all(), "RSI below 0"
            assert (valid_rsi <= 100).all(), "RSI above 100"

    @given(
        prices=st.lists(
            st.floats(min_value=100, max_value=200),
            min_size=50
        )
    )
    def test_rsi_deterministic(self, prices):
        """RSI calculation should be deterministic."""
        df = pd.DataFrame({'close': prices})

        rsi1 = calculate_rsi(df, period=14)
        rsi2 = calculate_rsi(df, period=14)

        pd.testing.assert_series_equal(rsi1, rsi2)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

@pytest.mark.integration
class TestRSIIntegration:
    """Integration tests with real-world scenarios."""

    def test_rsi_with_market_data(self, sample_ohlcv_data):
        """Should calculate RSI on realistic market data."""
        rsi = calculate_rsi(sample_ohlcv_data, period=14)

        # Basic sanity checks
        assert len(rsi) == len(sample_ohlcv_data)
        assert rsi.dtype == np.float64

        # Should have reasonable values
        valid_rsi = rsi.dropna()
        assert len(valid_rsi) > 0
        assert valid_rsi.mean() > 20  # Not all oversold
        assert valid_rsi.mean() < 80  # Not all overbought


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

@pytest.mark.slow
class TestRSIPerformance:
    """Performance benchmarks for RSI calculation."""

    def test_rsi_performance_large_dataset(self, benchmark):
        """RSI should calculate quickly on large dataset."""
        prices = np.random.randn(100000).cumsum() + 1000
        df = pd.DataFrame({'close': prices})

        result = benchmark(calculate_rsi, df, period=14)

        assert len(result) == len(df)
```

This example demonstrates:
- Clear organization with test classes
- Comprehensive coverage (happy path, edge cases, properties)
- Appropriate use of pytest markers
- Property-based testing with Hypothesis
- Integration and performance tests
- Good documentation

---

**Document Version:** 1.0
**Last Updated:** 2026-01-20
**Author:** Claude Code (Test Automation Expert)
