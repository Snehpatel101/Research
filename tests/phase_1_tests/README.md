# Phase 1 Test Suite

Comprehensive, well-organized test suite for Phase 1 pipeline stages.

## Overview

This test suite provides complete coverage for all 8 Phase 1 pipeline stages, pipeline orchestration, utilities, and validation logic. All tests follow TDD best practices with clear organization and modular structure.

**Total: 21 test files, 7,567 lines, 303 test cases**

All files adhere to the 650-line limit mandated by project engineering rules.

## Directory Structure

```
tests/phase_1_tests/
├── conftest.py                          # Shared fixtures and test utilities
├── __init__.py                          # Package initialization
│
├── stages/                              # Stage-specific tests (9 files)
│   ├── test_stage1_ingest.py           # Data ingestion (19 tests)
│   ├── test_stage2_data_cleaning.py    # Data cleaning (19 tests)
│   ├── test_stage3_feature_engineering_core.py  # Core indicators (20 tests)
│   ├── test_stage3_feature_engineering_advanced.py  # Advanced features (16 tests)
│   ├── test_stage4_triple_barrier_labeling.py   # Triple barrier labeling (17 tests)
│   ├── test_stage5_ga_optimization.py  # GA optimization (12 tests)
│   ├── test_stage6_final_labels.py     # Final label application (11 tests)
│   ├── test_stage7_data_splitting.py   # Train/val/test splits (11 tests)
│   └── test_stage8_data_validation.py  # Data validation (11 tests)
│
├── pipeline/                            # Pipeline orchestration tests (6 files)
│   ├── test_pipeline_core.py           # Core pipeline components (23 tests)
│   ├── test_pipeline_execution.py      # Execution and dependencies (17 tests)
│   ├── test_pipeline_integration.py    # End-to-end integration (22 tests)
│   ├── test_pipeline_state.py          # State persistence (11 tests)
│   ├── test_pipeline_system.py         # System configuration (5 tests)
│   └── test_pipeline.py                # Basic pipeline tests (3 tests)
│
├── utilities/                           # Utility module tests (2 files)
│   ├── test_feature_scaler.py          # Feature scaling (24 tests)
│   └── test_time_series_cv.py          # Time series cross-validation (25 tests)
│
├── validation/                          # Validation and quality tests (3 files)
│   ├── test_input_validation.py        # Parameter validation (37 tests)
│   ├── test_edge_cases.py              # Edge case handling (20 tests)
│   └── test_error_handling.py          # Exception handling (20 tests)
│
└── fixtures/                            # Test data generators (1 file)
    ├── data_generators.py              # OHLCV and feature data generators
    └── __init__.py
```

## Running Tests

### Run All Phase 1 Tests
```bash
pytest tests/phase_1_tests/ -v
```

### Run Specific Categories
```bash
# All stage tests
pytest tests/phase_1_tests/stages/ -v

# Specific stage
pytest tests/phase_1_tests/stages/test_stage1_ingest.py -v

# Pipeline tests
pytest tests/phase_1_tests/pipeline/ -v

# Utilities
pytest tests/phase_1_tests/utilities/ -v

# Validation tests
pytest tests/phase_1_tests/validation/ -v
```

### With Coverage
```bash
# Coverage for all Phase 1 code
pytest tests/phase_1_tests/ --cov=src/stages --cov=src/pipeline --cov=src/utils --cov-report=term-missing

# Coverage for specific stage
pytest tests/phase_1_tests/stages/test_stage1_ingest.py --cov=src/stages/stage1_ingest --cov-report=html
```

### Parallel Execution
```bash
# Run tests in parallel (requires pytest-xdist)
pytest tests/phase_1_tests/ -n auto
```

### Markers and Filters
```bash
# Run only fast tests (if marked)
pytest tests/phase_1_tests/ -m "not slow"

# Run tests matching pattern
pytest tests/phase_1_tests/ -k "ingest"
```

## Test Organization Principles

### 1. Modular Structure
- Each stage has dedicated test file(s)
- No file exceeds 650 lines
- Clear separation of concerns

### 2. Shared Fixtures
- Common fixtures in `conftest.py`
- Specialized data generators in `fixtures/data_generators.py`
- Proper fixture scoping for performance

### 3. Test Categories

#### Stage Tests
Test individual pipeline stages in isolation:
- Input validation at boundaries
- Core algorithm correctness
- Output quality and format
- Error handling

#### Pipeline Tests
Test pipeline orchestration and integration:
- Stage execution order
- Dependency management
- State persistence
- Error recovery

#### Utility Tests
Test reusable utility modules:
- Feature scaling and normalization
- Time series cross-validation
- Feature selection

#### Validation Tests
Test data quality and validation logic:
- Parameter validation
- Edge case handling
- Exception handling
- Error messages

### 4. Test Naming Convention
```python
class Test<Component><Aspect>:
    """Tests for <component> <aspect> functionality."""

    def test_<scenario>_<expected_outcome>(self):
        """Test that <scenario> results in <expected outcome>."""
        # Arrange
        # Act
        # Assert
```

## Shared Fixtures

### Temporary Directories
- `temp_dir`: Basic temporary directory
- `temp_project_dir`: Complete project directory structure

### OHLCV Data
- `sample_ohlcv_df`: Standard OHLCV DataFrame (500 bars)
- `sample_ohlcv_with_gaps`: OHLCV with missing data
- `sample_ohlcv_data`: OHLCV with ATR for advanced tests
- `sample_labeled_data`: Labeled data for stages 5-8

### Configuration
- `sample_config`: Single-symbol pipeline configuration
- `multi_symbol_config`: Multi-symbol configuration

### Utilities
- `assert_valid_ohlcv()`: Validate OHLCV structure
- `assert_no_lookahead()`: Check for lookahead bias
- `assert_monotonic_datetime()`: Validate datetime ordering

## Test Data Generators

The `fixtures/data_generators.py` module provides specialized data generation:

```python
from tests.phase_1_tests.fixtures.data_generators import (
    generate_trending_market,
    generate_ranging_market,
    generate_volatile_market,
    add_gaps,
    add_outliers,
    generate_labeled_data,
)
```

## Coverage Goals

- **Overall Coverage**: >80%
- **Stage Tests**: >70% per stage
- **Critical Paths**: 100% (validation, labeling, splitting)

## Adding New Tests

1. **Determine Category**: stage, pipeline, utility, or validation
2. **Choose Appropriate File**: Add to existing file if under 650 lines
3. **Use Shared Fixtures**: Reuse fixtures from `conftest.py`
4. **Follow Naming Convention**: Clear, descriptive test names
5. **Document**: Add docstring explaining test purpose

Example:
```python
class TestStage1NewFeature:
    """Tests for Stage 1 new feature functionality."""

    def test_new_feature_with_valid_input(self, temp_dir, sample_ohlcv_df):
        """Test that new feature works correctly with valid input."""
        # Arrange
        ingestor = DataIngestor(raw_data_dir=temp_dir, output_dir=temp_dir / "output")

        # Act
        result = ingestor.new_feature(sample_ohlcv_df)

        # Assert
        assert result is not None
        assert len(result) == len(sample_ohlcv_df)
```

## Continuous Integration

Tests are designed to run in CI environments:
- Deterministic (fixed random seeds)
- Fast (efficient fixtures, minimal I/O)
- Isolated (no shared state between tests)
- Clean (proper teardown)

## Engineering Rules Compliance

✓ **Modular**: Clear separation by stage and function
✓ **File Limits**: All files under 650 lines
✓ **Fail Fast**: Tests detect errors at boundaries
✓ **Less Code**: Simple, focused tests
✓ **No Exception Swallowing**: Explicit error handling
✓ **Clear Validation**: Boundary validation in all tests
✓ **Definition of Done**: Complete with passing tests

## Migration from Old Tests

Old test files in `tests/` root:
- ✓ Consolidated into organized structure
- ✓ Duplicates removed
- ✓ All under 650-line limit
- ✓ Modern Python patterns applied
- ✓ Import paths updated

Original files preserved for reference until Phase 1 completion.

## Support

For questions or issues with tests:
1. Check test docstrings for intent
2. Review shared fixtures in `conftest.py`
3. Examine similar tests for patterns
4. Ensure imports and paths are correct

---

**Last Updated**: 2025-12-21
**Test Suite Version**: 1.0
**Pipeline Version**: Phase 1
