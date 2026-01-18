#!/usr/bin/env python
"""
PHASE_2 Adapters Comprehensive Validation Test
==============================================

This script performs a complete validation of the PHASE_2 (Adapters) module
in the ML Factory pipeline.

Components Tested:
1. Base (base.py): BaseAdapter, AdapterResult
2. Registry (registry.py): AdapterRegistry, get_adapter()
3. TabularAdapter (tabular.py): 2D arrays for boosting/classical
4. SequenceAdapter (sequence.py): 3D arrays for RNN/TCN
5. MultiStreamAdapter (multi_stream.py): 4D arrays for transformers
6. Factory (factory.py): AdapterFactory, create_adapter_factory()
7. Scaling (scaling.py): AdapterScaler, ScalerConfig, create_scaler()
8. Preparation (preparation.py): PreparedData, UnifiedDataPreparation
9. Alignment (alignment.py): AlignedOOFResult, OOFAligner

Run with:
    python tests/validate_phase2_adapters.py
"""

import sys
import traceback
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import tempfile
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd


# =============================================================================
# TEST RESULT TRACKING
# =============================================================================

@dataclass
class TestResult:
    """Individual test result."""
    name: str
    passed: bool
    message: str = ""
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class ValidationReport:
    """Collects and reports validation results."""

    def __init__(self):
        self.results: List[TestResult] = []
        self.import_errors: List[str] = []
        self.components_tested: List[str] = []
        self.registered_adapters: Dict[str, str] = {}
        self.model_mappings: Dict[str, Dict[str, Any]] = {}

    def add_result(self, result: TestResult):
        self.results.append(result)

    def add_import_error(self, module: str, error: str):
        self.import_errors.append(f"{module}: {error}")

    def mark_component_tested(self, component: str):
        if component not in self.components_tested:
            self.components_tested.append(component)

    def set_registered_adapters(self, adapters: Dict[str, str]):
        self.registered_adapters = adapters

    def set_model_mappings(self, mappings: Dict[str, Dict[str, Any]]):
        self.model_mappings = mappings

    @property
    def total_tests(self) -> int:
        return len(self.results)

    @property
    def passed_tests(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_tests(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def all_passed(self) -> bool:
        return self.failed_tests == 0 and len(self.import_errors) == 0

    def print_report(self):
        """Print comprehensive validation report."""
        print("\n" + "=" * 80)
        print("PHASE_2 ADAPTERS VALIDATION REPORT")
        print("=" * 80)

        # Import Errors Section
        print("\n--- IMPORT STATUS ---")
        if self.import_errors:
            print(f"[FAIL] {len(self.import_errors)} import error(s):")
            for error in self.import_errors:
                print(f"  - {error}")
        else:
            print("[PASS] All imports successful")

        # Components Tested
        print(f"\n--- COMPONENTS TESTED ({len(self.components_tested)}) ---")
        for comp in self.components_tested:
            print(f"  [x] {comp}")

        # Registered Adapters
        print(f"\n--- REGISTERED ADAPTERS ({len(self.registered_adapters)}) ---")
        for adapter_id, adapter_class in sorted(self.registered_adapters.items()):
            print(f"  - {adapter_id}: {adapter_class}")

        # Model Mappings
        print(f"\n--- MODEL-ADAPTER MAPPINGS ({len(self.model_mappings)}) ---")
        by_adapter: Dict[str, List[str]] = {}
        for model, info in self.model_mappings.items():
            adapter = info.get("adapter", "unknown")
            if adapter not in by_adapter:
                by_adapter[adapter] = []
            by_adapter[adapter].append(model)

        for adapter, models in sorted(by_adapter.items()):
            rank = 2 if adapter == "tabular" else (3 if adapter == "sequence" else 4)
            print(f"  {adapter} (rank={rank}D):")
            for model in sorted(models):
                print(f"    - {model}")

        # Test Results Summary
        print(f"\n--- TEST RESULTS ---")
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")

        # Detailed Results
        if self.failed_tests > 0:
            print("\n--- FAILED TESTS ---")
            for result in self.results:
                if not result.passed:
                    print(f"\n[FAIL] {result.name}")
                    print(f"  Message: {result.message}")
                    if result.details:
                        for key, value in result.details.items():
                            print(f"  {key}: {value}")

        # All passed tests (abbreviated)
        print("\n--- PASSED TESTS ---")
        for result in self.results:
            if result.passed:
                print(f"  [PASS] {result.name}")

        # Final Status
        print("\n" + "=" * 80)
        if self.all_passed:
            print("PHASE_2 HEALTH STATUS: [HEALTHY] All validations passed!")
        else:
            print("PHASE_2 HEALTH STATUS: [UNHEALTHY] Some validations failed")
        print("=" * 80)


# =============================================================================
# TEST DATA HELPERS
# =============================================================================

def create_test_dataframe(n_samples: int = 500, n_features: int = 10, seed: int = 42) -> pd.DataFrame:
    """Create a test DataFrame with required columns for adapter testing."""
    np.random.seed(seed)

    # Create synthetic OHLCV data
    close = 4000 + np.cumsum(np.random.randn(n_samples) * 5)
    high = close + np.abs(np.random.randn(n_samples) * 3)
    low = close - np.abs(np.random.randn(n_samples) * 3)
    open_price = close + np.random.randn(n_samples) * 2
    volume = np.abs(np.random.randn(n_samples) * 1000 + 5000)

    data = {
        "datetime": pd.date_range("2024-01-01", periods=n_samples, freq="1min"),
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "symbol": "MES",
        "timeframe": "1min",
    }

    # Add synthetic features
    for i in range(n_features):
        data[f"feature_{i}"] = np.random.randn(n_samples)

    # Add labels and weights
    data["label_h20"] = np.random.choice([-1, 0, 1], size=n_samples)
    data["sample_weight_h20"] = np.random.uniform(0.5, 1.5, size=n_samples)

    df = pd.DataFrame(data)
    df.set_index("datetime", inplace=True)
    df.index = np.arange(n_samples)  # Simple integer index for testing

    return df


def create_multi_timeframe_dfs(n_samples: int = 500, seed: int = 42) -> Dict[str, pd.DataFrame]:
    """Create test DataFrames for multiple timeframes."""
    np.random.seed(seed)

    dfs = {}
    for tf, divisor in [("1min", 1), ("5min", 5), ("15min", 15)]:
        tf_samples = n_samples // divisor

        close = 4000 + np.cumsum(np.random.randn(tf_samples) * 5)
        high = close + np.abs(np.random.randn(tf_samples) * 3)
        low = close - np.abs(np.random.randn(tf_samples) * 3)
        open_price = close + np.random.randn(tf_samples) * 2
        volume = np.abs(np.random.randn(tf_samples) * 1000 + 5000)

        data = {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }

        df = pd.DataFrame(data)
        df["label_h20"] = np.random.choice([-1, 0, 1], size=tf_samples)
        df["sample_weight_h20"] = np.random.uniform(0.5, 1.5, size=tf_samples)

        dfs[tf] = df

    return dfs


# =============================================================================
# VALIDATION TESTS
# =============================================================================

def validate_imports(report: ValidationReport) -> bool:
    """Test 1: Validate all imports work correctly."""
    try:
        # Base imports
        from src.adapters.base import BaseAdapter, AdapterResult
        report.mark_component_tested("base.py: BaseAdapter, AdapterResult")

        # Registry imports
        from src.adapters.registry import AdapterRegistry, get_adapter
        report.mark_component_tested("registry.py: AdapterRegistry, get_adapter")

        # Adapter imports
        from src.adapters.tabular import TabularAdapter
        report.mark_component_tested("tabular.py: TabularAdapter")

        from src.adapters.sequence import SequenceAdapter
        report.mark_component_tested("sequence.py: SequenceAdapter")

        from src.adapters.multi_stream import MultiStreamAdapter
        report.mark_component_tested("multi_stream.py: MultiStreamAdapter")

        # Factory imports
        from src.adapters.factory import AdapterFactory, create_adapter_factory
        report.mark_component_tested("factory.py: AdapterFactory, create_adapter_factory")

        # Scaling imports
        from src.adapters.scaling import AdapterScaler, ScalerConfig, create_scaler
        report.mark_component_tested("scaling.py: AdapterScaler, ScalerConfig, create_scaler")

        # Preparation imports
        from src.adapters.preparation import PreparedData, UnifiedDataPreparation, prepare_for_model
        report.mark_component_tested("preparation.py: PreparedData, UnifiedDataPreparation, prepare_for_model")

        # Alignment imports
        from src.adapters.alignment import (
            AlignedOOFResult, OOFAligner, align_oof_predictions,
            compute_coverage_stats, validate_oof_results
        )
        report.mark_component_tested("alignment.py: AlignedOOFResult, OOFAligner, utilities")

        # Test package-level imports
        from src.adapters import (
            AdapterRegistry, get_adapter,
            BaseAdapter, AdapterResult,
            TabularAdapter, SequenceAdapter, MultiStreamAdapter,
            AdapterFactory, create_adapter_factory,
            AdapterScaler, ScalerConfig, create_scaler,
            PreparedData, UnifiedDataPreparation, prepare_for_model,
            AlignedOOFResult, OOFAligner, align_oof_predictions,
            compute_coverage_stats, validate_oof_results,
        )
        report.mark_component_tested("__init__.py: Package-level exports")

        report.add_result(TestResult(
            name="Import Validation",
            passed=True,
            message="All imports successful"
        ))
        return True

    except ImportError as e:
        report.add_import_error("src.adapters", str(e))
        report.add_result(TestResult(
            name="Import Validation",
            passed=False,
            message=f"Import failed: {e}"
        ))
        return False


def validate_base_classes(report: ValidationReport):
    """Test 2: Validate BaseAdapter and AdapterResult."""
    from src.adapters.base import BaseAdapter, AdapterResult
    from src.contracts import DataRank

    # Test AdapterResult creation and validation
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randint(0, 3, size=100).astype(np.int64)
    weights = np.random.uniform(0.5, 1.5, size=100).astype(np.float32)

    result = AdapterResult(
        X=X,
        y=y,
        weights=weights,
        data_rank=DataRank.TABULAR_2D,
        feature_columns=[f"feat_{i}" for i in range(50)],
        adapter_name="test_adapter"
    )

    # Test validation
    is_valid, issues = result.validate()

    report.add_result(TestResult(
        name="AdapterResult.validate() - Valid Data",
        passed=is_valid,
        message="Validation passed" if is_valid else f"Issues: {issues}",
        details={"n_samples": result.n_samples, "n_features": result.n_features}
    ))

    # Test validation with mismatched data
    bad_result = AdapterResult(
        X=np.random.randn(100, 50).astype(np.float32),
        y=np.random.randint(0, 3, size=50).astype(np.int64),  # Wrong size
        data_rank=DataRank.TABULAR_2D
    )
    is_valid_bad, issues_bad = bad_result.validate()

    report.add_result(TestResult(
        name="AdapterResult.validate() - Invalid Data Detection",
        passed=not is_valid_bad and len(issues_bad) > 0,
        message=f"Correctly detected issues: {issues_bad}" if not is_valid_bad else "Failed to detect issues",
    ))

    # Test to_dict serialization
    result_dict = result.to_dict()
    required_keys = ["n_samples", "n_features", "data_rank", "X_shape", "y_shape", "adapter_name"]
    has_all_keys = all(k in result_dict for k in required_keys)

    report.add_result(TestResult(
        name="AdapterResult.to_dict() Serialization",
        passed=has_all_keys,
        message="All required keys present" if has_all_keys else "Missing keys",
        details={"keys": list(result_dict.keys())}
    ))

    # Verify BaseAdapter is abstract
    try:
        adapter = BaseAdapter()
        report.add_result(TestResult(
            name="BaseAdapter Abstract Enforcement",
            passed=False,
            message="BaseAdapter should not be instantiable"
        ))
    except TypeError:
        report.add_result(TestResult(
            name="BaseAdapter Abstract Enforcement",
            passed=True,
            message="BaseAdapter correctly raises TypeError on instantiation"
        ))


def validate_registry(report: ValidationReport):
    """Test 3: Validate AdapterRegistry and get_adapter()."""
    from src.adapters.registry import AdapterRegistry, get_adapter
    from src.adapters.tabular import TabularAdapter
    from src.adapters.sequence import SequenceAdapter
    from src.adapters.multi_stream import MultiStreamAdapter

    # Get registered adapters
    registered = AdapterRegistry.list_adapters()

    report.set_registered_adapters({
        adapter_id: AdapterRegistry.get(adapter_id).__name__
        for adapter_id in registered
    })

    # Check expected adapters are registered
    expected_adapters = ["tabular", "sequence", "multi_stream"]
    all_registered = all(a in registered for a in expected_adapters)

    report.add_result(TestResult(
        name="AdapterRegistry - Expected Adapters Registered",
        passed=all_registered,
        message=f"Registered: {registered}",
        details={"expected": expected_adapters, "actual": registered}
    ))

    # Test get by ID
    try:
        tabular = AdapterRegistry.get("tabular")
        is_correct = tabular == TabularAdapter
        report.add_result(TestResult(
            name="AdapterRegistry.get('tabular')",
            passed=is_correct,
            message=f"Got {tabular.__name__}"
        ))
    except Exception as e:
        report.add_result(TestResult(
            name="AdapterRegistry.get('tabular')",
            passed=False,
            message=str(e)
        ))

    # Test create
    try:
        adapter = AdapterRegistry.create("sequence", sequence_length=30)
        is_correct = isinstance(adapter, SequenceAdapter) and adapter.sequence_length == 30
        report.add_result(TestResult(
            name="AdapterRegistry.create() with kwargs",
            passed=is_correct,
            message=f"Created {type(adapter).__name__} with seq_len={adapter.sequence_length}"
        ))
    except Exception as e:
        report.add_result(TestResult(
            name="AdapterRegistry.create() with kwargs",
            passed=False,
            message=str(e)
        ))

    # Test get_adapter convenience function with model_name
    try:
        adapter = get_adapter(model_name="xgboost")
        is_correct = isinstance(adapter, TabularAdapter)
        report.add_result(TestResult(
            name="get_adapter(model_name='xgboost')",
            passed=is_correct,
            message=f"Got {type(adapter).__name__}"
        ))
    except Exception as e:
        report.add_result(TestResult(
            name="get_adapter(model_name='xgboost')",
            passed=False,
            message=str(e)
        ))

    # Test get_adapter with adapter_id
    try:
        adapter = get_adapter(adapter_id="sequence", sequence_length=45)
        is_correct = isinstance(adapter, SequenceAdapter) and adapter.sequence_length == 45
        report.add_result(TestResult(
            name="get_adapter(adapter_id='sequence')",
            passed=is_correct,
            message=f"Got {type(adapter).__name__}"
        ))
    except Exception as e:
        report.add_result(TestResult(
            name="get_adapter(adapter_id='sequence')",
            passed=False,
            message=str(e)
        ))

    # Test error on unknown adapter
    try:
        AdapterRegistry.get("nonexistent")
        report.add_result(TestResult(
            name="AdapterRegistry - Unknown Adapter Error",
            passed=False,
            message="Should have raised ValueError"
        ))
    except ValueError:
        report.add_result(TestResult(
            name="AdapterRegistry - Unknown Adapter Error",
            passed=True,
            message="Correctly raised ValueError for unknown adapter"
        ))


def validate_tabular_adapter(report: ValidationReport):
    """Test 4: Validate TabularAdapter (2D arrays)."""
    from src.adapters.tabular import TabularAdapter
    from src.contracts import DataRank

    df = create_test_dataframe(n_samples=200, n_features=15)
    feature_cols = [f"feature_{i}" for i in range(15)]

    adapter = TabularAdapter(
        feature_columns=feature_cols,
        label_column="label_h20",
        weight_column="sample_weight_h20"
    )

    try:
        result = adapter.transform(df)

        # Check output shape
        expected_shape = (200, 15)
        shape_correct = result.X.shape == expected_shape

        report.add_result(TestResult(
            name="TabularAdapter - Output Shape (2D)",
            passed=shape_correct,
            message=f"Shape: {result.X.shape}",
            details={"expected": expected_shape, "actual": result.X.shape}
        ))

        # Check data rank
        rank_correct = result.data_rank == DataRank.TABULAR_2D
        report.add_result(TestResult(
            name="TabularAdapter - Data Rank",
            passed=rank_correct,
            message=f"Rank: {result.data_rank}"
        ))

        # Check y shape
        y_correct = result.y.shape == (200,)
        report.add_result(TestResult(
            name="TabularAdapter - Labels Shape",
            passed=y_correct,
            message=f"y.shape: {result.y.shape}"
        ))

        # Check weights present
        weights_present = result.weights is not None and result.weights.shape == (200,)
        report.add_result(TestResult(
            name="TabularAdapter - Weights Extracted",
            passed=weights_present,
            message=f"weights.shape: {result.weights.shape if result.weights is not None else None}"
        ))

        # Check data contract created
        contract_created = result.data_contract is not None
        report.add_result(TestResult(
            name="TabularAdapter - DataContract Created",
            passed=contract_created,
            message=f"Contract: {result.data_contract}"
        ))

        # Validate result
        is_valid, issues = result.validate()
        report.add_result(TestResult(
            name="TabularAdapter - Result Validation",
            passed=is_valid,
            message="Valid" if is_valid else f"Issues: {issues}"
        ))

    except Exception as e:
        report.add_result(TestResult(
            name="TabularAdapter - Transform",
            passed=False,
            message=f"Exception: {e}",
            details={"traceback": traceback.format_exc()}
        ))


def validate_sequence_adapter(report: ValidationReport):
    """Test 5: Validate SequenceAdapter (3D arrays)."""
    from src.adapters.sequence import SequenceAdapter
    from src.contracts import DataRank

    df = create_test_dataframe(n_samples=200, n_features=10)
    feature_cols = [f"feature_{i}" for i in range(10)]

    seq_len = 30
    adapter = SequenceAdapter(
        feature_columns=feature_cols,
        label_column="label_h20",
        weight_column="sample_weight_h20",
        sequence_length=seq_len,
        stride=1
    )

    try:
        result = adapter.transform(df)

        # Calculate expected number of sequences
        expected_n_sequences = (200 - seq_len) // 1 + 1  # 171 sequences
        expected_shape = (expected_n_sequences, seq_len, 10)

        shape_correct = result.X.shape == expected_shape

        report.add_result(TestResult(
            name="SequenceAdapter - Output Shape (3D)",
            passed=shape_correct,
            message=f"Shape: {result.X.shape}",
            details={"expected": expected_shape, "actual": result.X.shape}
        ))

        # Check data rank
        rank_correct = result.data_rank == DataRank.SEQUENCE_3D
        report.add_result(TestResult(
            name="SequenceAdapter - Data Rank",
            passed=rank_correct,
            message=f"Rank: {result.data_rank}"
        ))

        # Check sequence length stored
        seq_len_correct = result.sequence_length == seq_len
        report.add_result(TestResult(
            name="SequenceAdapter - Sequence Length Stored",
            passed=seq_len_correct,
            message=f"sequence_length: {result.sequence_length}"
        ))

        # Check original_indices for alignment
        indices_correct = (result.original_indices is not None and
                         len(result.original_indices) == result.X.shape[0])
        report.add_result(TestResult(
            name="SequenceAdapter - Original Indices",
            passed=indices_correct,
            message=f"indices length: {len(result.original_indices) if result.original_indices is not None else None}"
        ))

        # Validate result
        is_valid, issues = result.validate()
        report.add_result(TestResult(
            name="SequenceAdapter - Result Validation",
            passed=is_valid,
            message="Valid" if is_valid else f"Issues: {issues}"
        ))

    except Exception as e:
        report.add_result(TestResult(
            name="SequenceAdapter - Transform",
            passed=False,
            message=f"Exception: {e}",
            details={"traceback": traceback.format_exc()}
        ))


def validate_multi_stream_adapter(report: ValidationReport):
    """Test 6: Validate MultiStreamAdapter (4D arrays)."""
    from src.adapters.multi_stream import MultiStreamAdapter
    from src.contracts import DataRank

    dfs = create_multi_timeframe_dfs(n_samples=500)
    df_1min = dfs["1min"]
    additional_dfs = {"5min": dfs["5min"], "15min": dfs["15min"]}

    seq_len = 30
    adapter = MultiStreamAdapter(
        feature_columns=["open", "high", "low", "close", "volume"],
        label_column="label_h20",
        weight_column="sample_weight_h20",
        sequence_length=seq_len,
        timeframes=["1min", "5min", "15min"]
    )

    try:
        result = adapter.transform(df_1min, additional_dfs=additional_dfs)

        # Check 4D shape
        n_sequences = (500 - seq_len) // 1 + 1
        expected_shape = (n_sequences, 3, seq_len, 5)  # (sequences, timeframes, seq_len, features)

        shape_correct = result.X.shape == expected_shape

        report.add_result(TestResult(
            name="MultiStreamAdapter - Output Shape (4D)",
            passed=shape_correct,
            message=f"Shape: {result.X.shape}",
            details={"expected": expected_shape, "actual": result.X.shape}
        ))

        # Check data rank
        rank_correct = result.data_rank == DataRank.MULTI_TF_4D
        report.add_result(TestResult(
            name="MultiStreamAdapter - Data Rank",
            passed=rank_correct,
            message=f"Rank: {result.data_rank}"
        ))

        # Check n_timeframes
        tf_correct = result.n_timeframes == 3
        report.add_result(TestResult(
            name="MultiStreamAdapter - N_Timeframes",
            passed=tf_correct,
            message=f"n_timeframes: {result.n_timeframes}"
        ))

        # Check timeframe names stored
        names_correct = result.timeframe_names == ["1min", "5min", "15min"]
        report.add_result(TestResult(
            name="MultiStreamAdapter - Timeframe Names",
            passed=names_correct,
            message=f"timeframe_names: {result.timeframe_names}"
        ))

    except Exception as e:
        report.add_result(TestResult(
            name="MultiStreamAdapter - Transform",
            passed=False,
            message=f"Exception: {e}",
            details={"traceback": traceback.format_exc()}
        ))


def validate_factory(report: ValidationReport):
    """Test 7: Validate AdapterFactory."""
    from src.adapters.factory import AdapterFactory, create_adapter_factory
    from src.adapters.tabular import TabularAdapter
    from src.adapters.sequence import SequenceAdapter
    from src.adapters.multi_stream import MultiStreamAdapter
    from src.core.config import PipelineConfig
    from src.core.constants import MODEL_ADAPTER_MAP, MODEL_DATA_RANKS

    # Store model mappings in report
    model_mappings = {}
    for model in MODEL_ADAPTER_MAP:
        model_mappings[model] = {
            "adapter": MODEL_ADAPTER_MAP[model],
            "rank": MODEL_DATA_RANKS.get(model, 2)
        }
    report.set_model_mappings(model_mappings)

    # Create a minimal config
    with tempfile.TemporaryDirectory() as tmpdir:
        config = PipelineConfig(
            symbol="MES",
            data_path="./data/test.parquet",
            output_dir=tmpdir,
            sequence_length=30,
            mtf_timeframes=["1min", "5min", "15min"]
        )

        factory = AdapterFactory(config)

        # Test get_adapter for different model types
        tests = [
            ("xgboost", TabularAdapter),
            ("lightgbm", TabularAdapter),
            ("lstm", SequenceAdapter),
            ("gru", SequenceAdapter),
            ("patchtst", MultiStreamAdapter),
        ]

        for model_name, expected_class in tests:
            try:
                adapter = factory.get_adapter(model_name)
                is_correct = isinstance(adapter, expected_class)
                report.add_result(TestResult(
                    name=f"AdapterFactory.get_adapter('{model_name}')",
                    passed=is_correct,
                    message=f"Got {type(adapter).__name__}, expected {expected_class.__name__}"
                ))
            except Exception as e:
                report.add_result(TestResult(
                    name=f"AdapterFactory.get_adapter('{model_name}')",
                    passed=False,
                    message=str(e)
                ))

        # Test prepare_data
        df = create_test_dataframe(n_samples=200, n_features=10)

        try:
            result = factory.prepare_data("xgboost", df)
            is_correct = result.X.ndim == 2 and result.X.shape[0] == 200
            report.add_result(TestResult(
                name="AdapterFactory.prepare_data('xgboost')",
                passed=is_correct,
                message=f"Shape: {result.X.shape}"
            ))
        except Exception as e:
            report.add_result(TestResult(
                name="AdapterFactory.prepare_data('xgboost')",
                passed=False,
                message=str(e)
            ))

        # Test prepare_heterogeneous
        try:
            results = factory.prepare_heterogeneous(
                models=["xgboost", "lightgbm"],
                df=df
            )
            all_correct = all(
                results[m].X.ndim == 2 for m in ["xgboost", "lightgbm"]
            )
            report.add_result(TestResult(
                name="AdapterFactory.prepare_heterogeneous()",
                passed=all_correct,
                message=f"Models: {list(results.keys())}"
            ))
        except Exception as e:
            report.add_result(TestResult(
                name="AdapterFactory.prepare_heterogeneous()",
                passed=False,
                message=str(e)
            ))

        # Test list_models_by_adapter
        by_adapter = AdapterFactory.list_models_by_adapter()
        has_all_adapters = all(a in by_adapter for a in ["tabular", "sequence", "multi_stream"])
        report.add_result(TestResult(
            name="AdapterFactory.list_models_by_adapter()",
            passed=has_all_adapters,
            message=f"Adapters: {list(by_adapter.keys())}"
        ))

        # Test create_adapter_factory convenience function
        factory2 = create_adapter_factory(config)
        is_factory = isinstance(factory2, AdapterFactory)
        report.add_result(TestResult(
            name="create_adapter_factory()",
            passed=is_factory,
            message=f"Created {type(factory2).__name__}"
        ))


def validate_scaling(report: ValidationReport):
    """Test 8: Validate AdapterScaler, ScalerConfig, create_scaler()."""
    from src.adapters.scaling import AdapterScaler, ScalerConfig, create_scaler

    # Test ScalerConfig
    config = ScalerConfig(method="robust", clip_value=5.0)
    config_dict = config.to_dict()
    config_loaded = ScalerConfig.from_dict(config_dict)

    config_round_trip = (
        config.method == config_loaded.method and
        config.clip_value == config_loaded.clip_value
    )
    report.add_result(TestResult(
        name="ScalerConfig - Serialization Round-Trip",
        passed=config_round_trip,
        message="to_dict/from_dict works correctly"
    ))

    # Test 2D scaling
    X_2d = np.random.randn(100, 50).astype(np.float32)
    scaler = create_scaler("robust", clip_value=5.0)
    X_scaled = scaler.fit_transform(X_2d)

    scaled_2d_correct = X_scaled.shape == X_2d.shape
    report.add_result(TestResult(
        name="AdapterScaler - 2D Scaling",
        passed=scaled_2d_correct,
        message=f"Input: {X_2d.shape}, Output: {X_scaled.shape}"
    ))

    # Test 3D scaling
    X_3d = np.random.randn(100, 30, 50).astype(np.float32)
    scaler_3d = create_scaler("robust")
    X_3d_scaled = scaler_3d.fit_transform(X_3d)

    scaled_3d_correct = X_3d_scaled.shape == X_3d.shape
    report.add_result(TestResult(
        name="AdapterScaler - 3D Scaling",
        passed=scaled_3d_correct,
        message=f"Input: {X_3d.shape}, Output: {X_3d_scaled.shape}"
    ))

    # Test 4D scaling
    X_4d = np.random.randn(100, 3, 30, 5).astype(np.float32)
    scaler_4d = create_scaler("robust")
    X_4d_scaled = scaler_4d.fit_transform(X_4d)

    scaled_4d_correct = X_4d_scaled.shape == X_4d.shape
    report.add_result(TestResult(
        name="AdapterScaler - 4D Scaling",
        passed=scaled_4d_correct,
        message=f"Input: {X_4d.shape}, Output: {X_4d_scaled.shape}"
    ))

    # Test transform on new data
    X_2d_new = np.random.randn(50, 50).astype(np.float32)
    X_new_scaled = scaler.transform(X_2d_new)
    transform_works = X_new_scaled.shape == X_2d_new.shape
    report.add_result(TestResult(
        name="AdapterScaler - Transform New Data",
        passed=transform_works,
        message=f"Transformed shape: {X_new_scaled.shape}"
    ))

    # Test inverse_transform
    X_unscaled = scaler.inverse_transform(X_scaled)
    inverse_works = X_unscaled.shape == X_2d.shape
    report.add_result(TestResult(
        name="AdapterScaler - Inverse Transform",
        passed=inverse_works,
        message=f"Inverse shape: {X_unscaled.shape}"
    ))

    # Test save/load
    with tempfile.TemporaryDirectory() as tmpdir:
        scaler_path = Path(tmpdir) / "scaler"
        scaler.save(scaler_path)

        loaded_scaler = AdapterScaler.load(scaler_path)

        # Transform with loaded scaler should match
        X_loaded_scaled = loaded_scaler.transform(X_2d_new)
        load_works = np.allclose(X_loaded_scaled, scaler.transform(X_2d_new))
        report.add_result(TestResult(
            name="AdapterScaler - Save/Load Round-Trip",
            passed=load_works,
            message="Loaded scaler produces same results"
        ))

    # Test different scaling methods
    for method in ["robust", "standard", "minmax", "none"]:
        try:
            s = create_scaler(method)
            X_test = np.random.randn(50, 20).astype(np.float32)
            X_out = s.fit_transform(X_test)
            report.add_result(TestResult(
                name=f"create_scaler('{method}')",
                passed=X_out.shape == X_test.shape,
                message=f"Output shape: {X_out.shape}"
            ))
        except Exception as e:
            report.add_result(TestResult(
                name=f"create_scaler('{method}')",
                passed=False,
                message=str(e)
            ))


def validate_preparation(report: ValidationReport):
    """Test 9: Validate PreparedData and UnifiedDataPreparation."""
    from src.adapters.preparation import PreparedData, UnifiedDataPreparation, prepare_for_model
    from src.core.config import PipelineConfig

    # Create test data
    df = create_test_dataframe(n_samples=1000, n_features=10)
    feature_cols = [f"feature_{i}" for i in range(10)]

    # Rename label column to match default
    df["label"] = df["label_h20"]

    with tempfile.TemporaryDirectory() as tmpdir:
        config = PipelineConfig(
            symbol="MES",
            data_path="./data/test.parquet",
            output_dir=tmpdir,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=10,
            embargo_bars=10,
            sequence_length=30
        )

        prep = UnifiedDataPreparation(config)

        # Test prepare for tabular model
        try:
            data = prep.prepare(
                df=df,
                model_name="xgboost",
                feature_columns=feature_cols,
                label_column="label"
            )

            # Check PreparedData structure
            checks = {
                "has_X_train": data.X_train is not None,
                "has_X_val": data.X_val is not None,
                "has_y_train": data.y_train is not None,
                "has_y_val": data.y_val is not None,
                "correct_rank": data.data_rank == 2,
                "correct_adapter": data.adapter_type == "tabular",
            }

            all_passed = all(checks.values())
            report.add_result(TestResult(
                name="UnifiedDataPreparation.prepare('xgboost')",
                passed=all_passed,
                message=data.summary() if all_passed else f"Checks: {checks}"
            ))

            # Test validation
            is_valid, issues = data.validate()
            report.add_result(TestResult(
                name="PreparedData.validate()",
                passed=is_valid,
                message="Valid" if is_valid else f"Issues: {issues}"
            ))

            # Test scaler stored
            scaler = prep.get_scaler("xgboost")
            report.add_result(TestResult(
                name="UnifiedDataPreparation - Scaler Stored",
                passed=scaler is not None and scaler.is_fitted,
                message=f"Scaler fitted: {scaler.is_fitted if scaler else False}"
            ))

        except Exception as e:
            report.add_result(TestResult(
                name="UnifiedDataPreparation.prepare('xgboost')",
                passed=False,
                message=str(e),
                details={"traceback": traceback.format_exc()}
            ))

        # Test prepare for sequence model
        try:
            data_seq = prep.prepare(
                df=df,
                model_name="lstm",
                feature_columns=feature_cols,
                label_column="label"
            )

            is_3d = data_seq.data_rank == 3 and data_seq.X_train.ndim == 3
            report.add_result(TestResult(
                name="UnifiedDataPreparation.prepare('lstm')",
                passed=is_3d,
                message=data_seq.summary()
            ))

        except Exception as e:
            report.add_result(TestResult(
                name="UnifiedDataPreparation.prepare('lstm')",
                passed=False,
                message=str(e)
            ))

        # Test prepare_heterogeneous
        try:
            results = prep.prepare_heterogeneous(
                df=df,
                models=["xgboost", "lightgbm"],
                feature_columns=feature_cols,
                label_column="label"
            )

            all_prepared = all(m in results for m in ["xgboost", "lightgbm"])
            report.add_result(TestResult(
                name="UnifiedDataPreparation.prepare_heterogeneous()",
                passed=all_prepared,
                message=f"Prepared: {list(results.keys())}"
            ))

        except Exception as e:
            report.add_result(TestResult(
                name="UnifiedDataPreparation.prepare_heterogeneous()",
                passed=False,
                message=str(e)
            ))

        # Test prepare_for_model convenience function
        try:
            data_conv = prepare_for_model(
                df=df,
                model_name="lightgbm",
                config=config,
                feature_columns=feature_cols,
                label_column="label"
            )
            report.add_result(TestResult(
                name="prepare_for_model() convenience function",
                passed=isinstance(data_conv, PreparedData),
                message=data_conv.summary()
            ))
        except Exception as e:
            report.add_result(TestResult(
                name="prepare_for_model() convenience function",
                passed=False,
                message=str(e)
            ))


def validate_alignment(report: ValidationReport):
    """Test 10: Validate OOF alignment utilities."""
    from src.adapters.alignment import (
        AlignedOOFResult, OOFAligner, align_oof_predictions,
        compute_coverage_stats, validate_oof_results
    )
    from src.core.interfaces import OOFResult

    # Create mock OOF results with different coverage
    n_full = 1000  # Tabular model has full coverage
    n_seq = 941   # Sequence model loses first 59 samples (seq_len=60)

    # OOF for tabular model (full coverage)
    oof_tabular = OOFResult(
        predictions=np.random.randint(0, 3, size=n_full),
        probabilities=np.random.dirichlet(np.ones(3), size=n_full).astype(np.float32),
        indices=np.arange(n_full),
        fold_ids=np.repeat(np.arange(5), n_full // 5),
        model_name="xgboost",
        coverage=1.0
    )

    # OOF for sequence model (partial coverage)
    oof_sequence = OOFResult(
        predictions=np.random.randint(0, 3, size=n_seq),
        probabilities=np.random.dirichlet(np.ones(3), size=n_seq).astype(np.float32),
        indices=np.arange(59, n_full),  # Missing first 59 samples
        fold_ids=np.repeat(np.arange(5), n_seq // 5 + 1)[:n_seq],
        model_name="lstm",
        coverage=n_seq / n_full
    )

    # Test OOFAligner
    aligner = OOFAligner()

    # Test compute_offset
    offset_tab = aligner.compute_offset("xgboost", sequence_length=60)
    offset_seq = aligner.compute_offset("lstm", sequence_length=60)

    report.add_result(TestResult(
        name="OOFAligner.compute_offset() - Tabular",
        passed=offset_tab == 0,
        message=f"Offset: {offset_tab}"
    ))

    report.add_result(TestResult(
        name="OOFAligner.compute_offset() - Sequence",
        passed=offset_seq == 59,
        message=f"Offset: {offset_seq}"
    ))

    # Test estimate_common_samples
    estimated = aligner.estimate_common_samples(
        total_samples=1000,
        model_names=["xgboost", "lstm"],
        sequence_length=60
    )
    report.add_result(TestResult(
        name="OOFAligner.estimate_common_samples()",
        passed=estimated == 941,
        message=f"Estimated: {estimated}"
    ))

    # Test align with intersection strategy
    try:
        aligned = aligner.align([oof_tabular, oof_sequence], strategy="intersection")

        report.add_result(TestResult(
            name="OOFAligner.align() - Intersection",
            passed=aligned.n_common == n_seq,
            message=f"n_common: {aligned.n_common}, expected: {n_seq}"
        ))

        # Check stacking features shape
        stacking = aligned.stacking_features
        expected_cols = aligned.n_models * aligned.n_classes + 2  # probs + mean_conf + agreement
        report.add_result(TestResult(
            name="AlignedOOFResult.stacking_features",
            passed=stacking.shape == (aligned.n_common, expected_cols),
            message=f"Shape: {stacking.shape}"
        ))

        # Test to_dataframe
        df = aligned.to_dataframe()
        report.add_result(TestResult(
            name="AlignedOOFResult.to_dataframe()",
            passed=isinstance(df, pd.DataFrame) and len(df) == aligned.n_common,
            message=f"DataFrame shape: {df.shape}"
        ))

        # Test get_valid_mask
        valid_mask = aligned.get_valid_mask()
        report.add_result(TestResult(
            name="AlignedOOFResult.get_valid_mask()",
            passed=len(valid_mask) == aligned.n_common,
            message=f"Mask shape: {valid_mask.shape}"
        ))

        # Test summary
        summary = aligned.summary()
        report.add_result(TestResult(
            name="AlignedOOFResult.summary()",
            passed="AlignedOOFResult" in summary and "Coverage" in summary,
            message="Summary generated correctly"
        ))

    except Exception as e:
        report.add_result(TestResult(
            name="OOFAligner.align() - Intersection",
            passed=False,
            message=str(e),
            details={"traceback": traceback.format_exc()}
        ))

    # Test align_oof_predictions convenience function
    try:
        aligned2 = align_oof_predictions([oof_tabular, oof_sequence])
        report.add_result(TestResult(
            name="align_oof_predictions() convenience function",
            passed=isinstance(aligned2, AlignedOOFResult),
            message=f"n_common: {aligned2.n_common}"
        ))
    except Exception as e:
        report.add_result(TestResult(
            name="align_oof_predictions() convenience function",
            passed=False,
            message=str(e)
        ))

    # Test compute_coverage_stats
    try:
        stats = compute_coverage_stats([oof_tabular, oof_sequence])
        has_all_models = "xgboost" in stats and "lstm" in stats
        report.add_result(TestResult(
            name="compute_coverage_stats()",
            passed=has_all_models,
            message=f"Stats: {stats}"
        ))
    except Exception as e:
        report.add_result(TestResult(
            name="compute_coverage_stats()",
            passed=False,
            message=str(e)
        ))

    # Test validate_oof_results
    try:
        warnings = validate_oof_results([oof_tabular, oof_sequence])
        report.add_result(TestResult(
            name="validate_oof_results()",
            passed=isinstance(warnings, list),
            message=f"Warnings: {warnings if warnings else 'None'}"
        ))
    except Exception as e:
        report.add_result(TestResult(
            name="validate_oof_results()",
            passed=False,
            message=str(e)
        ))


# =============================================================================
# MAIN VALIDATION RUNNER
# =============================================================================

def run_validation():
    """Run all validation tests and print report."""
    report = ValidationReport()

    print("\nStarting PHASE_2 Adapters Validation...")
    print("-" * 50)

    # Run imports first
    if not validate_imports(report):
        print("[CRITICAL] Import validation failed. Cannot continue.")
        report.print_report()
        return report

    # Run all validation tests
    print("Testing base classes...")
    validate_base_classes(report)

    print("Testing registry...")
    validate_registry(report)

    print("Testing TabularAdapter...")
    validate_tabular_adapter(report)

    print("Testing SequenceAdapter...")
    validate_sequence_adapter(report)

    print("Testing MultiStreamAdapter...")
    validate_multi_stream_adapter(report)

    print("Testing AdapterFactory...")
    validate_factory(report)

    print("Testing Scaling...")
    validate_scaling(report)

    print("Testing Preparation...")
    validate_preparation(report)

    print("Testing Alignment...")
    validate_alignment(report)

    # Print comprehensive report
    report.print_report()

    return report


if __name__ == "__main__":
    report = run_validation()
    sys.exit(0 if report.all_passed else 1)
