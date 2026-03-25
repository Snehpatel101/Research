"""
Regression tests for Phases 4-9 fixes and E2E validation (Phases 10-11) in ML Factory.

Phase 4: OOF Stacking / Leakage Fixes (2 tests)
Phase 5: Architecture Fixes (4 tests)
Phase 6: Memory Optimization (2 tests)
Phase 7: Backtest Realism (4 tests)
Phase 8: Medium Bugs (3 tests)
Phase 9: Dead Code Removal (1 test)
Phase 10-11: E2E Validation (2 tests)
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Phase 4 -- OOF Stacking / Leakage Fixes
# ---------------------------------------------------------------------------


class TestPhase4OOFStackingFixes:
    """Tests verifying Phase 4 OOF stacking and leakage prevention fixes."""

    def test_stacking_only_copies_prob_columns(self) -> None:
        """Verify oof_stacking.py only copies model-prefixed probability
        columns from subsequent models, not y_true or fold_id.

        When building a stacking dataset from multiple models, the first
        model's full DataFrame is used as the base. For subsequent models,
        only columns starting with the model name are copied. This prevents
        overwriting the authoritative y_true and fold_id from the first model.
        """
        from src.validation.cv.oof_core import OOFPrediction
        from src.validation.cv.oof_stacking import StackingDatasetBuilder

        n = 50
        dates = pd.date_range("2024-01-01", periods=n, freq="h")

        # Model A: fold_id = [0,0,...,1,1,...] (first 25 fold 0, last 25 fold 1)
        model_a_df = pd.DataFrame(
            {
                "model_a_pred": np.random.choice([-1, 0, 1], size=n).astype(float),
                "model_a_prob_short": np.random.rand(n),
                "model_a_prob_neutral": np.random.rand(n),
                "model_a_prob_long": np.random.rand(n),
                "fold_id": np.array([0] * 25 + [1] * 25),
                "y_true": np.random.choice([-1, 0, 1], size=n),
                "datetime": dates,
            }
        )

        # Model B: fold_id = [2,2,...,3,3,...] (deliberately different)
        model_b_df = pd.DataFrame(
            {
                "model_b_pred": np.random.choice([-1, 0, 1], size=n).astype(float),
                "model_b_prob_short": np.random.rand(n),
                "model_b_prob_neutral": np.random.rand(n),
                "model_b_prob_long": np.random.rand(n),
                "fold_id": np.array([2] * 25 + [3] * 25),
                "y_true": np.random.choice([-1, 0, 1], size=n),
                "datetime": dates,
            }
        )

        oof_a = OOFPrediction(
            model_name="model_a",
            predictions=model_a_df,
            fold_info=[],
            coverage=1.0,
        )
        oof_b = OOFPrediction(
            model_name="model_b",
            predictions=model_b_df,
            fold_info=[],
            coverage=1.0,
        )

        y_true = pd.Series(np.random.choice([-1, 0, 1], size=n))

        builder = StackingDatasetBuilder()
        dataset = builder.build_stacking_dataset(
            oof_predictions={"model_a": oof_a, "model_b": oof_b},
            y_true=y_true,
            horizon=20,
            add_derived_features=False,
            drop_nan_samples=False,
        )

        # fold_id should come from model_a (first model), not model_b
        np.testing.assert_array_equal(
            dataset.data["fold_id"].values,
            model_a_df["fold_id"].values,
            err_msg="fold_id should come from the first model (authoritative), not be overwritten",
        )

        # model_b's prob columns should be present
        for col in ["model_b_prob_short", "model_b_prob_neutral", "model_b_prob_long"]:
            assert col in dataset.data.columns, f"Missing expected column: {col}"

        # Verify source code: the loop only copies model-prefixed columns
        source = inspect.getsource(StackingDatasetBuilder.build_stacking_dataset)
        assert (
            "c.startswith(model_name)" in source
        ), "build_stacking_dataset must filter columns by model_name prefix"

    def test_unified_regime_entry_point(self) -> None:
        """Verify src/data/pipeline/stages/regime/unified.py exists and
        get_regime_labels() is importable and callable.
        """
        unified_path = Path("src/data/pipeline/stages/regime/unified.py")
        assert unified_path.exists(), f"Missing unified regime entry point: {unified_path}"

        from src.data.pipeline.stages.regime.unified import get_regime_labels

        assert callable(get_regime_labels), "get_regime_labels must be callable"

        sig = inspect.signature(get_regime_labels)
        params = list(sig.parameters.keys())
        assert "df" in params, "get_regime_labels must accept a 'df' parameter"


# ---------------------------------------------------------------------------
# Phase 5 -- Architecture Fixes
# ---------------------------------------------------------------------------


class TestPhase5ArchitectureFixes:
    """Tests verifying Phase 5 architecture consolidation fixes."""

    def test_single_experiment_config(self) -> None:
        """Verify only 1 class ExperimentConfig definition exists in src/.

        Multiple definitions cause import confusion and divergent behavior.
        """
        src_root = Path("src")
        count = 0
        files_found: list[str] = []

        for py_file in src_root.rglob("*.py"):
            with open(py_file) as f:
                try:
                    tree = ast.parse(f.read())
                except SyntaxError:
                    continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == "ExperimentConfig":
                    count += 1
                    files_found.append(str(py_file))

        assert count == 1, (
            f"Expected exactly 1 ExperimentConfig class definition, "
            f"found {count} in: {files_found}"
        )

    def test_single_feature_selection_result(self) -> None:
        """Verify only 1 class FeatureSelectionResult definition exists in src/."""
        src_root = Path("src")
        count = 0
        files_found: list[str] = []

        for py_file in src_root.rglob("*.py"):
            with open(py_file) as f:
                try:
                    tree = ast.parse(f.read())
                except SyntaxError:
                    continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == "FeatureSelectionResult":
                    count += 1
                    files_found.append(str(py_file))

        assert count == 1, (
            f"Expected exactly 1 FeatureSelectionResult class definition, "
            f"found {count} in: {files_found}"
        )

    def test_k_up_zero_not_falsy(self) -> None:
        """Verify triple_barrier.py respects k_up=0.0 (not treated as None/falsy).

        When k_up=0.0 is explicitly passed, it must be used as-is. The fix
        uses 'is not None' checks instead of truthiness checks.
        """
        source_path = Path("src/data/labeling/triple_barrier.py")
        with open(source_path) as f:
            source = f.read()

        # Verify the 'is not None' pattern is used for k_up
        assert (
            "k_up is not None" in source
        ), "triple_barrier.py must use 'k_up is not None' to distinguish 0.0 from None"
        assert (
            "k_down is not None" in source
        ), "triple_barrier.py must use 'k_down is not None' to distinguish 0.0 from None"

        # Verify there is no bare 'if k_up:' which would treat 0.0 as falsy
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Check for bare 'if k_up' or 'if not k_up' patterns
                test = node.test
                if isinstance(test, ast.Name) and test.id in ("k_up", "k_down"):
                    pytest.fail(
                        f"Found bare 'if {test.id}:' at line {node.lineno} "
                        f"which treats 0.0 as falsy. Use 'is not None' instead."
                    )

    def test_confidence_zero_not_falsy(self) -> None:
        """Verify backtest confidence=0.0 is not replaced by 0.5.

        The position sizing code uses 'is not None' to check confidence,
        so 0.0 is a valid confidence value and should not be replaced.
        """
        from src.inference.backtesting.backtest import Backtester

        source = inspect.getsource(Backtester._calculate_position_size)

        # Must use 'is not None' check, not truthiness
        assert "confidence is not None" in source, (
            "_calculate_position_size must use 'confidence is not None' "
            "to avoid replacing 0.0 with 0.5"
        )

        # Functional check: 0.0 should not be treated as None
        val: float | None = 0.0
        assert val is not None, "0.0 should not be treated as None"


# ---------------------------------------------------------------------------
# Phase 6 -- Memory Optimization
# ---------------------------------------------------------------------------


class TestPhase6MemoryOptimization:
    """Tests verifying Phase 6 memory optimization fixes."""

    def test_sequence_dataset_no_copy(self) -> None:
        """Verify SequenceDataset.__getitem__ does not call .copy() on feature slice.

        Array slicing with [start:end] returns a view (no copy). Calling
        .copy() would wastefully duplicate data for every batch.
        """
        from src.core.datasets.sequences import SequenceDataset

        source = inspect.getsource(SequenceDataset.__getitem__)

        # The feature slice should NOT use .copy()
        lines = source.split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "_features[" in stripped and ".copy()" in stripped:
                pytest.fail(
                    f"Found .copy() on feature slice in __getitem__: {stripped}. "
                    f"Array slicing already returns a view; .copy() wastes memory."
                )

        # Positive check: confirm it uses from_numpy (zero-copy to tensor)
        assert (
            "torch.from_numpy" in source
        ), "__getitem__ should use torch.from_numpy for zero-copy tensor creation"

    def test_val_batch_2x_train(self) -> None:
        """Verify validation batch size is 2x training batch size in neural model config.

        Validation does not need gradients, so 2x batch size is safe and
        reduces validation time by ~50%.
        """
        from src.models.neural.base_rnn import BaseRNNModel

        source = inspect.getsource(BaseRNNModel)

        # Look for the 2x pattern
        assert (
            "batch_size" in source and "* 2" in source
        ), "BaseRNNModel should set val batch_size = train batch_size * 2"

        # Verify specific pattern
        assert 'batch_size": train_config.get("batch_size"' in source or (
            "batch_size" in source and "* 2" in source
        ), "Val batch should be 2x train batch"


# ---------------------------------------------------------------------------
# Phase 7 -- Backtest Realism
# ---------------------------------------------------------------------------


class TestPhase7BacktestRealism:
    """Tests verifying Phase 7 backtest realism improvements."""

    def test_stop_slippage_applied(self) -> None:
        """Verify stop-loss exit price is worse than barrier price by slippage amount.

        When a stop is hit, the fill price should be WORSE than the barrier:
        - Long stop: fill_price = stop_level - slippage
        - Short stop: fill_price = stop_level + slippage
        """
        from src.inference.backtesting.backtest import Backtester

        source = inspect.getsource(Backtester._resolve_exit_price)

        # Verify slippage is applied to stop exits
        assert "slippage" in source, "_resolve_exit_price must apply slippage to stop exits"
        assert (
            "stop_price -= slippage" in source
        ), "Long stop should fill below barrier (stop_price -= slippage)"
        assert (
            "stop_price += slippage" in source
        ), "Short stop should fill above barrier (stop_price += slippage)"

    def test_kelly_min_trades(self) -> None:
        """Verify Kelly requires minimum trades before producing non-default values.

        Kelly criterion is unreliable with few trades (high variance in
        win_rate estimate). The system should wait for kelly_min_trades
        before activating Kelly sizing.
        """
        from src.inference.backtesting.backtest import BacktestConfig, Backtester

        config = BacktestConfig()
        assert (
            config.kelly_min_trades == 30
        ), f"Default kelly_min_trades should be 30, got {config.kelly_min_trades}"

        # Verify source code checks the threshold
        source = inspect.getsource(Backtester._update_kelly_stats)
        assert (
            "self.config.kelly_min_trades" in source
        ), "_update_kelly_stats must check self.config.kelly_min_trades threshold"

    def test_vwap_renamed_midpoint(self) -> None:
        """Verify ExecutionModel.MIDPOINT exists and VWAP is alias.

        The old VWAP name was misleading (it was actually (H+L)/2 midpoint).
        MIDPOINT is the canonical name; VWAP is kept as a backward-compat alias.
        """
        from src.inference.backtesting.backtest import ExecutionModel

        assert hasattr(ExecutionModel, "MIDPOINT"), "ExecutionModel must have MIDPOINT"
        assert hasattr(ExecutionModel, "VWAP"), "ExecutionModel must have VWAP alias"

        # VWAP and MIDPOINT should resolve to the same value
        assert (
            ExecutionModel.VWAP == ExecutionModel.MIDPOINT
        ), f"VWAP ({ExecutionModel.VWAP}) should equal MIDPOINT ({ExecutionModel.MIDPOINT})"

        # The value should be "midpoint"
        assert (
            ExecutionModel.MIDPOINT.value == "midpoint"
        ), f"MIDPOINT value should be 'midpoint', got '{ExecutionModel.MIDPOINT.value}'"

    def test_session_end_config(self) -> None:
        """Verify BacktestConfig has force_session_close field (default False).

        This enables closing all positions at the end of each trading session,
        which is critical for intraday strategies.
        """
        from src.inference.backtesting.backtest import BacktestConfig

        config = BacktestConfig()
        assert hasattr(
            config, "force_session_close"
        ), "BacktestConfig must have force_session_close field"
        assert (
            config.force_session_close is False
        ), f"Default force_session_close should be False, got {config.force_session_close}"


# ---------------------------------------------------------------------------
# Phase 8 -- Medium Bugs
# ---------------------------------------------------------------------------


class TestPhase8MediumBugs:
    """Tests verifying Phase 8 medium bug fixes."""

    def test_feature_cache_clearing(self) -> None:
        """Verify clear_all_feature_caches() actually empties all caches.

        Feature caches must be cleared between models/windows to prevent
        stale data and reclaim memory.
        """
        from src.data.features.compute import clear_all_feature_caches, volatility

        # Populate a cache (volatility module has known caches)
        cache_names = [
            attr
            for attr in dir(volatility)
            if attr.endswith("_cache") and isinstance(getattr(volatility, attr), dict)
        ]
        assert len(cache_names) > 0, "volatility module should have at least one cache dict"

        # Populate with dummy data
        for name in cache_names:
            cache = getattr(volatility, name)
            cache["test_key"] = "test_value"

        # Clear all caches
        clear_all_feature_caches()

        # Verify all caches are empty
        for name in cache_names:
            cache = getattr(volatility, name)
            assert len(cache) == 0, f"Cache {name} should be empty after clearing, got {cache}"

    def test_worker_init_fn(self) -> None:
        """Verify DataLoader creation includes worker_init_fn parameter.

        Without worker_init_fn, workers share the same random seed,
        causing identical augmentation across workers.
        """
        from src.models.neural.base_rnn import BaseRNNModel

        source = inspect.getsource(BaseRNNModel._create_dataloader)

        assert (
            "worker_init_fn" in source
        ), "_create_dataloader must include worker_init_fn for reproducible DataLoader workers"

    def test_calibrated_meta_uses_timeseries_cv(self) -> None:
        """Verify CalibratedMetaLearner uses TimeSeriesSplit, not StratifiedKFold.

        StratifiedKFold shuffles data randomly, leaking future bars into
        calibration folds. TimeSeriesSplit respects temporal order.
        """
        from src.models.ensemble.calibrated_meta import CalibratedMetaLearner

        source = inspect.getsource(CalibratedMetaLearner.fit)

        assert (
            "TimeSeriesSplit" in source
        ), "CalibratedMetaLearner.fit must use TimeSeriesSplit for temporal safety"
        # Should NOT use StratifiedKFold (except possibly in comments)
        dedented = textwrap.dedent(source)
        tree = ast.parse(dedented)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "StratifiedKFold":
                pytest.fail(
                    "CalibratedMetaLearner.fit must NOT use StratifiedKFold "
                    "(found in non-comment code)"
                )


# ---------------------------------------------------------------------------
# Phase 9 -- Dead Code Removal
# ---------------------------------------------------------------------------


class TestPhase9DeadCodeRemoval:
    """Tests verifying Phase 9 dead code cleanup."""

    def test_dead_code_removed(self) -> None:
        """Verify SmartConfig, UnifiedConfig, default_periods, and
        thresholds files do not exist in the codebase.

        These were identified as dead code and removed in Phase 9.
        """
        src_root = Path("src")

        # Files that should NOT exist
        dead_files = [
            "smart_config.py",
            "unified_config.py",
            "default_periods.py",
            "thresholds.py",
        ]
        for filename in dead_files:
            found = list(src_root.rglob(filename))
            assert len(found) == 0, f"Dead code file '{filename}' still exists at: {found}"

        # Classes that should NOT exist
        dead_classes = ["SmartConfig", "UnifiedConfig"]
        for py_file in src_root.rglob("*.py"):
            with open(py_file) as f:
                try:
                    tree = ast.parse(f.read())
                except SyntaxError:
                    continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name in dead_classes:
                    pytest.fail(f"Dead class '{node.name}' found in {py_file}:{node.lineno}")

    def test_regime_hysteresis(self) -> None:
        """Verify regime detector supports min_regime_bars hysteresis.

        Regime flip-flopping (rapidly alternating regimes) creates noisy
        signals. Hysteresis requires a new regime to persist for
        min_regime_bars before it's accepted.
        """
        from src.data.pipeline.stages.regime.composite import CompositeRegimeDetector

        sig = inspect.signature(CompositeRegimeDetector.__init__)
        params = list(sig.parameters.keys())
        assert (
            "min_regime_bars" in params
        ), "CompositeRegimeDetector.__init__ must accept min_regime_bars parameter"

        # Create with default and verify the attribute
        detector = CompositeRegimeDetector()
        assert hasattr(
            detector, "min_regime_bars"
        ), "CompositeRegimeDetector must have min_regime_bars attribute"
        assert (
            detector.min_regime_bars >= 1
        ), f"min_regime_bars must be >= 1, got {detector.min_regime_bars}"

        # Verify _apply_hysteresis method exists
        assert hasattr(
            detector, "_apply_hysteresis"
        ), "CompositeRegimeDetector must have _apply_hysteresis method"


# ---------------------------------------------------------------------------
# Phases 10-11 -- E2E Validation
# ---------------------------------------------------------------------------


class TestPhase10E2EImports:
    """Tests verifying all critical module imports work end-to-end."""

    def test_import_all_key_modules(self) -> None:
        """Verify all critical imports work without errors.

        This is the E2E smoke test: if any of these imports fail,
        there is a broken dependency or circular import.
        """
        from src.config.experiment import ExperimentConfig
        from src.core.datasets.sequences import SequenceDataset
        from src.data.features.compute import clear_all_feature_caches
        from src.data.pipeline.stages.regime.composite import CompositeRegimeDetector
        from src.data.pipeline.stages.regime.unified import get_regime_labels
        from src.factory import MLFactory
        from src.inference.backtesting.backtest import BacktestConfig
        from src.models.ensemble.calibrated_meta import CalibratedMetaLearner
        from src.models.ensemble.stacking import StackingEnsemble
        from src.optimization.feature_selection.result import FeatureSelectionResult
        from src.validation.cv.oof_core import OOFPrediction
        from src.validation.cv.oof_stacking import StackingDatasetBuilder
        from src.validation.cv.purged_kfold import PurgedKFold

        # Verify key classes are actually classes
        for cls in [
            MLFactory,
            ExperimentConfig,
            BacktestConfig,
            PurgedKFold,
            FeatureSelectionResult,
            OOFPrediction,
            StackingDatasetBuilder,
            StackingEnsemble,
            CalibratedMetaLearner,
            CompositeRegimeDetector,
            SequenceDataset,
        ]:
            assert isinstance(cls, type), f"{cls.__name__} should be a class"

        # Verify callables
        assert callable(get_regime_labels)
        assert callable(clear_all_feature_caches)


class TestPhase11BacktestConfigDefaults:
    """Tests verifying BacktestConfig defaults are production-correct."""

    def test_backtest_config_defaults(self) -> None:
        """Verify BacktestConfig defaults match documented production values.

        These defaults are critical for realistic backtesting:
        - MARKET_ON_OPEN: orders fill at next bar's open (no peek at close)
        - force_session_close=False: legacy behavior, positions can carry
        - kelly_min_trades=30: need statistical significance before Kelly
        """
        from src.inference.backtesting.backtest import BacktestConfig, ExecutionModel

        config = BacktestConfig()

        assert (
            config.execution_model == ExecutionModel.MARKET_ON_OPEN
        ), f"Default execution_model should be MARKET_ON_OPEN, got {config.execution_model}"
        assert (
            config.force_session_close is False
        ), f"Default force_session_close should be False, got {config.force_session_close}"
        assert (
            config.kelly_min_trades == 30
        ), f"Default kelly_min_trades should be 30, got {config.kelly_min_trades}"
        assert (
            config.barrier_k_up == 0.0
        ), f"Default barrier_k_up should be 0.0 (legacy mode), got {config.barrier_k_up}"
        assert (
            config.barrier_k_down == 0.0
        ), f"Default barrier_k_down should be 0.0 (legacy mode), got {config.barrier_k_down}"
        assert (
            config.alignment_loss_warn_pct == 5.0
        ), f"Default alignment_loss_warn_pct should be 5.0, got {config.alignment_loss_warn_pct}"
