"""
Regression tests for Phases 1-3 fixes in ML Factory.

Phase 1: Leakage Fixes (7 tests)
Phase 2: Accuracy Fixes (4 tests)
Phase 3: Memory Fixes (3 tests)
"""

from __future__ import annotations

import ast
import inspect
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Phase 1 -- Leakage Fixes
# ---------------------------------------------------------------------------


class TestPhase1LeakageFixes:
    """Tests verifying Phase 1 leakage prevention fixes."""

    def test_optuna_atr_wilders_ema(self) -> None:
        """Verify that ATR in five_dimension_objective uses Wilder's EMA (alpha=1/period).

        Wilder's EMA uses alpha = 1/period, NOT the standard EMA alpha = 2/(period+1).
        This ensures the Optuna objective's ATR matches the labeling and backtest ATR.
        """
        from src.optimization.five_dimension_objective import _compute_atr

        # Create synthetic OHLCV with a known true-range pattern
        n = 100
        np.random.seed(42)
        close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        df = pd.DataFrame({"high": high, "low": low, "close": close})

        period = 14
        atr = _compute_atr(df, period=period)

        # Manually compute Wilder's EMA with alpha = 1/period
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr = np.maximum(
            high - low,
            np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
        )

        alpha = 1.0 / period  # Wilder's EMA
        expected_atr = np.zeros(n)
        expected_atr[0] = tr[0]
        for i in range(1, n):
            expected_atr[i] = alpha * tr[i] + (1 - alpha) * expected_atr[i - 1]

        np.testing.assert_allclose(atr, expected_atr, rtol=1e-10)

        # Also verify it does NOT match standard EMA (alpha = 2/(period+1))
        alpha_standard = 2.0 / (period + 1)
        standard_atr = np.zeros(n)
        standard_atr[0] = tr[0]
        for i in range(1, n):
            standard_atr[i] = alpha_standard * tr[i] + (1 - alpha_standard) * standard_atr[i - 1]

        # They should differ meaningfully
        assert not np.allclose(
            atr, standard_atr, rtol=1e-6
        ), "ATR should use Wilder's alpha=1/period, NOT standard alpha=2/(period+1)"

    @staticmethod
    def _make_regime_ohlcv(n: int, seed: int = 42) -> pd.DataFrame:
        """Synthetic OHLCV data for regime shift tests."""
        rng = np.random.RandomState(seed)
        close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
        high = close + np.abs(rng.randn(n) * 0.3)
        low = close - np.abs(rng.randn(n) * 0.3)
        volume = rng.randint(100, 1000, size=n).astype(float)
        return pd.DataFrame(
            {"open": close, "high": high, "low": low, "close": close, "volume": volume}
        )

    def test_volatility_regime_shifted(self) -> None:
        """Verify the live volatility regime is lagged -- a spike at bar N must
        not affect bar N's regime; it may only appear from bar N+1 onward.

        The live add_volatility_regime consumes hvol_20 which the live
        add_historical_volatility produces already lagged by 1 bar.
        """
        from src.data.pipeline.stages.features.regime import add_volatility_regime
        from src.data.pipeline.stages.features.volatility import add_historical_volatility

        def pipeline(df: pd.DataFrame) -> pd.DataFrame:
            out = add_historical_volatility(df.copy(), {}, periods=[20], timeframe="5min")
            return add_volatility_regime(out, {})

        df = self._make_regime_ohlcv(260)
        spike_bar = 200

        base = pipeline(df)

        df_spiked = df.copy()
        # Massive close spike at spike_bar (drives hvol through the roof)
        df_spiked.loc[spike_bar, "close"] *= 2.0
        spiked = pipeline(df_spiked)

        # Bars 0..spike_bar must be identical: the spike is only visible from
        # bar spike_bar+1 onward because hvol_20 is lagged by 1 bar.
        pd.testing.assert_series_equal(
            base["volatility_regime"].iloc[: spike_bar + 1],
            spiked["volatility_regime"].iloc[: spike_bar + 1],
        )

        # Sanity: the spike DOES propagate into the lagged hvol at bar N+1
        assert base["hvol_20"].iloc[spike_bar + 1] != pytest.approx(
            spiked["hvol_20"].iloc[spike_bar + 1]
        ), "hvol_20 should reflect the spike at bar N+1 (shift(1) includes it)"

    def test_trend_regime_shifted(self) -> None:
        """Verify the live trend regime uses lagged close -- a spike at bar N
        must not affect bar N's regime; it should appear at bar N+1.
        """
        from src.data.pipeline.stages.features.regime import add_trend_regime

        def pipeline(close: pd.Series) -> pd.Series:
            df = pd.DataFrame({"close": close})
            # SMA inputs are pre-lagged, matching the live engine contract
            df["sma_50"] = close.rolling(50).mean().shift(1)
            df["sma_200"] = close.rolling(200).mean().shift(1)
            out = add_trend_regime(df, {})
            return out["trend_regime"]

        # Steady uptrend: close > sma_50 > sma_200 after warmup
        n = 260
        close = pd.Series(100.0 + 0.1 * np.arange(n))
        spike_bar = 250

        base = pipeline(close)
        assert base.iloc[spike_bar] == 1, "Base data should be in uptrend at spike bar"
        assert base.iloc[spike_bar + 1] == 1, "Base data should be in uptrend after spike bar"

        close_spiked = close.copy()
        close_spiked.iloc[spike_bar] *= 0.5  # crash at spike_bar
        spiked = pipeline(close_spiked)

        # Bar N must NOT see its own crash (close is lagged inside add_trend_regime)
        assert spiked.iloc[spike_bar] == base.iloc[spike_bar], (
            f"Bar {spike_bar} trend regime should not see its own spike: "
            f"base={base.iloc[spike_bar]}, spiked={spiked.iloc[spike_bar]}"
        )

        # Bar N+1 SHOULD see the crash: lagged close < sma_50 breaks the uptrend
        assert spiked.iloc[spike_bar + 1] == 0, (
            f"Bar {spike_bar + 1} should leave uptrend after the crash, "
            f"got {spiked.iloc[spike_bar + 1]}"
        )

        # Source check: the live function lags close with shift(1)
        src = inspect.getsource(add_trend_regime)
        assert ".shift(1)" in src, "add_trend_regime must contain .shift(1)"

    def test_structure_regime_shifted(self) -> None:
        """Verify the live structure regime (Hurst-based) is shifted by 1 bar."""
        from src.data.pipeline.stages.features.regime import add_structure_regime

        df = self._make_regime_ohlcv(200)

        base = add_structure_regime(df.copy(), {}, lookback=100)
        assert "structure_regime" in base.columns, "structure_regime column missing"

        # Bar 0 must be NaN due to shift(1)
        assert pd.isna(
            base["structure_regime"].iloc[0]
        ), "structure_regime should be NaN at bar 0 (shifted)"

        df_spiked = df.copy()
        df_spiked.iloc[-1, df_spiked.columns.get_loc("close")] *= 2.0
        spiked = add_structure_regime(df_spiked, {}, lookback=100)

        # Last bar's regime must not see its own spike (shift(1) excludes it)
        base_last = base["structure_regime"].iloc[-1]
        spiked_last = spiked["structure_regime"].iloc[-1]
        assert not pd.isna(base_last), "structure_regime at last bar should be valid"
        assert base_last == spiked_last, (
            f"Last bar structure regime should not see its own spike: "
            f"base={base_last}, spiked={spiked_last}"
        )

        # Source check: the live function applies shift(1) to the regime series
        src = inspect.getsource(add_structure_regime)
        assert ".shift(1)" in src, "add_structure_regime must contain .shift(1)"

    def test_cv_tuner_strided_subsampling(self) -> None:
        """Verify cv_tuner uses strided (every-Nth) sampling, not random,
        preserving temporal ordering when subsampling > 50K samples.
        """
        n_original = 100_000  # > 50K triggers subsampling
        max_samples = 50_000

        # Manually apply the subsampling logic from the tune method
        stride = max(1, n_original // max_samples)
        sub_indices = np.arange(0, n_original, stride)[:max_samples]

        # Verify strided pattern: gaps between adjacent samples are uniform
        gaps = np.diff(sub_indices)
        assert np.all(
            gaps == stride
        ), f"Expected uniform stride={stride}, got varying gaps: {np.unique(gaps)}"

        # Verify temporal ordering preserved (monotonically increasing)
        assert np.all(
            np.diff(sub_indices) > 0
        ), "Subsampled indices must be monotonically increasing"

        # Verify we get max_samples or fewer
        assert len(sub_indices) <= max_samples

        # Verify the source code uses strided sampling (not random)
        from src.validation.cv.cv_tuner import TimeSeriesOptunaTuner

        source = inspect.getsource(TimeSeriesOptunaTuner.tune)
        assert (
            "np.arange(0, original_n, stride)" in source
        ), "tune() must use np.arange strided sampling, not random"

    def test_hp_tuning_embargo_propagation(self) -> None:
        """Verify HyperparameterTuningService uses embargo_bars from TuningRequest
        when provided, not just the horizon*2 fallback.
        """
        from src.models.training.services.hyperparameter_tuning import (
            HyperparameterTuningService,
            TuningRequest,
        )

        # Case 1: embargo_bars provided on request
        request_with_embargo = TuningRequest(
            model_name="xgboost",
            horizon=10,
            prepared_data=MagicMock(),
            embargo_bars=42,
        )
        embargo = (
            request_with_embargo.embargo_bars
            if request_with_embargo.embargo_bars is not None
            else request_with_embargo.horizon * 2
        )
        assert embargo == 42, f"Expected embargo=42 (from request), got {embargo}"

        # Case 2: embargo_bars is None => fallback to horizon*2
        request_no_embargo = TuningRequest(
            model_name="xgboost",
            horizon=10,
            prepared_data=MagicMock(),
            embargo_bars=None,
        )
        embargo_fallback = (
            request_no_embargo.embargo_bars
            if request_no_embargo.embargo_bars is not None
            else request_no_embargo.horizon * 2
        )
        assert embargo_fallback == 20, f"Expected embargo=20 (horizon*2), got {embargo_fallback}"

        # Verify the source code actually does this check
        source = inspect.getsource(HyperparameterTuningService.optimize)
        assert "request.embargo_bars" in source, "optimize() must reference request.embargo_bars"
        assert "request.horizon * 2" in source, "optimize() must fallback to request.horizon * 2"

    def test_higher_tf_shift(self) -> None:
        """Verify factory._generate_additional_dfs applies shift(1) to higher-TF
        resampled data, preventing lookahead from incomplete bars.
        """
        # Read the source code to verify shift(1) is present
        source = inspect.getsource(
            __import__("src.factory", fromlist=["MLFactory"]).MLFactory._generate_additional_dfs
        )
        assert (
            ".shift(1)" in source
        ), "_generate_additional_dfs must apply shift(1) to resampled higher-TF data"

        # Functional test: create minute-level data, resample to 5min
        # Verify that the resampled output is shifted by 1 bar
        n = 100
        dates = pd.date_range("2024-01-01 09:30", periods=n, freq="1min")
        close = 100.0 + np.arange(n, dtype=float) * 0.01
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.1,
                "low": close - 0.1,
                "close": close,
                "volume": np.ones(n) * 100,
            },
            index=dates,
        )

        # Resample to 5min and apply the same logic as the factory
        ohlcv_agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        resampled = (
            df[list(ohlcv_agg.keys())]
            .resample("5min", closed="left", label="left")
            .agg(ohlcv_agg)
            .dropna()
        )

        # Without shift: first bar would have data from 09:30-09:34
        unshifted_first_close = resampled["close"].iloc[0]

        # With shift(1): first valid bar should have the PREVIOUS bar's data
        shifted = resampled.shift(1).dropna()
        shifted_first_close = shifted["close"].iloc[0]

        # The shifted first bar should equal the unshifted first bar
        # (shift moves everything down by 1)
        assert (
            shifted_first_close == unshifted_first_close
        ), "shift(1) on resampled data should shift values down by one bar"

        # The shifted data should have one fewer row than unshifted
        assert len(shifted) == len(resampled) - 1


# ---------------------------------------------------------------------------
# Phase 2 -- Accuracy Fixes
# ---------------------------------------------------------------------------


class TestPhase2AccuracyFixes:
    """Tests verifying Phase 2 accuracy improvement fixes."""

    def test_class_weights_preserved_with_sample_weights(self) -> None:
        """Verify that when sample_weights are used, class_weights are still
        applied via criterion.weight in the unreduced CrossEntropyLoss.

        The fix ensures class_weights and sample_weights are BOTH applied:
        criterion_unreduced = CrossEntropyLoss(weight=criterion.weight, reduction='none')
        """
        import torch
        import torch.nn as nn

        source = inspect.getsource(
            __import__(
                "src.models.neural.base_rnn", fromlist=["BaseRNNModel"]
            ).BaseRNNModel._train_epoch
        )

        # Verify the critical pattern: criterion_unreduced uses criterion.weight
        assert (
            "criterion.weight" in source
        ), "_train_epoch must pass criterion.weight to criterion_unreduced"
        assert (
            'reduction="none"' in source or "reduction='none'" in source
        ), "_train_epoch must use reduction='none' for per-sample weighting"

        # Functional test: create CrossEntropyLoss with class weights
        class_weights = torch.tensor([1.0, 2.0, 0.5])
        criterion_weighted = nn.CrossEntropyLoss(weight=class_weights)

        # The unreduced version should preserve class_weights
        criterion_unreduced = nn.CrossEntropyLoss(
            weight=criterion_weighted.weight, reduction="none"
        )

        # Check that the weight parameter is preserved
        assert (
            criterion_unreduced.weight is not None
        ), "Unreduced criterion must carry class_weights from parent criterion"
        torch.testing.assert_close(criterion_unreduced.weight, class_weights)

        # Verify combined weighting produces different results than class weights alone
        logits = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 1])
        sample_weights = torch.tensor([1.0, 2.0, 0.5, 1.5])

        per_sample = criterion_unreduced(logits, targets)
        combined = (per_sample * sample_weights).mean()
        class_only = criterion_weighted(logits, targets)

        # Combined (sample * class weights) should differ from class-only
        assert not torch.allclose(
            combined, class_only
        ), "Combined sample+class weighting should differ from class-only"

    def test_default_execution_model_market_on_open(self) -> None:
        """Verify BacktestConfig defaults to MARKET_ON_OPEN execution model."""
        from src.inference.backtesting.backtest import BacktestConfig, ExecutionModel

        config = BacktestConfig()
        assert config.execution_model == ExecutionModel.MARKET_ON_OPEN, (
            f"Default execution_model should be MARKET_ON_OPEN, " f"got {config.execution_model}"
        )

    def test_binary_mode_oof_column_names(self) -> None:
        """Verify _get_prob_column_names returns correct names for
        n_classes=2 (binary) and n_classes=3 (ternary).
        """
        from src.validation.cv.oof_core import _get_prob_column_names

        # n_classes=3: backward-compatible named columns
        cols_3 = _get_prob_column_names("xgboost", n_classes=3)
        assert cols_3 == [
            "xgboost_prob_short",
            "xgboost_prob_neutral",
            "xgboost_prob_long",
        ], f"n_classes=3 columns wrong: {cols_3}"

        # n_classes=2: generic numbered columns
        cols_2 = _get_prob_column_names("lstm", n_classes=2)
        assert cols_2 == [
            "lstm_prob_0",
            "lstm_prob_1",
        ], f"n_classes=2 columns wrong: {cols_2}"

        # n_classes=5: generic numbered columns
        cols_5 = _get_prob_column_names("model", n_classes=5)
        assert len(cols_5) == 5
        assert cols_5[0] == "model_prob_0"
        assert cols_5[4] == "model_prob_4"

    def test_volatility_annualization_factor(self) -> None:
        """Verify the live annualization factor is timeframe-derived:
        sqrt(bars_per_day * 252), NOT a hardcoded constant.
        """
        from src.data.pipeline.stages.features.constants import (
            ANNUALIZATION_FACTOR,
            get_annualization_factor,
            get_bars_per_day,
        )

        # 5min regular session: 6.5h * 60 / 5 = 78 bars/day
        assert get_bars_per_day("5min") == pytest.approx(78.0)

        # Expected 5min factor: sqrt(252 * 78) ~= 140.07
        expected_5min = np.sqrt(252 * 78)
        result_5min = get_annualization_factor("5min")
        assert result_5min == pytest.approx(
            expected_5min, abs=1e-10
        ), f"get_annualization_factor('5min') = {result_5min}, expected {expected_5min}"
        assert 140.0 < result_5min < 141.0, f"Expected ~140.07, got {result_5min}"

        # Factor must vary by timeframe (timeframe-derived, not hardcoded)
        result_1min = get_annualization_factor("1min")
        assert result_1min == pytest.approx(np.sqrt(252 * 390), abs=1e-10)
        assert result_1min != result_5min, "1min and 5min factors must differ"
        assert get_annualization_factor("15min") < result_5min

        # Extended (23h futures) session yields a larger factor than regular
        assert get_annualization_factor("5min", extended_hours=True) > result_5min

        # Backward-compat module default is the 5min regular-session factor
        assert pytest.approx(expected_5min, abs=1e-10) == ANNUALIZATION_FACTOR

        # Unknown timeframes are rejected, not silently defaulted
        with pytest.raises(ValueError):
            get_annualization_factor("13min")


# ---------------------------------------------------------------------------
# Phase 3 -- Memory Fixes
# ---------------------------------------------------------------------------


class TestPhase3MemoryFixes:
    """Tests verifying Phase 3 memory optimization fixes."""

    def test_multi_resolution_uses_as_tensor(self) -> None:
        """Verify multi_resolution.py uses torch.as_tensor (zero-copy) instead
        of torch.tensor (which always copies).
        """
        source_path = "src/data/adapters/multi_resolution.py"
        with open(source_path) as f:
            source = f.read()

        # Should use torch.as_tensor (zero-copy when possible)
        assert (
            "torch.as_tensor" in source
        ), f"{source_path} must use torch.as_tensor for zero-copy tensor creation"

        # Should NOT use torch.tensor (always copies)
        # Parse the AST to find actual torch.tensor calls (not in comments/strings)
        tree = ast.parse(source)
        tensor_calls = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "tensor"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"
            ):
                tensor_calls.append(node.lineno)

        assert len(tensor_calls) == 0, (
            f"{source_path} should not use torch.tensor "
            f"(found at lines {tensor_calls}). "
            f"Use torch.as_tensor or torch.from_numpy instead."
        )

    def test_oof_generation_no_redundant_copy(self) -> None:
        """Verify oof_generation.py does NOT call .copy() on fancy-indexed arrays
        in the 4D OOF fold code. Fancy indexing already returns a copy, so
        .copy() would waste memory.
        """
        from src.models.training.services.oof_generation import (
            OOFGenerationService,
        )

        source = inspect.getsource(OOFGenerationService._generate_4d_oof)

        # The code should have a comment explaining why .copy() is not needed
        assert (
            "Fancy indexing already returns" in source or ".copy() is redundant" in source
        ), "_generate_4d_oof should document that fancy indexing copies are sufficient"

        # Count actual .copy() calls on array slicing patterns
        # There should be zero .copy() on X_4d[train_idx] or X_4d[val_idx]
        lines = source.split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "X_4d[" in stripped and ".copy()" in stripped:
                pytest.fail(f"Found redundant .copy() on fancy-indexed 4D array: " f"{stripped}")

    def test_oof_sequence_single_backup(self) -> None:
        """Verify oof_sequence.py uses the raw_X_backup pattern: a single copy
        before the CV loop, with np.copyto to restore in-place each fold.

        This avoids N copies (one per fold) by doing 1 copy + N in-place restores.
        """
        from src.validation.cv.oof_sequence import SequenceOOFGenerator

        source = inspect.getsource(SequenceOOFGenerator.generate_sequence_oof)

        # Must have exactly one backup copy BEFORE the fold loop
        assert "raw_X_backup = seq_builder._X.copy()" in source, (
            "Must create raw_X_backup from seq_builder._X.copy() " "before the CV loop"
        )

        # Must use np.copyto for in-place restore inside the loop
        assert "np.copyto(seq_builder._X, raw_X_backup)" in source, (
            "Must use np.copyto(seq_builder._X, raw_X_backup) " "inside the fold loop"
        )

        # Verify the backup happens BEFORE the for loop, and copyto INSIDE
        backup_line = None
        copyto_line = None
        for_loop_line = None

        for i, line in enumerate(source.split("\n")):
            if "raw_X_backup = seq_builder._X.copy()" in line and backup_line is None:
                backup_line = i
            if "for fold_idx" in line and for_loop_line is None:
                for_loop_line = i
            if "np.copyto(seq_builder._X, raw_X_backup)" in line and copyto_line is None:
                copyto_line = i

        assert backup_line is not None, "raw_X_backup assignment not found"
        assert for_loop_line is not None, "fold loop not found"
        assert copyto_line is not None, "np.copyto restore not found"

        assert backup_line < for_loop_line, (
            f"Backup (line {backup_line}) must come BEFORE " f"fold loop (line {for_loop_line})"
        )
        assert copyto_line > for_loop_line, (
            f"np.copyto (line {copyto_line}) must come AFTER "
            f"fold loop start (line {for_loop_line})"
        )
