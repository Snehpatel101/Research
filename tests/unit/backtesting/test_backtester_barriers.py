"""Backtester ATR, barrier stop/take-profit, per-contract sessions and factory barrier wiring."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# =============================================================================
# Phase 86: Barrier Alignment, Per-Contract Sessions, Per-Symbol Slippage
# =============================================================================


class TestComputeATR:
    """Test _compute_atr produces correct ATR values with known data."""

    def test_compute_atr_known_values(self):
        from src.inference.backtesting.backtest import BacktestConfig, Backtester

        # Build a 20-bar dataset with known high/low/close
        n = 20
        timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]
        high = np.array([110.0] * n)
        low = np.array([90.0] * n)
        close = np.array([100.0] * n)

        prices_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.full(n, 1000),
            }
        )
        preds_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "prediction": np.zeros(n, dtype=int),
                "confidence": np.ones(n),
            }
        )

        config = BacktestConfig(initial_equity=100000.0, enable_market_hours_filter=False)
        bt = Backtester(predictions=preds_df, prices=prices_df, config=config)
        data = bt._align_data()
        atr = bt._compute_atr(data, period=14)

        # True range for every bar (after the first):
        #   max(high-low, |high-prev_close|, |low-prev_close|)
        #   = max(20, 10, 10) = 20
        # First bar TR = high - low = 20 (no prev_close)
        # ATR(14) = rolling mean of TR with min_periods=1
        # After warm-up all values should be exactly 20.0
        assert len(atr) == n
        assert abs(atr.iloc[-1] - 20.0) < 1e-6


class TestBarrierStopTP:
    """Test barrier stop-loss / take-profit computation in _open_position."""

    def _make_backtester(self, k_up: float, k_down: float):
        from src.inference.backtesting.backtest import BacktestConfig, Backtester

        n = 5
        timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]
        prices_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": [100.0] * n,
                "high": [105.0] * n,
                "low": [95.0] * n,
                "close": [100.0] * n,
                "volume": [1000] * n,
            }
        )
        preds_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "prediction": np.zeros(n, dtype=int),
                "confidence": np.ones(n),
            }
        )
        config = BacktestConfig(
            initial_equity=100000.0,
            enable_market_hours_filter=False,
            barrier_k_up=k_up,
            barrier_k_down=k_down,
        )
        return Backtester(predictions=preds_df, prices=prices_df, config=config)

    def test_barrier_stop_tp_long(self):
        """LONG: stop = price - k_down*ATR, TP = price + k_up*ATR."""
        bt = self._make_backtester(k_up=3.0, k_down=2.0)
        price = 5000.0
        atr = 10.0
        bt._open_position(
            direction=1,
            price=price,
            timestamp=datetime(2024, 1, 1),
            bar_idx=0,
            atr=atr,
        )
        pos = bt._current_position
        assert pos is not None
        assert abs(pos.stop_loss - (price - 2.0 * atr)) < 1e-6  # 5000 - 20 = 4980
        assert abs(pos.take_profit - (price + 3.0 * atr)) < 1e-6  # 5000 + 30 = 5030

    def test_barrier_stop_tp_short(self):
        """SHORT: stop = price + k_up*ATR, TP = price - k_down*ATR."""
        bt = self._make_backtester(k_up=3.0, k_down=2.0)
        price = 5000.0
        atr = 10.0
        bt._open_position(
            direction=-1,
            price=price,
            timestamp=datetime(2024, 1, 1),
            bar_idx=0,
            atr=atr,
        )
        pos = bt._current_position
        assert pos is not None
        assert abs(pos.stop_loss - (price + 3.0 * atr)) < 1e-6  # 5000 + 30 = 5030
        assert abs(pos.take_profit - (price - 2.0 * atr)) < 1e-6  # 5000 - 20 = 4980

    def test_asymmetric_barriers(self):
        """MES H20 asymmetric: k_up=3.0, k_down=2.1."""
        bt = self._make_backtester(k_up=3.0, k_down=2.1)
        price = 4500.0
        atr = 5.0
        bt._open_position(
            direction=1,
            price=price,
            timestamp=datetime(2024, 1, 1),
            bar_idx=0,
            atr=atr,
        )
        pos = bt._current_position
        assert pos is not None
        # Long: stop = 4500 - 2.1*5 = 4489.5, TP = 4500 + 3.0*5 = 4515
        assert abs(pos.stop_loss - 4489.5) < 1e-6
        assert abs(pos.take_profit - 4515.0) < 1e-6

    def test_legacy_mode_no_barriers(self):
        """k_up=0, k_down=0 -> legacy 2% stop, no take profit."""
        bt = self._make_backtester(k_up=0.0, k_down=0.0)
        price = 5000.0
        bt._open_position(
            direction=1,
            price=price,
            timestamp=datetime(2024, 1, 1),
            bar_idx=0,
            atr=10.0,  # should be ignored
        )
        pos = bt._current_position
        assert pos is not None
        # Legacy long: stop = price * (1 - 1 * 0.02) = 4900
        assert abs(pos.stop_loss - 4900.0) < 1e-6
        assert pos.take_profit is None


class TestPerContractSessionTimes:
    """Test CONTRACT_SESSION_TIMES per-symbol session windows."""

    def test_per_contract_session_times(self):
        from datetime import time as dt_time

        from src.inference.backtesting.execution import CONTRACT_SESSION_TIMES

        # MGC = COMEX primary session: 8:20 - 13:30 ET
        mgc_start, mgc_end = CONTRACT_SESSION_TIMES["MGC"]
        assert mgc_start == dt_time(8, 20)
        assert mgc_end == dt_time(13, 30)

        # MES = NYSE cash session: 9:30 - 16:00 ET
        mes_start, mes_end = CONTRACT_SESSION_TIMES["MES"]
        assert mes_start == dt_time(9, 30)
        assert mes_end == dt_time(16, 0)

        # MNQ shares MES session
        mnq_start, mnq_end = CONTRACT_SESSION_TIMES["MNQ"]
        assert mnq_start == dt_time(9, 30)
        assert mnq_end == dt_time(16, 0)

    def test_market_hours_filter_uses_per_contract_times(self):
        """MarketHoursFilter resolves session from CONTRACT_SESSION_TIMES."""
        from datetime import time as dt_time

        from src.inference.backtesting.execution import MarketHoursFilter

        filt_mgc = MarketHoursFilter(contract="MGC", enable_market_hours_filter=True)
        assert filt_mgc._session_start == dt_time(8, 20)
        assert filt_mgc._session_end == dt_time(13, 30)

        filt_mes = MarketHoursFilter(contract="MES", enable_market_hours_filter=True)
        assert filt_mes._session_start == dt_time(9, 30)
        assert filt_mes._session_end == dt_time(16, 0)


class TestBarrierConfigFactoryWiring:
    """Test get_barrier_params feeds into BacktestConfig round-trip."""

    def test_barrier_config_factory_wiring(self):
        from src.data.pipeline.config.barriers_config import get_barrier_params
        from src.inference.backtesting.backtest import BacktestConfig

        params = get_barrier_params("MES", 20)
        config = BacktestConfig(
            barrier_k_up=params["k_up"],
            barrier_k_down=params["k_down"],
            max_holding_period=params["max_bars"],
        )

        # MES H20: k_up=3.0, k_down=2.1, max_bars=50
        assert config.barrier_k_up == 3.0
        assert config.barrier_k_down == 2.1
        assert config.max_holding_period == 50

    def test_barrier_params_propagate_to_position(self):
        """Barrier params in config flow through to Position stop/TP."""
        from src.data.pipeline.config.barriers_config import get_barrier_params
        from src.inference.backtesting.backtest import BacktestConfig, Backtester

        params = get_barrier_params("MES", 20)
        n = 5
        timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]
        prices_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": [4500.0] * n,
                "high": [4510.0] * n,
                "low": [4490.0] * n,
                "close": [4500.0] * n,
                "volume": [1000] * n,
            }
        )
        preds_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "prediction": np.zeros(n, dtype=int),
                "confidence": np.ones(n),
            }
        )
        config = BacktestConfig(
            initial_equity=100000.0,
            enable_market_hours_filter=False,
            barrier_k_up=params["k_up"],
            barrier_k_down=params["k_down"],
        )
        bt = Backtester(predictions=preds_df, prices=prices_df, config=config)
        price = 4500.0
        atr = 8.0
        bt._open_position(
            direction=1,
            price=price,
            timestamp=datetime(2024, 1, 1),
            bar_idx=0,
            atr=atr,
        )
        pos = bt._current_position
        assert pos is not None
        # k_up=3.0, k_down=2.1 from get_barrier_params
        assert abs(pos.stop_loss - (4500.0 - 2.1 * 8.0)) < 1e-6
        assert abs(pos.take_profit - (4500.0 + 3.0 * 8.0)) < 1e-6
