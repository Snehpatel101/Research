"""
Regression tests for lookahead prevention.

Tests that features at time t only use data up to t-1.
This is a critical invariant for ML models on time series data.

The golden rule: Features at bar[t] must ONLY use data available at bar[t-1].
Any violation of this rule causes lookahead bias and inflated backtest results.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from stages.features.price_features import add_returns, add_price_ratios
from stages.features.momentum import add_rsi, add_macd, add_roc, add_stochastic
from stages.features.trend import add_adx, add_supertrend
from stages.features.moving_averages import add_sma, add_ema
from stages.features.volume import add_volume_features, add_vwap


def create_test_data(n_bars: int = 100, price_jump_at: int = -1) -> pd.DataFrame:
    """
    Create test OHLCV data with optional price jump at specified index.

    If price_jump_at is provided (negative index supported), that bar will
    have a dramatic 100% price increase. This allows testing whether features
    at that bar "see" the jump (lookahead) or not.
    """
    np.random.seed(42)

    # Base price series with small random walks
    base_price = 100.0
    prices = [base_price]
    for i in range(n_bars - 1):
        change = np.random.randn() * 0.5  # Small changes
        prices.append(prices[-1] + change)

    prices = np.array(prices)

    # Create OHLC from close prices
    df = pd.DataFrame({
        'datetime': pd.date_range('2024-01-01', periods=n_bars, freq='5min'),
        'open': prices - np.random.rand(n_bars) * 0.5,
        'high': prices + np.random.rand(n_bars) * 1.0,
        'low': prices - np.random.rand(n_bars) * 1.0,
        'close': prices.copy(),
        'volume': np.random.randint(500, 2000, n_bars)
    })

    # Ensure OHLC relationships are valid
    df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
    df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)

    # Apply price jump if specified
    if price_jump_at is not None:
        # Handle negative indexing
        actual_idx = price_jump_at if price_jump_at >= 0 else n_bars + price_jump_at
        # Double the price at the jump bar
        df.loc[actual_idx, 'close'] = df.loc[actual_idx, 'close'] * 2
        df.loc[actual_idx, 'high'] = df.loc[actual_idx, 'close'] * 1.01

    return df


class TestLookaheadInvariance:
    """Tests that verify no lookahead bias in features."""

    def test_returns_exclude_current_bar(self):
        """Return features at bar t should not use close[t]."""
        # Create data where last bar has dramatic price change
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=50, freq='5min'),
            'open': [100.0] * 50,
            'high': [102.0] * 50,
            'low': [98.0] * 50,
            'close': [100.0] * 49 + [200.0],  # Last bar jumps to 200
            'volume': [1000] * 50
        })

        feature_metadata = {}
        result = add_returns(df.copy(), feature_metadata)

        # At bar 49 (last bar), return_1 should be based on bar 48's calculation
        # NOT on bar 49's close (200). If lookahead exists, return would be ~100%
        last_return = result['return_1'].iloc[-1]

        # With proper shift, last return should be 0 (bar 47->48, both 100)
        # Without shift, it would be 1.0 (100% gain)
        assert abs(last_return) < 0.5 or pd.isna(last_return), \
            f"Lookahead detected: return_1 at last bar = {last_return}"

    def test_log_returns_exclude_current_bar(self):
        """Log return features at bar t should not use close[t]."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=50, freq='5min'),
            'open': [100.0] * 50,
            'high': [102.0] * 50,
            'low': [98.0] * 50,
            'close': [100.0] * 49 + [200.0],
            'volume': [1000] * 50
        })

        feature_metadata = {}
        result = add_returns(df.copy(), feature_metadata)

        last_log_return = result['log_return_1'].iloc[-1]

        # With proper shift, should be ~0. Without shift, would be ~0.69 (ln(2))
        assert abs(last_log_return) < 0.3 or pd.isna(last_log_return), \
            f"Lookahead detected: log_return_1 at last bar = {last_log_return}"

    def test_price_ratios_exclude_current_bar(self):
        """Price ratio features at bar t should not use OHLC[t]."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=50, freq='5min'),
            'open': [100.0] * 49 + [50.0],   # Last bar opens at 50
            'high': [102.0] * 49 + [250.0],  # Last bar high at 250
            'low': [98.0] * 49 + [45.0],     # Last bar low at 45
            'close': [100.0] * 49 + [200.0], # Last bar closes at 200
            'volume': [1000] * 50
        })

        feature_metadata = {}
        result = add_price_ratios(df.copy(), feature_metadata)

        # hl_ratio at last bar should be ~102/98 (bar 48), not 250/45 (bar 49)
        last_hl_ratio = result['hl_ratio'].iloc[-1]

        # With shift: hl_ratio = 102/98 ~= 1.04
        # Without shift: hl_ratio = 250/45 ~= 5.56
        assert last_hl_ratio < 2.0 or pd.isna(last_hl_ratio), \
            f"Lookahead detected: hl_ratio at last bar = {last_hl_ratio}"

    def test_rsi_excludes_current_bar(self):
        """RSI at bar t should not use close[t]."""
        # Create data with a spike at the last bar
        df = create_test_data(100, price_jump_at=-1)

        feature_metadata = {}
        result = add_rsi(df.copy(), feature_metadata)

        # RSI at last bar should not show extreme overbought from the jump
        last_rsi = result['rsi_14'].iloc[-1]
        second_last_rsi = result['rsi_14'].iloc[-2]

        # The last RSI should be similar to second-to-last (both see same data)
        # If lookahead exists, last RSI would spike to near 100
        if not pd.isna(last_rsi) and not pd.isna(second_last_rsi):
            diff = abs(last_rsi - second_last_rsi)
            assert diff < 30, \
                f"Lookahead suspected: RSI jumped from {second_last_rsi:.1f} to {last_rsi:.1f}"

    def test_sma_excludes_current_bar(self):
        """SMA at bar t should not include close[t]."""
        # Stable prices then big jump
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=50, freq='5min'),
            'open': [100.0] * 50,
            'high': [102.0] * 50,
            'low': [98.0] * 50,
            'close': [100.0] * 49 + [200.0],
            'volume': [1000] * 50
        })

        feature_metadata = {}
        result = add_sma(df.copy(), feature_metadata)

        # SMA_10 at last bar should be ~100 (avg of previous 10 closes)
        # Not affected by the 200 spike if properly shifted
        last_sma = result['sma_10'].iloc[-1]

        # With shift: SMA should be ~100
        # Without shift: SMA would be ~110 (avg includes 200)
        assert last_sma < 105 or pd.isna(last_sma), \
            f"Lookahead detected: sma_10 at last bar = {last_sma}"

    def test_ema_excludes_current_bar(self):
        """EMA at bar t should not include close[t]."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=50, freq='5min'),
            'open': [100.0] * 50,
            'high': [102.0] * 50,
            'low': [98.0] * 50,
            'close': [100.0] * 49 + [200.0],
            'volume': [1000] * 50
        })

        feature_metadata = {}
        result = add_ema(df.copy(), feature_metadata)

        # EMA_9 at last bar should be ~100
        last_ema = result['ema_9'].iloc[-1]

        # With shift: EMA should be ~100
        # Without shift: EMA would be significantly higher due to 200 influence
        assert last_ema < 110 or pd.isna(last_ema), \
            f"Lookahead detected: ema_9 at last bar = {last_ema}"

    def test_macd_excludes_current_bar(self):
        """MACD at bar t should not include close[t]."""
        df = create_test_data(100, price_jump_at=-1)

        feature_metadata = {}
        result = add_macd(df.copy(), feature_metadata)

        # MACD line at last bar should not spike due to the jump
        last_macd = result['macd_line'].iloc[-1]
        second_last_macd = result['macd_line'].iloc[-2]

        if not pd.isna(last_macd) and not pd.isna(second_last_macd):
            diff = abs(last_macd - second_last_macd)
            # MACD should not change dramatically between consecutive bars
            assert diff < 5, \
                f"Lookahead suspected: MACD jumped from {second_last_macd:.2f} to {last_macd:.2f}"

    def test_roc_excludes_current_bar(self):
        """ROC at bar t should not use close[t]."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=50, freq='5min'),
            'open': [100.0] * 50,
            'high': [102.0] * 50,
            'low': [98.0] * 50,
            'close': [100.0] * 49 + [200.0],
            'volume': [1000] * 50
        })

        feature_metadata = {}
        result = add_roc(df.copy(), feature_metadata)

        # ROC_5 at last bar should be ~0 (comparing bar 43 to bar 48, both 100)
        # Not ~100 (comparing bar 44 to bar 49's 200)
        last_roc = result['roc_5'].iloc[-1]

        assert abs(last_roc) < 50 or pd.isna(last_roc), \
            f"Lookahead detected: roc_5 at last bar = {last_roc}"

    def test_stochastic_excludes_current_bar(self):
        """Stochastic at bar t should not include OHLC[t]."""
        # Create data with price spike at end
        df = create_test_data(100, price_jump_at=-1)

        feature_metadata = {}
        result = add_stochastic(df.copy(), feature_metadata)

        # Stochastic K at last bar should not spike to 100 from the jump
        last_k = result['stoch_k'].iloc[-1]
        second_last_k = result['stoch_k'].iloc[-2]

        if not pd.isna(last_k) and not pd.isna(second_last_k):
            # Both should see similar market state
            diff = abs(last_k - second_last_k)
            assert diff < 40, \
                f"Lookahead suspected: Stoch K jumped from {second_last_k:.1f} to {last_k:.1f}"

    def test_adx_excludes_current_bar(self):
        """ADX at bar t should not include OHLC[t]."""
        df = create_test_data(100, price_jump_at=-1)

        feature_metadata = {}
        result = add_adx(df.copy(), feature_metadata)

        # ADX at last bar should not spike dramatically
        last_adx = result['adx_14'].iloc[-1]
        second_last_adx = result['adx_14'].iloc[-2]

        if not pd.isna(last_adx) and not pd.isna(second_last_adx):
            diff = abs(last_adx - second_last_adx)
            assert diff < 20, \
                f"Lookahead suspected: ADX jumped from {second_last_adx:.1f} to {last_adx:.1f}"

    def test_supertrend_excludes_current_bar(self):
        """Supertrend at bar t should not include OHLC[t]."""
        df = create_test_data(100, price_jump_at=-1)

        feature_metadata = {}
        result = add_supertrend(df.copy(), feature_metadata)

        # Supertrend direction at last bar should not flip due to the jump
        last_dir = result['supertrend_direction'].iloc[-1]
        second_last_dir = result['supertrend_direction'].iloc[-2]

        # Direction should be the same (both see the same market state)
        if not pd.isna(last_dir) and not pd.isna(second_last_dir):
            assert last_dir == second_last_dir, \
                f"Lookahead suspected: Supertrend direction changed at last bar"

    def test_volume_features_exclude_current_bar(self):
        """Volume features at bar t should not use volume[t]."""
        # Create data with volume spike at end
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=50, freq='5min'),
            'open': [100.0] * 50,
            'high': [102.0] * 50,
            'low': [98.0] * 50,
            'close': [100.0] * 50,
            'volume': [1000] * 49 + [100000]  # Massive volume spike
        })

        feature_metadata = {}
        result = add_volume_features(df.copy(), feature_metadata)

        # Volume ratio at last bar should be ~1.0 (based on previous bars)
        # Not ~100 (if it sees the spike)
        last_ratio = result['volume_ratio'].iloc[-1]

        assert last_ratio < 10 or pd.isna(last_ratio), \
            f"Lookahead detected: volume_ratio at last bar = {last_ratio}"

    def test_features_have_nan_at_first_bar(self):
        """First bar should have NaN for lagged features."""
        df = create_test_data(50)

        feature_metadata = {}
        result = add_returns(df.copy(), feature_metadata)

        # First bar should be NaN since there's no previous bar
        assert pd.isna(result['return_1'].iloc[0]), \
            "First bar should have NaN for return_1"
        assert pd.isna(result['log_return_1'].iloc[0]), \
            "First bar should have NaN for log_return_1"


class TestLookaheadEdgeCases:
    """Edge case tests for lookahead prevention."""

    def test_consecutive_jumps(self):
        """Test with multiple consecutive price jumps."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=10, freq='5min'),
            'open': [100.0] * 10,
            'high': [102.0] * 10,
            'low': [98.0] * 10,
            'close': [100.0, 100.0, 100.0, 100.0, 100.0,
                      100.0, 100.0, 150.0, 200.0, 250.0],
            'volume': [1000] * 10
        })

        feature_metadata = {}
        result = add_returns(df.copy(), feature_metadata)

        # Return at bar 7 should be 0 (sees bar 5->6, both 100)
        # Return at bar 8 should be 0.5 (sees bar 6->7, 100->150)
        # Return at bar 9 should be 0.33 (sees bar 7->8, 150->200)

        # The key test: bar 8's return should NOT see bar 8's close
        assert abs(result['return_1'].iloc[8] - 0.5) < 0.1 or pd.isna(result['return_1'].iloc[8]), \
            f"Bar 8 return should reflect 100->150, got {result['return_1'].iloc[8]}"

    def test_zero_volume_handling(self):
        """Test that zero volume doesn't cause issues."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=30, freq='5min'),
            'open': [100.0] * 30,
            'high': [102.0] * 30,
            'low': [98.0] * 30,
            'close': [100.0] * 30,
            'volume': [0] * 30  # Zero volume
        })

        feature_metadata = {}
        # Should not raise and should skip volume features
        result = add_volume_features(df.copy(), feature_metadata)

        # Check that volume features were skipped (no column added)
        assert 'obv' not in result.columns or result['obv'].isna().all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
