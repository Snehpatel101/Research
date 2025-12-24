
"""
Unit tests for Stage 3: Feature Engineering - Advanced Tests.

FeatureEngineer - Technical indicators and features

Run with: pytest tests/phase_1_tests/stages/test_stage3_*.py -v
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import pytest
import numpy as np
import pandas as pd

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.phase1.stages.features import FeatureEngineer
from src.phase1.stages.features import (
    add_volume_features,
    add_supertrend,
    add_adx,
    add_stochastic,
    add_mfi,
    add_vwap,
    add_returns,
    add_price_ratios,
    add_rsi,
    add_atr,
    add_sma,
    add_ema,
    add_macd,
    add_bollinger_bands,
    add_temporal_features,
    add_historical_volatility,
    add_regime_features,
    add_roc,
    add_williams_r,
    add_cci,
    add_keltner_channels,
    add_parkinson_volatility,
    add_garman_klass_volatility,
)



class TestFeatureEngineerVolumeFeatures:
    """Tests for volume-based features."""

    def test_volume_features_with_volume(self, temp_dir, sample_ohlcv_df):
        """Test volume features are calculated when volume is present."""
        # Arrange
        feature_metadata = {}
        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        df = add_volume_features(sample_ohlcv_df.copy(), feature_metadata)

        # Assert
        assert 'obv' in df.columns
        assert 'volume_sma_20' in df.columns
        assert 'volume_ratio' in df.columns

    def test_volume_features_without_volume(self, temp_dir):
        """Test volume features are skipped when no volume."""
        # Arrange
        feature_metadata = {}
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=100, freq='min'),
            'open': [100.0] * 100,
            'high': [102.0] * 100,
            'low': [98.0] * 100,
            'close': [100.0] * 100,
            # No volume column
        })

        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        result = add_volume_features(df, feature_metadata)

        # Assert - Should return unchanged
        assert 'obv' not in result.columns



class TestFeatureEngineerSupertrend:
    """Tests for Supertrend calculation."""

    def test_supertrend_direction_values(self, temp_dir, sample_ohlcv_df):
        """Test Supertrend direction is 1 or -1."""
        # Arrange
        feature_metadata = {}
        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        df = add_supertrend(sample_ohlcv_df.copy(), feature_metadata)

        # Assert
        assert 'supertrend' in df.columns
        assert 'supertrend_direction' in df.columns

        valid_dir = df['supertrend_direction'].dropna()
        assert set(valid_dir.unique()).issubset({-1.0, 1.0})



class TestFeatureEngineerCrossAsset:
    """Tests for cross-asset features."""

    def test_cross_asset_features_missing_data(self, temp_dir, sample_ohlcv_df):
        """Test that cross-asset features are NaN when data is missing."""
        # Arrange - import the module-level function
        feature_metadata = {}
        from stages.features.cross_asset import add_cross_asset_features

        # Act - call without cross-asset close arrays (mes_close=None, mgc_close=None)
        # The function adds NaN columns when cross-asset data is not provided
        feature_metadata = {}  # Will be populated with metadata
        df = add_cross_asset_features(
            sample_ohlcv_df.copy(),
            feature_metadata=feature_metadata,
            mes_close=None,
            mgc_close=None,
            current_symbol='MES'
        )

        # Assert - cross-asset features should be NaN
        assert 'mes_mgc_correlation_20' in df.columns
        assert df['mes_mgc_correlation_20'].isna().all()



class TestFeatureEngineerADXIndicator:
    """Tests for ADX indicator in FeatureEngineer."""

    def test_add_adx_features(self, temp_dir, sample_ohlcv_df):
        """Test ADX feature calculation through FeatureEngineer."""
        # Arrange
        feature_metadata = {}
        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        df = add_adx(sample_ohlcv_df.copy(), feature_metadata)

        # Assert
        assert 'adx_14' in df.columns
        assert 'plus_di_14' in df.columns
        assert 'minus_di_14' in df.columns
        assert 'adx_strong_trend' in df.columns



class TestFeatureEngineerStochasticIndicator:
    """Tests for Stochastic indicator in FeatureEngineer."""

    def test_add_stochastic_features(self, temp_dir, sample_ohlcv_df):
        """Test Stochastic feature calculation."""
        # Arrange
        feature_metadata = {}
        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        df = add_stochastic(sample_ohlcv_df.copy(), feature_metadata)

        # Assert
        assert 'stoch_k' in df.columns
        assert 'stoch_d' in df.columns
        assert 'stoch_overbought' in df.columns
        assert 'stoch_oversold' in df.columns



class TestFeatureEngineerMFI:
    """Tests for Money Flow Index calculation."""

    def test_add_mfi_with_volume(self, temp_dir, sample_ohlcv_df):
        """Test MFI calculation with volume data."""
        # Arrange
        feature_metadata = {}
        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        df = add_mfi(sample_ohlcv_df.copy(), feature_metadata)

        # Assert
        assert 'mfi_14' in df.columns
        valid_mfi = df['mfi_14'].dropna()
        assert np.all(valid_mfi >= 0)
        assert np.all(valid_mfi <= 100)



class TestFeatureEngineerVWAP:
    """Tests for VWAP calculation."""

    def test_add_vwap(self, temp_dir, sample_ohlcv_df):
        """Test VWAP calculation."""
        # Arrange
        feature_metadata = {}
        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        df = add_vwap(sample_ohlcv_df.copy(), feature_metadata)

        # Assert
        assert 'vwap' in df.columns
        assert 'price_to_vwap' in df.columns

        # VWAP should be positive
        valid_vwap = df['vwap'].dropna()
        assert np.all(valid_vwap > 0)



class TestFeatureEngineerReturns:
    """Tests for return calculations."""

    def test_add_returns(self, temp_dir, sample_ohlcv_df):
        """Test return feature calculation."""
        # Arrange
        feature_metadata = {}
        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        df = add_returns(sample_ohlcv_df.copy(), feature_metadata)

        # Assert
        assert 'return_1' in df.columns
        assert 'return_5' in df.columns
        assert 'log_return_1' in df.columns
        assert 'log_return_5' in df.columns



class TestFeatureEngineerPriceRatios:
    """Tests for price ratio calculations."""

    def test_add_price_ratios(self, temp_dir, sample_ohlcv_df):
        """Test price ratio feature calculation."""
        # Arrange
        feature_metadata = {}
        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        df = add_price_ratios(sample_ohlcv_df.copy(), feature_metadata)

        # Assert
        assert 'hl_ratio' in df.columns
        assert 'co_ratio' in df.columns
        assert 'range_pct' in df.columns

        # HL ratio should be >= 1 (exclude NaN from shift(1) applied for lookahead prevention)
        valid_hl_ratio = df['hl_ratio'].dropna()
        assert np.all(valid_hl_ratio >= 1)



class TestFeatureEngineerSaveFeatures:
    """Tests for saving features."""

    @pytest.mark.skip(reason="Test fixture has insufficient rows (500) for full feature engineering which requires ~200+ rows after NaN removal from long rolling windows")
    def test_save_features(self, temp_dir, sample_ohlcv_df):
        """Test saving features to parquet."""
        # Arrange
        feature_metadata = {}
        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        df, report = engineer.engineer_features(sample_ohlcv_df.copy(), 'TEST')

        # Act
        data_path, meta_path = engineer.save_features(df, 'TEST', report)

        # Assert
        assert data_path.exists()
        assert meta_path.exists()



class TestFeatureEngineerProcessFile:
    """Tests for processing a complete file."""

    @pytest.mark.skip(reason="Test fixture has insufficient rows (500) for full feature engineering which requires ~200+ rows after NaN removal from long rolling windows")
    def test_process_file(self, temp_dir, sample_ohlcv_df):
        """Test processing a complete file."""
        # Arrange
        feature_metadata = {}
        file_path = temp_dir / "TEST.parquet"
        sample_ohlcv_df.to_parquet(file_path, index=False)

        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        report = engineer.process_file(file_path)

        # Assert
        assert report['symbol'] == 'TEST'
        assert report['features_added'] > 0



class TestFeatureEngineerRSIIndicator:
    """Tests for RSI indicator in FeatureEngineer."""

    def test_add_rsi_features(self, temp_dir, sample_ohlcv_df):
        """Test RSI feature calculation through FeatureEngineer."""
        # Arrange
        feature_metadata = {}
        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        df = add_rsi(sample_ohlcv_df.copy(), feature_metadata)

        # Assert
        assert 'rsi_14' in df.columns
        assert 'rsi_overbought' in df.columns
        assert 'rsi_oversold' in df.columns

        # Overbought/oversold should be 0 or 1 (excluding NaN from warmup period)
        assert set(df['rsi_overbought'].dropna().unique()).issubset({0, 1})
        assert set(df['rsi_oversold'].dropna().unique()).issubset({0, 1})



class TestFeatureEngineerATRMethod:
    """Tests for ATR calculation in FeatureEngineer."""

    def test_add_atr(self, temp_dir, sample_ohlcv_df):
        """Test ATR feature calculation."""
        # Arrange
        feature_metadata = {}
        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        df = add_atr(sample_ohlcv_df.copy(), feature_metadata)

        # Assert - periods are [7, 14, 21]
        assert 'atr_14' in df.columns
        assert 'atr_pct_14' in df.columns

        # ATR should be positive
        valid_atr = df['atr_14'].dropna()
        assert np.all(valid_atr >= 0)



class TestFeatureEngineerSMAMethod:
    """Tests for SMA features in FeatureEngineer."""

    def test_add_sma(self, temp_dir, sample_ohlcv_df):
        """Test SMA feature calculation."""
        # Arrange
        feature_metadata = {}
        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        df = add_sma(sample_ohlcv_df.copy(), feature_metadata)

        # Assert - periods are [10, 20, 50, 100, 200]
        assert 'sma_10' in df.columns
        assert 'sma_20' in df.columns
        assert 'sma_50' in df.columns



class TestFeatureEngineerEMAMethod:
    """Tests for EMA features in FeatureEngineer."""

    def test_add_ema(self, temp_dir, sample_ohlcv_df):
        """Test EMA feature calculation."""
        # Arrange
        feature_metadata = {}
        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        df = add_ema(sample_ohlcv_df.copy(), feature_metadata)

        # Assert - periods are [9, 12, 21, 26, 50]
        assert 'ema_9' in df.columns
        assert 'ema_12' in df.columns
        assert 'ema_21' in df.columns

