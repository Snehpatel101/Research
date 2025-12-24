"""
Unit tests for Stage 7.5: Feature Scaling with Train-Only Fitting.

Tests critical aspects:
- Train-only fitting prevents data leakage
- Feature column identification
- Multiple scaler types
- Outlier clipping
- Edge cases (NaN, single value, etc.)
- Save/load persistence
- Output file creation

Run with: pytest tests/phase_1_tests/stages/test_stage7_5_scaling.py -v
"""

import sys
from pathlib import Path
import json
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.phase1.stages.scaling import identify_feature_columns, scale_splits, run_scaling_stage
from src.phase1.stages.scaling import FeatureScaler


@pytest.fixture
def sample_data_with_features():
    """Create sample data with OHLCV, features, and labels."""
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        'datetime': pd.date_range('2024-01-01', periods=n, freq='5min'),
        'symbol': 'MES',
        'open': 4500 + np.cumsum(np.random.randn(n) * 0.5),
        'high': 4505 + np.cumsum(np.random.randn(n) * 0.5),
        'low': 4495 + np.cumsum(np.random.randn(n) * 0.5),
        'close': 4500 + np.cumsum(np.random.randn(n) * 0.5),
        'volume': np.random.randint(1000, 10000, n).astype(float),
        'rsi_14': 50.0 + np.random.randn(n) * 10,
        'macd_line': np.random.randn(n) * 0.5,
        'sma_20': 4500 + np.cumsum(np.random.randn(n) * 0.4),
        'bb_width': np.random.uniform(0.02, 0.05, n),
        'volume_ratio': np.random.uniform(0.5, 2.0, n),
        'price_change_pct': np.random.randn(n) * 0.01,
        'label_h5': np.random.choice([1, -1, 0], size=n, p=[0.3, 0.3, 0.4]),
        'label_h20': np.random.choice([1, -1, 0], size=n, p=[0.3, 0.3, 0.4]),
        'bars_to_hit_h5': np.random.randint(1, 10, n),
        'mae_h5': -np.abs(np.random.randn(n) * 0.01),
        'quality_h5': np.random.uniform(0.3, 0.9, n),
        'sample_weight_h5': np.random.choice([0.5, 1.0, 1.5], n),
    })
    df['macd_signal'] = df['macd_line'].rolling(9).mean().fillna(0)
    return df


@pytest.fixture
def sample_splits(sample_data_with_features):
    """Create train/val/test split indices."""
    df = sample_data_with_features
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return {
        'data': df,
        'train_idx': np.arange(0, train_end),
        'val_idx': np.arange(train_end, val_end),
        'test_idx': np.arange(val_end, n),
    }


# =============================================================================
# TESTS: identify_feature_columns
# =============================================================================

class TestIdentifyFeatureColumns:
    """Tests for feature column identification."""

    def test_excludes_ohlcv_and_metadata(self, sample_data_with_features):
        """Test that OHLCV and metadata columns are excluded."""
        feature_cols = identify_feature_columns(sample_data_with_features)
        excluded = ['open', 'high', 'low', 'close', 'volume', 'datetime', 'symbol']
        for col in excluded:
            assert col not in feature_cols, f"{col} should be excluded"

    def test_excludes_label_columns(self, sample_data_with_features):
        """Test that label columns are excluded."""
        feature_cols = identify_feature_columns(sample_data_with_features)
        label_prefixes = ['label_', 'bars_to_hit_', 'mae_', 'quality_', 'sample_weight_']
        for col in feature_cols:
            assert not any(col.startswith(p) for p in label_prefixes)

    def test_includes_features_and_numeric_only(self, sample_data_with_features):
        """Test that feature columns are included and only numeric."""
        df = sample_data_with_features.copy()
        df['text_col'] = 'test'
        feature_cols = identify_feature_columns(df)
        expected = ['rsi_14', 'macd_line', 'macd_signal', 'sma_20', 'bb_width', 'volume_ratio', 'price_change_pct']
        for feat in expected:
            assert feat in feature_cols
        assert 'text_col' not in feature_cols


# =============================================================================
# TESTS: scale_splits - Train-Only Fitting
# =============================================================================

class TestScaleSplitsTrainOnlyFitting:
    """Tests for train-only fitting to prevent data leakage."""

    def test_scaler_fitted_only_on_train(self, sample_splits):
        """CRITICAL: Verify scaler is fitted only on training data."""
        train_scaled, val_scaled, test_scaled, scaler = scale_splits(
            combined_df=sample_splits['data'], train_indices=sample_splits['train_idx'],
            val_indices=sample_splits['val_idx'], test_indices=sample_splits['test_idx'],
            scaler_type='robust'
        )
        assert scaler.is_fitted
        assert scaler.n_samples_train == len(sample_splits['train_idx'])

    def test_val_test_use_training_statistics(self, sample_splits):
        """CRITICAL: Verify val/test use training statistics, not their own."""
        df_modified = sample_splits['data'].copy()
        df_modified.loc[sample_splits['val_idx'], 'rsi_14'] += 200.0
        df_modified.loc[sample_splits['test_idx'], 'rsi_14'] -= 200.0

        train_scaled, val_scaled, test_scaled, scaler = scale_splits(
            combined_df=df_modified, train_indices=sample_splits['train_idx'],
            val_indices=sample_splits['val_idx'], test_indices=sample_splits['test_idx'],
            scaler_type='standard', clip_outliers=False
        )

        assert abs(train_scaled['rsi_14'].mean()) < 0.1
        assert abs(train_scaled['rsi_14'].std() - 1.0) < 0.2
        assert val_scaled['rsi_14'].mean() > 5.0  # Shifted positive
        assert test_scaled['rsi_14'].mean() < -5.0  # Shifted negative

    def test_no_data_leakage_between_splits(self, sample_splits):
        """Test that no data leakage occurs between splits."""
        train_scaled, val_scaled, test_scaled, _ = scale_splits(
            combined_df=sample_splits['data'], train_indices=sample_splits['train_idx'],
            val_indices=sample_splits['val_idx'], test_indices=sample_splits['test_idx']
        )
        assert len(train_scaled) == len(sample_splits['train_idx'])
        assert len(val_scaled) == len(sample_splits['val_idx'])
        assert len(test_scaled) == len(sample_splits['test_idx'])
        train_dates, val_dates, test_dates = set(train_scaled['datetime']), set(val_scaled['datetime']), set(test_scaled['datetime'])
        assert len(train_dates & val_dates) == 0 and len(train_dates & test_dates) == 0 and len(val_dates & test_dates) == 0


# =============================================================================
# TESTS: scale_splits - Scaler Types
# =============================================================================

class TestScaleSplitsScalerTypes:
    """Tests for different scaler types."""

    @pytest.mark.parametrize("scaler_type", ['robust', 'standard', 'minmax'])
    def test_scaler_types(self, sample_splits, scaler_type):
        """Test different scaler types work correctly."""
        _, _, _, scaler = scale_splits(
            combined_df=sample_splits['data'], train_indices=sample_splits['train_idx'],
            val_indices=sample_splits['val_idx'], test_indices=sample_splits['test_idx'],
            scaler_type=scaler_type
        )
        assert scaler.is_fitted and len(scaler.feature_names) > 0

    def test_robust_scaler_handles_outliers(self, sample_splits):
        """Test that RobustScaler handles outliers well."""
        df = sample_splits['data'].copy()
        df.loc[10:20, 'rsi_14'] = 1000.0
        train_scaled, _, _, _ = scale_splits(
            combined_df=df, train_indices=sample_splits['train_idx'],
            val_indices=sample_splits['val_idx'], test_indices=sample_splits['test_idx'],
            scaler_type='robust'
        )
        assert train_scaled['rsi_14'].std() < 100

    def test_standard_scaler_normalization(self, sample_splits):
        """Test that StandardScaler produces normalized features."""
        train_scaled, _, _, scaler = scale_splits(
            combined_df=sample_splits['data'], train_indices=sample_splits['train_idx'],
            val_indices=sample_splits['val_idx'], test_indices=sample_splits['test_idx'],
            scaler_type='standard', clip_outliers=False
        )
        normalized = sum(1 for f in scaler.feature_names
                        if abs(train_scaled[f].mean()) < 0.2 and abs(train_scaled[f].std() - 1.0) < 0.3)
        assert normalized >= len(scaler.feature_names) * 0.7

    def test_minmax_scaler_bounds(self, sample_splits):
        """Test that MinMaxScaler bounds features reasonably."""
        train_scaled, _, _, scaler = scale_splits(
            combined_df=sample_splits['data'], train_indices=sample_splits['train_idx'],
            val_indices=sample_splits['val_idx'], test_indices=sample_splits['test_idx'],
            scaler_type='minmax', clip_outliers=False
        )
        bounded = sum(1 for f in scaler.feature_names
                     if train_scaled[f].min() >= -0.1 and train_scaled[f].max() <= 1.1)
        assert bounded >= len(scaler.feature_names) * 0.7


# =============================================================================
# TESTS: scale_splits - Outlier Clipping
# =============================================================================

class TestScaleSplitsOutlierClipping:
    """Tests for outlier clipping functionality."""

    def test_outlier_clipping_enabled(self, sample_splits):
        """Test that outlier clipping works when enabled."""
        clip_range = (-3.0, 3.0)
        train_scaled, val_scaled, _, scaler = scale_splits(
            combined_df=sample_splits['data'], train_indices=sample_splits['train_idx'],
            val_indices=sample_splits['val_idx'], test_indices=sample_splits['test_idx'],
            scaler_type='robust', clip_outliers=True, clip_range=clip_range
        )
        for feat in scaler.feature_names:
            assert train_scaled[feat].min() >= clip_range[0] and train_scaled[feat].max() <= clip_range[1]
            assert val_scaled[feat].min() >= clip_range[0] and val_scaled[feat].max() <= clip_range[1]

    def test_outlier_clipping_disabled(self, sample_splits):
        """Test that outlier clipping can be disabled."""
        df = sample_splits['data'].copy()
        df.loc[10, 'rsi_14'] = 10000.0
        train_scaled, _, _, _ = scale_splits(
            combined_df=df, train_indices=sample_splits['train_idx'],
            val_indices=sample_splits['val_idx'], test_indices=sample_splits['test_idx'],
            scaler_type='standard', clip_outliers=False
        )
        assert train_scaled['rsi_14'].max() > 5.0

    def test_custom_clip_range(self, sample_splits):
        """Test custom clip range."""
        custom_range = (-10.0, 10.0)
        train_scaled, _, _, scaler = scale_splits(
            combined_df=sample_splits['data'], train_indices=sample_splits['train_idx'],
            val_indices=sample_splits['val_idx'], test_indices=sample_splits['test_idx'],
            scaler_type='robust', clip_outliers=True, clip_range=custom_range
        )
        for feat in scaler.feature_names:
            assert train_scaled[feat].min() >= custom_range[0] and train_scaled[feat].max() <= custom_range[1]


# =============================================================================
# TESTS: scale_splits - Edge Cases
# =============================================================================

class TestScaleSplitsEdgeCases:
    """Tests for edge cases."""

    def test_error_on_empty_feature_columns(self, sample_splits):
        """Test that error is raised when no feature columns found."""
        df_no_features = sample_splits['data'][
            ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'label_h5', 'label_h20']
        ].copy()
        with pytest.raises(ValueError, match="No feature columns"):
            scale_splits(combined_df=df_no_features, train_indices=sample_splits['train_idx'],
                        val_indices=sample_splits['val_idx'], test_indices=sample_splits['test_idx'])

    def test_handles_nan_values(self, sample_splits):
        """Test that NaN values are handled correctly."""
        df = sample_splits['data'].copy()
        df.loc[10:20, 'rsi_14'] = np.nan
        train_scaled, val_scaled, test_scaled, _ = scale_splits(
            combined_df=df, train_indices=sample_splits['train_idx'],
            val_indices=sample_splits['val_idx'], test_indices=sample_splits['test_idx']
        )
        assert not train_scaled['rsi_14'].isna().any()
        assert not val_scaled['rsi_14'].isna().any()
        assert not test_scaled['rsi_14'].isna().any()

    def test_handles_constant_features(self, sample_splits):
        """Test that constant features (zero variance) are handled."""
        df = sample_splits['data'].copy()
        df['constant_feat'] = 42.0
        _, _, _, scaler = scale_splits(
            combined_df=df, train_indices=sample_splits['train_idx'],
            val_indices=sample_splits['val_idx'], test_indices=sample_splits['test_idx']
        )
        assert 'constant_feat' in scaler.feature_names
        assert len(scaler.warnings) > 0

    def test_handles_single_sample(self):
        """Test that single sample per split is handled."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=3, freq='5min'),
            'symbol': 'MES', 'close': [100.0, 101.0, 102.0], 'rsi_14': [50.0, 55.0, 60.0],
        })
        train_scaled, val_scaled, test_scaled, _ = scale_splits(
            combined_df=df, train_indices=np.array([0]), val_indices=np.array([1]), test_indices=np.array([2])
        )
        assert len(train_scaled) == 1 and len(val_scaled) == 1 and len(test_scaled) == 1


# =============================================================================
# TESTS: scale_splits - Save/Load
# =============================================================================

class TestScaleSplitsSaveLoad:
    """Tests for scaler persistence."""

    def test_saves_all_files(self, sample_splits, temp_directory):
        """Test that scaled data and scaler are saved correctly."""
        output_dir = temp_directory / "scaled"
        scale_splits(combined_df=sample_splits['data'], train_indices=sample_splits['train_idx'],
                    val_indices=sample_splits['val_idx'], test_indices=sample_splits['test_idx'],
                    output_dir=output_dir)
        assert (output_dir / "train_scaled.parquet").exists()
        assert (output_dir / "val_scaled.parquet").exists()
        assert (output_dir / "test_scaled.parquet").exists()
        assert (output_dir / "feature_scaler.pkl").exists()
        assert (output_dir / "feature_scaler.json").exists()

    def test_saved_scaler_can_be_loaded(self, sample_splits, temp_directory):
        """Test that saved scaler can be loaded and used."""
        output_dir = temp_directory / "scaled"
        _, _, _, scaler = scale_splits(combined_df=sample_splits['data'],
                                       train_indices=sample_splits['train_idx'],
                                       val_indices=sample_splits['val_idx'],
                                       test_indices=sample_splits['test_idx'],
                                       output_dir=output_dir)
        loaded_scaler = FeatureScaler.load(output_dir / "feature_scaler.pkl")
        assert loaded_scaler.is_fitted
        assert loaded_scaler.feature_names == scaler.feature_names
        assert loaded_scaler.n_samples_train == scaler.n_samples_train

    def test_loaded_scaler_produces_same_results(self, sample_splits, temp_directory):
        """Test that loaded scaler produces identical results."""
        output_dir = temp_directory / "scaled"
        _, _, _, scaler = scale_splits(combined_df=sample_splits['data'],
                                       train_indices=sample_splits['train_idx'],
                                       val_indices=sample_splits['val_idx'],
                                       test_indices=sample_splits['test_idx'],
                                       output_dir=output_dir)
        loaded_scaler = FeatureScaler.load(output_dir / "feature_scaler.pkl")
        test_df = sample_splits['data'].iloc[sample_splits['test_idx']]
        result_original = scaler.transform(test_df)
        result_loaded = loaded_scaler.transform(test_df)
        for feat in scaler.feature_names:
            np.testing.assert_allclose(result_original[feat].values, result_loaded[feat].values, rtol=1e-10)


# =============================================================================
# TESTS: run_scaling_stage - Full Stage
# =============================================================================

class TestRunScalingStage:
    """Tests for full scaling stage execution."""

    def _setup_stage_inputs(self, df, temp_directory):
        """Helper to setup stage inputs."""
        data_path = temp_directory / "combined_data.parquet"
        df.to_parquet(data_path)
        splits_dir = temp_directory / "splits"
        splits_dir.mkdir(parents=True)
        n = len(df)
        np.save(splits_dir / "train_indices.npy", np.arange(0, int(n * 0.70)))
        np.save(splits_dir / "val_indices.npy", np.arange(int(n * 0.70), int(n * 0.85)))
        np.save(splits_dir / "test_indices.npy", np.arange(int(n * 0.85), n))
        return data_path, splits_dir, n

    def test_full_stage_execution(self, sample_data_with_features, temp_directory):
        """Test complete stage execution with file I/O."""
        data_path, splits_dir, n = self._setup_stage_inputs(sample_data_with_features, temp_directory)
        metadata = run_scaling_stage(data_path=data_path, splits_dir=splits_dir,
                                     scaler_type='robust', clip_outliers=True, clip_range=(-5.0, 5.0))
        assert metadata['scaler_type'] == 'robust'
        assert metadata['clip_outliers'] is True
        assert metadata['clip_range'] == [-5.0, 5.0]
        assert metadata['n_features_scaled'] > 0
        assert metadata['train_samples'] == int(n * 0.70)

    def test_stage_creates_output_files(self, sample_data_with_features, temp_directory):
        """Test that stage creates all expected output files."""
        data_path, splits_dir, _ = self._setup_stage_inputs(sample_data_with_features, temp_directory)
        run_scaling_stage(data_path=data_path, splits_dir=splits_dir)
        output_dir = splits_dir / "scaled"
        assert (output_dir / "train_scaled.parquet").exists()
        assert (output_dir / "val_scaled.parquet").exists()
        assert (output_dir / "test_scaled.parquet").exists()
        assert (output_dir / "feature_scaler.pkl").exists()
        assert (output_dir / "feature_scaler.json").exists()
        assert (output_dir / "scaling_metadata.json").exists()

    def test_stage_custom_output_directory(self, sample_data_with_features, temp_directory):
        """Test that custom output directory is respected."""
        data_path, splits_dir, _ = self._setup_stage_inputs(sample_data_with_features, temp_directory)
        custom_output = temp_directory / "custom_scaled"
        run_scaling_stage(data_path=data_path, splits_dir=splits_dir, output_dir=custom_output)
        assert (custom_output / "train_scaled.parquet").exists()
        assert (custom_output / "val_scaled.parquet").exists()
        assert (custom_output / "test_scaled.parquet").exists()

    def test_stage_metadata_json_format(self, sample_data_with_features, temp_directory):
        """Test that metadata JSON has correct format."""
        data_path, splits_dir, _ = self._setup_stage_inputs(sample_data_with_features, temp_directory)
        run_scaling_stage(data_path=data_path, splits_dir=splits_dir)
        with open(splits_dir / "scaled" / "scaling_metadata.json", 'r') as f:
            metadata = json.load(f)
        required_fields = ['timestamp', 'scaler_type', 'clip_outliers', 'clip_range', 'n_features_scaled',
                          'train_samples', 'val_samples', 'test_samples', 'feature_columns']
        for field in required_fields:
            assert field in metadata
        assert isinstance(metadata['feature_columns'], list)
