"""
Tests for PurgedKFold cross-validation.

Tests:
- Fold generation and sizing
- Purge zone removal (samples before test set)
- Embargo zone removal (samples after test set)
- Label leakage prevention with overlapping labels
- Minimum training size validation
- Coverage validation
"""
import numpy as np
import pandas as pd
import pytest

from src.cross_validation.purged_kfold import (
    PurgedKFold,
    PurgedKFoldConfig,
    ModelAwareCV,
    CV_STRATEGIES,
    get_cv_config_for_family,
)


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestPurgedKFoldConfig:
    """Tests for PurgedKFoldConfig validation."""

    def test_valid_config(self):
        """Test valid configuration creates successfully."""
        config = PurgedKFoldConfig(
            n_splits=5,
            purge_bars=60,
            embargo_bars=100,
            min_train_size=0.3,
        )
        assert config.n_splits == 5
        assert config.purge_bars == 60
        assert config.embargo_bars == 100
        assert config.min_train_size == 0.3

    def test_default_config(self):
        """Test default configuration values."""
        config = PurgedKFoldConfig()
        assert config.n_splits == 5
        assert config.purge_bars == 60
        assert config.embargo_bars == 1440
        assert config.min_train_size == 0.3

    def test_invalid_n_splits_too_low(self):
        """Test that n_splits < 2 raises ValueError."""
        with pytest.raises(ValueError, match="n_splits must be >= 2"):
            PurgedKFoldConfig(n_splits=1)

    def test_invalid_purge_bars_negative(self):
        """Test that negative purge_bars raises ValueError."""
        with pytest.raises(ValueError, match="purge_bars must be >= 0"):
            PurgedKFoldConfig(purge_bars=-1)

    def test_invalid_embargo_bars_negative(self):
        """Test that negative embargo_bars raises ValueError."""
        with pytest.raises(ValueError, match="embargo_bars must be >= 0"):
            PurgedKFoldConfig(embargo_bars=-1)

    def test_invalid_min_train_size_zero(self):
        """Test that min_train_size <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="min_train_size must be in"):
            PurgedKFoldConfig(min_train_size=0)

    def test_invalid_min_train_size_one(self):
        """Test that min_train_size >= 1 raises ValueError."""
        with pytest.raises(ValueError, match="min_train_size must be in"):
            PurgedKFoldConfig(min_train_size=1.0)


# =============================================================================
# FOLD GENERATION TESTS
# =============================================================================

class TestFoldGeneration:
    """Tests for basic fold generation."""

    def test_generates_correct_number_of_folds(self, default_cv, time_series_data):
        """Test that CV generates expected number of folds."""
        X = time_series_data["X"]
        folds = list(default_cv.split(X))

        assert len(folds) == default_cv.config.n_splits
        assert len(folds) == 5

    def test_fold_indices_are_arrays(self, small_cv, small_time_series_data):
        """Test that fold indices are numpy arrays."""
        X = small_time_series_data["X"]
        for train_idx, test_idx in small_cv.split(X):
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(test_idx, np.ndarray)

    def test_train_test_indices_non_overlapping(self, small_cv, small_time_series_data):
        """Test that train and test indices do not overlap."""
        X = small_time_series_data["X"]
        for train_idx, test_idx in small_cv.split(X):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0, f"Found {len(overlap)} overlapping indices"

    def test_test_folds_are_contiguous(self, small_cv, small_time_series_data):
        """Test that test indices form contiguous blocks (time-series property)."""
        X = small_time_series_data["X"]
        for train_idx, test_idx in small_cv.split(X):
            # Test indices should be consecutive
            test_idx_sorted = np.sort(test_idx)
            expected = np.arange(test_idx_sorted[0], test_idx_sorted[-1] + 1)
            np.testing.assert_array_equal(test_idx_sorted, expected)

    def test_all_samples_covered_in_test(self, default_cv, time_series_data):
        """Test that all samples appear in at least one test fold."""
        X = time_series_data["X"]
        all_test_indices = set()

        for _, test_idx in default_cv.split(X):
            all_test_indices.update(test_idx)

        assert len(all_test_indices) == len(X)

    def test_get_n_splits(self, default_cv, time_series_data):
        """Test get_n_splits returns correct value."""
        X = time_series_data["X"]
        assert default_cv.get_n_splits(X) == 5


# =============================================================================
# PURGE ZONE TESTS
# =============================================================================

class TestPurgeZone:
    """Tests for purge zone removal before test set."""

    def test_purge_zone_removed(self, small_time_series_data):
        """Test that samples in purge zone are removed from training."""
        config = PurgedKFoldConfig(n_splits=3, purge_bars=20, embargo_bars=0, min_train_size=0.1)
        cv = PurgedKFold(config)
        X = small_time_series_data["X"]

        for train_idx, test_idx in cv.split(X):
            test_start = test_idx[0]
            purge_zone = range(max(0, test_start - 20), test_start)

            # No training samples should be in purge zone
            for idx in purge_zone:
                assert idx not in train_idx, f"Index {idx} in purge zone but in training"

    def test_larger_purge_removes_more_samples(self, small_time_series_data):
        """Test that larger purge_bars removes more training samples."""
        X = small_time_series_data["X"]

        config_small = PurgedKFoldConfig(n_splits=3, purge_bars=5, embargo_bars=0, min_train_size=0.1)
        config_large = PurgedKFoldConfig(n_splits=3, purge_bars=30, embargo_bars=0, min_train_size=0.1)

        cv_small = PurgedKFold(config_small)
        cv_large = PurgedKFold(config_large)

        folds_small = list(cv_small.split(X))
        folds_large = list(cv_large.split(X))

        # Larger purge should result in smaller or equal training sets
        # For folds where test_start > purge_bars, training sets should be smaller
        # For the first fold (test_start near 0), purge may be at boundary
        total_train_small = sum(len(t) for t, _ in folds_small)
        total_train_large = sum(len(t) for t, _ in folds_large)

        # Total training samples across all folds should be fewer with larger purge
        assert total_train_large < total_train_small

    def test_zero_purge_no_removal(self, small_time_series_data):
        """Test that purge_bars=0 removes no extra samples."""
        config = PurgedKFoldConfig(n_splits=3, purge_bars=0, embargo_bars=0, min_train_size=0.1)
        cv = PurgedKFold(config)
        X = small_time_series_data["X"]
        n_samples = len(X)

        for train_idx, test_idx in cv.split(X):
            # With no purge/embargo, train + test should cover all samples
            # (except test itself)
            expected_train_size = n_samples - len(test_idx)
            assert len(train_idx) == expected_train_size


# =============================================================================
# EMBARGO ZONE TESTS
# =============================================================================

class TestEmbargoZone:
    """Tests for embargo zone removal after test set."""

    def test_embargo_zone_removed(self, small_time_series_data):
        """Test that samples in embargo zone are removed from training."""
        config = PurgedKFoldConfig(n_splits=3, purge_bars=0, embargo_bars=20, min_train_size=0.1)
        cv = PurgedKFold(config)
        X = small_time_series_data["X"]
        n_samples = len(X)

        for train_idx, test_idx in cv.split(X):
            test_end = test_idx[-1] + 1
            embargo_zone = range(test_end, min(n_samples, test_end + 20))

            # No training samples should be in embargo zone
            for idx in embargo_zone:
                assert idx not in train_idx, f"Index {idx} in embargo zone but in training"

    def test_larger_embargo_removes_more_samples(self, small_time_series_data):
        """Test that larger embargo_bars removes more training samples."""
        X = small_time_series_data["X"]

        config_small = PurgedKFoldConfig(n_splits=3, purge_bars=0, embargo_bars=5, min_train_size=0.1)
        config_large = PurgedKFoldConfig(n_splits=3, purge_bars=0, embargo_bars=30, min_train_size=0.1)

        cv_small = PurgedKFold(config_small)
        cv_large = PurgedKFold(config_large)

        folds_small = list(cv_small.split(X))
        folds_large = list(cv_large.split(X))

        # Compare non-final folds (final fold has no data after)
        for i in range(len(folds_small) - 1):
            train_small, _ = folds_small[i]
            train_large, _ = folds_large[i]
            assert len(train_large) < len(train_small)


# =============================================================================
# LABEL LEAKAGE PREVENTION TESTS
# =============================================================================

class TestLabelLeakagePrevention:
    """Tests for preventing label leakage with overlapping labels."""

    def test_overlapping_labels_purged(self, label_end_times_data):
        """Test that samples with labels overlapping test set are purged."""
        config = PurgedKFoldConfig(n_splits=3, purge_bars=10, embargo_bars=10, min_train_size=0.1)
        cv = PurgedKFold(config)

        X = label_end_times_data["X"]
        label_end_times = label_end_times_data["label_end_times"]

        for train_idx, test_idx in cv.split(X, label_end_times=label_end_times):
            test_start_time = X.index[test_idx[0]]

            # Check that no training sample has label_end_time >= test_start_time
            for idx in train_idx:
                if idx < test_idx[0]:  # Only check samples before test
                    label_end = label_end_times.iloc[idx]
                    assert label_end < test_start_time, (
                        f"Sample {idx} has label_end_time {label_end} >= "
                        f"test_start_time {test_start_time}"
                    )

    def test_without_label_end_times_uses_basic_purge(self, small_time_series_data):
        """Test that without label_end_times, basic purge still works."""
        config = PurgedKFoldConfig(n_splits=3, purge_bars=20, embargo_bars=0, min_train_size=0.1)
        cv = PurgedKFold(config)
        X = small_time_series_data["X"]

        # Should work without label_end_times
        folds = list(cv.split(X, label_end_times=None))
        assert len(folds) == 3

    def test_label_aware_purge_removes_additional_samples(self, label_end_times_data):
        """
        Prove that label_end_times purging removes MORE samples than basic purge.

        This demonstrates overlapping-event leakage prevention: samples before the
        purge zone whose labels extend into the test period are removed.
        """
        config = PurgedKFoldConfig(n_splits=3, purge_bars=10, embargo_bars=0, min_train_size=0.1)
        cv = PurgedKFold(config)

        X = label_end_times_data["X"]
        label_end_times = label_end_times_data["label_end_times"]
        horizon = label_end_times_data["horizon"]

        # Compare training sizes with and without label_end_times
        folds_without = list(cv.split(X, label_end_times=None))
        folds_with = list(cv.split(X, label_end_times=label_end_times))

        total_train_without = sum(len(train) for train, _ in folds_without)
        total_train_with = sum(len(train) for train, _ in folds_with)

        # Label-aware purging should remove additional samples
        # (samples whose labels overlap with test set even though they're before purge zone)
        assert total_train_with < total_train_without, (
            f"Label-aware purging should remove more samples. "
            f"Without: {total_train_without}, With: {total_train_with}"
        )

        # The difference should be roughly proportional to horizon size
        # (samples within horizon bars before purge zone whose labels reach into test)
        removed_by_label_purge = total_train_without - total_train_with
        assert removed_by_label_purge > 0, "Label-aware purging should remove additional samples"

    def test_no_lookahead_with_label_end_times(self, label_end_times_data):
        """
        Explicit test: no training sample can see future information via its label.

        A label "sees" information from entry_time to label_end_time. If label_end_time
        overlaps with the test set, that's lookahead bias.
        """
        config = PurgedKFoldConfig(n_splits=3, purge_bars=5, embargo_bars=5, min_train_size=0.1)
        cv = PurgedKFold(config)

        X = label_end_times_data["X"]
        label_end_times = label_end_times_data["label_end_times"]

        for fold_idx, (train_idx, test_idx) in enumerate(
            cv.split(X, label_end_times=label_end_times)
        ):
            test_start_time = X.index[test_idx[0]]
            test_end_time = X.index[test_idx[-1]]

            # Every training sample's label must be resolved before test set starts
            for train_i in train_idx:
                label_end = label_end_times.iloc[train_i]
                sample_time = X.index[train_i]

                # If sample is before test set, its label_end_time must not reach into test
                if sample_time < test_start_time:
                    assert label_end < test_start_time, (
                        f"Fold {fold_idx}: Sample at {sample_time} has label ending at "
                        f"{label_end} which overlaps test set starting at {test_start_time}. "
                        "This is lookahead bias!"
                    )


# =============================================================================
# MINIMUM TRAINING SIZE TESTS
# =============================================================================

class TestMinimumTrainingSize:
    """Tests for minimum training size validation."""

    def test_raises_when_train_too_small(self, small_time_series_data):
        """Test that ValueError is raised when training set is too small."""
        # Configure with very strict purge/embargo that will make training too small
        config = PurgedKFoldConfig(
            n_splits=5,
            purge_bars=50,
            embargo_bars=50,
            min_train_size=0.5,  # Require at least 50% for training
        )
        cv = PurgedKFold(config)
        X = small_time_series_data["X"]

        with pytest.raises(ValueError, match="Training set too small"):
            list(cv.split(X))

    def test_passes_when_train_sufficient(self, time_series_data):
        """Test that validation passes with sufficient training size."""
        config = PurgedKFoldConfig(
            n_splits=5,
            purge_bars=20,
            embargo_bars=20,
            min_train_size=0.3,
        )
        cv = PurgedKFold(config)
        X = time_series_data["X"]

        # Should not raise
        folds = list(cv.split(X))
        assert len(folds) == 5


# =============================================================================
# COVERAGE VALIDATION TESTS
# =============================================================================

class TestCoverageValidation:
    """Tests for validate_coverage method."""

    def test_validate_coverage_full(self, default_cv, time_series_data):
        """Test coverage validation reports 100% coverage."""
        X = time_series_data["X"]
        coverage = default_cv.validate_coverage(X)

        assert coverage["total_samples"] == len(X)
        assert coverage["coverage_fraction"] == 1.0
        assert coverage["uncovered_samples"] == 0

    def test_validate_coverage_no_duplicates(self, default_cv, time_series_data):
        """Test that samples don't appear in multiple test folds."""
        X = time_series_data["X"]
        coverage = default_cv.validate_coverage(X)

        # Standard k-fold should have no samples in multiple test folds
        assert coverage["samples_in_multiple_folds"] == 0


# =============================================================================
# FOLD INFO TESTS
# =============================================================================

class TestGetFoldInfo:
    """Tests for get_fold_info method."""

    def test_fold_info_structure(self, default_cv, time_series_data):
        """Test that fold info has expected structure."""
        X = time_series_data["X"]
        info = default_cv.get_fold_info(X)

        assert len(info) == 5

        for i, fold_info in enumerate(info):
            assert fold_info["fold"] == i
            assert "train_size" in fold_info
            assert "test_size" in fold_info
            assert "train_start" in fold_info
            assert "train_end" in fold_info
            assert "test_start" in fold_info
            assert "test_end" in fold_info
            assert fold_info["train_size"] > 0
            assert fold_info["test_size"] > 0

    def test_fold_info_with_integer_index(self, default_cv):
        """Test fold info works with integer index."""
        X = pd.DataFrame(np.random.randn(500, 10))  # Integer index

        info = default_cv.get_fold_info(X)

        for fold_info in info:
            # Should have _idx suffix for integer index
            assert "train_start_idx" in fold_info
            assert "test_start_idx" in fold_info


# =============================================================================
# MODEL-AWARE CV TESTS
# =============================================================================

class TestModelAwareCV:
    """Tests for ModelAwareCV wrapper."""

    def test_boosting_family_5_splits(self, default_cv, time_series_data):
        """Test boosting family uses 5 splits."""
        X = time_series_data["X"]
        model_cv = ModelAwareCV("boosting", default_cv)

        folds = list(model_cv.get_cv_splits(X))
        assert len(folds) == CV_STRATEGIES["boosting"]["n_splits"]
        assert len(folds) == 5

    def test_neural_family_3_splits(self, default_cv, time_series_data):
        """Test neural family uses 3 splits."""
        X = time_series_data["X"]
        model_cv = ModelAwareCV("neural", default_cv)

        folds = list(model_cv.get_cv_splits(X))
        assert len(folds) == CV_STRATEGIES["neural"]["n_splits"]
        assert len(folds) == 3

    def test_transformer_family_3_splits(self, default_cv, time_series_data):
        """Test transformer family uses 3 splits."""
        X = time_series_data["X"]
        model_cv = ModelAwareCV("transformer", default_cv)

        folds = list(model_cv.get_cv_splits(X))
        assert len(folds) == CV_STRATEGIES["transformer"]["n_splits"]
        assert len(folds) == 3

    def test_unknown_family_uses_default(self, default_cv, time_series_data):
        """Test unknown family falls back to default (boosting) strategy."""
        X = time_series_data["X"]
        model_cv = ModelAwareCV("unknown_model", default_cv)

        folds = list(model_cv.get_cv_splits(X))
        assert len(folds) == CV_STRATEGIES["boosting"]["n_splits"]

    def test_get_tuning_trials(self, default_cv):
        """Test get_tuning_trials returns family-appropriate value."""
        model_cv_boost = ModelAwareCV("boosting", default_cv)
        model_cv_neural = ModelAwareCV("neural", default_cv)

        assert model_cv_boost.get_tuning_trials() == 100
        assert model_cv_neural.get_tuning_trials() == 50


# =============================================================================
# GET_CV_CONFIG_FOR_FAMILY TESTS
# =============================================================================

class TestGetCVConfigForFamily:
    """Tests for get_cv_config_for_family helper."""

    def test_boosting_family_config(self, default_cv_config):
        """Test boosting family gets 5-split config."""
        config = get_cv_config_for_family("boosting", default_cv_config)
        assert config.n_splits == 5

    def test_neural_family_config(self, default_cv_config):
        """Test neural family gets 3-split config."""
        config = get_cv_config_for_family("neural", default_cv_config)
        assert config.n_splits == 3

    def test_preserves_purge_embargo(self, default_cv_config):
        """Test that purge and embargo settings are preserved."""
        config = get_cv_config_for_family("neural", default_cv_config)

        assert config.purge_bars == default_cv_config.purge_bars
        assert config.embargo_bars == default_cv_config.embargo_bars
        assert config.min_train_size == default_cv_config.min_train_size

    def test_unknown_family_returns_base(self, default_cv_config):
        """Test unknown family returns base config unchanged."""
        config = get_cv_config_for_family("unknown", default_cv_config)
        assert config == default_cv_config


# =============================================================================
# REPR TESTS
# =============================================================================

class TestRepr:
    """Tests for string representation."""

    def test_repr_includes_config(self, default_cv):
        """Test __repr__ includes configuration values."""
        repr_str = repr(default_cv)

        assert "PurgedKFold" in repr_str
        assert "n_splits=5" in repr_str
        assert "purge=60" in repr_str
        assert "embargo=100" in repr_str
