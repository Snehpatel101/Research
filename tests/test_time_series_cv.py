"""
Comprehensive tests for Time-Series Cross-Validation module.
Tests purging, embargo, and proper temporal splitting.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from stages.time_series_cv import (
    TimeSeriesCV,
    WalkForwardCV,
    CVConfig,
    CVSplit,
    create_cv_splits
)


class TestTimeSeriesCV:
    """Tests for TimeSeriesCV class."""

    @pytest.fixture
    def sample_df(self):
        """Create sample time-series DataFrame."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100000, freq='5min')
        return pd.DataFrame({
            'datetime': dates,
            'close': np.random.randn(100000).cumsum() + 1000
        })

    def test_basic_split(self, sample_df):
        """Test basic CV splitting."""
        config = CVConfig(n_splits=5, test_size_ratio=0.15)
        cv = TimeSeriesCV(config)

        splits = list(cv.split(sample_df))

        assert len(splits) == 5
        for split in splits:
            assert isinstance(split, CVSplit)
            assert len(split.train_indices) > 0
            assert len(split.test_indices) > 0

    def test_no_overlap(self, sample_df):
        """Test that train and test don't overlap."""
        config = CVConfig(n_splits=5)
        cv = TimeSeriesCV(config)

        for split in cv.split(sample_df):
            train_set = set(split.train_indices)
            test_set = set(split.test_indices)

            assert len(train_set & test_set) == 0, "Train and test overlap!"

    def test_purge_gap(self, sample_df):
        """Test that purge gap exists between train and test."""
        purge = 60
        embargo = 288
        config = CVConfig(n_splits=3, purge_bars=purge, embargo_bars=embargo)
        cv = TimeSeriesCV(config)

        for split in cv.split(sample_df):
            train_end = split.train_indices[-1]
            test_start = split.test_indices[0]
            gap = test_start - train_end

            assert gap >= purge + embargo, f"Gap {gap} < purge+embargo {purge+embargo}"

    def test_chronological_order(self, sample_df):
        """Test that train always comes before test."""
        config = CVConfig(n_splits=5)
        cv = TimeSeriesCV(config)

        for split in cv.split(sample_df):
            assert split.train_indices[-1] < split.test_indices[0]

    def test_test_sizes_equal(self, sample_df):
        """Test that test sizes are approximately equal."""
        config = CVConfig(n_splits=5, test_size_ratio=0.15)
        cv = TimeSeriesCV(config)

        splits = list(cv.split(sample_df))
        test_sizes = [len(s.test_indices) for s in splits]

        # All test sizes should be within 10% of each other
        mean_size = np.mean(test_sizes)
        for size in test_sizes:
            assert abs(size - mean_size) / mean_size < 0.1

    def test_get_splits_returns_all(self, sample_df):
        """Test that get_splits() returns all generated splits."""
        config = CVConfig(n_splits=5)
        cv = TimeSeriesCV(config)

        # Consume the generator
        splits_from_iter = list(cv.split(sample_df))
        splits_from_method = cv.get_splits()

        assert len(splits_from_iter) == len(splits_from_method)
        for a, b in zip(splits_from_iter, splits_from_method):
            assert a.fold == b.fold
            np.testing.assert_array_equal(a.train_indices, b.train_indices)
            np.testing.assert_array_equal(a.test_indices, b.test_indices)

    def test_cv_split_date_fields(self, sample_df):
        """Test that CVSplit includes proper date fields."""
        config = CVConfig(n_splits=3)
        cv = TimeSeriesCV(config)

        for split in cv.split(sample_df):
            assert split.train_start_date is not None
            assert split.train_end_date is not None
            assert split.test_start_date is not None
            assert split.test_end_date is not None
            # Verify dates are in string format
            assert isinstance(split.train_start_date, str)
            assert isinstance(split.test_end_date, str)

    def test_default_config(self, sample_df):
        """Test TimeSeriesCV with default config."""
        cv = TimeSeriesCV()

        splits = list(cv.split(sample_df))

        assert len(splits) == 5  # Default n_splits
        assert cv.config.purge_bars == 60  # Default purge
        assert cv.config.embargo_bars == 288  # Default embargo


class TestWalkForwardCV:
    """Tests for WalkForwardCV class."""

    @pytest.fixture
    def sample_df(self):
        """Create sample time-series DataFrame."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100000, freq='5min')
        return pd.DataFrame({
            'datetime': dates,
            'close': np.random.randn(100000).cumsum() + 1000
        })

    def test_sliding_window(self, sample_df):
        """Test sliding window walk-forward."""
        wf = WalkForwardCV(
            train_window=50000,
            test_window=10000,
            step_size=10000,
            purge_bars=60,
            embargo_bars=288,
            expanding=False
        )

        splits = list(wf.split(sample_df))

        assert len(splits) > 0

        # Check train window size is constant (sliding)
        for split in splits:
            assert len(split.train_indices) == 50000

    def test_expanding_window(self, sample_df):
        """Test expanding window walk-forward."""
        wf = WalkForwardCV(
            train_window=30000,
            test_window=10000,
            step_size=10000,
            purge_bars=60,
            embargo_bars=288,
            expanding=True
        )

        splits = list(wf.split(sample_df))

        assert len(splits) > 0

        # Check train window grows (expanding)
        train_sizes = [len(s.train_indices) for s in splits]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i-1]

    def test_purge_embargo_applied(self, sample_df):
        """Test that purge and embargo are applied."""
        purge = 60
        embargo = 288
        wf = WalkForwardCV(
            train_window=50000,
            test_window=10000,
            step_size=10000,
            purge_bars=purge,
            embargo_bars=embargo
        )

        for split in wf.split(sample_df):
            train_end = split.train_indices[-1]
            test_start = split.test_indices[0]
            gap = test_start - train_end

            assert gap >= purge + embargo

    def test_step_size_progression(self, sample_df):
        """Test that walk-forward steps correctly."""
        step_size = 10000
        wf = WalkForwardCV(
            train_window=30000,
            test_window=5000,
            step_size=step_size,
            purge_bars=60,
            embargo_bars=288,
            expanding=False
        )

        splits = list(wf.split(sample_df))

        # Verify each split starts step_size later than the previous
        for i in range(1, len(splits)):
            prev_train_start = splits[i-1].train_indices[0]
            curr_train_start = splits[i].train_indices[0]
            assert curr_train_start - prev_train_start == step_size

    def test_get_splits_method(self, sample_df):
        """Test get_splits returns all generated splits."""
        wf = WalkForwardCV(
            train_window=50000,
            test_window=10000,
            step_size=10000,
            purge_bars=60,
            embargo_bars=288
        )

        # Consume generator
        list(wf.split(sample_df))

        splits = wf.get_splits()
        assert len(splits) > 0
        assert all(isinstance(s, CVSplit) for s in splits)


class TestCVSplitPersistence:
    """Tests for saving/loading CV splits."""

    def test_create_and_save_splits(self):
        """Test creating and saving CV splits."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=50000, freq='5min')
        df = pd.DataFrame({
            'datetime': dates,
            'close': np.random.randn(50000).cumsum() + 1000
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            splits = create_cv_splits(
                df, output_dir, cv_type='tscv', n_splits=3
            )

            assert len(splits) == 3

            # Check files were created
            for i in range(3):
                fold_dir = output_dir / f"fold_{i}"
                assert fold_dir.exists()
                assert (fold_dir / "train_indices.npy").exists()
                assert (fold_dir / "test_indices.npy").exists()
                assert (fold_dir / "metadata.json").exists()

    def test_saved_indices_match(self):
        """Test that saved indices match the split indices."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=50000, freq='5min')
        df = pd.DataFrame({
            'datetime': dates,
            'close': np.random.randn(50000).cumsum() + 1000
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            splits = create_cv_splits(
                df, output_dir, cv_type='tscv', n_splits=3
            )

            # Verify saved data matches
            for split in splits:
                fold_dir = output_dir / f"fold_{split.fold}"
                loaded_train = np.load(fold_dir / "train_indices.npy")
                loaded_test = np.load(fold_dir / "test_indices.npy")

                np.testing.assert_array_equal(loaded_train, split.train_indices)
                np.testing.assert_array_equal(loaded_test, split.test_indices)

    def test_metadata_json_contents(self):
        """Test that metadata JSON has correct contents."""
        import json

        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=50000, freq='5min')
        df = pd.DataFrame({
            'datetime': dates,
            'close': np.random.randn(50000).cumsum() + 1000
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            splits = create_cv_splits(
                df, output_dir, cv_type='tscv', n_splits=3
            )

            for split in splits:
                fold_dir = output_dir / f"fold_{split.fold}"
                with open(fold_dir / "metadata.json") as f:
                    meta = json.load(f)

                assert meta['fold'] == split.fold
                assert meta['train_size'] == len(split.train_indices)
                assert meta['test_size'] == len(split.test_indices)
                assert 'train_start_date' in meta
                assert 'train_end_date' in meta
                assert 'test_start_date' in meta
                assert 'test_end_date' in meta

    def test_walkforward_cv_type(self):
        """Test creating walk-forward splits via create_cv_splits."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100000, freq='5min')
        df = pd.DataFrame({
            'datetime': dates,
            'close': np.random.randn(100000).cumsum() + 1000
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            splits = create_cv_splits(
                df, output_dir, cv_type='walkforward',
                train_window=50000, test_window=10000, step_size=10000
            )

            assert len(splits) > 0

            # Verify files exist for each fold
            for split in splits:
                fold_dir = output_dir / f"fold_{split.fold}"
                assert fold_dir.exists()

    def test_unknown_cv_type_raises(self):
        """Test that unknown cv_type raises ValueError."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=1000, freq='5min')
        df = pd.DataFrame({'datetime': dates, 'close': np.random.randn(1000)})

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Unknown cv_type"):
                create_cv_splits(df, Path(tmpdir), cv_type='invalid')


class TestEdgeCases:
    """Edge case tests."""

    def test_small_dataset_warning(self):
        """Test warning for small dataset."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=1000, freq='5min')
        df = pd.DataFrame({'datetime': dates, 'close': np.random.randn(1000)})

        config = CVConfig(n_splits=5, min_train_size=10000)
        cv = TimeSeriesCV(config)

        splits = list(cv.split(df))
        # Should get fewer splits due to insufficient data
        assert len(splits) < 5

    def test_empty_dataframe_raises(self):
        """Test that empty DataFrame raises error."""
        df = pd.DataFrame({'datetime': [], 'close': []})

        config = CVConfig(n_splits=5)
        cv = TimeSeriesCV(config)

        with pytest.raises((ValueError, ZeroDivisionError)):
            list(cv.split(df))

    def test_very_small_test_raises(self):
        """Test that very small test size raises error."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=500, freq='5min')
        df = pd.DataFrame({'datetime': dates, 'close': np.random.randn(500)})

        # With 500 samples and 0.01 ratio, test size = 5 (< 100)
        config = CVConfig(n_splits=5, test_size_ratio=0.01)
        cv = TimeSeriesCV(config)

        with pytest.raises(ValueError, match="Test size too small"):
            list(cv.split(df))

    def test_walkforward_no_splits_possible(self):
        """Test walk-forward when dataset is too small for any splits."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=1000, freq='5min')
        df = pd.DataFrame({'datetime': dates, 'close': np.random.randn(1000)})

        wf = WalkForwardCV(
            train_window=50000,  # Much larger than data
            test_window=10000,
            step_size=10000,
            purge_bars=60,
            embargo_bars=288
        )

        splits = list(wf.split(df))
        assert len(splits) == 0


class TestCVConfig:
    """Tests for CVConfig dataclass."""

    def test_default_values(self):
        """Test CVConfig default values."""
        config = CVConfig()

        assert config.n_splits == 5
        assert config.test_size_ratio == 0.15
        assert config.purge_bars == 60
        assert config.embargo_bars == 288
        assert config.min_train_size == 10000

    def test_custom_values(self):
        """Test CVConfig with custom values."""
        config = CVConfig(
            n_splits=3,
            test_size_ratio=0.2,
            purge_bars=100,
            embargo_bars=500,
            min_train_size=5000
        )

        assert config.n_splits == 3
        assert config.test_size_ratio == 0.2
        assert config.purge_bars == 100
        assert config.embargo_bars == 500
        assert config.min_train_size == 5000


class TestCVSplitDataclass:
    """Tests for CVSplit dataclass."""

    def test_cv_split_creation(self):
        """Test CVSplit can be created with all required fields."""
        split = CVSplit(
            fold=0,
            train_indices=np.arange(100),
            test_indices=np.arange(100, 150),
            train_start_date='2020-01-01 00:00:00',
            train_end_date='2020-01-01 08:15:00',
            test_start_date='2020-01-01 10:00:00',
            test_end_date='2020-01-01 14:05:00'
        )

        assert split.fold == 0
        assert len(split.train_indices) == 100
        assert len(split.test_indices) == 50
        assert split.train_start_date == '2020-01-01 00:00:00'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
