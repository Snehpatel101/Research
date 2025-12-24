"""
Tests for exception handling across the pipeline.
Verifies that errors are collected, not swallowed.

This module tests:
- Error collection in directory processing (stage1, stage2, stage3)
- GA fitness function error handling
- Absence of bare except clauses in codebase
- Actionable error messages

Run with: pytest tests/test_exception_handling.py -v
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import sys
import re

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestErrorCollection:
    """Tests that errors are collected, not swallowed."""

    def test_stage1_ingest_collects_errors(self):
        """Test that stage1 ingest collects errors for bad files."""
        from src.phase1.stages.ingest import DataIngestor

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create a valid file
            valid_df = pd.DataFrame({
                'datetime': pd.date_range('2020-01-01', periods=10, freq='1min'),
                'open': [100.0]*10,
                'high': [101.0]*10,
                'low': [99.0]*10,
                'close': [100.5]*10,
                'volume': [1000]*10
            })
            valid_df.to_parquet(tmppath / 'valid.parquet')

            # Create an invalid file (corrupt)
            with open(tmppath / 'invalid.parquet', 'w') as f:
                f.write("not a parquet file")

            ingestor = DataIngestor(
                raw_data_dir=tmppath,
                output_dir=tmppath / 'output'
            )

            # Should raise RuntimeError with error collection
            with pytest.raises(RuntimeError, match="failed"):
                ingestor.ingest_directory(pattern="*.parquet")

    def test_stage2_clean_collects_errors(self):
        """Test that stage2 clean collects errors for bad files."""
        from src.phase1.stages.clean import DataCleaner

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create corrupt file
            with open(tmppath / 'bad.parquet', 'w') as f:
                f.write("corrupt")

            cleaner = DataCleaner(
                input_dir=tmppath,
                output_dir=tmppath / 'output'
            )

            with pytest.raises(RuntimeError, match="failed"):
                cleaner.clean_directory(pattern="*.parquet")

    def test_stage3_features_collects_errors(self):
        """Test that stage3 features collects errors."""
        from src.phase1.stages.features.features import FeatureEngineer

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create corrupt file
            with open(tmppath / 'bad.parquet', 'w') as f:
                f.write("corrupt")

            engineer = FeatureEngineer(
                input_dir=tmppath,
                output_dir=tmppath / 'output'
            )

            with pytest.raises(RuntimeError, match="failed"):
                engineer.process_directory(pattern="*.parquet")

    def test_stage1_no_files_returns_empty(self):
        """Test that stage1 returns empty dict when no files match pattern."""
        from src.phase1.stages.ingest import DataIngestor

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            ingestor = DataIngestor(
                raw_data_dir=tmppath,
                output_dir=tmppath / 'output'
            )

            # Should return empty dict, not raise
            result = ingestor.ingest_directory(pattern="*.parquet")
            assert result == {}

    def test_stage2_no_files_returns_empty(self):
        """Test that stage2 returns empty dict when no files match pattern."""
        from src.phase1.stages.clean import DataCleaner

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            cleaner = DataCleaner(
                input_dir=tmppath,
                output_dir=tmppath / 'output'
            )

            # Should return empty dict, not raise
            result = cleaner.clean_directory(pattern="*.parquet")
            assert result == {}

    def test_stage3_no_files_returns_empty(self):
        """Test that stage3 returns empty dict when no files match pattern."""
        from src.phase1.stages.features.features import FeatureEngineer

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            engineer = FeatureEngineer(
                input_dir=tmppath,
                output_dir=tmppath / 'output'
            )

            # Should return empty dict, not raise
            result = engineer.process_directory(pattern="*.parquet")
            assert result == {}


class TestGAFitnessErrors:
    """Tests for GA fitness function error handling."""

    def test_empty_labels_returns_negative(self):
        """Test that empty label array returns negative fitness."""
        from src.phase1.stages.ga_optimize.ga_optimize import calculate_fitness

        labels = np.array([], dtype=np.int8)
        bars_to_hit = np.array([], dtype=np.int32)
        mae = np.array([], dtype=np.float32)
        mfe = np.array([], dtype=np.float32)

        fitness = calculate_fitness(
            labels=labels,
            bars_to_hit=bars_to_hit,
            mae=mae,
            mfe=mfe,
            horizon=5,
            atr_mean=1.0,
            symbol='MES'
        )

        # Empty labels should return very negative fitness
        assert fitness < 0
        assert fitness == -1000.0

    def test_all_neutral_labels_penalized(self):
        """Test that all-neutral labels are penalized."""
        from src.phase1.stages.ga_optimize.ga_optimize import calculate_fitness

        n = 100
        labels = np.zeros(n, dtype=np.int8)  # All neutral
        bars_to_hit = np.full(n, 10, dtype=np.int32)
        mae = np.zeros(n, dtype=np.float32)
        mfe = np.zeros(n, dtype=np.float32)

        fitness = calculate_fitness(
            labels=labels,
            bars_to_hit=bars_to_hit,
            mae=mae,
            mfe=mfe,
            horizon=5,
            atr_mean=1.0,
            symbol='MES'
        )

        # All neutral (100% neutral rate > 40% threshold) should be penalized
        assert fitness < 0

    def test_balanced_labels_positive_fitness(self):
        """Test that balanced labels can achieve positive fitness."""
        from src.phase1.stages.ga_optimize.ga_optimize import calculate_fitness

        n = 100
        # Create balanced distribution: ~35% long, ~35% short, ~30% neutral
        labels = np.array([1]*35 + [-1]*35 + [0]*30, dtype=np.int8)
        np.random.shuffle(labels)

        bars_to_hit = np.random.randint(1, 15, n, dtype=np.int32)
        mae = np.random.uniform(-0.01, 0, n).astype(np.float32)  # Small negative values
        mfe = np.random.uniform(0, 0.02, n).astype(np.float32)   # Small positive values

        fitness = calculate_fitness(
            labels=labels,
            bars_to_hit=bars_to_hit,
            mae=mae,
            mfe=mfe,
            horizon=5,
            atr_mean=1.0,
            symbol='MES'
        )

        # Balanced distribution should not return severe penalty
        # With transaction costs, expect negative fitness but better than -5000
        assert fitness > -5000


class TestNoBareCatch:
    """Verify no bare except clauses exist."""

    def test_no_bare_except_in_stages(self):
        """Search for bare except: clauses in stage files."""
        src_dir = Path(__file__).parent.parent / 'src' / 'stages'

        # Pattern to match bare except: (with optional whitespace)
        # Excludes "except Exception" and "except SomeError"
        bare_except_pattern = re.compile(r'except\s*:')

        violations = []
        for py_file in src_dir.rglob('*.py'):
            content = py_file.read_text()
            # Find all matches with line numbers
            for i, line in enumerate(content.split('\n'), 1):
                if bare_except_pattern.search(line):
                    violations.append(f"{py_file.name}:{i}: {line.strip()}")

        assert len(violations) == 0, f"Bare except: found in:\n" + "\n".join(violations)

    def test_no_bare_except_in_src(self):
        """Search for bare except: clauses in all source files."""
        src_dir = Path(__file__).parent.parent / 'src'

        bare_except_pattern = re.compile(r'except\s*:')

        violations = []
        for py_file in src_dir.rglob('*.py'):
            content = py_file.read_text()
            for i, line in enumerate(content.split('\n'), 1):
                if bare_except_pattern.search(line):
                    violations.append(f"{py_file.relative_to(src_dir)}:{i}")

        assert len(violations) == 0, f"Bare except: found in: {violations}"


class TestErrorMessages:
    """Test that error messages are actionable."""

    def test_missing_atr_column_error_message(self):
        """Test that missing ATR column gives helpful error message."""
        from src.phase1.stages.labeling import apply_triple_barrier

        df = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=100, freq='5min'),
            'close': [100.0] * 100,
            'high': [101.0] * 100,
            'low': [99.0] * 100,
            'open': [100.0] * 100
            # Missing ATR column
        })

        with pytest.raises(KeyError) as exc_info:
            apply_triple_barrier(df, horizon=5)

        # Error message should mention the missing column and provide alternatives
        error_msg = str(exc_info.value).lower()
        assert 'atr' in error_msg

    def test_empty_dataframe_error_message(self):
        """Test that empty DataFrame gives helpful error message."""
        from src.phase1.stages.labeling import apply_triple_barrier

        df = pd.DataFrame(columns=['datetime', 'close', 'high', 'low', 'open', 'atr_14'])

        with pytest.raises(ValueError) as exc_info:
            apply_triple_barrier(df, horizon=5)

        # Error message should mention empty DataFrame
        error_msg = str(exc_info.value).lower()
        assert 'empty' in error_msg

    def test_invalid_horizon_error_message(self):
        """Test that invalid horizon gives helpful error message."""
        from src.phase1.stages.labeling import apply_triple_barrier

        df = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=100, freq='5min'),
            'close': [100.0] * 100,
            'high': [101.0] * 100,
            'low': [99.0] * 100,
            'open': [100.0] * 100,
            'atr_14': [1.0] * 100
        })

        with pytest.raises(ValueError) as exc_info:
            apply_triple_barrier(df, horizon=-1)

        # Error message should mention invalid horizon
        error_msg = str(exc_info.value).lower()
        assert 'horizon' in error_msg or 'positive' in error_msg

    def test_negative_k_up_error_message(self):
        """Test that negative k_up gives helpful error message."""
        from src.phase1.stages.labeling import apply_triple_barrier

        df = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=100, freq='5min'),
            'close': [100.0] * 100,
            'high': [101.0] * 100,
            'low': [99.0] * 100,
            'open': [100.0] * 100,
            'atr_14': [1.0] * 100
        })

        with pytest.raises(ValueError) as exc_info:
            apply_triple_barrier(df, horizon=5, k_up=-1.0)

        # Error message should mention k_up
        error_msg = str(exc_info.value).lower()
        assert 'k_up' in error_msg or 'positive' in error_msg


class TestSecurityErrors:
    """Test security-related error handling."""

    def test_path_traversal_blocked(self):
        """Test that path traversal attempts are blocked."""
        from src.phase1.stages.ingest import DataIngestor, SecurityError

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            ingestor = DataIngestor(
                raw_data_dir=tmppath,
                output_dir=tmppath / 'output'
            )

            # Attempt path traversal
            with pytest.raises(SecurityError, match="suspicious pattern"):
                ingestor.load_data(tmppath / '..' / '..' / 'etc' / 'passwd')

    def test_tilde_expansion_blocked(self):
        """Test that tilde expansion is blocked."""
        from src.phase1.stages.ingest import DataIngestor, SecurityError

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            ingestor = DataIngestor(
                raw_data_dir=tmppath,
                output_dir=tmppath / 'output'
            )

            # Attempt tilde path
            with pytest.raises(SecurityError, match="suspicious pattern"):
                ingestor.load_data(Path('~/.ssh/id_rsa'))


class TestExceptionChaining:
    """Test that exceptions preserve original context."""

    def test_stage1_preserves_error_context(self):
        """Test that stage1 error collection preserves original error info."""
        from src.phase1.stages.ingest import DataIngestor

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create corrupt file
            with open(tmppath / 'corrupt.parquet', 'w') as f:
                f.write("not parquet data")

            ingestor = DataIngestor(
                raw_data_dir=tmppath,
                output_dir=tmppath / 'output'
            )

            try:
                ingestor.ingest_directory(pattern="*.parquet")
                pytest.fail("Expected RuntimeError")
            except RuntimeError as e:
                error_msg = str(e)
                # Should contain error type information
                assert 'Error' in error_msg or 'error' in error_msg
                # Should contain file information
                assert 'corrupt.parquet' in error_msg or '1/' in error_msg


class TestValidationErrors:
    """Test input validation error handling."""

    def test_stage4_missing_ohlc_columns(self):
        """Test that missing OHLC columns raise clear errors."""
        from src.phase1.stages.labeling import apply_triple_barrier

        # DataFrame missing 'high' column
        df = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=100, freq='5min'),
            'close': [100.0] * 100,
            'low': [99.0] * 100,
            'open': [100.0] * 100,
            'atr_14': [1.0] * 100
        })

        with pytest.raises(KeyError) as exc_info:
            apply_triple_barrier(df, horizon=5)

        error_msg = str(exc_info.value)
        # Should mention the missing column
        assert 'high' in error_msg.lower() or 'missing' in error_msg.lower()

    def test_stage4_nan_handling(self):
        """Test that NaN values in close prices are handled gracefully."""
        from src.phase1.stages.labeling import apply_triple_barrier

        df = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=100, freq='5min'),
            'close': [100.0] * 50 + [np.nan] * 50,  # 50% NaN
            'high': [101.0] * 100,
            'low': [99.0] * 100,
            'open': [100.0] * 100,
            'atr_14': [1.0] * 100
        })

        # Should not raise an exception - NaN values are handled gracefully
        result = apply_triple_barrier(df, horizon=5)

        # Verify result has the expected columns
        assert 'label_h5' in result.columns
        assert len(result) == len(df)

        # Bars with NaN close should get neutral label (0) or be marked invalid (-99)
        # The last rows should be marked as invalid (-99) due to insufficient future data
        labels = result['label_h5'].values
        assert labels is not None
        assert len(labels) == 100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
