"""
Time-Series Cross-Validation for Financial ML
Implements proper CV with purging and embargo for time-series data.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Iterator
from dataclasses import dataclass
import logging
import json

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class CVConfig:
    """Configuration for time-series CV."""
    n_splits: int = 5
    test_size_ratio: float = 0.15
    purge_bars: int = 60
    embargo_bars: int = 288
    min_train_size: int = 10000


@dataclass
class CVSplit:
    """A single CV split with train and test indices."""
    fold: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_start_date: str
    train_end_date: str
    test_start_date: str
    test_end_date: str


class TimeSeriesCV:
    """
    Time-series cross-validation with purging and embargo.

    Implements the methodology from:
    de Prado, M.L. (2018). "Advances in Financial Machine Learning" - Chapter 7
    """

    def __init__(self, config: Optional[CVConfig] = None):
        self.config = config or CVConfig()
        self.splits: List[CVSplit] = []

    def split(self, df: pd.DataFrame, datetime_col: str = 'datetime') -> Iterator[CVSplit]:
        """
        Generate time-series CV splits with purging and embargo.

        Args:
            df: DataFrame with datetime column
            datetime_col: Name of datetime column

        Yields:
            CVSplit objects with train and test indices
        """
        n = len(df)
        test_size = int(n * self.config.test_size_ratio)

        if test_size < 100:
            raise ValueError(f"Test size too small: {test_size}. Increase data or test_size_ratio.")

        self.splits = []

        for i in range(self.config.n_splits):
            # Calculate test boundaries (work backwards from end)
            test_end = n - test_size * (self.config.n_splits - i - 1)
            test_start = test_end - test_size

            if test_start <= 0:
                continue

            # Training ends before test with purge + embargo
            train_end = test_start - self.config.purge_bars - self.config.embargo_bars

            if train_end < self.config.min_train_size:
                logger.warning(f"Fold {i}: Insufficient training data ({train_end} < {self.config.min_train_size}), skipping")
                continue

            # Create indices
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)

            # Get date ranges
            split = CVSplit(
                fold=i,
                train_indices=train_indices,
                test_indices=test_indices,
                train_start_date=str(df.iloc[0][datetime_col]),
                train_end_date=str(df.iloc[train_end-1][datetime_col]),
                test_start_date=str(df.iloc[test_start][datetime_col]),
                test_end_date=str(df.iloc[test_end-1][datetime_col])
            )

            self.splits.append(split)

            logger.info(
                f"Fold {i}: Train[0:{train_end}] ({train_end:,} samples), "
                f"Test[{test_start}:{test_end}] ({test_size:,} samples)"
            )

            yield split

    def get_splits(self) -> List[CVSplit]:
        """Return all splits after split() has been called."""
        return self.splits


class WalkForwardCV:
    """
    Walk-forward validation with expanding or sliding window.

    Simulates real trading where model is retrained periodically.
    """

    def __init__(
        self,
        train_window: int,
        test_window: int,
        step_size: int,
        purge_bars: int = 60,
        embargo_bars: int = 288,
        expanding: bool = False
    ):
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.purge_bars = purge_bars
        self.embargo_bars = embargo_bars
        self.expanding = expanding
        self.splits: List[CVSplit] = []

    def split(self, df: pd.DataFrame, datetime_col: str = 'datetime') -> Iterator[CVSplit]:
        """
        Generate walk-forward splits.

        Args:
            df: DataFrame with datetime column
            datetime_col: Name of datetime column

        Yields:
            CVSplit objects
        """
        n = len(df)
        gap = self.purge_bars + self.embargo_bars

        self.splits = []
        fold = 0
        start = 0

        while True:
            if self.expanding:
                train_start = 0
            else:
                train_start = start

            train_end = start + self.train_window
            test_start = train_end + gap
            test_end = test_start + self.test_window

            if test_end > n:
                break

            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)

            split = CVSplit(
                fold=fold,
                train_indices=train_indices,
                test_indices=test_indices,
                train_start_date=str(df.iloc[train_start][datetime_col]),
                train_end_date=str(df.iloc[train_end-1][datetime_col]),
                test_start_date=str(df.iloc[test_start][datetime_col]),
                test_end_date=str(df.iloc[test_end-1][datetime_col])
            )

            self.splits.append(split)

            logger.info(
                f"WF Fold {fold}: Train[{train_start}:{train_end}] ({len(train_indices):,}), "
                f"Test[{test_start}:{test_end}] ({len(test_indices):,})"
            )

            yield split

            fold += 1
            start += self.step_size

    def get_splits(self) -> List[CVSplit]:
        """Return all splits."""
        return self.splits


def create_cv_splits(
    df: pd.DataFrame,
    output_dir: Path,
    cv_type: str = 'tscv',
    n_splits: int = 5,
    datetime_col: str = 'datetime',
    **kwargs
) -> List[CVSplit]:
    """
    Create and save CV splits.

    Args:
        df: DataFrame
        output_dir: Directory to save splits
        cv_type: 'tscv' or 'walkforward'
        n_splits: Number of folds
        datetime_col: Datetime column name
        **kwargs: Additional arguments for CV class

    Returns:
        List of CVSplit objects
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cv_type == 'tscv':
        config = CVConfig(n_splits=n_splits, **kwargs)
        cv = TimeSeriesCV(config)
    elif cv_type == 'walkforward':
        cv = WalkForwardCV(**kwargs)
    else:
        raise ValueError(f"Unknown cv_type: {cv_type}")

    splits = list(cv.split(df, datetime_col))

    # Save splits
    for split in splits:
        fold_dir = output_dir / f"fold_{split.fold}"
        fold_dir.mkdir(exist_ok=True)

        np.save(fold_dir / "train_indices.npy", split.train_indices)
        np.save(fold_dir / "test_indices.npy", split.test_indices)

        # Save metadata
        meta = {
            'fold': split.fold,
            'train_size': len(split.train_indices),
            'test_size': len(split.test_indices),
            'train_start_date': split.train_start_date,
            'train_end_date': split.train_end_date,
            'test_start_date': split.test_start_date,
            'test_end_date': split.test_end_date
        }
        with open(fold_dir / "metadata.json", 'w') as f:
            json.dump(meta, f, indent=2)

    logger.info(f"Saved {len(splits)} CV folds to {output_dir}")
    return splits


if __name__ == "__main__":
    # Example usage
    import sys
    logging.basicConfig(level=logging.INFO)

    # Set seed for reproducibility
    np.random.seed(42)

    # Create sample data
    dates = pd.date_range('2020-01-01', periods=100000, freq='5min')
    df = pd.DataFrame({'datetime': dates, 'close': np.random.randn(100000).cumsum() + 1000})

    print("\n=== Time-Series CV ===")
    tscv = TimeSeriesCV(CVConfig(n_splits=5))
    for split in tscv.split(df):
        print(f"Fold {split.fold}: Train {len(split.train_indices):,}, Test {len(split.test_indices):,}")

    print("\n=== Walk-Forward CV ===")
    wfcv = WalkForwardCV(train_window=50000, test_window=10000, step_size=10000)
    for split in wfcv.split(df):
        print(f"Fold {split.fold}: Train {len(split.train_indices):,}, Test {len(split.test_indices):,}")
