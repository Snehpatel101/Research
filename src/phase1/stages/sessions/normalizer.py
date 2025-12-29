"""
Session-Specific Volatility Normalizer

This module provides session-aware normalization for trading features.
Different trading sessions have distinct volatility profiles:
- NY session: Highest volatility, especially at open
- London session: Medium-high volatility
- Asia session: Lower volatility, range-bound

Normalizing features by session-specific volatility helps:
- Create more comparable features across sessions
- Account for session-specific market dynamics
- Improve model performance for session-aware strategies

Author: ML Pipeline
Created: 2025-12-22
"""

import logging
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_SESSIONS_CONFIG,
    SESSIONS,
    SessionName,
    SessionsConfig,
)
from .filter import SessionFilter

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class SessionVolatilityStats:
    """
    Volatility statistics for a single session.

    Stores the rolling volatility parameters used for normalization.
    """
    session_name: str
    n_samples: int
    mean_volatility: float
    std_volatility: float
    median_volatility: float
    q25_volatility: float
    q75_volatility: float
    min_volatility: float
    max_volatility: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'SessionVolatilityStats':
        """Create from dictionary."""
        return cls(**d)


@dataclass
class NormalizationReport:
    """Report from session normalization."""
    timestamp: str
    n_samples: int
    n_features_normalized: int
    sessions_stats: dict[str, dict]
    warnings: list[str]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


class SessionNormalizer:
    """
    Normalize features by session-specific volatility.

    This class computes volatility statistics per session and uses them
    to normalize features, creating more comparable values across sessions.

    Normalization approaches:
    1. Z-score per session: (x - session_mean) / session_std
    2. Volatility ratio: x / session_volatility
    3. Robust scaling: (x - session_median) / session_iqr

    Usage:
        normalizer = SessionNormalizer(config)
        normalizer.fit(train_df, feature_cols, volatility_col='atr_20')
        normalized_df = normalizer.transform(df)
    """

    def __init__(
        self,
        config: SessionsConfig | None = None,
        datetime_column: str = 'datetime',
        method: str = 'zscore'
    ):
        """
        Initialize SessionNormalizer.

        Args:
            config: Session configuration. Uses defaults if None.
            datetime_column: Name of the datetime column.
            method: Normalization method ('zscore', 'volatility', 'robust')

        Raises:
            ValueError: If method is invalid.
        """
        valid_methods = ('zscore', 'volatility', 'robust')
        if method not in valid_methods:
            raise ValueError(
                f"method must be one of {valid_methods}, got '{method}'"
            )

        self.config = config or DEFAULT_SESSIONS_CONFIG
        self.datetime_column = datetime_column
        self.method = method

        self._session_filter = SessionFilter(self.config, datetime_column)
        self._is_fitted = False
        self._feature_names: list[str] = []
        self._volatility_col: str | None = None

        # Statistics per session
        self._session_stats: dict[SessionName, SessionVolatilityStats] = {}
        self._feature_stats: dict[SessionName, dict[str, dict]] = {}

    @property
    def is_fitted(self) -> bool:
        """Check if normalizer has been fitted."""
        return self._is_fitted

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate input DataFrame."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(df)}")

        if df.empty:
            raise ValueError("DataFrame is empty")

        if self.datetime_column not in df.columns:
            raise ValueError(
                f"DataFrame missing required column '{self.datetime_column}'"
            )

    def _get_session_mask(
        self,
        df: pd.DataFrame,
        session_name: SessionName
    ) -> pd.Series:
        """Get boolean mask for rows in a session."""
        datetimes = df[self.datetime_column]
        time_minutes = datetimes.dt.hour * 60 + datetimes.dt.minute

        session_config = SESSIONS[session_name]
        start = session_config.start_minutes
        end = session_config.end_minutes

        if session_config.crosses_midnight:
            return (time_minutes >= start) | (time_minutes < end)
        else:
            return (time_minutes >= start) & (time_minutes < end)

    def _compute_session_volatility_stats(
        self,
        df: pd.DataFrame,
        volatility_col: str,
        session_name: SessionName
    ) -> SessionVolatilityStats:
        """Compute volatility statistics for a session."""
        mask = self._get_session_mask(df, session_name)
        session_data = df.loc[mask, volatility_col].dropna()

        if len(session_data) == 0:
            logger.warning(
                f"No data for session {session_name.value} in volatility column"
            )
            return SessionVolatilityStats(
                session_name=session_name.value,
                n_samples=0,
                mean_volatility=1.0,
                std_volatility=1.0,
                median_volatility=1.0,
                q25_volatility=1.0,
                q75_volatility=1.0,
                min_volatility=1.0,
                max_volatility=1.0,
            )

        return SessionVolatilityStats(
            session_name=session_name.value,
            n_samples=len(session_data),
            mean_volatility=float(session_data.mean()),
            std_volatility=float(session_data.std()),
            median_volatility=float(session_data.median()),
            q25_volatility=float(session_data.quantile(0.25)),
            q75_volatility=float(session_data.quantile(0.75)),
            min_volatility=float(session_data.min()),
            max_volatility=float(session_data.max()),
        )

    def _compute_feature_stats_for_session(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        session_name: SessionName
    ) -> dict[str, dict]:
        """Compute feature statistics for a session."""
        mask = self._get_session_mask(df, session_name)
        session_df = df.loc[mask]

        stats = {}
        for col in feature_cols:
            if col not in df.columns:
                continue

            col_data = session_df[col].dropna()
            if len(col_data) == 0:
                stats[col] = {
                    'mean': 0.0,
                    'std': 1.0,
                    'median': 0.0,
                    'q25': 0.0,
                    'q75': 1.0,
                    'n_samples': 0,
                }
            else:
                q25 = float(col_data.quantile(0.25))
                q75 = float(col_data.quantile(0.75))
                iqr = q75 - q25
                if iqr == 0:
                    iqr = 1.0  # Prevent division by zero

                stats[col] = {
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()) if col_data.std() > 0 else 1.0,
                    'median': float(col_data.median()),
                    'q25': q25,
                    'q75': q75,
                    'iqr': iqr,
                    'n_samples': len(col_data),
                }

        return stats

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        volatility_col: str | None = None
    ) -> 'SessionNormalizer':
        """
        Fit the normalizer on training data.

        Computes session-specific statistics for each feature.

        Args:
            df: Training DataFrame
            feature_cols: List of feature columns to normalize
            volatility_col: Column containing volatility measure (e.g., 'atr_20').
                           Required for 'volatility' method.

        Returns:
            Self for method chaining

        Raises:
            ValueError: If volatility_col is required but not provided
        """
        self._validate_dataframe(df)

        if self.method == 'volatility' and volatility_col is None:
            raise ValueError(
                "volatility_col is required for method='volatility'"
            )

        self._feature_names = [c for c in feature_cols if c in df.columns]
        self._volatility_col = volatility_col

        if len(self._feature_names) == 0:
            raise ValueError("No valid feature columns found in DataFrame")

        logger.info(
            f"Fitting SessionNormalizer on {len(df)} samples, "
            f"{len(self._feature_names)} features"
        )

        active_sessions = self.config.get_active_sessions()

        # Compute statistics for each session
        for session_name in active_sessions:
            # Volatility stats (if using volatility method)
            if volatility_col is not None:
                self._session_stats[session_name] = self._compute_session_volatility_stats(
                    df, volatility_col, session_name
                )
                logger.debug(
                    f"Session {session_name.value}: "
                    f"mean_vol={self._session_stats[session_name].mean_volatility:.4f}"
                )

            # Feature stats for normalization
            self._feature_stats[session_name] = self._compute_feature_stats_for_session(
                df, self._feature_names, session_name
            )

        self._is_fitted = True
        logger.info("SessionNormalizer fitted successfully")

        return self

    def transform(
        self,
        df: pd.DataFrame,
        suffix: str = '_session_norm'
    ) -> pd.DataFrame:
        """
        Transform features using session-specific normalization.

        Args:
            df: DataFrame to transform
            suffix: Suffix to add to normalized feature names

        Returns:
            DataFrame with normalized features added

        Raises:
            RuntimeError: If normalizer has not been fitted
        """
        if not self._is_fitted:
            raise RuntimeError("SessionNormalizer must be fitted before transform")

        self._validate_dataframe(df)
        df = df.copy()

        active_sessions = self.config.get_active_sessions()

        # Create normalized columns
        for col in self._feature_names:
            if col not in df.columns:
                continue

            new_col = f"{col}{suffix}"
            df[new_col] = np.nan

            for session_name in active_sessions:
                mask = self._get_session_mask(df, session_name)

                if mask.sum() == 0:
                    continue

                stats = self._feature_stats.get(session_name, {}).get(col)
                if stats is None or stats.get('n_samples', 0) == 0:
                    # No stats for this session - use original values
                    df.loc[mask, new_col] = df.loc[mask, col]
                    continue

                if self.method == 'zscore':
                    # Z-score normalization
                    mean = stats['mean']
                    std = stats['std']
                    df.loc[mask, new_col] = (df.loc[mask, col] - mean) / std

                elif self.method == 'robust':
                    # Robust scaling using median and IQR
                    median = stats['median']
                    iqr = stats.get('iqr', 1.0)
                    df.loc[mask, new_col] = (df.loc[mask, col] - median) / iqr

                elif self.method == 'volatility':
                    # Scale by session volatility
                    vol_stats = self._session_stats.get(session_name)
                    if vol_stats is None or vol_stats.mean_volatility == 0:
                        df.loc[mask, new_col] = df.loc[mask, col]
                    else:
                        df.loc[mask, new_col] = df.loc[mask, col] / vol_stats.mean_volatility

        logger.info(
            f"Transformed {len(self._feature_names)} features with session normalization"
        )

        return df

    def fit_transform(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        volatility_col: str | None = None,
        suffix: str = '_session_norm'
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Args:
            df: Training DataFrame
            feature_cols: List of feature columns to normalize
            volatility_col: Column containing volatility measure
            suffix: Suffix to add to normalized feature names

        Returns:
            DataFrame with normalized features added
        """
        self.fit(df, feature_cols, volatility_col)
        return self.transform(df, suffix)

    def get_session_stats(self) -> dict[str, dict]:
        """
        Get volatility statistics for all sessions.

        Returns:
            Dictionary of session name -> statistics
        """
        return {
            name.value: stats.to_dict()
            for name, stats in self._session_stats.items()
        }

    def get_normalization_report(self) -> NormalizationReport:
        """
        Generate a normalization report.

        Returns:
            NormalizationReport with statistics and warnings
        """
        warnings = []

        # Check for sessions with few samples
        for session_name, stats in self._session_stats.items():
            if stats.n_samples < 100:
                warnings.append(
                    f"Session {session_name.value} has only {stats.n_samples} samples"
                )

        # Check for features with missing stats
        for session_name, feature_stats in self._feature_stats.items():
            for fname, stats in feature_stats.items():
                if stats.get('n_samples', 0) < 10:
                    warnings.append(
                        f"Feature '{fname}' in session {session_name.value} "
                        f"has only {stats.get('n_samples', 0)} samples"
                    )

        return NormalizationReport(
            timestamp=datetime.now().isoformat(),
            n_samples=sum(s.n_samples for s in self._session_stats.values()),
            n_features_normalized=len(self._feature_names),
            sessions_stats=self.get_session_stats(),
            warnings=warnings,
        )


def normalize_by_session(
    df: pd.DataFrame,
    feature_cols: list[str],
    datetime_column: str = 'datetime',
    method: str = 'zscore',
    volatility_col: str | None = None,
    suffix: str = '_session_norm'
) -> tuple[pd.DataFrame, SessionNormalizer]:
    """
    Convenience function to normalize features by session.

    Args:
        df: Input DataFrame
        feature_cols: List of feature columns to normalize
        datetime_column: Name of datetime column
        method: Normalization method ('zscore', 'volatility', 'robust')
        volatility_col: Column containing volatility (required for 'volatility' method)
        suffix: Suffix for normalized column names

    Returns:
        Tuple of (normalized DataFrame, fitted normalizer)
    """
    config = SessionsConfig(
        session_volatility_normalization=True
    )

    normalizer = SessionNormalizer(
        config=config,
        datetime_column=datetime_column,
        method=method
    )

    df_normalized = normalizer.fit_transform(
        df, feature_cols, volatility_col, suffix
    )

    return df_normalized, normalizer


def get_session_volatility_ratios(
    df: pd.DataFrame,
    volatility_col: str = 'atr_20',
    datetime_column: str = 'datetime'
) -> dict[str, float]:
    """
    Calculate volatility ratios between sessions.

    Useful for understanding relative session activity.

    Args:
        df: Input DataFrame
        volatility_col: Column containing volatility measure
        datetime_column: Name of datetime column

    Returns:
        Dictionary of session -> volatility ratio (relative to overall mean)
    """
    if volatility_col not in df.columns:
        raise ValueError(f"Column '{volatility_col}' not found in DataFrame")

    if datetime_column not in df.columns:
        raise ValueError(f"Column '{datetime_column}' not found in DataFrame")

    overall_mean = df[volatility_col].mean()
    if overall_mean == 0:
        overall_mean = 1.0

    ratios = {}
    time_minutes = df[datetime_column].dt.hour * 60 + df[datetime_column].dt.minute

    for session_name, session_config in SESSIONS.items():
        start = session_config.start_minutes
        end = session_config.end_minutes

        if session_config.crosses_midnight:
            mask = (time_minutes >= start) | (time_minutes < end)
        else:
            mask = (time_minutes >= start) & (time_minutes < end)

        session_mean = df.loc[mask, volatility_col].mean()
        ratios[session_name.value] = float(session_mean / overall_mean)

    return ratios


__all__ = [
    'SessionVolatilityStats',
    'NormalizationReport',
    'SessionNormalizer',
    'normalize_by_session',
    'get_session_volatility_ratios',
]
