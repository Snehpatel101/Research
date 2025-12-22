# Session-Aware Normalization System Design

## Executive Summary

This design document specifies a modular session-aware normalization system for futures trading data. The system handles timezone conversion, session filtering, holiday calendars, and session-specific feature normalization to enable more precise trading models that account for distinct market regime characteristics across global trading sessions.

---

## 1. Architecture Overview

### 1.1 Design Principles

Following the project's engineering rules:
- **Modular**: Separate concerns (session definition, filtering, normalization, holidays)
- **<650 lines per file**: Each component stays within complexity limits
- **Fail fast**: Explicit timezone and session validation at boundaries
- **Testable**: Each component has clear contracts and comprehensive tests

### 1.2 Component Structure

```
src/stages/sessions/
├── __init__.py              # Package exports
├── session_config.py        # Session definitions and schemas (~200 lines)
├── timezone_handler.py      # DST and UTC conversion (~150 lines)
├── holiday_calendar.py      # CME holiday calendar (~200 lines)
├── session_filter.py        # Session filtering logic (~250 lines)
├── session_normalizer.py    # Session-aware normalization (~300 lines)
└── constants.py             # Session constants (~100 lines)

config/sessions/
├── cme_sessions.json        # CME session definitions
└── cme_holidays.json        # CME holiday calendar

tests/test_sessions/
├── test_session_config.py
├── test_timezone_handler.py
├── test_holiday_calendar.py
├── test_session_filter.py
└── test_session_normalizer.py
```

---

## 2. Configuration Schema

### 2.1 Session Definition Schema

**File: `config/sessions/cme_sessions.json`**

```json
{
  "version": "1.0",
  "exchange": "CME",
  "timezone": "America/Chicago",
  "sessions": {
    "new_york": {
      "name": "New York",
      "description": "US equity market hours",
      "start_time_et": "09:30",
      "end_time_et": "16:00",
      "start_time_utc_edt": "13:30",
      "end_time_utc_edt": "20:00",
      "start_time_utc_est": "14:30",
      "end_time_utc_est": "21:00",
      "days_of_week": [0, 1, 2, 3, 4],
      "enabled": true,
      "priority": 1,
      "characteristics": {
        "typical_volume_pct": 60,
        "expected_volatility": "high",
        "trend_behavior": "directional"
      }
    },
    "london": {
      "name": "London",
      "description": "European market hours",
      "start_time_et": "03:00",
      "end_time_et": "11:30",
      "start_time_utc_edt": "07:00",
      "end_time_utc_edt": "15:30",
      "start_time_utc_est": "08:00",
      "end_time_utc_est": "16:30",
      "days_of_week": [0, 1, 2, 3, 4],
      "enabled": true,
      "priority": 2,
      "characteristics": {
        "typical_volume_pct": 25,
        "expected_volatility": "medium",
        "trend_behavior": "mixed"
      }
    },
    "asia": {
      "name": "Asia",
      "description": "Asian market hours",
      "start_time_et": "18:00",
      "end_time_et": "02:00",
      "start_time_utc_edt": "22:00",
      "end_time_utc_edt": "06:00",
      "start_time_utc_est": "23:00",
      "end_time_utc_est": "07:00",
      "days_of_week": [0, 1, 2, 3, 4],
      "wraps_midnight": true,
      "enabled": true,
      "priority": 3,
      "characteristics": {
        "typical_volume_pct": 15,
        "expected_volatility": "low",
        "trend_behavior": "range_bound"
      }
    },
    "overnight": {
      "name": "Overnight",
      "description": "After-hours electronic trading",
      "start_time_et": "18:00",
      "end_time_et": "09:30",
      "start_time_utc_edt": "22:00",
      "end_time_utc_edt": "13:30",
      "start_time_utc_est": "23:00",
      "end_time_utc_est": "14:30",
      "days_of_week": [0, 1, 2, 3, 4],
      "wraps_midnight": true,
      "enabled": false,
      "priority": 4,
      "characteristics": {
        "typical_volume_pct": 40,
        "expected_volatility": "variable",
        "trend_behavior": "mixed"
      }
    }
  },
  "dst_transitions": {
    "spring_forward": {
      "description": "Second Sunday in March at 2:00 AM",
      "rule": "second_sunday_march"
    },
    "fall_back": {
      "description": "First Sunday in November at 2:00 AM",
      "rule": "first_sunday_november"
    }
  }
}
```

### 2.2 Holiday Calendar Schema

**File: `config/sessions/cme_holidays.json`**

```json
{
  "version": "1.0",
  "exchange": "CME",
  "timezone": "America/Chicago",
  "holidays": [
    {
      "name": "New Year's Day",
      "type": "full_closure",
      "dates": ["2024-01-01", "2025-01-01", "2026-01-01"],
      "recurring": {
        "month": 1,
        "day": 1,
        "observance": "nearest_weekday"
      }
    },
    {
      "name": "Martin Luther King Jr. Day",
      "type": "full_closure",
      "recurring": {
        "month": 1,
        "weekday": "monday",
        "occurrence": 3
      }
    },
    {
      "name": "Presidents Day",
      "type": "full_closure",
      "recurring": {
        "month": 2,
        "weekday": "monday",
        "occurrence": 3
      }
    },
    {
      "name": "Good Friday",
      "type": "full_closure",
      "notes": "Calculated based on Easter"
    },
    {
      "name": "Memorial Day",
      "type": "full_closure",
      "recurring": {
        "month": 5,
        "weekday": "monday",
        "occurrence": "last"
      }
    },
    {
      "name": "Juneteenth",
      "type": "full_closure",
      "dates": ["2024-06-19", "2025-06-19", "2026-06-19"],
      "recurring": {
        "month": 6,
        "day": 19,
        "observance": "nearest_weekday"
      }
    },
    {
      "name": "Independence Day",
      "type": "full_closure",
      "recurring": {
        "month": 7,
        "day": 4,
        "observance": "nearest_weekday"
      }
    },
    {
      "name": "Labor Day",
      "type": "full_closure",
      "recurring": {
        "month": 9,
        "weekday": "monday",
        "occurrence": 1
      }
    },
    {
      "name": "Thanksgiving",
      "type": "full_closure",
      "recurring": {
        "month": 11,
        "weekday": "thursday",
        "occurrence": 4
      }
    },
    {
      "name": "Day After Thanksgiving",
      "type": "early_close",
      "close_time_et": "13:00",
      "close_time_utc_est": "18:00",
      "recurring": {
        "month": 11,
        "weekday": "friday",
        "occurrence": 4
      }
    },
    {
      "name": "Christmas Eve",
      "type": "early_close",
      "close_time_et": "13:00",
      "close_time_utc_est": "18:00",
      "dates": ["2024-12-24", "2025-12-24", "2026-12-24"],
      "recurring": {
        "month": 12,
        "day": 24,
        "only_if_weekday": true
      }
    },
    {
      "name": "Christmas Day",
      "type": "full_closure",
      "recurring": {
        "month": 12,
        "day": 25,
        "observance": "nearest_weekday"
      }
    }
  ],
  "special_closures": [
    {
      "name": "September 11, 2001",
      "date": "2001-09-11",
      "type": "emergency_closure"
    }
  ]
}
```

---

## 3. Implementation Approach

### 3.1 New Pipeline Stage vs. Modify Existing

**Recommendation: Create NEW Stage 2.5 (Session Filtering)**

**Rationale:**
- Stage 2 (data_cleaning) should remain focused on OHLC validation and resampling
- Session filtering is a distinct concern with different failure modes
- Allows optional use (some models may want 24-hour data)
- Keeps stage2_clean under 650 lines
- Enables A/B testing (session-filtered vs. full-day models)

**Pipeline Integration:**

```python
# Updated stage_registry.py
{
    "name": "session_filtering",
    "dependencies": ["data_cleaning"],
    "description": "Stage 2.5: Session-aware filtering and normalization",
    "required": False,  # Optional - can skip for 24-hour models
    "stage_number": 2.5
}
```

### 3.2 Component Implementations

#### 3.2.1 Session Configuration (`session_config.py`)

```python
"""
Session configuration schema and validation.

Responsibilities:
- Load session definitions from JSON
- Validate session schemas
- Provide session lookup utilities
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path
import json
from datetime import time


@dataclass
class SessionCharacteristics:
    """Market characteristics for a trading session."""
    typical_volume_pct: float
    expected_volatility: str  # 'low', 'medium', 'high'
    trend_behavior: str  # 'directional', 'range_bound', 'mixed'


@dataclass
class SessionDefinition:
    """Definition of a trading session."""
    name: str
    description: str
    start_time_et: time
    end_time_et: time
    start_time_utc_edt: time
    end_time_utc_edt: time
    start_time_utc_est: time
    end_time_utc_est: time
    days_of_week: List[int]  # 0=Monday, 6=Sunday
    wraps_midnight: bool = False
    enabled: bool = True
    priority: int = 1
    characteristics: Optional[SessionCharacteristics] = None

    def is_active_day(self, day_of_week: int) -> bool:
        """Check if session is active on given day of week."""
        return day_of_week in self.days_of_week

    def get_utc_times(self, is_dst: bool) -> tuple[time, time]:
        """Get UTC start/end times based on DST status."""
        if is_dst:
            return self.start_time_utc_edt, self.end_time_utc_edt
        else:
            return self.start_time_utc_est, self.end_time_utc_est


class SessionConfig:
    """Session configuration manager."""

    def __init__(self, config_path: Path):
        """Load and validate session configuration."""
        self.config_path = config_path
        self.sessions: Dict[str, SessionDefinition] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load session definitions from JSON."""
        with open(self.config_path) as f:
            data = json.load(f)

        # Validate schema version
        if data.get('version') != '1.0':
            raise ValueError(f"Unsupported config version: {data.get('version')}")

        # Parse sessions
        for session_id, session_data in data['sessions'].items():
            self.sessions[session_id] = self._parse_session(session_data)

    def _parse_session(self, data: dict) -> SessionDefinition:
        """Parse session definition from dict."""
        # Parse time strings
        start_et = time.fromisoformat(data['start_time_et'])
        end_et = time.fromisoformat(data['end_time_et'])
        start_utc_edt = time.fromisoformat(data['start_time_utc_edt'])
        end_utc_edt = time.fromisoformat(data['end_time_utc_edt'])
        start_utc_est = time.fromisoformat(data['start_time_utc_est'])
        end_utc_est = time.fromisoformat(data['end_time_utc_est'])

        # Parse characteristics
        chars = None
        if 'characteristics' in data:
            chars = SessionCharacteristics(**data['characteristics'])

        return SessionDefinition(
            name=data['name'],
            description=data['description'],
            start_time_et=start_et,
            end_time_et=end_et,
            start_time_utc_edt=start_utc_edt,
            end_time_utc_edt=end_utc_edt,
            start_time_utc_est=start_utc_est,
            end_time_utc_est=end_utc_est,
            days_of_week=data['days_of_week'],
            wraps_midnight=data.get('wraps_midnight', False),
            enabled=data.get('enabled', True),
            priority=data.get('priority', 1),
            characteristics=chars
        )

    def get_enabled_sessions(self) -> Dict[str, SessionDefinition]:
        """Get all enabled sessions."""
        return {k: v for k, v in self.sessions.items() if v.enabled}

    def get_session(self, session_id: str) -> SessionDefinition:
        """Get session by ID."""
        if session_id not in self.sessions:
            raise KeyError(f"Session '{session_id}' not found")
        return self.sessions[session_id]
```

#### 3.2.2 Timezone Handler (`timezone_handler.py`)

```python
"""
Timezone and DST handling.

Responsibilities:
- Convert ET to UTC accounting for DST
- Detect DST transitions
- Provide DST-aware time utilities
"""

from datetime import datetime, timezone
from typing import Optional
import pytz


class TimezoneHandler:
    """Handle timezone conversions and DST transitions."""

    def __init__(self):
        """Initialize timezone handler."""
        self.et_tz = pytz.timezone('America/New_York')
        self.utc_tz = pytz.UTC

    def is_dst(self, dt: datetime) -> bool:
        """
        Check if datetime is in DST.

        Parameters
        ----------
        dt : datetime
            Timezone-aware datetime in ET or UTC

        Returns
        -------
        bool
            True if datetime is in DST period
        """
        # Convert to ET if needed
        if dt.tzinfo != self.et_tz:
            dt = dt.astimezone(self.et_tz)

        # Check DST offset
        return bool(dt.dst())

    def et_to_utc(self, dt: datetime) -> datetime:
        """
        Convert ET datetime to UTC.

        Parameters
        ----------
        dt : datetime
            Timezone-aware or naive datetime in ET

        Returns
        -------
        datetime
            Timezone-aware datetime in UTC
        """
        # Localize if naive
        if dt.tzinfo is None:
            dt = self.et_tz.localize(dt)
        elif dt.tzinfo != self.et_tz:
            dt = dt.astimezone(self.et_tz)

        # Convert to UTC
        return dt.astimezone(self.utc_tz)

    def utc_to_et(self, dt: datetime) -> datetime:
        """
        Convert UTC datetime to ET.

        Parameters
        ----------
        dt : datetime
            Timezone-aware datetime in UTC

        Returns
        -------
        datetime
            Timezone-aware datetime in ET
        """
        if dt.tzinfo is None:
            dt = self.utc_tz.localize(dt)

        return dt.astimezone(self.et_tz)

    def get_utc_offset_hours(self, dt: datetime) -> int:
        """
        Get UTC offset in hours for ET datetime.

        Returns -4 during EDT, -5 during EST.
        """
        if dt.tzinfo is None:
            dt = self.et_tz.localize(dt)
        elif dt.tzinfo != self.et_tz:
            dt = dt.astimezone(self.et_tz)

        offset = dt.utcoffset()
        return int(offset.total_seconds() / 3600)
```

#### 3.2.3 Holiday Calendar (`holiday_calendar.py`)

**Approach: Use `pandas_market_calendars` library**

This is a well-maintained library that handles CME holidays, early closes, and special schedules.

```python
"""
Holiday calendar integration.

Uses pandas_market_calendars for CME Group holidays.
"""

from datetime import datetime, date
from typing import List, Optional
import pandas as pd
import pandas_market_calendars as mcal
from pathlib import Path


class HolidayCalendar:
    """CME Group holiday calendar."""

    def __init__(self, exchange: str = 'CME'):
        """
        Initialize holiday calendar.

        Parameters
        ----------
        exchange : str
            Exchange code (default: 'CME')
        """
        self.exchange = exchange
        self.calendar = mcal.get_calendar(exchange)

    def is_trading_day(self, dt: datetime) -> bool:
        """
        Check if date is a trading day.

        Parameters
        ----------
        dt : datetime
            Date to check

        Returns
        -------
        bool
            True if trading day, False if holiday
        """
        schedule = self.calendar.schedule(
            start_date=dt.date(),
            end_date=dt.date()
        )
        return len(schedule) > 0

    def is_early_close(self, dt: datetime) -> bool:
        """
        Check if date has early close.

        Parameters
        ----------
        dt : datetime
            Date to check

        Returns
        -------
        bool
            True if early close day
        """
        schedule = self.calendar.schedule(
            start_date=dt.date(),
            end_date=dt.date()
        )

        if len(schedule) == 0:
            return False

        # Check if close time is earlier than normal
        normal_close = pd.Timestamp('16:00', tz='America/New_York').time()
        actual_close = schedule.iloc[0]['market_close'].time()

        return actual_close < normal_close

    def get_early_close_time(self, dt: datetime) -> Optional[datetime]:
        """
        Get early close time if applicable.

        Parameters
        ----------
        dt : datetime
            Date to check

        Returns
        -------
        Optional[datetime]
            Early close time or None
        """
        if not self.is_early_close(dt):
            return None

        schedule = self.calendar.schedule(
            start_date=dt.date(),
            end_date=dt.date()
        )

        return schedule.iloc[0]['market_close']

    def filter_trading_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter DataFrame to trading days only.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'datetime' column

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame
        """
        # Get schedule for date range
        schedule = self.calendar.schedule(
            start_date=df['datetime'].min().date(),
            end_date=df['datetime'].max().date()
        )

        # Extract trading dates
        trading_dates = set(schedule.index.date)

        # Filter DataFrame
        df_filtered = df[df['datetime'].dt.date.isin(trading_dates)].copy()

        return df_filtered
```

#### 3.2.4 Session Filter (`session_filter.py`)

```python
"""
Session filtering for trading data.

Responsibilities:
- Filter data to specific sessions
- Handle session overlaps
- Add session indicator features
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from datetime import datetime
import logging

from .session_config import SessionConfig, SessionDefinition
from .timezone_handler import TimezoneHandler
from .holiday_calendar import HolidayCalendar

logger = logging.getLogger(__name__)


class SessionFilter:
    """Filter trading data by session."""

    def __init__(
        self,
        session_config: SessionConfig,
        timezone_handler: TimezoneHandler,
        holiday_calendar: Optional[HolidayCalendar] = None,
        respect_holidays: bool = True
    ):
        """
        Initialize session filter.

        Parameters
        ----------
        session_config : SessionConfig
            Session configuration
        timezone_handler : TimezoneHandler
            Timezone handler
        holiday_calendar : Optional[HolidayCalendar]
            Holiday calendar (optional)
        respect_holidays : bool
            Filter out holidays (default: True)
        """
        self.session_config = session_config
        self.tz_handler = timezone_handler
        self.holiday_calendar = holiday_calendar
        self.respect_holidays = respect_holidays

    def filter_to_session(
        self,
        df: pd.DataFrame,
        session_id: str
    ) -> pd.DataFrame:
        """
        Filter data to specific session.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'datetime' column (UTC)
        session_id : str
            Session identifier

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame
        """
        logger.info(f"Filtering to session: {session_id}")

        session = self.session_config.get_session(session_id)

        # Ensure datetime is UTC
        df = df.copy()
        if df['datetime'].dt.tz is None:
            df['datetime'] = pd.to_datetime(df['datetime'], utc=True)

        # Filter holidays if requested
        if self.respect_holidays and self.holiday_calendar:
            df = self.holiday_calendar.filter_trading_days(df)

        # Create session mask
        mask = self._create_session_mask(df, session)

        df_filtered = df[mask].copy()

        logger.info(
            f"Session filter: {len(df):,} -> {len(df_filtered):,} bars "
            f"({len(df_filtered)/len(df)*100:.1f}%)"
        )

        return df_filtered

    def _create_session_mask(
        self,
        df: pd.DataFrame,
        session: SessionDefinition
    ) -> np.ndarray:
        """Create boolean mask for session times."""
        mask = np.zeros(len(df), dtype=bool)

        for i, dt in enumerate(df['datetime']):
            # Check day of week
            if not session.is_active_day(dt.dayofweek):
                continue

            # Get session times based on DST
            is_dst = self.tz_handler.is_dst(dt)
            start_utc, end_utc = session.get_utc_times(is_dst)

            # Check time
            current_time = dt.time()

            if session.wraps_midnight:
                # Session crosses midnight
                mask[i] = current_time >= start_utc or current_time < end_utc
            else:
                # Normal session
                mask[i] = start_utc <= current_time < end_utc

        return mask

    def add_session_indicators(
        self,
        df: pd.DataFrame,
        sessions: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Add session indicator columns.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'datetime' column (UTC)
        sessions : Optional[List[str]]
            List of session IDs (default: all enabled)

        Returns
        -------
        pd.DataFrame
            DataFrame with session_<name> columns
        """
        logger.info("Adding session indicator features...")

        df = df.copy()

        if sessions is None:
            sessions = list(self.session_config.get_enabled_sessions().keys())

        for session_id in sessions:
            session = self.session_config.get_session(session_id)
            mask = self._create_session_mask(df, session)
            df[f'session_{session_id}'] = mask.astype(int)

        return df
```

#### 3.2.5 Session Normalizer (`session_normalizer.py`)

```python
"""
Session-aware feature normalization.

Responsibilities:
- Calculate session-specific statistics
- Normalize features by session
- Preserve train/test split integrity
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import logging
import json

from .session_filter import SessionFilter

logger = logging.getLogger(__name__)


class SessionNormalizer:
    """Session-aware feature normalization."""

    def __init__(
        self,
        session_filter: SessionFilter,
        feature_columns: List[str]
    ):
        """
        Initialize session normalizer.

        Parameters
        ----------
        session_filter : SessionFilter
            Session filter instance
        feature_columns : List[str]
            List of feature column names to normalize
        """
        self.session_filter = session_filter
        self.feature_columns = feature_columns
        self.session_stats: Dict[str, Dict[str, Dict[str, float]]] = {}

    def fit(
        self,
        df: pd.DataFrame,
        sessions: List[str]
    ) -> None:
        """
        Calculate session-specific statistics from training data.

        Parameters
        ----------
        df : pd.DataFrame
            Training data
        sessions : List[str]
            List of session IDs
        """
        logger.info("Calculating session statistics...")

        for session_id in sessions:
            session_data = self.session_filter.filter_to_session(df, session_id)

            if len(session_data) == 0:
                logger.warning(f"No data for session {session_id}")
                continue

            self.session_stats[session_id] = {}

            for col in self.feature_columns:
                if col not in session_data.columns:
                    continue

                self.session_stats[session_id][col] = {
                    'mean': float(session_data[col].mean()),
                    'std': float(session_data[col].std()),
                    'min': float(session_data[col].min()),
                    'max': float(session_data[col].max()),
                    'median': float(session_data[col].median()),
                }

        logger.info(f"Calculated stats for {len(sessions)} sessions")

    def transform(
        self,
        df: pd.DataFrame,
        method: str = 'zscore'
    ) -> pd.DataFrame:
        """
        Normalize features using session-specific statistics.

        Parameters
        ----------
        df : pd.DataFrame
            Data to normalize
        method : str
            Normalization method ('zscore', 'minmax', 'robust')

        Returns
        -------
        pd.DataFrame
            Normalized DataFrame
        """
        logger.info(f"Normalizing features using method: {method}")

        df = df.copy()

        # Add session indicators
        df = self.session_filter.add_session_indicators(df)

        # Normalize each row based on its session
        for col in self.feature_columns:
            if col not in df.columns:
                continue

            normalized = np.zeros(len(df))

            for session_id, stats in self.session_stats.items():
                if col not in stats:
                    continue

                # Get session mask
                session_col = f'session_{session_id}'
                if session_col not in df.columns:
                    continue

                mask = df[session_col] == 1

                if method == 'zscore':
                    mean = stats[col]['mean']
                    std = stats[col]['std']
                    if std > 0:
                        normalized[mask] = (df.loc[mask, col] - mean) / std

                elif method == 'minmax':
                    min_val = stats[col]['min']
                    max_val = stats[col]['max']
                    if max_val > min_val:
                        normalized[mask] = (df.loc[mask, col] - min_val) / (max_val - min_val)

                elif method == 'robust':
                    median = stats[col]['median']
                    # Use IQR-based scaling
                    q75 = stats[col].get('q75', median)
                    q25 = stats[col].get('q25', median)
                    iqr = q75 - q25
                    if iqr > 0:
                        normalized[mask] = (df.loc[mask, col] - median) / iqr

            df[f'{col}_norm'] = normalized

        return df

    def save_stats(self, output_path: Path) -> None:
        """Save session statistics to JSON."""
        with open(output_path, 'w') as f:
            json.dump(self.session_stats, f, indent=2)
        logger.info(f"Saved session stats to {output_path}")

    def load_stats(self, input_path: Path) -> None:
        """Load session statistics from JSON."""
        with open(input_path) as f:
            self.session_stats = json.load(f)
        logger.info(f"Loaded session stats from {input_path}")
```

---

## 4. Integration with Existing Pipeline

### 4.1 Stage 2.5 Implementation

**File: `src/stages/stage2_5_sessions.py`**

```python
"""
Stage 2.5: Session Filtering and Normalization

Optional stage that filters data to specific trading sessions
and performs session-aware feature normalization.
"""

from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import logging

from src.stages.sessions import (
    SessionConfig,
    TimezoneHandler,
    HolidayCalendar,
    SessionFilter,
    SessionNormalizer
)
from src.config import CONFIG_DIR, CLEAN_DATA_DIR, FEATURES_DIR

logger = logging.getLogger(__name__)


def session_filtering(
    input_dir: Path = CLEAN_DATA_DIR,
    output_dir: Path = FEATURES_DIR,
    sessions: Optional[list] = None,
    normalize: bool = False,
    respect_holidays: bool = True,
    **kwargs
) -> Dict[str, Path]:
    """
    Filter data to specific trading sessions.

    Parameters
    ----------
    input_dir : Path
        Input directory with cleaned data
    output_dir : Path
        Output directory
    sessions : Optional[list]
        List of session IDs to filter (default: ['new_york'])
    normalize : bool
        Perform session-aware normalization (default: False)
    respect_holidays : bool
        Filter out holidays (default: True)

    Returns
    -------
    Dict[str, Path]
        Dictionary mapping symbols to output paths
    """
    if sessions is None:
        sessions = ['new_york']  # Default to NY session only

    logger.info(f"Session filtering: sessions={sessions}")

    # Initialize components
    config_path = CONFIG_DIR / 'sessions' / 'cme_sessions.json'
    session_config = SessionConfig(config_path)
    tz_handler = TimezoneHandler()
    holiday_cal = HolidayCalendar('CME') if respect_holidays else None

    session_filter = SessionFilter(
        session_config=session_config,
        timezone_handler=tz_handler,
        holiday_calendar=holiday_cal,
        respect_holidays=respect_holidays
    )

    # Process each symbol
    results = {}
    for file_path in input_dir.glob('*.parquet'):
        symbol = file_path.stem
        logger.info(f"Processing {symbol}...")

        df = pd.read_parquet(file_path)

        # Filter to requested sessions
        if len(sessions) == 1:
            df_filtered = session_filter.filter_to_session(df, sessions[0])
        else:
            # Multiple sessions: combine with indicators
            df_filtered = session_filter.add_session_indicators(df, sessions)
            # Keep only rows in any session
            session_cols = [f'session_{s}' for s in sessions]
            mask = df_filtered[session_cols].sum(axis=1) > 0
            df_filtered = df_filtered[mask]

        # Save filtered data
        output_path = output_dir / f'{symbol}_session_filtered.parquet'
        df_filtered.to_parquet(output_path, index=False)
        results[symbol] = output_path

        logger.info(f"Saved {symbol}: {len(df_filtered):,} bars")

    return results
```

### 4.2 Config Integration

**Add to `src/config.py`:**

```python
# Session filtering configuration
SESSION_FILTERING = {
    'enabled': False,  # Optional feature
    'sessions': ['new_york'],  # Default to NY session only
    'respect_holidays': True,
    'normalize_by_session': False
}
```

---

## 5. Example Session-Filtered Output

### 5.1 Before Session Filtering

```
MES.parquet: 100,000 bars (24-hour coverage)
- Asia session: 15,000 bars (15%)
- London session: 25,000 bars (25%)
- NY session: 60,000 bars (60%)

Volatility (ATR):
- Overall: 2.5 points
- Asia: 1.2 points
- London: 2.0 points
- NY: 3.5 points
```

### 5.2 After Session Filtering (NY Only)

```
MES_session_filtered.parquet: 60,000 bars
- NY session: 60,000 bars (100%)
- Holidays removed: 2,500 bars

Feature normalization:
- ATR normalized to NY session stats (mean=3.5, std=0.8)
- Volume normalized to NY session stats
- Returns normalized to NY session stats

Session features added:
- session_ny: 1 for all bars
- session_london: 0 for all bars
- session_asia: 0 for all bars
```

### 5.3 After Multi-Session with Indicators

```
MES_session_filtered.parquet: 85,000 bars
- NY session: 60,000 bars
- London session: 25,000 bars

Session features:
- session_ny: [1, 1, 1, ..., 0, 0]
- session_london: [0, 0, 0, ..., 1, 1]
- session_asia: [0, 0, 0, ..., 0, 0]

Normalized features:
- ATR_norm: Normalized using session-specific stats
- volume_norm: Normalized using session-specific stats
```

---

## 6. Testing Strategy

### 6.1 Unit Tests

```python
# test_timezone_handler.py
def test_dst_detection():
    """Test DST detection."""
    tz = TimezoneHandler()

    # Summer (EDT)
    summer_dt = datetime(2024, 7, 15, 12, 0, tzinfo=pytz.UTC)
    assert tz.is_dst(summer_dt) == True

    # Winter (EST)
    winter_dt = datetime(2024, 1, 15, 12, 0, tzinfo=pytz.UTC)
    assert tz.is_dst(winter_dt) == False

def test_et_to_utc_conversion():
    """Test ET to UTC conversion."""
    tz = TimezoneHandler()

    # 9:30 ET in summer (EDT) = 13:30 UTC
    et_dt = datetime(2024, 7, 15, 9, 30)
    utc_dt = tz.et_to_utc(et_dt)
    assert utc_dt.hour == 13
    assert utc_dt.minute == 30
```

### 6.2 Integration Tests

```python
# test_session_filter.py
def test_session_filtering():
    """Test session filtering."""
    # Create test data covering 24 hours
    df = create_test_24h_data()

    filter = create_session_filter()
    df_ny = filter.filter_to_session(df, 'new_york')

    # Verify NY session only (9:30-16:00 ET)
    assert len(df_ny) < len(df)
    assert all(df_ny['datetime'].dt.hour >= 13)  # EDT
    assert all(df_ny['datetime'].dt.hour < 20)
```

---

## 7. Performance Considerations

### 7.1 Optimization Strategies

1. **Vectorized Operations**: Use NumPy masks instead of row-by-row filtering
2. **Caching**: Cache DST transition dates
3. **Parallel Processing**: Process symbols in parallel
4. **Lazy Loading**: Only load session config when needed

### 7.2 Expected Performance

```
Dataset: 100K bars per symbol, 2 symbols
Session filtering: ~500ms per symbol
Normalization: ~200ms per symbol
Total: <2 seconds for full pipeline
```

---

## 8. Future Extensions

### 8.1 Regime-Adaptive Sessions

```python
# Automatically adjust session boundaries based on volume
def detect_active_hours(df: pd.DataFrame) -> tuple:
    """Detect actual active trading hours from volume."""
    hourly_vol = df.groupby(df['datetime'].dt.hour)['volume'].mean()
    threshold = hourly_vol.mean()
    active_hours = hourly_vol[hourly_vol > threshold].index
    return active_hours.min(), active_hours.max()
```

### 8.2 Session Transition Features

```python
# Add features for session transitions
def add_session_transition_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features for session open/close."""
    df['session_open'] = df['session_ny'].diff() == 1
    df['session_close'] = df['session_ny'].diff() == -1
    df['time_in_session'] = df.groupby('session_ny').cumcount()
    return df
```

---

## 9. Summary

This design provides a comprehensive, modular session-aware normalization system that:

1. **Separates concerns**: Each component has a single responsibility
2. **Stays modular**: All files under 650 lines
3. **Fails fast**: Validates timezones, sessions, and holidays at boundaries
4. **Integrates cleanly**: Optional Stage 2.5 doesn't break existing pipeline
5. **Tests thoroughly**: Unit and integration tests for all components

**Next Steps:**

1. Implement core components (session_config, timezone_handler)
2. Add holiday calendar integration
3. Build session filter with tests
4. Add session normalizer
5. Create Stage 2.5 integration
6. Write comprehensive tests
7. Update documentation

**Total LOC Estimate:**
- Core components: ~1,200 lines
- Stage 2.5: ~200 lines
- Tests: ~800 lines
- Config files: ~200 lines (JSON)
- **Total: ~2,400 lines** (well within modular limits)
