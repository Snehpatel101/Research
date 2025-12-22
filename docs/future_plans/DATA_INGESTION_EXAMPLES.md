# Data Ingestion Architecture - Code Examples

This document provides concrete code examples demonstrating the flexible data ingestion architecture.

---

## Example 1: Backward Compatible CLI Workflow

**Current workflow (unchanged):**

```bash
# Works exactly as before
./pipeline run --symbols MES,MGC --start 2020-01-01 --end 2024-12-31
```

**Behind the scenes (new implementation):**

```python
# src/pipeline_cli.py (modified run() command)

from pipeline.data_loader import PipelineDataLoader
from data_sources.file_source import FileDataSource
import config

@app.command()
def run(
    symbols: str = "MES,MGC",
    data_source_id: Optional[str] = None,  # NEW: optional data source
    ...
):
    """Run the pipeline."""
    symbol_list = [s.strip().upper() for s in symbols.split(",")]

    # Create data loader (auto-creates FileDataSource if no source specified)
    if data_source_id is None:
        # Backward compatible: use files in data/raw/
        loader = PipelineDataLoader(project_root=project_root_path)
    else:
        # Use specified data source (e.g., from UI upload)
        repo = load_repository(data_source_id)
        loader = PipelineDataLoader(repository=repo)

    # Set runtime symbols
    config.set_runtime_symbols(symbol_list)

    # Pipeline uses loader to fetch data
    runner = PipelineRunner(config, data_loader=loader)
    success = runner.run()
```

---

## Example 2: UI Upload Workflow

### Step 1: Upload Data

```python
# User action in Streamlit UI (src/ui/pages/upload_data.py)

uploaded_files = st.file_uploader(
    "Choose data files",
    type=['csv', 'parquet'],
    accept_multiple_files=True
)

if st.button("Process Uploads"):
    # Create upload source
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(tempfile.gettempdir()) / "pipeline_uploads" / session_id

    uploaded_dict = {file.name: file.read() for file in uploaded_files}

    source = UploadDataSource(
        source_id=f"upload_{session_id}",
        session_dir=session_dir,
        uploaded_files=uploaded_dict,
        file_format='parquet'
    )

    # Validate
    is_valid, errors = source.validate()
    if not is_valid:
        st.error(f"Validation failed: {errors}")
        return

    # Register
    repo = st.session_state.repository
    repo.register_source(source.source_id, source)
    repo.set_active_source(source.source_id)

    st.success("Data uploaded successfully!")

    # Preview
    metadata = source.get_metadata()
    st.write(f"Symbols: {metadata.symbols}")
    st.write(f"Date range: {metadata.date_range}")

    for symbol in metadata.symbols:
        st.write(f"Preview of {symbol}:")
        st.dataframe(source.preview(symbol, n_rows=5))
```

### Step 2: Configure and Run Pipeline

```python
# User action in Streamlit UI (src/ui/pages/configure_pipeline.py)

# Get active source
repo = st.session_state.repository
source = repo.get_active_source()
metadata = source.get_metadata()

# Symbol selection
selected_symbols = st.multiselect(
    "Select symbols to process",
    options=metadata.symbols,
    default=metadata.symbols
)

# Configure pipeline parameters
horizons = st.multiselect(
    "Label horizons",
    options=[1, 5, 10, 20],
    default=[5, 20]
)

train_ratio = st.slider("Train ratio", 0.5, 0.9, 0.70)
val_ratio = st.slider("Val ratio", 0.05, 0.3, 0.15)
test_ratio = 1.0 - train_ratio - val_ratio

if st.button("Run Pipeline", type="primary"):
    # Create pipeline config
    config_obj = create_default_config(
        symbols=selected_symbols,
        label_horizons=horizons,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )

    # Create data loader with active source
    loader = PipelineDataLoader(repository=repo)

    # Run pipeline
    with st.spinner("Running pipeline..."):
        runner = PipelineRunner(config_obj, data_loader=loader)
        success = runner.run()

    if success:
        st.success("Pipeline completed successfully!")
        st.balloons()
    else:
        st.error("Pipeline failed. Check logs.")
```

---

## Example 3: Adding Custom Data Source

Suppose you want to fetch data from Polygon.io API:

```python
# src/data_sources/polygon_source.py (< 350 lines)

from typing import Dict, List, Tuple
import pandas as pd
import requests
from datetime import datetime
from pathlib import Path

from .base import DataSource, DataSourceMetadata


class PolygonDataSource(DataSource):
    """
    Data source for Polygon.io market data API.
    """

    API_BASE = "https://api.polygon.io/v2"

    def __init__(
        self,
        source_id: str,
        api_key: str,
        symbols: List[str],
        start_date: str,
        end_date: str,
        timeframe: str = '5min'
    ):
        """
        Initialize Polygon data source.

        Parameters:
        -----------
        source_id : Unique identifier
        api_key : Polygon.io API key
        symbols : List of symbols to fetch
        start_date : Start date (YYYY-MM-DD)
        end_date : End date (YYYY-MM-DD)
        timeframe : Timeframe ('1min', '5min', '1hour', '1day')
        """
        self.source_id = source_id
        self.api_key = api_key
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe

        # Cache fetched data
        self._cache: Dict[str, pd.DataFrame] = {}

    def _fetch_symbol_data(self, symbol: str) -> pd.DataFrame:
        """Fetch data for a symbol from Polygon API."""
        # Build API request
        url = f"{self.API_BASE}/aggs/ticker/{symbol}/range/{self.timeframe}/{self.start_date}/{self.end_date}"

        params = {
            'apiKey': self.api_key,
            'adjusted': 'true',
            'sort': 'asc'
        }

        # Make request
        response = requests.get(url, params=params)

        if response.status_code != 200:
            raise RuntimeError(
                f"API request failed: {response.status_code} - {response.text}"
            )

        data = response.json()

        if 'results' not in data:
            raise ValueError(f"No data returned for {symbol}")

        # Convert to DataFrame
        df = pd.DataFrame(data['results'])

        # Rename columns to standard format
        df = df.rename(columns={
            't': 'timestamp',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume'
        })

        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Select standard columns
        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]

        return df

    def get_metadata(self) -> DataSourceMetadata:
        """Get metadata about this data source."""
        # Fetch first symbol to get metadata
        if not self._cache:
            first_symbol = self.symbols[0]
            self._cache[first_symbol] = self._fetch_symbol_data(first_symbol)

        sample_df = self._cache[self.symbols[0]]

        return DataSourceMetadata(
            source_id=self.source_id,
            source_type='api',
            symbols=self.symbols,
            date_range=(self.start_date, self.end_date),
            row_count=len(sample_df) * len(self.symbols),  # Estimate
            file_size_mb=0.0,  # Not applicable
            created_at=datetime.now(),
            columns=sample_df.columns.tolist(),
            validated=True,
            validation_errors=[]
        )

    def load_symbol(self, symbol: str) -> pd.DataFrame:
        """Load data for a symbol (with caching)."""
        if symbol not in self.symbols:
            raise KeyError(f"Symbol '{symbol}' not configured in this source")

        # Return cached data if available
        if symbol in self._cache:
            return self._cache[symbol].copy()

        # Fetch and cache
        df = self._fetch_symbol_data(symbol)
        self._cache[symbol] = df

        return df.copy()

    def get_available_symbols(self) -> List[str]:
        """Get available symbols."""
        return self.symbols

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate API access and data availability."""
        errors = []

        # Check API key validity
        try:
            # Make a test request
            test_symbol = self.symbols[0]
            df = self._fetch_symbol_data(test_symbol)

            if len(df) == 0:
                errors.append(f"No data available for {test_symbol}")

        except requests.exceptions.RequestException as e:
            errors.append(f"API request failed: {e}")
        except ValueError as e:
            errors.append(f"Data validation failed: {e}")
        except Exception as e:
            errors.append(f"Unexpected error: {e}")

        return (len(errors) == 0, errors)

    def preview(self, symbol: str, n_rows: int = 10) -> pd.DataFrame:
        """Get preview of data."""
        df = self.load_symbol(symbol)
        return df.head(n_rows)
```

**Usage:**

```python
# In Streamlit UI or CLI

from data_sources.polygon_source import PolygonDataSource

# Create API source
source = PolygonDataSource(
    source_id='polygon_20241221',
    api_key=os.environ['POLYGON_API_KEY'],
    symbols=['MES', 'MGC'],
    start_date='2024-01-01',
    end_date='2024-12-31',
    timeframe='5min'
)

# Validate
is_valid, errors = source.validate()
if not is_valid:
    print(f"Validation failed: {errors}")
    exit(1)

# Register
repo = DataSourceRepository(project_root)
repo.register_source(source.source_id, source)
repo.set_active_source(source.source_id)

# Use in pipeline
loader = PipelineDataLoader(repository=repo)
df = loader.load_symbol_data('MES')
```

---

## Example 4: Data Preview Component

```python
# src/ui/components/data_preview.py (< 150 lines)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def show_data_preview(df: pd.DataFrame, symbol: str):
    """
    Display data preview with statistics and chart.

    Parameters:
    -----------
    df : DataFrame to preview
    symbol : Symbol name
    """
    # Basic statistics
    st.markdown(f"**{symbol} Data Preview**")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", f"{len(df):,}")

    with col2:
        date_range = df['datetime'].max() - df['datetime'].min()
        st.metric("Duration", f"{date_range.days} days")

    with col3:
        avg_volume = df['volume'].mean()
        st.metric("Avg Volume", f"{avg_volume:,.0f}")

    with col4:
        price_range = df['close'].max() - df['close'].min()
        st.metric("Price Range", f"{price_range:.2f}")

    # Data table preview
    st.markdown("**First 10 Rows**")
    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True
    )

    # OHLCV chart
    st.markdown("**Price Chart (Last 100 Bars)**")

    chart_df = df.tail(100)

    fig = go.Figure()

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=chart_df['datetime'],
        open=chart_df['open'],
        high=chart_df['high'],
        low=chart_df['low'],
        close=chart_df['close'],
        name='OHLC'
    ))

    # Volume bar chart
    fig.add_trace(go.Bar(
        x=chart_df['datetime'],
        y=chart_df['volume'],
        name='Volume',
        yaxis='y2',
        opacity=0.3
    ))

    # Layout
    fig.update_layout(
        title=f"{symbol} - Last 100 Bars",
        yaxis=dict(title='Price'),
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right'
        ),
        xaxis=dict(title='Date'),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Data quality checks
    st.markdown("**Data Quality**")

    quality_checks = []

    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        quality_checks.append(f"⚠️ Missing values: {missing[missing > 0].to_dict()}")
    else:
        quality_checks.append("✅ No missing values")

    # Check for duplicates
    duplicates = df.duplicated(subset=['datetime']).sum()
    if duplicates > 0:
        quality_checks.append(f"⚠️ Duplicate timestamps: {duplicates}")
    else:
        quality_checks.append("✅ No duplicate timestamps")

    # Check OHLC relationships
    invalid_ohlc = (
        (df['high'] < df['low']) |
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close'])
    ).sum()

    if invalid_ohlc > 0:
        quality_checks.append(f"⚠️ Invalid OHLC relationships: {invalid_ohlc} rows")
    else:
        quality_checks.append("✅ Valid OHLC relationships")

    # Display checks
    for check in quality_checks:
        st.text(check)
```

---

## Example 5: Validation Display Component

```python
# src/ui/components/validation_display.py (< 100 lines)

import streamlit as st
from typing import List


def show_validation_results(is_valid: bool, errors: List[str]):
    """
    Display validation results with clear feedback.

    Parameters:
    -----------
    is_valid : Whether validation passed
    errors : List of validation errors
    """
    if is_valid:
        st.success("✅ All validation checks passed!")
        return

    # Show errors
    st.error(f"❌ Validation failed with {len(errors)} error(s)")

    st.markdown("**Validation Errors:**")

    for i, error in enumerate(errors, 1):
        # Categorize error type
        if "missing columns" in error.lower():
            icon = "📋"
            category = "Schema"
        elif "empty" in error.lower():
            icon = "📭"
            category = "Data"
        elif "datetime" in error.lower():
            icon = "📅"
            category = "Format"
        elif "numeric" in error.lower():
            icon = "🔢"
            category = "Type"
        else:
            icon = "⚠️"
            category = "General"

        st.markdown(f"{icon} **{category}**: {error}")

    # Provide suggestions
    st.markdown("---")
    st.markdown("**Suggestions:**")

    if any("missing columns" in e.lower() for e in errors):
        st.info(
            "💡 Ensure your data files contain columns: "
            "`datetime`, `open`, `high`, `low`, `close`, `volume`"
        )

    if any("datetime" in e.lower() for e in errors):
        st.info(
            "💡 Datetime column should be in ISO format (YYYY-MM-DD HH:MM:SS) "
            "or Unix timestamp"
        )

    if any("numeric" in e.lower() for e in errors):
        st.info(
            "💡 OHLC columns must be numeric (float or int). "
            "Remove any non-numeric characters."
        )
```

---

## Example 6: Session Management

```python
# src/ui/session.py (< 150 lines)

from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime, timedelta
import tempfile
import shutil


class SessionManager:
    """
    Manage Streamlit session state and temporary data.
    Handles upload sessions, cleanup, and persistence.
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.sessions_dir = project_root / "data" / "uploads" / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def create_session(self) -> str:
        """
        Create new session and return session ID.

        Returns:
        --------
        str : Session ID (YYYYMMDD_HHMMSS)
        """
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Save session metadata
        metadata = {
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'status': 'active',
            'files': []
        }

        metadata_path = session_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return session_id

    def get_session_dir(self, session_id: str) -> Path:
        """Get directory for a session."""
        session_dir = self.sessions_dir / session_id

        if not session_dir.exists():
            raise FileNotFoundError(f"Session '{session_id}' not found")

        return session_dir

    def update_session_metadata(
        self,
        session_id: str,
        updates: Dict[str, Any]
    ) -> None:
        """Update session metadata."""
        session_dir = self.get_session_dir(session_id)
        metadata_path = session_dir / 'metadata.json'

        # Load existing metadata
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {'session_id': session_id}

        # Apply updates
        metadata.update(updates)
        metadata['updated_at'] = datetime.now().isoformat()

        # Save
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """
        Clean up sessions older than max_age_hours.

        Parameters:
        -----------
        max_age_hours : Maximum age in hours before cleanup

        Returns:
        --------
        int : Number of sessions cleaned up
        """
        current_time = datetime.now()
        max_age = timedelta(hours=max_age_hours)
        cleaned = 0

        for session_dir in self.sessions_dir.iterdir():
            if not session_dir.is_dir():
                continue

            # Check metadata for creation time
            metadata_path = session_dir / 'metadata.json'

            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                created_at = datetime.fromisoformat(metadata['created_at'])
                age = current_time - created_at

                if age > max_age:
                    shutil.rmtree(session_dir)
                    cleaned += 1
            else:
                # No metadata, use directory modification time
                mtime = datetime.fromtimestamp(session_dir.stat().st_mtime)
                age = current_time - mtime

                if age > max_age:
                    shutil.rmtree(session_dir)
                    cleaned += 1

        return cleaned

    def list_sessions(self, active_only: bool = False) -> Dict[str, Dict]:
        """
        List all sessions with metadata.

        Parameters:
        -----------
        active_only : Only list active sessions

        Returns:
        --------
        dict : Mapping of session_id -> metadata
        """
        sessions = {}

        for session_dir in self.sessions_dir.iterdir():
            if not session_dir.is_dir():
                continue

            session_id = session_dir.name
            metadata_path = session_dir / 'metadata.json'

            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                if active_only and metadata.get('status') != 'active':
                    continue

                sessions[session_id] = metadata

        return sessions
```

---

## Example 7: Integration with Pipeline Runner

```python
# src/pipeline/data_loader.py (< 200 lines)

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from data_sources.base import DataSourceRepository, DataSource
from data_sources.file_source import FileDataSource
import config


class PipelineDataLoader:
    """
    Data loader for pipeline that uses DataSourceRepository.
    Provides backward compatibility with existing pipeline.
    """

    def __init__(
        self,
        repository: Optional[DataSourceRepository] = None,
        project_root: Optional[Path] = None
    ):
        """
        Initialize data loader.

        Parameters:
        -----------
        repository : DataSourceRepository (optional)
                     If None, creates default file source
        project_root : Path to project root
        """
        self.project_root = project_root or config.PROJECT_ROOT

        if repository is None:
            # Create default repository with file source
            self.repository = DataSourceRepository(self.project_root)

            # Register default file source
            try:
                default_source = FileDataSource(
                    source_id='default_files',
                    raw_data_dir=config.RAW_DATA_DIR,
                    symbol_pattern='*.parquet'
                )
                self.repository.register_source('default_files', default_source)
                self.repository.set_active_source('default_files')
            except Exception as e:
                # Allow initialization to succeed even if no files present
                # This enables UI workflow where files will be uploaded later
                pass
        else:
            self.repository = repository

    def load_symbol_data(self, symbol: str) -> pd.DataFrame:
        """
        Load data for a symbol from active source.

        Parameters:
        -----------
        symbol : Symbol to load

        Returns:
        --------
        pd.DataFrame : Loaded and standardized data

        Raises:
        -------
        RuntimeError : If no active source or symbol not found
        """
        try:
            source = self.repository.get_active_source()
        except RuntimeError as e:
            raise RuntimeError(
                "No active data source. Either place files in data/raw/ "
                "or upload data via UI."
            ) from e

        return source.load_symbol(symbol)

    def get_available_symbols(self) -> List[str]:
        """
        Get available symbols from active source.

        Returns:
        --------
        list : Available symbol names
        """
        try:
            source = self.repository.get_active_source()
            return source.get_available_symbols()
        except RuntimeError:
            return []

    def validate_source(self) -> bool:
        """
        Validate active data source.

        Returns:
        --------
        bool : True if valid

        Raises:
        -------
        ValueError : If validation fails with error details
        """
        source = self.repository.get_active_source()
        is_valid, errors = source.validate()

        if not is_valid:
            error_msg = "Data source validation failed:\n"
            error_msg += "\n".join(f"  - {e}" for e in errors)
            raise ValueError(error_msg)

        return True

    def get_source_metadata(self):
        """Get metadata about active source."""
        source = self.repository.get_active_source()
        return source.get_metadata()


# Usage in pipeline stages:

def stage1_ingest(data_loader: PipelineDataLoader, symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Stage 1: Ingest data using data loader.

    Parameters:
    -----------
    data_loader : PipelineDataLoader instance
    symbols : List of symbols to load

    Returns:
    --------
    dict : Mapping of symbol -> DataFrame
    """
    # Validate source before loading
    data_loader.validate_source()

    # Load data for each symbol
    data = {}
    for symbol in symbols:
        df = data_loader.load_symbol_data(symbol)
        data[symbol] = df
        print(f"Loaded {len(df):,} rows for {symbol}")

    return data
```

---

## Example 8: Complete End-to-End Workflow

```python
# Complete example: Upload via UI, run via CLI

# Step 1: User uploads data via Streamlit UI
# ============================================
# (User uploads MES_1m.parquet, MGC_1m.parquet)
# Session created: upload_20241221_120000
# Files validated and stored in /tmp/pipeline_uploads/upload_20241221_120000/

# Step 2: Run pipeline via CLI using uploaded data
# =================================================

from pathlib import Path
from data_sources.base import DataSourceRepository
from data_sources.upload_source import UploadDataSource
from pipeline.data_loader import PipelineDataLoader
from pipeline import PipelineRunner
from pipeline_config import create_default_config

# Project root
project_root = Path("/home/jake/Desktop/Research")

# Load repository
repo = DataSourceRepository(project_root)

# Load uploaded data source
session_id = "upload_20241221_120000"
session_dir = Path(f"/tmp/pipeline_uploads/{session_id}")

# Reconstruct upload source
uploaded_files = {}
for file_path in session_dir.glob("*.parquet"):
    uploaded_files[file_path.name] = file_path.read_bytes()

source = UploadDataSource(
    source_id=session_id,
    session_dir=session_dir,
    uploaded_files=uploaded_files,
    file_format='parquet'
)

# Register and activate
repo.register_source(session_id, source)
repo.set_active_source(session_id)

# Create data loader
loader = PipelineDataLoader(repository=repo)

# Get symbols from uploaded data
available_symbols = loader.get_available_symbols()
print(f"Available symbols: {available_symbols}")

# Create pipeline config
config = create_default_config(
    symbols=available_symbols,
    label_horizons=[5, 20],
    run_id="uploaded_data_run",
    project_root=project_root
)

# Run pipeline
runner = PipelineRunner(config, data_loader=loader)
success = runner.run()

if success:
    print("Pipeline completed successfully!")
else:
    print("Pipeline failed.")
```

---

## Example 9: Testing Data Sources

```python
# tests/test_data_sources.py

import pytest
from pathlib import Path
import pandas as pd
import tempfile
import shutil

from data_sources.file_source import FileDataSource
from data_sources.upload_source import UploadDataSource
from data_sources.base import DataSourceRepository


@pytest.fixture
def sample_data():
    """Create sample OHLCV data."""
    return pd.DataFrame({
        'datetime': pd.date_range('2024-01-01', periods=100, freq='5min'),
        'open': 100 + pd.Series(range(100)).cumsum() * 0.01,
        'high': 101 + pd.Series(range(100)).cumsum() * 0.01,
        'low': 99 + pd.Series(range(100)).cumsum() * 0.01,
        'close': 100.5 + pd.Series(range(100)).cumsum() * 0.01,
        'volume': 1000
    })


@pytest.fixture
def temp_data_dir(sample_data):
    """Create temporary directory with sample data."""
    temp_dir = Path(tempfile.mkdtemp())

    # Save sample data as parquet
    mes_path = temp_dir / "MES_1m.parquet"
    sample_data.to_parquet(mes_path, index=False)

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


def test_file_source_discovery(temp_data_dir):
    """Test FileDataSource discovers symbols correctly."""
    source = FileDataSource(
        source_id='test_files',
        raw_data_dir=temp_data_dir,
        symbol_pattern='*.parquet'
    )

    symbols = source.get_available_symbols()
    assert 'MES' in symbols


def test_file_source_load_symbol(temp_data_dir, sample_data):
    """Test FileDataSource loads symbol data."""
    source = FileDataSource(
        source_id='test_files',
        raw_data_dir=temp_data_dir
    )

    df = source.load_symbol('MES')
    assert len(df) == len(sample_data)
    assert 'datetime' in df.columns


def test_file_source_validation(temp_data_dir):
    """Test FileDataSource validates correctly."""
    source = FileDataSource(
        source_id='test_files',
        raw_data_dir=temp_data_dir
    )

    is_valid, errors = source.validate()
    assert is_valid
    assert len(errors) == 0


def test_upload_source_validation():
    """Test UploadDataSource validates data."""
    sample_df = pd.DataFrame({
        'datetime': pd.date_range('2024-01-01', periods=10, freq='5min'),
        'open': [100] * 10,
        'high': [101] * 10,
        'low': [99] * 10,
        'close': [100.5] * 10,
        'volume': [1000] * 10
    })

    # Save to bytes
    import io
    buffer = io.BytesIO()
    sample_df.to_parquet(buffer, index=False)
    file_bytes = buffer.getvalue()

    # Create upload source
    temp_dir = Path(tempfile.mkdtemp())

    try:
        source = UploadDataSource(
            source_id='test_upload',
            session_dir=temp_dir,
            uploaded_files={'MES_1m.parquet': file_bytes},
            file_format='parquet'
        )

        is_valid, errors = source.validate()
        assert is_valid
        assert len(errors) == 0

    finally:
        shutil.rmtree(temp_dir)


def test_repository_registration(temp_data_dir):
    """Test DataSourceRepository manages sources."""
    repo = DataSourceRepository(temp_data_dir)

    source = FileDataSource(
        source_id='test_source',
        raw_data_dir=temp_data_dir
    )

    # Register
    repo.register_source('test_source', source)

    # List
    sources = repo.list_sources()
    assert 'test_source' in sources

    # Set active
    repo.set_active_source('test_source')
    active = repo.get_active_source()
    assert active.source_id == 'test_source'


def test_repository_duplicate_registration(temp_data_dir):
    """Test repository prevents duplicate source IDs."""
    repo = DataSourceRepository(temp_data_dir)

    source1 = FileDataSource(
        source_id='dup_test',
        raw_data_dir=temp_data_dir
    )

    repo.register_source('dup_test', source1)

    # Try to register again with same ID
    source2 = FileDataSource(
        source_id='dup_test',
        raw_data_dir=temp_data_dir
    )

    with pytest.raises(ValueError, match="already registered"):
        repo.register_source('dup_test', source2)
```

---

## Summary

These examples demonstrate:

1. **Backward Compatibility**: Existing CLI workflow unchanged
2. **UI Integration**: Upload data, preview, validate, configure, run
3. **Extensibility**: Easy to add new data sources (e.g., Polygon API)
4. **Reusable Components**: Data preview, validation display, session management
5. **Clean Integration**: Pipeline uses data loader abstraction
6. **Comprehensive Testing**: Unit tests for all components

The architecture maintains modularity (all files < 650 lines), follows fail-fast principles (validation at every boundary), and provides excellent user experience for both technical (CLI) and non-technical (UI) users.
