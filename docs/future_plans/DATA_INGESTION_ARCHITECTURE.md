# Data Ingestion Architecture - Flexible Data Source Management

## Executive Summary

Design a modular, extensible data ingestion system that supports both CLI and Streamlit UI workflows, replacing hardcoded data sources with dynamic upload/selection while maintaining backward compatibility with existing pipeline.

**Key Objectives:**
1. Abstract data sources using Strategy pattern
2. Support multiple input methods (file upload, directory selection, future API/DB)
3. Dynamic symbol detection and validation
4. Seamless integration with existing pipeline (no breaking changes)
5. Maintain CLAUDE.md constraints (650-line limit, fail-fast, modular)

---

## Current State Analysis

### Existing Components
```
src/config.py
├── SYMBOLS = ['MES', 'MGC']  # Hardcoded
├── DATA_DIR, RAW_DATA_DIR    # Hardcoded paths
└── validate_config()          # Global validation

src/stages/stage1_ingest.py
├── DataIngestor class         # File loading (CSV/Parquet)
├── Path validation            # Security checks
├── OHLCV standardization      # Column mapping
└── Timezone handling          # UTC conversion

src/pipeline_cli.py
├── run(symbols, ...)          # CLI entry point
├── status(), rerun()          # Run management
└── Typer-based commands       # Terminal interface
```

### Pain Points
1. **Hardcoded symbols** in `src/config.py` require code changes
2. **Manual file placement** in `data/raw/` before pipeline execution
3. **No preview/validation** before starting expensive pipeline runs
4. **No UI** for non-technical users
5. **Tight coupling** between config and data sources

---

## Architecture Design

### 1. Data Source Abstraction Layer

**Design Pattern:** Strategy Pattern + Repository Pattern
- **Strategy Pattern** for pluggable data sources
- **Repository Pattern** for unified data access interface
- **Factory Pattern** for data source instantiation

```
┌─────────────────────────────────────────────────────────────┐
│                   DataSourceRepository                       │
│  Unified interface for accessing market data                │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼─────────┐ ┌──────▼────────┐ ┌────────▼────────┐
│ FileDataSource  │ │ UploadDataSrc │ │ APIDataSource   │
│ (existing files)│ │ (UI uploads)  │ │ (future: APIs)  │
└─────────────────┘ └───────────────┘ └─────────────────┘
```

#### Interface Definition

```python
# src/data_sources/base.py (< 200 lines)

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DataSourceMetadata:
    """Metadata for a data source."""
    source_id: str              # Unique identifier
    source_type: str            # 'file', 'upload', 'api', 'database'
    symbols: List[str]          # Detected symbols
    date_range: Tuple[str, str] # (start_date, end_date)
    row_count: int              # Total rows across all symbols
    file_size_mb: float         # Size in MB
    created_at: datetime        # When source was added
    columns: List[str]          # Available columns
    validated: bool             # Passed validation checks
    validation_errors: List[str] # Any validation issues


class DataSource(ABC):
    """Abstract base class for data sources."""

    @abstractmethod
    def get_metadata(self) -> DataSourceMetadata:
        """Get metadata about this data source."""
        pass

    @abstractmethod
    def load_symbol(self, symbol: str) -> pd.DataFrame:
        """Load data for a specific symbol."""
        pass

    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols in this source."""
        pass

    @abstractmethod
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate data source. Returns (is_valid, errors)."""
        pass

    @abstractmethod
    def preview(self, symbol: str, n_rows: int = 10) -> pd.DataFrame:
        """Get preview of data for a symbol."""
        pass


class DataSourceRepository:
    """
    Repository for managing multiple data sources.
    Provides unified interface for data access.
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self._sources: Dict[str, DataSource] = {}
        self._active_source_id: Optional[str] = None

    def register_source(self, source_id: str, source: DataSource) -> None:
        """Register a new data source."""
        if source_id in self._sources:
            raise ValueError(f"Source '{source_id}' already registered")

        # Validate source before registration
        is_valid, errors = source.validate()
        if not is_valid:
            raise ValueError(f"Source validation failed: {errors}")

        self._sources[source_id] = source

    def set_active_source(self, source_id: str) -> None:
        """Set the active data source for pipeline execution."""
        if source_id not in self._sources:
            raise KeyError(f"Source '{source_id}' not found")
        self._active_source_id = source_id

    def get_active_source(self) -> DataSource:
        """Get the currently active data source."""
        if self._active_source_id is None:
            raise RuntimeError("No active data source set")
        return self._sources[self._active_source_id]

    def list_sources(self) -> Dict[str, DataSourceMetadata]:
        """List all registered sources with metadata."""
        return {
            source_id: source.get_metadata()
            for source_id, source in self._sources.items()
        }

    def remove_source(self, source_id: str) -> None:
        """Remove a data source."""
        if source_id == self._active_source_id:
            self._active_source_id = None
        del self._sources[source_id]
```

---

### 2. Concrete Data Source Implementations

#### A. File Data Source (Existing Files)

```python
# src/data_sources/file_source.py (< 250 lines)

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime

from .base import DataSource, DataSourceMetadata
from ..stages.stage1_ingest import DataIngestor


class FileDataSource(DataSource):
    """
    Data source for files already present in data/raw directory.
    Backward compatible with existing pipeline workflow.
    """

    def __init__(
        self,
        source_id: str,
        raw_data_dir: Path,
        symbol_pattern: str = "*.parquet"
    ):
        """
        Initialize file data source.

        Parameters:
        -----------
        source_id : Unique identifier for this source
        raw_data_dir : Directory containing raw data files
        symbol_pattern : Glob pattern to match symbol files
        """
        self.source_id = source_id
        self.raw_data_dir = Path(raw_data_dir)
        self.symbol_pattern = symbol_pattern

        if not self.raw_data_dir.exists():
            raise FileNotFoundError(f"Directory not found: {raw_data_dir}")

        # Discover symbols from files
        self._symbol_files = self._discover_symbols()

        if not self._symbol_files:
            raise ValueError(f"No files found matching pattern: {symbol_pattern}")

    def _discover_symbols(self) -> Dict[str, Path]:
        """Discover symbols from files in directory."""
        symbol_files = {}

        for file_path in self.raw_data_dir.glob(self.symbol_pattern):
            # Extract symbol from filename (e.g., "MES_1m.parquet" -> "MES")
            symbol = file_path.stem.split('_')[0].upper()
            symbol_files[symbol] = file_path

        return symbol_files

    def get_metadata(self) -> DataSourceMetadata:
        """Get metadata about this data source."""
        total_size = sum(
            f.stat().st_size for f in self._symbol_files.values()
        )

        # Load first symbol to get date range and columns
        first_symbol = list(self._symbol_files.keys())[0]
        sample_df = pd.read_parquet(self._symbol_files[first_symbol])

        return DataSourceMetadata(
            source_id=self.source_id,
            source_type='file',
            symbols=list(self._symbol_files.keys()),
            date_range=(
                sample_df['datetime'].min().isoformat(),
                sample_df['datetime'].max().isoformat()
            ),
            row_count=len(sample_df),  # Approximate
            file_size_mb=total_size / (1024 * 1024),
            created_at=datetime.now(),
            columns=sample_df.columns.tolist(),
            validated=True,
            validation_errors=[]
        )

    def load_symbol(self, symbol: str) -> pd.DataFrame:
        """Load data for a specific symbol."""
        if symbol not in self._symbol_files:
            raise KeyError(f"Symbol '{symbol}' not found in this source")

        file_path = self._symbol_files[symbol]

        # Use DataIngestor for standardization
        ingestor = DataIngestor(
            raw_data_dir=self.raw_data_dir,
            output_dir=self.raw_data_dir,
            source_timezone='UTC'
        )

        df, _ = ingestor.ingest_file(file_path, symbol=symbol)
        return df

    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols."""
        return list(self._symbol_files.keys())

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate data source."""
        errors = []

        # Check all files are readable
        for symbol, file_path in self._symbol_files.items():
            try:
                df = pd.read_parquet(file_path)

                # Check for required columns
                required_cols = {'datetime', 'open', 'high', 'low', 'close'}
                missing_cols = required_cols - set(df.columns)
                if missing_cols:
                    errors.append(
                        f"Symbol {symbol}: missing columns {missing_cols}"
                    )

                # Check for empty data
                if len(df) == 0:
                    errors.append(f"Symbol {symbol}: empty dataset")

            except Exception as e:
                errors.append(f"Symbol {symbol}: failed to load - {e}")

        return (len(errors) == 0, errors)

    def preview(self, symbol: str, n_rows: int = 10) -> pd.DataFrame:
        """Get preview of data for a symbol."""
        if symbol not in self._symbol_files:
            raise KeyError(f"Symbol '{symbol}' not found")

        df = pd.read_parquet(self._symbol_files[symbol])
        return df.head(n_rows)
```

#### B. Upload Data Source (Streamlit UI)

```python
# src/data_sources/upload_source.py (< 300 lines)

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime
import tempfile
import shutil

from .base import DataSource, DataSourceMetadata
from ..stages.stage1_ingest import DataIngestor


class UploadDataSource(DataSource):
    """
    Data source for files uploaded via Streamlit UI.
    Stores uploaded files in temporary session directory.
    """

    def __init__(
        self,
        source_id: str,
        session_dir: Path,
        uploaded_files: Dict[str, bytes],
        file_format: str = 'parquet'
    ):
        """
        Initialize upload data source.

        Parameters:
        -----------
        source_id : Unique identifier
        session_dir : Temporary directory for this session
        uploaded_files : Dict mapping filename -> file bytes
        file_format : File format ('csv' or 'parquet')
        """
        self.source_id = source_id
        self.session_dir = Path(session_dir)
        self.file_format = file_format

        # Create session directory
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Save uploaded files to session directory
        self._symbol_files = self._save_uploaded_files(uploaded_files)

        if not self._symbol_files:
            raise ValueError("No valid symbol files provided")

    def _save_uploaded_files(
        self,
        uploaded_files: Dict[str, bytes]
    ) -> Dict[str, Path]:
        """Save uploaded files to session directory."""
        symbol_files = {}

        for filename, file_bytes in uploaded_files.items():
            # Extract symbol from filename
            symbol = Path(filename).stem.split('_')[0].upper()

            # Save to session directory
            file_path = self.session_dir / filename
            file_path.write_bytes(file_bytes)

            symbol_files[symbol] = file_path

        return symbol_files

    def get_metadata(self) -> DataSourceMetadata:
        """Get metadata about this data source."""
        total_size = sum(
            f.stat().st_size for f in self._symbol_files.values()
        )

        # Load first symbol for metadata
        first_symbol = list(self._symbol_files.keys())[0]
        sample_df = self._load_file(self._symbol_files[first_symbol])

        # Determine date range across all symbols
        all_dates = []
        for file_path in self._symbol_files.values():
            df = self._load_file(file_path)
            all_dates.append(df['datetime'].min())
            all_dates.append(df['datetime'].max())

        return DataSourceMetadata(
            source_id=self.source_id,
            source_type='upload',
            symbols=list(self._symbol_files.keys()),
            date_range=(
                min(all_dates).isoformat(),
                max(all_dates).isoformat()
            ),
            row_count=sum(
                len(self._load_file(f)) for f in self._symbol_files.values()
            ),
            file_size_mb=total_size / (1024 * 1024),
            created_at=datetime.now(),
            columns=sample_df.columns.tolist(),
            validated=False,  # Will be validated separately
            validation_errors=[]
        )

    def _load_file(self, file_path: Path) -> pd.DataFrame:
        """Load a file based on format."""
        if self.file_format == 'csv':
            df = pd.read_csv(file_path)
        elif self.file_format == 'parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported format: {self.file_format}")

        # Basic column standardization
        df.columns = df.columns.str.lower().str.strip()
        return df

    def load_symbol(self, symbol: str) -> pd.DataFrame:
        """Load data for a specific symbol."""
        if symbol not in self._symbol_files:
            raise KeyError(f"Symbol '{symbol}' not found")

        file_path = self._symbol_files[symbol]

        # Use DataIngestor for full standardization
        ingestor = DataIngestor(
            raw_data_dir=self.session_dir,
            output_dir=self.session_dir,
            source_timezone='UTC'
        )

        df, _ = ingestor.ingest_file(file_path, symbol=symbol)
        return df

    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols."""
        return list(self._symbol_files.keys())

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate uploaded data."""
        errors = []

        for symbol, file_path in self._symbol_files.items():
            try:
                df = self._load_file(file_path)

                # Check required columns
                required_cols = {'datetime', 'open', 'high', 'low', 'close'}
                df_cols = set(df.columns)
                missing_cols = required_cols - df_cols

                if missing_cols:
                    errors.append(
                        f"{symbol}: missing columns {missing_cols}"
                    )

                # Check for empty data
                if len(df) == 0:
                    errors.append(f"{symbol}: empty dataset")

                # Check datetime is parseable
                try:
                    pd.to_datetime(df['datetime'])
                except Exception as e:
                    errors.append(
                        f"{symbol}: invalid datetime format - {e}"
                    )

                # Check OHLC are numeric
                for col in ['open', 'high', 'low', 'close']:
                    if col in df.columns:
                        if not pd.api.types.is_numeric_dtype(df[col]):
                            errors.append(
                                f"{symbol}: {col} is not numeric"
                            )

            except Exception as e:
                errors.append(f"{symbol}: failed to load - {e}")

        return (len(errors) == 0, errors)

    def preview(self, symbol: str, n_rows: int = 10) -> pd.DataFrame:
        """Get preview of data."""
        if symbol not in self._symbol_files:
            raise KeyError(f"Symbol '{symbol}' not found")

        df = self._load_file(self._symbol_files[symbol])
        return df.head(n_rows)

    def cleanup(self) -> None:
        """Clean up temporary session directory."""
        if self.session_dir.exists():
            shutil.rmtree(self.session_dir)
```

---

### 3. Configuration Integration

Update `src/config.py` to support dynamic symbol configuration:

```python
# src/config.py (additions, < 50 lines)

from typing import Optional, List
from pathlib import Path

# Keep existing SYMBOLS as default
SYMBOLS = ['MES', 'MGC']  # Default symbols

# New: Runtime configuration
_runtime_symbols: Optional[List[str]] = None
_runtime_data_source_id: Optional[str] = None


def set_runtime_symbols(symbols: List[str]) -> None:
    """
    Set symbols at runtime (from UI or CLI).
    Overrides default SYMBOLS constant.
    """
    global _runtime_symbols
    _runtime_symbols = symbols


def get_symbols() -> List[str]:
    """
    Get active symbols (runtime or default).
    """
    if _runtime_symbols is not None:
        return _runtime_symbols
    return SYMBOLS


def set_runtime_data_source(source_id: str) -> None:
    """Set active data source ID."""
    global _runtime_data_source_id
    _runtime_data_source_id = source_id


def get_runtime_data_source() -> Optional[str]:
    """Get active data source ID."""
    return _runtime_data_source_id
```

---

### 4. Streamlit UI Structure

```
src/ui/
├── __init__.py                 # Package init
├── app.py                      # Main Streamlit app (<650 lines)
├── components/                 # Reusable UI components
│   ├── __init__.py
│   ├── data_upload.py         # File upload component
│   ├── data_preview.py        # Data preview component
│   ├── symbol_selector.py     # Symbol selection component
│   └── validation_display.py  # Validation results component
├── pages/                      # Multi-page app
│   ├── 1_upload_data.py       # Data upload page
│   ├── 2_configure_pipeline.py # Pipeline configuration
│   └── 3_monitor_run.py       # Run monitoring
└── session.py                  # Session state management
```

#### Main Streamlit App

```python
# src/ui/app.py (< 400 lines)

import streamlit as st
from pathlib import Path
import sys

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from data_sources.base import DataSourceRepository
from data_sources.file_source import FileDataSource
from data_sources.upload_source import UploadDataSource
from ui.session import SessionManager
import config


st.set_page_config(
    page_title="Trading Pipeline Manager",
    page_icon="📊",
    layout="wide"
)


def init_session():
    """Initialize session state."""
    if 'session_manager' not in st.session_state:
        st.session_state.session_manager = SessionManager(project_root)

    if 'repository' not in st.session_state:
        st.session_state.repository = DataSourceRepository(project_root)


def main():
    """Main Streamlit app."""
    init_session()

    st.title("📊 Trading Pipeline Manager")
    st.markdown("---")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        [
            "📤 Upload Data",
            "⚙️ Configure Pipeline",
            "🔍 Monitor Run",
            "📁 Manage Data Sources"
        ]
    )

    # Route to pages
    if page == "📤 Upload Data":
        from ui.pages.upload_data import render_upload_page
        render_upload_page()

    elif page == "⚙️ Configure Pipeline":
        from ui.pages.configure_pipeline import render_config_page
        render_config_page()

    elif page == "🔍 Monitor Run":
        from ui.pages.monitor_run import render_monitor_page
        render_monitor_page()

    elif page == "📁 Manage Data Sources":
        from ui.pages.manage_sources import render_sources_page
        render_sources_page()


if __name__ == "__main__":
    main()
```

#### Upload Data Page

```python
# src/ui/pages/upload_data.py (< 350 lines)

import streamlit as st
from pathlib import Path
import pandas as pd
from datetime import datetime
import tempfile

from data_sources.upload_source import UploadDataSource
from ui.components.data_preview import show_data_preview
from ui.components.validation_display import show_validation_results


def render_upload_page():
    """Render data upload page."""
    st.header("📤 Upload Data")

    st.markdown("""
    Upload your market data files (CSV or Parquet format).
    Files should contain OHLCV data with columns:
    `datetime`, `open`, `high`, `low`, `close`, `volume`.
    """)

    # File upload
    uploaded_files = st.file_uploader(
        "Choose data files",
        type=['csv', 'parquet'],
        accept_multiple_files=True,
        help="Upload one file per symbol (e.g., MES_1m.parquet, MGC_1m.parquet)"
    )

    if not uploaded_files:
        st.info("👆 Upload data files to get started")
        return

    # Detect file format
    file_format = Path(uploaded_files[0].name).suffix.lower().replace('.', '')

    st.success(f"✅ Uploaded {len(uploaded_files)} file(s)")

    # Preview uploaded files
    st.subheader("Uploaded Files")
    for file in uploaded_files:
        st.text(f"📄 {file.name} ({file.size / 1024:.1f} KB)")

    # Process uploads
    if st.button("Process Uploads", type="primary"):
        with st.spinner("Processing uploaded files..."):
            # Create temporary session directory
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = Path(tempfile.gettempdir()) / "pipeline_uploads" / session_id

            # Convert uploaded files to dict
            uploaded_dict = {
                file.name: file.read()
                for file in uploaded_files
            }

            try:
                # Create upload data source
                source = UploadDataSource(
                    source_id=f"upload_{session_id}",
                    session_dir=session_dir,
                    uploaded_files=uploaded_dict,
                    file_format=file_format
                )

                # Validate
                is_valid, errors = source.validate()

                if not is_valid:
                    show_validation_results(is_valid, errors)
                    return

                # Register source
                repo = st.session_state.repository
                repo.register_source(source.source_id, source)
                repo.set_active_source(source.source_id)

                # Store in session
                st.session_state.active_source = source
                st.session_state.active_source_id = source.source_id

                st.success("✅ Data uploaded and validated successfully!")

                # Show metadata
                metadata = source.get_metadata()
                st.subheader("Data Summary")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Symbols", len(metadata.symbols))
                with col2:
                    st.metric("Total Rows", f"{metadata.row_count:,}")
                with col3:
                    st.metric("Size", f"{metadata.file_size_mb:.2f} MB")

                st.markdown(f"**Date Range:** {metadata.date_range[0]} to {metadata.date_range[1]}")
                st.markdown(f"**Symbols:** {', '.join(metadata.symbols)}")

                # Preview data
                st.subheader("Data Preview")
                for symbol in metadata.symbols:
                    with st.expander(f"Preview: {symbol}"):
                        preview_df = source.preview(symbol, n_rows=10)
                        show_data_preview(preview_df, symbol)

                # Next steps
                st.info("👉 Go to 'Configure Pipeline' to set up and run the pipeline")

            except Exception as e:
                st.error(f"❌ Error processing uploads: {e}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
```

---

### 5. Pipeline Integration

Update pipeline runner to use data source repository:

```python
# src/pipeline/data_loader.py (NEW, < 200 lines)

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
            default_source = FileDataSource(
                source_id='default_files',
                raw_data_dir=config.RAW_DATA_DIR,
                symbol_pattern='*.parquet'
            )
            self.repository.register_source('default_files', default_source)
            self.repository.set_active_source('default_files')
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
        """
        source = self.repository.get_active_source()
        return source.load_symbol(symbol)

    def get_available_symbols(self) -> List[str]:
        """Get available symbols from active source."""
        source = self.repository.get_active_source()
        return source.get_available_symbols()

    def validate_source(self) -> bool:
        """Validate active data source."""
        source = self.repository.get_active_source()
        is_valid, errors = source.validate()

        if not is_valid:
            raise ValueError(f"Data source validation failed: {errors}")

        return True
```

---

## File Organization

```
/home/jake/Desktop/Research/
├── src/
│   ├── data_sources/           # NEW: Data source abstraction
│   │   ├── __init__.py
│   │   ├── base.py            # Interfaces (<200 lines)
│   │   ├── file_source.py     # File-based source (<250 lines)
│   │   ├── upload_source.py   # Upload source (<300 lines)
│   │   └── api_source.py      # Future: API source
│   │
│   ├── ui/                     # NEW: Streamlit UI
│   │   ├── __init__.py
│   │   ├── app.py             # Main app (<400 lines)
│   │   ├── session.py         # Session management (<150 lines)
│   │   ├── components/        # Reusable components
│   │   │   ├── __init__.py
│   │   │   ├── data_upload.py      (<200 lines)
│   │   │   ├── data_preview.py     (<150 lines)
│   │   │   ├── symbol_selector.py  (<100 lines)
│   │   │   └── validation_display.py (<100 lines)
│   │   └── pages/             # App pages
│   │       ├── upload_data.py       (<350 lines)
│   │       ├── configure_pipeline.py (<400 lines)
│   │       ├── monitor_run.py       (<300 lines)
│   │       └── manage_sources.py    (<250 lines)
│   │
│   ├── pipeline/
│   │   └── data_loader.py     # NEW: Pipeline integration (<200 lines)
│   │
│   ├── config.py              # MODIFIED: Add runtime config
│   ├── stages/
│   │   └── stage1_ingest.py   # NO CHANGE: Reused by data sources
│   └── pipeline_cli.py        # MODIFIED: Use data loader
│
├── data/
│   └── uploads/               # NEW: Uploaded data storage
│       └── sessions/          # Session-specific uploads
│
└── docs/
    └── DATA_INGESTION_ARCHITECTURE.md  # This document
```

**Total New Lines:** ~3,000 lines across 15 files
**Largest File:** `configure_pipeline.py` (400 lines) - well under 650 limit

---

## Usage Patterns

### Pattern 1: CLI with Existing Files (Backward Compatible)

```bash
# Existing workflow - unchanged
./pipeline run --symbols MES,MGC
```

Pipeline automatically uses `FileDataSource` for files in `data/raw/`.

### Pattern 2: UI Upload Workflow

```bash
# Start Streamlit UI
streamlit run src/ui/app.py
```

1. **Upload Data Page:**
   - Upload MES_1m.parquet, MGC_1m.parquet
   - System validates and previews data
   - Registers `UploadDataSource` with repository

2. **Configure Pipeline Page:**
   - Select symbols from uploaded data
   - Configure horizons, split ratios
   - Preview configuration

3. **Monitor Run Page:**
   - Trigger pipeline execution
   - Real-time progress monitoring
   - View results and artifacts

### Pattern 3: CLI with Uploaded Data

```bash
# After UI upload, run via CLI using session ID
./pipeline run --data-source upload_20241221_120000
```

### Pattern 4: Programmatic API

```python
from data_sources.base import DataSourceRepository
from data_sources.file_source import FileDataSource
from pipeline.data_loader import PipelineDataLoader

# Create repository
repo = DataSourceRepository(project_root)

# Register file source
source = FileDataSource(
    source_id='my_data',
    raw_data_dir='/path/to/data'
)
repo.register_source('my_data', source)
repo.set_active_source('my_data')

# Use in pipeline
loader = PipelineDataLoader(repository=repo)
df = loader.load_symbol_data('MES')
```

---

## Validation Strategy

### Level 1: File-Level Validation (Upload Time)
- File format (CSV/Parquet)
- Required columns present
- Datetime parseable
- OHLC numeric types
- Non-empty dataset

### Level 2: Data-Level Validation (Pre-Pipeline)
- OHLCV relationships (high >= low, etc.)
- Timezone consistency
- Duplicate timestamps
- Missing data gaps
- Date range coverage

### Level 3: Pipeline Validation (Runtime)
- Existing `DataIngestor` validation
- Feature engineering checks
- Label distribution validation

---

## State Management

### Session State (Streamlit)

```python
# src/ui/session.py (< 150 lines)

from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime
import tempfile


class SessionManager:
    """Manage Streamlit session state and temporary data."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.sessions_dir = project_root / "data" / "uploads" / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def create_session(self) -> str:
        """Create new session and return session ID."""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Save session metadata
        metadata = {
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'status': 'active'
        }

        metadata_path = session_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return session_id

    def get_session_dir(self, session_id: str) -> Path:
        """Get directory for a session."""
        return self.sessions_dir / session_id

    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Clean up sessions older than max_age_hours."""
        import time

        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        cleaned = 0

        for session_dir in self.sessions_dir.iterdir():
            if not session_dir.is_dir():
                continue

            age_seconds = current_time - session_dir.stat().st_mtime

            if age_seconds > max_age_seconds:
                import shutil
                shutil.rmtree(session_dir)
                cleaned += 1

        return cleaned
```

---

## Security Considerations

1. **Path Traversal Prevention**
   - Existing `DataIngestor._validate_path()` prevents traversal
   - Upload source restricts writes to session directory
   - All paths resolved and validated before use

2. **File Size Limits**
   - Streamlit upload limit: 200MB default
   - Can be configured via `server.maxUploadSize`
   - Validation before processing

3. **Input Sanitization**
   - Symbol names validated (alphanumeric only)
   - File extensions checked
   - Column names standardized/sanitized

4. **Session Isolation**
   - Each upload session gets unique directory
   - Session cleanup after expiration
   - No cross-session data access

---

## Testing Strategy

### Unit Tests

```python
# tests/test_data_sources.py

def test_file_source_discovery():
    """Test FileDataSource discovers symbols correctly."""
    ...

def test_upload_source_validation():
    """Test UploadDataSource validates data."""
    ...

def test_repository_registration():
    """Test DataSourceRepository manages sources."""
    ...
```

### Integration Tests

```python
# tests/test_pipeline_integration.py

def test_pipeline_with_upload_source():
    """Test pipeline execution with uploaded data."""
    ...

def test_pipeline_with_file_source():
    """Test backward compatibility with file source."""
    ...
```

### UI Tests

```python
# tests/test_ui.py (using Streamlit testing framework)

def test_upload_workflow():
    """Test complete upload workflow."""
    ...

def test_validation_display():
    """Test validation results display."""
    ...
```

---

## Migration Path

### Phase 1: Core Infrastructure (Week 1)
- [ ] Implement `data_sources/base.py` (interfaces)
- [ ] Implement `FileDataSource` (backward compatible)
- [ ] Implement `DataSourceRepository`
- [ ] Unit tests for data sources

### Phase 2: Upload Support (Week 2)
- [ ] Implement `UploadDataSource`
- [ ] Implement `SessionManager`
- [ ] Integration tests

### Phase 3: UI Development (Week 3)
- [ ] Streamlit app structure
- [ ] Upload data page
- [ ] Data preview components
- [ ] Validation display

### Phase 4: Pipeline Integration (Week 4)
- [ ] `PipelineDataLoader` implementation
- [ ] Update `pipeline_cli.py` to support data sources
- [ ] Update config for runtime symbols
- [ ] End-to-end tests

### Phase 5: UI Completion (Week 5)
- [ ] Configure pipeline page
- [ ] Monitor run page
- [ ] Manage sources page
- [ ] UI polish and testing

---

## Success Criteria

1. **Backward Compatibility**
   - Existing CLI workflow unchanged
   - No breaking changes to pipeline stages
   - Existing tests pass

2. **Modularity**
   - All files under 650 lines
   - Clear separation of concerns
   - Reusable components

3. **Fail-Fast**
   - Validation at every boundary
   - Clear error messages
   - Early detection of invalid data

4. **User Experience**
   - Upload data without code changes
   - Preview data before pipeline execution
   - Clear validation feedback
   - Session management for temporary data

5. **Extensibility**
   - Easy to add new data sources (API, database)
   - Plugin architecture for custom sources
   - Well-documented interfaces

---

## Future Enhancements

### API Data Source
```python
class APIDataSource(DataSource):
    """Fetch data from external APIs (Alpha Vantage, Polygon, etc.)."""
    ...
```

### Database Data Source
```python
class DatabaseDataSource(DataSource):
    """Load data from SQL/NoSQL databases."""
    ...
```

### Data Caching
```python
class CachedDataSource(DataSource):
    """Wrapper that caches loaded data."""
    ...
```

### Data Transformation Pipeline
```python
class TransformDataSource(DataSource):
    """Apply transformations to underlying source."""
    ...
```

---

## Appendix: Key Design Decisions

### Why Strategy Pattern?
- **Pluggable implementations:** Easy to add new data sources
- **Runtime selection:** Choose source dynamically
- **Single responsibility:** Each source handles one type

### Why Repository Pattern?
- **Centralized management:** Single source of truth
- **Abstraction:** Pipeline doesn't know about source details
- **Testability:** Easy to mock in tests

### Why Session-Based Storage?
- **Isolation:** Each upload session is independent
- **Cleanup:** Automatic expiration of old sessions
- **Security:** No persistent storage of uploaded data

### Why Not Modify DataIngestor?
- **Backward compatibility:** Existing code continues to work
- **Single responsibility:** DataIngestor focuses on file processing
- **Reusability:** Data sources reuse DataIngestor logic

---

## Questions & Answers

**Q: Will this break existing workflows?**
A: No. Existing CLI commands work unchanged. `FileDataSource` provides same interface as current hardcoded approach.

**Q: Where is uploaded data stored?**
A: Temporarily in `data/uploads/sessions/{session_id}/`. Cleaned up after 24 hours by default.

**Q: Can I use both CLI and UI?**
A: Yes. Upload via UI, then run via CLI using `--data-source upload_{session_id}`.

**Q: How do I add a new data source type?**
A: Implement `DataSource` interface. See `FileDataSource` and `UploadDataSource` as examples.

**Q: What about data source credentials (for APIs)?**
A: Future `APIDataSource` will handle credentials via environment variables or secrets management.

---

## Conclusion

This architecture provides a flexible, modular, and extensible data ingestion system that:

1. **Maintains backward compatibility** with existing CLI workflow
2. **Adds UI support** for non-technical users
3. **Follows CLAUDE.md principles** (modularity, fail-fast, <650 lines)
4. **Enables future extensions** (APIs, databases, caching)
5. **Provides excellent UX** (preview, validation, session management)

The Strategy + Repository pattern provides clean separation between data source types and pipeline logic, making the system easy to test, maintain, and extend.
