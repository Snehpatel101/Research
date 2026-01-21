import json
from pathlib import Path

import pandas as pd
import pytest

from src.core.lineage import (
    DatasetChecksum,
    PipelineLineage,
    compute_file_checksum,
    create_dataset_checksum,
    validate_dataset_checksum,
)


@pytest.fixture
def sample_dataset(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "feature1": [1.0, 2.0, 3.0],
            "feature2": [4.0, 5.0, 6.0],
            "label": [0, 1, 0],
        }
    )
    path = tmp_path / "test_data.parquet"
    df.to_parquet(path, index=False)
    return path


def test_compute_file_checksum(sample_dataset: Path):
    checksum = compute_file_checksum(sample_dataset)
    assert isinstance(checksum, str)
    assert len(checksum) == 16

    checksum2 = compute_file_checksum(sample_dataset)
    assert checksum == checksum2


def test_create_dataset_checksum(sample_dataset: Path):
    cs = create_dataset_checksum(sample_dataset, "test")
    assert cs.n_rows == 3
    assert cs.n_cols == 3
    assert set(cs.columns) == {"feature1", "feature2", "label"}
    assert len(cs.checksum) == 16


def test_validate_dataset_checksum(sample_dataset: Path):
    expected = create_dataset_checksum(sample_dataset, "test")

    is_valid, issues = validate_dataset_checksum(sample_dataset, expected, strict=True)
    assert is_valid
    assert len(issues) == 0


def test_validate_dataset_checksum_row_mismatch(sample_dataset: Path, tmp_path: Path):
    expected = create_dataset_checksum(sample_dataset, "test")

    df = pd.DataFrame(
        {
            "feature1": [1.0, 2.0],
            "feature2": [4.0, 5.0],
            "label": [0, 1],
        }
    )
    modified_path = tmp_path / "modified.parquet"
    df.to_parquet(modified_path, index=False)

    is_valid, issues = validate_dataset_checksum(modified_path, expected, strict=False)
    assert not is_valid
    assert any("Row count mismatch" in issue for issue in issues)


def test_pipeline_lineage_save_load(tmp_path: Path, sample_dataset: Path):
    cs = create_dataset_checksum(sample_dataset, "train")
    lineage = PipelineLineage(
        pipeline_run_id="test_run_001",
        target_timeframe="5min",
        output_timeframes=["5min"],
        symbols=["MES"],
        feature_generation="full",
        label_horizons=[5, 10, 20],
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        purge_bars=60,
        embargo_bars=1440,
        random_seed=42,
        dataset_checksums={"train": cs},
        created_at="2026-01-15T22:00:00",
    )

    lineage_path = tmp_path / "lineage.json"
    lineage.save(lineage_path)

    assert lineage_path.exists()

    loaded = PipelineLineage.load(lineage_path)
    assert loaded.pipeline_run_id == "test_run_001"
    assert loaded.target_timeframe == "5min"
    assert loaded.symbols == ["MES"]
    assert len(loaded.dataset_checksums) == 1
    assert "train" in loaded.dataset_checksums


def test_pipeline_lineage_to_dict(sample_dataset: Path):
    cs = create_dataset_checksum(sample_dataset, "train")
    lineage = PipelineLineage(
        pipeline_run_id="test_run_001",
        target_timeframe="5min",
        output_timeframes=["5min"],
        symbols=["MES"],
        feature_generation="full",
        label_horizons=[5, 10, 20],
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        purge_bars=60,
        embargo_bars=1440,
        random_seed=42,
        dataset_checksums={"train": cs},
        created_at="2026-01-15T22:00:00",
    )

    data = lineage.to_dict()
    assert data["pipeline_run_id"] == "test_run_001"
    assert "dataset_checksums" in data
    assert "train" in data["dataset_checksums"]
    assert data["dataset_checksums"]["train"]["n_rows"] == 3
