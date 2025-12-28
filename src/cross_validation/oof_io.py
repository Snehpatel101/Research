"""
OOF prediction I/O operations.

Save and load stacking datasets and metadata.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from src.cross_validation.oof_stacking import StackingDataset

logger = logging.getLogger(__name__)


# =============================================================================
# I/O OPERATIONS
# =============================================================================

class OOFDatasetIO:
    """Handle saving and loading of stacking datasets."""

    @staticmethod
    def save_stacking_dataset(
        stacking_ds: StackingDataset,
        output_dir: Path,
    ) -> Path:
        """
        Save stacking dataset to parquet.

        Args:
            stacking_ds: StackingDataset to save
            output_dir: Output directory

        Returns:
            Path to saved parquet file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save dataset
        parquet_path = output_dir / f"stacking_dataset_h{stacking_ds.horizon}.parquet"
        stacking_ds.data.to_parquet(parquet_path, index=False)

        # Save metadata
        metadata_path = output_dir / f"stacking_metadata_h{stacking_ds.horizon}.json"
        with open(metadata_path, "w") as f:
            json.dump(stacking_ds.metadata, f, indent=2)

        logger.info(f"Saved stacking dataset to {parquet_path}")
        return parquet_path


__all__ = [
    "OOFDatasetIO",
]
