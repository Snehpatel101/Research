"""Default configuration factory for DataConfig."""

from typing import Any


def create_default_config(
    symbols: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    run_id: str | None = None,
    **kwargs: Any,
):
    """
    Create a default DataConfig with optional overrides.

    Args:
        symbols: List of symbols to trade
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        run_id: Run identifier (auto-generated if None)
        **kwargs: Additional parameters to override defaults

    Returns:
        DataConfig instance
    """
    from src.pipeline.data_config import DataConfig

    config_kwargs = {}

    if symbols is not None:
        config_kwargs["symbols"] = symbols

    if start_date is not None:
        config_kwargs["start_date"] = start_date

    if end_date is not None:
        config_kwargs["end_date"] = end_date

    if run_id is not None:
        config_kwargs["run_id"] = run_id

    config_kwargs.update(kwargs)

    return DataConfig(**config_kwargs)
