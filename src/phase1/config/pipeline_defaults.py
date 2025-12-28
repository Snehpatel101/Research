"""Default configuration factory for PipelineConfig."""
from typing import List, Optional, Any


def create_default_config(
    symbols: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    run_id: Optional[str] = None,
    **kwargs: Any
):
    """
    Create a default configuration with optional overrides.

    Args:
        symbols: List of symbols to trade
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        run_id: Run identifier (auto-generated if None)
        **kwargs: Additional parameters to override defaults

    Returns:
        PipelineConfig instance
    """
    # Import here to avoid circular imports
    from src.phase1.pipeline_config import PipelineConfig

    config_kwargs = {}

    if symbols is not None:
        config_kwargs['symbols'] = symbols

    if start_date is not None:
        config_kwargs['start_date'] = start_date

    if end_date is not None:
        config_kwargs['end_date'] = end_date

    if run_id is not None:
        config_kwargs['run_id'] = run_id

    # Merge with additional kwargs
    config_kwargs.update(kwargs)

    return PipelineConfig(**config_kwargs)
