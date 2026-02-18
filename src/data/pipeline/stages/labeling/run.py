"""
Stage 4: Initial Triple-Barrier Labeling - Pipeline Wrapper.

This module provides the pipeline integration for initial labeling,
extracting the orchestration logic from pipeline/stages/labeling.py.

Required Input Columns:
    - close, high, low, open: OHLC price data
    - atr_14: 14-period Average True Range (computed in feature engineering stage)

The ATR column is required for dynamic barrier calculation. Barriers are set as
multiples of ATR to adapt to current volatility conditions. Without ATR, the
triple-barrier labeling cannot compute appropriate profit targets and stop losses.
"""

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

# Import barrier config for symbol/horizon-specific defaults
from src.data.pipeline.config.barriers_config import get_barrier_params
from src.data.pipeline.utils import StageResult, create_failed_result, create_stage_result

from .triple_barrier import triple_barrier_numba

if TYPE_CHECKING:
    from src.core.common.manifest import ArtifactManifest
    from src.data.pipeline.data_config import DataConfig as PipelineConfig

logger = logging.getLogger(__name__)

# =============================================================================
# REQUIRED COLUMNS FOR LABELING
# =============================================================================
# ATR is required for dynamic barrier calculation. Barriers are set as multiples
# of ATR (e.g., k_up * atr_14 for take-profit, k_down * atr_14 for stop-loss).
# This dependency must be satisfied by the feature engineering stage (Stage 3).
REQUIRED_LABELING_COLUMNS = ["atr_14"]

# OHLC columns required for triple-barrier calculation
REQUIRED_OHLC_COLUMNS = ["open", "high", "low", "close"]


def validate_labeling_prerequisites(
    df: pd.DataFrame,
    required_cols: list[str] | None = None,
    symbol: str = "",
    timeframe: str = "",
) -> None:
    """
    Validate that all required columns exist before labeling.

    This function performs early validation of the input DataFrame to ensure
    all required columns for triple-barrier labeling are present. Failing fast
    with a clear error message is preferred over cryptic KeyError exceptions
    during the labeling computation.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with features and OHLC data
    required_cols : list[str], optional
        List of required column names. If None, uses REQUIRED_LABELING_COLUMNS + REQUIRED_OHLC_COLUMNS.
    symbol : str, optional
        Symbol name for error message context
    timeframe : str, optional
        Timeframe for error message context

    Raises
    ------
    ValueError
        If any required column is missing from the DataFrame

    Examples
    --------
    >>> df = pd.DataFrame({"open": [1], "high": [2], "low": [0.5], "close": [1.5]})
    >>> validate_labeling_prerequisites(df)  # Raises ValueError: missing atr_14
    ValueError: Labeling prerequisites not met: Missing required columns: ['atr_14']

    >>> df["atr_14"] = 0.1
    >>> validate_labeling_prerequisites(df)  # OK, no exception raised
    """
    if required_cols is None:
        required_cols = REQUIRED_LABELING_COLUMNS + REQUIRED_OHLC_COLUMNS

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        context = ""
        if symbol or timeframe:
            context = (
                f" for {symbol} @ {timeframe}"
                if symbol and timeframe
                else f" for {symbol or timeframe}"
            )

        raise ValueError(
            f"Labeling prerequisites not met{context}: "
            f"Missing required columns: {missing_cols}. "
            f"Ensure feature engineering (Stage 3) completed successfully and includes ATR calculation. "
            f"ATR is required for dynamic barrier calculation (barriers are ATR multiples)."
        )


def _validate_horizons_vs_data(
    horizons: list[int],
    data_length: int,
    symbol: str = "",
    timeframe: str = "",
    raise_on_violation: bool = True,
) -> list[str]:
    """
    Validate that horizons are appropriate for the data length.

    This function checks that horizons are not too large relative to the data size.
    The rule is: max(horizons) should be < 10% of data_length to ensure at least
    10 samples per horizon for meaningful label distribution statistics.

    Parameters
    ----------
    horizons : list[int]
        List of horizon values
    data_length : int
        Number of rows in the data
    symbol : str, optional
        Symbol for context in log messages
    timeframe : str, optional
        Timeframe for context in log messages
    raise_on_violation : bool, optional
        If True (default), raise ValueError when validation fails.
        Set to False for warning-only behavior.

    Returns
    -------
    list[str]
        List of validation warnings (empty if all valid)

    Raises
    ------
    ValueError
        If raise_on_violation=True and horizons exceed recommended limits

    Notes
    -----
    Phase 12.5H: Added raise_on_violation parameter for stricter validation.
    Phase 25: Changed default to True for data integrity (fail-fast).
    """
    warnings = []

    if data_length <= 0:
        msg = f"Invalid data_length={data_length}, skipping horizon validation"
        logger.warning(msg)
        return [msg]

    max_recommended = data_length // 10
    context = f" for {symbol} @ {timeframe}" if symbol and timeframe else ""

    for h in horizons:
        if h >= max_recommended:
            msg = (
                f"Horizon {h} may be too large{context}: "
                f"data has {data_length:,} rows, recommended max horizon is {max_recommended}. "
                f"Consider using smaller horizons or more data for reliable label distributions."
            )
            warnings.append(msg)
            logger.warning(msg)

    if warnings and raise_on_violation:
        raise ValueError(
            f"Horizon validation failed with {len(warnings)} violations: {warnings[0]}"
        )

    return warnings


def run_initial_labeling(
    config: "PipelineConfig",
    manifest: "ArtifactManifest",
) -> "StageResult":
    """
    Stage 4: Initial Triple-Barrier Labeling.

    Applies initial triple-barrier labeling with default parameters for
    use as input to GA optimization.

    Args:
        config: Pipeline configuration
        manifest: Artifact manifest for tracking outputs

    Returns:
        StageResult with status and artifacts
    """
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("STAGE 4: Initial Triple-Barrier Labeling")
    logger.info("=" * 70)

    # Early validation: Check if ATR dependency might be missing due to feature toggles
    # This provides a clear error before processing any data (Issue #1: LBL-001)
    feature_toggles = getattr(config, "feature_toggles", None) or {}
    volatility_enabled = feature_toggles.get("volatility", True)
    if not volatility_enabled:
        raise ValueError(
            "Labeling stage requires 'atr_14' column but volatility features are disabled "
            "via feature_toggles['volatility']=False. Triple-barrier labeling uses ATR for "
            "dynamic barrier calculation (barriers are ATR multiples). Either enable "
            "volatility features or ensure 'atr_14' is available through another source."
        )

    try:
        # Create labels directory for GA input (run-scoped)
        labels_dir = config.run_data_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Get effective output timeframes (supports multi-TF mode)
        output_timeframes = config.effective_output_timeframes
        is_multi_tf = len(output_timeframes) > 1

        logger.info(f"Output timeframes: {output_timeframes}")
        logger.info(f"Multi-TF mode: {is_multi_tf}")

        artifacts = []
        label_stats: dict[str, dict[str, Any]] = {}

        # Get barrier overrides from config (if specified)
        barrier_overrides = getattr(config, "barrier_overrides", None) or {}

        # Track provenance for all labeling operations
        all_provenance: dict[str, dict[str, Any]] = {}

        for symbol in config.symbols:
            label_stats[symbol] = {}
            all_provenance[symbol] = {}

            # Process each output timeframe
            for tf in output_timeframes:
                # Load features data for this timeframe
                features_path = config.features_dir / f"{symbol}_{tf}_features.parquet"

                # Try in-memory cache first (optimization 5.2), fall back to disk
                df = None
                if getattr(config, "enable_in_memory_handoff", False):
                    cache = getattr(config, "_stage_data_cache", None)
                    if cache is not None:
                        df = cache.pop(str(features_path), None)
                        if df is not None:
                            logger.info(f"\nProcessing {symbol} @ {tf}... (from memory)")

                if df is None:
                    if not features_path.exists():
                        logger.warning(
                            f"Features file not found for {symbol} @ {tf}: {features_path}"
                        )
                        continue
                    logger.info(f"\nProcessing {symbol} @ {tf}...")
                    df = pd.read_parquet(features_path)

                logger.info(f"  Loaded {len(df):,} rows")

                # Validate labeling prerequisites early (LBL-001)
                # This provides a clear error if ATR or OHLC columns are missing
                validate_labeling_prerequisites(df, symbol=symbol, timeframe=tf)

                # Validate horizons against data length (LBL-005)
                # Log warnings if horizons are too large relative to data size
                _validate_horizons_vs_data(
                    horizons=config.label_horizons,
                    data_length=len(df),
                    symbol=symbol,
                    timeframe=tf,
                )

                tf_stats = {}
                tf_provenance: dict[str, Any] = {}
                atr_col = "atr_14"  # Required column validated above

                # Apply initial labeling with default parameters for each horizon
                for horizon in config.label_horizons:
                    # Determine barrier source and values (LBL-002: provenance tracking)
                    # Priority order:
                    # 1. barrier_overrides from config (explicit user override)
                    # 2. Symbol/horizon-specific defaults from barriers_config
                    # 3. Hard-coded fallbacks (only if barriers_config has no entry)
                    #
                    # Defaults will be optimized by GA in later stages
                    override_applied = bool(barrier_overrides)

                    if override_applied:
                        # Use explicit barrier overrides from config
                        k_up = barrier_overrides.get("k_up", 2.0)
                        k_down = barrier_overrides.get("k_down", 1.0)
                        max_bars = barrier_overrides.get("max_bars", horizon * 3)
                        barrier_source = "config_override"
                    else:
                        # Use symbol/horizon-specific defaults from barriers_config
                        barrier_defaults = get_barrier_params(symbol, horizon)
                        k_up = barrier_defaults.get("k_up", 2.0)
                        k_down = barrier_defaults.get("k_down", 1.0)
                        max_bars = barrier_defaults.get("max_bars", horizon * 3)
                        barrier_source = "barriers_config"

                    # Enforce max_bars_ahead cap if configured (Issue #2)
                    # max_bars should not exceed the pipeline's max_bars_ahead setting
                    if hasattr(config, "max_bars_ahead") and config.max_bars_ahead is not None:
                        max_bars = min(max_bars, config.max_bars_ahead)

                    logger.info(
                        f"  Horizon {horizon}: k_up={k_up}, k_down={k_down}, max_bars={max_bars}"
                    )

                    labels, bars_to_hit, mae, mfe, touch_type = triple_barrier_numba(
                        df["close"].values,
                        df["high"].values,
                        df["low"].values,
                        df[atr_col].values,
                        k_up,
                        k_down,
                        max_bars,
                    )

                    # Add columns
                    df[f"label_h{horizon}"] = labels
                    df[f"bars_to_hit_h{horizon}"] = bars_to_hit
                    df[f"mae_h{horizon}"] = mae
                    df[f"mfe_h{horizon}"] = mfe

                    # Calculate distribution
                    n_long = (labels == 1).sum()
                    n_short = (labels == -1).sum()
                    n_neutral = (labels == 0).sum()
                    total = len(labels)

                    tf_stats[horizon] = {
                        "long": int(n_long),
                        "short": int(n_short),
                        "neutral": int(n_neutral),
                        "long_pct": n_long / total * 100,
                        "short_pct": n_short / total * 100,
                        "neutral_pct": n_neutral / total * 100,
                    }

                    # Record provenance for this horizon (LBL-002)
                    tf_provenance[f"h{horizon}"] = {
                        "barrier_source": barrier_source,
                        "k_up_value": k_up,
                        "k_down_value": k_down,
                        "max_bars_value": max_bars,
                        "override_applied": override_applied,
                        "timestamp": datetime.now().isoformat(),
                    }

                    logger.info(
                        f"    Distribution: L={n_long/total*100:.1f}% "
                        f"S={n_short/total*100:.1f}% N={n_neutral/total*100:.1f}%"
                    )

                # Save to labels directory for GA input (with timeframe in filename)
                output_path = labels_dir / f"{symbol}_{tf}_labels_init.parquet"
                df.to_parquet(output_path, index=False)
                artifacts.append(output_path)

                label_stats[symbol][tf] = tf_stats
                all_provenance[symbol][tf] = tf_provenance

                manifest.add_artifact(
                    name=f"initial_labels_{symbol}_{tf}",
                    file_path=output_path,
                    stage="initial_labeling",
                    metadata={
                        "symbol": symbol,
                        "timeframe": tf,
                        "horizons": config.label_horizons,
                    },
                )

                logger.info(f"  Saved initial labels to {output_path}")

        # Save labeling provenance metadata (LBL-002)
        provenance_path = labels_dir / "labeling_provenance.json"
        provenance_metadata = {
            "stage": "initial_labeling",
            "created_at": datetime.now().isoformat(),
            "symbols": list(config.symbols),
            "timeframes": output_timeframes,
            "horizons": config.label_horizons,
            "provenance_by_symbol": all_provenance,
        }
        with open(provenance_path, "w") as f:
            json.dump(provenance_metadata, f, indent=2)
        artifacts.append(provenance_path)
        logger.info(f"Saved labeling provenance to {provenance_path}")

        return create_stage_result(
            stage_name="initial_labeling",
            start_time=start_time,
            artifacts=artifacts,
            metadata={
                "horizons": config.label_horizons,
                "label_stats": label_stats,
                "output_timeframes": output_timeframes,
                "multi_tf_mode": is_multi_tf,
                "labeling_provenance": all_provenance,
                "provenance_path": str(provenance_path),
            },
        )

    except Exception as e:
        logger.error(f"Initial labeling failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return create_failed_result(
            stage_name="initial_labeling", start_time=start_time, error=str(e)
        )
