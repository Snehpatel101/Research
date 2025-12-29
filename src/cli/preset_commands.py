"""
CLI Preset Commands - commands for viewing and managing trading presets.
"""
from typing import Optional

import typer
from rich.table import Table
from rich.panel import Panel

from .utils import console, show_info
from .run_commands_core import LazyImports


def presets_command(
    preset_name: Optional[str] = typer.Argument(
        None,
        help="Preset name to show details for (optional)"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed preset configuration"
    )
) -> None:
    """
    List available trading presets or show details for a specific preset.

    Examples:
        pipeline presets                  # List all presets
        pipeline presets scalping         # Show scalping preset details
        pipeline presets day_trading -v   # Show day_trading with full config
    """
    lazy = LazyImports()
    presets_mod = lazy.presets

    console.print("\n[bold cyan]Trading Presets[/bold cyan]\n")

    if preset_name:
        # Show details for specific preset
        _show_preset_details(preset_name, presets_mod, verbose)
    else:
        # List all presets
        _list_all_presets(presets_mod)


def _list_all_presets(presets_mod) -> None:
    """Display a table of all available presets."""
    available = presets_mod.list_available_presets()

    table = Table(title="Available Trading Presets", show_header=True)
    table.add_column("Name", style="cyan", width=15)
    table.add_column("Timeframe", style="green", width=12)
    table.add_column("Horizons", style="yellow", width=15)
    table.add_column("Sessions", style="magenta", width=25)
    table.add_column("Description", style="white")

    for preset_name in available:
        config = presets_mod.get_preset(preset_name)

        sessions_str = ", ".join(config.get('sessions', []))
        horizons_str = ", ".join(map(str, config.get('horizons', [])))

        table.add_row(
            preset_name,
            config.get('target_timeframe', 'N/A'),
            horizons_str,
            sessions_str,
            config.get('description', 'N/A')
        )

    console.print(table)
    console.print()

    # Usage hint
    console.print("[dim]Usage: pipeline run --preset <name> [other options][/dim]")
    console.print("[dim]       pipeline presets <name> -v   # for detailed view[/dim]")
    console.print()


def _show_preset_details(preset_name: str, presets_mod, verbose: bool) -> None:
    """Display detailed information for a specific preset."""
    try:
        # Validate and get preset
        presets_mod.validate_preset(preset_name)
        config = presets_mod.get_preset(preset_name)
    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)

    # Header
    console.print(Panel(
        f"[bold]{config.get('name', preset_name)}[/bold]\n"
        f"[dim]{config.get('description', 'N/A')}[/dim]",
        title=f"Preset: {preset_name}",
        border_style="cyan"
    ))
    console.print()

    # Main settings table
    main_table = Table(title="Core Settings", show_header=True, box=None)
    main_table.add_column("Setting", style="cyan")
    main_table.add_column("Value", style="green")

    main_table.add_row("Target Timeframe", config.get('target_timeframe', 'N/A'))
    main_table.add_row("Label Horizons", ", ".join(map(str, config.get('horizons', []))))
    main_table.add_row("Max Bars Ahead", str(config.get('max_bars_ahead', 'N/A')))
    main_table.add_row("Sessions", ", ".join(config.get('sessions', [])))
    main_table.add_row("Labeling Strategy", config.get('labeling_strategy', 'N/A'))
    main_table.add_row("Barrier Multiplier", str(config.get('barrier_multiplier', 1.0)))
    main_table.add_row("Min Trade Duration", f"{config.get('min_trade_duration_bars', 1)} bars")

    console.print(main_table)
    console.print()

    if verbose:
        # Feature config table
        feature_config = config.get('feature_config', {})
        if feature_config:
            feat_table = Table(title="Feature Configuration", show_header=True, box=None)
            feat_table.add_column("Feature", style="cyan")
            feat_table.add_column("Value", style="yellow")

            feat_table.add_row("SMA Periods", str(feature_config.get('sma_periods', [])))
            feat_table.add_row("EMA Periods", str(feature_config.get('ema_periods', [])))
            feat_table.add_row("ATR Periods", str(feature_config.get('atr_periods', [])))
            feat_table.add_row("RSI Period", str(feature_config.get('rsi_period', 14)))

            console.print(feat_table)
            console.print()

        # Risk config table
        risk_config = config.get('risk_config', {})
        if risk_config:
            risk_table = Table(title="Risk Configuration", show_header=True, box=None)
            risk_table.add_column("Parameter", style="cyan")
            risk_table.add_column("Value", style="red")

            risk_table.add_row("Max Positions", str(risk_config.get('max_positions', 'N/A')))
            risk_table.add_row("Stop Loss ATR Mult", str(risk_config.get('stop_loss_atr_mult', 'N/A')))
            risk_table.add_row("Take Profit ATR Mult", str(risk_config.get('take_profit_atr_mult', 'N/A')))

            console.print(risk_table)
            console.print()

    # Usage example
    console.print("[bold]Usage Example:[/bold]")
    console.print(f"  pipeline run --preset {preset_name} --symbols MES,MGC")
    console.print()
