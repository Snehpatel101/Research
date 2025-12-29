"""
Information commands - list models and their requirements.

Provides CLI commands for displaying available model types,
their data requirements, and ensemble configurations.
"""
from rich.table import Table

from .utils import console
from .run_commands_core import LazyImports


def models_command() -> None:
    """
    List available model types and their data requirements.

    Shows all supported models for Phase 2 training with their
    recommended feature sets, scaling requirements, and sequence lengths.
    """
    lazy = LazyImports()
    model_config = lazy.model_config

    console.print("\n[bold cyan]Available Model Types[/bold cyan]\n")

    # Create table for models
    table = Table(show_header=True, title="Model Data Requirements")
    table.add_column("Model", style="cyan")
    table.add_column("Family", style="yellow")
    table.add_column("Feature Set", style="green")
    table.add_column("Scaling", style="magenta")
    table.add_column("Sequences", style="blue")
    table.add_column("Description")

    for name in model_config.get_all_model_names():
        req = model_config.get_model_requirements(name)
        table.add_row(
            name,
            req.family.value,
            req.feature_set,
            req.scaler_type.value if req.requires_scaling else "none",
            str(req.sequence_length) if req.requires_sequences else "N/A",
            req.description[:50] + "..." if len(req.description) > 50 else req.description
        )

    console.print(table)

    # Show ensembles
    console.print("\n[bold cyan]Pre-defined Ensembles[/bold cyan]\n")

    ensemble_table = Table(show_header=True, title="Ensemble Configurations")
    ensemble_table.add_column("Ensemble", style="cyan")
    ensemble_table.add_column("Base Models", style="green")
    ensemble_table.add_column("Meta-Learner", style="yellow")
    ensemble_table.add_column("Description")

    for name in model_config.get_all_ensemble_names():
        ens = model_config.get_ensemble_config(name)
        ensemble_table.add_row(
            name,
            ", ".join(ens.base_models),
            ens.meta_learner,
            ens.description
        )

    console.print(ensemble_table)

    console.print("\n[dim]Use --model-type with 'pipeline run' to prepare data for specific models.[/dim]")
    console.print("[dim]Example: pipeline run --model-type lstm --feature-set neural_optimal[/dim]\n")
