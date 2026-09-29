from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

import typer

app = typer.Typer(
    help="Metrion - energy attribution for Linux processes, threads and cgroups.",
    no_args_is_help=True,
    add_completion=False,
)

TIME_FORMATS = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]


@app.command()
def monitor(
    config_path: Annotated[Path, typer.Option("--config", "-c", help="Path to the configuration file.")] = Path(
        __file__
    ).parent.parent
    / "configs"
    / "config.yaml",
    timeout_minutes: Annotated[
        float | None, typer.Option("--timeout", "-t", help="Stop after this many minutes instead of on Ctrl+C.")
    ] = None,
):
    """Collect energy and resource usage data until Ctrl+C or the timeout."""
    from metrion.orchestrator import Orchestrator

    Orchestrator(str(config_path)).start(timeout_minutes=timeout_minutes)


@app.command()
def analyze(
    db_path: Annotated[
        Path, typer.Option(exists=True, help="Database file of a run, or the directory of a Parquet run.")
    ],
    start_time: Annotated[
        datetime, typer.Option(formats=TIME_FORMATS, metavar="DATE", help="Start of the analysis window (UTC).")
    ],
    end_time: Annotated[
        datetime, typer.Option(formats=TIME_FORMATS, metavar="DATE", help="End of the analysis window (UTC).")
    ],
    filter_cmdline: Annotated[str, typer.Option(help="Substring of the command line or executable.")] = "",
    filter_pids: Annotated[str, typer.Option(help="Comma-separated PIDs, e.g. '1234,5678'.")] = "",
    unit: Annotated[Literal["J", "Wh"], typer.Option(help="Energy unit.")] = "J",
    force: Annotated[
        bool, typer.Option("--force", help="Rebuild the tables the analysis keeps in the database.")
    ] = False,
    backend: Annotated[Literal["matplotlib", "plotly"], typer.Option(help="Plotting backend.")] = "plotly",
    export_summary: Annotated[
        bool, typer.Option("--export-summary", help="Write the summary as CSV and JSON.")
    ] = False,
    no_plots: Annotated[bool, typer.Option("--no-plots", help="Skip plots and the HTML report.")] = False,
    export_raw_data: Annotated[
        bool, typer.Option("--export-raw-data", help="Write the filtered intervals as one CSV per device.")
    ] = False,
):
    """Attribute the energy of a run to processes and print a summary."""
    from metrion.energy_attribution.energy_attribution_cli import main

    if not filter_cmdline and not filter_pids:
        raise typer.BadParameter("give --filter-cmdline or --filter-pids")
    main(
        db_path,
        filter_cmdline,
        filter_pids,
        start_time,
        end_time,
        unit,
        force,
        backend,
        export_summary,
        no_plots,
        export_raw_data,
    )


if __name__ == "__main__":
    app()
