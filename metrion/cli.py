import shutil
import sys
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
DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"
# Relative, so every command looks in the directory it runs in.
CONFIG = Path("metrion.yaml")


@app.command()
def init():
    """Write the default configuration to ./metrion.yaml, to adjust before the first run."""
    if CONFIG.exists():
        typer.echo(f"{CONFIG.absolute()} exists already; delete it to start from the default again.", err=True)
        raise typer.Exit(1)
    shutil.copyfile(DEFAULT_CONFIG, CONFIG)
    # sudo only searches its own PATH, so name the program by its full path.
    program = shutil.which("metrion") or "metrion"
    typer.echo(f"Wrote {CONFIG.absolute()}. Adjust it, then start a run here with: sudo {program} monitor")


@app.command()
def monitor(
    config_path: Annotated[
        Path, typer.Option("--config", "-c", help="Configuration file; `metrion init` writes one.")
    ] = CONFIG,
    timeout_minutes: Annotated[
        float | None, typer.Option("--timeout", "-t", help="Stop after this many minutes instead of on Ctrl+C.")
    ] = None,
):
    """Collect energy and resource usage data until Ctrl+C or the timeout."""
    if not config_path.exists():
        typer.echo(f"No configuration at {config_path.absolute()}; create one with `metrion init`.", err=True)
        raise typer.Exit(1)

    from metrion.orchestrator import Orchestrator

    Orchestrator(str(config_path)).start(timeout_minutes=timeout_minutes)


@app.command()
def analyze(
    path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            help="Database file of a run, the directory of a Parquet run, or the result directory of the "
            "live attribution.",
        ),
    ],
    cmdline: Annotated[
        str | None, typer.Option(help="Processes whose command line or name contains this, with their children.")
    ] = None,
    pids: Annotated[str | None, typer.Option(help="Comma-separated PIDs, with their children.")] = None,
    cgroup: Annotated[str | None, typer.Option(help="cgroups whose path contains this, one row each.")] = None,
    start: Annotated[
        datetime | None, typer.Option(formats=TIME_FORMATS, metavar="DATE", help="Start of the analysis (UTC).")
    ] = None,
    end: Annotated[
        datetime | None, typer.Option(formats=TIME_FORMATS, metavar="DATE", help="End of the analysis (UTC).")
    ] = None,
    unit: Annotated[Literal["J", "Wh"], typer.Option(help="Unit of the energy shown.")] = "J",
    csv: Annotated[
        Path | None,
        typer.Option(
            metavar="DIR",
            help="Also write the whole table and the selected rows per window and thread as CSV, in joules.",
        ),
    ] = None,
):
    """Show how much energy each process of a run used; without a filter, the processes that used the most."""
    if sum(option is not None for option in (cmdline, pids, cgroup)) > 1:
        raise typer.BadParameter("give at most one of --cmdline, --pids and --cgroup")
    try:
        pid_list = None if pids is None else [int(pid) for pid in pids.split(",")]
    except ValueError as e:
        raise typer.BadParameter("--pids takes comma-separated numbers, e.g. 1234,5678") from e

    from loguru import logger

    from metrion.energy_attribution.report import load, print_report, processes, select, summarize

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    rows, threads = load(path, start, end)
    procs = processes(threads)
    selected = select(rows, procs, cmdline, pid_list, cgroup)
    if cmdline is not None:
        label = f'cmdline "{cmdline}"'
    elif pids is not None:
        label = f"pids {pids}"
    elif cgroup is not None:
        label = f'cgroup "{cgroup}"'
    else:
        label = "all processes"
    if selected.is_empty():
        typer.echo(f"Nothing matched {label}.", err=True)
        raise typer.Exit(1)

    table = summarize(selected, procs, cgroup is not None)
    print_report(rows, selected, table, label, unit)
    if csv is not None:
        csv.mkdir(parents=True, exist_ok=True)
        table.write_csv(csv / "table.csv")
        selected.sort("window_start", "native_pid", "native_thread_id").write_csv(csv / "rows.csv")
        typer.echo(f"Wrote table.csv and rows.csv to {csv}")


if __name__ == "__main__":
    app()
