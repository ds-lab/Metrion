import multiprocessing
import os
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
    typer.echo(f"Wrote {CONFIG.absolute()}. Adjust it, then")
    typer.echo(f"  measure the idle power once per machine: sudo {program} calibrate")
    typer.echo(f"  start a run in this directory:           sudo {program} monitor")


@app.command()
def calibrate(
    freeze: Annotated[
        list[str] | None, typer.Option(help="Also freeze active units matching this pattern, e.g. machine.slice.")
    ] = None,
    skip: Annotated[
        list[str] | None, typer.Option(help="Leave units matching this pattern running, e.g. 'docker-*.scope'.")
    ] = None,
    seconds: Annotated[int, typer.Option(help="Length of the measurement.")] = 60,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Freeze without asking.")] = False,
):
    """Measure the idle power of CPU, DRAM and GPUs while the rest of the machine is frozen."""
    from metrion import calibration

    if os.geteuid() != 0:
        typer.echo("Freezing units and reading RAPL needs root: sudo metrion calibrate", err=True)
        raise typer.Exit(1)
    try:
        active = calibration.active_units()
        units = calibration.select_units(active, freeze or [], skip or [])
        info = calibration.describe(units) if units else {}
    except calibration.CalibrationError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from e

    left_running = [unit for unit in units if info[unit].get("CanFreeze") == "no"]
    units = [unit for unit in units if unit not in left_running]
    typer.echo("Frozen during the measurement:")
    width = max(len(unit) for unit in units)
    for unit in units:
        tasks = info[unit].get("TasksCurrent", "")
        note = "  logins and desktop, including this terminal" if unit == "user.slice" else ""
        typer.echo(f"  {unit:<{width}} {tasks if tasks.isdigit() else '-':>6} tasks{note}")
    if left_running:
        typer.echo(f"Cannot be frozen, left running: {', '.join(left_running)}")
    for unit in active:
        if unit in calibration.OPT_IN and unit not in units:
            typer.echo(f"Left running: {unit} (virtual machines); add --freeze {unit} to include it")
    holding = calibration.gpu_processes()
    if holding:
        typer.echo("Holding a GPU, they keep their GPU context: " + ", ".join(holding))

    duration = seconds + calibration.SETTLE_S
    if not yes and not typer.confirm(f"Freeze these {len(units)} units for about {duration} s?"):
        raise typer.Exit(1)
    try:
        result = calibration.run(units, seconds)
    except calibration.CalibrationError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from e

    for unit, error in result["not_frozen"].items():
        typer.echo(f"Not frozen, measured anyway: {unit} ({error})", err=True)
    for component in ("cpu", "dram", "gpu"):
        for device, watts in result[component].items():
            typer.echo(f"{component.upper():<5} {device:<44} {'no counter' if watts is None else f'{watts:.2f} W'}")
    typer.echo(f"Saved to {calibration.cache_file()}")


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

    from metrion.calibration import CalibrationError
    from metrion.orchestrator import Orchestrator

    # Every helper process starts as a fresh interpreter. A forked one inherits the parent's
    # memory but not its threads, so a thread pool the parent started (polars has one) hangs in it.
    multiprocessing.set_start_method("spawn", force=True)
    try:
        Orchestrator(str(config_path)).start(timeout_minutes=timeout_minutes)
    except CalibrationError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from e


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
