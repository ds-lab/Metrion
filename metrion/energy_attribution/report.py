from datetime import datetime
from pathlib import Path
from typing import Literal

import duckdb
import polars as pl
from rich.console import Console
from rich.table import Table

from metrion.data_storage.parquet_to_duckdb import ensure_duckdb
from metrion.energy_attribution.sql_attribution import get_energy_per_thread
from metrion.energy_attribution.wide_result import ENERGY_COLUMNS, IDENTITY_COLUMNS, to_wide

COMPONENTS = ["cpu", "dram", "gpu"]
PARTS = [f"{component}_{part}_j" for component in COMPONENTS for part in ("idle", "dynamic")]
TOTALS = [f"{component}_total_j" for component in COMPONENTS]

# The grid a database run is attributed on; --start and --end cut on it.
WINDOW_S = 1.0
# Rows the terminal shows, the rest is summed up in one line.
TOP = 20
SECONDS_PER_UNIT = {"J": 1, "Wh": 3600}
# About 0.1 J in either unit.
DECIMALS = {"J": 1, "Wh": 5}
NAME_WIDTH = 16
MIN_FLEXIBLE = 16

THREADS_SQL = """
SELECT
    id AS logical_entity_id,
    native_pid,
    native_thread_id,
    native_parent_pid,
    nullif(executable_name, '') AS executable_name,
    nullif(cmdline, '') AS cmdline,
    resource_group_name AS cgroup_name,
    user_id,
    group_id
FROM logical_entity
"""


def load(path: Path, start: datetime | None, end: datetime | None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Energy per window and thread, and every thread the run knows.

    Parameters
    ----------
    path : Path
        Database file of a run, the directory of a Parquet run, or the result directory of the
        live attribution.
    start, end : datetime or None
        UTC bounds of the windows to keep.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        Rows in the schema of the live result stream, and the identity of every thread. A live
        result only knows the threads that received energy.
    """
    if path.is_dir() and any(path.glob("part-*.parquet")):
        rows = pl.read_parquet(path / "part-*.parquet")
        threads = rows.select("logical_entity_id", *IDENTITY_COLUMNS).unique("logical_entity_id")
    else:
        if path.is_dir():
            path = ensure_duckdb(path)
        con = duckdb.connect()
        # Read only: the tables the attribution builds stay in memory, and the file may belong to root.
        con.execute(f"ATTACH '{path}' AS run (READ_ONLY)")
        con.execute("SET search_path = 'memory.main,run.main'")
        threads = con.sql(THREADS_SQL).pl().unique("logical_entity_id")
        per_thread = get_energy_per_thread(con, start or datetime(1970, 1, 1), end or datetime(9999, 1, 1))
        rows = to_wide(per_thread, WINDOW_S).join(threads, on="logical_entity_id", how="left")

    if start is not None:
        rows = rows.filter(pl.col("window_start") >= start)
    if end is not None:
        rows = rows.filter(pl.col("window_start") < end)
    return rows.select("window_start", "window_end", "logical_entity_id", *IDENTITY_COLUMNS, *ENERGY_COLUMNS), threads


def processes(threads: pl.DataFrame) -> pl.DataFrame:
    """One row per process with its parent, name, command line and cgroup, taken from its main thread."""
    return (
        threads.sort(pl.col("native_thread_id") != pl.col("native_pid"))
        .group_by("native_pid", maintain_order=True)
        .agg(
            # Every other thread names its own process as creator.
            ppid=pl.col("native_parent_pid").filter(pl.col("native_parent_pid") != pl.col("native_pid")).first(),
            name=pl.col("executable_name").drop_nulls().first(),
            cmdline=pl.col("cmdline").drop_nulls().first(),
            cgroup=pl.col("cgroup_name").drop_nulls().first(),
        )
        .rename({"native_pid": "pid"})
    )


def with_descendants(procs: pl.DataFrame, pids: set[int]) -> set[int]:
    children = dict(procs.drop_nulls("ppid").group_by("ppid").agg("pid").iter_rows())
    family, frontier = set(pids), set(pids)
    while frontier:
        frontier = {child for pid in frontier for child in children.get(pid, [])} - family
        family |= frontier
    return family


def select(
    rows: pl.DataFrame, procs: pl.DataFrame, cmdline: str | None, pids: list[int] | None, cgroup: str | None
) -> pl.DataFrame:
    """The rows of the selection; a process selected by command line or PID brings its descendants."""
    if cgroup is not None:
        return rows.filter(pl.col("cgroup_name").str.contains(cgroup, literal=True))
    if cmdline is not None:
        matches = pl.col("cmdline").str.contains(cmdline, literal=True) | pl.col("name").str.contains(
            cmdline, literal=True
        )
        pids = procs.filter(matches)["pid"].to_list()
    if pids is None:
        return rows
    return rows.filter(pl.col("native_pid").is_in(with_descendants(procs, set(pids))))


def summarize(selected: pl.DataFrame, procs: pl.DataFrame, by_cgroup: bool) -> pl.DataFrame:
    """One row per process, or per cgroup, with its energy; the largest first."""
    energy = [pl.col(column).sum() for column in PARTS + TOTALS]
    threads = pl.col("native_thread_id").n_unique().alias("threads")
    if by_cgroup:
        table = selected.group_by(pl.col("cgroup_name").alias("cgroup")).agg(
            pl.col("native_pid").n_unique().alias("processes"), threads, *energy
        )
        keys = ["cgroup", "processes", "threads"]
    else:
        table = (
            selected.group_by(pl.col("native_pid").alias("pid"))
            .agg(threads, *energy)
            .join(procs, on="pid", how="left", nulls_equal=True)
        )
        keys = ["pid", "ppid", "name", "threads", "cgroup", "cmdline"]
    return (
        table.with_columns(total_j=pl.sum_horizontal(TOTALS))
        .select(*keys, *PARTS, "total_j")
        .sort("total_j", descending=True, nulls_last=True)
    )


def print_report(
    rows: pl.DataFrame,
    selected: pl.DataFrame,
    table: pl.DataFrame,
    label: str,
    unit: Literal["J", "Wh"] = "J",
    console: Console | None = None,
) -> None:
    console = console or Console()
    total = table["total_j"].sum()
    energy = _amount(total, unit, True)
    if selected.height < rows.height:
        energy += f", {_share(total, rows.select(pl.sum_horizontal(TOTALS)).to_series().sum())} of the run"
    # Window bounds, so on the grid of the run: 1 s for a database, the configured window for a live result.
    active = selected.filter(pl.sum_horizontal(TOTALS) > 0)
    active = selected if active.is_empty() else active

    header = Table.grid(padding=(0, 2))
    header.add_row("Run", f"{_time(rows['window_start'].min())} → {_time(rows['window_end'].max())} UTC")
    header.add_row(
        "Selection",
        f"{label}: {selected['native_pid'].n_unique()} processes, {selected['native_thread_id'].n_unique()} threads",
    )
    header.add_row("Active", f"{_time(active['window_start'].min())} → {_time(active['window_end'].max())} UTC")
    header.add_row("Energy", energy)
    for component in COMPONENTS:
        idle = table[f"{component}_idle_j"].sum()
        used = idle + table[f"{component}_dynamic_j"].sum()
        if used:
            header.add_row(f"  {component.upper()}", f"{_amount(used, unit, True)}, {_share(idle, used)} idle")
    console.print(header)
    console.print()
    console.print(_energy_table(table, unit, console.width))


def _energy_table(table: pl.DataFrame, unit: Literal["J", "Wh"], width: int) -> Table:
    numeric = [column for column in PARTS if table[column].sum()] + ["total_j"]
    keys = [column for column in table.columns if column not in PARTS and column != "total_j"]
    by_cgroup = keys[0] == "cgroup"
    top, rest = table.head(TOP), table.slice(TOP)

    cells = {key: [_text(value, key == "cgroup" and not by_cgroup) for value in top[key]] for key in keys}
    cells |= {_heading(column): [_amount(value, unit) for value in top[column]] for column in numeric}
    footers = {keys[0]: "sum"} | {_heading(column): _amount(table[column].sum(), unit) for column in numeric}
    if not rest.is_empty():
        more = {keys[0]: f"… {rest.height} more"} | {_heading(c): _amount(rest[c].sum(), unit) for c in numeric}
        for name, column in cells.items():
            column.append(more.get(name, ""))
    if "name" in cells:
        cells["name"] = [_fit(value, NAME_WIDTH, False) for value in cells["name"]]

    def used(names: list[str]) -> int:
        # Each column is padded by one space on either side.
        return sum(max(len(name), len(footers.get(name, "")), *map(len, cells[name])) + 2 for name in names)

    # The last text column takes the room the others leave, and is left out below a useful width.
    # A cgroup path is cut at its start, which is the part all cgroups share.
    flexible = "cgroup" if by_cgroup else "cmdline"
    names = [name for name in cells if name != flexible]
    room = width - used(names) - 2
    if room >= MIN_FLEXIBLE or by_cgroup:
        cells[flexible] = [_fit(value, max(room, MIN_FLEXIBLE), by_cgroup) for value in cells[flexible]]
        names.insert(keys.index(flexible), flexible)
    for optional in ("cgroup", "name"):
        if used(names) > width and optional in names and optional != keys[0]:
            names.remove(optional)

    view = Table(title=f"Energy in {unit}", title_justify="left", show_footer=True, box=None, footer_style="bold")
    for name in names:
        view.add_column(name, footer=footers.get(name, ""), justify="left" if name in keys else "right", no_wrap=True)
    for row in zip(*(cells[name] for name in names), strict=True):
        view.add_row(*row)
    return view


def _heading(column: str) -> str:
    return column.removesuffix("_j").replace("_dynamic", " dyn").replace("_", " ")


def _text(value, last_segment: bool) -> str:
    if value is None:
        return "?"
    # The end of a cgroup path names the service, session or container.
    text = str(value).rsplit("/", 1)[-1] if last_segment else str(value)
    # A command line can hold newlines, e.g. python -c with a script.
    return " ".join(text.split())


def _fit(text: str, width: int, cut_start: bool) -> str:
    if len(text) <= width:
        return text
    return "…" + text[1 - width :] if cut_start else text[: width - 1] + "…"


def _time(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _amount(joules: float, unit: Literal["J", "Wh"], with_unit: bool = False) -> str:
    value = joules / SECONDS_PER_UNIT[unit]
    decimals = DECIMALS[unit]
    # A value too small for the decimals shown is still not nothing.
    text = f"<{10**-decimals:.{decimals}f}" if 0 < value < 0.5 * 10**-decimals else f"{value:,.{decimals}f}"
    return f"{text} {unit}" if with_unit else text


def _share(part: float, whole: float) -> str:
    return f"{part / whole:.1%}" if whole else "–"
