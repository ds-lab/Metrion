import argparse
from pathlib import Path

import duckdb
import polars as pl

from metrion.energy_attribution.energy_attribution_cli import get_energy_per_thread
from metrion.energy_attribution.wide_result import ENERGY_COLUMNS, to_wide

WINDOW_KEY = ["window_start", "logical_entity_id"]
TOTAL_COLUMNS = ["cpu_total_j", "dram_total_j", "gpu_total_j"]
LIVE = "_live"
BOTH_SIDES = [*ENERGY_COLUMNS, *(column + LIVE for column in ENERGY_COLUMNS)]
# The queries of metrion analyze leave the measured energy of every interval in these tables.
MEASURED_INTERVALS = {
    "cpu": "energy_interval_pkg_tmp",
    "dram": "energy_interval_dram_tmp",
    "gpu": "energy_interval_gpu_tmp",
}


def load_live(folder: Path) -> tuple[pl.DataFrame, float, int]:
    """Read the result stream and sum it per window and thread.

    Returns
    -------
    tuple[pl.DataFrame, float, int]
        The summed stream, its window length in seconds, and how many rows repeated a window and
        thread. A component that comes back after it had stalled writes such a second row.
    """
    rows = pl.read_parquet(str(folder / "part-*.parquet")).select(["window_end", *WINDOW_KEY, *ENERGY_COLUMNS])
    window = (rows["window_end"][0] - rows["window_start"][0]).total_seconds()
    summed = rows.group_by(WINDOW_KEY).agg(pl.col(ENERGY_COLUMNS).sum())
    return summed, window, rows.height - summed.height


def analyze_database(con: duckdb.DuckDBPyConnection, window: float) -> pl.DataFrame:
    """What metrion analyze computes for the whole run, on the grid of the live stream."""
    first, last = con.sql("SELECT min(start_ts), max(end_ts) FROM measurement").fetchone()
    return to_wide(get_energy_per_thread(con, first, last), window).drop("window_end")


def coverage(con: duckdb.DuckDBPyConnection, live: pl.DataFrame) -> pl.DataFrame:
    """How much of the energy the counters measured reaches a thread in the live stream, per component."""
    return pl.DataFrame(
        {
            "component": list(MEASURED_INTERVALS),
            "measured_j": [
                con.sql(f"SELECT coalesce(sum(energy_j), 0) FROM {table}").fetchone()[0]
                for table in MEASURED_INTERVALS.values()
            ],
            "live_j": [live[f"{component}_total_j"].sum() for component in MEASURED_INTERVALS],
        }
    ).with_columns(live_pct=pl.when(pl.col("measured_j") > 0).then(pl.col("live_j") / pl.col("measured_j") * 100))


def _threads(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    return (
        con.sql("SELECT id AS logical_entity_id, native_thread_id, native_pid, executable_name FROM logical_entity")
        .pl()
        .unique("logical_entity_id")
    )


def side_by_side(db: pl.DataFrame, live: pl.DataFrame, key: list[str]) -> pl.DataFrame:
    """Both sides per key; energy that one side does not have counts as 0 J there."""
    joined = db.with_columns(in_db=pl.lit(True)).join(
        live.with_columns(in_live=pl.lit(True)), on=key, how="full", coalesce=True, nulls_equal=True, suffix=LIVE
    )
    return joined.with_columns(pl.col(BOTH_SIDES).fill_null(0.0), pl.col("in_db", "in_live").fill_null(False))


def sums(compared: pl.DataFrame) -> pl.DataFrame:
    return (
        pl.DataFrame(
            {
                "column": ENERGY_COLUMNS,
                "db_j": [compared[column].sum() for column in ENERGY_COLUMNS],
                "live_j": [compared[column + LIVE].sum() for column in ENERGY_COLUMNS],
            }
        )
        .with_columns(diff_j=pl.col("live_j") - pl.col("db_j"))
        .with_columns(diff_pct=pl.when(pl.col("db_j") != 0).then(pl.col("diff_j") / pl.col("db_j") * 100))
    )


def deviations(compared: pl.DataFrame, min_joules: float) -> tuple[pl.DataFrame, pl.DataFrame]:
    """How far live and database lie apart per energy column, in joules and in percent of the database value.

    deviating_pct is sum(|live - db|) / sum(db); energy that moved between two rows counts twice,
    once where it is missing and once where it is extra. The percentages cover only rows with at
    least min_joules in the database, below that they are noise.
    """
    long = pl.concat(
        compared.select(
            pl.lit(column).alias("column"),
            pl.col(column).alias("db"),
            (pl.col(column + LIVE) - pl.col(column)).abs().alias("abs"),
        )
        for column in ENERGY_COLUMNS
    ).with_columns(rel=pl.when(pl.col("db") >= min_joules).then(pl.col("abs") / pl.col("db") * 100))

    absolute = long.group_by("column", maintain_order=True).agg(
        pl.len().alias("rows"),
        pl.when(pl.col("db").sum() > 0).then(pl.col("abs").sum() / pl.col("db").sum() * 100).alias("deviating_pct"),
        *_stats(pl.col("abs")),
    )
    relative = long.group_by("column", maintain_order=True).agg(
        pl.col("rel").count().alias("rows"), *_stats(pl.col("rel"))
    )
    return absolute, relative


def _stats(values: pl.Expr) -> list[pl.Expr]:
    return [
        values.min().alias("min"),
        values.median().alias("median"),
        values.mean().alias("mean"),
        values.quantile(0.95).alias("p95"),
        values.quantile(0.99).alias("p99"),
        values.max().alias("max"),
    ]


def presence(compared: pl.DataFrame) -> str:
    only_db = compared.filter(~pl.col("in_live"))
    only_live = compared.filter(~pl.col("in_db"))
    return (
        f"{compared.height} rows, {only_db.height} only in the database ({_joules(only_db, ''):.3f} J), "
        f"{only_live.height} only live ({_joules(only_live, LIVE):.3f} J)"
    )


def _joules(frame: pl.DataFrame, suffix: str) -> float:
    return frame.select(pl.sum_horizontal([column + suffix for column in TOTAL_COLUMNS])).sum().item()


def top_consumers(processes: pl.DataFrame, rows: int = 10) -> pl.DataFrame:
    """The processes that got the most energy in the live stream."""
    return (
        processes.select(
            "native_pid", "executable_name", *(pl.col(column + LIVE).alias(column) for column in TOTAL_COLUMNS)
        )
        .with_columns(total_j=pl.sum_horizontal(TOTAL_COLUMNS))
        .sort("total_j", descending=True)
        .head(rows)
    )


def furthest_apart(compared: pl.DataFrame, key: list[str], rows: int = 10) -> pl.DataFrame:
    """The rows whose total energy over CPU, DRAM and GPU lies furthest apart."""
    return (
        compared.with_columns(
            db_j=pl.sum_horizontal(TOTAL_COLUMNS),
            live_j=pl.sum_horizontal([column + LIVE for column in TOTAL_COLUMNS]),
        )
        .with_columns(diff_j=pl.col("live_j") - pl.col("db_j"))
        .sort(pl.col("diff_j").abs(), descending=True)
        .head(rows)
        .select([*key, "executable_name", "db_j", "live_j", "diff_j"])
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the live result stream of a run with what metrion analyze computes from its database, "
        "and show how much of the measured energy the stream attributes."
    )
    parser.add_argument("database", type=Path, help="DuckDB file of the run")
    parser.add_argument("live_folder", type=Path, help="directory of the live result stream")
    parser.add_argument(
        "--min-joules",
        type=float,
        default=1.0,
        help="relative deviations only for rows with at least this much energy in the database (default: 1 J)",
    )
    args = parser.parse_args()
    if not args.database.is_file():
        parser.error(f"{args.database} is not a file")
    if not any(args.live_folder.glob("part-*.parquet")):
        parser.error(f"no part-*.parquet in {args.live_folder}")

    live, window, repeated = load_live(args.live_folder)
    print(f"Running the queries of metrion analyze on {args.database} ...")
    with duckdb.connect() as con:
        # The run's tables are read from the file, the tables the queries create stay in memory.
        con.execute(f"ATTACH '{args.database}' AS run (READ_ONLY)")
        con.execute("SET search_path = 'memory.main,run.main'")
        db = analyze_database(con, window)
        threads_in_db = _threads(con)
        covered = coverage(con, live)

    windows = side_by_side(db, live, WINDOW_KEY)
    threads = (
        windows.group_by("logical_entity_id")
        .agg(pl.col(BOTH_SIDES).sum(), pl.col("in_db", "in_live").any())
        .join(threads_in_db, on="logical_entity_id", how="left")
    )
    # native_pid, not pid_id: pid_id splits a process for which BPF reported a start of 0 for some
    # of its threads, as it did for metrion's own processes. A PID seldom comes back within one run.
    processes = threads.group_by("native_pid").agg(
        pl.col(BOTH_SIDES).sum(),
        pl.col("in_db", "in_live").any(),
        # The main thread's name; the others often carry their own, like polars-3.
        pl.col("executable_name").sort_by(pl.col("native_thread_id") != pl.col("native_pid")).first(),
    )

    with pl.Config(
        tbl_rows=-1,
        tbl_cols=-1,
        tbl_width_chars=200,
        float_precision=4,
        tbl_hide_dataframe_shape=True,
        tbl_hide_column_data_types=True,
    ):
        print(
            f"Live stream: {window:g} s windows from {live['window_start'].min()} to {live['window_start'].max()}, "
            f"{repeated} rows repeated a window and thread"
        )
        print("\nEnergy the counters measured and what of it the live stream attributes")
        print(covered)
        print("\nSums over the run")
        print(sums(windows))
        for level, compared in (("window and thread", windows), ("thread", threads), ("process", processes)):
            absolute, relative = deviations(compared, args.min_joules)
            print(f"\nPer {level}: {presence(compared)}")
            print("|live - db| in J")
            print(absolute)
            print(f"|live - db| in % of db, rows with at least {args.min_joules:g} J in the database")
            print(relative)
        print("\nTop consumers in the live stream")
        print(top_consumers(processes))
        print("\nThreads furthest apart")
        print(furthest_apart(threads, ["logical_entity_id", "native_pid", "native_thread_id"]))
        print("\nProcesses furthest apart")
        print(furthest_apart(processes, ["native_pid"]))


if __name__ == "__main__":
    main()
