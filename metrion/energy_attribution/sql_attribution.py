import datetime
from pathlib import Path
from typing import Literal

import duckdb
import polars as pl
import sqlparse
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from metrion.metric import (
    CPU_PKG_PID_DYNAMIC_ENERGY,
    CPU_PKG_PID_IDLE_ENERGY,
    CPU_PKG_PID_TOTAL_ENERGY,
    DRAM_SOCKET_PID_DYNAMIC_ENERGY,
    DRAM_SOCKET_PID_IDLE_ENERGY,
    DRAM_SOCKET_PID_TOTAL_ENERGY,
    GPU_DEVICE_PID_DYNAMIC_ENERGY,
    GPU_DEVICE_PID_IDLE_ENERGY,
    GPU_DEVICE_PID_TOTAL_ENERGY,
)

# ============================================================================
# Configuration Constants
# ============================================================================

SENTINEL_DURATION_MS = 500  # milliseconds for collapsing timestamps
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

# ============================================================================
# Database Connection & Query Execution
# ============================================================================


def connect_to_db(db_path: Path) -> Engine | None:
    """
    Connect to a DuckDB database using SQLAlchemy.

    Parameters
    ----------
    db_path : Path
        Path to the DuckDB database file.

    Returns
    -------
    Engine or None
        SQLAlchemy Engine object if successful, None otherwise.

    Notes
    -----
    Checks if the database file exists before attempting to connect.
    Logs errors if the connection fails.
    """
    try:
        if not db_path.exists():
            logger.error(f"Database file does not exist: {db_path}")
            return None

        database_url = f"duckdb:///{db_path}"
        engine = create_engine(database_url, echo=False)

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        logger.info(f"Successfully connected to database: {db_path}")
        return engine

    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemy error connecting to database {db_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error connecting to database {db_path}: {e}")
        return None


def close_db_connection(engine: Engine) -> None:
    """
    Safely dispose of a SQLAlchemy engine.

    Parameters
    ----------
    engine : Engine
        SQLAlchemy Engine to dispose.

    Notes
    -----
    Disposes the engine if it exists. Logs success or error.
    """
    try:
        if engine:
            engine.dispose()
            logger.info("Database engine disposed successfully")
    except SQLAlchemyError as e:
        logger.error(f"Error disposing database engine: {e}")


def load_sql_queries() -> tuple[str, str, str, str, str, str]:
    """
    Load SQL queries from files.

    Returns
    -------
    tuple of str
        (CPU energy analysis by PID query,
         CPU energy per group query,
         DRAM energy analysis by PID query,
         DRAM energy per group query,
         GPU energy analysis by PID query,
         GPU energy per group query)
    """
    cpu_energy_analysis_by_thread_path = Path(__file__).parent / "cpu_energy_analysis_by_thread.sql"
    query_cpu_energy_per_group_path = Path(__file__).parent / "cpu_energy_per_group.sql"
    dram_energy_analysis_by_thread_path = Path(__file__).parent / "dram_energy_analysis_by_thread.sql"
    dram_energy_per_group_path = Path(__file__).parent / "dram_energy_per_group.sql"
    gpu_energy_analysis_by_thread_path = Path(__file__).parent / "gpu_energy_analysis_by_thread.sql"
    gpu_energy_per_group_path = Path(__file__).parent / "gpu_energy_per_group.sql"

    def _read_sql(path: Path) -> str:
        return path.read_text(encoding="utf-8")

    query_cpu_energy_analysis_by_thread = _read_sql(cpu_energy_analysis_by_thread_path)
    query_cpu_energy_per_group = _read_sql(query_cpu_energy_per_group_path)
    query_dram_energy_analysis_by_thread = _read_sql(dram_energy_analysis_by_thread_path)
    query_dram_energy_per_group = _read_sql(dram_energy_per_group_path)
    query_gpu_energy_analysis_by_thread = _read_sql(gpu_energy_analysis_by_thread_path)
    query_gpu_energy_per_group = _read_sql(gpu_energy_per_group_path)

    return (
        query_cpu_energy_analysis_by_thread,
        query_cpu_energy_per_group,
        query_dram_energy_analysis_by_thread,
        query_dram_energy_per_group,
        query_gpu_energy_analysis_by_thread,
        query_gpu_energy_per_group,
    )


def _execute_group_query(
    engine: Engine,
    view_name: str,
    create_view_sql_statements: list[str],
    group_query_sql: str,
    force: bool,
    schema: dict[str, pl.DataType],
) -> pl.DataFrame:
    with engine.connect() as conn:
        if force:
            logger.debug(f"Recreating table {view_name} (force={force})")
            conn.execute(text(f"DROP TABLE IF EXISTS {view_name}"))
        for create_view_sql_statement in create_view_sql_statements:
            conn.execute(text(create_view_sql_statement))
        result = conn.execute(text(group_query_sql))
        df = pl.DataFrame(result, schema=schema)

    return df


# The table each per-thread query leaves behind, with the metrics of its total, idle and active column.
_THREAD_TABLES = {
    "total_energy_per_thread_view": (CPU_PKG_PID_TOTAL_ENERGY, CPU_PKG_PID_IDLE_ENERGY, CPU_PKG_PID_DYNAMIC_ENERGY),
    "dram_energy_per_thread_view": (
        DRAM_SOCKET_PID_TOTAL_ENERGY,
        DRAM_SOCKET_PID_IDLE_ENERGY,
        DRAM_SOCKET_PID_DYNAMIC_ENERGY,
    ),
    "gpu_energy_per_thread_view": (
        GPU_DEVICE_PID_TOTAL_ENERGY,
        GPU_DEVICE_PID_IDLE_ENERGY,
        GPU_DEVICE_PID_DYNAMIC_ENERGY,
    ),
}


def get_energy_per_thread(
    con: duckdb.DuckDBPyConnection, start_time: datetime.datetime, end_time: datetime.datetime
) -> list[pl.DataFrame]:
    """Run the per-thread queries of `metrion analyze` and return their result as attribution frames.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Sees the run's tables. The queries create their own tables only where they are missing,
        so the connection must not hold them yet.
    start_time, end_time : datetime.datetime
        Range of the energy samples to attribute, by their start.

    Returns
    -------
    list[pl.DataFrame]
        One frame per component and part, one row per energy interval and thread, in the schema
        of the attribution models without physical_entity_id; to_wide takes them as they are.
    """
    cpu_query, _, dram_query, _, gpu_query, _ = load_sql_queries()
    bounds = {"start_time": start_time.strftime(DATETIME_FORMAT), "end_time": end_time.strftime(DATETIME_FORMAT)}
    for query in (cpu_query, dram_query, gpu_query):
        for statement in sqlparse.split(query.format(**bounds)):
            con.execute(statement)

    frames = []
    for table, metrics in _THREAD_TABLES.items():
        # A row without a thread belongs to no one; the GPU query writes two with 0 J per idle GPU and interval.
        rows = con.sql(f"SELECT * FROM {table} WHERE logical_entity_id IS NOT NULL").pl()
        for metric, column in zip(metrics, ("total_energy_j", "idle_energy_j", "active_energy_j"), strict=True):
            frames.append(
                rows.select(
                    pl.lit(metric.name).alias("metric_id"),
                    "logical_entity_id",
                    pl.col(column).alias("value"),
                    pl.col("interval_start").alias("start_ts"),
                    pl.col("interval_end").alias("end_ts"),
                )
            )
    return frames


def execute_queries(
    engine: Engine, start_time: datetime.datetime, end_time: datetime.datetime, filter: str = "", force: bool = False
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame] | None:
    """
    Execute energy analysis queries against the database.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        SQLAlchemy Engine connected to the database.
    start_time : datetime.datetime
        Start datetime for the analysis window.
    end_time : datetime.datetime
        End datetime for the analysis window.
    filter : str, optional
        SQL filter condition for results.
    force : bool, optional
        If True, recreate views even if they exist.

    Returns
    -------
    tuple of polars.DataFrame or None
        (cpu_energy_per_group_df, dram_energy_per_group_df, gpu_energy_per_group_df)
        Returns None if query execution fails.
    """
    # try:
    (
        query_cpu_energy_analysis_by_thread,
        query_cpu_energy_per_group,
        query_dram_energy_analysis_by_thread,
        query_dram_energy_per_group,
        query_gpu_energy_analysis_by_thread,
        query_gpu_energy_per_group,
    ) = load_sql_queries()

    # Convert datetime objects to strings for SQLite
    start_str = start_time.strftime("%Y-%m-%d %H:%M:%S.%f")
    end_str = end_time.strftime("%Y-%m-%d %H:%M:%S.%f")

    # Format the query with time parameters
    formatted_query_cpu = query_cpu_energy_analysis_by_thread.format(start_time=start_str, end_time=end_str)
    formatted_query_dram = query_dram_energy_analysis_by_thread.format(start_time=start_str, end_time=end_str)
    formatted_query_gpu = query_gpu_energy_analysis_by_thread.format(start_time=start_str, end_time=end_str)

    formatted_query_cpu_group = query_cpu_energy_per_group.format(filter_condition=filter)
    formatted_query_dram_group = query_dram_energy_per_group.format(filter_condition=filter)
    formatted_query_gpu_group = query_gpu_energy_per_group.format(filter_condition=filter)

    # Common schema for energy results
    energy_schema = {
        "interval_start": pl.Datetime,
        "interval_end": pl.Datetime,
        "cmdline": pl.String,
        "executable_name": pl.String,
        "resource_group_name": pl.String,
        "resource_group_id": pl.Int64,
        "total_energy_j": pl.Float64,
        "active_energy_j": pl.Float64,
        "idle_energy_j": pl.Float64,
    }

    # CPU
    cpu_query_statements = sqlparse.split(formatted_query_cpu)
    cpu_energy_per_group_df = _execute_group_query(
        engine, "total_energy_per_thread_view", cpu_query_statements, formatted_query_cpu_group, force, energy_schema
    )
    logger.info(f"Successfully loaded {len(cpu_energy_per_group_df)} CPU energy records")

    # DRAM
    dram_query_statements = sqlparse.split(formatted_query_dram)
    dram_energy_per_group_df = _execute_group_query(
        engine, "dram_energy_per_thread_view", dram_query_statements, formatted_query_dram_group, force, energy_schema
    )
    logger.info(f"Successfully loaded {len(dram_energy_per_group_df)} DRAM energy records")

    # GPU
    gpu_query_statements = sqlparse.split(formatted_query_gpu)
    try:
        gpu_energy_per_group_df = _execute_group_query(
            engine, "gpu_energy_per_thread_view", gpu_query_statements, formatted_query_gpu_group, force, energy_schema
        )
        logger.info(f"Successfully loaded {len(gpu_energy_per_group_df)} GPU energy records")
    except Exception as e:
        logger.warning(f"GPU energy query failed (may not have GPU data): {e}")
        gpu_energy_per_group_df = pl.DataFrame()

    return cpu_energy_per_group_df, dram_energy_per_group_df, gpu_energy_per_group_df


def filter_data_by_cmdline_and_comm(df: pl.DataFrame | None, filter_cmdline: str) -> pl.DataFrame | None:
    """
    Filter the energy data by command line pattern.

    Parameters
    ----------
    df : polars.DataFrame or None
        Input DataFrame containing energy data.
    filter_cmdline : str
        Command line substring or pattern to filter by.

    Returns
    -------
    polars.DataFrame or None
        Filtered DataFrame containing only rows where 'cmdline' matches the pattern.
        Returns None if input is empty or no matches found.

    Notes
    -----
    Uses string containment to filter the 'cmdline' column.
    Logs a warning if no matching records are found.
    """
    if df is None or df.is_empty():
        logger.warning("Input DataFrame is empty or None")
        return None

    filtered_df = df.filter(
        pl.col("cmdline").str.contains(filter_cmdline) | pl.col("executable_name").str.contains(filter_cmdline)
    )

    if filtered_df.is_empty():
        logger.warning(f"No data found for cmdline pattern: {filter_cmdline}")
        return None

    logger.info(f"Filtered to {len(filtered_df)} records matching '{filter_cmdline}'")
    return filtered_df


# ============================================================================
# Data Processing & Filtering
# ============================================================================


def collapse_time_stamps(df: pl.DataFrame | None) -> pl.DataFrame | None:
    if df is None or df.is_empty():
        logger.warning("Input DataFrame is empty or None")
        return None
    sentinel = pl.duration(milliseconds=SENTINEL_DURATION_MS)
    group_keys = ["cmdline", "resource_group_id", "resource_group_name"]

    temp_df = (
        df
        # ensure correct order inside each group
        .sort(group_keys + ["interval_start"])
        .with_columns(
            [
                # compute gap shift gets previous value)
                (pl.col("interval_start") - pl.col("interval_start").shift(1).over(group_keys)).alias("gap"),
            ]
        )
        # start a new cluster if:
        #  - this is the first row in the group (gap is null), or
        #  - the gap is >= sentinel
        .with_columns(
            [
                (
                    (pl.col("gap").is_null() | (pl.col("gap") >= sentinel))
                    .cast(pl.Int64)
                    .cum_sum()  # running count over the whole frame
                    .over(group_keys)  # but reset per group
                ).alias("cluster_id"),
            ]
        )
    )

    collapsed_df = (
        temp_df.group_by(group_keys + ["cluster_id"])
        .agg(
            [
                pl.col("interval_start").min().alias("interval_start"),
                pl.col("interval_end").max().alias("interval_end"),
                pl.col("total_energy_j").sum().alias("total_energy_j"),
                pl.col("active_energy_j").sum().alias("active_energy_j"),
                pl.col("idle_energy_j").sum().alias("idle_energy_j"),
                pl.col("executable_name").first().alias("executable_name"),
            ]
        )
        .sort(group_keys + ["interval_start"])
    )

    collapsed_df = collapsed_df.select(
        pl.col("interval_start").cast(pl.Datetime),
        pl.col("interval_end").cast(pl.Datetime),
        pl.col("cmdline"),
        pl.col("executable_name"),
        pl.col("resource_group_name"),
        pl.col("resource_group_id"),
        pl.col("total_energy_j").cast(pl.Float64),
        pl.col("active_energy_j").cast(pl.Float64),
        pl.col("idle_energy_j").cast(pl.Float64),
    )
    return collapsed_df


def prepare_data(
    db_path: Path,
    filter_cmdline: str,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    unit: Literal["J", "Wh"],
    filter: str = "",
    force: bool = False,
) -> tuple[list[tuple[str, pl.DataFrame]], list[tuple[str, pl.DataFrame]]] | None:
    """
    Prepare energy data from database for visualization.

    Parameters
    ----------
    db_path : Path
        Database file, or the directory of a Parquet run to build one from.
    filter_cmdline : str
        Command line pattern to filter processes.
    start_time : datetime.datetime
        Start time for analysis.
    end_time : datetime.datetime
        End time for analysis.
    unit : {'J', 'Wh'}
        Energy unit for display.
    filter : str, optional
        SQL filter condition.
    force : bool, optional
        Force recreation of views.

    Returns
    -------
    tuple or None
        (filtered_dfs, all_dfs) where each is a list of (origin, DataFrame) tuples.
    """
    if db_path.is_dir():
        # A Parquet run. The analyses are SQL, so build the database now - this is the point
        # where it is needed, and it is skipped when the file is already up to date.
        from metrion.data_storage.parquet_to_duckdb import ensure_duckdb

        db_path = ensure_duckdb(db_path)

    # Connect to database
    engine = connect_to_db(db_path)
    if not engine:
        logger.error("Failed to connect to database. Exiting.")
        return

    result = execute_queries(engine, start_time, end_time, filter, force)
    if result is None:
        logger.error("Failed to execute queries. Exiting.")
        return

    close_db_connection(engine)

    cpu_energy_per_group_df, dram_energy_per_group_df, gpu_energy_per_group_df = result

    # Collapse time stamps
    logger.debug(
        f"Status raw: cpu: {len(cpu_energy_per_group_df)}, "
        f"dram: {len(dram_energy_per_group_df)}, "
        f"gpu: {len(gpu_energy_per_group_df)}"
    )
    collapsed_df_cpu = collapse_time_stamps(cpu_energy_per_group_df)
    collapsed_df_dram = collapse_time_stamps(dram_energy_per_group_df)
    collapsed_df_gpu = collapse_time_stamps(gpu_energy_per_group_df)

    logger.debug(
        f"Status collapsed: cpu: {len(collapsed_df_cpu) if collapsed_df_cpu is not None else 0}, "
        f"dram: {len(collapsed_df_dram) if collapsed_df_dram is not None else 0}, "
        f"gpu: {len(collapsed_df_gpu) if collapsed_df_gpu is not None else 0}"
    )

    # Filter data by command line (skip if no cmdline filter provided)
    if filter_cmdline:
        filtered_df_cpu = filter_data_by_cmdline_and_comm(collapsed_df_cpu, filter_cmdline)
        filtered_df_dram = filter_data_by_cmdline_and_comm(collapsed_df_dram, filter_cmdline)
        filtered_df_gpu = filter_data_by_cmdline_and_comm(collapsed_df_gpu, filter_cmdline)
    else:
        filtered_df_cpu = collapsed_df_cpu
        filtered_df_dram = collapsed_df_dram
        filtered_df_gpu = collapsed_df_gpu

    filtered_dfs = []
    if filtered_df_cpu is not None and filtered_df_cpu.is_empty() is False:
        filtered_dfs.append(("cpu", filtered_df_cpu))
    if filtered_df_dram is not None and filtered_df_dram.is_empty() is False:
        filtered_dfs.append(("dram", filtered_df_dram))
    if filtered_df_gpu is not None and filtered_df_gpu.is_empty() is False:
        filtered_dfs.append(("gpu", filtered_df_gpu))

    if len(filtered_dfs) == 0:
        logger.warning(f"No data found for filter: cmdline='{filter_cmdline}', sql='{filter}'")
        return

    all_dfs = [
        ("cpu", cpu_energy_per_group_df),
        ("dram", dram_energy_per_group_df),
        ("gpu", gpu_energy_per_group_df),
    ]

    return filtered_dfs, all_dfs
