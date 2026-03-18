import os
import tempfile

# Set matplotlib cache directory before importing matplotlib to avoid warnings
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import datetime
import json
from datetime import UTC
from pathlib import Path
from typing import Literal

import click
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import sqlparse
from loguru import logger
from matplotlib.figure import Figure as MatplotlibFigure
from matplotlib.ticker import FixedLocator
from plotly.subplots import make_subplots
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

# ============================================================================
# Configuration Constants
# ============================================================================

# Data processing constants
SENTINEL_DURATION_MS = 500  # milliseconds for collapsing timestamps
PAD_FRACTION = 0.04  # padding for plot ranges

# Plot style constants
DEFAULT_FIGURE_SIZE = (15, 12)
LINE_WIDTH = 2.5
MARKER_SIZE = 6
GRID_ALPHA = 0.2
TIME_FORMAT = "%H:%M:%S"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

# Color schemes for consistent styling
PLOT_COLORS = {
    "matplotlib_cmap": "tab20",
    "plotly_palette": px.colors.qualitative.Plotly,
    "primary": "#2C3E50",
    "secondary": "#3498DB",
    "success": "#27AE60",
    "warning": "#F39C12",
    "danger": "#E74C3C",
}

PLOT_STYLE = {
    "title_font": {"size": 18, "weight": "bold", "color": PLOT_COLORS["primary"]},
    "axis_font": {"size": 12, "color": PLOT_COLORS["primary"]},
    "legend_font": {"size": 10},
    "grid_alpha": GRID_ALPHA,
    "line_width": LINE_WIDTH,
    "marker_size": MARKER_SIZE,
}

# ============================================================================
# Database Connection & Query Execution
# ============================================================================


def connect_to_db(db_path: Path, type: str) -> Engine | None:
    """
    Connect to a SQLite database using SQLAlchemy.

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database file.

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

        database_url = f"{type}:///{db_path}"
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


def load_sql_queries(dialect: str) -> tuple[str, str, str, str, str, str]:
    """
    Load SQL queries from files.

    Parameters
    ----------
    dialect : str
        Database dialect ('sqlite' or 'duckdb').

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
        content = path.read_text(encoding="utf-8")

        if dialect != "sqlite":
            return content

        lines = content.splitlines()

        def _strip_sqlite_marker(line: str) -> str:
            stripped = line.lstrip()
            if stripped.startswith("--sqlite"):
                leading_ws_len = len(line) - len(stripped)
                leading_ws = line[:leading_ws_len]
                rest = stripped[len("--sqlite") :].lstrip()
                return leading_ws + rest
            return line

        return "\n".join(_strip_sqlite_marker(line) for line in lines) + "\n"

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

    if engine.dialect.name == "sqlite":
        df = df.with_columns(
            pl.col("interval_start").str.to_datetime("%Y-%m-%dT%H:%M:%S%.6f"),
            pl.col("interval_end").str.to_datetime("%Y-%m-%dT%H:%M:%S%.6f"),
        )

    return df


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
    ) = load_sql_queries(engine.dialect.name)

    # Convert datetime objects to strings for SQLite
    start_str = start_time.strftime("%Y-%m-%d %H:%M:%S.%f")
    end_str = end_time.strftime("%Y-%m-%d %H:%M:%S.%f")
    dialect = engine.dialect.name

    if dialect == "sqlite":
        date_method = "julianday"  # SQLite date method julian
        max_func = "MAX"
    elif dialect == "duckdb":
        date_method = "julian"
        max_func = "greatest"
    else:
        raise ValueError(f"Unsupported database dialect: {dialect}")

    # Format the query with time parameters
    formatted_query_cpu = query_cpu_energy_analysis_by_thread.format(
        start_time=start_str, end_time=end_str, date_method=date_method, max_func=max_func
    )
    formatted_query_dram = query_dram_energy_analysis_by_thread.format(
        start_time=start_str, end_time=end_str, date_method=date_method, max_func=max_func
    )
    formatted_query_gpu = query_gpu_energy_analysis_by_thread.format(
        start_time=start_str, end_time=end_str, date_method=date_method, max_func=max_func
    )

    formatted_query_cpu_group = query_cpu_energy_per_group.format(filter_condition=filter)
    formatted_query_dram_group = query_dram_energy_per_group.format(filter_condition=filter)
    formatted_query_gpu_group = query_gpu_energy_per_group.format(filter_condition=filter)

    # Common schema for energy results
    energy_schema = {
        "interval_start": pl.Datetime if dialect == "duckdb" else pl.String,
        "interval_end": pl.Datetime if dialect == "duckdb" else pl.String,
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


def get_energy_for_pids(
    engine: Engine,
    pids: list[int],
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    force: bool = False,
) -> pl.DataFrame:
    """
    Get energy attribution for a list of PIDs as a wide-format DataFrame.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        SQLAlchemy Engine connected to the database.
    pids : list[int]
        OS-native PIDs to include in the result.
    start_time : datetime.datetime
        Start datetime for the analysis window.
    end_time : datetime.datetime
        End datetime for the analysis window.
    force : bool, optional
        If True, recreate intermediate views even if they already exist.

    Returns
    -------
    polars.DataFrame
        Wide-format DataFrame with one row per (native_pid, interval). Columns:
        native_pid, cmdline, executable_name, interval_start, interval_end,
        cpu_total_energy_j, cpu_active_energy_j, cpu_idle_energy_j,
        dram_total_energy_j, dram_active_energy_j, dram_idle_energy_j,
        gpu_total_energy_j, gpu_active_energy_j, gpu_idle_energy_j,
        total_energy_j, total_active_energy_j, total_idle_energy_j.
    """
    if not pids:
        return pl.DataFrame()

    pid_list = ", ".join(str(p) for p in pids)
    filter_condition = f"AND le.native_pid IN ({pid_list})"

    (
        query_cpu_energy_analysis_by_thread,
        _,
        query_dram_energy_analysis_by_thread,
        _,
        query_gpu_energy_analysis_by_thread,
        _,
    ) = load_sql_queries(engine.dialect.name)

    start_str = start_time.strftime("%Y-%m-%d %H:%M:%S.%f")
    end_str = end_time.strftime("%Y-%m-%d %H:%M:%S.%f")
    dialect = engine.dialect.name

    if dialect == "sqlite":
        date_method = "julianday"
        max_func = "MAX"
    elif dialect == "duckdb":
        date_method = "julian"
        max_func = "greatest"
    else:
        raise ValueError(f"Unsupported database dialect: {dialect}")

    formatted_query_cpu = query_cpu_energy_analysis_by_thread.format(
        start_time=start_str, end_time=end_str, date_method=date_method, max_func=max_func
    )
    formatted_query_dram = query_dram_energy_analysis_by_thread.format(
        start_time=start_str, end_time=end_str, date_method=date_method, max_func=max_func
    )
    formatted_query_gpu = query_gpu_energy_analysis_by_thread.format(
        start_time=start_str, end_time=end_str, date_method=date_method, max_func=max_func
    )

    sql_dir = Path(__file__).parent
    query_cpu_per_pid = (sql_dir / "cpu_energy_per_pid.sql").read_text(encoding="utf-8").format(
        filter_condition=filter_condition
    )
    query_dram_per_pid = (sql_dir / "dram_energy_per_pid.sql").read_text(encoding="utf-8").format(
        filter_condition=filter_condition
    )
    query_gpu_per_pid = (sql_dir / "gpu_energy_per_pid.sql").read_text(encoding="utf-8").format(
        filter_condition=filter_condition
    )

    per_pid_schema = {
        "interval_start": pl.Datetime if dialect == "duckdb" else pl.String,
        "interval_end": pl.Datetime if dialect == "duckdb" else pl.String,
        "native_pid": pl.Int64,
        "cmdline": pl.String,
        "total_energy_j": pl.Float64,
        "active_energy_j": pl.Float64,
        "idle_energy_j": pl.Float64,
    }

    join_keys = ["native_pid", "cmdline", "interval_start", "interval_end"]

    cpu_df = _execute_group_query(
        engine, "total_energy_per_thread_view", sqlparse.split(formatted_query_cpu), query_cpu_per_pid, force, per_pid_schema
    ).rename({"total_energy_j": "cpu_total_energy_j", "active_energy_j": "cpu_active_energy_j", "idle_energy_j": "cpu_idle_energy_j"})

    dram_df = _execute_group_query(
        engine, "dram_energy_per_thread_view", sqlparse.split(formatted_query_dram), query_dram_per_pid, force, per_pid_schema
    ).rename({"total_energy_j": "dram_total_energy_j", "active_energy_j": "dram_active_energy_j", "idle_energy_j": "dram_idle_energy_j"})

    try:
        gpu_df = _execute_group_query(
            engine, "gpu_energy_per_thread_view", sqlparse.split(formatted_query_gpu), query_gpu_per_pid, force, per_pid_schema
        ).rename({"total_energy_j": "gpu_total_energy_j", "active_energy_j": "gpu_active_energy_j", "idle_energy_j": "gpu_idle_energy_j"})
    except Exception as e:
        logger.warning(f"GPU energy query failed (may not have GPU data): {e}")
        gpu_df = pl.DataFrame()

    result = cpu_df.join(dram_df, on=join_keys, how="full", coalesce=True)
    if not gpu_df.is_empty():
        result = result.join(gpu_df, on=join_keys, how="full", coalesce=True)

    energy_cols = [
        "cpu_total_energy_j", "cpu_active_energy_j", "cpu_idle_energy_j",
        "dram_total_energy_j", "dram_active_energy_j", "dram_idle_energy_j",
        "gpu_total_energy_j", "gpu_active_energy_j", "gpu_idle_energy_j",
    ]

    for col in energy_cols:
        if col not in result.columns:
            result = result.with_columns(pl.lit(0.0).alias(col))

    result = result.with_columns(pl.col(col).fill_null(0.0) for col in energy_cols)

    result = (
        result.group_by(["native_pid", "cmdline"])
        .agg(pl.col(energy_cols).sum())
    )

    result = result.with_columns(
        (pl.col("cpu_total_energy_j") + pl.col("dram_total_energy_j") + pl.col("gpu_total_energy_j")).alias("total_energy_j"),
        (pl.col("cpu_active_energy_j") + pl.col("dram_active_energy_j") + pl.col("gpu_active_energy_j")).alias("total_active_energy_j"),
        (pl.col("cpu_idle_energy_j") + pl.col("dram_idle_energy_j") + pl.col("gpu_idle_energy_j")).alias("total_idle_energy_j"),
    )

    return result


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


# ============================================================================
# Visualization - Time Series & Bar Charts
# ============================================================================


def generate_html_report(
    filtered_figures: list,
    all_figures: list,
    filtered_tables: list,
    all_tables: list,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    filter_cmdline: str,
    output_path: str | None = None,
) -> None:
    """
    Generate an HTML report with all the figures.

    Parameters
    ----------
    filtered_figures : list
        List of plotly figures for filtered data.
    all_figures : list
        List of plotly figures for all data.
    start_time : datetime.datetime
        Analysis start time.
    end_time : datetime.datetime
        Analysis end time.
    filter_cmdline : str
        Process filter used.
    output_path : str or None, optional
        Output HTML file path. If None, a timestamped file is created in the output directory.

    Returns
    -------
    None

    Notes
    -----
    - Generates a styled HTML report containing interactive Plotly charts for filtered and all data.
    - Responsive design for desktop and mobile.
    - Charts are rendered using Plotly.js.
    - The report includes analysis parameters and generation timestamp.
    """
    import plotly.io as pio

    # Set default output path if not provided
    if output_path is None:
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S.%f")
        output_path = str(output_dir / f"energy_analysis_report_{timestamp}.html")

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Energy Analysis Report</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px 40px;
                background-color: #f5f5f5;
                max-width: calc(100% - 80px);
                overflow-x: auto;
            }}
            .header {{
                background-color: #2c3e50;
                color: white;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
            }}
            .info-section {{
                background-color: white;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .chart-container {{
                background-color: white;
                padding: 15px 25px;
                border-radius: 8px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                width: calc(100% - 50px);
                overflow-x: auto;
            }}
            .chart-title {{
                font-size: 16px;
                font-weight: bold;
                margin-bottom: 15px;
                color: #2c3e50;
            }}
            .section-title {{
                font-size: 22px;
                font-weight: bold;
                margin: 30px 0 20px 0;
                color: #34495e;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }}
            .icon {{
                color: #3498db;
                font-weight: bold;
            }}

            /* Responsive design */
            @media (max-width: 768px) {{
                body {{
                    margin: 10px 20px;
                    max-width: calc(100% - 40px);
                }}
                .header {{
                    padding: 15px;
                }}
                .header h1 {{
                    font-size: 24px;
                }}
                .chart-container {{
                    padding: 10px 15px;
                    width: calc(100% - 30px);
                }}
                .chart-title {{
                    font-size: 14px;
                }}
                .section-title {{
                    font-size: 18px;
                }}
            }}

            @media (max-width: 480px) {{
                body {{
                    margin: 5px 15px;
                    max-width: calc(100% - 30px);
                }}
                .header h1 {{
                    font-size: 20px;
                }}
                .section-title {{
                    font-size: 16px;
                }}
                .chart-title {{
                    font-size: 12px;
                }}
                .chart-container {{
                    padding: 8px 12px;
                    width: calc(100% - 24px);
                }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1><span class="icon">&#9889;</span> Energy Analysis Report</h1>
            <p>Comprehensive power consumption analysis for system processes</p>
        </div>

        <div class="info-section">
            <h2><span class="icon">&#128202;</span> Analysis Parameters</h2>
            <ul>
                <li><strong>Time Range:</strong> {start_time.strftime("%Y-%m-%d %H:%M:%S.%f")}
                    to {end_time.strftime("%Y-%m-%d %H:%M:%S.%f")}</li>
                <li><strong>Duration:</strong> {end_time - start_time}</li>
                <li><strong>Process Filter:</strong> "{filter_cmdline}"</li>
                <li><strong>Generated:</strong> {datetime.datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S.%f")}</li>
            </ul>
        </div>
    """

    # Add filtered data charts
    if filtered_figures:
        html_content += f"""
        <div class="section-title">
            <span class="icon">&#128269;</span> Filtered Data Analysis (Process: "{filter_cmdline}")
        </div>
        """

        for i, _ in enumerate(filtered_figures):
            chart_title = f"Filtered Data - Chart {i + 1}"
            if i == 0:
                chart_title = "Energy Consumption Over Time (Filtered)"
            elif i == 1:
                chart_title = "Total Energy by Process (Filtered)"

            html_content += f"""
            <div class="chart-container">
                <div class="chart-title">{chart_title}</div>
                <div id="chart-filtered-{i}"></div>
            </div>
            """
    # Add filtered data tables
    for table in filtered_tables:
        html_content += f"""
        <div class="chart-container">
            <div class="chart-title">Filtered Data - Table</div>
            {table._repr_html_()}
        </div>
        """
    # Add all data charts
    if all_figures:
        html_content += """
        <div class="section-title"><span class="icon">&#128200;</span> Complete Dataset Analysis</div>
        """

        for i, _ in enumerate(all_figures):
            chart_title = f"All Data - Chart {i + 1}"
            if i == 0:
                chart_title = "Energy Consumption Over Time (All Processes)"
            elif i == 1:
                chart_title = "Total Energy by Process (All Processes)"

            html_content += f"""
            <div class="chart-container">
                <div class="chart-title">{chart_title}</div>
                <div id="chart-all-{i}"></div>
            </div>
            """
    # Add filtered data tables
    for table in all_tables:
        html_content += f"""
        <div class="chart-container">
            <div class="chart-title">All Data - Table</div>
            {table._repr_html_()}
        </div>
    """

    # Add JavaScript to render the charts
    html_content += """
        <script>
    """

    # Add filtered figures
    for i, fig in enumerate(filtered_figures):
        fig_json = pio.to_json(fig)
        html_content += f"""
            (function() {{
                var fig = JSON.parse({fig_json!r});
                Plotly.newPlot('chart-filtered-{i}', fig.data, fig.layout, {{
                    responsive: true,
                    displayModeBar: false
                }});
            }})();
        """

    # Add all data figures
    for i, fig in enumerate(all_figures):
        fig_json = pio.to_json(fig)
        html_content += f"""
            (function() {{
                var fig = JSON.parse({fig_json!r});
                Plotly.newPlot('chart-all-{i}', fig.data, fig.layout, {{
                    responsive: true,
                    displayModeBar: false
                }});
            }})();
        """

    html_content += """
        // Add window resize handler to ensure charts resize properly
        window.addEventListener('resize', function() {
            var charts = document.querySelectorAll('[id^="chart-"]');
            charts.forEach(function(chart) {
                Plotly.Plots.resize(chart);
            });
        });
        </script>
    </body>
    </html>
    """

    # Write the HTML file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"HTML report generated: {output_path}")


def plot_energy_consumption_over_time(
    data: list[tuple[str, pl.DataFrame]],
    title="Energy Consumption Over Time",
    backend: Literal["matplotlib", "plotly"] = "matplotlib",
    unit: Literal["J", "Wh"] = "J",
) -> go.Figure | MatplotlibFigure | None:
    """
    Plot energy consumption over time for each process and origin.

    Parameters
    ----------
    data_by_origin : list of tuple[str, polars.DataFrame]
        List of (origin_label, DataFrame) tuples. Each DataFrame should contain energy data with columns:
            - 'interval_start': str or datetime, interval start time
            - 'interval_end': str or datetime, interval end time
            - 'total_energy_j': float, energy consumed in the interval
            - 'cmdline': str, process command line
            - 'resource_group_name': str, resource_group_name identifier
            - 'resource_group_id': int, resource_group_name ID
    title : str, optional
        Title for the plot (default: "Energy Consumption Over Time").
    backend : {"matplotlib", "plotly"}, optional
        Plotting backend to use (default: "matplotlib").

    Returns
    -------
    go.Figure or matplotlib.figure.Figure or None
        Plotly Figure, Matplotlib Figure, or None if input is empty.

    Notes
    -----
    - Converts interval start/end to datetime.
    - Computes midpoint of each interval for cumulative plots.
    - Plots step and cumulative energy for each unique (origin, process).
    - Returns the figure object; does not display or save.
    """
    processed_data = []
    for origin, df in data:
        if df.is_empty():
            continue
        # Parse timestamps & midpoint, sort
        df_processed = (
            df.with_columns(
                [
                    (pl.col("total_energy_j") / 3600 if unit == "Wh" else pl.col("total_energy_j")).alias(
                        "total_energy"
                    ),
                ]
            )
            .with_columns(
                [
                    # Be robust to either string or datetime inputs
                    pl.col("interval_start").cast(pl.Datetime, strict=False),
                    pl.col("interval_end").cast(pl.Datetime, strict=False),
                ]
            )
            # drop rows with missing timestamps or energy
            .filter(
                pl.col("interval_start").is_not_null()
                & pl.col("interval_end").is_not_null()
                & pl.col("total_energy").is_not_null()
            )
            .with_columns(
                [
                    (
                        (pl.col("interval_start") + (pl.col("interval_end") - pl.col("interval_start")) / 2).alias(
                            "interval_mid"
                        )
                    )
                ]
            )
            .sort("interval_start")
        )

        processed_data.append((origin, df_processed))

    if backend == "matplotlib":
        return plot_energy_consumption_over_time_matplotlib(processed_data, title, unit)
    elif backend == "plotly":
        return plot_energy_consumption_over_time_plotly(processed_data, title, unit)


def plot_energy_consumption_over_time_matplotlib(
    data_by_origin: list[tuple[str, pl.DataFrame]],
    title,
    unit: Literal["J", "Wh"] = "J",
):
    """
    Plot step & cumulative energy for multiple origins and cmdlines.

    Parameters
    ----------
    data_by_origin : list of tuple[str, pl.DataFrame]
        List of (origin_label, DataFrame) tuples. Each DataFrame should contain energy data with columns:
        - 'interval_start': str or datetime, interval start time
        - 'interval_end': str or datetime, interval end time
        - 'total_energy': float, energy consumed in the interval in J or Wh
        - 'cmdline': str, process command line
        - 'resource_group_name': str, resource_group_name identifier
        - 'resource_group_id': int, resource_group_name ID

    title : str
        Title text shown above both sub-plots.

    Returns
    -------
    matplotlib.figure.Figure
        The Matplotlib figure object (does not display by default).
    """

    # ---------------------------------------------------------------------
    # 1) Determine all (origin, cmdline) pairs – they uniquely map to a line
    # ---------------------------------------------------------------------
    pairs: list[tuple[str, str]] = []
    for origin, df in data_by_origin:
        if df.is_empty():
            continue
        for cmdline in df.select("cmdline").unique()["cmdline"]:
            pairs.append((origin, cmdline))
    pairs = sorted(set(pairs))  # stable ordering for reproducible colours

    cmap = plt.get_cmap(PLOT_COLORS["matplotlib_cmap"])  # up to 20 distinct hues
    colours = {pair: cmap(i % 20) for i, pair in enumerate(pairs)}

    # 2) Create figure with shared‑x sub‑plots
    fig, (ax_step, ax_cum) = plt.subplots(2, 1, sharex=True, figsize=DEFAULT_FIGURE_SIZE, dpi=100)

    all_step_times: list = []
    all_mid_times: list = []

    # ------------------------------------------------------------------
    # 3) STEP PLOT – repeat the energy value for the entire interval
    # ------------------------------------------------------------------
    for origin, df in data_by_origin:
        for cmdline in df.select("cmdline").unique()["cmdline"]:
            cmd_df = df.filter(pl.col("cmdline") == cmdline)
            if cmd_df.is_empty():
                continue

            times, energies = [], []
            for row in cmd_df.iter_rows(named=True):
                times.extend([row["interval_start"], row["interval_end"]])
                energies.extend([row["total_energy"], row["total_energy"]])
                all_step_times.extend([row["interval_start"], row["interval_end"]])

            label = f"{origin} · {cmdline}"
            ax_step.plot(
                times,
                energies,
                label=label,
                color=colours[(origin, cmdline)],
                drawstyle="steps-post",
                marker="o",
                markersize=MARKER_SIZE - 2,
                linewidth=LINE_WIDTH,
            )

    ax_step.set_ylabel(f"Energy ({unit})", fontsize=12)
    ax_step.set_title(f"{title} – Step plot", fontsize=14, fontweight="bold")
    ax_step.grid(True, alpha=PLOT_STYLE["grid_alpha"], linestyle="--", linewidth=0.5)

    ## ------------------------------------------------------------------
    # 4) CUMULATIVE PLOT – energy over time
    # ------------------------------------------------------------------
    for origin, df in data_by_origin:
        for cmdline in df.select("cmdline").unique()["cmdline"]:
            cmd_df = df.filter(pl.col("cmdline") == cmdline)
            if cmd_df.is_empty():
                continue

            cmd_df = cmd_df.with_columns(pl.col("total_energy").cum_sum().alias("cumulative_energy"))

            label = f"{origin} · {cmdline}"
            ax_cum.plot(
                cmd_df["interval_mid"],
                cmd_df["cumulative_energy"],
                label=label,
                color=colours[(origin, cmdline)],
                marker="o",
                markersize=MARKER_SIZE - 2,
                linewidth=LINE_WIDTH,
            )
            all_mid_times.extend(cmd_df["interval_mid"])

    ax_cum.set_xlabel("Time", fontsize=12)
    ax_cum.set_ylabel(f"Cumulative energy ({unit})", fontsize=12)
    ax_cum.set_title(f"{title} – Cumulative energy", fontsize=14, fontweight="bold")
    ax_cum.grid(True, alpha=PLOT_STYLE["grid_alpha"], linestyle="--", linewidth=0.5)

    # 5) Combine the legends (place to the right of the plots)
    ax_step.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left")
    ax_cum.legend(bbox_to_anchor=(1.02, 0.0), loc="lower left")

    # 6) Align x‑ticks exactly with interval boundaries / mid‑points
    xticks = mdates.date2num(sorted(set(all_step_times + all_mid_times)))
    locator = FixedLocator(list(xticks))
    formatter = mdates.DateFormatter("%H:%M:%S")
    for ax in (ax_step, ax_cum):
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    return fig


def plot_energy_consumption_over_time_plotly(
    data_by_origin: list[tuple[str, pl.DataFrame]],
    title: str,
    unit: Literal["J", "Wh"] = "J",
) -> go.Figure:
    """
    Generates an interactive Plotly figure showing step and cumulative energy consumption over time for multiple origins
    and command lines.

    This function creates two vertically stacked subplots:
        1. Step plot: Visualizes energy consumption per interval for each (origin, cmdline) pair.
        2. Cumulative plot: Shows the cumulative energy consumed over time for each (origin, cmdline) pair.

    Each trace is color-coded by (origin, cmdline), and interactive tooltips provide detailed information
    for each data point.

    Parameters
    ----------
    data_by_origin : list[tuple[str, pl.DataFrame]]
        List of tuples containing an origin label and a Polars DataFrame.
        Each DataFrame must include the following columns:
            - 'interval_start': datetime or str, start time of the interval
            - 'interval_end': datetime or str, end time of the interval
            - 'interval_mid': datetime or str, midpoint of the interval
            - 'total_energy': float, energy consumed in the interval (in J or Wh)
            - 'cmdline': str, process command line
            - 'resource_group_name': str, resource_group_name identifier
            - 'resource_group_id': int, resource_group_name ID

    title : str
        Title text displayed above both subplots.

    Returns
    -------
    go.Figure
        Plotly Figure object containing the interactive step and cumulative energy plots.
    """

    # 1) Colour map for every (origin, cmdline) combination
    pairs = sorted({(origin, cmd) for origin, df in data_by_origin for cmd in df["cmdline"].unique()})
    palette = PLOT_COLORS["plotly_palette"]  # large categorical palette
    colour_map = {pair: palette[i % len(palette)] for i, pair in enumerate(pairs)}

    # 2) Create empty sub‑plots (independent axes)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        shared_yaxes=False,
        subplot_titles=[f"{title} – Step plot", f"{title} – Cumulative energy"],
        vertical_spacing=0.12,
    )

    all_step_times, all_mid_times = [], []

    # ------------------------------------------------------------------
    # 3) STEP LINES
    # ------------------------------------------------------------------
    for origin, df in data_by_origin:
        for cmdline in df["cmdline"].unique():
            cmd_df = df.filter(pl.col("cmdline") == cmdline)
            if cmd_df.is_empty():
                continue

            time, energy = [], []
            cmd_df = cmd_df.sort("interval_start")
            for row in cmd_df.iter_rows(named=True):
                s = row["interval_start"]
                e = row["interval_end"]
                v = row["total_energy"]
                if s is None or e is None or v is None:
                    continue
                time.extend([s, e])
                energy.extend([v, v])
                all_step_times.extend([s, e])

            label = f"{origin} · {cmdline}"
            if time and energy:
                fig.add_trace(
                    go.Scatter(
                        x=time,
                        y=energy,
                        mode="lines+markers",
                        name=label,
                        line=dict(color=colour_map[(origin, cmdline)], width=LINE_WIDTH, shape="hv"),
                        marker=dict(size=MARKER_SIZE),
                        hovertemplate=(
                            f"<b>{label}</b><br>Time: %{{x|%H:%M:%S}}<br>Energy: %{{y:.2f}} {unit}<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=1,
                )

    # ------------------------------------------------------------------
    # 4) CUMULATIVE LINES
    # ------------------------------------------------------------------
    for origin, df in data_by_origin:
        for cmdline in df["cmdline"].unique():
            cmd_df = df.filter(pl.col("cmdline") == cmdline)
            if cmd_df.is_empty():
                continue

            cmd_df = cmd_df.sort("interval_mid")
            cmd_df = cmd_df.with_columns(pl.col("total_energy").cum_sum().alias("cumulative_energy"))

            label = f"{origin} · {cmdline}"
            if cmd_df.height > 0:
                fig.add_trace(
                    go.Scatter(
                        x=cmd_df["interval_mid"].to_list(),
                        y=cmd_df["cumulative_energy"].to_list(),
                        mode="lines+markers",
                        name=label,
                        showlegend=False,  # hide duplicate legends (already shown above)
                        line=dict(color=colour_map[(origin, cmdline)], width=LINE_WIDTH - 0.5),
                        marker=dict(size=MARKER_SIZE),
                        hovertemplate=(
                            f"<b>{label}</b><br>"
                            f"Time: %{{x|%H:%M:%S}}<br>"
                            f"Cumulative energy: %{{y:.2f}} {unit}<extra></extra>"
                        ),
                    ),
                    row=2,
                    col=1,
                )
                all_mid_times.extend(cmd_df["interval_mid"])

    # ------------------------------------------------------------------
    # 5) Layout & axes adjustments
    # ------------------------------------------------------------------
    fig.update_layout(
        height=800,
        title=dict(text=title, x=0.5, font=dict(size=18, color=PLOT_COLORS["primary"])),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
            font=dict(size=10),
        ),
        margin=dict(l=70, r=180, t=90, b=70),
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="white",
        autosize=True,
    )

    pad_frac = 0.04  # 4%

    # x‑axis ranges & formatting
    if all_step_times:
        filtered_step_times = [t for t in all_step_times if t is not None]
        if filtered_step_times:
            lo, hi = min(filtered_step_times), max(filtered_step_times)
            pad = (hi - lo) * pad_frac
            fig.update_xaxes(
                row=1,
                col=1,
                type="date",
                tickformat="%H:%M:%S",
                range=[lo - pad, hi + pad],
                title_text="Time",
            )
    if all_mid_times:
        filtered_mid_times = [t for t in all_mid_times if t is not None]
        if filtered_mid_times:
            lo, hi = min(filtered_mid_times), max(filtered_mid_times)
            pad = (hi - lo) * pad_frac
            fig.update_xaxes(
                row=2,
                col=1,
                type="date",
                tickformat="%H:%M:%S",
                range=[lo - pad, hi + pad],
                title_text="Time",
            )

    fig.update_yaxes(title_text=f"Energy ({unit})", row=1, col=1)
    fig.update_yaxes(title_text=f"Cumulative energy ({unit})", row=2, col=1)

    return fig


def plot_energy_bar_chart_and_table(
    data: list[tuple[str, pl.DataFrame]],
    title: str = "Energy consumption by process",
    backend: Literal["matplotlib", "plotly"] = "matplotlib",
    unit: Literal["J", "Wh"] = "J",
) -> tuple[go.Figure | MatplotlibFigure | None, pl.DataFrame | None]:
    """
    Plot total energy consumption per process as a stacked bar chart.
    Each stack segment shows the contribution of one “origin” (e.g. CPU, DRAM).

    Parameters
    ----------
    data
        A list of (origin_label, DataFrame) tuples.  Each DataFrame must have a
        'cmdline' column and a 'total_energy_j' column (others are ignored here).
    title
        Plot title.
    backend
        "matplotlib" or "plotly".

    Returns
    -------
    The figure object, or None if every DataFrame is empty.
    """
    # ---- 1. Aggregate -------------------------------------------------------
    frames = []
    origin_order = []  # preserves the input order for stacking
    for origin, df in data:
        if df.is_empty():
            continue
        origin_order.append(origin)
        df = df.with_columns(
            (pl.col("total_energy_j") / 3600 if unit == "Wh" else pl.col("total_energy_j")).alias("total_energy")
        )
        frames.append(df.group_by("cmdline").agg(pl.col("total_energy").sum().alias(origin)))

    if not frames:  # nothing to plot
        logger.info("All supplied DataFrames are empty.")
        return None, None

    # Join all frames properly handling column conflicts
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.join(frame, on="cmdline", how="full", coalesce=True)
    merged = merged.fill_null(0)

    # total energy (all origins) → used for sorting
    merged = merged.with_columns(pl.sum_horizontal(*origin_order).alias("_total_energy")).sort(
        "_total_energy", descending=True
    )

    cmdlines = merged["cmdline"].to_list()
    stacked_values = [merged[o].to_list() for o in origin_order]

    # ---- 2. Plot ------------------------------------------------------------
    table = merged.select(
        pl.col("cmdline"),
        *[pl.col(o).alias(f"{o} [{unit}]") for o in origin_order],
        pl.col("_total_energy").alias(f"total_energy [{unit}]"),
    )

    if backend == "matplotlib":
        return _stacked_bar_matplotlib(cmdlines, stacked_values, origin_order, title, unit), table
    elif backend == "plotly":
        return _stacked_bar_plotly(cmdlines, stacked_values, origin_order, title, unit), table


def _stacked_bar_matplotlib(cmdlines, stacked_values, origins, title, unit: Literal["J", "Wh"] = "J"):
    """
    Plot a stacked bar chart of total energy consumption per process using matplotlib.

    Parameters
    ----------
    cmdlines : list of str
        List of process command lines to display on the x-axis.
    stacked_values : list of list of float
        Each inner list contains energy values (Joules) for each process, corresponding to a specific origin.
    origins : list of str
        List of origin labels, each representing a source/category of energy consumption.
        Must match the length of stacked_values.
    title : str
        Title of the plot.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object containing the stacked bar chart.
    """
    fig = plt.figure(figsize=(13, 7), dpi=100)
    bottom = [0] * len(cmdlines)
    cmap = plt.get_cmap(PLOT_COLORS["matplotlib_cmap"])
    bars = []
    for idx, (vals, origin) in enumerate(zip(stacked_values, origins, strict=True)):
        color = cmap(idx / max(len(origins) - 1, 1))
        bars.append(plt.bar(cmdlines, vals, bottom=bottom, label=origin, color=color, edgecolor="white", linewidth=1))
        bottom = [b + v for b, v in zip(bottom, vals, strict=True)]

    plt.xlabel("Process (cmdline)", fontsize=12)
    plt.ylabel(f"Total energy ({unit})", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xticks(range(len(cmdlines)), cmdlines, rotation=45, ha="right", fontsize=10)
    plt.legend(title="Device", fontsize=10, title_fontsize=11)
    plt.grid(True, axis="y", alpha=PLOT_STYLE["grid_alpha"], linestyle="--", linewidth=0.5)

    # value labels on top of each stack
    for idx, total in enumerate(bottom):
        if total > 0:  # Only show labels for non-zero values
            plt.text(idx, total, f"{total:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    return fig


def _stacked_bar_plotly(
    cmdlines,
    stacked_values,
    origins,
    title,
    unit: Literal["J", "Wh"] = "J",
):
    """
    Create a stacked bar chart using Plotly to visualize energy consumption per process.

    Parameters
    ----------
    cmdlines : list of str
        List of process command lines to be used as x-axis labels.
    stacked_values : list of list of float
        2D list where each sublist contains energy values (in Joules) for a specific origin,
        corresponding to each process in `cmdlines`.
    origins : list of str
        List of origin labels (e.g., measurement sources) for each stack in the bar chart.
    title : str
        Title of the plot.

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        Plotly Figure object representing the stacked bar chart.
    """
    palette = PLOT_COLORS["plotly_palette"]
    fig = go.Figure()
    for idx, (vals, origin) in enumerate(zip(stacked_values, origins, strict=True)):
        color = palette[idx % len(palette)]
        fig.add_trace(
            go.Bar(
                x=cmdlines,
                y=vals,
                name=origin,
                marker=dict(color=color, line=dict(color="white", width=1)),
                text=[f"{v:.1f}" if v > 0 else "" for v in vals],
                textposition="inside",
                textfont=dict(size=9),
            )
        )

    fig.update_layout(
        barmode="stack",
        title=dict(text=title, x=0.5, font=dict(size=18, color=PLOT_COLORS["primary"])),
        xaxis_title="Process (cmdline)",
        yaxis_title=f"Total energy ({unit})",
        height=600,
        margin=dict(l=70, r=70, t=90, b=120),
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
    )

    fig.update_xaxes(title_font=dict(size=12), tickfont=dict(size=10))
    fig.update_yaxes(title_font=dict(size=12), tickfont=dict(size=10))

    return fig


# ============================================================================
# CLI Output & Statistics
# ============================================================================


def calculate_energy_statistics(
    energy_data: list[tuple[str, pl.DataFrame]],
    duration: datetime.timedelta,
    unit: Literal["J", "Wh"],
) -> dict:
    """
    Calculate comprehensive energy statistics from energy data.

    Parameters
    ----------
    energy_data : list[tuple[str, pl.DataFrame]]
        List of (origin, DataFrame) tuples containing energy data.
    duration : datetime.timedelta
        Measurement duration.
    unit : {'J', 'Wh'}
        Energy unit for values.

    Returns
    -------
    dict
        Statistics dict with:
        - device_totals: {device: {total, idle, dynamic}}
        - grand_total: total across all devices
        - average_power_w: average power draw
        - duration_seconds: measurement duration
    """
    stats = {"device_totals": {}, "unit": unit}
    grand_total = 0
    grand_idle = 0
    grand_dynamic = 0

    for origin, df in energy_data:
        if df.is_empty():
            continue

        # Calculate totals for this device
        total = df["total_energy_j"].sum()
        idle = df["idle_energy_j"].sum()
        dynamic = df["active_energy_j"].sum()

        # Convert to selected unit
        if unit == "Wh":
            total /= 3600
            idle /= 3600
            dynamic /= 3600

        stats["device_totals"][origin.upper()] = {
            "total": total,
            "idle": idle,
            "dynamic": dynamic,
        }

        grand_total += total
        grand_idle += idle
        grand_dynamic += dynamic

    stats["grand_total"] = grand_total
    stats["grand_idle"] = grand_idle
    stats["grand_dynamic"] = grand_dynamic
    stats["duration_seconds"] = duration.total_seconds()
    stats["average_power_w"] = (grand_total * (3600 if unit == "Wh" else 1)) / stats["duration_seconds"]

    return stats


def print_energy_summary(
    filtered_energy_data: list[tuple[str, pl.DataFrame]],
    summary_table: pl.DataFrame | None,
    filter_cmdline: str,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    unit: Literal["J", "Wh"],
) -> None:
    """
    Print professional formatted energy consumption summary using Rich.

    Displays:
    1. Header with filter and time range
    2. Device totals table (CPU, DRAM, GPU, Total)
    3. Statistics (duration, average power)
    4. Per-process breakdown table
    """
    console = Console()
    duration = end_time - start_time

    # Calculate statistics
    stats = calculate_energy_statistics(filtered_energy_data, duration, unit)

    # Header
    console.print()
    header_text = Text("Energy Attribution Summary", style="bold cyan", justify="center")
    console.print(Panel(header_text, box=box.DOUBLE, style="cyan"))

    # Analysis parameters
    console.print(f"\n[bold]Filter:[/bold] '{filter_cmdline}'")
    console.print(
        f"[bold]Time Range:[/bold] {start_time.strftime(DATETIME_FORMAT)} to {end_time.strftime(DATETIME_FORMAT)}"
    )
    console.print(f"[bold]Duration:[/bold] {duration} ({stats['duration_seconds']:.1f} seconds)")

    # Device totals table
    console.print("\n[bold cyan]═══ Energy Consumption by Device ═══[/bold cyan]")
    device_table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    device_table.add_column("Device", style="cyan", justify="left")
    device_table.add_column(f"Total ({unit})", style="green", justify="right")
    device_table.add_column(f"Idle ({unit})", style="yellow", justify="right")
    device_table.add_column(f"Dynamic ({unit})", style="red", justify="right")
    device_table.add_column("Percentage", style="blue", justify="right")

    # Add device rows
    for device in ["CPU", "DRAM", "GPU"]:
        if device in stats["device_totals"]:
            totals = stats["device_totals"][device]
            percentage = (totals["total"] / stats["grand_total"] * 100) if stats["grand_total"] > 0 else 0
            device_table.add_row(
                device,
                f"{totals['total']:.2f}",
                f"{totals['idle']:.2f}",
                f"{totals['dynamic']:.2f}",
                f"{percentage:.1f}" + "%",
            )

    # Add total row
    device_table.add_section()
    device_table.add_row(
        "[bold]TOTAL[/bold]",
        "[bold green]" + f"{stats['grand_total']:.2f}" + "[/bold green]",
        "[bold yellow]" + f"{stats['grand_idle']:.2f}" + "[/bold yellow]",
        "[bold red]" + f"{stats['grand_dynamic']:.2f}" + "[/bold red]",
        "[bold]100.0%" + "[/bold]",
    )

    console.print(device_table)

    # Statistics
    console.print("\n[bold cyan]═══ Statistics ═══[/bold cyan]")
    console.print(f"[bold]Average Power:[/bold] {stats['average_power_w']:.2f} W")

    if stats["grand_total"] > 0:
        idle_percentage = stats["grand_idle"] / stats["grand_total"] * 100
        dynamic_percentage = stats["grand_dynamic"] / stats["grand_total"] * 100
        console.print(f"[bold]Idle Energy:[/bold] {idle_percentage:.1f}% of total")
        console.print(f"[bold]Dynamic Energy:[/bold] {dynamic_percentage:.1f}% of total")

    # Per-process breakdown
    if summary_table is not None and not summary_table.is_empty():
        console.print("\n[bold cyan]═══ Per-Process Energy Breakdown ═══[/bold cyan]")
        process_table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")

        # Add columns from summary_table
        for col in summary_table.columns:
            if col == "cmdline":
                process_table.add_column("Process", style="cyan", justify="left", overflow="fold")
            elif "total_energy" in col.lower():
                process_table.add_column(col, style="bold green", justify="right")
            elif col.endswith(f" [{unit}]"):
                process_table.add_column(col, style="white", justify="right")
            else:
                process_table.add_column(col, justify="left")

        # Add rows (limit to top 20 for readability)
        max_rows = 20
        for i, row in enumerate(summary_table.iter_rows()):
            if i >= max_rows:
                remaining = len(summary_table) - max_rows
                console.print(f"[dim]... and {remaining} more processes[/dim]")
                break
            process_table.add_row(*[f"{val:.2f}" if isinstance(val, float) else str(val) for val in row])

        console.print(process_table)

    console.print()


def export_summary_to_file(
    summary_table: pl.DataFrame,
    stats: dict,
    filter_cmdline: str,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    output_dir: Path,
) -> None:
    """
    Export energy summary to CSV and JSON files.

    Parameters
    ----------
    summary_table : polars.DataFrame
        Per-process energy breakdown table.
    stats : dict
        Energy statistics from calculate_energy_statistics.
    filter_cmdline : str
        Process filter used.
    start_time : datetime.datetime
        Analysis start time.
    end_time : datetime.datetime
        Analysis end time.
    output_dir : Path
        Directory to save export files.

    Creates:
    - summary_TIMESTAMP.csv: Per-process breakdown
    - stats_TIMESTAMP.json: Device totals and statistics
    """
    timestamp = datetime.datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    # Export CSV
    csv_path = output_dir / f"energy_summary_{timestamp}.csv"
    summary_table.write_csv(csv_path)
    logger.info(f"Exported summary table to {csv_path}")

    # Export JSON with metadata
    json_path = output_dir / f"energy_stats_{timestamp}.json"
    json_data = {
        "metadata": {
            "filter": filter_cmdline,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": stats["duration_seconds"],
            "unit": stats["unit"],
        },
        "device_totals": stats["device_totals"],
        "summary": {
            "total_energy": stats["grand_total"],
            "idle_energy": stats["grand_idle"],
            "dynamic_energy": stats["grand_dynamic"],
            "average_power_w": stats["average_power_w"],
        },
    }

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    logger.info(f"Exported statistics to {json_path}")


def create_visualizations(
    filtered_dfs: list[tuple[str, pl.DataFrame]],
    all_dfs: list[tuple[str, pl.DataFrame]],
    filter_cmdline: str,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    backend: Literal["matplotlib", "plotly"] = "matplotlib",
    unit: Literal["J", "Wh"] = "J",
    no_plots: bool = False,
) -> pl.DataFrame | None:
    """
    Create visualizations for both filtered and all data and generate an HTML report.

    Args:
        filtered_df: Filtered DataFrame
        all_df: Complete DataFrame
        filter_cmdline: Command line filter used
        start_time: Analysis start time
        end_time: Analysis end time
        backend: Plotting backend to use ('matplotlib' or 'plotly')
        no_plots: If True, skip plot and HTML report generation (summary table still computed)
    """
    if no_plots:
        _, bar_table = plot_energy_bar_chart_and_table(
            filtered_dfs, f"Total Energy by Process (containing '{filter_cmdline}')", backend=backend, unit=unit
        )
        return bar_table

    filtered_figures = []
    all_figures = []

    # Create visualizations for the filtered data
    logger.info(f"Creating plots for '{filter_cmdline}'")

    # Create time series plot
    time_fig = plot_energy_consumption_over_time(
        filtered_dfs, f"Energy Consumption for processes containing '{filter_cmdline}'", backend=backend, unit=unit
    )
    if time_fig and backend == "plotly":
        filtered_figures.append(time_fig)
    logger.info("Time series plot created")

    # Create bar chart
    bar_fig, bar_table = plot_energy_bar_chart_and_table(
        filtered_dfs, f"Total Energy by Process (containing '{filter_cmdline}')", backend=backend, unit=unit
    )
    if bar_fig and backend == "plotly":
        filtered_figures.append(bar_fig)
    logger.info("Bar chart created")

    # Create plots for all data if available
    logger.info(f"Creating plots for all {len(all_dfs)} records")

    # Create time series plot for all data
    all_time_fig = plot_energy_consumption_over_time(
        all_dfs, "Energy Consumption for All Processes", backend=backend, unit=unit
    )
    if all_time_fig and backend == "plotly":
        all_figures.append(all_time_fig)

    # Create bar chart for all data
    all_bar_fig, all_bar_table = plot_energy_bar_chart_and_table(
        all_dfs, "Total Energy by Process (All)", backend=backend, unit=unit
    )
    if all_bar_fig and backend == "plotly":
        all_figures.append(all_bar_fig)
    logger.info("All data plots created")

    # Generate HTML report if using plotly backend
    if backend == "plotly" and (filtered_figures or all_figures):
        generate_html_report(
            filtered_figures=filtered_figures,
            filtered_tables=[bar_table],
            all_figures=all_figures,
            all_tables=[all_bar_table],
            start_time=start_time,
            end_time=end_time,
            filter_cmdline=filter_cmdline,
        )
    elif backend == "matplotlib":
        logger.info("HTML report generation is only available with plotly backend")
    else:
        logger.warning("No figures generated for HTML report")

    return bar_table


def prepare_data(
    db_path: Path,
    filter_cmdline: str,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    unit: Literal["J", "Wh"],
    filter: str = "",
    force: bool = False,
    type: Literal["sqlite", "duckdb"] = "duckdb",
) -> tuple[list[tuple[str, pl.DataFrame]], list[tuple[str, pl.DataFrame]]] | None:
    """
    Prepare energy data from database for visualization.

    Parameters
    ----------
    db_path : Path
        Path to the database file.
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
    type : {'sqlite', 'duckdb'}, optional
        Database type.

    Returns
    -------
    tuple or None
        (filtered_dfs, all_dfs) where each is a list of (origin, DataFrame) tuples.
    """
    # Connect to database
    engine = connect_to_db(db_path, type=type)
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


# ============================================================================
# Main Pipeline & Orchestration
# ============================================================================


# ============================================================================
# CLI Entry Point
# ============================================================================


@click.command()
@click.option(
    "--db-path", type=click.Path(exists=True, path_type=Path), required=True, help="Path to the SQLite database file"
)
@click.option(
    "--filter-cmdline",
    type=str,
    default="",
    help="Command line pattern to filter processes (optional if --filter-pids is set)",
)
@click.option(
    "--filter-pids",
    type=str,
    default="",
    help="Comma-separated list of PIDs to filter by (e.g. '1234,5678')",
)
@click.option(
    "--start-time",
    type=click.DateTime(formats=["%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]),
    required=True,
    help="Start time for analysis (format: YYYY-MM-DD HH:MM:SS or YYYY-MM-DD)",
)
@click.option(
    "--end-time",
    type=click.DateTime(formats=["%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]),
    required=True,
    help="End time for analysis (format: YYYY-MM-DD HH:MM:SS or YYYY-MM-DD)",
)
@click.option(
    "--unit",
    type=click.Choice(["J", "Wh"], case_sensitive=False),
    default="J",
    help="Energy unit to use in plots (default: J)",
)
@click.option(
    "--force",
    type=bool,
    default=False,
    help="Force recreation of views (default: False)",
)
@click.option(
    "--type",
    type=click.Choice(["sqlite", "duckdb"], case_sensitive=False),
    default="duckdb",
    help="Type of database to connect to (default: duckdb)",
)
@click.option(
    "--backend",
    type=click.Choice(["matplotlib", "plotly"], case_sensitive=False),
    default="plotly",
    help="Plotting backend to use (default: plotly)",
)
@click.option(
    "--export-summary",
    is_flag=True,
    default=False,
    help="Export summary to CSV and JSON files",
)
@click.option(
    "--no-plots",
    is_flag=True,
    default=False,
    help="Skip plot and HTML report generation",
)
@click.option(
    "--export-raw-data",
    is_flag=True,
    default=False,
    help="Export raw filtered interval data to CSV files (one per device: cpu, dram, gpu)",
)
def main(
    db_path: Path,
    filter_cmdline: str,
    filter_pids: str,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    unit: Literal["J", "Wh"],
    force: bool = False,
    type: Literal["sqlite", "duckdb"] = "duckdb",
    backend: Literal["matplotlib", "plotly"] = "plotly",
    export_summary: bool = False,
    no_plots: bool = False,
    export_raw_data: bool = False,
) -> None:
    """
    Energy analysis tool for system processes.

    Analyzes CPU, DRAM, and GPU energy consumption from a database,
    filtering by command line patterns, PIDs, and time ranges.

    Examples
    --------
    Basic analysis with required parameters::

        python analysis.py --db-path /path/to/db.db \\
            --start-time "2025-07-16 13:50:00" \\
            --end-time "2025-07-29 14:10:00" \\
            --filter-cmdline "python"

    Filter by PIDs::

        python analysis.py --db-path /path/to/db.db \\
            --filter-pids "1234,5678" \\
            --start-time "2025-07-16" --end-time "2025-07-29"
    """
    # Validate that at least one filter is provided
    if not filter_cmdline and not filter_pids:
        raise click.UsageError("At least one of --filter-cmdline or --filter-pids must be provided.")

    # Parse PIDs and build SQL filter condition
    pid_filter_sql = ""
    if filter_pids:
        try:
            pid_list = [int(p.strip()) for p in filter_pids.split(",") if p.strip()]
        except ValueError as e:
            raise click.BadParameter(
                "--filter-pids must be a comma-separated list of integers", param_hint="--filter-pids"
            ) from e
        if pid_list:
            pid_csv = ", ".join(str(p) for p in pid_list)
            pid_filter_sql = f"AND le.native_pid IN ({pid_csv})"

    # Build display label for reports/plots
    filter_label_parts = []
    if filter_cmdline:
        filter_label_parts.append(f"cmdline='{filter_cmdline}'")
    if filter_pids:
        filter_label_parts.append(f"pids=[{filter_pids}]")
    filter_label = ", ".join(filter_label_parts)

    prepared_data = prepare_data(
        db_path=db_path,
        filter_cmdline=filter_cmdline,
        start_time=start_time,
        end_time=end_time,
        unit=unit,
        filter=pid_filter_sql,
        force=force,
        type=type,
    )
    if prepared_data is None:
        logger.error("No filtered data available. Exiting.")
        return
    filtered_dfs, all_dfs = prepared_data

    # Export raw filtered data if requested
    if export_raw_data:
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        for device, df in filtered_dfs:
            csv_path = output_dir / f"raw_{device}_{timestamp}.csv"
            df.write_csv(csv_path)
            logger.info(f"Raw {device.upper()} data exported to {csv_path}")

    # Create visualizations
    summary_table = create_visualizations(
        filtered_dfs=filtered_dfs,
        all_dfs=all_dfs,
        filter_cmdline=filter_label,
        start_time=start_time,
        end_time=end_time,
        backend=backend,
        unit=unit,
        no_plots=no_plots,
    )

    # Print professional summary
    print_energy_summary(
        filtered_energy_data=filtered_dfs,
        summary_table=summary_table,
        filter_cmdline=filter_label,
        start_time=start_time,
        end_time=end_time,
        unit=unit,
    )

    # Export if requested
    if export_summary and summary_table is not None:
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        stats = calculate_energy_statistics(filtered_dfs, end_time - start_time, unit)
        export_summary_to_file(
            summary_table=summary_table,
            stats=stats,
            filter_cmdline=filter_label,
            start_time=start_time,
            end_time=end_time,
            output_dir=output_dir,
        )

    logger.info("Analysis completed successfully")


if __name__ == "__main__":
    main()
