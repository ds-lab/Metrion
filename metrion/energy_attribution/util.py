from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import polars as pl

from metrion.energy_attribution import energy_attribution_cli

ENERGY_DATA_SCHEMA = pl.Schema(
    {
        "interval_start": pl.Datetime("us"),
        "interval_end": pl.Datetime("us"),
        "cmdline": pl.String,
        "comm": pl.String,
        "cgroup": pl.String,
        "cgroup_id": pl.Int64,
        "total_energy_j": pl.Float32,
        "active_energy_j": pl.Float32,
        "idle_energy_j": pl.Float32,
    }
)

EnergyData = list[tuple[str, pl.DataFrame]]


def _default_cache_folder() -> Path:
    """Return the default cache folder used by analysis utilities.

    The folder is resolved relative to this module and mirrors the
    `benchmarks/evaluation` output location used for cached CSV files.

    Returns
    -------
    Path
        Path to the default cache folder.
    """

    current_dir = Path(__file__).resolve().parents[1]
    return current_dir / "output" / "database_output"


def load_cached_energy_data(
    db_name: str,
    types: Sequence[str] | None = None,
    cache_folder: Path | None = None,
) -> EnergyData | None:
    """Load cached energy data from previous evaluations, if available.

    Parameters
    ----------
    db_name : str
        Name of the database for which cached CSV files should be loaded.
    types : Sequence[str], optional
        Iterable of measurement types (for example, ``["cpu", "dram"]``)
        to load. When omitted, the default CPU and DRAM datasets are
        requested.
    cache_folder : Path, optional
        Folder to look for cached CSV files. If omitted, a project-local
        default is used.

    Returns
    -------
    EnergyData | None
        A list of ``(measurement_type, DataFrame)`` tuples when all requested
        cache files exist; ``None`` otherwise.
    """

    selected_types = list(types) if types is not None else ["cpu", "dram"]

    cache_folder = cache_folder or _default_cache_folder()

    loaded: EnergyData = []
    for measurement_type in selected_types:
        path = cache_folder / f"energy_data_{db_name}_{measurement_type}.csv"
        if not path.exists():
            return None
        loaded.append((measurement_type, pl.read_csv(path, schema=ENERGY_DATA_SCHEMA)))
    return loaded


def save_energy_data(energy_data: EnergyData, db_name: str, cache_folder: Path | None = None) -> None:
    """Persist prepared energy data for a database to CSV files.

    Parameters
    ----------
    energy_data : list[tuple[str, pl.DataFrame]]
        Sequence of ``(measurement_type, DataFrame)`` pairs to persist.
    db_name : str
        Name of the database used to derive the data; it is embedded in the
        generated CSV filenames.
    cache_folder : Path, optional
        Destination folder for the CSV files. If omitted, a project-local
        default is used.

    Returns
    -------
    None
        The function writes CSV files as a side effect and returns ``None``.
    """

    cache_folder = cache_folder or _default_cache_folder()
    cache_folder.mkdir(parents=True, exist_ok=True)

    for measurement_type, df in energy_data:
        energy_data_path = cache_folder / f"energy_data_{db_name}_{measurement_type}.csv"
        df.write_csv(energy_data_path)


def load_energy_data_per_database(
    databases: Sequence[Path],
    cache_folder: Path | None,
    workload: str,
    time_start: datetime,
    time_end: datetime,
    unit: Literal["J", "Wh"] = "J",
    filter_condition: str | None = None,
    force: bool = False,
    dialect: Literal["sqlite", "duckdb"] = "duckdb",
) -> list[EnergyData]:
    """Collect energy data for each monitoring database.

    Parameters
    ----------
    databases : Sequence[Path]
        Iterable of database file paths to process.
    cache_folder : Path, optional
        Optional cache folder to read/write prepared CSV files. If omitted,
        a project-local default is used.
    workload : str
        Workload identifier used to filter processes in the database query.
    time_start : datetime
        Inclusive lower bound of the time range used for data extraction.
    time_end : datetime
        Inclusive upper bound of the time range used for data extraction.
    unit : {"J", "Wh"}, optional
        Energy unit requested from the analysis pipeline. Defaults to ``"J"``.
    filter_condition : str, optional
        Additional SQL filter clause appended to the prepared statement. When
        omitted, the workload identifier is matched against command-line and
        command fields.
    force : bool, optional
        When ``True``, bypass cached CSV data and always recompute the
        aggregates. Default is ``False``.

    Returns
    -------
    list[EnergyData]
        A list containing per-database energy datasets.
    """

    if filter_condition is None:
        filter_condition = f"AND (le.cmdline LIKE '%{workload}%' OR le.executable_name LIKE '%{workload}%')"

    cache_folder = cache_folder or _default_cache_folder()

    energy_data_per_database: list[EnergyData] = []
    for db_path in databases:
        cached: EnergyData | None = None
        db_name = db_path.stem
        if not force:
            cached = load_cached_energy_data(db_name, cache_folder=cache_folder)
        if cached is not None:
            energy_data_per_database.append(cached)
        else:
            data = energy_attribution_cli.prepare_data(
                db_path, workload, time_start, time_end, unit=unit, filter=filter_condition, force=force, type=dialect
            )
            if data:
                filtered_dfs, _ = data
                save_energy_data(filtered_dfs, db_name, cache_folder=cache_folder)
                energy_data_per_database.append(filtered_dfs)

    return energy_data_per_database


def combine_energy_data(energy_data_dbs: list[EnergyData], schema: dict[str, Any]) -> pl.DataFrame:
    if not energy_data_dbs:
        return pl.DataFrame(schema=schema)

    result_rows: list[dict[str, Any]] = []

    for run_id, energy_data_db in enumerate(energy_data_dbs):
        row: dict[str, Any] = {}

        for column, dtype in schema.items():
            if dtype in (pl.Float32, pl.Float64):
                row[column] = 0.0
            elif dtype in (pl.Int32, pl.Int64, pl.UInt32, pl.UInt64):
                row[column] = 0
            else:
                row[column] = None

        row["run_id"] = run_id

        for component_type, df in energy_data_db:
            if df.is_empty():
                continue

            component_key = "pkg" if component_type == "cpu" else component_type

            start_df = df.select(pl.col("interval_start").min())
            end_df = df.select(pl.col("interval_end").max())

            start_ts = start_df.item() if not start_df.is_empty() else None
            end_ts = end_df.item() if not end_df.is_empty() else None

            if start_ts is not None and row["date"] is None:
                row["date"] = start_ts

            if start_ts is not None and end_ts is not None:
                duration = (end_ts - start_ts).total_seconds()
                row["duration_s"] = max(row.get("duration_s", 0.0), duration)

            row[f"{component_key}_energy_idle_j"] = df.select(pl.col("idle_energy_j").sum()).item()
            row[f"{component_key}_energy_active_j"] = df.select(pl.col("active_energy_j").sum()).item()
            row[f"{component_key}_energy_total_j"] = df.select(pl.col("total_energy_j").sum()).item()

        result_rows.append(row)

    return pl.DataFrame(result_rows, schema=schema)
