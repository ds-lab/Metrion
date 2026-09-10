# %%
import json
from datetime import datetime
from pathlib import Path

import polars as pl

results_schema_df = {
    "run_id": pl.Int64,
    "date": pl.Datetime,
    "duration_s": pl.Float64,
    "pkg_energy_idle_j": pl.Float64,
    "pkg_energy_active_j": pl.Float64,
    "pkg_energy_total_j": pl.Float64,
    "dram_energy_idle_j": pl.Float64,
    "dram_energy_active_j": pl.Float64,
    "dram_energy_total_j": pl.Float64,
    "gpu_energy_idle_j": pl.Float64,
    "gpu_energy_active_j": pl.Float64,
    "gpu_energy_total_j": pl.Float64,
}

ACTIVE_ENERGY_COLUMNS: tuple[str, ...] = (
    "pkg_energy_active_j",
    "dram_energy_active_j",
    "gpu_energy_active_j",
)

ACTIVE_ENERGY_LABELS: dict[str, str] = {
    "pkg_energy_active_j": "CPU Package Energy (J)",
    "dram_energy_active_j": "DRAM Energy (J)",
    "gpu_energy_active_j": "GPU Energy (J)",
}


def parse_carbontracker_logs(data_dir: Path) -> pl.DataFrame:
    from carbontracker import parser

    output_df = pl.DataFrame(schema=results_schema_df)
    id = 0

    logs = parser.parse_all_logs(data_dir)

    for log in logs:
        # get filename from the log (fall back to standard_filename)
        fname = log["output_filename"]

        date_part = fname.split("_")[-3]
        parsed_date = None
        if date_part:
            try:
                parsed_date = datetime.strptime(date_part, "%Y-%m-%dT%H%M%SZ")
            except Exception:
                parsed_date = None

        df = pl.DataFrame(
            [
                {
                    "run_id": id,
                    "date": parsed_date,
                    "duration_s": log["actual"]["duration (s)"],
                    "pkg_energy_total_j": log["components"]["cpu"].get("avg_energy_usages (J)"),
                    "gpu_energy_total_j": 0.0,  # log["components"]["gpu"].get("avg_energy_usages (J)"),
                }
            ]
        )
        df = df.select(
            pl.col("run_id"),
            pl.col("date"),
            pl.col("duration_s"),
            pl.lit(0.0).alias("pkg_energy_idle_j"),
            pl.col("pkg_energy_total_j").alias("pkg_energy_active_j"),
            pl.col("pkg_energy_total_j"),
            pl.lit(0.0).alias("dram_energy_idle_j"),
            pl.lit(0.0).alias("dram_energy_active_j"),
            pl.lit(0.0).alias("dram_energy_total_j"),
            pl.lit(0.0).alias("gpu_energy_idle_j"),
            pl.lit(0.0).alias("gpu_energy_active_j"),
            pl.col("gpu_energy_total_j"),
        )
        id += 1
        output_df = pl.concat([output_df, df], how="vertical")

    return output_df


def parse_codecarbon_logs(data_dir: Path) -> pl.DataFrame:
    output_df = pl.DataFrame(schema=results_schema_df)
    id = 0

    for csv_file in data_dir.glob("emission*.csv"):
        df = pl.read_csv(csv_file)
        df = df.select(
            pl.lit(id).alias("run_id").cast(pl.Int64),
            df["timestamp"].str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S").alias("date"),
            df["duration"].alias("duration_s"),
            pl.lit(0.0).alias("pkg_energy_idle_j"),
            (df["cpu_energy"] * 3.6e6).alias("pkg_energy_active_j"),  # kwh to joules
            (df["cpu_energy"] * 3.6e6).alias("pkg_energy_total_j"),
            pl.lit(0.0).alias("dram_energy_idle_j"),
            (df["ram_energy"] * 3.6e6).alias("dram_energy_active_j"),  # kwh to joules
            (df["ram_energy"] * 3.6e6).alias("dram_energy_total_j"),
            pl.lit(0.0).alias("gpu_energy_idle_j"),
            (df["gpu_energy"] * 3.6e6).alias("gpu_energy_active_j"),  # kwh to joules
            (df["gpu_energy"] * 3.6e6).alias("gpu_energy_total_j"),
        )

        output_df = pl.concat([output_df, df], how="vertical")
        id += 1

    return output_df


def parse_energat_logs(data_dir: Path) -> pl.DataFrame:
    output_df = pl.DataFrame(schema=results_schema_df)
    id = 0

    for csv_file in data_dir.glob("energat_traces_*.csv"):
        filename = csv_file.stem  # e.g., "energat_traces_stress-ng_20251003_110839"
        parts = filename.split("_")
        date_str = parts[-2] + "_" + parts[-1]
        parsed_date = datetime.strptime(date_str, "%Y%m%d_%H%M%S")

        df = pl.read_csv(csv_file)

        # Sum the required columns
        duration_total = df["duration_sec"].sum()
        pkg_energy_total = df["ascribed_pkg_joules"].sum()
        dram_energy_total = df["ascribed_dram_joules"].sum()

        # Create a single row dataframe with the aggregated values
        result_df = pl.DataFrame(
            [
                {
                    "run_id": id,
                    "date": parsed_date,
                    "duration_s": duration_total,
                    "pkg_energy_idle_j": 0.0,
                    "pkg_energy_active_j": pkg_energy_total,
                    "pkg_energy_total_j": pkg_energy_total,
                    "dram_energy_idle_j": 0.0,
                    "dram_energy_active_j": dram_energy_total,
                    "dram_energy_total_j": dram_energy_total,
                    "gpu_energy_idle_j": 0.0,
                    "gpu_energy_active_j": 0.0,
                    "gpu_energy_total_j": 0.0,
                }
            ]
        )

        output_df = pl.concat([output_df, result_df], how="vertical")
        id += 1

    return output_df


def parse_scaphandre_json(path, process_pattern, step_s=0.1):
    """
    Parse Scaphandre JSON exporter output and compute energy (J) and average power (W)
    for a given process matched by substring in 'cmdline'.
    """
    try:
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Unable to parse Scaphandre report: {exc}") from exc

    energy_j = 0.0
    samples = 0
    timestamps = []

    for data in records:
        ts = data.get("host", {}).get("timestamp")
        for consumer in data.get("consumers", []):
            cmdline = str(consumer.get("cmdline", ""))
            if (process_pattern in cmdline) and ("scaphandre" not in cmdline):
                microwatts = consumer.get("consumption", 0.0)
                if microwatts > 0:
                    watts = microwatts / 1e6
                    energy_j += watts * step_s
                    samples += 1
                    timestamps = timestamps + [ts]

    duration_s = max(timestamps) - min(timestamps) if timestamps else 0

    return {"pkg_energy_active_j": energy_j, "duration_s": duration_s}


def parse_scaphandre_logs(data_dir: Path, process_pattern: str, step_s: float = 0.1) -> pl.DataFrame:
    """
    Parse all Scaphandre JSON reports in data_dir for the given process_pattern.
    Uses parse_scaphandre_json to extract energy (J) and duration (s) and
    constructs rows that match results_schema_df.
    """
    output_df = pl.DataFrame(schema=results_schema_df)
    id = 0

    for json_file in data_dir.glob("*.json"):
        parsed = parse_scaphandre_json(json_file, process_pattern, step_s=step_s)
        filename = json_file.stem  # e.g., "energat_traces_stress-ng_20251003_110839"
        parts = filename.split("_")
        date_str = parts[-2] + "_" + parts[-1]
        parsed_date = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
        # if no samples matched, skip the file
        if parsed.get("duration_s", 0.0) == 0.0 and parsed.get("pkg_energy_active_j", 0.0) == 0.0:
            id += 1
            continue

        row = pl.DataFrame(
            [
                {
                    "run_id": id,
                    "date": parsed_date,
                    "duration_s": parsed.get("duration_s", 0.0),
                    "pkg_energy_idle_j": 0.0,
                    "pkg_energy_active_j": parsed.get("pkg_energy_active_j", 0.0),
                    "pkg_energy_total_j": parsed.get("pkg_energy_active_j", 0.0),
                    "dram_energy_idle_j": 0.0,
                    "dram_energy_active_j": 0.0,
                    "dram_energy_total_j": 0.0,
                    "gpu_energy_idle_j": 0.0,
                    "gpu_energy_active_j": 0.0,
                    "gpu_energy_total_j": 0.0,
                }
            ]
        )

        output_df = pl.concat([output_df, row], how="vertical")
        id += 1

    return output_df


def parse_rapl_logs_idle(data_dir: Path) -> pl.DataFrame:
    output_df = pl.DataFrame(schema=results_schema_df)
    id = 0

    idle_path = data_dir / "measurements_idle.csv"
    idle_df = pl.read_csv(idle_path)

    n_ids = idle_df.height
    ids = list(range(0, n_ids))
    # parse datetimes and strip timezone info by casting through Int64
    start_date = idle_df["start"].str.to_datetime("%Y-%m-%dT%H:%M:%S%z")
    stop_date = idle_df["stop"].str.to_datetime("%Y-%m-%dT%H:%M:%S%z")
    date = start_date.cast(pl.Datetime)
    # compute duration in seconds as float
    duration_s = (stop_date - start_date).cast(pl.Float64) / 1e6

    pkg_energy_total_j = (idle_df["pkg_diff_uj"] / 1e6).cast(pl.Float64)  # microjoules to joules
    dram_energy_total_j = (idle_df["dram_diff_uj"] / 1e6).cast(pl.Float64)  # microjoules to joules

    # Create a column-wise dataframe (one row per idle measurement) so types align with schema
    result_df = pl.DataFrame(
        {
            "run_id": ids,
            "date": date,
            "duration_s": duration_s,
            "pkg_energy_idle_j": [0.0] * n_ids,
            "pkg_energy_active_j": pkg_energy_total_j,
            "pkg_energy_total_j": pkg_energy_total_j,
            "dram_energy_idle_j": [0.0] * n_ids,
            "dram_energy_active_j": dram_energy_total_j,
            "dram_energy_total_j": dram_energy_total_j,
            "gpu_energy_idle_j": [0.0] * n_ids,
            "gpu_energy_active_j": [0.0] * n_ids,
            "gpu_energy_total_j": [0.0] * n_ids,
        }
    )

    output_df = pl.concat([output_df, result_df], how="vertical")
    id += 1

    return output_df


def _compute_power_idle_measurement(idle_df: pl.DataFrame) -> pl.DataFrame:
    idle_power_df = idle_df.select(
        (pl.col("pkg_energy_active_j") / pl.col("duration_s")).alias("power_pkg_idle_w"),
        (pl.col("dram_energy_active_j") / pl.col("duration_s")).alias("power_dram_idle_w"),
        (pl.col("gpu_energy_active_j") / pl.col("duration_s")).alias("power_gpu_idle_w"),
    )

    idle_mean_power_df = idle_power_df.select(
        pl.col("power_pkg_idle_w").mean().alias("power_pkg_idle_mean_w"),
        pl.col("power_dram_idle_w").mean().alias("power_dram_idle_mean_w"),
        pl.col("power_gpu_idle_w").mean().alias("power_gpu_idle_mean_w"),
    )

    return idle_mean_power_df


def parse_rapl_logs_workload(data_dir: Path, workload_type: str) -> pl.DataFrame:
    output_df = pl.DataFrame(schema=results_schema_df)
    id = 0

    # parse idle measurements first to compute idle power
    idle_df = parse_rapl_logs_idle(data_dir)
    idle_mean_power_df = _compute_power_idle_measurement(idle_df)

    workload_path = data_dir / "measurements_workload.csv"
    workload_df = pl.read_csv(workload_path)
    workload_df = workload_df.filter(pl.col("workload_name") == f"rapl-{workload_type}")

    n_ids = workload_df.height
    ids = list(range(0, n_ids))
    # parse datetimes and strip timezone info by casting through Int64
    start_date = workload_df["start"].str.to_datetime("%Y-%m-%dT%H:%M:%S%z")
    stop_date = workload_df["stop"].str.to_datetime("%Y-%m-%dT%H:%M:%S%z")
    date = start_date.cast(pl.Datetime)
    # compute duration in seconds as float
    duration_s = (stop_date - start_date).cast(pl.Float64) / 1e6

    pkg_energy_idle_j = (idle_mean_power_df["power_pkg_idle_mean_w"] * duration_s).cast(pl.Float64)
    pkg_energy_active_j = ((workload_df["pkg_diff_uj"] / 1e6) - pkg_energy_idle_j).cast(pl.Float64)
    pkg_energy_total_j = pkg_energy_idle_j + pkg_energy_active_j
    dram_energy_idle_j = (idle_mean_power_df["power_dram_idle_mean_w"] * duration_s).cast(pl.Float64)
    dram_energy_active_j = ((workload_df["dram_diff_uj"] / 1e6) - dram_energy_idle_j).cast(pl.Float64)
    dram_energy_total_j = dram_energy_idle_j + dram_energy_active_j

    # Create a column-wise dataframe (one row per idle measurement) so types align with schema
    result_df = pl.DataFrame(
        {
            "run_id": ids,
            "date": date,
            "duration_s": duration_s,
            "pkg_energy_idle_j": pkg_energy_idle_j,
            "pkg_energy_active_j": pkg_energy_active_j,
            "pkg_energy_total_j": pkg_energy_total_j,
            "dram_energy_idle_j": dram_energy_idle_j,
            "dram_energy_active_j": dram_energy_active_j,
            "dram_energy_total_j": dram_energy_total_j,
            "gpu_energy_idle_j": [0.0] * n_ids,
            "gpu_energy_active_j": [0.0] * n_ids,
            "gpu_energy_total_j": [0.0] * n_ids,
        }
    )

    output_df = pl.concat([output_df, result_df], how="vertical")
    id += 1

    return output_df


def parse_metrion_logs(
    data_dir: Path,
    workload: str,
    start_time: datetime,
    end_time: datetime,
) -> pl.DataFrame:
    from metrion.energy_attribution import util

    cache_folder = data_dir.parent.parent / "database_output"
    databases = list(data_dir.glob("*.db"))

    energy_data = util.load_energy_data_per_database(databases, cache_folder, workload, start_time, end_time)

    return util.combine_energy_data(energy_data, results_schema_df)


def parse_runtime_logs(runtime_log_path: Path) -> pl.DataFrame:
    runtime_log_schema = {
        "name": pl.Utf8,
        "start": pl.Datetime,
        "stop": pl.Datetime,
        "execution_time": pl.Float64,
    }
    runtime_log_df = pl.read_csv(runtime_log_path, schema=runtime_log_schema)
    return runtime_log_df.select(
        pl.col("name"),
        pl.col("start"),
        pl.col("stop"),
        pl.col("execution_time").alias("duration_s"),
    )
