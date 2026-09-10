import datetime as dt
from pathlib import Path

import polars as pl
import polars.testing as plt
import pytest

from metrion.energy_attribution.energy_attribution import (
    CPUEnergyAttributionModel,
    DRAMEnergyAttributionModel,
    GPUEnergyAttributionModel,
)
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

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "data"


def _read_parquet(name: str) -> pl.DataFrame:
    return pl.read_parquet(FIXTURE_DIR / name)


@pytest.fixture(scope="module")
def pkg_input_data() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    measurement_df = _read_parquet("measurement.parquet")
    cpu_df = _read_parquet("cpu.parquet")
    core_df = _read_parquet("core.parquet")
    return measurement_df, cpu_df, core_df


@pytest.fixture(scope="module")
def pkg_expected_frames() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    expected_df = _read_parquet("cpu_energy_results.parquet")

    expected_idle_df = expected_df.select(
        pl.col("logical_entity_id"),
        pl.col("idle_energy_j").alias("value"),
        pl.lit(CPU_PKG_PID_IDLE_ENERGY.name).alias("metric_id"),
        pl.col("package_id").alias("physical_entity_id"),
        pl.col("interval_start").alias("start_ts"),
        pl.col("interval_end").alias("end_ts"),
    )

    expected_active_df = expected_df.select(
        pl.col("logical_entity_id"),
        pl.col("active_energy_j").alias("value"),
        pl.lit(CPU_PKG_PID_DYNAMIC_ENERGY.name).alias("metric_id"),
        pl.col("package_id").alias("physical_entity_id"),
        pl.col("interval_start").alias("start_ts"),
        pl.col("interval_end").alias("end_ts"),
    )

    expected_total_df = expected_df.select(
        pl.col("logical_entity_id"),
        (pl.col("idle_energy_j") + pl.col("active_energy_j")).alias("value"),
        pl.lit(CPU_PKG_PID_TOTAL_ENERGY.name).alias("metric_id"),
        pl.col("package_id").alias("physical_entity_id"),
        pl.col("interval_start").alias("start_ts"),
        pl.col("interval_end").alias("end_ts"),
    )

    return expected_idle_df, expected_active_df, expected_total_df


@pytest.fixture(scope="module")
def dram_input_data() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    measurement_df = _read_parquet("measurement.parquet")
    core_df = _read_parquet("core.parquet")
    dram_df = _read_parquet("dram.parquet")
    return measurement_df, core_df, dram_df


@pytest.fixture(scope="module")
def dram_expected_frames() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    expected_total_df = _read_parquet("dram_energy_results.parquet")
    expected_active_df = _read_parquet("dram_energy_active_results.parquet")
    expected_idle_df = _read_parquet("dram_energy_idle_results.parquet")

    expected_idle_df = expected_idle_df.select(
        pl.col("logical_entity_id"),
        pl.col("idle_energy_j").alias("value"),
        pl.lit(DRAM_SOCKET_PID_IDLE_ENERGY.name).alias("metric_id"),
        pl.col("dram_id").alias("physical_entity_id"),
        pl.col("interval_start").alias("start_ts"),
        pl.col("interval_end").alias("end_ts"),
    )

    expected_active_df = expected_active_df.select(
        pl.col("logical_entity_id"),
        pl.col("active_energy_j").alias("value"),
        pl.lit(DRAM_SOCKET_PID_DYNAMIC_ENERGY.name).alias("metric_id"),
        pl.col("dram_id").alias("physical_entity_id"),
        pl.col("interval_start").alias("start_ts"),
        pl.col("interval_end").alias("end_ts"),
    )

    expected_total_df = expected_total_df.select(
        pl.col("logical_entity_id"),
        (pl.col("total_energy_j")).alias("value"),
        pl.lit(DRAM_SOCKET_PID_TOTAL_ENERGY.name).alias("metric_id"),
        pl.col("dram_id").alias("physical_entity_id"),
        pl.col("interval_start").alias("start_ts"),
        pl.col("interval_end").alias("end_ts"),
    )
    return expected_idle_df, expected_active_df, expected_total_df


def _sort_and_assert(actual: pl.DataFrame, expected: pl.DataFrame) -> None:
    columns_order = ["logical_entity_id", "value", "metric_id", "physical_entity_id", "start_ts", "end_ts"]
    sort_keys = ["start_ts", "logical_entity_id", "physical_entity_id"]
    actual_sorted = actual.sort(sort_keys).select(columns_order)
    expected_sorted = expected.sort(sort_keys).select(columns_order)
    print(actual_sorted)
    print(expected_sorted)
    plt.assert_frame_equal(actual_sorted, expected_sorted, check_exact=False, rel_tol=1e-3, abs_tol=1e-3)


@pytest.fixture(scope="module")
def pkg_model_results(pkg_input_data) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Run CPUEnergyAttributionModel once per module and return (total, idle, active) frames."""
    measurement_df, cpu_df, core_df = pkg_input_data
    model = CPUEnergyAttributionModel()
    return model._attribute_pkg_energy(measurement_df, cpu_df, core_df)


@pytest.fixture(scope="module")
def dram_model_results(dram_input_data) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Run DRAMEnergyAttributionModel once per module and return (total, idle, active) frames."""
    measurement_df, core_df, dram_df = dram_input_data
    model = DRAMEnergyAttributionModel()
    return model._attribute_dram_energy(measurement_df, core_df, dram_df)


def test_pkg_idle_energy_matches(pkg_expected_frames, pkg_model_results) -> None:
    expected_idle_df, _, _ = pkg_expected_frames

    # use precomputed model results fixture to avoid rerunning model
    _, res_idle_energy_df, _ = pkg_model_results
    _sort_and_assert(res_idle_energy_df, expected_idle_df)


def test_pkg_dynamic_energy_matches(pkg_expected_frames, pkg_model_results) -> None:
    _, expected_active_df, _ = pkg_expected_frames

    _, _, res_dynamic_energy_df = pkg_model_results

    _sort_and_assert(res_dynamic_energy_df, expected_active_df)


def test_pkg_total_energy_matches(pkg_expected_frames, pkg_model_results) -> None:
    _, _, expected_total_df = pkg_expected_frames

    res_total_energy_df, _, _ = pkg_model_results

    _sort_and_assert(res_total_energy_df, expected_total_df)


def test_dram_idle_energy_matches(dram_expected_frames, dram_model_results) -> None:
    expected_idle_df, _, _ = dram_expected_frames

    # use precomputed model results fixture to avoid rerunning model
    _, res_idle_energy_df, _ = dram_model_results
    _sort_and_assert(res_idle_energy_df, expected_idle_df)


def test_dram_dynamic_energy_matches(dram_expected_frames, dram_model_results) -> None:
    _, expected_active_df, _ = dram_expected_frames

    _, _, res_dynamic_energy_df = dram_model_results

    _sort_and_assert(res_dynamic_energy_df, expected_active_df)


def test_dram_total_energy_matches(dram_expected_frames, dram_model_results) -> None:
    _, _, expected_total_df = dram_expected_frames

    res_total_energy_df, _, _ = dram_model_results

    _sort_and_assert(res_total_energy_df, expected_total_df)


@pytest.fixture(scope="module")
def gpu_input_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    measurement_df = _read_parquet("measurement.parquet")
    gpu_df = _read_parquet("gpu.parquet")
    return measurement_df, gpu_df


@pytest.fixture(scope="module")
def gpu_expected_frames() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    # The SQL leaves two 0 J rows without a thread per idle GPU and interval (NULL never matches
    # in its full outer join); the model writes none where no process ran.
    expected_df = _read_parquet("gpu_energy_results.parquet").filter(pl.col("logical_entity_id").is_not_null())

    expected_idle_df = expected_df.select(
        pl.col("logical_entity_id"),
        pl.col("idle_energy_j").alias("value"),
        pl.lit(GPU_DEVICE_PID_IDLE_ENERGY.name).alias("metric_id"),
        pl.col("gpu_id").alias("physical_entity_id"),
        pl.col("interval_start").alias("start_ts"),
        pl.col("interval_end").alias("end_ts"),
    )

    expected_active_df = expected_df.select(
        pl.col("logical_entity_id"),
        pl.col("active_energy_j").alias("value"),
        pl.lit(GPU_DEVICE_PID_DYNAMIC_ENERGY.name).alias("metric_id"),
        pl.col("gpu_id").alias("physical_entity_id"),
        pl.col("interval_start").alias("start_ts"),
        pl.col("interval_end").alias("end_ts"),
    )

    expected_total_df = expected_df.select(
        pl.col("logical_entity_id"),
        pl.col("total_energy_j").alias("value"),
        pl.lit(GPU_DEVICE_PID_TOTAL_ENERGY.name).alias("metric_id"),
        pl.col("gpu_id").alias("physical_entity_id"),
        pl.col("interval_start").alias("start_ts"),
        pl.col("interval_end").alias("end_ts"),
    )

    return expected_idle_df, expected_active_df, expected_total_df


@pytest.fixture(scope="module")
def gpu_model_results(gpu_input_data) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Run GPUEnergyAttributionModel once per module and return (total, idle, dynamic) frames."""
    measurement_df, gpu_df = gpu_input_data
    return GPUEnergyAttributionModel().attribute_gpu_energy(measurement_df, gpu_df)


def test_gpu_idle_energy_matches(gpu_expected_frames, gpu_model_results) -> None:
    expected_idle_df, _, _ = gpu_expected_frames

    _, res_idle_energy_df, _ = gpu_model_results

    _sort_and_assert(res_idle_energy_df, expected_idle_df)


def test_gpu_dynamic_energy_matches(gpu_expected_frames, gpu_model_results) -> None:
    _, expected_active_df, _ = gpu_expected_frames

    _, _, res_dynamic_energy_df = gpu_model_results

    _sort_and_assert(res_dynamic_energy_df, expected_active_df)


def test_gpu_total_energy_matches(gpu_expected_frames, gpu_model_results) -> None:
    _, _, expected_total_df = gpu_expected_frames

    res_total_energy_df, _, _ = gpu_model_results

    _sort_and_assert(res_total_energy_df, expected_total_df)


# The socket counter rises by this much over one second while idle_power_package claims far more,
# so without the cap the remainder handed to the threads would be negative.
MEASURED_ENERGY_J = 50.0
CLAIMED_IDLE_POWER_W = 200.0


START = dt.datetime(2026, 1, 1)
END = START + dt.timedelta(seconds=1)


def _measurement_frame(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "metric_id": pl.String,
            "physical_entity_id": pl.String,
            "logical_entity_id": pl.String,
            "value": pl.Float64,
            "start_ts": pl.Datetime,
            "end_ts": pl.Datetime,
        },
        orient="row",
    )


def test_cpu_idle_power_above_the_measured_energy_leaves_no_negative_share() -> None:
    measurements = _measurement_frame(
        [
            ("CPU_SOCKET_PKG_ENERGY", "cpu-A", None, 100.0, START, START),
            ("CPU_SOCKET_PKG_ENERGY", "cpu-A", None, 100.0 + MEASURED_ENERGY_J, END, END),
            ("CPU_CORE_PID_WORK", "core-A", "p1", 3000.0, START, END),
            ("CPU_CORE_PID_WORK", "core-A", "p2", 1000.0, START, END),
            ("CPU_CORE_PID_TIME", "core-A", "p1", 750.0, START, END),
            ("CPU_CORE_PID_TIME", "core-A", "p2", 250.0, START, END),
        ]
    )
    cpu_df = pl.DataFrame({"id": ["cpu-A"], "idle_power_package": [CLAIMED_IDLE_POWER_W]})
    core_df = pl.DataFrame({"id": ["core-A"], "parent_id": ["cpu-A"]})

    total, _, dynamic = CPUEnergyAttributionModel().attribute_energy(
        measurement_df=measurements, cpu_df=cpu_df, core_df=core_df
    )

    assert (dynamic["value"] >= 0.0).all()
    assert total["value"].sum() == pytest.approx(MEASURED_ENERGY_J)


def test_dram_idle_power_above_the_measured_energy_leaves_no_negative_share() -> None:
    measurements = _measurement_frame(
        [
            ("DRAM_SOCKET_TOTAL_ENERGY", "dram-0", None, 100.0, START, START),
            ("DRAM_SOCKET_TOTAL_ENERGY", "dram-0", None, 100.0 + MEASURED_ENERGY_J, END, END),
            ("DRAM_SOCKET_PID_MEMORY", "dram-0", "p1", 1000.0, START, START),
            ("DRAM_SOCKET_PID_MEMORY", "dram-0", "p2", 1000.0, START, START),
            (DRAMEnergyAttributionModel._READ_LOCAL_METRIC, "core-A", "p1", 300.0, START, START),
            (DRAMEnergyAttributionModel._READ_LOCAL_METRIC, "core-A", "p2", 100.0, START, START),
        ]
    )
    dram_df = pl.DataFrame(
        {
            "id": ["dram-0"],
            "cpu_id": ["cpu-A"],
            "parent_id": ["host-1"],
            "idle_power": [CLAIMED_IDLE_POWER_W],
        }
    )
    core_df = pl.DataFrame({"id": ["core-A"], "parent_id": ["cpu-A"]})

    total, _, dynamic = DRAMEnergyAttributionModel().attribute_energy(
        measurement_df=measurements, core_df=core_df, dram_df=dram_df
    )

    assert (dynamic["value"] >= 0.0).all()
    assert total["value"].sum() == pytest.approx(MEASURED_ENERGY_J)


# %%
