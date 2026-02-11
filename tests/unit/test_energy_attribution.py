from pathlib import Path

import polars as pl
import polars.testing as plt
import pytest

from metrion.energy_attribution.energy_attribution import (
    CPUEnergyAttributionModel,
    DRAMEnergyAttributionModel,
)
from metrion.metric import (
    CPU_PKG_PID_DYNAMIC_ENERGY,
    CPU_PKG_PID_IDLE_ENERGY,
    CPU_PKG_PID_TOTAL_ENERGY,
    DRAM_SOCKET_PID_DYNAMIC_ENERGY,
    DRAM_SOCKET_PID_IDLE_ENERGY,
    DRAM_SOCKET_PID_TOTAL_ENERGY,
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
        pl.col("dyn_energy_j").alias("value"),
        pl.lit(CPU_PKG_PID_DYNAMIC_ENERGY.name).alias("metric_id"),
        pl.col("package_id").alias("physical_entity_id"),
        pl.col("interval_start").alias("start_ts"),
        pl.col("interval_end").alias("end_ts"),
    )

    expected_total_df = expected_df.select(
        pl.col("logical_entity_id"),
        (pl.col("idle_energy_j") + pl.col("dyn_energy_j")).alias("value"),
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


# %%
