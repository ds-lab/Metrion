import datetime as dt

import polars as pl
import pytest

from metrion.energy_attribution.energy_attribution import (
    CPUEnergyAttributionModel,
    DRAMEnergyAttributionModel,
    GPUEnergyAttributionModel,
)


def _ts(sec, micro=0):
    return dt.datetime(2025, 1, 1, 0, 0, sec, micro)


@pytest.fixture
def cpu_model():
    return CPUEnergyAttributionModel()


@pytest.fixture
def intervals():
    return pl.DataFrame(
        {
            "package_id": ["p1", "p1", "p2", "p2"],
            "interval_start": [_ts(0), _ts(1), _ts(0), _ts(1)],
            "interval_end": [_ts(1), _ts(2), _ts(1), _ts(2)],
        }
    )


def _meas(package_id, start, end, value):
    return pl.DataFrame({"package_id": [package_id], "meas_start": [start], "meas_end": [end], "meas_value": [value]})


def test_contained_measurement_is_unchanged(cpu_model, intervals):
    """The common case must not move: one interval, full value."""
    out = cpu_model._split_over_intervals(_meas("p1", _ts(0, 200_000), _ts(0, 900_000), 50.0), intervals, "meas_value")
    assert out.height == 1
    assert out["meas_value"][0] == pytest.approx(50.0)
    assert out["interval_start"][0] == _ts(0)


def test_straddling_measurement_is_split_proportionally(cpu_model, intervals):
    """Regression: this measurement used to be discarded completely."""
    out = cpu_model._split_over_intervals(_meas("p1", _ts(0, 980_000), _ts(1, 30_000), 100.0), intervals, "meas_value")
    assert out.height == 2
    by_interval = dict(zip(out["interval_start"], out["meas_value"], strict=False))
    assert by_interval[_ts(0)] == pytest.approx(40.0)
    assert by_interval[_ts(1)] == pytest.approx(60.0)


def test_no_work_is_created_or_lost(cpu_model, intervals):
    out = cpu_model._split_over_intervals(_meas("p1", _ts(0, 980_000), _ts(1, 30_000), 100.0), intervals, "meas_value")
    assert out["meas_value"].sum() == pytest.approx(100.0)


def test_measurement_stays_on_its_own_package(cpu_model, intervals):
    """Regression: pl.col('package_id') == pl.col('package_id') was always true."""
    out = cpu_model._split_over_intervals(_meas("p1", _ts(0, 200_000), _ts(0, 900_000), 50.0), intervals, "meas_value")
    assert out.height == 1, "measurement was matched to more than one package"
    assert set(out["package_id"]) == {"p1"}


def test_measurement_spanning_three_intervals(cpu_model):
    ei = pl.DataFrame(
        {
            "package_id": ["p1"] * 3,
            "interval_start": [_ts(0), _ts(1), _ts(2)],
            "interval_end": [_ts(1), _ts(2), _ts(3)],
        }
    )
    out = cpu_model._split_over_intervals(_meas("p1", _ts(0, 500_000), _ts(2, 500_000), 200.0), ei, "meas_value")
    assert out.height == 3
    assert out["meas_value"].sum() == pytest.approx(200.0)
    by_interval = dict(zip(out["interval_start"], out["meas_value"], strict=False))
    assert by_interval[_ts(1)] == pytest.approx(100.0)  # the one full second out of two


def test_measurement_outside_all_intervals_is_dropped(cpu_model, intervals):
    out = cpu_model._split_over_intervals(_meas("p1", _ts(5), _ts(6), 10.0), intervals, "meas_value")
    assert out.height == 0


def test_point_measurement_is_kept(cpu_model, intervals):
    """Regression: start_ts == end_ts overlaps no half-open interval on its own."""
    out = cpu_model._split_over_intervals(_meas("p1", _ts(0, 500_000), _ts(0, 500_000), 7.0), intervals, "meas_value")
    assert out.height == 1
    assert out["interval_start"][0] == _ts(0)
    assert out["meas_value"][0] == pytest.approx(7.0)


def test_point_measurement_on_an_interval_border_is_kept(cpu_model, intervals):
    out = cpu_model._split_over_intervals(_meas("p1", _ts(1), _ts(1), 7.0), intervals, "meas_value")
    assert out.height == 1
    assert out["interval_start"][0] == _ts(1)
    assert out["meas_value"][0] == pytest.approx(7.0)


def _gpu_intervals():
    return pl.DataFrame(
        {
            "gpu_id": ["gpu-a", "gpu-b"],
            "interval_start": [_ts(0), _ts(0)],
            "interval_end": [_ts(10), _ts(10)],
        }
    )


def _util_measurement(gpu_id, start, end):
    return pl.DataFrame(
        {
            "metric_id": ["GPU_DEVICE_PID_COMPUTE_UTILIZATION"],
            "physical_entity_id": [gpu_id],
            "logical_entity_id": ["pid-1"],
            "value": [60.0],
            "start_ts": [start],
            "end_ts": [end],
        }
    )


def test_gpu_utilization_stays_on_its_own_device():
    """Regression: pl.col('gpu_id') == pl.col('gpu_id') matched every device."""
    aligned = GPUEnergyAttributionModel()._get_process_utilization(
        _util_measurement("gpu-a", _ts(5), _ts(5)), _gpu_intervals()
    )
    assert aligned.height == 1, "utilization sample was aligned to more than one GPU"
    assert aligned["gpu_id"][0] == "gpu-a"


def test_gpu_point_sample_is_aligned():
    """NVML writes utilization as a point sample, which must still find its interval."""
    aligned = GPUEnergyAttributionModel()._get_process_utilization(
        _util_measurement("gpu-a", _ts(3), _ts(3)), _gpu_intervals()
    )
    assert aligned.height == 1
    assert aligned["interval_start"][0] == _ts(0)


def _dram_frames(idle_power):
    dram_df = pl.DataFrame({"id": ["dram-0"], "parent_id": ["cpu-0"], "idle_power": [idle_power]})
    core_df = pl.DataFrame({"id": ["core-0"], "parent_id": ["cpu-0"]})
    return dram_df, core_df


def _dram_measurements(read_start, read_end):
    def rows(metric, entity, logical, values, starts, ends):
        return pl.DataFrame(
            {
                "metric_id": [metric] * len(values),
                "physical_entity_id": [entity] * len(values),
                "logical_entity_id": pl.Series(logical, dtype=pl.String),
                "value": values,
                "start_ts": starts,
                "end_ts": ends,
            }
        )

    energy = rows(
        "DRAM_SOCKET_TOTAL_ENERGY",
        "dram-0",
        [None] * 3,
        [0.0, 100.0, 300.0],
        [_ts(0), _ts(1), _ts(2)],
        [_ts(0), _ts(1), _ts(2)],
    )
    memory = rows(
        "DRAM_SOCKET_PID_MEMORY",
        "dram-0",
        ["pid-1"] * 2,
        [1000.0, 1000.0],
        [_ts(0, 500_000), _ts(1, 500_000)],
        [_ts(0, 500_000), _ts(1, 500_000)],
    )
    reads = rows("CPU_CORE_PID_MEM_LOAD_L3_MISS_LOCAL_DRAM", "core-0", ["pid-1"], [100.0], [read_start], [read_end])
    return pl.concat([energy, memory, reads])


def test_dram_reads_straddling_a_boundary_are_attributed():
    """Regression: a read window across a RAPL sample used to leave the PID at zero."""
    dram_df, core_df = _dram_frames(0.0)
    _, _, dynamic = DRAMEnergyAttributionModel().attribute_energy(
        measurement_df=_dram_measurements(_ts(0, 800_000), _ts(1, 200_000)),
        core_df=core_df,
        dram_df=dram_df,
    )
    attributed = dynamic.filter(pl.col("logical_entity_id") == "pid-1")
    assert attributed.height == 2, "the split read did not reach both energy intervals"
    assert attributed["value"].sum() == pytest.approx(300.0)


def test_dram_point_memory_samples_still_mark_presence():
    """Regression: DRAM_SOCKET_PID_MEMORY is written with start_ts == end_ts."""
    dram_df, core_df = _dram_frames(10.0)
    _, idle, _ = DRAMEnergyAttributionModel().attribute_energy(
        measurement_df=_dram_measurements(_ts(0, 200_000), _ts(0, 900_000)),
        core_df=core_df,
        dram_df=dram_df,
    )
    present = idle.filter(pl.col("logical_entity_id") == "pid-1")
    assert present.height == 2, "point memory samples no longer mark the PID as present"
    assert present["value"].sum() == pytest.approx(20.0)  # 10 W over two one-second intervals
