import datetime as dt

import polars as pl
import pytest

from metrion.discovery.system_scanner.bare_metal.linux.max_power_gpu_dram_cpu import read_energy_range_j
from metrion.energy_attribution.energy_attribution import (
    CPUEnergyAttributionModel,
    DRAMEnergyAttributionModel,
)

# The counter runs 95 -> 5 over one second. With this range that is a 10 J wrap, with the
# built-in constant it would be 262 kJ, so the two cases cannot be confused.
RANGE_J = 100.0
COUNTER_BEFORE = 95.0
COUNTER_AFTER = 5.0


def _ts(second):
    return dt.datetime(2026, 1, 1) + dt.timedelta(seconds=second)


def _rows(metric, entity, logical, values, start_second, end_second):
    return pl.DataFrame(
        {
            "metric_id": [metric] * len(values),
            "physical_entity_id": [entity] * len(values),
            "logical_entity_id": pl.Series(logical, dtype=pl.String),
            "value": values,
            "start_ts": [_ts(start_second)] * len(values),
            "end_ts": [_ts(end_second)] * len(values),
        }
    )


def _cpu_frames():
    cpu_df = pl.DataFrame(
        {
            "id": ["cpu-A"],
            "idle_power_package": [0.0],
            "max_energy_range_j": [RANGE_J],
        }
    )
    core_df = pl.DataFrame({"id": ["core-A"], "parent_id": ["cpu-A"]})
    return cpu_df, core_df


def _cpu_measurements():
    return pl.concat(
        [
            _rows("CPU_SOCKET_PKG_ENERGY", "cpu-A", [None], [COUNTER_BEFORE], 0, 0),
            _rows("CPU_SOCKET_PKG_ENERGY", "cpu-A", [None], [COUNTER_AFTER], 1, 1),
            # One thread does all the work, so it is attributed the whole interval.
            _rows("CPU_CORE_PID_WORK", "core-A", ["p1"], [1000.0], 0, 1),
            _rows("CPU_CORE_PID_TIME", "core-A", ["p1"], [1000.0], 0, 1),
        ]
    )


def _cpu_energy(cpu_df):
    core_df = _cpu_frames()[1]
    total, _, _ = CPUEnergyAttributionModel().attribute_energy(
        measurement_df=_cpu_measurements(), cpu_df=cpu_df, core_df=core_df
    )
    return total["value"].sum()


def test_the_cpu_wrap_uses_the_range_of_its_socket():
    cpu_df, _ = _cpu_frames()

    assert _cpu_energy(cpu_df) == pytest.approx(RANGE_J - COUNTER_BEFORE + COUNTER_AFTER)


def test_the_cpu_wrap_falls_back_to_the_constant():
    """Runs recorded before the range was read carry no such column."""
    cpu_df, _ = _cpu_frames()

    energy = _cpu_energy(cpu_df.drop("max_energy_range_j"))

    assert energy == pytest.approx(CPUEnergyAttributionModel._ENERGY_WRAP_CPU - COUNTER_BEFORE + COUNTER_AFTER)


def _dram_energy(dram_df):
    core_df = pl.DataFrame({"id": ["core-A"], "parent_id": ["cpu-A"]})
    measurements = pl.concat(
        [
            _rows("DRAM_SOCKET_TOTAL_ENERGY", "dram-0", [None], [COUNTER_BEFORE], 0, 0),
            _rows("DRAM_SOCKET_TOTAL_ENERGY", "dram-0", [None], [COUNTER_AFTER], 1, 1),
            _rows("DRAM_SOCKET_PID_MEMORY", "dram-0", ["p1"], [1000.0], 0, 0),
            _rows(DRAMEnergyAttributionModel._READ_LOCAL_METRIC, "core-A", ["p1"], [100.0], 0, 0),
        ]
    )
    total, _, _ = DRAMEnergyAttributionModel().attribute_energy(
        measurement_df=measurements, core_df=core_df, dram_df=dram_df
    )
    return total["value"].sum()


def test_the_dram_wrap_uses_the_range_of_its_module():
    dram_df = pl.DataFrame(
        {
            "id": ["dram-0"],
            "cpu_id": ["cpu-A"],
            "parent_id": ["host-1"],
            "idle_power": [0.0],
            "max_energy_range_j": [RANGE_J],
        }
    )

    assert _dram_energy(dram_df) == pytest.approx(RANGE_J - COUNTER_BEFORE + COUNTER_AFTER)


def test_the_dram_wrap_falls_back_to_the_constant():
    dram_df = pl.DataFrame(
        {
            "id": ["dram-0"],
            "cpu_id": ["cpu-A"],
            "parent_id": ["host-1"],
            "idle_power": [0.0],
        }
    )

    energy = _dram_energy(dram_df)

    assert energy == pytest.approx(DRAMEnergyAttributionModel._ENERGY_WRAP_DRAM - COUNTER_BEFORE + COUNTER_AFTER)


def test_the_range_is_read_next_to_the_energy_counter(tmp_path):
    (tmp_path / "max_energy_range_uj").write_text("262143328850\n")

    assert read_energy_range_j(tmp_path) == pytest.approx(262_143.328850)


def test_a_domain_without_the_file_reports_no_range(tmp_path):
    assert read_energy_range_j(tmp_path) is None
