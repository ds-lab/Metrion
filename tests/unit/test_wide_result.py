import datetime as dt

import polars as pl
import pytest

from metrion.energy_attribution.wide_result import ENERGY_COLUMNS, METRIC_TO_COLUMN, to_wide

T0 = dt.datetime(2026, 1, 1)
WINDOW = 5.0


def _long(metric, value, start_s, end_s, thread="t1", physical="pkg_0"):
    return pl.DataFrame(
        {
            "metric_id": [metric],
            "physical_entity_id": [physical],
            "logical_entity_id": [thread],
            "value": [float(value)],
            "start_ts": [T0 + dt.timedelta(seconds=start_s)],
            "end_ts": [T0 + dt.timedelta(seconds=end_s)],
        }
    )


def test_no_frames_still_yields_the_full_schema():
    """The saver concatenates these frames, so an empty one has to carry the columns."""
    wide = to_wide([], WINDOW)

    assert wide.height == 0
    assert wide.columns[3:] == ENERGY_COLUMNS


def test_the_nine_metrics_become_nine_columns_in_one_row():
    frames = [_long(metric, i, 0, 1) for i, metric in enumerate(METRIC_TO_COLUMN, start=1)]

    wide = to_wide(frames, WINDOW)

    assert wide.height == 1
    assert wide.columns[3:] == ENERGY_COLUMNS


def test_a_row_is_spread_over_the_windows_it_overlaps():
    """A GPU interval of 5 s straddles the grid: 4 of its 5 seconds fall into the first window."""
    frames = [_long("GPU_DEVICE_PID_TOTAL_ENERGY", 50.0, 1, 6, physical="gpu_0")]

    wide = to_wide(frames, WINDOW)

    assert wide["window_start"].to_list() == [T0, T0 + dt.timedelta(seconds=5)]
    assert wide["gpu_total_j"].to_list() == pytest.approx([40.0, 10.0])


def test_energy_is_conserved():
    frames = [_long("CPU_PKG_PID_TOTAL_ENERGY", 10.0, i, i + 1) for i in range(13)]
    frames += [_long("GPU_DEVICE_PID_TOTAL_ENERGY", 50.0, 1, 6, physical="gpu_0")]

    wide = to_wide(frames, WINDOW)

    assert wide["cpu_total_j"].sum() == pytest.approx(130.0)
    assert wide["gpu_total_j"].sum() == pytest.approx(50.0)


def test_sockets_are_summed_into_one_row():
    """A thread that ran on both sockets has two long rows and one wide row."""
    metric = "CPU_PKG_PID_TOTAL_ENERGY"
    frames = [_long(metric, 3.0, 0, 1, physical="pkg_0"), _long(metric, 4.0, 0, 1, physical="pkg_1")]

    wide = to_wide(frames, WINDOW)

    assert wide.height == 1
    assert wide["cpu_total_j"][0] == pytest.approx(7.0)


def test_a_component_without_rows_is_zero_not_null():
    wide = to_wide([_long("CPU_PKG_PID_TOTAL_ENERGY", 1.0, 0, 1)], WINDOW)

    assert wide["gpu_total_j"][0] == 0.0
    assert wide.null_count().sum_horizontal()[0] == 0


def test_windows_lie_on_an_absolute_grid():
    """S4.5 adds up partial results per window, which only works if the boundaries never move."""
    metric = "CPU_PKG_PID_TOTAL_ENERGY"

    first = to_wide([_long(metric, 1.0, 1, 2)], WINDOW)
    second = to_wide([_long(metric, 2.0, 3, 4)], WINDOW)

    assert first["window_start"][0] == second["window_start"][0] == T0


def test_threads_stay_apart():
    metric = "CPU_PKG_PID_TOTAL_ENERGY"
    frames = [_long(metric, 1.0, 0, 1, thread="t1"), _long(metric, 2.0, 0, 1, thread="t2")]

    assert to_wide(frames, WINDOW).height == 2
