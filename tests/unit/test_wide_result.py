import datetime as dt

import polars as pl
import pytest

from metrion.energy_attribution.wide_result import (
    ENERGY_COLUMNS,
    IDENTITY_COLUMNS,
    METRIC_TO_COLUMN,
    denormalize,
    identity_of,
    to_wide,
)
from metrion.entity.logical_entity import LogicalEntity

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


def _thread():
    t = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    return LogicalEntity(
        native_thread_id=1234,
        native_pid=1234,
        native_parent_pid=1,
        native_parent_thread_id=1,
        executable_name="python3",
        cmdline="/usr/bin/python3 train.py",
        start_time=t,
        pid_start_time=t,
        parent_tid_start_time=t,
        parent_pid_start_time=t,
        resource_group_name="/system.slice/train.service",
        user_id=1000,
        group_id=1000,
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


def test_a_thread_gets_its_identity():
    thread = _thread()
    wide = to_wide([_long("CPU_PKG_PID_TOTAL_ENERGY", 5.0, 0, 1, thread=thread.id)], WINDOW)

    row = denormalize(wide, {thread.id: identity_of(thread)}).row(0, named=True)

    assert (row["native_pid"], row["executable_name"]) == (1234, "python3")
    assert row["cgroup_name"] == "/system.slice/train.service"
    assert row["cpu_total_j"] == pytest.approx(5.0)


def test_a_thread_without_identity_keeps_its_energy():
    """Energy can be attributed before the thread's entity arrives; the row must not vanish."""
    wide = to_wide([_long("CPU_PKG_PID_TOTAL_ENERGY", 5.0, 0, 1)], WINDOW)

    out = denormalize(wide, {})

    assert out.height == 1
    assert out["native_pid"][0] is None
    assert out["cpu_total_j"][0] == pytest.approx(5.0)


def test_the_schema_does_not_depend_on_what_is_known():
    """Each cycle writes its own Parquet file, and they are only readable as one if their types agree."""
    thread = _thread()
    wide = to_wide([_long("CPU_PKG_PID_TOTAL_ENERGY", 5.0, 0, 1, thread=thread.id)], WINDOW)

    known = denormalize(wide, {thread.id: identity_of(thread)})
    unknown = denormalize(wide, {})
    empty = denormalize(to_wide([], WINDOW), {})

    assert known.schema == unknown.schema == empty.schema
    assert known.columns == ["window_start", "window_end", "logical_entity_id", *IDENTITY_COLUMNS, *ENERGY_COLUMNS]
