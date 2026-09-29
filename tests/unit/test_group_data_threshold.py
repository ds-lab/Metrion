from datetime import UTC, datetime

import polars as pl
import pytest
from pydantic import ValidationError

from metrion.config_reader import SourceIntervals
from metrion.monitoring.timing_context import TimingContext

pytest.importorskip("metrion.metrion_rust", reason="sched_switch needs the built metrion_rust extension")

from metrion.monitoring.data_source.cpu.sched_switch import SchedSwitchMonitor  # noqa: E402
from metrion.monitoring.data_source.cpu.utils import grouping_threshold_ms  # noqa: E402


@pytest.mark.parametrize(
    "interval_ms,expected",
    [(1000, 100), (500, 50), (250, 25), (200, 20), (100, 10), (30, 5), (5000, 100)],
)
def test_threshold_scales_with_interval(interval_ms, expected):
    assert grouping_threshold_ms(interval_ms) == expected


@pytest.mark.parametrize("interval_ms", [30, 100, 200, 250, 500, 1000, 5000])
def test_usable_window_is_never_empty(interval_ms):
    """Regression: with the fixed 100 ms band the window was empty from 200 ms downwards."""
    assert interval_ms - 2 * grouping_threshold_ms(interval_ms) > 0


@pytest.mark.parametrize("interval_ms", [100, 250, 500, 1000])
def test_window_keeps_80_percent_of_the_interval(interval_ms):
    assert interval_ms - 2 * grouping_threshold_ms(interval_ms) == pytest.approx(0.8 * interval_ms)


def test_every_rapl_interval_the_config_accepts_stays_groupable():
    """Ties the lower bound in SourceIntervals to this formula: lowering it must fail here first."""
    for interval_ms in range(1, 2001):
        try:
            SourceIntervals(rapl=interval_ms / 1000)
        except ValidationError:
            continue
        assert interval_ms - 2 * grouping_threshold_ms(interval_ms) > 0, f"{interval_ms} ms leaves no window"


def _bare_monitor():
    # object.__new__ skips the class's singleton __new__, so this cannot hand back - or
    # overwrite - the instance another test is using.
    monitor = object.__new__(SchedSwitchMonitor)
    monitor._boot_time_s = 0.0
    return monitor


def test_group_data_uses_the_scaled_threshold():
    """Two rows 50-60 ms and 70-80 ms into a 200 ms interval: groupable at 20 ms, not at 100 ms."""
    monitor = _bare_monitor()
    tc = TimingContext(
        source_name="test",
        start_time_ms=0,
        interval_seconds=0.2,
        start_datetime=datetime.now(UTC),
    )
    df = pl.DataFrame(
        {
            "pid": [1, 1],
            "pid_start": [0, 0],
            "cpu_core": [0, 0],
            "time_start": [50_000_000, 70_000_000],
            "time_end": [60_000_000, 80_000_000],
            "execution_time": [1.0, 2.0],
        }
    )

    result = monitor._group_data(df, ["execution_time"], tc)

    assert result.height == 1, "both rows sit inside one bucket and share a group key"
    assert result["execution_time"].item() == 3.0


def _one_second_context():
    return TimingContext(
        source_name="test",
        start_time_ms=0,
        interval_seconds=1.0,
        start_datetime=datetime.now(UTC),
    )


def _events(starts_ms, ends_ms, values):
    return pl.DataFrame(
        {
            "pid": [1] * len(starts_ms),
            "pid_start": [0] * len(starts_ms),
            "cpu_core": [0] * len(starts_ms),
            "time_start": [ms * 1_000_000 for ms in starts_ms],
            "time_end": [ms * 1_000_000 for ms in ends_ms],
            "execution_time": values,
        }
    )


def test_guard_band_events_are_summed():
    """Three fragments in the opening 100 ms guard band become one row, not three."""
    monitor = _bare_monitor()

    result = monitor._group_data(
        _events([10, 20, 30], [11, 21, 31], [1.0, 2.0, 3.0]), ["execution_time"], _one_second_context()
    )

    assert result.height == 1
    assert result["execution_time"].item() == 6.0


def test_guard_band_stays_apart_from_the_usable_window():
    """A guard band event and a window event share a group key but must not be summed together."""
    monitor = _bare_monitor()

    result = monitor._group_data(_events([10, 500], [11, 501], [1.0, 2.0]), ["execution_time"], _one_second_context())

    assert result.height == 2
    assert sorted(result["execution_time"].to_list()) == [1.0, 2.0]


def test_straddling_event_is_not_read_as_inside_the_window():
    """An event crossing the interval end wraps pos_end, which would otherwise look early."""
    monitor = _bare_monitor()

    result = monitor._group_data(
        _events([500, 500], [600, 1200], [1.0, 2.0]), ["execution_time"], _one_second_context()
    )

    assert result.height == 2
    assert sorted(result["execution_time"].to_list()) == [1.0, 2.0]
