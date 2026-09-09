import datetime as dt

import polars as pl
import pytest

from metrion.energy_attribution.energy_attribution import (
    CPUEnergyAttributionModel,
    DRAMEnergyAttributionModel,
    DyncamicEnergyAttributionModel,
    GPUEnergyAttributionModel,
)

GPU_STREAMS = (
    GPUEnergyAttributionModel._ENERGY_METRIC,
    GPUEnergyAttributionModel._UTILIZATION_METRIC,
)

CPU_STREAMS = (
    CPUEnergyAttributionModel._PKG_METRIC,
    CPUEnergyAttributionModel._WORK_METRIC,
    CPUEnergyAttributionModel._TIME_METRIC,
)


def _t(seconds: float) -> dt.datetime:
    return dt.datetime(2026, 1, 1) + dt.timedelta(seconds=seconds)


def _rows(metric: str, times, value: float = 1.0) -> pl.DataFrame:
    times = list(times)
    return pl.DataFrame(
        {
            "metric_id": [metric] * len(times),
            "physical_entity_id": ["pkg_0"] * len(times),
            "logical_entity_id": [None] * len(times),
            "value": [value] * len(times),
            "start_ts": times,
            "end_ts": times,
        },
        schema_overrides={"logical_entity_id": pl.String},
    )


def _feed(model: DyncamicEnergyAttributionModel, metrics, times) -> None:
    for metric in metrics:
        model.update_measurements(_rows(metric, times))


def test_energy_alone_does_not_open_a_window():
    """The actual regression test: RAPL without sched_switch must not attribute anything."""
    model = DyncamicEnergyAttributionModel(cutoff=0.0)
    model.update_measurements(_rows(CPUEnergyAttributionModel._PKG_METRIC, [_t(0), _t(1)]))
    model.update_measurements(_rows(CPUEnergyAttributionModel._WORK_METRIC, [_t(0)]))
    model.update_measurements(_rows(CPUEnergyAttributionModel._TIME_METRIC, [_t(0)]))

    _, bounds = model._get_relevant_measurements()

    assert bounds["cpu"] == _t(0), "the bound follows the slowest stream, not the fastest"


def test_bound_advances_when_the_slow_stream_catches_up():
    model = DyncamicEnergyAttributionModel(cutoff=0.0)
    _feed(model, CPU_STREAMS, [_t(0), _t(1), _t(2)])

    _, bounds = model._get_relevant_measurements()

    assert bounds["cpu"] == _t(2)


def test_never_seen_stream_is_not_required():
    """Without a GPU in the machine the GPU bound must not block the CPU."""
    model = DyncamicEnergyAttributionModel(cutoff=0.0)
    _feed(model, CPU_STREAMS, [_t(0)])

    _, bounds = model._get_relevant_measurements()

    assert bounds["cpu"] == _t(0)
    assert bounds["gpu"] is None


def test_watermark_is_monotonic_across_batches():
    """The bound must not fall back once the buffer has been trimmed."""
    model = DyncamicEnergyAttributionModel(cutoff=0.0)
    _feed(model, CPU_STREAMS, [_t(0), _t(1)])
    model._get_relevant_measurements()  # trims the buffer

    _, bounds = model._get_relevant_measurements()

    assert bounds["cpu"] == _t(1), "computed from the buffer contents the bound would have jumped back"


def test_stalled_stream_stops_blocking():
    """A dried-up stream must not hold up attribution forever."""
    model = DyncamicEnergyAttributionModel(cutoff=0.0, stall_after=10.0)
    _feed(model, CPU_STREAMS, [_t(0)])
    # WORK stops here, the other two keep going
    model.update_measurements(_rows(CPUEnergyAttributionModel._PKG_METRIC, [_t(100)]))
    model.update_measurements(_rows(CPUEnergyAttributionModel._TIME_METRIC, [_t(100)]))

    _, bounds = model._get_relevant_measurements()

    assert bounds["cpu"] == _t(100), "past stall_after the dried-up stream is no longer required"


def test_force_drains_past_the_slowest_stream():
    """The shutdown drain must not get stuck on the slowest stream."""
    model = DyncamicEnergyAttributionModel(cutoff=15.0)
    model.update_measurements(_rows(CPUEnergyAttributionModel._PKG_METRIC, [_t(0), _t(5)]))
    model.update_measurements(_rows(CPUEnergyAttributionModel._WORK_METRIC, [_t(0)]))
    model.update_measurements(_rows(CPUEnergyAttributionModel._TIME_METRIC, [_t(0)]))

    _, normal = model._get_relevant_measurements()
    _, forced = model._get_relevant_measurements(force=True)

    assert normal["cpu"] == _t(0)
    assert forced["cpu"] == _t(5)


def test_dram_bound_waits_for_the_read_counters():
    """DRAM energy is split by L3 misses, so those streams gate the DRAM bound too."""
    model = DyncamicEnergyAttributionModel(cutoff=0.0)
    model.update_measurements(_rows(DRAMEnergyAttributionModel._DRAM_METRIC, [_t(0), _t(10)]))
    model.update_measurements(_rows(DRAMEnergyAttributionModel._MEMORY_METRIC, [_t(0), _t(10)]))
    model.update_measurements(_rows(DRAMEnergyAttributionModel._READ_LOCAL_METRIC, [_t(0)]))
    model.update_measurements(_rows(DRAMEnergyAttributionModel._READ_REMOTE_METRIC, [_t(0)]))

    _, bounds = model._get_relevant_measurements()

    assert bounds["dram"] == _t(0)


def test_aware_and_naive_timestamps_mix():
    """Measurement.to_df yields tz-aware rows, Parquet yields naive ones; both must land."""
    model = DyncamicEnergyAttributionModel(cutoff=0.0)
    aware = [t.replace(tzinfo=dt.UTC) for t in (_t(0), _t(1))]
    for metric in CPU_STREAMS:
        model.update_measurements(_rows(metric, aware))
        model.update_measurements(_rows(metric, [_t(2)]))

    _, bounds = model._get_relevant_measurements()

    assert bounds["cpu"] == _t(2)


def _cpu_batch(seconds):
    """RAPL point samples plus the per-thread rows covering each second."""
    rows = []
    for i in seconds:
        rows.append(
            {
                "metric_id": CPUEnergyAttributionModel._PKG_METRIC,
                "physical_entity_id": "pkg0",
                "logical_entity_id": None,
                "value": 100.0 + 33.0 * i,
                "start_ts": _t(i),
                "end_ts": _t(i),
            }
        )
        for core, thread in (("c0", "p1"), ("c1", "p2")):
            for metric, value in (
                (CPUEnergyAttributionModel._WORK_METRIC, 1e9),
                (CPUEnergyAttributionModel._TIME_METRIC, 1000.0),
            ):
                rows.append(
                    {
                        "metric_id": metric,
                        "physical_entity_id": core,
                        "logical_entity_id": thread,
                        "value": value,
                        "start_ts": _t(i),
                        "end_ts": _t(i + 1),
                    }
                )
    return pl.DataFrame(rows, schema_overrides={"logical_entity_id": pl.String})


def _drain_two_batches(force):
    model = DyncamicEnergyAttributionModel(cutoff=0.0)
    model._cpu_df = pl.DataFrame({"id": ["pkg0"], "idle_power_package": [0.0]})
    model._core_df = pl.DataFrame({"id": ["c0", "c1"], "parent_id": ["pkg0", "pkg0"]})

    total = 0.0
    for seconds in (range(0, 4), range(4, 7)):
        model.update_measurements(_cpu_batch(list(seconds)))
        (attributed, _, _), _, _ = model.attribute_energy(force=force)
        if not attributed.is_empty():
            total += float(attributed["value"].sum())
    return total


def test_no_energy_is_lost_between_batches():
    """The counter runs from 100 to 298 over six seconds, so 198 J must come out."""
    assert _drain_two_batches(force=False) == pytest.approx(198.0)


def test_the_forced_drain_loses_no_energy_either():
    """force lifts the bound past the energy stream; the buffer must not be trimmed with it."""
    assert _drain_two_batches(force=True) == pytest.approx(198.0)


def test_a_surviving_anchor_is_not_injected_twice():
    """An anchor above the deletion point stays in the buffer and must not be added again."""
    model = DyncamicEnergyAttributionModel(cutoff=0.0)
    model.update_measurements(_cpu_batch([0, 1, 2, 3]))
    # A second component two seconds behind, holding the deletion point back
    model.update_measurements(_rows(GPUEnergyAttributionModel._ENERGY_METRIC, [_t(0), _t(1)]))
    model.update_measurements(_rows(GPUEnergyAttributionModel._UTILIZATION_METRIC, [_t(0), _t(1)]))

    model._get_relevant_measurements()

    pkg = model._buffer.get_all().filter(pl.col("metric_id") == CPUEnergyAttributionModel._PKG_METRIC)
    assert pkg.get_column("end_ts").n_unique() == pkg.height, "a timestamp appears twice in the buffer"


def _drain_with_a_lagging_component(cycles=5):
    """CPU runs ahead while a GPU component lags two seconds and pins the deletion point."""
    model = DyncamicEnergyAttributionModel(cutoff=0.0)
    model._cpu_df = pl.DataFrame({"id": ["pkg0"], "idle_power_package": [0.0]})
    model._core_df = pl.DataFrame({"id": ["c0", "c1"], "parent_id": ["pkg0", "pkg0"]})

    total = 0.0
    intervals = []
    for k in range(cycles):
        model.update_measurements(_cpu_batch([2 * k, 2 * k + 1]))
        lag = [_t(max(0, 2 * k - 2)), _t(max(0, 2 * k - 1))]
        model.update_measurements(_rows(GPUEnergyAttributionModel._ENERGY_METRIC, lag))
        model.update_measurements(_rows(GPUEnergyAttributionModel._UTILIZATION_METRIC, lag))
        (attributed, _, _), _, _ = model.attribute_energy()
        if attributed.is_empty():
            continue
        total += float(attributed["value"].sum())
        intervals += list(
            zip(attributed["start_ts"], attributed["end_ts"], attributed["logical_entity_id"], strict=True)
        )
    return total, intervals


def test_a_lagging_component_does_not_inflate_the_others():
    """The counter runs 100 -> 397 over nine seconds, so 297 J is all there is."""
    total, _ = _drain_with_a_lagging_component()

    assert total == pytest.approx(297.0)


def test_intervals_are_emitted_once():
    """No interval may appear in two cycles while the buffer still holds it."""
    _, intervals = _drain_with_a_lagging_component()

    assert len(intervals) == len(set(intervals)), "an interval was billed twice"


def test_continuity_samples_do_not_count_as_late():
    """The false alarm: re-injected samples must not trip the late counter.

    The lagging component is what makes this bite. Its anchor ends up below the edge,
    while the fast component's anchor sits exactly on it and would pass either way.
    """
    model = DyncamicEnergyAttributionModel(cutoff=0.0)
    _feed(model, CPU_STREAMS, [_t(0), _t(10)])
    _feed(
        model, (GPUEnergyAttributionModel._ENERGY_METRIC, GPUEnergyAttributionModel._UTILIZATION_METRIC), [_t(0), _t(2)]
    )

    model._get_relevant_measurements()  # injects the continuity samples
    model._get_relevant_measurements()  # the old warning fired here

    assert model._late_rows == 0


def test_a_slower_component_may_still_deliver():
    """The edge is per component, so a GPU row the GPU has not reached yet is not late."""
    model = DyncamicEnergyAttributionModel(cutoff=0.0)
    model._cpu_df = pl.DataFrame({"id": ["pkg0"], "idle_power_package": [0.0]})
    model._core_df = pl.DataFrame({"id": ["c0", "c1"], "parent_id": ["pkg0", "pkg0"]})
    model.update_measurements(_cpu_batch([0, 1, 2]))  # CPU runs well ahead
    _feed(model, GPU_STREAMS, [_t(0)])
    model.attribute_energy()

    model.update_measurements(_rows(GPUEnergyAttributionModel._UTILIZATION_METRIC, [_t(2)]))

    assert model._late_rows == 0, "the GPU has not attributed t=2 yet"


def test_a_row_behind_its_own_component_is_late():
    model = DyncamicEnergyAttributionModel(cutoff=0.0)
    model._cpu_df = pl.DataFrame({"id": ["pkg0"], "idle_power_package": [0.0]})
    model._core_df = pl.DataFrame({"id": ["c0", "c1"], "parent_id": ["pkg0", "pkg0"]})
    model.update_measurements(_cpu_batch([0, 1, 2, 3]))
    model.attribute_energy()

    model.update_measurements(_rows(CPUEnergyAttributionModel._WORK_METRIC, [_t(0)]))

    assert model._late_rows == 1
