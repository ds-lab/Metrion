import numpy as np
import polars as pl
import pytest

pytest.importorskip("metrion.metrion_rust", reason="the Rust extension has to be built")

import metrion.metrion_rust as mar  # noqa: E402

HT = 1.15
ALONE, OVERLAP = 0, 1
COUNTERS = ("aperf", "mperf", "unhalted_core_cycles")


def frame(events):
    """events: (pid, core, time_start, time_end, {counter: (start, end)})."""
    columns = {
        name: []
        for name in [
            "pid",
            "pid_start",
            "exec_count",
            "cpu_core",
            "time_start",
            "time_end",
            *[f"{c}_{edge}" for c in COUNTERS for edge in ("start", "end")],
        ]
    }
    for index, (pid, core, start, end, counters) in enumerate(events):
        for name, value in (
            ("pid", pid),
            ("pid_start", pid * 11),
            ("exec_count", index),
            ("cpu_core", core),
            ("time_start", start),
            ("time_end", end),
        ):
            columns[name].append(value)
        for counter in COUNTERS:
            columns[f"{counter}_start"].append(counters[counter][0])
            columns[f"{counter}_end"].append(counters[counter][1])
    schema = {name: (pl.UInt32 if name in ("pid", "cpu_core") else pl.UInt64) for name in columns}
    return pl.DataFrame(columns, schema_overrides=schema)


def counters(aperf, mperf, cycles):
    return {"aperf": aperf, "mperf": mperf, "unhalted_core_cycles": cycles}


def segments_of(result, pid):
    rows = result.filter(pl.col("pid") == pid).sort("start_time")
    return [
        (
            r["start_time"],
            r["end_time"],
            r["shared_state"],
            r["delta_aperf"],
            r["delta_mperf"],
            r["delta_unhalted_core_cycles"],
        )
        for r in rows.iter_rows(named=True)
    ]


def test_two_cuts_split_a_slice_by_time():
    """The sibling starts a task and stops it again: both cuts give each segment its time share."""
    events = [
        (1, 0, 1000, 2000, counters((1000, 2000), (1000, 2000), (1000, 2000))),
        (2, 1, 1500, 1750, counters((5000, 5250), (5000, 5250), (5000, 5250))),
    ]

    result = mar.prepare_monitoring_data_for_tuples(frame(events), [(0, 1)], HT)

    assert segments_of(result, 1) == [
        (1000, 1500, ALONE, 500, 500, 500),
        (1500, 1750, OVERLAP, 250, 250, 287),  # 250 * 1.15, truncated
        (1750, 2000, ALONE, 250, 250, 250),
    ]
    assert segments_of(result, 2) == [(1500, 1750, OVERLAP, 250, 250, 287)]


def test_five_cuts_keep_every_segment_proportional():
    """Counters that run at different rates, cut five times: every segment keeps its time share."""
    events = [
        (1, 0, 10_000, 18_000, counters((100_000, 116_000), (100_000, 108_000), (100_000, 124_000))),
        (2, 1, 11_000, 13_000, counters((200_000, 204_000), (200_000, 202_000), (200_000, 206_000))),
        (3, 1, 14_000, 15_000, counters((300_000, 302_000), (300_000, 301_000), (300_000, 303_000))),
        (4, 1, 16_000, 20_000, counters((400_000, 408_000), (400_000, 404_000), (400_000, 412_000))),
    ]

    result = mar.prepare_monitoring_data_for_tuples(frame(events), [(0, 1)], HT)

    assert segments_of(result, 1) == [
        (10_000, 11_000, ALONE, 2000, 1000, 3000),
        (11_000, 13_000, OVERLAP, 4000, 2000, 6899),
        (13_000, 14_000, ALONE, 2000, 1000, 3000),
        (14_000, 15_000, OVERLAP, 2000, 1000, 3449),
        (15_000, 16_000, ALONE, 2000, 1000, 3000),
        (16_000, 18_000, OVERLAP, 4000, 2000, 6899),
    ]
    assert segments_of(result, 4) == [
        (16_000, 18_000, OVERLAP, 4000, 2000, 6899),
        (18_000, 20_000, ALONE, 4000, 2000, 6000),
    ]


def test_every_segment_gets_its_time_share():
    """For random timelines on an SMT pair, each segment holds its slice's counters by duration."""
    rng = np.random.default_rng(20260912)
    events, slices = [], {}
    for pid, core in ((1, 0), (2, 1)):
        time = 1_000_000
        for _ in range(60):
            duration = int(rng.integers(1_000, 60_000))
            start, end = time, time + duration
            values = counters(*[(1_000_000, 1_000_000 + duration * rate) for rate in (2, 1, 3)])
            events.append((pid, core, start, end, values))
            slices[(core, pid, start)] = (end, {c: values[c][1] - values[c][0] for c in COUNTERS})
            time = end + int(rng.integers(0, 40_000))

    result = mar.prepare_monitoring_data_for_tuples(frame(events), [(0, 1)], HT)
    assert result.height > len(events), "a random pair of timelines should produce cuts"

    for row in result.iter_rows(named=True):
        slice_start = max(
            s for c, p, s in slices if c == row["cpu_core"] and p == row["pid"] and s <= row["start_time"]
        )
        slice_end, deltas = slices[(row["cpu_core"], row["pid"], slice_start)]
        assert row["end_time"] <= slice_end
        share = (row["end_time"] - row["start_time"]) / (slice_end - slice_start)
        # aperf and mperf carry no HT weight, so they show the split unchanged. Both the
        # interpolated value and the delta are truncated, so a segment can come out one count
        # short; the interpolation this guards against was off by thousands.
        for counter in ("aperf", "mperf"):
            assert abs(row[f"delta_{counter}"] - deltas[counter] * share) < 2


def test_the_result_does_not_depend_on_the_input_order():
    """Siblings that switch in the same nanosecond give the same result whatever order the rows come in."""
    rng = np.random.default_rng(20260929)
    events = []
    for pid, core in ((1, 0), (2, 1)):
        time = 1_000_000
        for _ in range(60):
            # A coarse grid, so that both cores often start or stop at the same time.
            duration = int(rng.integers(1, 6)) * 1_000
            values = counters(*[(1_000_000, 1_000_000 + duration * rate) for rate in (2, 1, 3)])
            events.append((pid, core, time, time + duration, values))
            time += duration + int(rng.integers(0, 3)) * 1_000

    forward = mar.prepare_monitoring_data_for_tuples(frame(events), [(0, 1)], HT)
    backward = mar.prepare_monitoring_data_for_tuples(frame(events[::-1]), [(0, 1)], HT)

    assert forward.equals(backward)
