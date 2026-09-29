import time

import numpy as np
import polars as pl
import pytest

pytest.importorskip("metrion.metrion_rust", reason="sched_switch needs the built metrion_rust extension")

from metrion.monitoring.data_source.cpu import sched_switch as ss  # noqa: E402

Monitor = ss.SchedSwitchBpfMonitor
RECORD = Monitor.RING_RECORD_DTYPE
RING_SIZE = 1024  # a power of two that does not divide into whole records, so records wrap


class FakeRing:
    """A ring buffer laid out like the kernel's: positions as u64, data mapped twice."""

    def __init__(self):
        self.buf = np.zeros(RING_SIZE, np.uint8)
        self.cons = np.zeros(1, np.uint64)
        self.prod = np.zeros(1, np.uint64)

    def write(self, pid, state="ready", event_type=0, comm=b"", cmdline=b""):
        rec = np.zeros(1, RECORD)
        rec["len"] = 328 | {"ready": 0, "busy": ss._RINGBUF_BUSY_BIT, "discarded": ss._RINGBUF_DISCARD_BIT}[state]
        rec["event"]["pid"] = pid
        rec["event"]["event_type"] = event_type
        rec["event"]["comm"] = comm
        rec["event"]["cmdline"] = cmdline
        raw = np.frombuffer(rec.tobytes(), np.uint8)
        start = int(self.prod[0]) % RING_SIZE
        for i, b in enumerate(raw):
            self.buf[(start + i) % RING_SIZE] = b
        self.prod[0] += RECORD.itemsize

    def attach(self, monitor):
        monitor._rb_mask = RING_SIZE - 1
        monitor._rb_cons = self.cons
        monitor._rb_prod = self.prod
        # The kernel maps the data area twice in a row; a copy of it stands in for that here.
        monitor._rb_data = np.concatenate([self.buf, self.buf])


def make_monitor(ring=None):
    monitor = Monitor.__new__(Monitor)
    monitor._raw_buffer = []
    monitor._pending_events = 0
    monitor._flush_requested = False
    monitor._buffer_flush_threshold = 10_000
    monitor._buffer_flush_time_threshold = 60.0
    monitor._last_flush_time = time.time()
    monitor._df_schema = {}
    if ring is not None:
        ring.attach(monitor)
    return monitor


def consumed_pids(monitor):
    return [int(p) for chunk in monitor._raw_buffer for p in chunk["pid"]]


def test_all_ready_records_are_taken_and_the_position_moves_past_them():
    ring = FakeRing()
    for pid in (1, 2):
        ring.write(pid)
    monitor = make_monitor(ring)

    monitor._consume_ring()

    assert consumed_pids(monitor) == [1, 2]
    assert monitor._pending_events == 2
    assert int(ring.cons[0]) == 2 * RECORD.itemsize


def test_reading_stops_at_a_record_the_kernel_is_still_filling():
    ring = FakeRing()
    ring.write(1)
    ring.write(2, state="busy")
    ring.write(3)
    monitor = make_monitor(ring)

    monitor._consume_ring()

    assert consumed_pids(monitor) == [1]
    assert int(ring.cons[0]) == RECORD.itemsize


def test_discarded_records_are_skipped_but_read_past():
    ring = FakeRing()
    ring.write(1)
    ring.write(2, state="discarded")
    ring.write(3)
    monitor = make_monitor(ring)

    monitor._consume_ring()

    assert consumed_pids(monitor) == [1, 3]
    assert monitor._pending_events == 2
    assert int(ring.cons[0]) == 3 * RECORD.itemsize


def test_a_record_running_past_the_end_of_the_ring_is_read_in_one_piece():
    ring = FakeRing()
    for pid in (1, 2, 3):  # 3 x 336 = 1008 of 1024 bytes
        ring.write(pid)
    ring.cons[0] = ring.prod[0]  # the reader has taken those already
    ring.write(4)  # starts at byte 1008 and wraps
    monitor = make_monitor(ring)

    monitor._consume_ring()

    assert consumed_pids(monitor) == [4]


def test_an_empty_ring_leaves_everything_as_it_was():
    monitor = make_monitor(FakeRing())

    monitor._consume_ring()

    assert monitor._raw_buffer == []
    assert monitor._pending_events == 0


def test_a_flush_is_requested_once_enough_events_are_pending():
    ring = FakeRing()
    for pid in (1, 2):
        ring.write(pid)
    monitor = make_monitor(ring)
    monitor._buffer_flush_threshold = 2

    monitor._consume()

    assert monitor._flush_requested is True
    assert monitor._pending_events == 0


def test_buffered_events_counts_events_not_chunks():
    ring = FakeRing()
    for pid in (1, 2):
        ring.write(pid)
    monitor = make_monitor(ring)

    monitor._consume_ring()

    assert len(monitor._raw_buffer) == 1
    assert monitor._buffered_events() == 2


def test_only_events_with_metadata_get_a_name_and_command_line():
    events = np.zeros(3, Monitor.SCHED_EVENT_DTYPE)
    events["pid"] = [10, 11, 12]
    events["event_type"] = [0, 1, 0]
    events["uid"] = [7, 8, 9]
    events["comm"] = [b"leftover", b"python3", b"leftover"]
    events["cmdline"] = [b"leftover", b"  python3 -m app  ", b"leftover"]
    monitor = make_monitor()

    df = monitor._parse_raw_events([events[:2], events[2:]])

    assert df["pid"].to_list() == [10, 11, 12]
    assert df["comm"].to_list() == [None, "python3", None]
    assert df["cmdline"].to_list() == [None, "python3 -m app", None]
    assert df["uid"].to_list() == [None, 8, None]
    assert df.schema["comm"] == pl.String


def test_a_batch_without_metadata_has_empty_name_columns():
    events = np.zeros(2, Monitor.SCHED_EVENT_DTYPE)
    monitor = make_monitor()

    df = monitor._parse_raw_events([events])

    assert df["comm"].to_list() == [None, None]
    assert df.schema["cmdline"] == pl.String
