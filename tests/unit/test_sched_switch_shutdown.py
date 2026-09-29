import multiprocessing as mp
import queue as q
import time
from unittest.mock import MagicMock

import pytest
from loguru import logger

from metrion.queue_util import drain_remaining, drain_until_sentinel

SENTINEL = "SENTINEL"


# --------------------------------------------------------------------------
# Drain behaviour, testable without the extension.
# --------------------------------------------------------------------------


def test_drain_stops_at_sentinel_and_ignores_it_as_path():
    """Regression: the sentinel must never reach read_temp_ipc as a file path."""
    queue = mp.Queue()
    queue.put("/tmp/batch_a.ipc")
    queue.put("/tmp/batch_b.ipc")
    queue.put(SENTINEL)

    paths, seen = drain_until_sentinel(queue, SENTINEL, deadline_s=3.0)

    assert seen is True
    assert paths == ["/tmp/batch_a.ipc", "/tmp/batch_b.ipc"]
    assert SENTINEL not in paths


def test_drain_terminates_when_sentinel_already_consumed():
    """If _get_raw_monitoring_data swallowed the sentinel, stop() must not hang."""
    queue = mp.Queue()
    queue.put("/tmp/batch_c.ipc")

    paths, seen = drain_until_sentinel(queue, SENTINEL, deadline_s=0.5, poll_timeout=0.05)

    assert paths == ["/tmp/batch_c.ipc"]
    assert seen is False


def test_drain_remaining_returns_once_the_queue_goes_idle():
    """The _producer_finished branch must not sit out its whole deadline."""
    queue = mp.Queue()
    queue.put("/tmp/batch_d.ipc")

    start = time.monotonic()
    paths = drain_remaining(queue, idle_timeout=0.3, deadline_s=5.0)
    elapsed = time.monotonic() - start

    assert paths == ["/tmp/batch_d.ipc"]
    assert elapsed < 2.0, f"took {elapsed:.1f}s although the queue was idle after one item"


def test_drain_remaining_on_an_empty_queue_returns_immediately():
    start = time.monotonic()
    paths = drain_remaining(q.Queue(), idle_timeout=0.2, deadline_s=5.0)
    elapsed = time.monotonic() - start

    assert paths == []
    assert elapsed < 1.0


def test_drain_remaining_stops_at_the_deadline_when_items_keep_arriving():
    """A producer that never stops must not keep stop() alive forever."""

    class NeverEmpty:
        def get(self, timeout=None):
            return "/tmp/endless.ipc"

    start = time.monotonic()
    paths = drain_remaining(NeverEmpty(), idle_timeout=0.1, deadline_s=0.5)
    elapsed = time.monotonic() - start

    assert paths, "items should still be returned"
    assert 0.4 < elapsed < 2.0


# --------------------------------------------------------------------------
# The real SchedSwitchMonitor methods, bound to a stand-in instance.
# The module needs the built metrion_rust extension, so these are
# skipped where it is missing.
# --------------------------------------------------------------------------

try:
    from metrion.monitoring.data_source.cpu import sched_switch as ss
except ImportError:
    ss = None

needs_bpf_stack = pytest.mark.skipif(ss is None, reason="needs the built metrion_rust extension")


@pytest.fixture
def make_monitor():
    """__init__ would attach BPF probes and open perf events, so it is bypassed."""

    class FakeMonitor:
        _get_raw_monitoring_data = ss.SchedSwitchMonitor._get_raw_monitoring_data
        _report_dead_producer = ss.SchedSwitchMonitor._report_dead_producer
        stop = ss.SchedSwitchMonitor.stop

        def __init__(self, producer_finished=False):
            self._queue_monitoring_data = mp.Queue()
            self._ring_buffer_config = MagicMock()
            self._ring_buffer_config.sentinel = SENTINEL
            self._producer_finished = producer_finished
            self._producer_death_reported = False
            self._rb_sched_switch = MagicMock()
            self._rb_sched_switch.is_alive.return_value = False
            self._rb_sched_switch.exitcode = -9
            self._calculate_and_cache_metrics = MagicMock()
            self.get_data = MagicMock(return_value="RECORD")

    return FakeMonitor


@pytest.fixture
def no_ipc_reads(monkeypatch):
    """read_temp_ipc would need real IPC files; record the paths instead."""
    paths = []
    monkeypatch.setattr(ss, "read_temp_ipc", lambda path: paths.append(path) or MagicMock())
    monkeypatch.setattr(ss.pl, "concat", lambda dfs, how: MagicMock())
    return paths


@needs_bpf_stack
def test_sentinel_during_normal_operation_is_not_read_as_a_path(make_monitor, no_ipc_reads):
    monitor = make_monitor()
    monitor._queue_monitoring_data.put(SENTINEL)
    time.sleep(0.3)

    df = monitor._get_raw_monitoring_data()

    assert monitor._producer_finished is True
    assert df.is_empty()
    assert no_ipc_reads == []


@needs_bpf_stack
def test_stop_does_not_hang_when_the_sentinel_was_consumed_early(make_monitor, no_ipc_reads):
    """This is the guaranteed hang: stop() used to wait for a sentinel that was gone."""
    monitor = make_monitor(producer_finished=True)

    start = time.monotonic()
    record = monitor.stop([])
    elapsed = time.monotonic() - start

    assert record == "RECORD"
    assert elapsed < 3.0, f"stop() took {elapsed:.1f}s although the producer was already done"


@needs_bpf_stack
def test_stop_drains_paths_and_skips_the_sentinel(make_monitor, no_ipc_reads):
    monitor = make_monitor()
    monitor._queue_monitoring_data.put("/tmp/a.ipc")
    monitor._queue_monitoring_data.put(SENTINEL)
    time.sleep(0.3)

    monitor.stop([])

    assert no_ipc_reads == ["/tmp/a.ipc"]
    assert monitor._calculate_and_cache_metrics.call_count == 1


@needs_bpf_stack
def test_stop_is_bounded_and_kills_an_unresponsive_producer(make_monitor, no_ipc_reads, monkeypatch):
    """A crashed producer never sends the sentinel; stop() must still return."""
    monkeypatch.setattr(
        ss,
        "drain_until_sentinel",
        lambda queue, sentinel, deadline_s, name: drain_until_sentinel(queue, sentinel, deadline_s=1.0, name=name),
    )
    monitor = make_monitor()
    monitor._rb_sched_switch.is_alive.return_value = True

    start = time.monotonic()
    monitor.stop([])
    elapsed = time.monotonic() - start

    assert 0.9 < elapsed < 4.0, f"took {elapsed:.1f}s"
    assert monitor._rb_sched_switch.kill.called
    assert monitor._calculate_and_cache_metrics.call_count == 0, "no data means no cache build"


def _errors(action):
    messages = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level="ERROR")
    try:
        action()
    finally:
        logger.remove(sink_id)
    return messages


@needs_bpf_stack
def test_a_producer_that_dies_mid_run_is_reported_once(make_monitor, no_ipc_reads):
    """The OOM killer ends the producer without a sentinel; the run must say so."""
    monitor = make_monitor()

    errors = _errors(lambda: [monitor._get_raw_monitoring_data() for _ in range(3)])

    assert len(errors) == 1
    assert "exit code -9" in errors[0]


@needs_bpf_stack
def test_a_producer_that_finished_normally_is_not_reported(make_monitor, no_ipc_reads):
    monitor = make_monitor()
    monitor._queue_monitoring_data.put(SENTINEL)
    time.sleep(0.3)

    def stop_and_collect_again():
        monitor.stop([])
        monitor._get_raw_monitoring_data()

    assert _errors(stop_and_collect_again) == []
