import multiprocessing as mp
import time
from unittest.mock import MagicMock

import pytest

from metrion.config_reader import LiveAttributionConfig
from metrion.queue_util import close_and_join_feeder

dc = pytest.importorskip("metrion.data_collector", reason="needs bcc")

SENT_E = dc.DataCollector.SENT_E
SENT_M = dc.DataCollector.SENT_M


@pytest.fixture
def saver():
    """DataSaver without a storage backend; _data_storage is built in store_data()."""
    ctx = mp.get_context("spawn")
    instance = dc.DataSaver.__new__(dc.DataSaver)
    instance._entities_queue = ctx.Queue()
    instance._measurements_queue = ctx.Queue()
    instance._stop_event = ctx.Event()
    instance._received_sent_e = False
    instance._received_sent_m = False
    instance._data_storage = MagicMock()
    instance._live = LiveAttributionConfig()
    instance._energy_writer = None
    instance._store_raw = True
    return instance


def test_stop_terminates_without_sentinels(saver):
    """Regression: the two 1000-iteration loops waited up to ~16 minutes each."""
    start = time.monotonic()
    saver._stop(deadline_s=0.5)
    elapsed = time.monotonic() - start

    assert elapsed < 10, f"_stop() took {elapsed:.1f}s, the deadline is not taking effect"
    saver._data_storage.close.assert_called_once()


def test_stop_saves_what_arrived_before_the_sentinel(saver, monkeypatch):
    monkeypatch.setattr(dc, "read_temp_ipc", lambda path: f"df:{path}")
    saver._entities_queue.put(["entity-a"])
    saver._entities_queue.put(SENT_E)
    saver._measurements_queue.put("/tmp/m1.ipc")
    saver._measurements_queue.put(SENT_M)
    time.sleep(0.3)

    saver._stop(deadline_s=5.0)

    saver._data_storage.save_entities.assert_called_once_with(["entity-a"])
    saver._data_storage.save_measurements.assert_called_once_with(["df:/tmp/m1.ipc"])
    assert saver._received_sent_e is True
    assert saver._received_sent_m is True


def test_stop_skips_drains_the_loop_already_finished(saver):
    """store_data() sets these when it sees a sentinel; _stop must not wait again."""
    saver._received_sent_e = True
    saver._received_sent_m = True

    start = time.monotonic()
    saver._stop(deadline_s=30.0)
    elapsed = time.monotonic() - start

    assert elapsed < 1.0, f"waited {elapsed:.1f}s although both sentinels were already seen"


def test_sentinel_is_not_saved_as_an_entity(saver):
    """Regression: SENT_E is a tuple and would land in save_entities as one."""
    saver._entities_queue.put(SENT_E)
    saver._measurements_queue.put(SENT_M)
    time.sleep(0.3)

    saver._stop(deadline_s=5.0)

    saver._data_storage.save_entities.assert_not_called()
    saver._data_storage.save_measurements.assert_not_called()


# --------------------------------------------------------------------------
# close_and_join_feeder
# --------------------------------------------------------------------------


def test_feeder_flush_reports_success():
    queue = mp.Queue()
    queue.put("x")
    consumed = queue.get(timeout=2)

    assert consumed == "x"
    assert close_and_join_feeder(queue, timeout_s=5.0) is True


def test_feeder_flush_gives_up_instead_of_blocking():
    """With nothing reading the pipe, join_thread() would block forever."""

    class StuckQueue:
        cancelled = False

        def close(self):
            pass

        def join_thread(self):
            time.sleep(30)

        def cancel_join_thread(self):
            StuckQueue.cancelled = True

    start = time.monotonic()
    flushed = close_and_join_feeder(StuckQueue(), timeout_s=0.5, name="stuck")
    elapsed = time.monotonic() - start

    assert flushed is False
    assert StuckQueue.cancelled is True
    assert elapsed < 3.0, f"took {elapsed:.1f}s"


# --------------------------------------------------------------------------
# DataCollector.stop
# --------------------------------------------------------------------------


@pytest.fixture
def collector():
    ctx = mp.get_context("spawn")
    instance = dc.DataCollector.__new__(dc.DataCollector)
    instance._ctx = ctx
    instance._entities_queue = ctx.Queue()
    instance._measurements_queue = ctx.Queue()
    instance._measurement_buffer = []
    instance._buffer_row_count = 0
    instance._stop_event = ctx.Event()
    instance.ipc_writer = MagicMock()
    instance.process = MagicMock()
    instance.process.is_alive.return_value = False
    return instance


def test_stop_delivers_both_sentinels(collector, monkeypatch):
    """Regression: cancel_join_thread() right after close() was allowed to drop them.

    The flush is stubbed out because a closed queue cannot be read from the process that
    closed it; in production the saver process is the reader.
    """
    monkeypatch.setattr(dc, "close_and_join_feeder", lambda queue, name: True)

    collector.stop()

    assert collector._entities_queue.get(timeout=5) == SENT_E
    assert collector._measurements_queue.get(timeout=5) == SENT_M


def test_stop_never_cancels_the_feeder_on_the_happy_path(collector, monkeypatch):
    cancelled = []
    for queue, label in ((collector._entities_queue, "e"), (collector._measurements_queue, "m")):
        monkeypatch.setattr(queue, "cancel_join_thread", lambda label=label: cancelled.append(label))

    collector.stop()

    assert cancelled == [], "buffered data must not be dropped when the feeder can flush"


def test_stop_sets_the_event_only_after_the_sentinels_are_flushed(collector, monkeypatch):
    order = []
    monkeypatch.setattr(dc, "close_and_join_feeder", lambda queue, name: order.append(f"flush-{name}") or True)
    monkeypatch.setattr(collector._stop_event, "set", lambda: order.append("stop_event"))

    collector.stop()

    assert order == ["flush-entities", "flush-measurements", "stop_event"]


def test_stop_escalates_from_terminate_to_kill(collector):
    process = collector.process
    process.is_alive.return_value = True

    collector.stop(join_timeout=0.1)

    assert process.terminate.called
    assert process.kill.called
    assert process.join.call_count == 3, "join after the initial wait, after terminate and after kill"
    assert collector.process is None
