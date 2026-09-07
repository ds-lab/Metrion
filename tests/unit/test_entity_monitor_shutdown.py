import queue as q
import threading
import time
from unittest.mock import MagicMock

import pytest

em = pytest.importorskip("metrion.monitoring.entity_monitor", reason="needs bcc")

from metrion.data_record import DataRecord  # noqa: E402
from metrion.monitoring.data_source.data_source_runner import DataSourceRunner  # noqa: E402


def _record(*measurements):
    return DataRecord(entities=[], measurements=list(measurements))


def _monitor(*queues):
    """EntityTypeMonitor without real data sources; __init__ would start threads."""
    monitor = em.EntityTypeMonitor.__new__(em.EntityTypeMonitor)
    monitor._data_source_runners = {}
    runners = []
    for i, queue in enumerate(queues):
        runner = MagicMock(spec=DataSourceRunner)
        runner._id = f"runner-{i}"
        runner.is_alive.return_value = False
        runner.get_final_data_records.return_value = None
        monitor._data_source_runners[runner] = queue
        runners.append(runner)
    monitor._entities = [MagicMock()]
    return monitor, runners


def test_get_data_skips_sent_end_without_crashing():
    """Regression: SENT_END is a str, so .measurements raised AttributeError."""
    queue = q.Queue()
    queue.put(_record("m1"))
    queue.put(DataSourceRunner.SENT_END)
    monitor, _ = _monitor(queue)

    record = monitor.get_data()

    assert list(record.measurements) == ["m1"]
    assert queue.unfinished_tasks == 0, "SENT_END must be acknowledged with task_done()"


def test_drain_of_dead_runner_terminates_and_keeps_its_data():
    """Regression: dead runners were skipped, so their data and SENT_END stayed in the queue."""
    queue = q.Queue()
    queue.put(_record("m2"))
    queue.put(DataSourceRunner.SENT_END)
    monitor, _ = _monitor(queue)

    record = monitor.stop()

    assert "m2" in record.measurements
    assert queue.unfinished_tasks == 0


def test_stop_leaves_no_unfinished_tasks_so_a_queue_join_would_return():
    """This is the hang: stop() skipped dead runners, then called queue.join() on them anyway."""
    queue = q.Queue()
    queue.put(_record("m3"))
    queue.put(DataSourceRunner.SENT_END)
    monitor, _ = _monitor(queue)

    monitor.stop()

    joined = threading.Event()
    threading.Thread(target=lambda: (queue.join(), joined.set()), daemon=True).start()
    assert joined.wait(2.0), "queue.join() would still block forever"


def test_drain_without_sent_end_returns_at_deadline():
    queue = q.Queue()
    queue.put(_record("m4"))
    monitor, runners = _monitor(queue)

    start = time.monotonic()
    record = monitor._drain_queue_for_runner(runners[0], deadline_s=0.5)
    elapsed = time.monotonic() - start

    assert "m4" in record.measurements
    assert 0.4 < elapsed < 2.0, f"returned after {elapsed:.1f}s"


def test_all_runners_are_asked_to_stop_before_any_draining():
    """Sequential stop-then-drain made runner N wait out every drain before it.

    order records when each runner was told to stop and when its queue was drained.
    """
    order = []
    queues = []
    monitor, runners = _monitor(q.Queue(), q.Queue())
    for i, (runner, queue) in enumerate(monitor._data_source_runners.items()):
        queue.put(DataSourceRunner.SENT_END)
        queues.append(queue)
        runner.request_stop.side_effect = lambda i=i: order.append(("stop", i))

    original = monitor._drain_queue_for_runner
    seen = list(monitor._data_source_runners)

    def tracking_drain(runner, deadline_s=15.0):
        order.append(("drain", seen.index(runner)))
        return original(runner, deadline_s)

    monitor._drain_queue_for_runner = tracking_drain
    monitor.stop()

    assert order == [("stop", 0), ("stop", 1), ("drain", 0), ("drain", 1)]


def test_stop_collects_final_records_from_runners_that_already_exited():
    """A runner sets _done_event right after SENT_END, so its final data must still be read."""
    queue = q.Queue()
    queue.put(DataSourceRunner.SENT_END)
    monitor, runners = _monitor(queue)
    runners[0].get_final_data_records.return_value = _record("final")

    record = monitor.stop()

    assert "final" in record.measurements


def test_get_final_data_records_gives_up_instead_of_blocking():
    runner = DataSourceRunner.__new__(DataSourceRunner)
    runner._id = "never-started"
    runner._done_event = threading.Event()
    runner._final_data_records = _record()

    start = time.monotonic()
    result = runner.get_final_data_records(timeout=0.5)
    elapsed = time.monotonic() - start

    assert result is not None
    assert 0.4 < elapsed < 2.0, f"waited {elapsed:.1f}s"
