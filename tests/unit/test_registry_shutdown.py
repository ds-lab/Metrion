import multiprocessing as mp
import time
from unittest.mock import MagicMock

import pytest

ler = pytest.importorskip("metrion.entity.logical_entity_registry", reason="needs bcc")

SENTINEL = "SENTINEL"


@pytest.fixture
def make_registry():
    """__init__ scans /proc and starts a BPF process, so it is bypassed."""

    class FakeRegistry:
        stop = ler.LogicalEntityRegistry.stop
        update = ler.LogicalEntityRegistry.update

        def __init__(self, producer_finished=False):
            self._queue = mp.Queue()
            self._config = MagicMock()
            self._config.sentinel = SENTINEL
            self._producer_finished = producer_finished
            self._bpf_process = MagicMock()
            self._bpf_process.is_alive.return_value = False
            self._process_events = MagicMock()
            self.get_new_entities = MagicMock(return_value=[])

    return FakeRegistry


@pytest.fixture
def no_ipc_reads(monkeypatch):
    paths = []
    monkeypatch.setattr(ler, "read_temp_ipc", lambda path: paths.append(path) or MagicMock())
    monkeypatch.setattr(ler, "build_cgroup_id_map", lambda: {})
    monkeypatch.setattr(ler.pl, "concat", lambda dfs, how: MagicMock())
    return paths


def test_update_records_the_sentinel_instead_of_discarding_it(make_registry, no_ipc_reads):
    """update() used to break on the sentinel silently, leaving stop() to wait for it forever."""
    registry = make_registry()
    registry._queue.put(SENTINEL)
    time.sleep(0.3)

    registry.update()

    assert registry._producer_finished is True
    assert no_ipc_reads == [], "the sentinel must not be read as a file path"


def test_update_keeps_the_flag_false_while_the_producer_runs(make_registry, no_ipc_reads):
    registry = make_registry()
    registry._queue.put("/tmp/a.ipc")
    time.sleep(0.3)

    registry.update()

    assert registry._producer_finished is False
    assert no_ipc_reads == ["/tmp/a.ipc"]


def test_stop_does_not_hang_when_the_sentinel_was_consumed_early(make_registry, no_ipc_reads):
    registry = make_registry(producer_finished=True)

    start = time.monotonic()
    registry.stop()
    elapsed = time.monotonic() - start

    assert elapsed < 3.0, f"stop() took {elapsed:.1f}s although the producer was already done"


def test_stop_does_not_busy_spin(make_registry, no_ipc_reads):
    """The old loop called get_nowait() in a tight except-continue and burned a full core."""
    calls = 0

    class CountingQueue:
        def get(self, timeout=None):
            nonlocal calls
            calls += 1
            time.sleep(min(timeout or 0, 0.1))
            raise ler.q.Empty

        def get_nowait(self):
            nonlocal calls
            calls += 1
            raise ler.q.Empty

    registry = make_registry(producer_finished=True)
    registry._queue = CountingQueue()

    registry.stop()

    assert calls <= 20, f"{calls} queue polls indicates a busy spin"


def test_stop_drains_paths_and_skips_the_sentinel(make_registry, no_ipc_reads):
    registry = make_registry()
    registry._queue.put("/tmp/a.ipc")
    registry._queue.put("/tmp/b.ipc")
    registry._queue.put(SENTINEL)
    time.sleep(0.3)

    registry.stop()

    assert no_ipc_reads == ["/tmp/a.ipc", "/tmp/b.ipc"]
    assert registry._process_events.call_count == 2


def test_stop_is_bounded_and_kills_an_unresponsive_producer(make_registry, no_ipc_reads, monkeypatch):
    monkeypatch.setattr(
        ler,
        "drain_until_sentinel",
        lambda queue, sentinel, deadline_s, name: ([], False),
    )
    registry = make_registry()
    registry._bpf_process.is_alive.return_value = True

    registry.stop()

    assert registry._bpf_process.kill.called
    assert registry._bpf_process.join.call_count == 2, "kill must be followed by a join to reap the zombie"
