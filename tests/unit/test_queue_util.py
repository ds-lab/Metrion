import multiprocessing as mp
import queue as q
import time

from metrion.queue_util import drain_until_sentinel

SENTINEL = ("__EOF__", "test")


def test_returns_items_before_sentinel():
    src = q.Queue()
    for i in range(3):
        src.put(i)
    src.put(SENTINEL)

    items, seen = drain_until_sentinel(src, SENTINEL, deadline_s=2.0)

    assert items == [0, 1, 2]
    assert seen is True


def test_sentinel_is_not_returned_as_an_item():
    """Regression: the sentinel used to reach read_temp_ipc as a file path."""
    src = q.Queue()
    src.put("/tmp/batch_a.ipc")
    src.put(SENTINEL)

    items, _ = drain_until_sentinel(src, SENTINEL, deadline_s=2.0)

    assert SENTINEL not in items


def test_returns_on_deadline_when_sentinel_missing():
    src = q.Queue()
    src.put("a")

    start = time.monotonic()
    items, seen = drain_until_sentinel(src, SENTINEL, deadline_s=0.5, poll_timeout=0.05)
    elapsed = time.monotonic() - start

    assert items == ["a"]
    assert seen is False
    assert 0.4 < elapsed < 1.5, "must return around the deadline, not earlier and not never"


def test_empty_queue_returns_at_deadline():
    items, seen = drain_until_sentinel(q.Queue(), SENTINEL, deadline_s=0.3, poll_timeout=0.05)

    assert items == []
    assert seen is False


def test_works_with_multiprocessing_queue():
    src = mp.Queue()
    src.put("x")
    src.put(SENTINEL)

    items, seen = drain_until_sentinel(src, SENTINEL, deadline_s=3.0)

    assert items == ["x"]
    assert seen is True


def test_sentinel_survives_serialisation_across_a_process_boundary():
    """multiprocessing pickles the sentinel, so it is equal but not identical."""
    src = mp.Queue()
    src.put("/tmp/x.ipc")
    src.put(("__EOF__", "pidmeta"))

    items, seen = drain_until_sentinel(src, ("__EOF__", "pidmeta"), deadline_s=3.0)

    assert seen is True
    assert items == ["/tmp/x.ipc"]


def test_does_not_busy_spin(monkeypatch):
    calls = 0
    src = q.Queue()
    original_get = src.get

    def counting_get(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_get(*args, **kwargs)

    monkeypatch.setattr(src, "get", counting_get)
    drain_until_sentinel(src, SENTINEL, deadline_s=0.5, poll_timeout=0.1)

    assert calls <= 10, f"{calls} get() calls in 0.5s indicates a busy spin"
