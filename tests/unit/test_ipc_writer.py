import multiprocessing as mp
import queue
import time

import polars as pl
import pytest

from metrion.util import IpcAsyncWriter, read_temp_ipc


def _small_df(n=10):
    return pl.DataFrame({"value": list(range(n)), "metric_id": ["M"] * n})


def _comm_df(n=4):
    return pl.DataFrame({"src_endpoint": ["a"] * n, "dst_endpoint": ["b"] * n, "value": list(range(n))})


@pytest.fixture
def out_queue():
    q = mp.Queue()
    yield q
    q.close()


def _assert_nothing_more(q, timeout=1.5):
    # qsize() is not implemented on macOS, so wait for a path instead of counting them.
    with pytest.raises(queue.Empty):
        q.get(timeout=timeout)


def test_flushes_on_age_without_reaching_row_threshold(out_queue):
    """Regression: without a time threshold a small batch waited for the next producer event."""
    writer = IpcAsyncWriter(out_queue, max_rows=500_000, max_age_s=0.5)
    try:
        writer.enqueue(_small_df())
        assert read_temp_ipc(out_queue.get(timeout=5.0)).height == 10
    finally:
        writer.close()


def test_flush_happens_within_max_age(out_queue):
    writer = IpcAsyncWriter(out_queue, max_rows=500_000, max_age_s=1.0)
    try:
        start = time.monotonic()
        writer.enqueue(_small_df())
        out_queue.get(timeout=5.0)
        assert time.monotonic() - start < 3.0
    finally:
        writer.close()


def test_the_same_buffer_is_not_written_twice(out_queue):
    """Regression: a flush that does not reset the batch rewrites it every max_age_s."""
    writer = IpcAsyncWriter(out_queue, max_rows=500_000, max_age_s=0.3)
    try:
        writer.enqueue(_small_df())
        assert read_temp_ipc(out_queue.get(timeout=5.0)).height == 10
        _assert_nothing_more(out_queue)
    finally:
        writer.close()


def test_age_counts_from_the_first_buffered_frame(out_queue):
    """Frames arriving inside one max_age_s window end up in a single file."""
    writer = IpcAsyncWriter(out_queue, max_rows=500_000, max_age_s=1.0)
    try:
        writer.enqueue(_small_df())
        time.sleep(0.2)
        writer.enqueue(_small_df())
        assert read_temp_ipc(out_queue.get(timeout=5.0)).height == 20
        _assert_nothing_more(out_queue)
    finally:
        writer.close()


def test_close_flushes_leftovers(out_queue):
    writer = IpcAsyncWriter(out_queue, max_rows=500_000, max_age_s=3600.0)
    writer.enqueue([_small_df()], "measurement")
    writer.close()
    assert read_temp_ipc(out_queue.get(timeout=5.0)).height == 10


def test_row_threshold_still_writes_immediately(out_queue):
    writer = IpcAsyncWriter(out_queue, max_rows=5, max_age_s=3600.0)
    try:
        writer.enqueue(_small_df(12))
        assert read_temp_ipc(out_queue.get(timeout=5.0)).height == 5
        assert read_temp_ipc(out_queue.get(timeout=5.0)).height == 5
        assert read_temp_ipc(out_queue.get(timeout=5.0)).height == 2
    finally:
        writer.close()


def test_ids_are_assigned_only_when_asked_for(out_queue):
    writer = IpcAsyncWriter(out_queue, max_age_s=0.3, assign_ids=True)
    try:
        writer.enqueue([_small_df()], "measurement")
        df = read_temp_ipc(out_queue.get(timeout=5.0))
        assert "id" in df.columns
        assert df["id"].n_unique() == 10
    finally:
        writer.close()

    plain = IpcAsyncWriter(out_queue, max_age_s=0.3)
    try:
        plain.enqueue(_small_df())
        assert "id" not in read_temp_ipc(out_queue.get(timeout=5.0)).columns
    finally:
        plain.close()


def test_frame_types_are_batched_separately(out_queue):
    """Measurements and communication measurements have different schemas and cannot share a file."""
    writer = IpcAsyncWriter(out_queue, max_rows=500_000, max_age_s=0.3)
    try:
        writer.enqueue([_small_df()], "measurement")
        writer.enqueue([_comm_df()], "communication_measurement")
        written = [read_temp_ipc(out_queue.get(timeout=5.0)) for _ in range(2)]
        by_height = {df.height: df for df in written}
        assert set(by_height) == {10, 4}
        assert "src_endpoint" in by_height[4].columns
        assert "src_endpoint" not in by_height[10].columns
    finally:
        writer.close()


def test_no_busy_spin_when_idle(out_queue):
    import resource

    writer = IpcAsyncWriter(out_queue, max_age_s=0.5)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_utime
    time.sleep(2.0)
    after = resource.getrusage(resource.RUSAGE_SELF).ru_utime
    writer.close()
    assert after - before < 0.5, "more than 0.5 s of CPU in 2 s of idling points at a busy spin"
